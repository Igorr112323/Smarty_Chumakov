"""Итерации доработки признаков: AUC и 95 % ДИ после каждой (пункт 2.2 промта).

Скрипт считает признаки по токенам ответа, метки берёт из разметки корпуса и
сравнивает четыре итерации:

    1-baseline            attention_entropy, ctx_attention_mass, embedding_density;
    2-mass-normalized     + масса на токен контекста и на длину ответа;
    3-position-evidence   + позиция токена и расстояние до подтверждающего участка;
    4-lexical-semantic    + числа/единицы, отрицание, модальность, даты.

Для каждой итерации — AUC свёртки (усреднение ориентированных признаков, без
подбора порога), для каждого признака — AUC с 95 % ДИ (бутстрэп, seed фиксирован).
Отдельно проверяется гипотеза «масса внимания на контекст информативнее сырой
энтропии внимания»: печатается сравнение AUC пары признаков и вывод.

Разбиение — по документам (``meta.group``/``meta.doc_id``): признаки и метрика
никогда не считаются на одном документе и в обучении, и в измерении. Порог решения
не подбирается: свёртка признаков фиксирована, сравнение идёт по AUC.

Запуск::

    python scripts/feature_iterations.py --dataset data/corpus_a3/pairs.jsonl \\
        --mode hf --layers first,middle,-4,last --limit 200 --bootstrap 5000 \\
        --seed 42 --out reports/experiments/feature_iterations
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spanverify import Verifier  # noqa: E402
from spanverify.core import tokenize_with_offsets  # noqa: E402
from spanverify.dataset import read_pairs  # noqa: E402
from spanverify.feature_pack import EXTRA_FEATURE_NAMES, FEATURE_ITERATIONS, iteration_features  # noqa: E402
from spanverify.features import FEATURE_NAMES, extract_features, is_scored_token  # noqa: E402

# Направление признака: 1 — больше значит «рискованнее», -1 — наоборот.
ORIENTATION: dict[str, int] = {
    "attention_entropy": 1,
    "ctx_attention_mass": -1,
    "embedding_density": 1,
    "mass_per_context_token": -1,
    "mass_per_answer_token": -1,
    "position": 1,
    "evidence_distance": 1,
    "number_support": -1,
    "unit_support": -1,
    "negation_conflict": 1,
    "modality_conflict": 1,
    "date_support": -1,
}


def auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """AUC по Манну—Уитни с обработкой совпадений (0.5 на ничью)."""
    pairs = sorted(zip(scores, labels, strict=False), key=lambda item: item[0])
    positive = sum(labels)
    negative = len(labels) - positive
    if positive == 0 or negative == 0:
        return 0.5
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index
        while end + 1 < len(pairs) and pairs[end + 1][0] == pairs[index][0]:
            end += 1
        average_rank = (index + end) / 2 + 1
        rank_sum += average_rank * sum(label for _score, label in pairs[index : end + 1])
        index = end + 1
    return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)


def bootstrap_ci(
    labels: Sequence[int],
    scores: Sequence[float],
    iterations: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """95 % ДИ бутстрэпом по парам «метка — значение признака»."""
    if not labels or iterations <= 0:
        return 0.0, 0.0
    rng = random.Random(seed)
    size = len(labels)
    values: list[float] = []
    for _ in range(iterations):
        sample = [rng.randrange(size) for _ in range(size)]
        sampled_labels = [labels[i] for i in sample]
        if len(set(sampled_labels)) < 2:
            continue
        values.append(auc(sampled_labels, [scores[i] for i in sample]))
    if not values:
        return 0.0, 0.0
    values.sort()
    low = values[int(alpha / 2 * len(values))]
    high = values[min(len(values) - 1, int((1 - alpha / 2) * len(values)))]
    return low, high


def collect_rows(
    pairs: Sequence[dict],
    mode: str,
    layer: str,
    model: str | None,
    limit: int,
) -> tuple[dict[str, list[float]], list[int], list[str]]:
    """Собрать значения признаков по токенам и метки (1 — недостоверный токен)."""
    verifier = Verifier(mode=mode, model_name=model)
    values: dict[str, list[float]] = {name: [] for name in (*FEATURE_NAMES, *EXTRA_FEATURE_NAMES)}
    labels: list[int] = []
    groups: list[str] = []
    processed = 0
    for pair in pairs:
        answer = pair.get("answer", "")
        context = pair.get("context", "")
        tokens = tokenize_with_offsets(answer)
        kwargs: dict[str, object] = {}
        if mode == "hf":
            kwargs = {"model_name": model, "layer": layer}
        matrix = extract_features(answer, context, mode=mode, answer_tokens=tokens, **kwargs)
        truth = [(int(start), int(end)) for start, end, label in pair.get("labels", []) if int(label) == 1]
        for index, token in enumerate(tokens):
            if not is_scored_token(token.text) or index >= len(matrix):
                continue
            labels.append(1 if any(token.start < end and token.end > start for start, end in truth) else 0)
            for name in (*FEATURE_NAMES, *EXTRA_FEATURE_NAMES):
                if name in values:
                    source = getattr(matrix, name, None) or matrix.extra.get(name)
                    values[name].append(float(source[index]) if source and index < len(source) else 0.0)
            groups.append(str((pair.get("meta") or {}).get("group") or (pair.get("meta") or {}).get("doc_id") or "?"))
        processed += 1
        if limit and processed >= limit:
            break
    _ = verifier  # конфигурация читается для согласованности предупреждений
    return values, labels, groups


def oriented(values: Sequence[float], name: str) -> list[float]:
    """Привести признак к «больше = рискованнее»."""
    sign = ORIENTATION.get(name, 1)
    return [sign * value for value in values]


def iteration_scores(
    iteration: str,
    values: dict[str, list[float]],
    labels: Sequence[int],
) -> list[float]:
    """Свёртка признаков итерации: среднее ориентированных значений.

    Порог не подбирается, веса не обучаются: это сравнение информативности
    наборов признаков, а не настройка решающего правила.
    """
    base = {name: values[name] for name in FEATURE_NAMES}
    extras = {name: values[name] for name in EXTRA_FEATURE_NAMES}
    available = iteration_features(iteration, base, extras)
    size = len(labels)
    combined = [0.0] * size
    used = 0
    for name, series in available.items():
        if name not in ORIENTATION or len(series) != size:
            continue
        oriented_series = oriented(series, name)
        for index, value in enumerate(oriented_series):
            combined[index] += value
        used += 1
    if used:
        combined = [value / used for value in combined]
    return combined


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Итерации доработки признаков: AUC с 95 % ДИ")
    parser.add_argument("--dataset", default="data/corpus_a3/pairs.jsonl")
    parser.add_argument("--mode", choices=["demo", "hf"], default="hf")
    parser.add_argument("--layers", default="last")
    parser.add_argument("--model", default=None, help="имя модели для режима hf")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="reports/experiments/feature_iterations")
    parser.add_argument("--test-share", type=float, default=0.3)
    args = parser.parse_args(argv)

    started = time.time()
    pairs = list(read_pairs(args.dataset))
    if not pairs:
        print(f"ошибка: датасет пуст: {args.dataset}", file=sys.stderr)
        return 2
    # Разбиение по документам: признаки измеряются на отложенной части, а не на всей.
    groups: dict[str, list[dict]] = {}
    for pair in pairs:
        meta = pair.get("meta") or {}
        key = str(meta.get("group") or meta.get("doc_id") or pair.get("id"))
        groups.setdefault(key, []).append(pair)
    keys = sorted(groups)
    random.Random(args.seed).shuffle(keys)
    cut = max(1, int(len(keys) * (1 - args.test_share)))
    held_out = [pair for key in keys[cut:] for pair in groups[key]] or pairs

    layers = [item.strip() for item in args.layers.split(",") if item.strip()]
    report: dict[str, object] = {
        "dataset": args.dataset,
        "mode": args.mode,
        "model": args.model,
        "seed": args.seed,
        "bootstrap_iterations": args.bootstrap,
        "limit": args.limit,
        "test_share": args.test_share,
        "held_out_pairs": len(held_out),
        "document_groups": len(keys),
        "layers": {},
        "iterations": [],
        "hypothesis": {},
        "contrast": {},
    }

    for layer in layers:
        values, labels, pair_groups = collect_rows(held_out, args.mode, layer, args.model, args.limit)
        if len(labels) < 30 or len(set(labels)) < 2:
            report["layers"][layer] = {"error": "недостаточно размеченных токенов", "tokens": len(labels)}
            continue
        per_feature: dict[str, dict[str, float]] = {}
        for name in (*FEATURE_NAMES, *EXTRA_FEATURE_NAMES):
            series = oriented(values[name], name)
            point = auc(labels, series)
            low, high = bootstrap_ci(labels, series, args.bootstrap, args.seed)
            per_feature[name] = {
                "auc_oriented": round(point, 4),
                "ci_low": round(low, 4),
                "ci_high": round(high, 4),
                "orientation": ORIENTATION.get(name, 1),
            }
        report["layers"][layer] = {"tokens": len(labels), "features": per_feature}
        # Контроль контраста: доля слов ответа, встретившихся в контексте.
        grounded = [pair for pair in held_out if (pair["meta"] or {}).get("mode") == "faithful"]
        unsupported = [
            pair for pair in held_out if (pair["meta"] or {}).get("mode") in {"unconfirmed", "excess", "missing"}
        ]
        report["contrast"][layer] = {
            "grounded_pairs": len(grounded),
            "grounded_word_overlap": _mean_word_overlap(grounded),
            "unsupported_pairs": len(unsupported),
            "unsupported_word_overlap": _mean_word_overlap(unsupported),
            "criterion": "≥ 0.80 для пар с опорой, ≤ 0.30 для пар без опоры",
        }
        if layer == layers[0]:
            for iteration, names in FEATURE_ITERATIONS:
                scores = iteration_scores(iteration, values, labels)
                point = auc(labels, scores)
                low, high = bootstrap_ci(labels, scores, args.bootstrap, args.seed + 1)
                report["iterations"].append(
                    {
                        "iteration": iteration,
                        "added_features": list(names),
                        "auc": round(point, 4),
                        "ci_low": round(low, 4),
                        "ci_high": round(high, 4),
                        "features_used": len(iteration_features(iteration, values, values)),
                    }
                )
            entropy_auc = per_feature["attention_entropy"]["auc_oriented"]
            mass_auc = per_feature["ctx_attention_mass"]["auc_oriented"]
            report["hypothesis"] = {
                "statement": "масса внимания на контекст информативнее сырой энтропии внимания",
                "entropy_auc": entropy_auc,
                "mass_auc": mass_auc,
                "difference": round(mass_auc - entropy_auc, 4),
                "supported": bool(mass_auc > entropy_auc + 0.02),
                "note": "сравнение по AUC на отложенной части; различия меньше 0.02 считаются шумом",
            }
    report["duration_s"] = round(time.time() - started, 1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.with_suffix(".json")).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (out.with_suffix(".md")).write_text(render_report(report), encoding="utf-8")
    print(render_report(report))
    return 0


def _mean_word_overlap(pairs: Sequence[dict]) -> float:
    """Средняя доля слов ответа, встретившихся в контексте (контроль контраста)."""
    if not pairs:
        return 0.0
    shares: list[float] = []
    for pair in pairs:
        answer_words = {word.lower().replace("ё", "е") for word in str(pair.get("answer", "")).split() if len(word) > 3}
        if not answer_words:
            continue
        context = str(pair.get("context", "")).lower().replace("ё", "е")
        found = sum(1 for word in answer_words if word in context)
        shares.append(found / len(answer_words))
    return round(sum(shares) / len(shares), 4) if shares else 0.0


def render_report(report: dict) -> str:
    """Текст отчёта генерируется из JSON: числа в тексте и в файле совпадают."""
    lines = [
        "# Итерации доработки признаков",
        "",
        f"Датасет: `{report['dataset']}`, режим: **{report['mode']}**"
        + (f", модель: `{report['model']}`" if report["model"] else ""),
        f"Отложенная часть: {report['held_out_pairs']} пар, {report['document_groups']} групп документов, "
        f"seed {report['seed']}, бутстрэп {report['bootstrap_iterations']}, время {report['duration_s']} с.",
        "",
        "## Итерации (слой " + str(report["layers"] and next(iter(report["layers"]))) + ")",
        "",
        "| Итерация | Добавлено | AUC | 95 % ДИ |",
        "|---|---|---|---|",
    ]
    for item in report["iterations"]:
        added = ", ".join(item["added_features"]) or "базовые признаки"
        lines.append(f"| {item['iteration']} | {added} | {item['auc']} | [{item['ci_low']}; {item['ci_high']}] |")
    hypothesis = report.get("hypothesis") or {}
    if hypothesis:
        lines += [
            "",
            "## Гипотеза",
            "",
            f"«{hypothesis['statement']}»: AUC энтропии {hypothesis['entropy_auc']}, "
            f"AUC массы {hypothesis['mass_auc']}, разница {hypothesis['difference']}. "
            f"**Подтверждена: {'да' if hypothesis['supported'] else 'нет'}.** {hypothesis['note']}.",
        ]
    lines += ["", "## Признаки по слоям", ""]
    for layer, payload in report["layers"].items():
        lines.append(f"### Слой {layer} ({payload.get('tokens', 0)} токенов)")
        lines.append("")
        lines.append("| Признак | AUC | 95 % ДИ | Направление |")
        lines.append("|---|---|---|---|")
        for name, values in sorted((payload.get("features") or {}).items(), key=lambda item: -item[1]["auc_oriented"]):
            direction = "риск" if values["orientation"] > 0 else "опора"
            lines.append(
                f"| `{name}` | {values['auc_oriented']} | [{values['ci_low']}; {values['ci_high']}] | {direction} |"
            )
        lines.append("")
        contrast = report["contrast"].get(layer) or {}
        if contrast:
            lines.append(
                f"Контроль контраста: с опорой {contrast['grounded_word_overlap']} "
                f"({contrast['grounded_pairs']} пар), без опоры {contrast['unsupported_word_overlap']} "
                f"({contrast['unsupported_pairs']} пар); критерий {contrast['criterion']}."
            )
            lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - точка входа
    raise SystemExit(main())
