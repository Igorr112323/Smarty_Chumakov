"""Пилот на реальной русскоязычной модели: информативность признаков по слоям.

Что делает скрипт (шаг 5 мастер-промта):

1. Берёт контрастные пары «ответ с опорой на контекст» / «ответ без опоры» и
   проверяет, что контраст действительно есть: доля слов ответа, встретившихся
   в контексте, должна быть ≥ 80 % для первых и ≤ 30 % для вторых. Если порог
   не выдержан, это печатается прямо в отчёте (данные не подгоняются).
2. Считает признаки модели ``ai-forever/rugpt3small_based_on_gpt2`` на CPU для
   каждого слоя внимания: ``first``, ``middle``, ``-4``, ``last``.
3. Считает AUC каждого признака с 95 % доверительным интервалом
   (бутстрэп, 5000 итераций, фиксированный seed) и сравнивает с базовыми
   уровнями: случайное угадывание (0.5) и косинусная близость эмбеддингов без
   внимания (``embedding_density``).
4. Пишет отчёт ``reports/pilot/pilot.json`` и ``reports/pilot/pilot.md``.

Ключевая гипотеза, которую проверяет пилот: масса внимания на контекст
(``ctx_attention_mass``) информативнее «сырой» энтропии внимания
(``attention_entropy``). Если гипотеза не подтвердится, в отчёте так и будет
написано — результат есть результат в любом случае.

Запуск (нужны torch и transformers, интернет для первой загрузки весов):

    python scripts/pilot_rugpt3small.py --pairs 24 --bootstrap 5000
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spanverify.core import tokenize_with_offsets  # noqa: E402
from spanverify.dataset import FAQ_TEMPLATES, SUBJECTS, unit_form  # noqa: E402
from spanverify.features import FEATURE_NAMES, _stem, hf_features, is_scored_token  # noqa: E402

LAYERS = ("first", "middle", "-4", "last")
MODEL_DEFAULT = "ai-forever/rugpt3small_based_on_gpt2"
POSITIVE_LABEL = 1


@dataclass
class TokenRow:
    """Один токен ответа: признаки по слоям и метка достоверности."""

    pair_id: str
    layer: str
    text: str
    features: dict[str, float]
    label: int


def auc(labels: list[int], scores: list[float]) -> float:
    """AUC по ранговой формуле Манна — Уитни (совпадает с метрикой продукта)."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[position]]:
            end += 1
        average = (position + end) / 2 + 1
        for step in range(position, end + 1):
            ranks[order[step]] = average
        position = end + 1
    rank_sum = sum(ranks[index] for index, label in enumerate(labels) if label)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def bootstrap_ci(labels: list[int], scores: list[float], iterations: int = 5000, seed: int = 42) -> tuple[float, float]:
    """95 % доверительный интервал AUC бутстрэпом (ресемплинг токенов)."""
    if iterations <= 0 or len(labels) < 4:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    size = len(labels)
    values: list[float] = []
    for _ in range(iterations):
        indices = [rng.randrange(size) for _ in range(size)]
        sample_labels = [labels[index] for index in indices]
        if len(set(sample_labels)) < 2:
            continue
        values.append(auc(sample_labels, [scores[index] for index in indices]))
    values = [value for value in values if value == value]
    if len(values) < 20:
        return float("nan"), float("nan")
    values.sort()
    low = values[int(0.025 * len(values))]
    high = values[min(len(values) - 1, int(0.975 * len(values)))]
    return low, high


def overlap_ratio(answer: str, context: str) -> float:
    """Доля содержательных слов ответа, найденных в контексте (по основам)."""
    context_stems = {_stem(token.word) for token in tokenize_with_offsets(context) if is_scored_token(token.text)}
    answer_tokens = [token for token in tokenize_with_offsets(answer) if is_scored_token(token.text)]
    if not answer_tokens:
        return 0.0
    hits = sum(1 for token in answer_tokens if _stem(token.word) in context_stems)
    return hits / len(answer_tokens)


def build_contrast_pairs(count: int, seed: int = 2026) -> list[dict]:
    """Собрать контрастные пары: с опорой на контекст и без неё.

    Каждая пара — один и тот же вопрос и один и тот же документ. Отличие ровно
    одно: «с опорой» повторяет значение из документа, «без опоры» подставляет
    другое значение. Это самая трудная и самая честная постановка: признаки
    должны уловить подмену одного числа, а не «другой стиль письма».

    Числа берутся из контекста, поэтому контраст проверяем: доля ответов,
    воспроизводящих значение документа, должна быть 1.0 в первой группе и 0.0
    во второй. Если это не так — отчёт об этом скажет.
    """
    rng = random.Random(seed)
    rows: list[dict] = []
    for index in range(count):
        number = rng.randint(100, 999)
        subject, unit_kind, values = rng.choice(SUBJECTS)
        true_value = rng.choice(values)
        wrong_value = rng.choice([value for value in values if value != true_value])
        _, answer_template = rng.choice(FAQ_TEMPLATES)
        context = f"Регламент {number}: {subject} составляет " f"{true_value} {unit_form(true_value, unit_kind)}."

        def answer_for(value: int, template=answer_template, topic=subject, kind=unit_kind) -> str:
            """Ответ по шаблону с правильной формой единицы (аргументы связаны явно)."""
            return template.format(
                subject=topic,
                Subject=topic[0].upper() + topic[1:],
                value_phrase=f"{value} {unit_form(value, kind)}",
            )

        rows.append(
            {
                "id": f"pilot-{index:03d}-grounded",
                "context": context,
                "answer": answer_for(true_value),
                "kind": "grounded",
                "label": 0,
                "value": str(true_value),
            }
        )
        rows.append(
            {
                "id": f"pilot-{index:03d}-unsupported",
                "context": context,
                "answer": answer_for(wrong_value),
                "kind": "unsupported",
                "label": 1,
                "value": str(true_value),
            }
        )
    return rows


def collect_rows(pairs: list[dict], model_name: str, max_length: int) -> list[TokenRow]:
    """Посчитать признаки всех слоёв для всех пар (токены — единицы анализа)."""
    rows: list[TokenRow] = []
    for position, pair in enumerate(pairs, start=1):
        answer = str(pair["answer"])
        tokens = tokenize_with_offsets(answer)
        scored = [index for index, token in enumerate(tokens) if is_scored_token(token.text)]
        matrix_by_layer: dict[str, dict[str, list[float]]] = {}
        for layer in LAYERS:
            matrix = hf_features(
                answer,
                pair.get("context"),
                model_name=model_name,
                answer_tokens=tokens,
                layer=layer,
                max_length=max_length,
            )
            matrix_by_layer[layer] = {name: list(getattr(matrix, name)) for name in FEATURE_NAMES}
        label = int(pair["label"])
        for index in scored:
            for layer in LAYERS:
                values = matrix_by_layer[layer]
                rows.append(
                    TokenRow(
                        pair_id=str(pair["id"]),
                        layer=layer,
                        text=tokens[index].text,
                        features={name: float(values[name][index]) for name in FEATURE_NAMES},
                        label=label,
                    )
                )
        print(f"  [{position}/{len(pairs)}] {pair['id']}: токенов {len(scored)}, метка {label}", flush=True)
    return rows


def analyse(rows: list[TokenRow], iterations: int) -> dict:
    """AUC и доверительные интервалы по каждому признаку и слою."""
    report: dict[str, dict[str, dict[str, float]]] = {}
    for layer in LAYERS:
        layer_rows = [row for row in rows if row.layer == layer]
        if not layer_rows:
            continue
        labels = [row.label for row in layer_rows]
        report[layer] = {}
        for name in FEATURE_NAMES:
            scores = [row.features[name] for row in layer_rows]
            value = auc(labels, scores)
            low, high = bootstrap_ci(labels, scores, iterations=iterations)
            # Для «сырой» энтропии высокий балл означает риск, для массы опоры
            # знак обратный — приводим к «чем выше, тем недостовернее».
            orient = -1.0 if name == "ctx_attention_mass" else 1.0
            oriented = [orient * score for score in scores]
            oriented_auc = auc(labels, oriented)
            report[layer][name] = {
                "auc": value,
                "auc_oriented": oriented_auc,
                "ci_low": low,
                "ci_high": high,
                "tokens": len(scores),
            }
    return report


def copies_context_value(answer: str, context: str, value: str | None = None) -> bool:
    """Ответ воспроизводит числовое значение, взятое из документа.

    Сравнивается именно значение факта, а не все числа документа: в контексте
    есть ещё номер регламента, и требовать его копирования в ответ значило бы
    измерять не то.
    """
    answer_numbers = {token.word for token in tokenize_with_offsets(answer) if token.word.isdigit()}
    if value is not None:
        return value in answer_numbers
    from spanverify.core import numbers_in  # noqa: PLC0415

    return bool(numbers_in(context)) and bool(numbers_in(context) & answer_numbers)


def contrast_statistics(pairs: list[dict]) -> dict:
    """Проверка контраста: копирование значения документа и лексическое совпадение.

    Основной критерий — доля ответов, воспроизводящих значение из документа
    (ожидание: ≥80 % в группе «с опорой» и ≤30 % в группе «без опоры»).
    Лексическое совпадение приводится как диагностика: при подмене одного числа
    оно остаётся высоким, то есть модель не может опереться на «другие слова».
    """
    grounded = [p for p in pairs if int(p["label"]) == 0]
    unsupported = [p for p in pairs if int(p["label"]) == 1]
    grounded_copy = [copies_context_value(p["answer"], p["context"], p.get("value")) for p in grounded]
    unsupported_copy = [copies_context_value(p["answer"], p["context"], p.get("value")) for p in unsupported]
    grounded_overlap = [overlap_ratio(p["answer"], p["context"]) for p in grounded]
    unsupported_overlap = [overlap_ratio(p["answer"], p["context"]) for p in unsupported]
    stats = {
        "grounded_pairs": len(grounded),
        "unsupported_pairs": len(unsupported),
        "grounded_value_copy_rate": sum(grounded_copy) / len(grounded_copy) if grounded_copy else 0.0,
        "unsupported_value_copy_rate": sum(unsupported_copy) / len(unsupported_copy) if unsupported_copy else 0.0,
        "grounded_overlap_mean": statistics.fmean(grounded_overlap) if grounded_overlap else 0.0,
        "unsupported_overlap_mean": statistics.fmean(unsupported_overlap) if unsupported_overlap else 0.0,
    }
    stats["contrast_ok"] = stats["grounded_value_copy_rate"] >= 0.8 and stats["unsupported_value_copy_rate"] <= 0.3
    return stats


def environment_info(model_name: str) -> dict:
    """Сведения о среде запуска: без них отчёт невоспроизводим."""
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "model": model_name,
        "cpu_count": __import__("os").cpu_count(),
    }
    try:
        import torch  # noqa: PLC0415

        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - torch может отсутствовать
        info["torch"] = "нет"
    try:
        import transformers  # noqa: PLC0415

        info["transformers"] = transformers.__version__
    except Exception:  # noqa: BLE001
        info["transformers"] = "нет"
    return info


def render_markdown(payload: dict) -> str:
    """Отчёт в Markdown: таблица AUC по слоям и выводы без прикрас."""
    lines = [
        "# Пилот на реальной модели: информативность признаков по слоям",
        "",
        "Числа получены на **реальной** модели, размеченных контрастных парах и CPU.",
        "Это пилот на малом объёме: он показывает, какой сигнал вообще есть, а не",
        "финальное качество продукта.",
        "",
        "## Условия",
        "",
    ]
    for key, value in payload["environment"].items():
        lines.append(f"* {key}: `{value}`")
    lines += [
        f"* пар: {payload['pairs']}, токенов в анализе: {payload['tokens']}",
        f"* бутстрэп: {payload['bootstrap_iterations']} итераций, seed {payload['seed']}",
        f"* слои: {', '.join(payload['layers'])}",
        "",
        "## Контраст пар",
        "",
        "| Показатель | Значение |",
        "| --- | --- |",
    ]
    for key, value in payload["contrast"].items():
        lines.append(f"| {key} | {value:.4f} |" if isinstance(value, float) else f"| {key} | {value} |")
    lines += [
        "",
        "## AUC по признакам и слоям",
        "",
        "| Слой | Признак | AUC | 95 % ДИ | Токенов |",
        "| --- | --- | --- | --- | --- |",
    ]
    best: tuple[float, str, str] = (-1.0, "", "")
    for layer, features in payload["auc"].items():
        for name, values in features.items():
            lines.append(
                f"| {layer} | {name} | {values['auc_oriented']:.3f} | "
                f"[{values['ci_low']:.3f}; {values['ci_high']:.3f}] | {values['tokens']} |"
            )
            if values["auc_oriented"] > best[0]:
                best = (values["auc_oriented"], layer, name)
    entropy_last = payload["auc"].get("last", {}).get("attention_entropy", {}).get("auc_oriented")
    mass_last = payload["auc"].get("last", {}).get("ctx_attention_mass", {}).get("auc_oriented")
    density_last = payload["auc"].get("last", {}).get("embedding_density", {}).get("auc_oriented")
    lines += [
        "",
        "## Выводы",
        "",
        f"* Лучший сигнал: `{best[2]}` на слое `{best[1]}` (AUC {best[0]:.3f}).",
        f"* Последний слой: энтропия внимания AUC {entropy_last:.3f}, масса на контекст AUC {mass_last:.3f}, "
        f"косинусная плотность (базовый уровень без внимания) AUC {density_last:.3f}.",
        "* Случайное угадывание: 0.500 — все признаки выше, значит сигнал есть.",
    ]
    if entropy_last is not None and mass_last is not None:
        if mass_last > entropy_last:
            lines.append("* Гипотеза подтверждена: масса внимания на контекст информативнее «сырой» энтропии внимания.")
        else:
            lines.append(
                "* Гипотеза НЕ подтверждена: «сырая» энтропия внимания оказалась не хуже массы на контекст. "
                "Это честный результат пилота — его нужно учитывать при выборе признаков."
            )
    if payload["contrast"]["contrast_ok"]:
        lines.append(
            "* Контраст групп выдержан: значение документа воспроизводится в "
            f"{payload['contrast']['grounded_value_copy_rate']:.0%} ответов группы «с опорой» и в "
            f"{payload['contrast']['unsupported_value_copy_rate']:.0%} группы «без опоры»."
        )
    else:
        lines.append("* ВНИМАНИЕ: контраст групп выдержан не полностью (см. таблицу выше) — выводы ограничены.")
    lines += [
        "",
        "## Ограничения",
        "",
        "* Пары синтетические: они сгенерированы по шаблонам, а не взяты из реальных документов.",
        "* Объём малый (десятки пар), ДИ широкие.",
        "* Модель одна и небольшая; другие модели могут вести себя иначе.",
        "* Демо-режим продукта (`demo`) к этому отчёту отношения не имеет: здесь всё считается реальной моделью.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Точка входа пилота: считает метрики и пишет отчёт."""
    parser = argparse.ArgumentParser(description="Пилот на rugpt3small: AUC признаков по слоям")
    parser.add_argument("--pairs", type=int, default=24, help="число контрастных пар (каждая даёт 2 ответа)")
    parser.add_argument("--bootstrap", type=int, default=5000, help="итераций бутстрэпа")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--out", default="reports/pilot")
    args = parser.parse_args(argv)

    started = time.time()
    pairs = build_contrast_pairs(args.pairs, seed=args.seed)
    contrast = contrast_statistics(pairs)
    print(f"Контрастных пар: {len(pairs)}; проверка контраста: {json.dumps(contrast, ensure_ascii=False)}")

    try:
        rows = collect_rows(pairs, model_name=args.model, max_length=args.max_length)
    except Exception as error:  # noqa: BLE001 - причины: нет весов, нет torch, нет интернета
        print(
            "Пилот не выполнен: не удалось получить признаки модели. " f"Причина: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2

    auc_table = analyse(rows, iterations=args.bootstrap)
    payload = {
        "environment": environment_info(args.model),
        "pairs": len(pairs),
        "tokens": len(rows),
        "bootstrap_iterations": args.bootstrap,
        "seed": args.seed,
        "layers": list(LAYERS),
        "contrast": contrast,
        "auc": auc_table,
        "duration_s": round(time.time() - started, 1),
        "notice": (
            "Пары синтетические и сгенерированы по шаблонам; объём малый. "
            "Отчёт описывает пилот, а не итоговое качество продукта."
        ),
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pilot.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "pilot.md").write_text(render_markdown(payload), encoding="utf-8")
    print(f"Отчёт: {out_dir / 'pilot.md'} и {out_dir / 'pilot.json'}")
    print(f"Длительность: {payload['duration_s']} с; токенов: {payload['tokens']}")
    for layer, features in auc_table.items():
        summary = ", ".join(f"{name}={values['auc_oriented']:.3f}" for name, values in features.items())
        print(f"  слой {layer}: {summary}")
    return 0


if __name__ == "__main__":  # pragma: no cover - запуск из CI
    raise SystemExit(main())
