"""Диагностика на валидации: операционная точка, сглаживание, вклад семейств, разброс seed'ов.

Зачем отдельный инструмент. Критерий ТЗ не достигнут с зазором 0,004959 по F1, и следующий шаг
обязан отвечать на вопрос «за счёт чего», а не «подкрутим до цифры». Тестовая часть при этом
закрыта (она уже использована для оценки v1), поэтому всё измеряется на валидации A3 dev:

* где лежит операционная точка — F1 при замороженном пороге, максимум F1 при FPR ≤ 0,40 и
  максимум F1 без ограничения (последнее — oracle по val, результатом не является);
* что дают сглаживание оценок по окну и отказ от целого семейства признаков; сравнивается
  **двумя** способами: при одном и том же пороге (сравнимы веса) и при собственном лучшем
  пороге с ограничением по FPR (сравнимы масштабы оценок — у сглаженных моделей он другой);
* измеряем ли вообще разброс по seed'ам: `hf_protocol.fit_model` для `logreg` seed выбрасывает
  (`del seed`), поэтому нулевое СКО по пяти seed'ам — свойство харнесса, а не устойчивость модели.

Чего инструмент не делает: не открывает тестовую выборку (файл с `test` в имени отвергается и
как сплит, и как файл кеша), не выбирает конфигурацию поставки, не пишет в `reports/hf_final/*`
и не меняет `config/hf_experiments.json`. Его вывод — диагностика: если по нему решат что-то
менять в модели, конфигурация обязана попасть в журнал гипотез цикла (не более 10 на val),
а не использоваться «в обход» — это закреплено в `docs/ПРОТОКОЛ_И_ТЗ_v2.md`.

Команда (кеш сетки v3 — артефакты `grid-a3-*` прогона «Протокол hf (критерий ТЗ)»):

    python3 scripts/val_diagnostics.py --grid-cache reports/hf-cache-a3 \
        --splits data/corpus_a3/splits --config config/hf_experiments.json \
        --baseline wide-linear --threshold-manifest reports/hf_final/manifest.json \
        --out-json reports/hf_protocol/val_diagnostics.json \
        --out-md reports/hf_protocol/val_diagnostics.md --annotate
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT / "scripts"))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import hf_protocol  # noqa: E402

from spanverify.hf_grid import GRID_FEATURE_NAMES  # noqa: E402

PROTOCOL = "val-diagnostics-v1"
WINDOWS = (0, 1, 2, 3)
FPR_CAP = 0.40
# Ключи конфигурации, которые диагностика обязана сохранить: иначе она меряет другую модель.
PASSTHROUGH_PARAMS = ("epochs", "learning_rate", "l2", "folds", "trees", "depth")


def load_feature_doc() -> dict[str, tuple[str, str]]:
    """Группы признаков берутся из генератора `docs/ПРИЗНАКИ.md`, а не переписываются здесь."""
    path = MODULE_ROOT / "scripts" / "render_features_doc.py"
    spec = importlib.util.spec_from_file_location("render_features_doc", path)
    if spec is None or spec.loader is None:  # pragma: no cover - отказ только в битном дереве
        raise SystemExit(f"не удалось загрузить {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.FEATURE_DOC)


def guard_no_test(paths: list[Path], label: str) -> None:
    """Отказ по имени файла: «случайно посчитать на тесте» должно быть невозможно."""
    forbidden = sorted(path.name for path in paths if "test" in path.name.lower())
    if forbidden:
        raise SystemExit(f"{label}: файлы тестовой выборки {forbidden} — диагностика идёт только на val")


def load_val_records(splits_dir: Path) -> tuple[list[dict[str, Any]], Path]:
    val_path = hf_protocol.val_path_of(splits_dir)
    guard_no_test([val_path], "сплит для оценки")
    return hf_protocol.load_split(val_path), val_path


def families(names: list[str], doc: dict[str, tuple[str, str]], *, min_size: int = 2) -> dict[str, list[str]]:
    """Семейства по первому элементу описания признака; одиночки не убираются (это шум)."""
    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(doc.get(name, ("прочее", ""))[0], []).append(name)
    return {key: value for key, value in sorted(groups.items()) if len(value) >= min_size}


def frontier(rows: list[dict[str, Any]], cap: float) -> dict[str, Any]:
    """Кривая F1–FPR по уникальным значениям оценок; возвращает две точки — с FPR-потолком и без."""
    labels = [int(row["gold"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return {"error": "вырожденная разметка: нужны и положительные, и отрицательные токены"}
    order = sorted(range(len(scores)), key=lambda index: -scores[index])
    tp = fp = 0
    best_capped: tuple[float, float, float] | None = None
    best_any: tuple[float, float, float] | None = None
    last_value = None
    for index in order:
        if labels[index]:
            tp += 1
        else:
            fp += 1
        # Порог «>= value»: группа равных оценок открывается целиком, поэтому считать надо
        # на последом элементе группы, иначе F1 зависел бы от порядка внутри связей.
        if index + 1 == len(order) or scores[order[index + 1]] != scores[index]:
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / positives
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            fpr = fp / negatives
            if fpr <= cap and (best_capped is None or f1 > best_capped[0] + 1e-12):
                best_capped = (f1, fpr, scores[index])
            if best_any is None or f1 > best_any[0] + 1e-12:
                best_any = (f1, fpr, scores[index])
        last_value = scores[index]
    if best_any is None or best_capped is None:
        return {"error": "ни одна операционная точка не попала под ограничение по FPR"}
    _ = last_value
    return {
        "distinct_candidates": len({scores[index] for index in order}),
        "capped": {"f1": round(best_capped[0], 6), "fpr": round(best_capped[1], 6), "threshold": best_capped[2]},
        "unconstrained": {
            "f1": round(best_any[0], 6),
            "fpr": round(best_any[1], 6),
            "threshold": best_any[2],
            "warning": "oracle по val: максимум выбран по той же выборке, на которой измерен; не результат",
        },
    }


def experiment_entry(
    name: str, features: list[str], window: int, base: dict[str, Any], hypothesis: str
) -> dict[str, Any]:
    """Конфигурация в формате сетки шага 1–2 (`hf_protocol.run_experiment`)."""
    return {
        "name": name,
        "hypothesis": hypothesis,
        "change": "диагностика на val: порог общий, меняется только состав признаков/окно",
        "features": list(features),
        "window": int(window),
        "merge_gap": int(base.get("merge_gap", 2)),
        "classifier": str(base.get("classifier", "logreg")),
        "params": {key: base[key] for key in PASSTHROUGH_PARAMS if key in base},
    }


def summarize(block: dict[str, Any], cap: float) -> dict[str, Any]:
    """Метрики при том же пороге + собственная лучшая точка под FPR-потолком."""
    eval_block = block["eval"]
    tokens = eval_block["tokens"]
    spans = eval_block.get("spans") or {}
    out: dict[str, Any] = {
        "threshold": block["threshold"],
        "token_f1": tokens.get("f1"),
        "token_precision": tokens.get("precision"),
        "token_recall": tokens.get("recall"),
        "token_fpr": tokens.get("fpr"),
        "token_auc": tokens.get("auc"),
        "span_f1": spans.get("f1"),
        "model_sha256": block["model_sha256"],
        "threshold_selection": block["threshold_selection"],
    }
    rows = block.get("_rows") or []
    if rows:
        shape = frontier(rows, cap)
        if "error" in shape:
            out["capped_error"] = shape["error"]
        else:
            out["capped_f1"] = shape["capped"]["f1"]
            out["capped_fpr"] = shape["capped"]["fpr"]
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    doc = load_feature_doc()

    cache_dir = Path(args.grid_cache)
    cache_paths = hf_protocol._cache_paths(cache_dir)
    guard_no_test(cache_paths, "каталог кеша сетки")
    cache = hf_protocol.load_grid_cache(cache_paths)

    splits_dir = Path(args.splits)
    guard_no_test([splits_dir / "train.jsonl"], "обучающая часть")
    train = hf_protocol.load_split(splits_dir / "train.jsonl")
    val, val_path = load_val_records(splits_dir)

    grid = json.loads(Path(args.config).read_text(encoding="utf-8"))
    known = {str(item.get("name")): item for item in grid.get("experiments") or []}
    if args.baseline not in known:
        raise SystemExit(f"базовой линии «{args.baseline}» нет в {args.config}: есть {sorted(known)}")
    base = known[args.baseline]
    base_features = [str(name) for name in base.get("features") or []]
    if not base_features:
        raise SystemExit(f"у базовой линии «{args.baseline}» пуст список признаков")
    unknown = sorted(set(base_features) - set(GRID_FEATURE_NAMES))
    if unknown:
        raise SystemExit(f"базовая линия ссылается на вне сетки: {unknown}")

    frozen: float | None = None
    threshold_source = "подбор по групповой CV по документам на train (тот же seed, что и обучение)"
    if args.threshold_manifest:
        manifest = json.loads(Path(args.threshold_manifest).read_text(encoding="utf-8"))
        frozen = float(manifest["threshold"])
        threshold_source = (
            f"замороженный порог {frozen} из {Path(args.threshold_manifest).name} — общий для всех "
            "конфигураций, чтобы ΔF1 не смешивался с переселекцией порога"
        )

    seeds = [int(item) for item in str(args.seeds).split(",") if item.strip()]
    if not seeds:
        raise SystemExit("пуст --seeds: разброс по seed'ам обязан быть заявлен явно")

    def evaluate(experiment: dict[str, Any], seed: int) -> dict[str, Any]:
        block = hf_protocol.run_experiment(cache, train, val, experiment, seed, fixed_threshold=frozen, keep_rows=True)
        if "error" in block:
            raise SystemExit(f"{experiment['name']}: {block['error']}")
        return block

    results: dict[str, Any] = {}
    window_base = int(base.get("window", 0))

    baseline_block = evaluate(
        experiment_entry("baseline", base_features, window_base, base, "базовая линия = лучшая конфигурация шага 1–2"),
        seeds[0],
    )
    results["baseline"] = summarize(baseline_block, args.fpr_cap) | {
        "name": args.baseline,
        "features": base_features,
        "window": window_base,
        "seed": seeds[0],
    }
    base_f1 = float(results["baseline"]["token_f1"] or 0.0)
    base_capped = results["baseline"].get("capped_f1")
    results["baseline"]["delta_f1"] = 0.0

    if args.anchor_cv and frozen is not None:
        # Единственная лишняя CV-процедура в диагностике: без неё базовая линия — «модель,
        # обученная на train, с порогом шага 3», и сравнивать её с числами sweep.json
        # нельзя (та модель обучалась на train+val, и порог у неё свой).
        anchor = hf_protocol.run_experiment(
            cache,
            train,
            val,
            experiment_entry("baseline-anchor", base_features, window_base, base, "якорь: подбор порога по протоколу"),
            seeds[0],
            keep_rows=False,
        )
        anchor_tokens = anchor["eval"]["tokens"]
        results["baseline"]["anchor_cv"] = {
            "threshold": anchor["threshold"],
            "token_f1": anchor_tokens.get("f1"),
            "token_fpr": anchor_tokens.get("fpr"),
            "token_auc": anchor_tokens.get("auc"),
            "note": (
                "обучение на train, порог по групповой CV по документам (seed как у всех) — "
                "сопоставимо с best_selection в reports/hf_protocol/sweep.json"
            ),
        }

    rows = baseline_block.get("_rows") or []
    results["frontier"] = frontier(rows, args.fpr_cap)
    results["frontier"]["protocol_point"] = {
        "f1": results["baseline"]["token_f1"],
        "fpr": results["baseline"]["token_fpr"],
        "threshold": results["baseline"]["threshold"],
    }
    results["frontier"]["gap_to_criterion_on_val"] = round(args.criterion_f1 - base_f1, 6)

    def delta(current: Any, reference: Any) -> Any:
        if current is None or reference is None:
            return None
        return round(float(current) - float(reference), 6)

    smoothing: list[dict[str, Any]] = []
    for window in WINDOWS:
        block = evaluate(
            experiment_entry(f"smooth-{window}", base_features, window, base, f"сглаживание оценок по окну {window}"),
            seeds[0],
        )
        summary = summarize(block, args.fpr_cap)
        summary |= {
            "window": window,
            "delta_f1": delta(summary["token_f1"], base_f1),
            "delta_capped_f1": delta(summary.get("capped_f1"), base_capped),
        }
        smoothing.append(summary)
    results["smoothing"] = smoothing

    ablation_pool = list(base_features) if args.ablate == "baseline" else list(GRID_FEATURE_NAMES)
    ablations: list[dict[str, Any]] = []
    for group, members in families(ablation_pool, doc).items():
        kept = [name for name in ablation_pool if name not in set(members)]
        if len(kept) < 3:
            continue
        block = evaluate(
            experiment_entry(
                f"minus-{group}", kept, window_base, base, f"без семейства «{group}» ({len(members)} колонок)"
            ),
            seeds[0],
        )
        summary = summarize(block, args.fpr_cap)
        summary |= {
            "family": group,
            "removed": members,
            "kept": len(kept),
            "delta_f1": delta(summary["token_f1"], base_f1),
            "delta_capped_f1": delta(summary.get("capped_f1"), base_capped),
        }
        ablations.append(summary)
    results["ablations"] = ablations

    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        block = evaluate(experiment_entry(f"seed-{seed}", base_features, window_base, base, "разброс по seed'ам"), seed)
        summary = summarize(block, args.fpr_cap)
        summary["seed"] = seed
        per_seed.append(summary)
    f1_values = [item["token_f1"] for item in per_seed]
    results["seeds"] = {
        "requested": seeds,
        "per_seed": per_seed,
        "token_f1_values": f1_values,
        "threshold_values": [item["threshold"] for item in per_seed],
        "model_sha256_unique": sorted({item["model_sha256"] for item in per_seed}),
        "std_token_f1": _std(f1_values),
        "std_threshold": _std([item["threshold"] for item in per_seed]),
        "note": (
            "hf_protocol.fit_model для classifier=logreg игнорирует seed (`del seed`), а при "
            "--threshold-manifest порог один и тот же: нулевое СКО здесь означает «харнесс "
            "детерминирован», и не означает «модель устойчива». В v2 разброс вносит бутстрап по "
            "документам (параметр bootstrap_docs в hf_protocol_v2.fit_and_score), а перед шагом 3 "
            "он проверяется командой `python3 scripts/hf_protocol_v2.py seedcheck` — требование "
            "записано в docs/METRIC_SPEC_v2.md (раздел 4) и закрыто тестом "
            "tests/test_v2_seed_variance.py"
        ),
    }

    return {
        "protocol": PROTOCOL,
        "purpose": (
            "диагностика на val: операционная точка, сглаживание, вклад семейств, разброс по "
            "seed'ам; не гипотеза цикла v2 и не оценка критерия"
        ),
        "not_for_criterion": True,
        "test_not_read": True,
        "rule": "оценка — только val (A3 dev); файлы с `test` в имени отвергаются до вычислений",
        "reading": (
            "у каждой конфигурации два числа: F1 при общем замороженном пороге (сравнимы веса) и "
            "максимум F1 при FPR ≤ 0,40 на val по самой конфигурации (сравнимы масштабы оценок); "
            "оба — на валидации, второй к тому же oracle и в отчётах о критерии не используется"
        ),
        "split": {"name": "val (A3 dev)", "file": str(val_path), "pairs": len(val), "tokens": len(rows)},
        "train": {"pairs": len(train), "used_for": "обучение модели и (если не заморожен) выбор порога"},
        "labels": {
            "positive_tokens": sum(int(row["gold"]) for row in rows),
            "negative_tokens": sum(1 - int(row["gold"]) for row in rows),
        },
        "model": hf_protocol.cache_model_provenance(cache),
        "baseline": args.baseline,
        "criterion_f1": args.criterion_f1,
        "threshold_source": threshold_source,
        "fpr_cap": args.fpr_cap,
        "ablate": args.ablate,
        "windows": list(WINDOWS),
        "inputs": {
            "grid_cache": str(cache_dir),
            "cache_files": sorted(path.name for path in cache_paths),
            "splits": str(splits_dir),
            "config": str(args.config),
            "config_sha256": hf_protocol.sha256_file(Path(args.config)),
            "splits_sha256": hf_protocol.sha256_file(val_path),
            "threshold_manifest": args.threshold_manifest or "не задан",
        },
        "code_commit": hf_protocol._git_commit(),
        "results": results,
    }


def _std(values: list[Any]) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    if len(numbers) < 2:
        return None
    return round(statistics.pstdev(numbers), 6)


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _delta(value: Any) -> str:
    return "—" if value is None else f"{value:+.6f}"


def markdown(payload: dict[str, Any]) -> str:
    results = payload["results"]
    split = payload["split"]
    front = results["frontier"]
    baseline = results["baseline"]
    lines = [
        "# Диагностика на валидации: операционная точка, сглаживание, семейства, seed'ы",
        "",
        f"Протокол `{payload['protocol']}`. {payload['purpose']}.",
        "",
        f"Тестовая выборка не читалась (`test_not_read = true`). Выборка оценки: пар `{split['pairs']}`, "
        f"токенов ответа `{split['tokens']}`, положительных `{payload['labels']['positive_tokens']}`. "
        f"Модель кеша: `{payload['model']['id']}` (revision `{str(payload['model']['revision'])[:9]}`). "
        f"Режим порога: {payload['threshold_source']}.",
        "",
        "## Базовая линия",
        "",
        f"Конфигурация `{payload['baseline']}`, колонок признака: `{len(baseline['features'])}`, "
        f"окно сглаживания: `{baseline['window']}`, seed: `{baseline['seed']}`.",
        "",
        "| Метрика | значение |",
        "|---|---|",
        f"| F1 (токены) | {_fmt(baseline['token_f1'])} |",
        f"| Precision | {_fmt(baseline['token_precision'])} |",
        f"| Recall | {_fmt(baseline['token_recall'])} |",
        f"| FPR | {_fmt(baseline['token_fpr'])} |",
        f"| AUC | {_fmt(baseline['token_auc'])} |",
        f"| F1 (фрагменты) | {_fmt(baseline['span_f1'])} |",
        f"| порог | {_fmt(baseline['threshold'])} |",
        f"| до критерия на val не хватает | {_fmt(front.get('gap_to_criterion_on_val'))} |",
        "",
        "## Операционная точка",
        "",
        "| Точка | F1 | FPR | Порог |",
        "|---|---|---|---|",
        f"| по протоколу (тот же порог, что в шаге 3) | {_fmt(front['protocol_point']['f1'])} "
        f"| {_fmt(front['protocol_point']['fpr'])} | {_fmt(front['protocol_point']['threshold'])} |",
        f"| максимум F1 при FPR ≤ {payload['fpr_cap']} | {_fmt(front['capped']['f1'])} "
        f"| {_fmt(front['capped']['fpr'])} | {_fmt(front['capped']['threshold'])} |",
        f"| максимум F1 без ограничения | {_fmt(front['unconstrained']['f1'])} "
        f"| {_fmt(front['unconstrained']['fpr'])} | {_fmt(front['unconstrained']['threshold'])} |",
        "",
        f"> {front['unconstrained']['warning']}",
        "",
        "## Сглаживание оценок по окну",
        "",
        "ΔF1 (порог) — при общем пороге; ΔF1 (best) — при собственном максимуме под FPR-потолком.",
        "",
        "| Окно | F1 | FPR | ΔF1 (порог) | ΔF1 (best) |",
        "|---|---|---|---|---|",
    ]
    for item in results["smoothing"]:
        lines.append(
            f"| {item['window']} | {_fmt(item['token_f1'])} | {_fmt(item['token_fpr'])} "
            f"| {_delta(item['delta_f1'])} | {_delta(item['delta_capped_f1'])} |"
        )
    lines += [
        "",
        "## Вклад семейств (убрать семейство целиком)",
        "",
        f"Набор для убирания: `{payload['ablate']}` (pool из {len(baseline['features']) if payload['ablate'] == 'baseline' else len(GRID_FEATURE_NAMES)} колонок).",
        "",
        "| Семейство | Убрано колонок | Оставлено | F1 | ΔF1 (порог) | ΔF1 (best) |",
        "|---|---|---|---|---|---|",
    ]
    for item in results["ablations"]:
        lines.append(
            f"| {item['family']} | {len(item['removed'])} | {item['kept']} | {_fmt(item['token_f1'])} "
            f"| {_delta(item['delta_f1'])} | {_delta(item['delta_capped_f1'])} |"
        )
    seeds = results["seeds"]
    lines += [
        "",
        "## Разброс по seed'ам",
        "",
        f"Запрошено: `{len(seeds['requested'])}` ({', '.join(str(item) for item in seeds['requested'])}); "
        f"различающихся хешей модели: `{len(seeds['model_sha256_unique'])}`; СКО F1: "
        f"`{_fmt(seeds['std_token_f1'], 6)}`; СКО порога: `{_fmt(seeds['std_threshold'], 6)}`.",
        "",
        f"> {seeds['note']}",
        "",
        "| Seed | F1 | FPR | Порог |",
        "|---|---|---|---|",
    ]
    for item in seeds["per_seed"]:
        lines.append(
            f"| {item['seed']} | {_fmt(item['token_f1'])} | {_fmt(item['token_fpr'])} | {_fmt(item['threshold'])} |"
        )
    lines += [
        "",
        "## Как это читать",
        "",
        payload["reading"].rstrip(".").capitalize() + ".",
        "- Нулевое СКО по seed'ам означает «харнесс детерминирован», а не «модель устойчива».",
        "- Отрицательный ΔF1 семейства — не разрешение его удалить: удаление меняет сетку,",
        "  зафиксированную протоколом, и требует решения заказчика (`docs/DECISIONS_NEEDED.md`).",
        "",
    ]
    return "\n".join(lines)


def annotate(payload: dict[str, Any]) -> None:
    results = payload["results"]
    baseline = results["baseline"]
    front = results["frontier"]
    print(
        f"::notice title=базовая линия (val)::F1={_fmt(baseline['token_f1'])} FPR={_fmt(baseline['token_fpr'])} "
        f"Recall={_fmt(baseline['token_recall'])} Precision={_fmt(baseline['token_precision'])} "
        f"порог={_fmt(baseline['threshold'])}"
    )
    print(
        f"::notice title=операционная точка::макс F1 при FPR≤{payload['fpr_cap']}={_fmt(front['capped']['f1'])} "
        f"(FPR={_fmt(front['capped']['fpr'])}), oracle={_fmt(front['unconstrained']['f1'])} — не результат"
    )
    print(
        "::notice title=сглаживание::"
        + " | ".join(
            f"окно {item['window']}: F1 {_fmt(item['token_f1'])} (Δ {_delta(item['delta_f1'])})"
            for item in results["smoothing"]
        )
    )
    print(
        "::notice title=вклад семейств::"
        + " | ".join(f"−{item['family']}: Δ {_delta(item['delta_f1'])}" for item in results["ablations"])
    )
    seeds = results["seeds"]
    print(
        f"::notice title=разброс по seed'ам::СКО F1={_fmt(seeds['std_token_f1'])} "
        f"СКО порога={_fmt(seeds['std_threshold'])} различных моделей={len(seeds['model_sha256_unique'])} "
        f"из {len(seeds['requested'])}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Диагностика на валидации (без доступа к тестовой части)")
    parser.add_argument("--grid-cache", required=True, help="каталог или файл кеша сетки v3")
    parser.add_argument("--splits", default="data/corpus_a3/splits", help="каталог разбиения по документам")
    parser.add_argument(
        "--config", default="config/hf_experiments.json", help="сетка шага 1–2 (источник базовой линии)"
    )
    parser.add_argument("--baseline", default="wide-linear", help="имя базовой линии в --config")
    parser.add_argument(
        "--threshold-manifest",
        default="",
        help=(
            "путь к manifest.json, откуда взять замороженный порог шага 3; пусто (по умолчанию) — "
            "каждая конфигурация подбирает порог тем же правилом протокола (групповая CV по "
            "документам на train), и тогда базовая линия сравнима с reports/hf_protocol/sweep.json"
        ),
    )
    parser.add_argument(
        "--no-anchor-cv",
        dest="anchor_cv",
        action="store_false",
        help="не считать базовую линию с собственным подбором порога (экономит время, когда порог заморожен)",
    )
    parser.add_argument("--seeds", default="42,43,44,45,46", help="seed'ы через запятую")
    parser.add_argument("--fpr-cap", type=float, default=FPR_CAP, help="потолок FPR для точки «максимум F1»")
    parser.add_argument(
        "--ablate", choices=("baseline", "grid"), default="grid", help="на каком наборе убирать семейства"
    )
    parser.add_argument("--criterion-f1", type=float, default=0.60, help="порог критерия (для строки «не хватает»)")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--annotate", action="store_true", help="ключевые числа — в аннотации прогона")
    args = parser.parse_args(argv)

    payload = run(args)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(markdown(payload) + "\n", encoding="utf-8")
    if args.annotate:
        annotate(payload)
    results = payload["results"]
    print(
        f"диагностика: токенов val {payload['split']['tokens']} | базовая F1 {_fmt(results['baseline']['token_f1'])} "
        f"| макс F1 при FPR≤{payload['fpr_cap']} {_fmt(results['frontier']['capped']['f1'])} "
        f"| семейств проверено {len(results['ablations'])} | СКО F1 по seed'ам {_fmt(results['seeds']['std_token_f1'])}"
    )
    print(f"записано: {out_json} и {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
