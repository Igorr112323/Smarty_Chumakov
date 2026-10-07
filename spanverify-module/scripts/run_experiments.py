"""Эксперименты: обучение, метрики на отложенной части, отчёт в reports/.

    python scripts/run_experiments.py --dataset data/demo_pairs.jsonl --out reports/experiments

Что делает:

1. обучает конвейер (перебор весов, порог маски, голова, калибровка);
2. считает сквозные метрики через публичный ``verify()`` — то же, что отдаёт API;
3. сохраняет отчёт в Markdown и JSON, а также обученные параметры.

Если рядом с корпусом лежат официальные разбиения
(``<папка корпуса>/splits/{train,dev,test}.jsonl``), эксперимент идёт по ним:

* веса, маска, голова и порог выбираются **только на официальной обучающей
  части** (внутри неё ``train()`` делает своё групповое разделение);
* официальный ``test`` измеряется отдельно и в подборе параметров не
  участвует — ни порог, ни маска по нему не выбираются;
* дополнительно в отчёт попадают: метрики по типам пар на официальном тесте,
  доля ложных пометок на чистых парах всего корпуса, кривая маски на
  обучающей части и сравнение правила вердикта (старое «любой фрагмент»
  против нового «одиночный токен маски не обвиняет ответ») на одном тесте.

Все числа берутся из запуска. Если корпус демонстрационный (синтетический), это
прямо печатается в отчёте: научные выводы требуют реальной разметки.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spanverify.dataset import corpus_statistics, generate_pairs, read_pairs, write_pairs  # noqa: E402
from spanverify.engine import VERDICT_ANY_SPAN, Verifier, WeightsBundle  # noqa: E402
from spanverify.features import (  # noqa: E402
    DEFAULT_WEIGHTS,
    HF_MODEL_DEFAULT,
    feature_cache_key,
    load_feature_cache,
    set_feature_cache,
)
from spanverify.train import (  # noqa: E402
    build_risk_fn,
    collect_samples,
    mask_curve,
    save_training_artifacts,
    train,
)

REPORT_TEMPLATE = """# Эксперимент: {dataset}

Режим: `{mode}`. Seed: {seed}. Дата запуска: {timestamp}. Прогон: {run_id}.
Корпус: {pairs_line}.

## Метрики по всему корпусу (сквозной путь `verify()`; `evaluate`)

| Показатель | Значение |
| --- | --- |
| token precision | {precision:.3f} |
| token recall | {recall:.3f} |
| **token F1** | **{f1:.3f}** |
| FPR по токенам | {fpr:.3f} |
| AUC по токенам | {auc:.3f} |
| F1 по ответам (порог {answer_threshold:.3f}) | {answer_f1:.3f} |
| строгий F1 фрагментов (IoU ≥ 0.5, узкая разметка) | {strict_span_f1:.3f} |
| полнота накрытия фрагментов | {containment:.3f} |
| F1 при расширенной разметке | {expanded:.3f} |

Критерий качества мастер-промта (token F1 ≥ 0.90 при FPR ≤ 0.10): **{gate}**.

{official_section}## Выбранные параметры

* веса: {weights}
* порог маски: z={span_z}, floor={span_floor}, cap={span_cap}
* порог решения: {threshold:.4f} (целевой FPR {target_fpr})
* сигнал: {signal}; голова: {head}
* фолды головы (AUC): {folds}

## Оговорка

{synthetic_note}
"""

OFFICIAL_TEMPLATE = """## Официальный test: {test_pairs} пар (не участвовал в выборе параметров)

Порог и маска выбраны только на официальной обучающей части ({train_pairs} пар).

| Показатель | Значение |
| --- | --- |
| token precision | {precision:.3f} |
| token recall | {recall:.3f} |
| **token F1** | **{f1:.3f}** |
| FPR по токенам | {fpr:.3f} |
| AUC по токенам | {auc:.3f} |
| строгий F1 фрагментов (IoU ≥ 0.5) | {strict_span_f1:.3f} |
| полнота накрытия фрагментов | {containment:.3f} |
| средняя ширина фрагмента к истинной | {width_ratio:.1f} |
| вердикт: recall / FPR | {verdict_recall:.3f} / {verdict_fpr:.3f} |
| вердикт: F1 | {verdict_f1:.3f} |

Критерий качества мастер-промта на официальном тесте (token F1 ≥ 0.90 при FPR ≤ 0.10): **{gate}**.

### По типам пар на официальном тесте

| Тип | Пар | Полнота по токенам | Токенный F1 | Фрагментов найдено |
| --- | --- | --- | --- | --- |
{by_type_rows}

### Чистые пары (без расхождений), весь корпус

На {clean_pairs} чистых парах доля ложных пометок вердикта {clean_flagged_share:.3f}
(токенный FPR {clean_token_fpr:.3f}).

### Кривая маски (официальная обучающая часть, сигнал «{mask_signal}»)

Текущая точка: z={mask_z}, floor={mask_floor}, cap={mask_cap}.
{mask_statement}

Лучшие точки сетки при FPR ≤ {target_fpr}:

| z | floor | cap | token F1 | FPR | полнота |
| --- | --- | --- | --- | --- | --- |
{mask_rows}

### Правило вердикта: сравнение на одном официальном тесте

| Правило | Вердикт F1 | Вердикт FPR | Помечено ответов |
| --- | --- | --- | --- |
| старое: любой фрагмент | {legacy_f1:.3f} | {legacy_fpr:.3f} | {legacy_flagged} |
| новое: одиночный токен маски не обвиняет | {new_f1:.3f} | {new_fpr:.3f} | {new_flagged} |

Токенные метрики от правила вердикта не зависят (токены считаются по маске).

"""


def load_records(dataset: Path, pairs: int, seed: int) -> list[dict]:
    """Прочитать корпус или сгенерировать его при отсутствии файла."""
    if dataset.is_file():
        return list(read_pairs(dataset))
    generated = generate_pairs(pairs, seed=seed)
    write_pairs(generated, dataset)
    return [pair.to_dict() for pair in generated]


def _git_commit() -> str:
    """Текущий коммит: числа отчёта обязаны привязываться к состоянию кода."""
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20)
        return result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _read_split(splits_dir: Path, name: str) -> list[dict] | None:
    """Официальное разбиение корпуса, если файл существует и не пуст."""
    path = splits_dir / f"{name}.jsonl"
    if not path.is_file() or not path.stat().st_size:
        return None
    return list(read_pairs(path))


def _is_clean(pair: dict) -> bool:
    """Чистая пара: в разметке нет ни одного фрагмента недостоверности."""
    return not any(int(label[2]) == 1 for label in pair.get("labels", []))


def _metrics_block(metrics: dict) -> dict:
    """Сжать дерево ``evaluate`` до чисел отчёта (округление как в метриках)."""
    tokens, spans, answers, verdicts = (
        metrics["tokens"],
        metrics["spans"],
        metrics["answers"],
        metrics["verdicts"],
    )
    return {
        "pairs": metrics.get("pairs"),
        "tokens": {
            "precision": round(tokens["precision"], 4),
            "recall": round(tokens["recall"], 4),
            "f1": round(tokens["f1"], 4),
            "fpr": round(tokens["fpr"], 4),
            "auc": round(tokens["auc"], 4) if tokens["auc"] == tokens["auc"] else None,
            "tp": tokens["tp"],
            "fp": tokens["fp"],
            "fn": tokens["fn"],
            "tn": tokens["tn"],
        },
        "spans": {
            "strict_f1_iou_0_5": round(spans["f1"], 4),
            "coverage": round(spans["recall_containment"], 4),
            "soft_f1_expanded": round(spans["f1_expanded_labels"], 4),
            "mean_width_ratio": round(spans["mean_width_ratio"], 2),
        },
        "answers": {
            "threshold": round(answers["threshold"], 4),
            "precision": round(answers["precision"], 4),
            "recall": round(answers["recall"], 4),
            "f1": round(answers["f1"], 4),
            "fpr": round(answers["fpr"], 4),
            "auc": round(answers["auc"], 4) if answers["auc"] == answers["auc"] else None,
        },
        "verdicts": {
            key: verdicts[key] for key in ("tp", "fp", "fn", "tn", "precision", "recall", "f1", "fpr", "flagged_share")
        },
    }


def _gate(metrics: dict) -> str:
    """Критерий мастер-промта (ориентир, не приёмка): достигнут или нет."""
    tokens = metrics["tokens"]
    return "ДОСТИГНУТ" if tokens["f1"] >= 0.90 and tokens["fpr"] <= 0.10 else "НЕ достигнут"


def _by_type_on_split(verifier: Verifier, split: list[dict]) -> dict[str, dict]:
    """Полнота и токенный F1 по типам пар на данном сплите (официальный тест)."""
    by_mode: dict[str, list[dict]] = {}
    for pair in split:
        mode_name = str((pair.get("meta") or {}).get("mode") or "unknown")
        by_mode.setdefault(mode_name, []).append(pair)
    report: dict[str, dict] = {}
    for mode_name in sorted(by_mode):
        subset = by_mode[mode_name]
        metrics = verifier.evaluate(subset)
        tokens = metrics["tokens"]
        report[mode_name] = {
            "pairs": len(subset),
            "token_precision": round(tokens["precision"], 4),
            "token_recall": round(tokens["recall"], 4),
            "token_f1": round(tokens["f1"], 4),
            "span_coverage": round(metrics["spans"]["recall_containment"], 4),
            "verdict_recall": metrics["verdicts"]["recall"],
            "verdict_fpr": metrics["verdicts"]["fpr"],
        }
    return report


def _mask_scan(
    verifier: Verifier,
    train_split: list[dict],
    bundle: WeightsBundle,
    target_fpr: float,
) -> dict:
    """Кривая маски на официальной обучающей части и решение о точке.

    Подбор точки — только на обучающей части: тест в нём не участвует. Если
    точка лучше текущей при FPR ≤ target на обучающей части не находится, это
    фиксируется в ``statement``, а не подбирается по тесту.
    """
    signal, risk_fn = build_risk_fn(bundle, entropy_quantile=verifier.entropy_quantile)
    samples, _ = collect_samples(verifier, train_split)
    samples_by_pair: dict[str, list] = {}
    for sample in samples:
        samples_by_pair.setdefault(sample.pair_id, []).append(sample)
    curve = mask_curve(verifier, train_split, risk_fn, samples_by_pair, target_fpr)
    by_params = {(point["span_z"], point["span_floor"], point["span_cap"]): point for point in curve}
    selected = by_params.get((bundle.span_z, bundle.span_floor, bundle.span_cap))
    eligible = [point for point in curve if point["within_target"]]
    best = eligible[0] if eligible else None
    adopted = False
    if best is not None and (
        selected is None
        or not selected["within_target"]
        or (best["token_f1"], -best["token_fpr"]) > (selected["token_f1"], -selected["token_fpr"])
    ):
        bundle.span_z = best["span_z"]
        bundle.span_floor = best["span_floor"]
        bundle.span_cap = best["span_cap"]
        adopted = True
    if best is None:
        statement = (
            f"Точка лучше текущей при FPR ≤ {target_fpr} на обучающей части не найдена: "
            "ни одна точка сетки не укладывается в ограничение."
        )
    elif not adopted:
        statement = (
            f"Точка лучше текущей при FPR ≤ {target_fpr} на обучающей части не найдена: " "выбранная точка остаётся."
        )
    else:
        statement = (
            f"На обучающей части найдена точка лучше при FPR ≤ {target_fpr}: "
            f"z={best['span_z']}, floor={best['span_floor']}, cap={best['span_cap']} "
            f"(token F1 {best['token_f1']:.4f}, FPR {best['token_fpr']:.4f}) — "
            "она и взята; по тесту ничего не подбиралось."
        )
    return {
        "signal": signal,
        "split": "official_train",
        "target_fpr": target_fpr,
        "points_total": len(curve),
        "points_within_target": len(eligible),
        "selected": selected,
        "best_within_target": best,
        "adopted": adopted,
        "statement": statement,
        "curve_top": [point for point in eligible[:5]] + [point for point in curve[:5] if not point["within_target"]],
    }


def main(argv: list[str] | None = None) -> int:
    """Запустить эксперимент и сохран отчёт."""
    parser = argparse.ArgumentParser(description="Эксперименты SpanVerify")
    parser.add_argument("--dataset", default="data/demo_pairs.jsonl")
    parser.add_argument("--out", default="reports/experiments")
    parser.add_argument("--mode", choices=["demo", "hf"], default="demo")
    parser.add_argument(
        "--model",
        default=HF_MODEL_DEFAULT,
        help="модель режима hf; входит в ключ кеша и должна совпадать с предпосчётом",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--target-fpr", type=float, default=0.1)
    parser.add_argument("--pairs", type=int, default=240, help="сгенерировать, если файла нет")
    parser.add_argument(
        "--splits-dir",
        default=None,
        help="каталог официальных разбиений (по умолчанию <папка корпуса>/splits)",
    )
    parser.add_argument(
        "--no-splits",
        action="store_true",
        help="игнорировать официальные разбиения и считать по всему корпусу",
    )
    parser.add_argument(
        "--features-cache",
        action="append",
        default=[],
        metavar="PATH",
        help="кеш предпосчитанных признаков (файл JSONL или каталог с шардами); " "можно указать несколько раз",
    )
    parser.add_argument(
        "--require-cache",
        action="store_true",
        help="завершиться с ошибкой, если хотя бы одна пара не нашлась в кеше",
    )
    parser.add_argument(
        "--compare-span-filter-chars",
        type=int,
        default=0,
        help=(
            "сравнить на обучающей части два варианта маски: текущий и вариант, в котором "
            "фрагмент короче стольких знаков без числа, даты или денежной суммы не помечается; "
            "0 — сравнение выключено; порог не подбирается по тесту"
        ),
    )
    args = parser.parse_args(argv)

    dataset = Path(args.dataset)
    records = load_records(dataset, args.pairs, 1312)

    # Кеш признаков: без него режим hf пересчитывает прямой проход модели в
    # каждом фолде кросс-валидации, и эксперимент не укладывается в лимит job'а
    # (прогон 37446204812 выбрал 120 минут целиком).
    cache = load_feature_cache(args.features_cache, counting=True) if args.features_cache else None
    set_feature_cache(cache)
    if cache:
        print(f"кеш признаков: {len(cache)} ключей из {len(args.features_cache)} пути(ей)", flush=True)
        if args.require_cache:
            missing = []
            for record in records:
                data = record.to_dict() if hasattr(record, "to_dict") else record
                key = feature_cache_key(
                    str(data.get("answer", "")),
                    data.get("context", ""),
                    args.mode,
                    args.model if args.mode == "hf" else "",
                )
                if key not in cache:
                    missing.append(str(data.get("id", "?")))
            if missing:
                print(
                    f"кеш неполон: нет {len(missing)} из {len(records)} " f"(первые: {', '.join(missing[:5])})",
                    file=sys.stderr,
                )
                return 2

    # Официальные разбиения корпуса: если они есть, параметры выбираются
    # только на официальной обучающей части, а официальный тест измеряется
    # отдельно и в выборе параметров не участвует.
    splits: dict[str, list[dict]] = {}
    if not args.no_splits:
        splits_dir = Path(args.splits_dir) if args.splits_dir else dataset.parent / "splits"
        if (splits_dir / "train.jsonl").is_file():
            train_split = _read_split(splits_dir, "train")
            if train_split:
                splits["train"] = train_split
                for name in ("dev", "test"):
                    part = _read_split(splits_dir, name)
                    if part:
                        splits[name] = part
                print(
                    "официальные разбиения: "
                    + ", ".join(f"{name} {len(part)}" for name, part in splits.items())
                    + f" ({splits_dir})",
                    flush=True,
                )

    started = time.time()
    # Верификатор с явным именем модели: иначе берётся config.hf_model
    # (rubert-tiny2), ключ кеша не совпадает с предпосчётом (rugpt3small),
    # и эксперимент либо считает другую модель, либо падает без torch.
    train_verifier = Verifier(
        mode=args.mode,
        model_name=args.model,
        features_cache=cache,
        weights=WeightsBundle(weights=dict(DEFAULT_WEIGHTS), threshold=0.5, mode=args.mode),
    )
    print(f"модель признаков: {train_verifier.model_name}", flush=True)
    learn_pairs = splits.get("train", records)
    report = train(
        learn_pairs,
        mode=args.mode,
        seed=args.seed,
        folds=args.folds,
        target_fpr=args.target_fpr,
        dataset_name=str(dataset) + (" (официальная обучающая часть)" if splits else ""),
        verifier=train_verifier,
    )
    head = report.bundle.head or {}
    head_inline = bool((head.get("model") or {}).get("weights"))
    print(
        "голова в оценке: " + ("модель встроена в бандл" if head_inline else "только файл " + str(head.get("file"))),
        flush=True,
    )
    if cache is not None and args.require_cache and getattr(cache, "misses", 0):
        print(
            f"кеш неполон: промахов {cache.misses} ещё до оценки — эксперимент остановлен",
            file=sys.stderr,
        )
        return 2

    bundle = report.bundle

    # Кривая маски на всей официальной обучающей части: если там есть точка
    # лучше выбранной при FPR ≤ target, берём её (подбор по-прежнему не выходит
    # за пределы обучающей части).
    mask_scan: dict | None = None
    span_filter_chars = int(args.compare_span_filter_chars or 0)
    span_filter_comparison: dict | None = None
    chosen_span_filter = 0
    if splits:
        base_span_params = (bundle.span_z, bundle.span_floor, bundle.span_cap)
        verifier_for_scan = Verifier(
            mode=args.mode,
            model_name=args.model,
            weights=bundle,
            features_cache=cache,
        )
        mask_scan = _mask_scan(verifier_for_scan, splits["train"], bundle, args.target_fpr)
        print(mask_scan["statement"], flush=True)
        baseline_statement = mask_scan["statement"]
        if span_filter_chars > 0:
            # Сравнение двух вариантов маски — только на обучающей части.
            # Вариант 1 (базовый) просканирован выше; перед вторым сканом
            # параметры бандла возвращаются к обученным.
            bundle.span_z, bundle.span_floor, bundle.span_cap = base_span_params
            verifier_filtered_scan = Verifier(
                mode=args.mode,
                model_name=args.model,
                weights=bundle,
                features_cache=cache,
                span_filter_min_chars=span_filter_chars,
            )
            filtered_scan = _mask_scan(verifier_filtered_scan, splits["train"], bundle, args.target_fpr)
            print("вариант с фильтром коротких фрагментов: " + filtered_scan["statement"], flush=True)
            baseline_best = mask_scan["best_within_target"]
            filtered_best = filtered_scan["best_within_target"]

            def _point_key(point: dict | None) -> tuple[float, float]:
                return (point["token_f1"], -point["token_fpr"]) if point else (float("-inf"), float("-inf"))

            # Выбор: выше токены F1 при FPR ≤ target; при равенстве — ниже FPR;
            # при равенстве и этого остаётся базовый вариант. Тест в выборе не
            # участвует. Порог длины фрагмента фиксирован и не подбирается.
            selected = "filtered" if _point_key(filtered_best) > _point_key(baseline_best) else "baseline"
            chosen_scan = filtered_scan if selected == "filtered" else mask_scan
            chosen_span_filter = span_filter_chars if selected == "filtered" else 0
            bundle.span_z, bundle.span_floor, bundle.span_cap = base_span_params
            chosen_best = chosen_scan["best_within_target"]
            if chosen_scan["adopted"] and chosen_best is not None:
                bundle.span_z = chosen_best["span_z"]
                bundle.span_floor = chosen_best["span_floor"]
                bundle.span_cap = chosen_best["span_cap"]
            mask_scan = chosen_scan
            span_filter_comparison = {
                "filter_min_chars": span_filter_chars,
                "rule": (
                    "фрагмент маски короче порога без числа, даты или денежной суммы не помечается; "
                    "порог фиксирован и не подбирался; выбор варианта — только на обучающей части "
                    "при прежнем ограничении token FPR ≤ target"
                ),
                "target_fpr": args.target_fpr,
                "selected": selected,
                "baseline": {
                    "best_within_target": baseline_best,
                    "statement": baseline_statement,
                },
                "filtered": {
                    "best_within_target": filtered_best,
                    "statement": filtered_scan["statement"],
                },
                "selection_key": "max token_f1, затем минимальный token_fpr; тест не участвовал",
            }
            print(
                f"выбран вариант маски: {selected} "
                f"(train: базовый F1={_point_key(baseline_best)[0]:.4f}, "
                f"с фильтром F1={_point_key(filtered_best)[0]:.4f})",
                flush=True,
            )

    verifier = Verifier(
        mode=args.mode,
        model_name=args.model,
        weights=bundle,
        features_cache=cache,
        span_filter_min_chars=chosen_span_filter,
    )
    metrics = verifier.evaluate(records)
    if cache is not None:
        hits = getattr(cache, "hits", 0)
        misses = getattr(cache, "misses", 0)
        print(f"кеш признаков: попаданий {hits}, промахов {misses}", flush=True)
        if misses:
            print(
                "ВНИМАНИЕ: часть пар считана моделью, а не из кеша — ключи разошлись "
                "или кеш неполон; время прогона это покажет",
                file=sys.stderr,
                flush=True,
            )

    # Официальный test, отдельно от всех пар: в выборе порога и маски не участвовал.
    test_block: dict | None = None
    dev_block: dict | None = None
    by_type: dict[str, dict] = {}
    clean_block: dict | None = None
    verdict_comparison: dict | None = None
    if splits:
        test_pairs = splits.get("test") or []
        dev_pairs = splits.get("dev") or []
        if test_pairs:
            test_metrics = verifier.evaluate(test_pairs)
            test_block = _metrics_block(test_metrics)
            by_type = _by_type_on_split(verifier, test_pairs)
            # Сравнение правила вердикта на одном и том же тесте: старое
            # («любой фрагмент») против нового. Токенные метрики от него не
            # зависят, меняются только вердикты уровня ответа.
            legacy = Verifier(
                mode=args.mode,
                model_name=args.model,
                weights=bundle,
                features_cache=cache,
                verdict_rule=VERDICT_ANY_SPAN,
                span_filter_min_chars=chosen_span_filter,
            )
            legacy_metrics = legacy.evaluate(test_pairs)
            legacy_flagged = legacy_metrics["verdicts"]["tp"] + legacy_metrics["verdicts"]["fp"]
            new_flagged = test_metrics["verdicts"]["tp"] + test_metrics["verdicts"]["fp"]
            verdict_comparison = {
                "test_pairs": len(test_pairs),
                "rule_any_span": {
                    key: legacy_metrics["verdicts"][key]
                    for key in ("tp", "fp", "fn", "tn", "precision", "recall", "f1", "fpr", "flagged_share")
                },
                "rule_min_two_tokens": {
                    key: test_metrics["verdicts"][key]
                    for key in ("tp", "fp", "fn", "tn", "precision", "recall", "f1", "fpr", "flagged_share")
                },
                "flagged_answers": {"any_span": legacy_flagged, "min_two_tokens": new_flagged},
                "tokens_identical": (
                    legacy_metrics["tokens"]["tp"] == test_metrics["tokens"]["tp"]
                    and legacy_metrics["tokens"]["fp"] == test_metrics["tokens"]["fp"]
                    and legacy_metrics["tokens"]["fn"] == test_metrics["tokens"]["fn"]
                ),
            }
        if dev_pairs:
            dev_block = _metrics_block(verifier.evaluate(dev_pairs))
        clean = [pair for pair in records if _is_clean(pair)]
        if clean:
            clean_metrics = verifier.evaluate(clean)
            clean_block = {
                "pairs": len(clean),
                "token_fpr": round(clean_metrics["tokens"]["fpr"], 4),
                "verdict_fpr": clean_metrics["verdicts"]["fpr"],
                "flagged_pairs": clean_metrics["verdicts"]["tp"] + clean_metrics["verdicts"]["fp"],
                "flagged_share": clean_metrics["verdicts"]["flagged_share"],
                "note": (
                    "чистые пары всего корпуса (без фрагментов недостоверности в разметке); "
                    "числа — измерение фиксированным бандлом, не подбор"
                ),
            }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = save_training_artifacts(report, out_dir / "weights.json", root=out_dir)

    tokens = metrics["tokens"]
    spans = metrics["spans"]
    answers = metrics["answers"]
    gate = _gate(metrics)
    synthetic_note = (
        "Корпус демонстрационный (синтетические пары). Числа описывают "
        "работоспособность и самосогласованность конвейера, а не качество на "
        "реальных документах: для научных выводов нужен режим `hf` и разметка "
        "настоящих пар."
        if args.mode == "demo"
        else "Режим `hf`: признаки считаются реальной моделью. Данные всё равно "
        "синтетические, если корпус не был заменён на размеченные документы."
    )
    stats = corpus_statistics(records)

    official_section = ""
    if test_block is not None and mask_scan is not None:
        by_type_rows = "\n".join(
            f"| {mode_name} | {row['pairs']} | {row['token_recall']:.3f} | {row['token_f1']:.3f} "
            f"| {row['span_coverage']:.3f} |"
            for mode_name, row in by_type.items()
        )
        mask_rows = "\n".join(
            f"| {point['span_z']} | {point['span_floor']} | {point['span_cap']} "
            f"| {point['token_f1']:.4f} | {point['token_fpr']:.4f} | {point['token_recall']:.4f} |"
            for point in mask_scan["curve_top"]
        )
        legacy = verdict_comparison["rule_any_span"]
        new = verdict_comparison["rule_min_two_tokens"]
        official_section = OFFICIAL_TEMPLATE.format(
            test_pairs=test_block["pairs"],
            train_pairs=len(splits["train"]),
            precision=test_block["tokens"]["precision"],
            recall=test_block["tokens"]["recall"],
            f1=test_block["tokens"]["f1"],
            fpr=test_block["tokens"]["fpr"],
            auc=test_block["tokens"]["auc"] or 0.0,
            strict_span_f1=test_block["spans"]["strict_f1_iou_0_5"],
            containment=test_block["spans"]["coverage"],
            width_ratio=test_block["spans"]["mean_width_ratio"],
            verdict_recall=test_block["verdicts"]["recall"],
            verdict_fpr=test_block["verdicts"]["fpr"],
            verdict_f1=test_block["verdicts"]["f1"],
            gate=(
                "ДОСТИГНУТ"
                if test_block["tokens"]["f1"] >= 0.90 and test_block["tokens"]["fpr"] <= 0.10
                else "НЕ достигнут"
            ),
            by_type_rows=by_type_rows,
            clean_pairs=clean_block["pairs"] if clean_block else 0,
            clean_flagged_share=clean_block["flagged_share"] if clean_block else 0.0,
            clean_token_fpr=clean_block["token_fpr"] if clean_block else 0.0,
            mask_signal=mask_scan["signal"],
            mask_z=bundle.span_z,
            mask_floor=bundle.span_floor,
            mask_cap=bundle.span_cap,
            mask_statement=mask_scan["statement"],
            target_fpr=args.target_fpr,
            mask_rows=mask_rows,
            legacy_f1=legacy["f1"],
            legacy_fpr=legacy["fpr"],
            legacy_flagged=verdict_comparison["flagged_answers"]["any_span"],
            new_f1=new["f1"],
            new_fpr=new["fpr"],
            new_flagged=verdict_comparison["flagged_answers"]["min_two_tokens"],
        )
    text = REPORT_TEMPLATE.format(
        dataset=dataset,
        mode=args.mode,
        seed=args.seed,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        run_id=os.environ.get("GITHUB_RUN_ID", "локальный запуск"),
        pairs_line=f"{stats['pairs']} пар, типы {stats['kinds']}",
        precision=tokens["precision"],
        recall=tokens["recall"],
        f1=tokens["f1"],
        fpr=tokens["fpr"],
        auc=tokens["auc"],
        answer_f1=answers["f1"],
        answer_threshold=answers["threshold"],
        containment=spans["recall_containment"],
        expanded=spans["f1_expanded_labels"],
        strict_span_f1=spans["f1"],
        gate=gate,
        official_section=official_section,
        weights={k: round(v, 3) for k, v in bundle.weights.items()},
        span_z=bundle.span_z,
        span_floor=bundle.span_floor,
        span_cap=bundle.span_cap,
        threshold=bundle.threshold,
        target_fpr=bundle.target_fpr,
        signal=(bundle.meta or {}).get("signal", "—"),
        head=(bundle.head or {}).get("type", "none"),
        folds=[round(fold["auc"], 3) for fold in (report.folds or [])],
        synthetic_note=synthetic_note,
    )
    markdown_path = out_dir / "experiment.md"
    markdown_path.write_text(text, encoding="utf-8")
    payload = {
        "dataset": str(dataset),
        "mode": args.mode,
        "model": args.model,
        "seed": args.seed,
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "git_commit": _git_commit(),
        "duration_s": round(time.time() - started, 2),
        "corpus": stats,
        "splits": (
            {
                "source": str(Path(args.splits_dir) if args.splits_dir else dataset.parent / "splits"),
                "sizes": {name: len(part) for name, part in splits.items()},
                "rule": "порог и маска выбраны только на официальной обучающей части; тест в выборе не участвовал",
            }
            if splits
            else None
        ),
        "metrics": metrics,
        "test_official": test_block,
        "dev_official": dev_block,
        "by_type_test": by_type,
        "clean_pairs": clean_block,
        "mask_scan": mask_scan,
        "span_filter_comparison": span_filter_comparison,
        "verdict_rule_comparison": verdict_comparison,
        "bundle": bundle.to_dict(),
        "folds": report.folds,
        "gate": gate,
        "gate_test_official": _gate({"tokens": test_block["tokens"]}) if test_block else None,
        "head_inline": head_inline,
        "artifacts": {name: str(path) for name, path in written.items()},
    }
    json_path = out_dir / "experiment.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Отчёт: {markdown_path}")
    print(f"Данные: {json_path}")
    print(f"token F1={tokens['f1']:.3f} FPR={tokens['fpr']:.3f} — критерий {gate}")
    if test_block is not None:
        print(
            f"официальный test ({test_block['pairs']} пар): "
            f"token F1={test_block['tokens']['f1']:.3f} FPR={test_block['tokens']['fpr']:.3f} "
            f"вердикт F1={test_block['verdicts']['f1']:.3f} FPR={test_block['verdicts']['fpr']:.3f}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - утилита
    raise SystemExit(main())
