#!/usr/bin/env python3
"""Полное обучение режима ``hf`` на корпусе A3 (фрагменты реальных актов).

Зачем отдельный скрипт
---------------------

``spanverify train`` обучает конвейер целиком, но состав признаков головы у него
фиксирован (шесть столбцов). Чтобы поднять качество на реальных актах, признаки
нужно **добавлять**, а не подбирать порог: отбор идёт по AUC на валидации, а
официальный тест в отборе не участвует вовсе. Этот скрипт и делает отбор, и
обучает финальную модель, и пишет артефакты с суффиксом ``_hf`` — боевые
``config/weights.json``, ``config/head.json`` и ``config/participation.json`` он
не трогает.

Протокол (он же — протокол CI-прогона)
-------------------------------------

1. Корпус читается вместе с официальными разбиениями
   ``<папка корпуса>/splits/{train,dev,test}.jsonl``; если их нет — используется
   групповое разбиение самого ``train()`` (seed зафиксирован).
2. **Отбор признаков** — жадный прямой поиск на фиксированной стратифицированной
   подвыборке обучающей части (``--select-pairs``). Критерий — AUC головы на
   внутренней валидации (``head.auc_test``), при равенстве — out-of-fold AUC.
   Признак остаётся, только если увеличил метрику больше чем на ``--min-auc-gain``.
   Порог маски и порог решения при отборе не перебираются ради метрик: они
   обучаются штатно и оцениваются так, как их получит API.
3. **Финальное обучение** — на всей обучающей части с отобранными признаками.
4. **Оценка** — сквозным ``Verifier.evaluate`` на официальном тесте и на всём
   корпусе; сравнение «база против расширенной головы» попадает в отчёт.
5. Артефакты: ``config/weights_hf.json``, ``config/head_hf.json``,
   ``config/participation_hf.json``, ``reports/hf_a3_train.log``,
   ``reports/hf_a3_eval.md``, ``reports/hf_a3_eval.json``.

Режим ``hf`` требует torch и весов модели; в песочнице и в обычном CI их нет,
поэтому признаках используется кеш предпосчёта
(``scripts/precompute_features.py``), а ``--require-cache`` делает промах по кешу
ошибкой, а не тихим пересчётом.

Команды
-------

::

    python scripts/train_hf_a3.py                                  # hf, корпус A3
    python scripts/train_hf_a3.py --mode demo                      # лексические суррогаты
    python scripts/precompute_features.py --dataset data/corpus_a3/pairs.jsonl \\
        --mode hf --shards 4 --out reports/hf-cache                # один проход модели
    python scripts/train_hf_a3.py --features-cache reports/hf-cache --require-cache
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spanverify.dataset import read_pairs  # noqa: E402
from spanverify.engine import Verifier, WeightsBundle  # noqa: E402
from spanverify.features import (  # noqa: E402
    DEFAULT_WEIGHTS,
    HEAD_FEATURE_CANDIDATES,
    HF_MODEL_DEFAULT,
    feature_cache_key,
    load_feature_cache,
    set_feature_cache,
    validate_feature_cache,
)
from spanverify.train import head_feature_names, train  # noqa: E402

GATE_TOKEN_F1 = 0.90
GATE_TOKEN_FPR = 0.10


class Tee:
    """Печатать в консоль и в файл журнала одновременно (артефакт прогона)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = None

    def open(self) -> Tee:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8")
        return self

    def write(self, text: str) -> None:
        print(text, flush=True)
        if self.handle is not None:
            self.handle.write(text + "\n")
            self.handle.flush()

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def load_split(dataset: Path, splits_dir: Path | None, name: str) -> list[dict]:
    """Официальная часть разбиения, если она есть и не пуста."""
    directory = splits_dir if splits_dir is not None else dataset.parent / "splits"
    path = directory / f"{name}.jsonl"
    if not path.is_file() or not path.stat().st_size:
        return []
    return list(read_pairs(path))


def stratified_sample(pairs: list[dict], size: int, seed: int) -> list[dict]:
    """Стратифицированная подвыборка по типу пары — для отбора признаков.

    Отбор по AUC должен видеть те же типы ошибок, что и весь корпус, иначе
    признак, полезный только для ``missing``, мог бы быть отсеян просто потому,
    что таких пар в случайной выборке оказалось мало. ПорядокDeterministic:
    перемешивание с тем же seed даёт один и тот же набор при повторном запуске.
    """
    if size <= 0 or size >= len(pairs):
        ordered = list(pairs)
        random.Random(seed).shuffle(ordered)
        return ordered
    by_mode: dict[str, list[dict]] = {}
    for pair in pairs:
        data = pair.to_dict() if hasattr(pair, "to_dict") else pair
        mode_name = str((data.get("meta") or {}).get("mode") or "unknown")
        by_mode.setdefault(mode_name, []).append(data)
    rng = random.Random(seed)
    take: list[dict] = []
    share = size / len(pairs)
    for mode_name in sorted(by_mode):
        bucket = list(by_mode[mode_name])
        rng.shuffle(bucket)
        take.extend(bucket[: max(1, round(len(bucket) * share))])
    rng.shuffle(take)
    return take[:size]


def cache_check(cache: Any, records: list[dict], mode: str, model: str, require: bool) -> tuple[int, list[str]]:
    """Сколько пар корпуса покрыто кешем (и какие именно отсутствуют)."""
    if not cache:
        return 0, ["<кеш не задан>"] if require else []
    missing: list[str] = []
    for record in records:
        data = record.to_dict() if hasattr(record, "to_dict") else record
        key = feature_cache_key(
            str(data.get("answer", "")),
            data.get("context", ""),
            mode,
            model if mode == "hf" else "",
        )
        if key not in cache:
            missing.append(str(data.get("id", "?")))
    return len(records) - len(missing), missing


def metrics_row(metrics: dict[str, Any]) -> dict[str, float]:
    """Сжатая строка метрик для таблиц отчёта."""
    tokens = metrics["tokens"]
    verdicts = metrics["verdicts"]
    spans = metrics["spans"]
    auc = tokens.get("auc")
    return {
        "pairs": metrics.get("pairs"),
        "token_precision": round(tokens["precision"], 4),
        "token_recall": round(tokens["recall"], 4),
        "token_f1": round(tokens["f1"], 4),
        "token_fpr": round(tokens["fpr"], 4),
        "token_auc": round(auc, 4) if auc == auc else None,
        "span_f1_iou_0_5": round(spans["f1"], 4),
        "span_coverage": round(spans["recall_containment"], 4),
        "verdict_f1": round(verdicts["f1"], 4),
        "verdict_fpr": round(verdicts["fpr"], 4),
        "coverage": metrics.get("coverage") or {},
    }


def evaluate_with(verifier: Verifier, pairs: list[dict]) -> dict[str, Any]:
    """Сквозная оценка публичным путём (тот же ``verify()``, что и у API)."""
    return verifier.evaluate(pairs)


def gate(row: dict[str, Any]) -> tuple[bool, str]:
    """Критерий качества из технического задания: F1 ≥ 0.90 при FPR ≤ 0.10."""
    f1 = row["token_f1"] or 0.0
    fpr = row["token_fpr"] if row["token_fpr"] is not None else 1.0
    passed = f1 >= GATE_TOKEN_F1 and fpr <= GATE_TOKEN_FPR
    statement = (
        f"token F1 {f1:.4f} при пороге {GATE_TOKEN_F1:.2f}, "
        f"токенный FPR {fpr:.4f} при ограничении {GATE_TOKEN_FPR:.2f}: "
        + ("критерий выполнен" if passed else "критерий НЕ выполнен")
    )
    return passed, statement


def weights_file(out: str | Path) -> Path:
    """Имя файла весов из ``--out`` (по умолчанию ``weights_hf.json``)."""
    path = Path(out)
    return path if path.suffix else path / "weights_hf.json"


def fit_participation(verifier: Verifier, cache: Any, mode: str, model: str, seed: int) -> dict[str, Any]:
    """Обучить голову доли участия ИИ по кешу признаков (или честно пропустить).

    ``train._fit_participation`` в режиме hf с кешом пропускает обучение:
    синтетических текстов участия в кеше нет, а считать их моделью посреди
    обучения — второй проход. Если предпосчёт запускался с ``--participation``,
    строки есть, и голова обучается здесь, по тому же кешу, без нового прохода.
    """
    from spanverify._version import __version__
    from spanverify.features import feature_cache_key as key_of
    from spanverify.participation import (
        ParticipationModel,
        build_participation_corpus,
        corpus_rows_and_labels,
    )

    samples = build_participation_corpus(count=240, seed=seed + 1985)
    if cache is None or not len(cache):
        return _participation_skipped("кеш признаков не использовался")
    needed = [
        key_of(str(item["text"]), item.get("context", ""), mode, model if mode == "hf" else "") for item in samples
    ]
    absent = sum(1 for key in needed if key not in cache)
    if absent:
        return _participation_skipped(f"в кеше нет {absent} из {len(needed)} строк корпуса участия")
    rows, labels = corpus_rows_and_labels(verifier, samples)
    if not rows or len(set(labels)) < 2:
        return _participation_skipped("недостаточно строк или только один класс")
    fitted = ParticipationModel.fit(rows, labels, seed=seed, version=__version__)
    return {
        "type": "participation-logreg",
        "payload": fitted.to_dict(),
        "auc_out_of_fold": fitted.auc_out_of_fold,
        "calibrated_on": fitted.calibrated_on,
        "rows": len(rows),
    }


def _participation_skipped(reason: str) -> dict[str, Any]:
    return {
        "type": "none",
        "payload": None,
        "auc_out_of_fold": 0.0,
        "calibrated_on": f"пропущено: {reason}",
    }


def save_artifacts(report: Any, weights_path: Path, dataset: Path, mode: str, model: str) -> dict[str, str]:
    """Записать веса, голову и участие ИИ рядом с ``weights_path``.

    Имена соседних файлов выводятся из имени весов (``weights_hf.json`` →
    ``head_hf.json`` и ``participation_hf.json``), поэтому ``--out`` управляет
    всеми артефактами сразу. Формат совпадает с ``config/weights.json`` и
    ``config/head.json`` — те же поля и то же округление: меняется только имя
    файла, чтобы модель, обученная на реальных актах, не перезаписала
    демо-артефакты поставки.
    """
    out_dir = weights_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = weights_path.stem
    suffix = weights_path.suffix or ".json"
    weights_name = stem.replace("weights_", "") or "hf"
    bundle = report.bundle
    written: dict[str, str] = {}
    bundle.save(weights_path)
    written["weights"] = str(weights_path)
    payload = (report.head or {}).get("payload") or {}
    if payload:
        head_path = out_dir / f"head_{weights_name}{suffix}"
        head_payload = {**payload, "version": bundle.version, "seed": report.seed}
        head_path.write_text(json.dumps(head_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written["head"] = str(head_path)
    participation = report.participation or {}
    if participation.get("payload"):
        part_path = out_dir / f"participation_{weights_name}{suffix}"
        part_payload = {**participation["payload"], "version": bundle.version, "seed": report.seed}
        part_path.write_text(json.dumps(part_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written["participation"] = str(part_path)
    written["dataset"] = str(dataset)
    written["mode"] = mode
    written["model"] = model if mode == "hf" else ""
    return written


def render_report(args: argparse.Namespace, result: dict[str, Any]) -> str:
    """Markdown-отчёт оценки (таблица «база против расширенной головы»)."""
    selection = result["selection"]
    lines = [
        f"# Обучение режима `{result['mode']}` на корпусе {Path(args.dataset).name}",
        "",
        f"Дата запуска: {result['generated_at']}. Seed: {args.seed}. "
        f"Прогон: {args.run_id or 'локальный'}. Модель: {result['model'] or '— (лексические суррогаты)'}",
        "",
        f"Корпус: {result['pairs_total']} пар; разбиения: "
        + (
            ", ".join(f"{name} {count}" for name, count in result["splits"].items())
            if result["splits"]
            else "официальных нет, использовано групповое разбиение train()"
        )
        + ".",
        "",
        "## Отбор признаков головы (критерий — AUC на валидации, не порог)",
        "",
        f"Кандидаты: {', '.join(f'`{name}`' for name in selection['candidates'])}.",
        f"Подвыборка для отбора: {selection['select_pairs']} пар (стратификация по типу пары, seed {args.seed}).",
        f"Признак оставляем, если прирост AUC больше {args.min_auc_gain}.",
        "",
        "| Раунд | Кандидат | AUC головы (валидация) | Прирост | Решение |",
        "| --- | --- | --- | --- | --- |",
    ]
    for step in selection["steps"]:
        gain = step.get("auc_gain")
        lines.append(
            f"| {step['round']} | `{step['candidate']}` | {step['auc']:.4f} | "
            + (f"{gain:+.4f}" if isinstance(gain, float) else "—")
            + f" | {'взяли' if step['accepted'] else 'отклонили'} |"
        )
    lines += [
        "",
        "Отобранные дополнительные признаки: "
        + (", ".join(f"`{name}`" for name in selection["selected"]) if selection["selected"] else "нет")
        + ".",
        "",
        "## Состав признаков головы",
        "",
        "```",
        json.dumps(selection["final_features"], ensure_ascii=False),
        "```",
        "",
        "AUC головы (out-of-fold): "
        f"{result['final']['head_auc_out_of_fold']:.4f}; правило (без головы): "
        f"{result['final']['head_auc_rule']:.4f}; базовый состав (6 признаков): "
        + (f"{result['baseline']['head_auc_out_of_fold']:.4f}" if result.get("baseline") else "—")
        + ".",
        "",
        "## Метрики",
        "",
        "| Набор | Признаки | Token F1 | FPR | Precision | Recall | AUC | Span F1 (IoU 0.5) |"
        " Полнота накрытия | Вердикт F1 | Вердикт FPR |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in result["metrics"]:
        lines.append(
            f"| {row['split']} | {row['features']} | **{row['token_f1']:.4f}** | "
            f"{row['token_fpr']:.4f} | {row['token_precision']:.4f} | "
            f"{row['token_recall']:.4f} | {row['token_auc']} | "
            f"{row['span_f1_iou_0_5']:.4f} | {row['span_coverage']:.4f} | "
            f"{row['verdict_f1']:.4f} | {row['verdict_fpr']:.4f} |"
        )
    lines += [
        "",
        "### Coverage-метрики (детектор покрытия фактов, включён по умолчанию)",
        "",
        "| Набор | recall missing | precision missing | recall partial | precision partial |" " FPR на чистых |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in result["metrics"]:
        cov = row.get("coverage") or {}
        if not cov:
            continue
        lines.append(
            f"| {row['split']} | {cov.get('recall_missing')} | {cov.get('precision_missing')} | "
            f"{cov.get('recall_partial')} | {cov.get('precision_partial')} | {cov.get('fpr_clean')} |"
        )
    lines += [
        "",
        "## Проверка критерия",
        "",
        f"На официальном тесте: {result['gate_statement']}.",
        "",
        f"Пути артефактов: weights `{result['artifacts'].get('weights','—')}`, "
        f"head `{result['artifacts'].get('head','—')}`, "
        f"participation `{result['artifacts'].get('participation','—')}`.",
        "",
        "## Оговорка",
        "",
        result["disclaimer"],
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Обучение режима hf на корпусе A3")
    parser.add_argument("--dataset", default="data/corpus_a3/pairs.jsonl")
    parser.add_argument("--out", default="config/weights_hf.json", help="файл весов (или каталог)")
    parser.add_argument("--mode", choices=["hf", "demo"], default="hf")
    parser.add_argument("--model", default=HF_MODEL_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--target-fpr", type=float, default=0.1)
    parser.add_argument("--splits-dir", default=None, help="каталог официальных разбиений")
    parser.add_argument("--no-splits", action="store_true", help="игнорировать официальные разбиения")
    parser.add_argument(
        "--features-cache",
        action="append",
        default=[],
        metavar="PATH",
        help="каталог/файл кеша признаков (можно несколько раз)",
    )
    parser.add_argument("--require-cache", action="store_true", help="пасть, если кеш покрывает корпус не полностью")
    parser.add_argument(
        "--select-pairs",
        type=int,
        default=300,
        help="размер стратифицированной подвыборки для отбора признаков (0 — вся обучающая часть)",
    )
    parser.add_argument(
        "--min-auc-gain", type=float, default=0.0005, help="минимальный прирост AUC, чтобы взять признак"
    )
    parser.add_argument("--rounds", type=int, default=3, help="число раундов жадного отбора")
    parser.add_argument("--no-selection", action="store_true", help="взять все кандидаты сразу (без отбора)")
    parser.add_argument("--log", default="reports/hf_a3_train.log")
    parser.add_argument("--report-md", default="reports/hf_a3_eval.md")
    parser.add_argument("--report-json", default="reports/hf_a3_eval.json")
    parser.add_argument("--run-id", default="", help="номер CI-прогона для подписи чисел")
    parser.add_argument("--fail-on-gate", action="store_true", help="ненулевой код возврата, если критерий не выполнен")
    args = parser.parse_args(argv)

    log = Tee(Path(args.log)).open()
    started = time.time()
    try:
        return run(args, log, started)
    finally:
        log.close()


def run(args: argparse.Namespace, log: Tee, started: float) -> int:
    dataset = Path(args.dataset)
    if not dataset.is_file():
        log.write(f"корпус не найден: {dataset}")
        return 2
    records = list(read_pairs(dataset))
    splits: dict[str, list[dict]] = {}
    if not args.no_splits:
        splits_dir = Path(args.splits_dir) if args.splits_dir else None
        for name in ("train", "dev", "test"):
            part = load_split(dataset, splits_dir, name)
            if part:
                splits[name] = part

    cache = load_feature_cache(args.features_cache, counting=True) if args.features_cache else None
    set_feature_cache(cache)
    if cache is not None:
        info = validate_feature_cache(cache, args.mode, args.model if args.mode == "hf" else "")
        log.write(
            f"кеш признаков: {len(cache)} ключей, модели {info.get('models')}, " f"полей {len(info.get('fields', []))}"
        )
        covered, missing = cache_check(cache, records, args.mode, args.model, args.require_cache)
        log.write(f"кеш покрывает {covered} из {len(records)} пар корпуса")
        if args.require_cache and missing and missing[0] != "<кеш не задан>":
            log.write(f"кеш неполон: нет {len(missing)} пар (первые: {', '.join(missing[:5])})")
            return 2
        log.write("кеш считается готовым: признаки читаются из него, модель не запускается")

    def verifier(bundle: WeightsBundle | None = None) -> Verifier:
        return Verifier(
            mode=args.mode,
            model_name=args.model,
            features_cache=cache,
            weights=bundle or WeightsBundle(weights=dict(DEFAULT_WEIGHTS), threshold=0.5, mode=args.mode),
        )

    learn_pairs = splits.get("train") or records
    test_pairs = splits.get("test") or []
    candidates = list(HEAD_FEATURE_CANDIDATES)

    log.write("=" * 72)
    log.write(f"корпус: {dataset} ({len(records)} пар), режим {args.mode}, модель {args.model}")
    log.write(f"обучающая часть: {len(learn_pairs)} пар; официальный тест: {len(test_pairs)} пар")
    log.write(f"отбор признаков: {'выключен' if args.no_selection else 'жадный прямой поиск'}")
    log.write("=" * 72)

    selection: dict[str, Any] = {"candidates": candidates, "steps": [], "selected": [], "select_pairs": 0}
    base_auc: float | None = None
    if args.no_selection:
        selected = tuple(candidates)
    else:
        pool = list(candidates)
        selected = []
        subset = stratified_sample(learn_pairs, args.select_pairs, args.seed)
        selection["select_pairs"] = len(subset)
        log.write(f"отбор: подвыборка {len(subset)} пар, кандидаты {len(pool)}")
        for round_index in range(max(1, args.rounds)):
            if not pool:
                break
            baseline = train(
                subset,
                mode=args.mode,
                seed=args.seed,
                folds=args.folds,
                target_fpr=args.target_fpr,
                verifier=verifier(),
                dataset_name=f"{dataset.name} (отбор, раунд {round_index + 1})",
                head_features=tuple(selected),
            )
            reference = (baseline.head or {}).get("auc_test", 0.0)
            if base_auc is None:
                base_auc = reference
            log.write(f"  раунд {round_index + 1}: база AUC головы на валидации {reference:.4f}")
            best: tuple[float, str] | None = None
            for candidate in pool:
                trial = train(
                    subset,
                    mode=args.mode,
                    seed=args.seed,
                    folds=args.folds,
                    target_fpr=args.target_fpr,
                    verifier=verifier(),
                    dataset_name=f"{dataset.name} (отбор: +{candidate})",
                    head_features=(*selected, candidate),
                )
                auc = (trial.head or {}).get("auc_test", 0.0)
                gain = auc - reference
                accepted = gain > args.min_auc_gain
                selection["steps"].append(
                    {
                        "round": round_index + 1,
                        "candidate": candidate,
                        "auc": round(auc, 4),
                        "auc_gain": round(gain, 4),
                        "accepted": accepted,
                        "token_f1": round((trial.validation or {}).get("f1", 0.0), 4),
                        "token_fpr": round((trial.validation or {}).get("fpr", 0.0), 4),
                    }
                )
                log.write(
                    f"    {candidate:26s} AUC {auc:.4f} (Δ {gain:+.4f}) "
                    f"token F1 {(trial.validation or {}).get('f1', 0.0):.4f} → "
                    + ("взяли" if accepted else "отклонили")
                )
                if accepted and (best is None or auc > best[0]):
                    best = (auc, candidate)
            if best is None:
                log.write("  ни один кандидат не улучшил AUC — отбор остановлен")
                break
            selected.append(best[1])
            pool.remove(best[1])
            log.write(f"  раунд {round_index + 1}: взят {best[1]} (AUC {best[0]:.4f})")

    log.write("-" * 72)
    log.write("финальное обучение: " + (", ".join(selected) if selected else "без дополнительных признаков"))
    names = head_feature_names(selected)
    selection["selected"] = list(selected)
    selection["final_features"] = names

    final_report = train(
        learn_pairs,
        mode=args.mode,
        seed=args.seed,
        folds=args.folds,
        target_fpr=args.target_fpr,
        verifier=verifier(),
        dataset_name=str(dataset) + (" (официальная обучающая часть)" if splits else ""),
        head_features=tuple(selected),
    )
    bundle = final_report.bundle
    if args.mode == "hf":
        # Модель записывается в бандл: оценка по --weights config/weights_hf.json
        # обязана использовать ту же модель, что и обучение (имя входит в ключ
        # кеша признаков и в признаки как таковые).
        bundle.meta["model"] = args.model
    # Для головы участия нужен верификатор с готовым кешем; бандл не важен:
    # участие считается по признакам токенов, а не по порогу и маске.
    served_for_participation = Verifier(mode=args.mode, model_name=args.model, features_cache=cache)
    # Доля участия ИИ: train() в режиме hf с кешом её не обучает (в кеше нет
    # строк корпуса участия). Если предпосчёт делался с --participation, голова
    # доучивается здесь по тому же кешу — без нового прохода по модели.
    participation = fit_participation(served_for_participation, cache, args.mode, args.model, args.seed)
    if participation.get("payload") and not (final_report.participation or {}).get("payload"):
        final_report.participation = participation
        bundle.meta["participation"] = {
            "type": "participation-logreg",
            "auc_out_of_fold": participation.get("auc_out_of_fold", 0.0),
            "calibrated_on": participation.get("calibrated_on", ""),
        }
    served = Verifier(mode=args.mode, model_name=args.model, weights=bundle, features_cache=cache)

    metrics: list[dict[str, Any]] = []
    baseline_row: dict[str, Any] | None = None
    if selected:
        # Сравнение «база против расширенной головы» на том же корпусе: иначе
        # прирост признака неотделим от удачного разбиения.
        base_bundle_report = train(
            learn_pairs,
            mode=args.mode,
            seed=args.seed,
            folds=args.folds,
            target_fpr=args.target_fpr,
            verifier=verifier(),
            dataset_name=str(dataset) + " (база: 6 признаков)",
        )
        base_served = Verifier(
            mode=args.mode, model_name=args.model, weights=base_bundle_report.bundle, features_cache=cache
        )
        base_eval_set = test_pairs or learn_pairs
        baseline_row = metrics_row(evaluate_with(base_served, base_eval_set))
        baseline_row.update(
            split=("официальный тест (база)" if test_pairs else "обучающая часть (база)"),
            features="база (6)",
        )
        metrics.append(dict(baseline_row))
        selection["baseline_auc"] = (base_bundle_report.head or {}).get("auc_test", 0.0)

    eval_sets = [("официальный тест", test_pairs)] if test_pairs else []
    eval_sets.append(("весь корпус", records))
    for name, part in eval_sets:
        row = metrics_row(evaluate_with(served, part))
        row.update(split=name, features=f"расширенная ({len(names)})")
        metrics.append(row)
    # Критерий проверяется по официальному тесту; если разбиений нет — по тому
    # набору, на котором обучались (и в отчёте это подписано отдельной строкой).
    scored = [row for row in metrics if row["features"].startswith("расширенная")]
    gate_row = scored[0] if test_pairs else scored[-1]
    gate_passed, gate_statement = gate(gate_row)

    artifacts = save_artifacts(final_report, weights_file(args.out), dataset, args.mode, args.model)

    result: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + " UTC",
        "mode": args.mode,
        "model": args.model if args.mode == "hf" else "",
        "dataset": str(dataset),
        "pairs_total": len(records),
        "splits": {name: len(part) for name, part in splits.items()},
        "seed": args.seed,
        "folds": args.folds,
        "target_fpr": args.target_fpr,
        "selection": selection,
        "final": {
            "head_auc_out_of_fold": (final_report.head or {}).get("auc_out_of_fold", 0.0),
            "head_auc_test": (final_report.head or {}).get("auc_test", 0.0),
            "head_auc_rule": (final_report.head or {}).get("auc_rule", 0.0),
            "weights": {name: round(value, 4) for name, value in bundle.weights.items()},
            "threshold": round(bundle.threshold, 6),
            "span": {"z": bundle.span_z, "floor": bundle.span_floor, "cap": bundle.span_cap},
        },
        "baseline": {
            "head_auc_out_of_fold": (
                (selection.get("baseline_auc") or base_auc or 0.0)
                if selected
                else (final_report.head or {}).get("auc_out_of_fold", 0.0)
            ),
            "metrics": baseline_row,
        },
        "metrics": metrics,
        "gate_token_f1": GATE_TOKEN_F1,
        "gate_token_fpr": GATE_TOKEN_FPR,
        "gate_passed": gate_passed,
        "gate_statement": gate_statement,
        "artifacts": artifacts,
        "cache": {
            "paths": list(args.features_cache),
            "keys": len(cache) if cache else 0,
            "hits": getattr(cache, "hits", None),
            "misses": getattr(cache, "misses", None),
        },
        "duration_s": round(time.time() - started, 1),
        "disclaimer": (
            "Режим hf: признаки посчитаны весами языковой модели "
            f"{args.model}; корпус A3 собран из фрагментов опубликованных актов "
            "(OCR), разметка — управляемые подмены, а не независимая экспертная. "
            "Числа описывают перенос метода на реальные тексты и не являются "
            "обещанием качества на других документах."
            if args.mode == "hf"
            else "Режим demo: лексические суррогаты признаков без весов модели. "
            "Числа измеряют полезность самих признаков на реальных текстах и "
            "работоспособность конвейера; научным результатом не являются."
        ),
    }

    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_json).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = render_report(args, result)
    Path(args.report_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_md).write_text(markdown + "\n", encoding="utf-8")

    log.write("")
    for row in metrics:
        log.write(
            f"{row['split']:28s} {row['features']:20s} token F1 {row['token_f1']:.4f} "
            f"FPR {row['token_fpr']:.4f} AUC {row['token_auc']} "
            f"вердикт F1 {row['verdict_f1']:.4f}"
        )
    log.write("")
    log.write(f"критерий: {gate_statement}")
    log.write(f"отчёт: {args.report_md}, артефакты: {', '.join(f'{k}={v}' for k, v in artifacts.items())}")
    log.write(f"длительность: {result['duration_s']} с")
    if args.fail_on_gate and not gate_passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
