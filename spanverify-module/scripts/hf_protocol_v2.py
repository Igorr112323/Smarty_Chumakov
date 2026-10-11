#!/usr/bin/env python3
"""Харнесс цикла v2: новая независимая выборка, цель F1 ≥ 0,70 при FPR ≤ 0,40.

Почему отдельный файл, а не флаги в `hf_protocol.py`. Тот харнесс замораживал
измерение цикла v1: его правила записаны в `docs/METRIC_SPEC.md` (sha256
спецификации входит в манифесты v1), и любое изменение внутри него сделало бы
зафиксированные числа v1 невоспроизводимыми. В v2 отличаются ровно четыре вещи —
выборка, правило порога (val вместо train-CV, по требованию заказчика), обязанность
seed'а влиять на обучение и стоп-правило шага 2. Всё остальное (определения метрик,
разбор сетки признаков, модель, классификатор, агрегация seed'ов, формат
предсказаний, пересчёт манифеста) импортируется из v1, чтобы числа двух циклов
сравнивались честно.

Команды: freeze → curve → sweep → seedcheck → final → verify-manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import hf_protocol as v1  # noqa: E402  (переиспользование замороженного харнесса v1)

PROTOCOL = "hf-criterion-v2"
GOAL_INTERNAL_F1 = 0.70
GOAL_CONTRACT_F1 = float(v1.CRITERION_F1)
FPR_MAX = float(v1.CRITERION_FPR)
MAX_EXPERIMENTS = int(v1.MAX_EXPERIMENTS)

# Нижние границы объёма (docs/METRIC_SPEC_v2.md, раздел 7). Не «цель», а именно пол:
# ниже него тест не различает 0,70 и 0,60, и замер теряет смысл.
FLOORS: dict[str, tuple[int, int]] = {"train": (2000, 700), "val": (600, 200), "test": (600, 200)}

# Что считается «уже виденным» в v1: каталоги-источники doc_id. Обход.generic'ный —
# по каталогу собираются имена файлов источников, ключи манифестов и document_of из
# всех *.jsonl, чтобы не пропустить черновик A2 и демо.
V1_UNIVERSES: tuple[str, ...] = (
    "data/corpus_a",
    "data/corpus_a2",
    "data/corpus_a3",
    "data/corpus_a3_local70",
    "data/demo_pairs.jsonl",
    "data/demo_dataset.jsonl",
)

# Кандидаты порога: фиксированная сетка + квантили распределения оценок. Оптимум F1
# всегда на оценке какого-то токена, а полный перебор по 50 000 оценок дал бы кривую,
# которую нельзя ни напечатать, ни прочитать.
CURVE_GRID_STEP = 0.01
CURVE_QUANTILES = 200
CURVE_MAX_POINTS = 400


# ------------------------------------------------------------------ независимость


def collect_document_ids(root: Path, relative: str) -> set[str]:
    """Все doc_id из одного каталога/файла корпуса v1 (только чтение, ничего не измеряем)."""
    found: set[str] = set()
    base = root / relative
    if base.is_file():
        if base.suffix == ".jsonl":
            for record in v1.read_jsonl(base):
                found.add(v1.document_of(record))
        return found
    if not base.is_dir():
        return found
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        # Имена файлов источников — это те же doc_id, что потом попадают в meta.group
        # пар (проверено на A3: `0001202212080001`), поэтому пересечение считается
        # в одном пространстве имён, а не «примерно по названию».
        if path.parent.name in {"sources", "docs", "akty_texts"} and path.suffix == ".txt":
            found.add(path.stem)
        elif path.name == "manifest.json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            # pair_sha256 намеренно не берётся: это идентификаторы пар, а не документов.
            for key in ("sources_sha256",):
                block = payload.get(key)
                if isinstance(block, dict):
                    found |= {str(item).removesuffix(".txt") for item in block}
            documents = payload.get("documents")
            if isinstance(documents, dict) and isinstance(documents.get("sha256"), dict):
                found |= {str(item).removesuffix(".txt") for item in documents["sha256"]}
        elif path.suffix == ".jsonl":
            try:
                for record in v1.read_jsonl(path):
                    # Любая запись-пара: у черновика A2 свои ключи, и пропускать их
                    # «потому что формат не узнали» — значит ослабить проверку.
                    if isinstance(record, dict):
                        found.add(v1.document_of(record))
            except (ValueError, OSError):
                continue
    return found


def independence_report(root: Path, split_dir: Path) -> dict[str, Any]:
    """Пересечение doc_id новой выборки со всем, что видели циклы v1.

    Это не косметика: «независимая тестовая выборка», собранная из уже измеренных
    документов, была бы подглядыванием. Проверка выполняется и на заморозке, и
    тестом `tests/test_v2_no_leakage.py`.
    """
    v2_ids: set[str] = set()
    per_part: dict[str, int] = {}
    for part, filename in (("train", "train.jsonl"), ("val", None), ("test", "test.jsonl")):
        path = v1.val_path_of(split_dir) if filename is None else split_dir / filename
        if not path.is_file():
            continue
        ids = {v1.document_of(record) for record in v1.load_split(path)}
        v2_ids |= ids
        per_part[part] = len(ids)
    universes: dict[str, Any] = {}
    overlapping: set[str] = set()
    for relative in V1_UNIVERSES:
        ids = collect_document_ids(root, relative)
        overlap = ids & v2_ids
        overlapping |= overlap
        universes[relative] = {
            "documents": len(ids),
            "overlap": len(overlap),
            "examples": sorted(overlap)[:5],
        }
    overlap_total = len(overlapping)
    return {
        "v2_documents": len(v2_ids),
        "v2_documents_per_part": per_part,
        "v1_universes": universes,
        "overlap_total": overlap_total,
        "rule": "ни один doc_id выборки v2 не встречается ни в одном корпусе v1 (A1, A2, A3, local70, демо)",
    }


def parts_of(split_dir: Path) -> dict[str, Path]:
    return {
        "train": split_dir / "train.jsonl",
        "val": v1.val_path_of(split_dir),
        "test": split_dir / "test.jsonl",
    }


def audit_sizes(parts: dict[str, Path]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Объём частей против замороженного пола + пересечения документов между частями."""
    info: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    documents: dict[str, set[str]] = {}
    for part, path in parts.items():
        if not path.is_file():
            problems.append(f"нет файла разбиения {path}")
            continue
        records = v1.load_split(path)
        names = sorted({v1.document_of(record) for record in records})
        documents[part] = set(names)
        min_pairs, min_docs = FLOORS.get(part, (0, 0))
        if len(records) < min_pairs:
            problems.append(f"{part}: пар {len(records)} — ниже замороженного пола {min_pairs}")
        if len(names) < min_docs:
            problems.append(f"{part}: документов {len(names)} — ниже замороженного пола {min_docs}")
        info[part] = {
            "file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "sha256": v1.sha256_file(path),
            "pairs": len(records),
            "documents": len(names),
            "document_list_sha256": hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest(),
            "gold_spans": sum(len(v1.spans_of(record)) for record in records),
            "gold_tokens_approx": sum(
                1
                for record in records
                for token in v1.token_rows(record, None)
                if any(token["start"] < end and token["end"] > start for start, end in v1.spans_of(record))
            ),
            "mode_counts": v1._mode_counts(records),
        }
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = documents.get(first, set()) & documents.get(second, set())
        if overlap:
            problems.append(f"документы пересекаются между {first} и {second}: {sorted(overlap)[:5]}")
    return info, problems


# ---------------------------------------------------------------------- модель


def entries_of(config: dict[str, Any]) -> list[dict[str, Any]]:
    experiments = list(config.get("experiments") or [])
    if not experiments:
        raise SystemExit("в файле эксперимента нет ни одной гипотезы (ключ «experiments»)")
    if len(experiments) > MAX_EXPERIMENTS:
        raise SystemExit(f"гипотез {len(experiments)} — больше разрешённых {MAX_EXPERIMENTS} (шаг 2 цикла)")
    return [v1._experiment_entry(number, experiment) for number, experiment in enumerate(experiments, start=1)]


def load_cache(directory: str | Path) -> dict[str, dict[str, Any]]:
    return v1.load_grid_cache(v1._cache_paths(Path(directory)))


class CacheSet:
    """Кеши сеток по каталогам; у гипотезы «другая модель» свой кеш и свой model id.

    `cache_model_provenance` в v1 падает, если в одном кеше оказались две модели,
    поэтому каталоги не смешиваются: каждый каталог — одна модель, одна revision.
    """

    def __init__(self, default_directory: str | Path) -> None:
        self.default = str(default_directory)
        self._loaded: dict[str, dict[str, Any]] = {}

    def for_entry(self, entry: dict[str, Any]) -> tuple[dict[str, Any], str]:
        directory = str(entry["params"].get("grid_cache") or self.default)
        if directory not in self._loaded:
            self._loaded[directory] = load_cache(directory)
        return self._loaded[directory], directory


def bootstrap_records(records: Sequence[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Бутстрап по документам обучающей части — способ заставить seed влиять на модель.

    В v1 пять seed'ов дали std ровно 0,0: `train_logreg` — полный пакет без
    перемешивания, и seed менял только инициализацию (то есть ничего). Линейная
    модель на тех же данных детерминирована, поэтому единственный честный способ
    получить осмысленный разброс, не меняя классификатор продукта, — менять состав
    обучающей выборки. Отбор конфигурации и порог при этом остаются общими для всех
    seed'ов: иначе seed участвовал бы в отборе.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(v1.document_of(record), []).append(dict(record))
    keys = sorted(groups)
    if not keys:
        return []
    rng = random.Random(seed)
    picked = [keys[rng.randrange(len(keys))] for _ in keys]
    out: list[dict[str, Any]] = []
    for key in picked:
        out.extend(groups[key])
    return out


def fit_and_score(
    cache: dict[str, dict[str, Any]],
    entry: dict[str, Any],
    fit_records: Sequence[dict[str, Any]],
    eval_records: Sequence[dict[str, Any]],
    seed: int,
    *,
    bootstrap: bool = False,
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
    """Обучение на fit-части и оценки на eval-части (сглаживание — как в v1)."""
    names = entry["features"]
    config = dict(entry["params"])
    config["classifier"] = entry["classifier"]
    rows_in = bootstrap_records(fit_records, seed) if bootstrap else list(fit_records)
    fit_rows = v1.prepare(rows_in, cache, names)
    eval_rows = v1.prepare(eval_records, cache, names)
    if not fit_rows or not eval_rows:
        raise SystemExit("пустые строки: кеш не покрывает выборку (нужен precompute_features.py --grid)")
    model = v1.fit_model(v1.feature_matrix(fit_rows, names), [int(row["gold"]) for row in fit_rows], config, seed)
    for row, value in zip(eval_rows, model.scores(v1.feature_matrix(eval_rows, names)), strict=True):
        row["score"] = float(value)
    if int(entry["window"]):
        v1.smooth(v1.group_by_pair(eval_rows), int(entry["window"]))
    return model, fit_rows, eval_rows


def threshold_candidates(scores: Sequence[float]) -> list[float]:
    grid = [round(value * CURVE_GRID_STEP, 4) for value in range(1, 100)]
    values = sorted(float(score) for score in scores)
    if values:
        step = max(1, len(values) // CURVE_QUANTILES)
        grid += [values[index] for index in range(0, len(values), step)]
    out = sorted({value for value in grid if 0.0 < value < 1.0})
    if len(out) > CURVE_MAX_POINTS:
        keep = {out[index] for index in range(0, len(out), max(1, len(out) // CURVE_MAX_POINTS))}
        keep.add(out[-1])
        out = sorted(keep | set(out[:3]))
    return out


def curve_on(rows: Sequence[dict[str, Any]], merge_gap: int, candidates: Iterable[float]) -> list[dict[str, Any]]:
    curve: list[dict[str, Any]] = []
    for threshold in candidates:
        numbers = v1.score_metrics(rows, float(threshold), merge_gap)["tokens"]
        curve.append(
            {
                "threshold": round(float(threshold), 6),
                "f1": numbers.get("f1"),
                "precision": numbers.get("precision"),
                "recall": numbers.get("recall"),
                "fpr": numbers.get("fpr"),
                "tp": numbers.get("tp"),
                "fp": numbers.get("fp"),
                "fn": numbers.get("fn"),
                "tn": numbers.get("tn"),
            }
        )
    return curve


def apply_rule(entry: dict[str, Any], rows: Sequence[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """Метрики всех уровней при заданном пороге (кривая не нужна — только точка)."""
    block = v1.score_metrics(rows, float(threshold), int(entry.get("merge_gap", 2)))
    tokens = block["tokens"]
    return {
        "threshold": round(float(threshold), 6),
        "f1": tokens.get("f1"),
        "precision": tokens.get("precision"),
        "recall": tokens.get("recall"),
        "fpr": tokens.get("fpr"),
        "auc": tokens.get("auc"),
        "answer_f1": (block.get("answers") or {}).get("f1"),
        "answer_fpr": (block.get("answers") or {}).get("fpr"),
        "span_f1": (block.get("spans") or {}).get("f1"),
        "rows": block["rows"],
        "positive_rows": block["positive_rows"],
    }


def goal_status(tokens: dict[str, Any] | None, goal_f1: float) -> dict[str, Any]:
    """Статус цели. Числа не «доокругляются» до PASS: сравнение по тому же правилу, что и тест."""
    numbers = tokens or {}
    f1 = float(numbers.get("f1", 0.0) or 0.0)
    fpr = float(numbers.get("fpr", 1.0) if numbers.get("fpr") is not None else 1.0)
    passed = f1 >= goal_f1 and fpr <= FPR_MAX
    return {
        "target": f"F1 ≥ {goal_f1} при FPR ≤ {FPR_MAX} (токены, hf, независимый test цикла v2)",
        "status": "PASS" if passed else "NOT MET",
        "token_f1": round(f1, 6),
        "token_fpr": round(fpr, 6),
        "f1_gap": round(goal_f1 - f1, 6),
        "fpr_gap": round(fpr - FPR_MAX, 6),
    }


# --------------------------------------------------------------------- команды


def command_freeze(args: argparse.Namespace) -> int:
    """Шаг 0: аудит разбиения v2, независимость от v1, фиксация правил и порогов."""
    started = time.time()
    split_dir = Path(args.splits)
    info, problems = audit_sizes(parts_of(split_dir))
    independence = independence_report(ROOT, split_dir)
    if independence["overlap_total"]:
        problems.append(
            f"независимость нарушена: {independence['overlap_total']} doc_id уже есть в корпусах v1 "
            f"(примеры: {json.dumps(independence['v1_universes'], ensure_ascii=False)[:300]})"
        )
    config_path = Path(args.config) if args.config else None
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path and config_path.is_file() else {}
    payload = {
        "protocol": PROTOCOL,
        "step": "freeze",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "code_commit": v1._git_commit(),
        "metric_spec_v2": "docs/METRIC_SPEC_v2.md",
        "metric_spec_v2_sha256": v1.sha256_file(ROOT / "docs" / "METRIC_SPEC_v2.md"),
        "metric_spec_inherited": "docs/METRIC_SPEC.md",
        "metric_spec_inherited_sha256": v1.sha256_file(ROOT / "docs" / "METRIC_SPEC.md"),
        "goals": {
            "internal": {"token_f1_min": GOAL_INTERNAL_F1, "token_fpr_max": FPR_MAX},
            "contract_min": {"token_f1_min": GOAL_CONTRACT_F1, "token_fpr_max": FPR_MAX},
            "contract_source": (
                "формулировка ТЗ (Прил. №3 к договору 0117812, п. 4.1) передана заказчиком текстом "
                "2026-10-11; PDF договора в рабочей папке отсутствует, поэтому минимум зафиксирован как дано"
            ),
            "primary_level": "tokens",
        },
        "selection_rule": (
            "из не более чем 10 гипотез выбирается конфигурация с максимальным F1(токены) на val "
            "при FPR(токены) на val ≤ 0,40; test в отборе не участвует"
        ),
        "threshold_rules": {
            "E1": (
                "правило цикла v2 (по требованию заказчика): порог = argmax F1_val из порогов, "
                "для которых FPR_val ≤ 0,40; при равенстве F1 — меньший FPR"
            ),
            "E0": (
                "правило цикла v1 (справочно, второй строкой): порог = argmax F1 по out-of-fold оценкам "
                "train (group CV по документам), затем применяется к val без переселекции"
            ),
            "applied_at_step_3": "E1",
            "why_both": (
                "в E1 порог и оценка отобраны на одном сплите, поэтому F1_val оптимистична; "
                "E0 даёт несмещённую по порогу companion-оценку на том же val"
            ),
        },
        "stop_rule": {
            "condition": f"после {MAX_EXPERIMENTS} гипотез F1_val < {GOAL_INTERNAL_F1} при FPR_val ≤ {FPR_MAX}",
            "consequence": "цикл закрывается со статусом «не достигнуто на val», тестовая выборка не открывается",
            "decided_by": "заказчик, 2026-10-11 (вариант «понижать порог входа» отвергнут)",
        },
        "seed_rule": {
            "seeds": list(args.seeds),
            "requirement": "std(F1) по 5 seed'ам обязан быть > 0; способ — бутстрап по документам train",
            "gate": "scripts/hf_protocol_v2.py seedcheck, тест tests/test_v2_seed_variance.py",
        },
        "floors": {part: {"pairs_min": pairs, "documents_min": docs} for part, (pairs, docs) in FLOORS.items()},
        "splits_dir": str(split_dir),
        "splits_sha256": {part: block["sha256"] for part, block in info.items()},
        "parts": info,
        "independence": independence,
        "experiments_limit": MAX_EXPERIMENTS,
        "config": config,
        "config_sha256": v1.sha256_file(config_path) if config_path and config_path.is_file() else "",
        "feature_names": list(v1.GRID_FEATURE_NAMES),
        "grid_cache_format": v1.GRID_CACHE_FORMAT,
        "hardware": v1.hardware_manifest(),
        "problems": problems,
        "duration_s": round(time.time() - started, 2),
    }
    out = Path(args.out)
    v1.write_json(out / "freeze.json", payload)
    print(f"заморозка v2 записана: {out / 'freeze.json'} (sha256 спеки {payload['metric_spec_v2_sha256'][:12]})")
    for part, block in info.items():
        print(
            f"  {part}: пар {block['pairs']}, документов {block['documents']}, "
            f"gold-фрагментов {block['gold_spans']}, sha256 {block['sha256'][:12]}"
        )
    print(
        f"  независимость: пересечение с v1 = {independence['overlap_total']} из {independence['v2_documents']} doc_id"
    )
    if problems:
        print("ПРОБЛЕМЫ:")
        for item in problems:
            print(f"  - {item}")
        return 1
    return 0


def command_curve(args: argparse.Namespace) -> int:
    """Шаг 1: кривые F1/FPR по порогам на val + обе правила порога. Test не читается."""
    if args.split != "val":
        raise SystemExit("кривая строится только на val: доступ к test открывается ровно один раз — в команде final")
    started = time.time()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    entries = entries_of(config)
    entry = next((item for item in entries if item["name"] == args.experiment), entries[0])
    caches = CacheSet(args.grid_cache)
    cache, cache_dir = caches.for_entry(entry)
    split_dir = Path(args.splits)
    parts = parts_of(split_dir)
    fit = v1.load_split(parts["train"])
    rows_eval = v1.load_split(parts[args.split])
    bootstrap = bool(entry["params"].get("bootstrap_docs"))
    model, fit_rows, eval_rows = fit_and_score(cache, entry, fit, rows_eval, int(config.get("seed", 42)), bootstrap)
    scores = [float(row["score"]) for row in eval_rows]
    labels = [int(row["gold"]) for row in eval_rows]
    cut = v1.choose_threshold(scores, labels, FPR_MAX)
    e1 = apply_rule(entry, eval_rows, float(cut["threshold"]))
    cv_model, e0_threshold, e0_details = v1.fit_with_cv(
        v1.feature_matrix(fit_rows, entry["features"]),
        [int(row["gold"]) for row in fit_rows],
        [str(row["doc"]) for row in fit_rows],
        [str(row["id"]) for row in fit_rows],
        {**entry["params"], "classifier": entry["classifier"]},
        int(config.get("seed", 42)),
        int(entry["window"]),
    )
    e0 = apply_rule(entry, eval_rows, float(e0_threshold))
    payload = {
        "protocol": PROTOCOL,
        "step": "curve",
        "split": args.split,
        "experiment": entry["name"],
        "features": entry["features"],
        "classifier": entry["classifier"],
        "params": entry["params"],
        "window": entry["window"],
        "merge_gap": entry["merge_gap"],
        "grid_cache_dir": cache_dir,
        "grid_model": v1.cache_model_provenance(cache),
        "grid_cache_files": [
            {"file": str(item), "sha256": v1.sha256_file(item)} for item in v1._cache_paths(Path(args.grid_cache))
        ],
        "rows": len(eval_rows),
        "positive_rows": sum(labels),
        "rule_e1_val_chosen": e1,
        "rule_e0_train_cv": e0,
        "e0_cv_details": e0_details,
        "model_sha256_e1": hashlib.sha256(json.dumps(model.to_dict(), sort_keys=True).encode("utf-8")).hexdigest(),
        "model_sha256_e0": hashlib.sha256(json.dumps(cv_model.to_dict(), sort_keys=True).encode("utf-8")).hexdigest(),
        "curve": curve_on(eval_rows, int(entry["merge_gap"]), threshold_candidates(scores)),
        "code_commit": v1._git_commit(),
        "duration_s": round(time.time() - started, 2),
    }
    out = Path(args.out)
    v1.write_json(out / "val_curve.json", payload)
    print(f"эксперимент: {entry['name']} (val, строк токенов {len(eval_rows)}, gold {sum(labels)})")
    print(f"  E1 (порог на val) : порог {e1['threshold']} F1 {e1['f1']} FPR {e1['fpr']} AUC {e1['auc']}")
    print(f"  E0 (порог из CV)  : порог {e0['threshold']} F1 {e0['f1']} FPR {e0['fpr']} AUC {e0['auc']}")
    print(f"  кривая: {len(payload['curve'])} точек → {out / 'val_curve.json'}")
    return 0


def command_sweep(args: argparse.Namespace) -> int:
    """Шаг 2: до 10 гипотез, отбор по val, стоп-правило. Чисел на test здесь не бывает."""
    started = time.time()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    entries = entries_of(config)
    caches = CacheSet(args.grid_cache)
    parts = parts_of(Path(args.splits))
    train = v1.load_split(parts["train"])
    val = v1.load_split(parts["val"])
    seed = int(config.get("seed", 42))
    e0_names = set(config.get("e0_experiments") or [])
    results: list[dict[str, Any]] = []

    def train_cv_threshold(entry: dict[str, Any], fit_rows: Sequence[dict[str, Any]]) -> float:
        """Порог по out-of-fold оценкам train — правило E0 (справочная строка).

        Group CV дорогa (пять обучений на 3000 пар), поэтому по умолчанию она
        считается для гипотез, перечисленных в `e0_experiments`, и обязательно —
        для отобранной конфигурации (см. ниже): читатель обязан видеть, насколько
        оптимистична оценка с порогом, выбранным на том же сплите.
        """
        _, threshold, _details = v1.fit_with_cv(
            v1.feature_matrix(fit_rows, entry["features"]),
            [int(row["gold"]) for row in fit_rows],
            [str(row["doc"]) for row in fit_rows],
            [str(row["id"]) for row in fit_rows],
            {**entry["params"], "classifier": entry["classifier"]},
            seed,
            int(entry["window"]),
        )
        return float(threshold)

    for number, entry in enumerate(entries, start=1):
        bootstrap = bool(entry["params"].get("bootstrap_docs"))
        cache, cache_dir = caches.for_entry(entry)
        _model, fit_rows, eval_rows = fit_and_score(cache, entry, train, val, seed, bootstrap=bootstrap)
        scores = [float(row["score"]) for row in eval_rows]
        labels = [int(row["gold"]) for row in eval_rows]
        cut = v1.choose_threshold(scores, labels, FPR_MAX)
        e1 = apply_rule(entry, eval_rows, float(cut["threshold"]))
        block: dict[str, Any] = {"e1_val_chosen": e1, "e0_train_cv": None}
        if not e0_names or entry["name"] in e0_names:
            block["e0_train_cv"] = apply_rule(entry, eval_rows, train_cv_threshold(entry, fit_rows))
        results.append(
            {
                "number": number,
                "name": entry["name"],
                "hypothesis": entry["hypothesis"],
                "change": entry["change"],
                "features_count": len(entry["features"]),
                "classifier": entry["classifier"],
                "window": entry["window"],
                "merge_gap": entry["merge_gap"],
                "params": entry["params"],
                "grid_cache_dir": cache_dir,
                "grid_model": v1.cache_model_provenance(cache)["id"],
                **block,
            }
        )
        print(
            f"#{number} {entry['name']}: F1_val {e1['f1']} при FPR_val {e1['fpr']} "
            f"(порог {e1['threshold']}, AUC {e1['auc']})"
        )
    allowed = [item for item in results if (item["e1_val_chosen"]["fpr"] or 1.0) <= FPR_MAX]
    selected = None
    if allowed:
        selected = max(allowed, key=lambda item: (item["e1_val_chosen"]["f1"], -item["e1_val_chosen"]["fpr"]))
    if selected is not None and selected["e0_train_cv"] is None:
        entry = entries[selected["number"] - 1]
        _model, fit_rows, eval_rows = fit_and_score(cache, entry, train, val, seed, bootstrap=False)
        selected["e0_train_cv"] = apply_rule(entry, eval_rows, train_cv_threshold(entry, fit_rows))
        selected["e0_added_after_selection"] = (
            "порог E0 для отобранной конфигурации досчитан после отбора; на отбор он не влияет "
            "(конфигурация выбрана по E1), а нужен как несмещённая по порогу companion-оценка"
        )
    best_f1 = float((selected or {}).get("e1_val_chosen", {}).get("f1") or 0.0)
    stop = {
        "condition": f"F1_val < {GOAL_INTERNAL_F1} после {MAX_EXPERIMENTS} гипотез",
        "triggered": best_f1 < GOAL_INTERNAL_F1,
    }
    payload = {
        "protocol": PROTOCOL,
        "step": "sweep",
        "config_file": args.config,
        "config_sha256": v1.sha256_file(Path(args.config)),
        "splits_sha256": {part: v1.sha256_file(path) for part, path in parts.items() if path.is_file()},
        "seed": seed,
        "selection_rule": "максимум F1(токены) на val при FPR(токены) на val ≤ 0,40; порог по правилу E1",
        "experiments_limit": MAX_EXPERIMENTS,
        "experiments_run": len(results),
        "results": results,
        "selected": selected["name"] if selected else None,
        "selected_number": selected["number"] if selected else None,
        "stop_rule": {**stop, "consequence": "тест не открывается, цикл закрывается «не достигнуто на val»"},
        "test_accessed": False,
        "code_commit": v1._git_commit(),
        "duration_s": round(time.time() - started, 2),
    }
    out = Path(args.out)
    v1.write_json(out / "sweep.json", payload)
    if selected:
        print(f"отобрано: #{selected['number']} {selected['name']} (F1_val {best_f1})")
    else:
        print("ни одна гипотеза не уложилась в FPR_val ≤ 0,40 — конфигурация не отобрана")
    if stop["triggered"]:
        print(f"СТОП-ПРАВИЛО: F1_val {best_f1} < {GOAL_INTERNAL_F1} → тест не открывается (см. docs/EXPERIMENTS_v2.md)")
        return 2
    return 0


def command_seedcheck(args: argparse.Namespace) -> int:
    """Проверка «seed обязан влиять на обучение» — до шага 3, на val."""
    started = time.time()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    entries = entries_of(config)
    name = args.experiment or str(config.get("selected") or entries[0]["name"])
    entry = next((item for item in entries if item["name"] == name), None)
    if entry is None:
        raise SystemExit(f"гипотезы «{name}» нет в файле конфигурации")
    caches = CacheSet(args.grid_cache)
    cache, cache_dir = caches.for_entry(entry)
    parts = parts_of(Path(args.splits))
    train = v1.load_split(parts["train"])
    val = v1.load_split(parts["val"])
    seeds = [int(value) for value in str(args.seeds).split(",") if value.strip()]
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        _, _, eval_rows = fit_and_score(
            cache, entry, train, val, seed, bootstrap=bool(entry["params"].get("bootstrap_docs"))
        )
        scores = [float(row["score"]) for row in eval_rows]
        threshold = float(v1.choose_threshold(scores, [int(row["gold"]) for row in eval_rows], FPR_MAX)["threshold"])
        numbers = v1.score_metrics(eval_rows, threshold, int(entry["merge_gap"]))["tokens"]
        per_seed.append({"seed": seed, "threshold": round(threshold, 6), "tokens": numbers})
    f1 = v1._mean_std([float(item["tokens"].get("f1") or 0.0) for item in per_seed])
    fpr = v1._mean_std([float(item["tokens"].get("fpr") or 0.0) for item in per_seed])
    payload = {
        "protocol": PROTOCOL,
        "step": "seedcheck",
        "experiment": entry["name"],
        "grid_cache_dir": cache_dir,
        "bootstrap_docs": bool(entry["params"].get("bootstrap_docs")),
        "seeds": seeds,
        "per_seed": per_seed,
        "f1_mean": None if f1[0] is None else round(f1[0], 6),
        "f1_std": None if f1[1] is None else round(f1[1], 6),
        "fpr_mean": None if fpr[0] is None else round(fpr[0], 6),
        "fpr_std": None if fpr[1] is None else round(fpr[1], 6),
        "distinct_f1_values": len({round(float(item["tokens"].get("f1") or 0.0), 6) for item in per_seed}),
        "requirement": "std(F1) > 0: пять seed'ов должны быть пятью измерениями, а не одним (дефект v1)",
        "duration_s": round(time.time() - started, 2),
    }
    out = Path(args.out)
    v1.write_json(out / "seedcheck.json", payload)
    print(
        f"seedcheck ({entry['name']}): F1 val {payload['f1_mean']} ± {payload['f1_std']}, значений {payload['distinct_f1_values']}"
    )
    if not payload["f1_std"]:
        print("ЗАПРЕТ ШАГА 3: std(F1) = 0 при 5 seed'ах — разброс не измерен (см. docs/METRIC_SPEC_v2.md, раздел 4)")
        return 1
    return 0


def command_final(args: argparse.Namespace) -> int:
    """Шаг 3: единственный прогон на test по замороженной конфигурации."""
    started = time.time()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("step") != "final-config-v2":
        raise SystemExit(
            "конфигурация не размечена как final-config-v2: шаг 3 выполняется один раз по описанию, "
            "зафиксированному до запуска (docs/EXPERIMENTS_v2.md, раздел «Шаг 3»)"
        )
    split_dir = Path(args.splits)
    parts = parts_of(split_dir)
    info, problems = audit_sizes(parts)
    freeze_path = Path(args.freeze) if args.freeze else Path(args.out) / "freeze.json"
    if not freeze_path.is_file():
        problems.append(f"нет файла заморозки {freeze_path}: шаг 0 обязан предшествовать любому числу")
    else:
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        recorded = freeze.get("splits_sha256") or {}
        for part, block in info.items():
            if recorded.get(part) and recorded[part] != block["sha256"]:
                problems.append(f"{part}: sha256 файла {block['sha256'][:12]} != зафиксированный на шаге 0")
        independence = freeze.get("independence") or {}
        if int(independence.get("overlap_total", -1)) != 0:
            problems.append(f"на шаге 0 пересечение с v1 = {independence.get('overlap_total')} (должно быть 0)")
    sweep_path = Path(args.sweep) if args.sweep else Path(args.out) / "sweep.json"
    if not sweep_path.is_file():
        problems.append(f"нет файла шага 2 ({sweep_path}): конфигурация не отобрана по val")
    else:
        sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
        if (sweep.get("stop_rule") or {}).get("triggered"):
            problems.append(
                "СТОП-ПРАВИЛО ШАГА 2: F1_val ниже цели, измерение на тесте не выполняется "
                "(тестовая выборка остаётся закрытой)"
            )
        if sweep.get("selected") != config.get("name"):
            problems.append(
                f"имя замороженной конфигурации «{config.get('name')}» != отобранная на val «{sweep.get('selected')}»"
            )
        sweep_features = {str(item.get("name")): item for item in sweep.get("results", [])}.get(
            str(config.get("name")), {}
        )
        if sweep_features and int(sweep_features.get("features_count", -1)) != len(config["features"]):
            problems.append("состав признаков в замороженной конфигурации != отобранный на val")
    seedcheck_path = Path(args.seedcheck) if args.seedcheck else Path(args.out) / "seedcheck.json"
    seed_note: dict[str, Any] = {"file": str(seedcheck_path), "f1_std": None, "accepted_zero": None}
    if seedcheck_path.is_file():
        seedcheck = json.loads(seedcheck_path.read_text(encoding="utf-8"))
        seed_note["f1_std"] = seedcheck.get("f1_std")
        if not seedcheck.get("f1_std"):
            if args.allow_zero_std:
                seed_note["accepted_zero"] = str(args.allow_zero_std)
            else:
                problems.append(
                    "std(F1) на val равен 0: сначала seed должен влиять на обучение "
                    "(seedcheck), либо нулевой разброс обязан быть принят явной причиной в --allow-zero-std"
                )
    else:
        problems.append(f"нет проверки seed'ов ({seedcheck_path}): разброс не подтверждён")
    if config.get("splits_sha256") and info:
        for part, digest in config["splits_sha256"].items():
            if info.get(part, {}).get("sha256") and info[part]["sha256"] != digest:
                problems.append(f"{part}: sha256 в замороженной конфигурации != файл на диске")
    if problems:
        print("ШАГ 3 НЕ ВЫПОЛНЯЕТСЯ:")
        for item in problems:
            print(f"  - {item}")
        return 1

    entry = {
        "name": str(config.get("name", "final-v2")),
        "features": [str(name) for name in config["features"]],
        "window": int(config.get("window", 0)),
        "merge_gap": int(config.get("merge_gap", 2)),
        "classifier": str(config.get("classifier", "logreg")),
        "params": dict(config.get("params") or {}),
    }
    caches = CacheSet(args.grid_cache)
    cache, cache_dir = caches.for_entry(entry)
    train = v1.load_split(parts["train"])
    val = v1.load_split(parts["val"]) if args.include_val else []
    test = v1.load_split(parts["test"])
    fit_records = list(train) + list(val)
    bootstrap = bool(entry["params"].get("bootstrap_docs"))
    seeds = [int(value) for value in str(args.seeds).split(",") if value.strip()]
    threshold = float(config["threshold"])
    print(f"порог заморожен до шага 3: {threshold} (правило {config.get('threshold_rule', 'E1')}, {config_path.name})")
    per_seed: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    curve_source: Sequence[dict[str, Any]] = []
    for seed in seeds:
        model, _, eval_rows = fit_and_score(cache, entry, fit_records, test, seed, bootstrap=bootstrap)
        for row in eval_rows:
            row["id"] = f"{args.corpus_name}:{row['id']}"
        metrics = v1.score_metrics(eval_rows, threshold, int(entry["merge_gap"]))
        per_seed.append(
            {
                "seed": seed,
                "threshold": threshold,
                "model_sha256": hashlib.sha256(json.dumps(model.to_dict(), sort_keys=True).encode("utf-8")).hexdigest(),
                "tokens": metrics["tokens"],
                "answers": metrics["answers"],
                "spans": metrics["spans"],
                "rows": metrics["rows"],
                "positive_rows": metrics["positive_rows"],
            }
        )
        if not curve_source:
            curve_source = list(eval_rows)
        for row in eval_rows:
            predictions.append(
                {
                    "id": row["id"],
                    "level": "tokens",
                    "seed": seed,
                    "token": row["token"],
                    "start": row["start"],
                    "end": row["end"],
                    "gold": int(row["gold"]),
                    "score": round(float(row["score"]), 12),
                    "pred": int(float(row["score"]) >= threshold),
                }
            )
        numbers = metrics["tokens"]
        print(
            f"seed {seed}: F1(test) {numbers.get('f1')} FPR(test) {numbers.get('fpr')} "
            f"P {numbers.get('precision')} R {numbers.get('recall')}"
        )
    sweep_results = {
        str(item.get("name")): item
        for item in (json.loads(sweep_path.read_text(encoding="utf-8")).get("results") or [])
    }
    chosen = sweep_results.get(str(config.get("name"))) or {}
    val_reference = {
        "file": str(sweep_path),
        "e1_val_chosen": chosen.get("e1_val_chosen"),
        "e0_train_cv": chosen.get("e0_train_cv"),
        "note": (
            "числа отбора на val (из sweep.json) — рядом с итогом на test, чтобы разрыв между "
            "отобранной оценкой и результатом на независимой выборке был виден, а не молчал"
        ),
    }
    aggregate = v1.aggregate_seeds(per_seed)
    # Поле `criterion` считается по правилу v1 (тот же `v1.criterion_status`), чтобы
    # `v1.compare_manifest` сверял манифест v2 без поблажек; цели цикла v2 записаны
    # отдельными полями — внутренняя 0,70 и контрактный минимум 0,60.
    legacy_status = v1.criterion_status(aggregate)
    internal = goal_status(aggregate, GOAL_INTERNAL_F1)
    contract = goal_status(aggregate, GOAL_CONTRACT_F1)
    out = Path(args.out)
    gold_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in test:
        identifier = f"{args.corpus_name}:{record.get('id')}"
        if identifier in seen:
            continue
        seen.add(identifier)
        gold_rows.append({"id": identifier, "spans": [list(span) for span in v1.spans_of(record)]})
    rows_written = v1.write_jsonl(out / "predictions_test.jsonl.gz", predictions)
    v1.write_jsonl(out / "gold_test.jsonl.gz", gold_rows)
    curve: dict[str, Any] = {}
    if len(curve_source) == len(eval_rows) and curve_source:
        scores = [float(row["score"]) for row in curve_source]
        curve = {
            "note": "кривая на test записывается ПОСЛЕ единственного прогона (шаг 3) и не использовалась для отбора",
            "seed": seeds[0],
            "points": curve_on(curve_source, int(entry["merge_gap"]), threshold_candidates(scores)),
        }
        v1.write_json(out / "test_curve.json", curve)
    payload = {
        "step": "final-v2",
        "protocol": PROTOCOL,
        "code_commit": v1._git_commit(),
        "freeze_file": str(freeze_path),
        "freeze_sha256": v1.sha256_file(freeze_path) if freeze_path.is_file() else "",
        "sweep_file": str(sweep_path),
        "sweep_sha256": v1.sha256_file(sweep_path) if sweep_path.is_file() else "",
        "config_file": str(config_path),
        "config_sha256": v1.sha256_file(config_path),
        "config": config,
        "metric_spec_v2": "docs/METRIC_SPEC_v2.md",
        "metric_spec_v2_sha256": v1.sha256_file(ROOT / "docs" / "METRIC_SPEC_v2.md"),
        "metric_spec_inherited": "docs/METRIC_SPEC.md",
        "metric_spec_inherited_sha256": v1.sha256_file(ROOT / "docs" / "METRIC_SPEC.md"),
        "corpus": args.corpus_name,
        "splits_dir": str(split_dir),
        "splits_sha256": {part: block["sha256"] for part, block in info.items()},
        "pairs": {part: block["pairs"] for part, block in info.items()},
        "documents": {part: block["documents"] for part, block in info.items()},
        "fit_on": "train+val" if val else "train",
        "seeds": seeds,
        "seed_rule": (
            "seed'ы зафиксированы до запуска; обучение с бутстрапом по документам train "
            f"(bootstrap_docs={bootstrap}), отбор и порог общие для всех seed'ов; лучший seed не выбирается"
        ),
        "seed_variance": seed_note,
        "threshold": threshold,
        "threshold_rule": config.get("threshold_rule", "E1"),
        "threshold_source": (
            f"выбран на val по правилу {config.get('threshold_rule', 'E1')} (максимум F1 при FPR ≤ {FPR_MAX}) "
            f"и зафиксирован в {config_path.name} до шага 3; переселекции по test нет"
        ),
        "threshold_frozen": True,
        "thresholds_per_seed": [float(item["threshold"]) for item in per_seed],
        "grid_cache_dir": cache_dir,
        "model": v1.cache_model_provenance(cache),
        "per_seed": per_seed,
        "metrics_mean_std": aggregate,
        "criterion": legacy_status,
        "goal_internal": internal,
        "goal_contract_min": contract,
        "val_reference": val_reference,
        "test_curve": curve.get("points", []),
        "predictions": "predictions_test.jsonl.gz",
        "prediction_rows": rows_written,
        "gold_file": "gold_test.jsonl.gz",
        "gold_rows": len(gold_rows),
        "gold_sha256": v1.sha256_file(out / "gold_test.jsonl.gz"),
        "predictions_format": list(v1.PREDICTIONS_FIELDS),
        "hardware": v1.hardware_manifest(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "duration_s": round(time.time() - started, 2),
        "sampling_note": None,
        "deviations": list(config.get("deviations") or []),
    }
    v1.write_json(out / "manifest.json", payload)
    print(
        f"итог v2: F1(токены, test) {internal['token_f1']} ± {aggregate.get('f1_std')} "
        f"FPR {internal['token_fpr']} → цель 0,70: {internal['status']}; контрактный минимум 0,60: {contract['status']}"
    )
    print(f"строк предсказаний {rows_written}; манифест {out / 'manifest.json'}")
    return 0


def command_verify(args: argparse.Namespace) -> int:
    """Пересчёт каждого числа манифеста из строк предсказаний + проверка разбиения."""
    manifest_path = Path(args.manifest)
    out_dir = manifest_path.parent
    predictions = Path(args.predictions) if args.predictions else out_dir / "predictions_test.jsonl.gz"
    gold = out_dir / "gold_test.jsonl.gz"
    problems, numbers = v1.compare_manifest(manifest_path, predictions, gold if gold.is_file() else None)
    split_problems: list[str] = []
    if args.check_splits:
        numbers_split, split_problems = v1.check_splits(manifest_path)
        numbers.update(numbers_split)
        problems += split_problems
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    aggregate = manifest.get("metrics_mean_std") or {}
    for key, goal in (("goal_internal", GOAL_INTERNAL_F1), ("goal_contract_min", GOAL_CONTRACT_F1)):
        expected = goal_status(aggregate, goal)
        recorded = manifest.get(key) or {}
        if str(recorded.get("status")) != expected["status"]:
            problems.append(f"{key}: манифест «{recorded.get('status')}», пересчёт «{expected['status']}»")
    print(json.dumps(numbers, ensure_ascii=False))
    if problems:
        for item in problems:
            print(f"ПРОБЛЕМА: {item}")
        return 1
    print(
        f"числа воспроизведены: строк {numbers['rows']}, seed'ов {numbers['seeds']}, F1 {numbers['f1']}"
        f" ± {numbers['f1_std']}, FPR {numbers['fpr']}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Протокол цикла v2 (независимая выборка, цель 0,70)")
    sub = parser.add_subparsers(dest="command", required=True)

    freeze = sub.add_parser("freeze", help="шаг 0: аудит разбиения + независимость + фиксация правил")
    freeze.add_argument("--splits", default="data/corpus_v2/splits")
    freeze.add_argument("--config", default="config/hf_experiments_v2.json")
    freeze.add_argument("--seeds", default="42,43,44,45,46")
    freeze.add_argument("--out", default="reports/v2")
    freeze.set_defaults(func=command_freeze)

    for name, help_text in (
        ("curve", "шаг 1: кривые F1/FPR на val и оба правила порога"),
        ("sweep", "шаг 2: до 10 гипотез, отбор по val, стоп-правило"),
        ("seedcheck", "проверка, что seed влияет на обучение (std > 0) — до шага 3"),
    ):
        node = sub.add_parser(name, help=help_text)
        node.add_argument("--config", required=True)
        node.add_argument("--grid-cache", required=True)
        node.add_argument("--splits", default="data/corpus_v2/splits")
        node.add_argument("--out", default="reports/v2")
        if name == "curve":
            node.add_argument("--split", default="val", choices=["val"])
            node.add_argument("--experiment", default="")
        if name == "seedcheck":
            node.add_argument("--seeds", default="42,43,44,45,46")
            node.add_argument("--experiment", default="")
        node.set_defaults(func={"curve": command_curve, "sweep": command_sweep, "seedcheck": command_seedcheck}[name])

    final = sub.add_parser("final", help="шаг 3: единственный прогон на test по замороженной конфигурации")
    final.add_argument("--config", required=True)
    final.add_argument("--grid-cache", required=True)
    final.add_argument("--splits", default="data/corpus_v2/splits")
    final.add_argument("--out", default="reports/v2")
    final.add_argument("--corpus-name", default="v2")
    final.add_argument("--seeds", default="42,43,44,45,46")
    final.add_argument("--include-val", action="store_true")
    final.add_argument("--freeze", default="")
    final.add_argument("--sweep", default="")
    final.add_argument("--seedcheck", default="")
    final.add_argument("--allow-zero-std", default="", help="явная причина принять std = 0 (иначе шаг 3 запрещён)")
    final.set_defaults(func=command_final)

    verify = sub.add_parser("verify-manifest", help="пересчёт чисел манифеста из предсказаний")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--predictions", default="")
    verify.add_argument("--check-splits", action="store_true")
    verify.set_defaults(func=command_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
