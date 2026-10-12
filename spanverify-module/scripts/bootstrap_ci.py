#!/usr/bin/env python3
"""Доверительный интервал для F1 по бутстрапу **по документам** (не по токенам).

Зачем: точечное `F1 = 0,595041` на test и `0,515702` на val говорит только «сколько получилось».
Чтобы понимать, насколько это число устойчиво к составу выборки, нужен интервал, считают его по
документам (токены одного документа не независимы — иначе интервал искусственно узкий), и считать
его надо при **том же замороженном пороге**: переселэкция порога внутри бутстрапа оценивала бы уже
другую процедуру, а протокол её запрещает.

Вход — пофайловые оценки, которые лежат в репозитории (`predictions_*.jsonl.gz`); метрика берётся
из `score >= threshold`, как в `spanverify.engine._token_metrics`. Ни признаков, ни модели скрипт
не читает: он описывает неопределённость уже зафиксированного числа и не участвует в отборе.

Команды:

    python3 scripts/bootstrap_ci.py --predictions reports/hf_protocol/predictions_val.jsonl.gz \\
        --manifest reports/hf_protocol/manifest_val.json --out reports/hf_protocol/bootstrap_val.json
    python3 scripts/bootstrap_ci.py --predictions reports/hf_final/predictions_test.jsonl.gz \\
        --manifest reports/hf_final/manifest.json --out reports/hf_protocol/bootstrap_test.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT / "scripts"))


PROTOCOL = "bootstrap-ci-v1"


def document_of(record: dict[str, Any]) -> str:
    """Документ — по id пары (`a3:corpus-a3-00034` → `corpus-a3-00034`).

    Отдельного поля `doc` в `predictions_*` нет, а id пары несёт префикс корпуса: снимать надо
    именно префикс, иначе все 177 пар A3 и A1 слились бы в одну выборку.
    """
    identifier = str(record.get("id") or record.get("pair_id") or "")
    document = str(record.get("doc") or "").strip()
    if document:
        return document
    _, _, tail = identifier.partition(":")
    return tail or identifier


def load_rows(path: Path, seed: int | None) -> list[dict[str, Any]]:
    opener = gzip.open if path.name.endswith(".gz") else open
    rows: list[dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if str(record.get("level")) != "tokens":
                continue
            if seed is not None and int(record.get("seed", seed)) != seed:
                continue
            rows.append({"doc": document_of(record), "gold": int(record["gold"]), "score": float(record["score"])})
    if not rows:
        raise SystemExit(f"в {path} нет строк уровня tokens (для выбранного seed)")
    return rows


def metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, float]:
    """Token-метрики при фиксированном пороге (TP/FP/FN — как в `_token_metrics`)."""
    tp = fp = fn = 0
    for row in rows:
        predicted = row["score"] >= threshold
        if predicted and row["gold"]:
            tp += 1
        elif predicted:
            fp += 1
        elif row["gold"]:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def percentile_of(values: list[float], level: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * level
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def bootstrap(rows: list[dict[str, Any]], threshold: float, reps: int, seed: int) -> dict[str, Any]:
    """Бутстрап по документам: повтор = выборка документов с возвращением, метрика — по их токенам."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["doc"], []).append(row)
    keys = sorted(groups)
    rng = random.Random(seed)
    point = metrics(rows, threshold)
    values: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    for _ in range(reps):
        picked: list[dict[str, Any]] = []
        for _ in keys:
            picked.extend(groups[keys[rng.randrange(len(keys))]])
        numbers = metrics(picked, threshold)
        values.append(numbers["f1"])
        precisions.append(numbers["precision"])
        recalls.append(numbers["recall"])
    mean = sum(values) / len(values)
    spread = (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
    return {
        "documents": len(keys),
        "tokens": len(rows),
        "reps": reps,
        "seed": seed,
        "point_f1": round(point["f1"], 6),
        "ci95_f1": [round(percentile_of(values, 0.025), 6), round(percentile_of(values, 0.975), 6)],
        "mean_f1": round(mean, 6),
        "std_f1": round(spread, 6),
        "ci95_precision": [round(percentile_of(precisions, 0.025), 6), round(percentile_of(precisions, 0.975), 6)],
        "ci95_recall": [round(percentile_of(recalls, 0.025), 6), round(percentile_of(recalls, 0.975), 6)],
        "point_precision": round(point["precision"], 6),
        "point_recall": round(point["recall"], 6),
        "confusion": {key: int(point[key]) for key in ("tp", "fp", "fn")},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="95 % ДИ для F1: бутстрап по документам при фиксированном пороге")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--manifest", default=None, help="откуда взять зафиксированный порог и сверить метрику")
    parser.add_argument(
        "--threshold", type=float, default=None, help="переопределить порог (по умолчанию — из манифеста)"
    )
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", default=None, help="подпись выборки в выводе (val / test)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    path = Path(args.predictions)
    if not path.is_absolute():
        path = MODULE_ROOT / path
    if not path.is_file():
        raise SystemExit(f"нет файла оценок {path}")
    manifest: dict[str, Any] = {}
    manifest_path = None
    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.is_absolute():
            manifest_path = MODULE_ROOT / manifest_path
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    threshold = args.threshold
    if threshold is None:
        threshold = manifest.get("threshold") or (manifest.get("metrics_mean_std") or {}).get("threshold")
    if threshold is None:
        raise SystemExit("порог не задан: нужен --manifest с полем threshold или --threshold")
    threshold = float(threshold)

    # seed: если манифест сообщает, что все seed'ы дали одно и то же число, берём один seed —
    # иначе 5 копий одной оценки сузили бы интервал в sqrt(5) раз без всякого нового измерения.
    per_seed = [float(value) for value in (manifest.get("metrics_mean_std") or {}).get("f1_per_seed", [])]
    collapse_seeds = bool(per_seed) and max(per_seed) - min(per_seed) <= 1e-9
    rows = load_rows(path, args.seed if collapse_seeds else None)
    result = bootstrap(rows, threshold, args.reps, args.seed)
    result |= {
        "protocol": PROTOCOL,
        "sample": args.sample or ("val" if "val" in path.name else "не указано"),
        "threshold": round(threshold, 6),
        "threshold_source": ("манифест " + manifest_path.name) if manifest_path else "флаг --threshold",
        "seeds_included": [args.seed] if collapse_seeds else "все (метрики по seed'ам различаются)",
        "seeds_collapse_note": (
            "все seed'ы дали одинаковую метрику (std = 0), поэтому взят один: 5 копий той же оценки "
            "не являются пятью измерениями и сузили бы интервал искусственно"
            if collapse_seeds
            else "метрики по seed'ам различаются, поэтому использованы строки всех seed'ов"
        ),
        "estimator": "бутстрап по документам (выбор документов с возвращением), метрика — по токенам пула",
        "threshold_fixed": True,
    }
    if manifest:
        numbers = manifest.get("metrics_mean_std") or manifest.get("tokens") or {}
        recorded = numbers.get("f1")
        if recorded is not None:
            drift = abs(float(recorded) - float(result["point_f1"]))
            result["manifest_cross_check"] = {
                "recorded_f1": float(recorded),
                "recomputed_f1": result["point_f1"],
                "abs_расхождение": round(drift, 9),
            }
            if drift > 2e-5:
                raise SystemExit(
                    f"пересчёт не совпал с манифестом: {result['point_f1']} против {recorded} — "
                    "файл оценок и манифест описывают разные прогоны"
                )
    lower, upper = result["ci95_f1"]
    print(
        f"{result['sample']}: документов {result['documents']}, токенов {result['tokens']}, "
        f"порог {result['threshold']}"
    )
    print(
        f"F1 = {result['point_f1']} (precision {result['point_precision']}, recall {result['point_recall']}), "
        f"95 % ДИ по документам: [{lower}, {upper}] (повторов {result['reps']}, seed {result['seed']})"
    )
    print(f"recall 95 % ДИ: {result['ci95_recall']}, precision 95 % ДИ: {result['ci95_precision']}")
    print(f"семена: {result['seeds_included']}; {result['seeds_collapse_note']}")
    if "manifest_cross_check" in result:
        print(f"сверка с манифестом: {result['manifest_cross_check']}")
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = MODULE_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"записано: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
