#!/usr/bin/env python3
"""Кривая порога на val: проверка утверждения «правило порога уже даёт максимум F1 при FPR ≤ 0,40».

Зачем отдельный скрипт: вывод «дефицит F1 — в полноте, а не в операционной точке» нельзя принимать
на слово. Он проверяется числом: на валидационных оценках (те же 7 969 токенов, что и в
`reports/hf_protocol/manifest_val.json`) перебираются ВСЕ различные значения скора, и для каждого
считаются те же метрики, что считает протокол (`hf_protocol.score_metrics`). Если бы под потолком
FPR существовала точка с F1 выше выбранной — утверждение было бы неверно, и это было бы видно здесь.

Данные — только val. Тест не читается: файл с `test` в имени отвергается до вычислений, и ни одна
ветка кода не открывает `predictions_test*` (тестовых пофайловых оценок в репозитории нет).

Что делает дополнительно к простому максимуму:
* сверяет быстрый монотонный проход по кандидатам с официальной функцией метрик на трёх
  контрольных порогах (если разойдутся — скрипт падает, а не рисует удобный ответ);
* отдельно посчитанную точку `choose_threshold` на этих же данных: она обязана совпасть с
  максимумом под потолком (это и есть проверяемое свойство правила);
* пишет таблицу (прореженную) и выводы в JSON, чтобы каждое число в документе имело команду.

Команда:

    python3 scripts/val_curve.py \\
      --predictions reports/hf_protocol/predictions_val.jsonl.gz \\
      --manifest reports/hf_protocol/manifest_val.json \\
      --out reports/hf_protocol/val_curve.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT / "scripts"))

import hf_protocol  # noqa: E402

PROTOCOL = "val-curve-v1"
CONTROL_THRESHOLDS = 3


def guard_no_test(path: Path, what: str) -> None:
    """Отказ до вычислений, если в имени входного файла есть `test`.

    Повторяет страж `scripts/val_diagnostics.py`: проверка на «правильность пути» не считается,
    потому что имя файла — единственное, чем val отличается от test в этом каталоге.
    """
    lowered = path.name.lower()
    if "test" in str(path).lower() and "test" in lowered:
        raise SystemExit(f"{what}: файл `{path}` содержит `test` в имени — диагностика читает только val")


def load_rows(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.name.endswith(".gz") else open
    rows: list[dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if str(record.get("level")) != "tokens":
                continue
            rows.append(
                {
                    "id": str(record["id"]),
                    "start": int(record["start"]),
                    "end": int(record["end"]),
                    "gold": int(record["gold"]),
                    "score": float(record["score"]),
                }
            )
    if not rows:
        raise SystemExit(f"в {path} нет строк уровня `tokens`")
    return rows


def scan_curve(rows: list[dict[str, Any]], merge_gap: int, cap: float) -> dict[str, Any]:
    """Кривая по всем различным скорам + быстрый монотонный обход, и сверка с `score_metrics`.

    Порог `t` означает «флаг там, где score ≥ t». При понижении порога TP и FP не убывают, поэтому
    FPR монотонен — на этом держится и правило `choose_threshold`, и проверка «нет ли точки выше».
    """
    order = sorted(range(len(rows)), key=lambda index: -rows[index]["score"])
    positives = sum(row["gold"] for row in rows)
    negatives = len(rows) - positives
    if not positives or not negatives:
        raise SystemExit("вырожденная разметка: нужны и положительные, и отрицательные токены")

    points: list[dict[str, Any]] = []
    tp = fp = 0
    cursor = 0
    while cursor < len(order):
        score = rows[order[cursor]]["score"]
        while cursor < len(order) and rows[order[cursor]]["score"] == score:
            if rows[order[cursor]]["gold"]:
                tp += 1
            else:
                fp += 1
            cursor += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / positives
        fpr = fp / negatives
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        points.append(
            {
                "threshold": round(float(score), 8),
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "fpr": round(fpr, 6),
                "flagged": tp + fp,
            }
        )

    # Контрольная сверка: быстрые TP/FP против официальной функции метрик на нескольких порогах.
    checks: list[dict[str, Any]] = []
    picks = [points[0], points[len(points) // 2], points[-1]]
    for pick in picks[:CONTROL_THRESHOLDS]:
        reference = hf_protocol.score_metrics([dict(row) for row in rows], float(pick["threshold"]), merge_gap)[
            "tokens"
        ]
        checks.append(
            {
                "threshold": pick["threshold"],
                "fast": {
                    "precision": pick["precision"],
                    "recall": pick["recall"],
                    "f1": pick["f1"],
                    "fpr": pick["fpr"],
                },
                "score_metrics": {
                    "precision": round(float(reference["precision"]), 6),
                    "recall": round(float(reference["recall"]), 6),
                    "f1": round(float(reference["f1"]), 6),
                    "fpr": round(float(reference["fpr"]), 6),
                },
            }
        )
        for key in ("precision", "recall", "f1", "fpr"):
            if abs(float(checks[-1]["fast"][key]) - float(checks[-1]["score_metrics"][key])) > 2e-5:
                raise SystemExit(
                    f"расхождение метрик на пороге {pick['threshold']} по {key}: "
                    f"{checks[-1]['fast'][key]} против {checks[-1]['score_metrics'][key]}"
                )

    under_cap = [point for point in points if float(point["fpr"]) <= cap]
    best_capped = max(under_cap, key=lambda point: (float(point["f1"]), -float(point["fpr"]))) if under_cap else None
    best_any = max(points, key=lambda point: float(point["f1"]))
    return {
        "candidates": len(points),
        "distinct_thresholds": len({point["threshold"] for point in points}),
        "checks": checks,
        "curve": points,
        "best_capped": best_capped,
        "best_any": best_any,
        "positive_tokens": positives,
        "negative_tokens": negatives,
        "tokens": len(rows),
    }


def _relative(path: Path) -> str:
    """Путь для вывода: вне каталога модуля (временные каталоги тестов) показываем как есть."""
    try:
        return str(path.relative_to(MODULE_ROOT))
    except ValueError:
        return str(path)


def brief(points: list[dict[str, Any]], size: int = 28) -> list[dict[str, Any]]:
    """Прореженная таблица для человека и для документа (полная кривая остаётся в JSON)."""
    if len(points) <= size:
        return points
    step = max(1, len(points) // (size - 1))
    kept = [points[index] for index in range(0, len(points), step)]
    if points[-1] not in kept:
        kept.append(points[-1])
    return kept


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="кривая F1/FPR по порогам на val (тест не читается)")
    parser.add_argument("--predictions", default="reports/hf_protocol/predictions_val.jsonl.gz")
    parser.add_argument("--manifest", default="reports/hf_protocol/manifest_val.json")
    parser.add_argument("--out", default="reports/hf_protocol/val_curve.json")
    parser.add_argument("--fpr-cap", type=float, default=hf_protocol.CRITERION_FPR)
    parser.add_argument("--criterion-f1", type=float, default=hf_protocol.CRITERION_F1)
    args = parser.parse_args(argv)

    predictions = (MODULE_ROOT / args.predictions).resolve()
    manifest_path = (MODULE_ROOT / args.manifest).resolve()
    out_path = (MODULE_ROOT / args.out).resolve()
    guard_no_test(predictions, "предсказания")
    guard_no_test(manifest_path, "манифест")
    guard_no_test(out_path, "файл вывода")
    if not predictions.is_file():
        raise SystemExit(f"нет файла оценок {predictions} — он создаётся прогоном шага 1 (`make hf-sweep`)")

    rows = load_rows(predictions)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    merge_gap = int(manifest.get("merge_gap", 2))
    numbers = manifest.get("metrics_mean_std") or {}
    recorded_threshold = manifest.get("threshold", numbers.get("threshold"))

    scan = scan_curve(rows, merge_gap, args.fpr_cap)
    rule = hf_protocol.choose_threshold(
        [float(row["score"]) for row in rows], [int(row["gold"]) for row in rows], args.fpr_cap
    )

    capped = scan["best_capped"] or {}
    any_point = scan["best_any"] or {}
    # Точку на записанном пороге считаем официальной функцией метрик, а не поиском по кандидатам:
    # порог в манифесте округлён (0,1106), и совпадения со скором конкретного токена может не быть.
    recorded_point: dict[str, Any] | None = None
    if recorded_threshold is not None:
        tokens_at = hf_protocol.score_metrics([dict(row) for row in rows], float(recorded_threshold), merge_gap)[
            "tokens"
        ]
        recorded_point = {
            "threshold": float(recorded_threshold),
            "precision": round(float(tokens_at["precision"]), 6),
            "recall": round(float(tokens_at["recall"]), 6),
            "f1": round(float(tokens_at["f1"]), 6),
            "fpr": round(float(tokens_at["fpr"]), 6),
            "auc": round(float(tokens_at["auc"]), 6),
            "flagged": int(tokens_at.get("tp", 0)) + int(tokens_at.get("fp", 0)),
        }
    higher = [
        point
        for point in scan["curve"]
        if capped and float(point["fpr"]) <= args.fpr_cap and float(point["f1"]) > float(capped["f1"]) + 1e-12
    ]
    at_recorded = recorded_point or {}

    cross_check: dict[str, Any] = {"available": False}
    if recorded_point and numbers:
        # Совпадение с манифестом — это проверка, что мы читаем ту же выборку и те же метрики.
        pass
    if recorded_point and numbers:
        pairs = {
            "precision": (float(at_recorded["precision"]), float(numbers.get("precision", at_recorded["precision"]))),
            "recall": (float(at_recorded["recall"]), float(numbers.get("recall", at_recorded["recall"]))),
            "f1": (float(at_recorded["f1"]), float(numbers.get("f1", at_recorded["f1"]))),
            "fpr": (float(at_recorded["fpr"]), float(numbers.get("fpr", at_recorded["fpr"]))),
        }
        for key, (mine, theirs) in pairs.items():
            if abs(mine - theirs) > 2e-5:
                raise SystemExit(
                    f"кривая не сходится с манифестом на пороге {recorded_threshold} по {key}: "
                    f"{mine} против {theirs} — сравниваются разные выборки или разные метрики"
                )
        cross_check = {
            "available": True,
            "threshold": recorded_threshold,
            "max_расхождение": max(abs(mine - theirs) for mine, theirs in pairs.values()),
            "с": str(manifest_path.name),
        }

    payload = {
        "protocol": PROTOCOL,
        "purpose": "проверка, что правило порога даёт максимум F1 при FPR ≤ 0,40 на val",
        "not_for_criterion": True,
        "test_not_read": True,
        "inputs": {
            "predictions": _relative(predictions),
            "predictions_sha256": hf_protocol.sha256_file(predictions),
            "manifest": _relative(manifest_path) if manifest_path.is_file() else None,
            "threshold_recorded": recorded_threshold,
            "code_commit": hf_protocol._git_commit(),
            "grid_sha256": manifest.get("grid_sha256"),
        },
        "split": {
            "name": "val (A3 dev)",
            "tokens": scan["tokens"],
            "positive_tokens": scan["positive_tokens"],
            "negative_tokens": scan["negative_tokens"],
        },
        "fpr_cap": args.fpr_cap,
        "criterion_f1": args.criterion_f1,
        "rule_on_val": rule,
        "best_capped_by_full_scan": capped,
        "best_unconstrained_oracle": any_point,
        "at_recorded_threshold": at_recorded,
        "answer": {
            "есть_точка_выше_под_потолком": bool(higher),
            "число_таких_точек": len(higher),
            "правило_совпало_с_полным_перебором": bool(capped) and abs(float(capped["f1"]) - float(rule["f1"])) <= 2e-5,
            "порог_правила_совпал_с_замороженным": (
                recorded_threshold is not None and abs(float(rule["threshold"]) - float(recorded_threshold)) <= 1e-4
            ),
            "не_хватает_до_критерия_на_val": round(args.criterion_f1 - float(at_recorded.get("f1", rule["f1"])), 6),
        },
        "manifest_cross_check": cross_check,
        "table": brief(scan["curve"]),
        "reading": (
            "кривая построена по val; «oracle» — максимум F1 без ограничения по FPR, измеренный на той же "
            "выборке, поэтому он не результат, а верхняя граница того, что вообще даёт этот скор"
        ),
        "curve_full": scan["curve"],
        "checks": scan["checks"],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    print(
        f"val: токенов {scan['tokens']}, положительных {scan['positive_tokens']}, кандидатов порога {scan['candidates']}"
    )
    print(f"правило choose_threshold на val: порог {rule['threshold']}, F1 {rule['f1']}, FPR {rule['fpr']}")
    print(
        f"полный перебор под потолком FPR ≤ {args.fpr_cap}: F1 {capped.get('f1')} при FPR {capped.get('fpr')}"
        f" (порог {capped.get('threshold')})"
    )
    print(f"oracle без ограничения: F1 {any_point.get('f1')} при FPR {any_point.get('fpr')}")
    if recorded_threshold is not None:
        print(
            f"записанный порог {recorded_threshold}: F1 {at_recorded.get('f1')}, FPR {at_recorded.get('fpr')},"
            f" precision {at_recorded.get('precision')}, recall {at_recorded.get('recall')}"
        )
    print(
        "точек с FPR ≤ {cap} и F1 выше выбранного: {n} → {verdict}".format(
            cap=args.fpr_cap, n=len(higher), verdict="нет" if not higher else "ЕСТЬ"
        )
    )
    print(f"записано: {_relative(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
