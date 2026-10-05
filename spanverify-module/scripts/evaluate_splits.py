"""Оценка режима на готовых разбиениях корпуса (train/dev/test) без утечки.

Скрипт считает те же метрики, что и ``run_experiments``, но по уже
зафиксированным файлам ``splits/*.jsonl`` и с разбором источников ложных
срабатываний на верных ответах (``relaxed_fpr_sources``). Он нужен, чтобы
правки признаков и правил покрытия проверялись на dev, а тест читался только
для итогового числа (порог на тесте не подбирается).

Запуск::

    python scripts/evaluate_splits.py --splits data/corpus_a/splits --mode demo --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spanverify.dataset import read_pairs  # noqa: E402
from spanverify.engine import Verifier  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Метрики по разбиениям корпуса")
    parser.add_argument("--splits", default="data/corpus_a/splits")
    parser.add_argument("--mode", choices=["demo", "hf"], default="demo")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--weights", default="config/weights.json", help="файл параметров (для режима hf — config/weights_hf.json)"
    )
    parser.add_argument("--feature-cache", default=None, help="каталог кэша признаков hf (повторный прогон без модели)")
    parser.add_argument("--splits-names", default="dev,test")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=0, help="ограничить число пар в каждом разбиении")
    args = parser.parse_args(argv)

    import dataclasses

    from spanverify.config import Config

    config = Config.load()
    if args.feature_cache:
        config = dataclasses.replace(config, hf_feature_cache=args.feature_cache)
    verifier = Verifier(mode=args.mode, model_name=args.model, weights_path=args.weights, config=config)
    report: dict = {
        "mode": args.mode,
        "model": args.model,
        "weights": args.weights,
        "feature_cache": args.feature_cache,
        "splits": {},
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    for name in args.splits_names.split(","):
        path = Path(args.splits) / f"{name.strip()}.jsonl"
        if not path.is_file():
            print(f"нет файла разбиения: {path}", file=sys.stderr)
            continue
        pairs = list(read_pairs(path))
        if args.limit:
            pairs = pairs[: args.limit]
        metrics = verifier.evaluate(pairs)
        report["splits"][name] = metrics
        printed = {
            "verdict_f1": metrics["verdicts"]["f1"],
            "verdict_precision": metrics["verdicts"]["precision"],
            "verdict_recall": metrics["verdicts"]["recall"],
            "fpr_faithful": metrics["verdicts"]["fpr"],
            "faithful_pairs": metrics["by_type"].get("faithful", {}).get("pairs", 0),
            "faithful_flagged": metrics["by_type"].get("faithful", {}).get("fp", 0),
            "token_f1": metrics["tokens"]["f1"],
            "span_f1_strict": metrics["spans"].get("f1"),
            "span_recall_containment": metrics["spans"].get("recall_containment"),
            "width_ratio_narrow": metrics["spans"].get("mean_width_ratio"),
            "width_ratio_expanded": metrics["spans"].get("mean_width_ratio_expanded"),
            "coverage_expanded": metrics["spans"].get("coverage_expanded"),
            "recall_by_type": {key: value["recall"] for key, value in sorted(metrics["by_type"].items())},
        }
        print(f"\n=== {name} ({len(pairs)} пар) ===")
        print(json.dumps(printed, ensure_ascii=False, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False)[:400])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
