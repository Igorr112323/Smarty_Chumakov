"""Обучение отдельного набора параметров для режима ``hf`` (без подмены демо-весов).

Зачем скрипт: ``config/weights.json`` обучен на демонстрационных (лексических)
признаках. Подавать те же веса в режим ``hf`` нельзя — признаки там другой природы,
и риск вырождается в 1.0 на всех парах (это наблюдалось и записано в отчёте аудита).
Поэтому для режима ``hf`` обучается собственный набор ``config/weights_hf.json``
(и ``config/head_hf.json``), причём **только на обучающем разбиении** корпуса
(``splits/train.jsonl``), с разбиением по документам. Числа на dev/test считаются
отдельным скриптом ``evaluate_splits.py`` и на обучение не влияют.

Запуск::

    python scripts/train_hf_bundle.py --model /path/или/имя --splits data/corpus_a/splits \\
        --feature-cache data/cache/hf_features --out-json reports/hf_bundle_train.json

Признаки кэшируются: повторный прогон оценки по dev/test не требует модели.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import sys
import time
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spanverify import __version__  # noqa: E402
from spanverify.config import Config  # noqa: E402
from spanverify.dataset import read_pairs  # noqa: E402
from spanverify.engine import Verifier  # noqa: E402
from spanverify.train import train  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Обучить параметры режима hf на обучающем разбиении")
    parser.add_argument("--model", required=True, help="имя модели на HuggingFace или путь к папке с весами")
    parser.add_argument("--splits", default="data/corpus_a/splits", help="каталог с train/dev/test")
    parser.add_argument("--train-file", default="train.jsonl")
    parser.add_argument("--feature-cache", default="data/cache/hf_features", help="каталог кэша признаков")
    parser.add_argument("--weights-out", default="config/weights_hf.json")
    parser.add_argument("--head-out", default="config/head_hf.json")
    parser.add_argument("--out-json", default="reports/hf_bundle_train.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-fpr", type=float, default=0.1)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="ограничить число обучающих пар (по умолчанию все)")
    args = parser.parse_args(argv)

    train_path = Path(args.splits) / args.train_file
    pairs = list(read_pairs(train_path))
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        print(f"нет обучающих пар: {train_path}", file=sys.stderr)
        return 2

    config = Config.load()
    if args.feature_cache:
        config = dataclasses.replace(config, hf_feature_cache=args.feature_cache)
    verifier = Verifier(mode="hf", model_name=args.model, config=config)

    started = time.perf_counter()
    report = train(
        pairs,
        mode="hf",
        seed=args.seed,
        folds=args.folds,
        target_fpr=args.target_fpr,
        dataset_name=str(train_path),
        verifier=verifier,
        group_split=True,
    )
    elapsed = time.perf_counter() - started

    weights_path = Path(args.weights_out)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = report.bundle
    payload = bundle.to_dict()
    payload["mode"] = "hf"
    payload["meta"] = {
        **(payload.get("meta") or {}),
        "mode": "hf",
        "model": args.model,
        "calibrated_on": f"{train_path} (обучающее разбиение корпуса A; разбиение по документам)",
        "n_train_pairs": len(pairs),
        "held_out": "dev/test корпуса A оцениваются отдельно scripts/evaluate_splits.py",
        "hardware": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor() or platform.machine(),
        },
        "seconds": round(elapsed, 1),
        "note": (
            "Параметры режима hf обучены только на обучающем разбиении. Тест не использовался "
            "ни для подбора весов, ни для порога."
        ),
    }
    weights_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    head_written = None
    head_payload = (report.head or {}).get("payload")
    if bundle.head.get("type") == "logreg" and head_payload:
        head_path = Path(args.head_out)
        head_path.parent.mkdir(parents=True, exist_ok=True)
        head_path.write_text(
            json.dumps(
                {**head_payload, "version": __version__, "seed": args.seed},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        head_written = str(head_path)

    summary = {
        "mode": "hf",
        "model": args.model,
        "n_train_pairs": len(pairs),
        "train_file": str(train_path),
        "folds": report.folds,
        "head_auc_folds": [round(fold.get("auc", float("nan")), 4) for fold in report.folds if fold.get("auc") is not None],
        "threshold": bundle.threshold,
        "target_fpr": bundle.target_fpr,
        "weights": bundle.weights,
        "head": (bundle.head or {}).get("type", "none"),
        "head_file": head_written,
        "validation": report.validation,
        "seconds": round(elapsed, 1),
        "hardware": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor() or platform.machine(),
        },
        "command": "python " + " ".join(sys.argv),
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"параметры режима hf: {weights_path}")
    if head_written:
        print(f"голова: {head_written}")
    print(f"отчёт: {out_json}")
    print(f"пар в обучении: {len(pairs)}, время: {elapsed:.0f} с, порог: {bundle.threshold:.4f}")
    print(f"веса: { {k: round(v, 3) for k, v in bundle.weights.items()} }")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
