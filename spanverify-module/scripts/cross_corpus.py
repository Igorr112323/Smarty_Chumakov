"""Кросс-корпусный тест: обучение на корпусе A, оценка на корпусе B (P0-2).

Скрипт отвечает на главный научный вопрос проекта: переносится ли качество
метода на корпус, который построен **другим генератором** (иные шаблоны, числа
словами, другие формулировки). Результат пишется в ``reports/cross_corpus.json``
и попадает в единый файл чисел ``reports/METRICS.json``.

Запуск::

    python scripts/cross_corpus.py --dataset data/demo_pairs.jsonl --out reports/cross_corpus.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spanverify.corpus_alt import generate_alt_pairs  # noqa: E402
from spanverify.dataset import read_pairs  # noqa: E402
from spanverify.engine import Verifier  # noqa: E402
from spanverify.train import train  # noqa: E402


def _metrics_tree(metrics: dict) -> dict:
    """Выбрать из метрик evaluate() то, что сравнивается между корпусами."""
    tokens = metrics["tokens"]
    spans = metrics["spans"]
    answers = metrics["answers"]
    return {
        "tokens": {
            "precision": round(tokens["precision"], 4),
            "recall": round(tokens["recall"], 4),
            "f1": round(tokens["f1"], 4),
            "fpr": round(tokens["fpr"], 4),
            "auc": round(tokens["auc"], 4) if tokens["auc"] == tokens["auc"] else None,
            "n": tokens["tp"] + tokens["fp"] + tokens["tn"] + tokens["fn"],
        },
        "spans": {
            "strict_f1": round(spans["f1"], 4),
            "coverage": round(spans["recall_containment"], 4),
            "soft_f1": round(spans["f1_expanded_labels"], 4),
            "width_ratio": round(spans["mean_width_ratio"], 2),
        },
        "answers": {
            "f1": round(answers["f1"], 4),
            "fpr": round(answers["fpr"], 4),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Кросс-корпусный тест SpanVerify")
    parser.add_argument("--dataset", default="data/demo_pairs.jsonl", help="корпус A (обучение)")
    parser.add_argument("--alt-pairs", type=int, default=240, help="размер корпуса B")
    parser.add_argument("--alt-seed", type=int, default=4242, help="seed корпуса B")
    parser.add_argument("--seed", type=int, default=42, help="seed обучения")
    parser.add_argument("--mode", default="demo", help="режим признаков (demo/hf)")
    parser.add_argument("--out", default="reports/cross_corpus.json", help="куда записать JSON")
    args = parser.parse_args()

    pairs_a = list(read_pairs(args.dataset))
    report = train(pairs_a, mode=args.mode, seed=args.seed, dataset_name=str(args.dataset))
    in_corpus = report.validation["end_to_end"]

    pairs_b = generate_alt_pairs(n_pairs=args.alt_pairs, seed=args.alt_seed)
    verifier = Verifier(mode=args.mode, weights=report.bundle)
    cross = verifier.evaluate(pairs_b)

    payload = {
        "seed": args.seed,
        "mode": args.mode,
        "corpus_a": {
            "name": str(args.dataset),
            "pairs": len(pairs_a),
            "split": report.stats.get("split", {}),
        },
        "corpus_b": {
            "name": "alt-generator 1.0 (числа словами, другие шаблоны)",
            "pairs": len(pairs_b),
            "seed": args.alt_seed,
        },
        "in_corpus": _metrics_tree(in_corpus),
        "cross_corpus": _metrics_tree(cross),
        "note": (
            "Корпуса A и B синтетические. Число «кросс-корпус» показывает перенос между "
            "генераторами; измерением на реальных регламентах оно не является."
        ),
    }
    delta = payload["in_corpus"]["tokens"]["f1"] - payload["cross_corpus"]["tokens"]["f1"]
    payload["token_f1_drop"] = round(delta, 4)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Внутри корпуса A (отложенная часть): token F1={payload['in_corpus']['tokens']['f1']:.3f}")
    print(f"Кросс-корпус (A → B): token F1={payload['cross_corpus']['tokens']['f1']:.3f}")
    print(f"Падение token F1: {delta:+.3f}")
    print(f"Записано: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
