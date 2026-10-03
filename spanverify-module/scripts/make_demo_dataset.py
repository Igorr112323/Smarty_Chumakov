"""Сгенерировать демонстрационный корпус пар «контекст — ответ».

    python scripts/make_demo_dataset.py --pairs 240 --seed 1312 --out data/demo_pairs.jsonl

Корпус синтетический: он проверяет работоспособность конвейера и метрики
согласованности, но не заменяет размеченные реальные документы.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spanverify.dataset import corpus_statistics, generate_pairs, write_pairs  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Собрать корпус и напечатать его статистику."""
    parser = argparse.ArgumentParser(description="Генерация демонстрационного корпуса")
    parser.add_argument("--pairs", type=int, default=240)
    parser.add_argument("--seed", type=int, default=1312)
    parser.add_argument("--rate", type=float, default=0.5, help="доля недостоверных ответов")
    parser.add_argument("--out", default="data/demo_pairs.jsonl")
    args = parser.parse_args(argv)

    pairs = generate_pairs(args.pairs, seed=args.seed, hallucination_rate=args.rate)
    path = write_pairs(pairs, args.out)
    print(f"Корпус записан: {path}")
    for key, value in corpus_statistics(pairs).items():
        print(f"  {key}: {value}")
    print("Помните: это синтетика — числа описывают конвейер, а не реальные документы.")
    return 0


if __name__ == "__main__":  # pragma: no cover - утилита
    raise SystemExit(main())
