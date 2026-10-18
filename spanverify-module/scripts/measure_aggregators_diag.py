#!/usr/bin/env python3
"""Диагностическая версия - только проверка аргументов и создание файла"""

import argparse
import sys
from pathlib import Path


def main():
    print("=" * 60)
    print("Диагностическая версия measure_aggregators")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="P0-2: Диагностика")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--mode", choices=["demo", "hf"], default="hf")
    parser.add_argument("--features-cache", required=True)
    parser.add_argument("--baseline-experiment", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-source-run-id")
    parser.add_argument("--measurement-run-id")
    parser.add_argument("--measurement-commit")

    args = parser.parse_args()

    print("\nПолучены аргументы:")
    print(f"  dataset: {args.dataset}")
    print(f"  splits: {args.splits}")
    print(f"  mode: {args.mode}")
    print(f"  features-cache: {args.features_cache}")
    print(f"  baseline-experiment: {args.baseline_experiment}")
    print(f"  out: {args.out}")
    print(f"  cache-source-run-id: {args.cache_source_run_id}")
    print(f"  measurement-run-id: {args.measurement_run_id}")
    print(f"  measurement-commit: {args.measurement_commit}")

    print("\nПроверка файлов...")
    print(f"  dataset exists: {Path(args.dataset).exists()}")
    print(f"  splits exists: {Path(args.splits).exists()}")
    print(f"  features-cache exists: {Path(args.features_cache).exists()}")
    print(f"  baseline-experiment exists: {Path(args.baseline_experiment).exists()}")

    print("\nСоздание выходной директории...")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ Директория создана: {out_dir.absolute()}")

    print("\nСоздание тестового файла...")
    test_file = out_dir / "diag_test.txt"
    test_file.write_text("Диагностический тест пройден успешно\n", encoding="utf-8")
    print(f"  ✓ Файл создан: {test_file}")

    print("\n" + "=" * 60)
    print("✓ Диагностика завершена успешно!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
