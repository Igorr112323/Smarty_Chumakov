#!/usr/bin/env python3
"""Сборка замороженного P1-кандидата из опубликованных артефактов.

Загружает:
- baseline_bundle.json (HF token-head)
- p1_model.json (answer-head)
- calibrator_p1.json (isotonic calibrator)
- thresholds.json (train threshold)

Создаёт единый P1 bundle с полной точностью float-значений.
"""
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <source_dir>")
        print("  source_dir: путь к опубликованным артефактам")
        sys.exit(1)
    
    source_dir = Path(sys.argv[1])
    
    # Проверяем наличие всех файлов
    required_files = [
        "baseline_bundle.json",
        "p1_model.json",
        "calibrator_p1.json",
        "thresholds.json",
    ]
    
    for fname in required_files:
        fpath = source_dir / fname
        if not fpath.exists():
            print(f"ERROR: Required file not found: {fname}")
            sys.exit(1)
    
    # Загружаем исходные файлы
    print("Загрузка исходных артефактов...")
    
    baseline_bundle = json.loads((source_dir / "baseline_bundle.json").read_text())
    p1_model = json.loads((source_dir / "p1_model.json").read_text())
    calibrator_p1 = json.loads((source_dir / "calibrator_p1.json").read_text())
    thresholds = json.loads((source_dir / "thresholds.json").read_text())
    
    # Создаём P1 bundle
    p1_bundle = baseline_bundle.copy()
    
    # Устанавливаем P1 компоненты
    p1_bundle["answer_head"] = p1_model
    p1_bundle["isotonic"] = calibrator_p1
    p1_bundle["threshold"] = thresholds["p1"]  # Полная точность
    
    # Добавляем provenance
    p1_bundle["p1_provenance"] = {
        "source_measurement_run": "39116955685",
        "source_measurement_sha": "b87850781b73d8630832cb972037158d0ec88927",
        "p1_model_features": p1_model.get("n_features", 37),
        "p1_schema_version": p1_model.get("schema_version", "1.0"),
    }
    
    # Сохраняем без округления
    output_path = source_dir / "p1_candidate_bundle.json"
    output_path.write_text(json.dumps(p1_bundle, indent=2))
    
    print(f"✓ P1 кандидат собран: {output_path}")
    print(f"  - Threshold: {p1_bundle['threshold']:.15f}")
    print(f"  - Answer head features: {p1_model.get('n_features', 37)}")
    print(f"  - Schema version: {p1_model.get('schema_version', '1.0')}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
