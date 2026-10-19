#!/usr/bin/env python3
"""Локальный тест сборки P1 кандидата без HF."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "spanverify-module"))

def test_build_p1_candidate():
    """Проверить что P1 кандидат собирается корректно."""
    
    source_dir = Path("spanverify-module/reports/experiments/p1_raw/run39116955685")
    
    # Проверяем наличие всех файлов
    required_files = [
        "baseline_bundle.json",
        "p1_model.json",
        "calibrator_p1.json",
        "thresholds.json",
    ]
    
    print("Проверка наличия файлов...")
    for fname in required_files:
        fpath = source_dir / fname
        if not fpath.exists():
            print(f"✗ {fname} NOT FOUND")
            return False
        print(f"✓ {fname} found")
    
    # Загружаем файлы
    print("\nЗагрузка файлов...")
    baseline_bundle = json.loads((source_dir / "baseline_bundle.json").read_text())
    p1_model = json.loads((source_dir / "p1_model.json").read_text())
    calibrator_p1 = json.loads((source_dir / "calibrator_p1.json").read_text())
    thresholds = json.loads((source_dir / "thresholds.json").read_text())
    
    print(f"✓ baseline_bundle loaded")
    print(f"✓ p1_model loaded (n_features={p1_model.get('n_features')})")
    print(f"✓ calibrator_p1 loaded")
    print(f"✓ thresholds loaded (p1={thresholds['p1']})")
    
    # Проверяем импорт spanverify
    print("\nПроверка импорта spanverify...")
    try:
        from spanverify.engine import Verifier, WeightsBundle
        from spanverify.answer_head import AnswerHeadModel
        print("✓ spanverify.engine imported")
        print("✓ spanverify.answer_head imported")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # Пробуем создать P1 bundle
    print("\nСборка P1 кандидата...")
    try:
        p1_bundle = baseline_bundle.copy()
        p1_bundle["answer_head"] = p1_model
        p1_bundle["isotonic"] = calibrator_p1
        p1_bundle["threshold"] = thresholds["p1"]
        
        print(f"✓ P1 bundle assembled")
        print(f"  - Threshold: {p1_bundle['threshold']:.15f}")
        print(f"  - Answer head features: {p1_model.get('n_features')}")
        
        # Пробуем загрузить bundle
        bundle = WeightsBundle.from_dict(p1_bundle)
        print(f"✓ WeightsBundle loaded")
        print(f"  - Threshold: {bundle.threshold:.15f}")
        print(f"  - Answer head: {bundle.answer_head is not None}")
        
        # Пробуем создать AnswerHeadModel
        ah_model = AnswerHeadModel.from_dict(p1_model)
        print(f"✓ AnswerHeadModel loaded")
        print(f"  - n_features: {ah_model.n_features}")
        print(f"  - weights length: {len(ah_model.weights)}")
        
    except Exception as e:
        print(f"✗ Build failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ All checks passed!")
    return True

if __name__ == "__main__":
    success = test_build_p1_candidate()
    sys.exit(0 if success else 1)
