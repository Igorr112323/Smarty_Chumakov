#!/usr/bin/env python3
"""Минимальный тест для диагностики проблемы с measure_aggregators.py"""
import sys
import json
from pathlib import Path

print("=" * 60)
print("Минимальный тест measure_aggregators")
print("=" * 60)

errors = []

# Тест 1: Импорт модулей
print("\n[1/6] Тест импорта модулей...")
try:
    from spanverify.dataset import read_pairs
    print("  ✓ read_pairs")
except Exception as e:
    print(f"  ❌ read_pairs: {e}")
    errors.append(f"read_pairs: {e}")

try:
    from spanverify.engine import Verifier, WeightsBundle
    print("  ✓ Verifier, WeightsBundle")
except Exception as e:
    print(f"  ❌ Verifier: {e}")
    errors.append(f"Verifier: {e}")

try:
    from spanverify.features import load_feature_cache, set_feature_cache, is_scored_token
    print("  ✓ load_feature_cache, set_feature_cache, is_scored_token")
except Exception as e:
    print(f"  ❌ features: {e}")
    errors.append(f"features: {e}")

try:
    from spanverify.calibration import IsotonicCalibrator, choose_threshold_report
    print("  ✓ IsotonicCalibrator, choose_threshold_report")
except Exception as e:
    print(f"  ❌ calibration: {e}")
    errors.append(f"calibration: {e}")

try:
    from spanverify.train import auc_score
    print("  ✓ auc_score")
except Exception as e:
    print(f"  ❌ auc_score: {e}")
    errors.append(f"auc_score: {e}")

# Тест 2: Загрузка данных
print("\n[2/6] Тест загрузки данных...")
try:
    pairs = read_pairs("data/corpus_a3/pairs.jsonl")
    print(f"  ✓ Загружено {len(pairs)} пар")
except Exception as e:
    print(f"  ❌ Загрузка данных: {e}")
    import traceback
    traceback.print_exc()
    errors.append(f"Загрузка данных: {e}")
    pairs = []

# Тест 3: Загрузка кеша
print("\n[3/6] Тест загрузки кеша...")
try:
    cache = load_feature_cache(["reports/hf-cache-v2"], counting=True, strict=True)
    set_feature_cache(cache)
    print(f"  ✓ Кеш: {len(cache)} ключей")
except Exception as e:
    print(f"  ❌ Загрузка кеша: {e}")
    import traceback
    traceback.print_exc()
    errors.append(f"Загрузка кеша: {e}")
    cache = {}

# Тест 4: Загрузка бандла
print("\n[4/6] Тест загрузки бандла...")
try:
    bundle = WeightsBundle.from_file(
        "reports/experiments/p0_span/run39060670054/clause_capped/experiment.json"
    )
    print(f"  ✓ Бандл загружен")
    print(f"    Model: {bundle.model}")
    print(f"    Threshold: {bundle.threshold}")
except Exception as e:
    print(f"  ❌ Загрузка бандла: {e}")
    import traceback
    traceback.print_exc()
    errors.append(f"Загрузка бандла: {e}")
    bundle = None

# Тест 5: Создание верификатора
print("\n[5/6] Тест создания верификатора...")
verifier = None
if bundle is not None:
    try:
        verifier = Verifier(mode="hf", weights=bundle)
        print(f"  ✓ Verifier создан")
    except Exception as e:
        print(f"  ❌ Создание верификатора: {e}")
        import traceback
        traceback.print_exc()
        errors.append(f"Создание верификатора: {e}")
else:
    print("  ⚠️  Пропущено (bundle не загружен)")

# Тест 6: Проверка одной пары
print("\n[6/6] Тест проверки одной пары...")
if verifier is not None and pairs:
    try:
        test_pair = pairs[0]
        answer = test_pair.get("answer", "")
        context = test_pair.get("context", "")
        print(f"  Тестовая пара: {test_pair.get('id', 'unknown')}")
        print(f"  Answer length: {len(answer)}")
        print(f"  Context length: {len(context)}")
        
        result = verifier.verify(answer, context, with_tokens=True)
        print(f"  ✓ Проверка выполнена")
        print(f"    Score: {result.score:.4f}")
        print(f"    Tokens: {len(result.tokens)}")
    except Exception as e:
        print(f"  ❌ Проверка пары: {e}")
        import traceback
        traceback.print_exc()
        errors.append(f"Проверка пары: {e}")
else:
    print("  ⚠️  Пропущено (verifier или pairs не загружены)")

print("\n" + "=" * 60)
if errors:
    print(f"❌ Обнаружено {len(errors)} ошибок:")
    for i, err in enumerate(errors, 1):
        print(f"  {i}. {err}")
    print("=" * 60)
    sys.exit(1)
else:
    print("✓ Все тесты пройдены!")
    print("=" * 60)
    sys.exit(0)
