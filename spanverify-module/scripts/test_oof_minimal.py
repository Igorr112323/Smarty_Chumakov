#!/usr/bin/env python3
"""Минимальный тест OOF-логики без HF-модели."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spanverify.dataset import read_pairs
from spanverify.engine import Verifier, WeightsBundle
from spanverify.features import load_feature_cache, set_feature_cache
from spanverify.oof import _get_risks_with_head, _train_model_on_subset
from spanverify.train import collect_samples


def main():
    print("=" * 60)
    print("Минимальный тест OOF-логики")
    print("=" * 60)

    # Загружаем данные
    print("\n[1/5] Загрузка данных...")
    train_pairs = list(read_pairs("data/corpus_a3/splits/train.jsonl"))[:50]  # Только 50 пар для быстрого теста
    print(f"✓ Train: {len(train_pairs)} pairs")

    # Загружаем бандл
    print("\n[2/5] Загрузка бандла...")
    import json

    exp = json.loads(
        Path("reports/experiments/p0_span/run39060670054/clause_capped/experiment.json").read_text(encoding="utf-8")
    )
    bundle = WeightsBundle.from_dict(exp["bundle"])
    print("✓ Бандл загружен")

    # Загружаем кеш
    print("\n[3/5] Загрузка кеша...")
    cache = load_feature_cache(["reports/hf-cache-v2"], counting=True, strict=True)
    set_feature_cache(cache)
    print(f"✓ Кеш: {len(cache)} ключей")

    # Создаём верификатор
    print("\n[4/5] Создание верификатора...")
    verifier = Verifier(
        mode="hf",
        model_name=exp["model"],
        weights=bundle,
        features_cache=cache,
        span_expansion="clause_capped",
    )
    print("✓ Verifier создан")

    # Собираем samples
    print("\n[5/5] Сбор samples и обучение модели...")
    start = time.time()
    samples, stats = collect_samples(verifier, train_pairs)
    print(f"✓ Samples собраны: {len(samples)} tokens за {time.time() - start:.1f}s")

    # Обучаем модель
    print("\n  Обучение модели...")
    start = time.time()
    weights, head = _train_model_on_subset(samples, entropy_quantile=verifier.entropy_quantile)
    print(f"✓ Модель обучена за {time.time() - start:.1f}s")
    print(f"  Weights: {weights}")

    # Получаем риски
    print("\n  Вычисление рисков...")
    start = time.time()
    risks = _get_risks_with_head(samples, weights, head, entropy_quantile=verifier.entropy_quantile)
    print(f"✓ Риски вычислены: {len(risks)} значений за {time.time() - start:.1f}s")
    print(f"  Risk range: [{min(risks):.4f}, {max(risks):.4f}]")

    print("\n" + "=" * 60)
    print("✓ Минимальный тест завершен успешно!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        import traceback

        print("\n" + "=" * 60, file=sys.stderr)
        print("❌ КРИТИЧЕСКАЯ ОШИБКА", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(f"Тип ошибки: {type(e).__name__}", file=sys.stderr)
        print(f"Сообщение: {e}", file=sys.stderr)
        print("\nПолный traceback:", file=sys.stderr)
        traceback.print_exc()
        print("=" * 60, file=sys.stderr)
        sys.exit(1)
