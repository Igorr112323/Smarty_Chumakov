#!/usr/bin/env python3
"""Упрощённая версия measure_aggregators - обрабатывает только несколько пар"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Импорты
from spanverify.dataset import read_pairs
from spanverify.engine import Verifier, WeightsBundle
from spanverify.features import load_feature_cache, set_feature_cache


# Определим локальные версии функций, чтобы избежать проблем с импортами
def is_scored_token(text: str) -> bool:
    """Проверить, является ли токен scored (не пунктуация и не служебное слово)"""
    import re

    text = text.strip()
    if not text:
        return False
    # Простая проверка: токен должен содержать хотя бы одну букву или цифру
    return bool(re.search(r"[а-яА-Яa-zA-Z0-9]", text))


def auc_score(labels: list[int], scores: list[float]) -> float:
    """Вычислить AUC-ROC"""
    if not labels or not scores:
        return 0.0

    # Простая реализация AUC
    pairs = list(zip(labels, scores, strict=False))
    pairs.sort(key=lambda x: x[1], reverse=True)

    tp = 0
    fp = 0
    auc = 0.0
    total_pos = sum(labels)
    total_neg = len(labels) - total_pos

    if total_pos == 0 or total_neg == 0:
        return 0.0

    for label, _score in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc += tp

    return auc / (total_pos * total_neg)


def load_hf_bundle(experiment_path: Path) -> tuple[WeightsBundle, dict]:
    """Загрузить HF-бандл из experiment.json"""
    data = json.loads(experiment_path.read_text(encoding="utf-8"))
    bundle_data = data.get("weights_bundle", {})

    bundle = WeightsBundle(
        model=bundle_data["model"],
        threshold=bundle_data["threshold"],
        span_z=bundle_data.get("span_z", 1.5),
        span_floor=bundle_data.get("span_floor", 0.3),
        span_cap=bundle_data.get("span_cap", 0.7),
        span_expansion=bundle_data.get("span_expansion", "clause_capped"),
        token_heads=bundle_data.get("token_heads"),
    )

    metadata = {
        "source_run_id": data.get("source_run_id"),
        "source_commit": data.get("source_commit"),
        "model": bundle_data["model"],
    }

    return bundle, metadata


def aggregate_current(scores: list[float]) -> float:
    """Текущий агрегатор: среднее всех токенов"""
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def main():
    parser = argparse.ArgumentParser(description="P0-2: Упрощённое сравнение агрегаторов")
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
    start_time = time.time()

    print("=" * 60)
    print("P0-2: Упрощённое сравнение агрегаторов")
    print("=" * 60)

    # 1. Загрузка бандла
    print("\n[1/5] Загрузка HF-бандла...")
    bundle, metadata = load_hf_bundle(Path(args.baseline_experiment))
    print("✓ Бандл загружен")
    print(f"  Model: {metadata['model']}")

    # 2. Загрузка данных
    print("\n[2/5] Загрузка данных...")
    splits_dir = Path(args.splits)
    train_pairs = list(read_pairs(splits_dir / "train.jsonl"))
    dev_pairs = list(read_pairs(splits_dir / "dev.jsonl"))
    print(f"✓ Train: {len(train_pairs)} pairs")
    print(f"✓ Dev: {len(dev_pairs)} pairs")

    # Ограничим до 10 пар для быстрого теста
    train_pairs = train_pairs[:10]
    dev_pairs = dev_pairs[:10]
    print(f"  (ограничено до {len(train_pairs)} пар для теста)")

    # 3. Загрузка кеша
    print("\n[3/5] Загрузка кеша признаков...")
    cache = load_feature_cache([args.features_cache], counting=True, strict=True)
    set_feature_cache(cache)
    print(f"✓ Кеш: {len(cache)} ключей")

    # 4. Создание верификатора
    print("\n[4/5] Создание верификатора...")
    verifier = Verifier(mode="hf", weights=bundle)
    print("✓ Verifier создан")

    # 5. Простая обработка пар
    print("\n[5/5] Обработка пар с агрегатором 'current'...")
    scores = []
    labels = []

    for i, pair in enumerate(dev_pairs):
        answer = pair.get("answer", "")
        context = pair.get("context", "")
        label = 1 if pair.get("labels") else 0

        result = verifier.verify(answer, context, with_tokens=True)

        # Извлечь scores для scored токенов
        token_scores = []
        for token_info in result.tokens:
            if is_scored_token(token_info.get("text", "")):
                token_scores.append(token_info.get("risk", 0.0))

        # Применить агрегатор
        raw_score = aggregate_current(token_scores)
        scores.append(raw_score)
        labels.append(label)

        if i % 5 == 0:
            print(f"  Обработано {i+1}/{len(dev_pairs)} пар")

    print(f"✓ Обработано {len(dev_pairs)} пар")

    # Вычислить метрики
    print("\nВычисление метрик...")
    raw_auc = auc_score(labels, scores)
    print(f"  Raw AUC: {raw_auc:.4f}")

    # Создание результатов
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "measurement_run_id": args.measurement_run_id,
        "measurement_commit": args.measurement_commit,
        "cache_source_run_id": args.cache_source_run_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "aggregators": {
            "current": {
                "name": "current",
                "raw_auc": raw_auc,
                "num_pairs": len(dev_pairs),
                "status": "success",
            }
        },
        "best_aggregator": "current",
        "summary": {
            "best_aggregator": "current",
            "best_raw_auc": raw_auc,
            "num_aggregators_tested": 1,
        },
    }

    # Сохранить результаты
    results_path = out_dir / "aggregator_comparison.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ Результаты сохранены: {results_path}")

    # Создать markdown отчёт
    md_path = out_dir / "p0_answer_comparison.md"
    md_content = f"""# P0-2: Упрощённое сравнение агрегаторов

**Дата:** {results['timestamp']}
**Measurement Run ID:** {results['measurement_run_id']}
**Measurement Commit:** {results['measurement_commit']}

## Результаты

| Агрегатор | Raw AUC | Статус |
|-----------|---------|--------|
| current | {raw_auc:.4f} | ✓ |

**Лучший агрегатор:** current

---

**Статус:** ✅ Упрощённое сравнение завершено (тестовый режим)
"""
    md_path.write_text(md_content, encoding="utf-8")
    print(f"✓ Markdown отчёт: {md_path}")

    duration = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"✓ Завершено за {duration:.1f}s")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        import traceback

        print("\n" + "=" * 60, file=sys.stderr)
        print("❌ КРИТИЧЕСКАЯ ОШИБКА В СКРИПТЕ", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(f"Тип ошибки: {type(e).__name__}", file=sys.stderr)
        print(f"Сообщение: {e}", file=sys.stderr)
        print("\nПолный traceback:", file=sys.stderr)
        traceback.print_exc()
        print("=" * 60, file=sys.stderr)
        sys.exit(1)
