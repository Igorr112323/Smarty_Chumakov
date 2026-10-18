#!/usr/bin/env python3
"""P2 Experiment: минимальный эксперимент token-head.

Упрощённая версия для проверки интеграции P2.
Полный эксперимент с 5 doc-folds требует отдельной реализации.

Usage:
    python scripts/p2_experiment.py \
        --train data/corpus_a3/splits/train.jsonl \
        --dev data/corpus_a3/splits/dev.jsonl \
        --p1-bundle reports/experiments/p1_public_replay/run39144872719/p1_candidate_bundle.json \
        --output reports/experiments/p2_experiment/
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import hashlib

import numpy as np

# Добавляем родительскую директорию в path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spanverify.token_head import TokenGBDTModel, DecisionTree
from spanverify.engine import Verifier, WeightsBundle
from spanverify.core import tokenize_with_offsets


def load_pairs(path: Path) -> list[dict]:
    """Загрузить пары из JSONL."""
    pairs = []
    with open(path) as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="P2 Experiment (Minimal)")
    parser.add_argument("--train", type=Path, required=True, help="Train JSONL")
    parser.add_argument("--dev", type=Path, required=True, help="Dev JSONL")
    parser.add_argument("--p1-bundle", type=Path, required=True, help="P1 bundle JSON")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    args = parser.parse_args()
    
    # Создаём output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Загружаем данные
    print(f"Загрузка train: {args.train}")
    train_pairs = load_pairs(args.train)
    print(f"  {len(train_pairs)} пар")
    
    print(f"Загрузка dev: {args.dev}")
    dev_pairs = load_pairs(args.dev)
    print(f"  {len(dev_pairs)} пар")
    
    # Загружаем P1 bundle
    print(f"Загрузка P1 bundle: {args.p1_bundle}")
    p1_bundle = WeightsBundle.load(args.p1_bundle)
    # Используем mode="demo" для быстрого теста без HF backend
    p1_verifier = Verifier(weights=p1_bundle, mode="demo")
    
    print("✅ P1 bundle загружен")
    
    # Создаём placeholder P2 модель (простое дерево)
    print("Создание placeholder P2 модели...")
    tree = DecisionTree(
        threshold=0.5,
        feature_index=0,
        left=DecisionTree(threshold=0.0, feature_index=0, value=0.2),
        right=DecisionTree(threshold=0.0, feature_index=0, value=0.8),
    )
    p2_model = TokenGBDTModel(
        trees=[tree],
        learning_rate=0.1,
        initial_prediction=0.0,
    )
    p2_model_dict = p2_model.to_dict()
    
    # Создаём bundle с P1 + P2
    p2_bundle = WeightsBundle(
        weights=p1_bundle.weights,
        threshold=p1_bundle.threshold,
        mode="demo",
        answer_head=p1_bundle.answer_head,
        token_head=p2_model_dict,
        token_threshold=0.5,
    )
    p2_verifier = Verifier(weights=p2_bundle, mode="demo")
    
    print("✅ P2 bundle создан")
    
    # Проверяем P1 invariance на dev
    print("Проверка P1 invariance на dev...")
    p1_invariant = True
    drift_count = 0
    
    for i, pair in enumerate(dev_pairs[:10]):  # Только первые 10 для быстрой проверки
        answer = pair["answer"]
        context = pair.get("context", "")
        
        # P1 prediction
        result_p1 = p1_verifier.verify(answer, context)
        
        # P2 prediction (с P2 token-head)
        result_p2 = p2_verifier.verify(answer, context)
        
        # Проверяем что P1 score не изменился
        score_diff = abs(result_p1.stats["raw_score_full"] - result_p2.stats["raw_score_full"])
        if score_diff > 1e-12:
            print(f"  ❌ P1 score drift: pair {i}, diff={score_diff:.2e}")
            p1_invariant = False
            drift_count += 1
        
        if result_p1.is_hallucination != result_p2.is_hallucination:
            print(f"  ❌ P1 prediction drift: pair {i}")
            p1_invariant = False
            drift_count += 1
    
    if p1_invariant:
        print("  ✅ P1 invariance: OK (0 drifts)")
    else:
        print(f"  ❌ P1 invariance: FAIL ({drift_count} drifts)")
    
    # Проверяем что P2 влияет на token risk
    print("Проверка P2 influence на token risk...")
    pair = dev_pairs[0]
    answer = pair["answer"]
    context = pair.get("context", "")
    
    result_p1 = p1_verifier.verify(answer, context, with_tokens=True)
    result_p2 = p2_verifier.verify(answer, context, with_tokens=True)
    
    # Извлекаем token risks
    p1_risks = [t.get("raw_risk", t.get("risk", 0.0)) for t in result_p1.tokens]
    p2_risks = [t.get("raw_risk", t.get("risk", 0.0)) for t in result_p2.tokens]
    
    risks_changed = p1_risks != p2_risks
    if risks_changed:
        print("  ✅ P2 влияет на token risk")
        print(f"    P1 risks (first 5): {p1_risks[:5]}")
        print(f"    P2 risks (first 5): {p2_risks[:5]}")
    else:
        print("  ❌ P2 НЕ влияет на token risk")
    
    # Сохраняем P2 модель
    p2_model_path = args.output / "p2_token_head.json"
    p2_model.save(p2_model_path)
    print(f"✅ P2 модель сохранена: {p2_model_path}")
    
    # Сохраняем отчёт
    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "train_pairs": len(train_pairs),
        "dev_pairs": len(dev_pairs),
        "p1_invariant": p1_invariant,
        "drift_count": drift_count,
        "p2_influences_token_risk": risks_changed,
        "status": "✅ P2 интеграция работает" if (p1_invariant and risks_changed) else "❌ P2 интеграция сломана",
    }
    
    report_path = args.output / "experiment_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"✅ Отчёт сохранён: {report_path}")
    
    # Итог
    print("\n" + "=" * 80)
    print("ИТОГ P2 МИНИМАЛЬНОГО ЭКСПЕРИМЕНТА")
    print("=" * 80)
    print(f"P1 Invariance: {'✅ OK' if p1_invariant else '❌ FAIL'}")
    print(f"P2 Influences Token Risk: {'✅ OK' if risks_changed else '❌ FAIL'}")
    print(f"Статус: {report['status']}")
    print("=" * 80)
    
    return 0 if (p1_invariant and risks_changed) else 1


if __name__ == "__main__":
    sys.exit(main())
