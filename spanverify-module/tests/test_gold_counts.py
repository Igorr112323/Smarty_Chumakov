"""Тест для проверки правильности gold counts на реальных данных (задача 2.1)."""
import json
from pathlib import Path
from spanverify.oof import answer_label


def test_gold_counts():
    """Проверить правильность gold counts на реальных train/dev данных.
    
    Согласно задаче 2.1:
    - Train: 467 положительных, 375 отрицательных
    - Dev: 96 положительных, 85 отрицательных
    """
    # Путь к данным
    data_dir = Path(__file__).parent.parent / "data" / "corpus_a3" / "splits"
    
    # Проверяем train
    train_path = data_dir / "train.jsonl"
    if train_path.exists():
        train_pairs = [json.loads(line) for line in train_path.read_text().splitlines() if line.strip()]
        train_positive = sum(1 for p in train_pairs if answer_label(p) == 1)
        train_negative = len(train_pairs) - train_positive
        
        print(f"Train: {len(train_pairs)} pairs, {train_positive} positive, {train_negative} negative")
        
        # Проверяем ожидаемые значения
        assert train_positive == 467, f"Expected 467 positive in train, got {train_positive}"
        assert train_negative == 375, f"Expected 375 negative in train, got {train_negative}"
    
    # Проверяем dev
    dev_path = data_dir / "dev.jsonl"
    if dev_path.exists():
        dev_pairs = [json.loads(line) for line in dev_path.read_text().splitlines() if line.strip()]
        dev_positive = sum(1 for p in dev_pairs if answer_label(p) == 1)
        dev_negative = len(dev_pairs) - dev_positive
        
        print(f"Dev: {len(dev_pairs)} pairs, {dev_positive} positive, {dev_negative} negative")
        
        # Проверяем ожидаемые значения
        assert dev_positive == 96, f"Expected 96 positive in dev, got {dev_positive}"
        assert dev_negative == 85, f"Expected 85 negative in dev, got {dev_negative}"


def test_specific_pairs():
    """Проверить конкретные пары corpus-a3-00473 и corpus-a3-00614.
    
    Согласно задаче 2.1, эти пары имеют односимвольные числа в размеченных
    фрагментах, которые не проходят scored_token фильтр, но должны считаться
    положительными по исходной разметке.
    """
    data_dir = Path(__file__).parent.parent / "data" / "corpus_a3" / "splits"
    train_path = data_dir / "train.jsonl"
    
    if train_path.exists():
        train_pairs = [json.loads(line) for line in train_path.read_text().splitlines() if line.strip()]
        
        # Ищем конкретные пары
        pair_473 = next((p for p in train_pairs if p.get("id") == "corpus-a3-00473"), None)
        pair_614 = next((p for p in train_pairs if p.get("id") == "corpus-a3-00614"), None)
        
        if pair_473:
            print(f"Pair corpus-a3-00473: labels={pair_473.get('labels', [])}")
            assert answer_label(pair_473) == 1, "Pair corpus-a3-00473 should be positive"
        
        if pair_614:
            print(f"Pair corpus-a3-00614: labels={pair_614.get('labels', [])}")
            assert answer_label(pair_614) == 1, "Pair corpus-a3-00614 should be positive"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
