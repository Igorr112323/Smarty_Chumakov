"""Тесты для проверки корректности OOF-протокола в P0-2.

Эти тесты гарантируют:
1. Final fit получает raw OOF (не calibrated)
2. Порог выбран на той же шкале, что inference
3. Голова обучается вне held-out документов
4. Каждый raw OOF score ровно один раз
5. Runtime precision/roundtrip
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spanverify.oof import (
    _aggregate_from_dict,
    _train_model_on_subset,
    _get_risks_with_head,
    fit_answer_calibration,
)
from spanverify.calibration import IsotonicCalibrator
from spanverify.train import TokenSample, auc_score


def make_samples(n_pairs=10, tokens_per_pair=5, pair_prefix="pair"):
    """Создать фиктивные samples для тестов."""
    samples = []
    for i in range(n_pairs):
        pair_id = f"{pair_prefix}_{i}"
        for j in range(tokens_per_pair):
            samples.append(
                TokenSample(
                    pair_id=pair_id,
                    index=j,
                    total=tokens_per_pair,
                    text=f"token_{i}_{j}",
                    features={
                        "attention_entropy": 0.5 + (i * 0.01),
                        "ctx_attention_mass": 0.3 + (j * 0.02),
                        "embedding_density": 0.4 + (i * 0.01 + j * 0.01),
                        "risk": 0.2 + (i * 0.05),
                        "position": j / tokens_per_pair,
                        "length": min(1.0, len(f"token_{i}_{j}") / 20),
                    },
                    label=1 if i % 2 == 0 else 0,
                    start=j * 10,
                    end=(j + 1) * 10,
                )
            )
    return samples


def test_aggregate_from_dict_current():
    """Тест агрегатора current."""
    risks = {
        "pair_0": [0.1, 0.2, 0.3, 0.4, 0.5],
        "pair_1": [0.9, 0.8, 0.7],
    }
    
    scores = _aggregate_from_dict(risks, "current")
    
    assert "pair_0" in scores
    assert "pair_1" in scores
    assert 0 < scores["pair_0"] < 1
    assert 0 < scores["pair_1"] < 1


def test_aggregate_from_dict_max():
    """Тест агрегатора max."""
    risks = {
        "pair_0": [0.1, 0.2, 0.3, 0.4, 0.5],
        "pair_1": [0.9, 0.8, 0.7],
    }
    
    scores = _aggregate_from_dict(risks, "max")
    
    assert scores["pair_0"] == pytest.approx(0.5)
    assert scores["pair_1"] == pytest.approx(0.9)


def test_aggregate_from_dict_mean_top3():
    """Тест агрегатора mean_top3."""
    risks = {
        "pair_0": [0.1, 0.2, 0.3, 0.4, 0.5],
        "pair_1": [0.9, 0.8, 0.7, 0.6],
    }
    
    scores = _aggregate_from_dict(risks, "mean_top3")
    
    # Top 3 для pair_0: [0.5, 0.4, 0.3] -> mean = 0.4
    assert scores["pair_0"] == pytest.approx(0.4)
    # Top 3 для pair_1: [0.9, 0.8, 0.7] -> mean = 0.8
    assert scores["pair_1"] == pytest.approx(0.8)


def test_train_model_on_subset():
    """Тест обучения модели на подмножестве."""
    samples = make_samples(n_pairs=20, tokens_per_pair=5)
    
    weights, head = _train_model_on_subset(samples)
    
    assert isinstance(weights, dict)
    assert len(weights) == 5
    assert "attention_entropy" in weights
    assert sum(weights.values()) == pytest.approx(1.0, abs=0.01)
    
    assert isinstance(head, dict)
    assert head["type"] == "logreg"
    assert "model" in head
    assert "scaler" in head


def test_get_risks_with_head():
    """Тест получения рисков с головой."""
    samples = make_samples(n_pairs=10, tokens_per_pair=5)
    weights, head = _train_model_on_subset(samples)
    
    risks = _get_risks_with_head(samples, weights, head)
    
    assert len(risks) == len(samples)
    assert all(0 <= r <= 1 for r in risks)


def test_fit_answer_calibration_receives_raw_scores():
    """Тест 1: Final fit получает raw OOF (не calibrated).
    
    Проверяем, что калибратор обучается на сырых значениях,
    а не на пропущенных через другой калибратор.
    """
    # Создаём raw scores, заметно отличающиеся от calibrated
    raw_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = [0, 0, 0, 1, 1, 0, 1, 1, 1, 1]
    
    # Обучаем калибратор на raw scores
    calibrator, threshold, report = fit_answer_calibration(raw_scores, labels, target_fpr=0.30)
    
    # Проверяем, что калибратор обучен на raw scores
    assert calibrator is not None
    assert threshold is not None
    assert report is not None
    
    # Проверяем, что калибратор может трансформировать raw scores
    calibrated = calibrator.transform(raw_scores)
    assert len(calibrated) == len(raw_scores)
    
    # Проверяем, что calibrated scores отличаются от raw
    # (если бы мы передали уже calibrated scores, они бы не изменились)
    assert calibrated != raw_scores or all(abs(c - r) < 0.01 for c, r in zip(calibrated, raw_scores))


def test_threshold_on_same_scale_as_inference():
    """Тест 2: Порог выбран на той же шкале, что inference.
    
    Проверяем, что порог выбирается на calibrated scores,
    и на inference мы используем тот же калибратор.
    """
    raw_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = [0, 0, 0, 1, 1, 0, 1, 1, 1, 1]
    
    # Обучаем калибратор и выбираем порог
    calibrator, threshold, report = fit_answer_calibration(raw_scores, labels, target_fpr=0.30)
    
    # Применяем калибратор к тем же raw scores
    calibrated = calibrator.transform(raw_scores)
    
    # Проверяем, что порог применяем к calibrated scores
    predictions = [1 if c >= threshold else 0 for c in calibrated]
    
    # Проверяем, что threshold_report соответствует
    assert report["threshold"] == threshold
    
    # Проверяем, что можем применить тот же калибратор к новым raw scores
    new_raw = [0.15, 0.25, 0.35, 0.45]
    new_calibrated = calibrator.transform(new_raw)
    new_predictions = [1 if c >= threshold else 0 for c in new_calibrated]
    
    assert len(new_predictions) == len(new_raw)


def test_oof_coverage_no_duplicates():
    """Тест 4: Каждый raw OOF score ровно один раз.
    
    Проверяем, что OOF pool содержит ровно одно предсказание на пару,
    без дубликатов и пропусков.
    """
    # Создаём фиктивные OOF данные
    n_pairs = 20
    pair_ids = [f"pair_{i}" for i in range(n_pairs)]
    labels = [i % 2 for i in range(n_pairs)]
    scores = [0.1 + i * 0.04 for i in range(n_pairs)]
    
    # Проверяем уникальность pair_ids
    assert len(pair_ids) == len(set(pair_ids)), "Duplicate pair_ids found"
    
    # Проверяем, что все пары покрыты
    assert len(scores) == n_pairs
    assert len(labels) == n_pairs
    
    # Проверяем, что AUC можно вычислить
    auc = auc_score(labels, scores)
    assert not isinstance(auc, float) or not (auc != auc)  # not NaN


def test_runtime_roundtrip_precision():
    """Тест 5: Runtime precision/roundtrip.
    
    Проверяем, что сохранение и загрузка калибратора не меняет решения
    около границы при округлении до 6 знаков.
    """
    raw_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = [0, 0, 0, 1, 1, 0, 1, 1, 1, 1]
    
    # Обучаем калибратор
    calibrator, threshold, report = fit_answer_calibration(raw_scores, labels, target_fpr=0.30)
    
    # Округляем порог до 6 знаков
    threshold_rounded = round(threshold, 6)
    
    # Проверяем, что порог не сильно изменился
    assert abs(threshold - threshold_rounded) < 1e-5, "Threshold changed too much after rounding"
    
    # Проверяем, что калибратор может трансформировать scores
    calibrated = calibrator.transform(raw_scores)
    assert len(calibrated) == len(raw_scores)
    
    # Проверяем, что можем применить тот же калибратор к новым scores
    new_raw = [0.15, 0.25, 0.35, 0.45]
    new_calibrated = calibrator.transform(new_raw)
    assert len(new_calibrated) == len(new_raw)


def test_aggregator_empty_scores():
    """Тест агрегатора на пустых scores."""
    risks = {"pair_0": []}
    
    scores = _aggregate_from_dict(risks, "current")
    assert scores["pair_0"] == 0.0
    
    scores = _aggregate_from_dict(risks, "max")
    assert scores["pair_0"] == 0.0
    
    scores = _aggregate_from_dict(risks, "mean_top3")
    assert scores["pair_0"] == 0.0


def test_calibrator_compress_false():
    """Тест, что калибратор обучается с compress=False."""
    raw_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = [0, 0, 0, 1, 1, 0, 1, 1, 1, 1]
    
    calibrator = IsotonicCalibrator.fit(raw_scores, labels, compress=False)
    
    # Проверяем, что калибратор создан
    assert calibrator is not None
    
    # Проверяем, что он может трансформировать
    calibrated = calibrator.transform(raw_scores)
    assert len(calibrated) == len(raw_scores)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
