"""Тесты калибровки: PAVA, порог, метрики, кросс-валидация."""

from __future__ import annotations

import math
import random

import pytest

from spanverify.calibration import (
    IsotonicCalibrator,
    choose_threshold,
    cross_validate,
    metrics_at,
    pava,
)


def test_pava_output_is_non_decreasing():
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    labels = [1, 0, 0, 1, 0, 1, 1, 1]
    _, values = pava(scores, labels)
    assert all(b >= a for a, b in zip(values, values[1:]))


def test_pava_with_separable_data_gives_extremes():
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    _, values = pava(scores, labels)
    assert values[0] == 0.0
    assert values[-1] == 1.0


def test_pava_rejects_length_mismatch():
    with pytest.raises(ValueError):
        pava([0.1, 0.2], [1])


def test_pava_empty_input():
    assert pava([], []) == ([], [])


def test_isotonic_calibrator_is_monotone_and_bounded():
    rng = random.Random(7)
    scores = [rng.random() for _ in range(400)]
    labels = [1 if s > 0.5 else 0 for s in scores]
    calibrator = IsotonicCalibrator.fit(scores, labels)
    grid = [i / 100 for i in range(101)]
    probs = calibrator.transform(grid)
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert all(b >= a - 1e-9 for a, b in zip(probs, probs[1:]))


def test_calibrator_roundtrip_through_dict():
    calibrator = IsotonicCalibrator.fit([0.1, 0.5, 0.9], [0, 0, 1], source="test")
    restored = IsotonicCalibrator.from_dict(calibrator.to_dict())
    assert restored.transform([0.05, 0.5, 0.99]) == calibrator.transform([0.05, 0.5, 0.99])
    assert restored.meta["source"] == "test"


def test_calibrator_save_and_load(tmp_path):
    calibrator = IsotonicCalibrator.fit([0.2, 0.4, 0.6, 0.8], [0, 0, 1, 1])
    path = tmp_path / "nested" / "calibration.json"
    calibrator.save(path)
    assert path.is_file()
    assert IsotonicCalibrator.load(path).transform([0.7]) == calibrator.transform([0.7])


def test_calibrator_load_missing_file_returns_none(tmp_path):
    assert IsotonicCalibrator.load(tmp_path / "нет.json") is None


def test_calibrator_compresses_pava_output():
    """Сжатие сохраняет монотонность, границы и не меняет ответ грубо."""
    rng = random.Random(5)
    scores = [min(1.0, max(0.0, rng.gauss(0.5, 0.2))) for _ in range(5000)]
    labels = [1 if s + rng.gauss(0, 0.2) > 0.5 else 0 for s in scores]
    calibrator = IsotonicCalibrator.fit(scores, labels, tolerance=0.01, max_points=200)
    assert len(calibrator.thresholds) <= 202
    assert all(b >= a - 1e-9 for a, b in zip(calibrator.values, calibrator.values[1:]))
    assert calibrator.transform_one(0.0) <= calibrator.transform_one(1.0)


def test_calibrator_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        IsotonicCalibrator(thresholds=[0.1, 0.2], values=[0.5])


def test_metrics_at_hand_computed():
    probs = [0.9, 0.8, 0.4, 0.1]
    labels = [1, 0, 1, 0]
    metrics = metrics_at(probs, labels, threshold=0.5)
    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]) == (1, 1, 1, 1)
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert math.isclose(metrics["fpr"], 0.5)
    assert math.isclose(metrics["hdr"], 0.5)


def test_choose_threshold_respects_fpr_limit():
    rng = random.Random(11)
    scores = []
    labels = []
    for _ in range(2000):
        is_ai = rng.random() < 0.5
        scores.append(min(1.0, max(0.0, rng.gauss(0.75 if is_ai else 0.35, 0.15))))
        labels.append(1 if is_ai else 0)
    threshold = choose_threshold(scores, labels, max_fpr=0.10)
    metrics = metrics_at(scores, labels, threshold)
    assert metrics["fpr"] <= 0.15  # сглаженный допуск на дискретность порога
    assert metrics["recall"] > 0.5


def test_choose_threshold_with_empty_input():
    assert choose_threshold([], []) == 0.5


def test_cross_validate_reports_folds_and_mean():
    rng = random.Random(3)
    scores = []
    labels = []
    for _ in range(600):
        is_ai = rng.random() < 0.5
        scores.append(min(1.0, max(0.0, rng.gauss(0.8 if is_ai else 0.3, 0.1))))
        labels.append(1 if is_ai else 0)
    report = cross_validate(scores, labels, folds=5, max_fpr=0.15, seed=42)
    assert len(report["folds"]) == 5
    assert report["mean"]["f1"] > 0.8
    assert 0.0 <= report["mean"]["fpr"] <= 1.0


def test_cross_validate_handles_tiny_dataset():
    report = cross_validate([0.1, 0.9], [0, 1], folds=5)
    assert report["folds"] == []
    assert report["n"] == 2
