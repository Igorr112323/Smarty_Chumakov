"""Оценка должна применять ту голову, которую только что обучили.

Прогон 37472951524: кросс-валидация головы на корпусе A3 дала AUC 0,85, а
сквозной token F1 — 0,013. Причина не в признаках. Бандл хранил только путь
``config/head.json``, и оценка прочитала голову из поставки (обученную на
демо-корпусе), а не модель этого прогона. Маска, подобранная под свежую
голову, применилась к чужим вероятностям.
"""

from __future__ import annotations

import math

import pytest

from spanverify.calibration import choose_threshold_report
from spanverify.core import Token
from spanverify.engine import _head_risk
from spanverify.features import FeatureMatrix, scale, scale_robust


def test_inline_head_is_used_and_shipped_file_is_not_read(monkeypatch) -> None:
    """Встроенная модель имеет приоритет над файлом поставки."""

    def boom(path: str) -> str:
        raise AssertionError(f"прочитан файл поставки: {path}")

    monkeypatch.setattr("spanverify.config.read_runtime_text", boom)
    head = {
        "type": "logreg",
        "file": "config/head.json",
        "model": {"weights": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "bias": 2.0},
        "scaler": {"means": [0.0] * 6, "scales": [1.0] * 6},
    }
    features = FeatureMatrix(
        attention_entropy=[0.1, 0.2],
        ctx_attention_mass=[0.3, 0.4],
        embedding_density=[0.5, 0.6],
    )
    tokens = [Token("пять", 0, 4), Token("лет", 5, 8)]
    probabilities = _head_risk(head, features, [0.2, 0.2], tokens)
    expected = 1.0 / (1.0 + math.exp(-2.0))
    assert probabilities == [pytest.approx(expected), pytest.approx(expected)]


def test_file_head_is_still_loaded_when_model_is_absent(monkeypatch) -> None:
    """Старый бандл без встроенной модели продолжает читать файл."""
    payload = {
        "model": {"weights": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "bias": -2.0},
        "scaler": {"means": [0.0] * 6, "scales": [1.0] * 6},
    }

    def fake_read(path: str) -> str:
        assert path == "config/head.json"
        import json

        return json.dumps(payload)

    monkeypatch.setattr("spanverify.config.read_runtime_text", fake_read)
    features = FeatureMatrix(attention_entropy=[0.1], ctx_attention_mass=[0.2], embedding_density=[0.3])
    probabilities = _head_risk({"type": "logreg", "file": "config/head.json"}, features, [0.4], [Token("акт", 0, 3)])
    assert probabilities == [pytest.approx(1.0 / (1.0 + math.exp(2.0)))]


def test_choose_threshold_does_not_go_silent_when_fpr_is_impossible() -> None:
    """Невыполнимый потолок FPR не должен выключать детектор."""
    report = choose_threshold_report([0.2, 0.9, 0.3, 0.8], [0, 1, 0, 1], max_fpr=-0.01)
    assert report["constraint_met"] is False
    assert report["f1"] > 0.0
    assert report["threshold"] < 1.0


def test_choose_threshold_reports_when_constraint_holds() -> None:
    """Когда ограничение достижимо, признак constraint_met истинен."""
    report = choose_threshold_report([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], max_fpr=0.1)
    assert report["constraint_met"] is True
    assert report["fpr"] <= 0.1
    assert report["f1"] == pytest.approx(1.0)


def test_scale_robust_matches_scale_at_quantile_one() -> None:
    """Квантиль 1.0 — это прежняя нормировка, демо-числа не должны сдвинуться."""
    values = [0.1, 0.4, 0.2, 0.8]
    assert scale_robust(values, 1.0) == scale(values)
    assert scale_robust([], 0.9) == []


def test_scale_robust_does_not_let_one_spike_zero_the_rest() -> None:
    """Одиночный пик не должен сжимать остальные значения к нулю."""
    values = [0.2, 0.4, 0.4, 20.0]
    robust = scale_robust(values, 0.5)
    naive = scale(values)
    assert naive[0] < 0.02
    assert robust[0] >= 0.5
    assert max(robust) == pytest.approx(1.0)
