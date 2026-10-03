"""Тесты ядра: оценка текста, локализация фрагментов, устойчивость."""

from __future__ import annotations

import random

import pytest

from spanverify import Config, Detector
from spanverify.backends.base import get_backend
from spanverify.demo_data import (
    generate_dataset,
    make_ai_paragraph,
    make_human_paragraph,
    make_mixed_document,
)
from spanverify.training import train_calibrator


@pytest.fixture(scope="module")
def calibrator_config():
    """Герметичная калибровка на небольшом синтетическом корпусе."""
    config = Config(backend="surrogate", calibration_path="config/__tests_absent.json")
    report = train_calibrator(Detector(config), generate_dataset(80, seed=2024), config=config, dataset_name="tests")
    return config.with_overrides(threshold=report.threshold), report.calibrator


@pytest.fixture(scope="module")
def detector(calibrator_config) -> Detector:
    config, calibrator = calibrator_config
    return Detector(config, calibrator=calibrator)


def test_empty_text_returns_zero_result(detector: Detector):
    result = detector.analyze("   ")
    assert result.ai_fraction == 0.0
    assert result.spans == []
    assert result.meta.get("empty") is True


def test_result_is_serializable(detector: Detector):
    payload = detector.analyze("Проверка сериализации результата.").to_dict()
    assert payload["verdict"] in {"likely_ai", "mixed", "likely_human"}
    assert 0.0 <= payload["ai_fraction"] <= 1.0
    assert isinstance(payload["spans"], list)
    assert "warnings" in payload


def test_ai_text_scores_higher_than_human_text(detector: Detector):
    """Средняя оценка машинных абзацев выше, человеческие не дают фрагментов.

    Порог подобран под ограничение FPR, поэтому возможны пропуски (низкая
    полнота) — это ожидаемое поведение, а не ошибка: строгий порог защищает
    от ложных обвинений в адрес человеческого текста.
    """
    rng = random.Random(31)
    ai_scores = [detector.analyze(make_ai_paragraph(rng, 6)).ai_fraction for _ in range(10)]
    human_scores = [detector.analyze(make_human_paragraph(rng, 6)).ai_fraction for _ in range(10)]
    mean_ai = sum(ai_scores) / len(ai_scores)
    mean_human = sum(human_scores) / len(human_scores)
    assert mean_ai > mean_human + 0.3
    assert max(human_scores) < 0.5  # человеческий текст не «обвиняется» целиком
    assert max(ai_scores) > 0.5


def test_spans_are_inside_text_and_ordered(detector: Detector):
    doc = make_mixed_document(random.Random(7))
    text = doc["text"]
    result = detector.analyze(text)
    assert result.spans, "на смешанном документе ожидаются найденные фрагменты"
    previous_end = -1
    for span in result.spans:
        assert 0 <= span.start_char < span.end_char <= len(text)
        assert text[span.start_char : span.end_char] == span.text
        assert span.start_char >= previous_end
        previous_end = span.end_char


def test_span_length_respects_min_span_tokens():
    config = Config(backend="surrogate", min_span_tokens=25, calibration_path="config/_none.json")
    detector = Detector(config)
    doc = make_mixed_document(random.Random(11))
    for span in detector.analyze(doc["text"]).spans:
        assert span.n_tokens >= 25


def test_higher_threshold_never_increases_ai_share(detector: Detector):
    doc = make_mixed_document(random.Random(3))
    low = detector.analyze(doc["text"], threshold=0.1).ai_fraction
    high = detector.analyze(doc["text"], threshold=0.9).ai_fraction
    assert high <= low


def test_surrogate_mode_adds_honest_warning(detector: Detector):
    result = detector.analyze("Любой достаточно длинный текст для проверки предупреждений.")
    assert any("ДЕМО-РЕЖИМ" in w for w in result.warnings)


def test_analysis_is_deterministic(detector: Detector):
    doc = make_mixed_document(random.Random(13))
    first = detector.analyze(doc["text"]).to_dict()
    second = detector.analyze(doc["text"]).to_dict()
    assert first["ai_fraction"] == second["ai_fraction"]
    assert [s["start_char"] for s in first["spans"]] == [s["start_char"] for s in second["spans"]]


def test_mixed_document_estimates_track_ground_truth(detector: Detector):
    """Оценка доли участия ИИ сохраняет порядок документов по «машинности».

    Абсолютная ошибка зависит от качества калибровки (в тестах калибратор
    обучен на небольшом корпусе), поэтому проверяется ранговая корреляция
    и слабое ограничение на абсолютную ошибку.
    """
    rng = random.Random(23)
    pairs = []
    for _ in range(20):
        doc = make_mixed_document(rng)
        truth = sum(e - s for s, e, label in doc["labels"] if int(label) == 1) / len(doc["text"])
        pairs.append((truth, detector.analyze(doc["text"]).ai_fraction))

    truths = [t for t, _ in pairs]
    estimates = [e for _, e in pairs]
    correlation = _pearson(truths, estimates)
    mae = sum(abs(t - e) for t, e in pairs) / len(pairs)
    assert correlation > 0.6, f"ранговая согласованность низкая: r={correlation:.2f}"
    assert mae < 0.4


def _pearson(xs, ys) -> float:
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=False))
    var_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    var_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
    return cov / (var_x * var_y) if var_x and var_y else 0.0


def test_explain_returns_per_token_table(detector: Detector):
    payload = detector.explain("Данный метод обеспечивает эффективное решение задачи.")
    assert payload["tokens"]
    token = payload["tokens"][0]
    assert {"token", "start", "end", "raw", "prob", "flag"} <= set(token)


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError):
        get_backend("нет-такого-бэкенда")


def test_load_detector_helper_uses_overrides():
    from spanverify.detector import load_detector

    detector = load_detector("config/__absent__.json", backend="surrogate", threshold=0.33)
    assert detector.config.threshold == pytest.approx(0.33)
