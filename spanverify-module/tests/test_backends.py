"""Тесты фабрики бэкендов и режима 'hf' без установленных зависимостей."""

from __future__ import annotations

import pytest

from spanverify.backends import BackendUnavailable, get_backend
from spanverify.backends.surrogate import SurrogateBackend


def test_factory_creates_surrogate_backend():
    backend = get_backend("surrogate")
    assert isinstance(backend, SurrogateBackend)
    assert backend.available()


def test_factory_accepts_aliases():
    assert isinstance(get_backend("demo"), SurrogateBackend)
    assert isinstance(get_backend("stub"), SurrogateBackend)


def test_surrogate_ignores_model_arguments():
    backend = get_backend("surrogate", model="неважно", max_tokens=8)
    assert backend.name == "surrogate"


def test_surrogate_returns_features_for_each_word():
    backend = SurrogateBackend()
    words = "Данный метод обеспечивает эффективное решение задачи".split()
    result = backend.process(words, " ".join(words), dim=256)
    assert len(result.predictability) == len(words)
    assert all(0.0 <= p <= 1.0 for p in result.predictability)
    assert result.vectors is not None and len(result.vectors) == len(words)
    assert result.informative is not None


def test_surrogate_marks_service_words_uninformative():
    backend = SurrogateBackend()
    words = ["и", "метод"]
    result = backend.process(words, "и метод", dim=128)
    assert result.informative == [False, True]


def test_surrogate_handles_empty_input():
    result = SurrogateBackend().process([], "", dim=128)
    assert result.predictability == []


def test_hf_backend_reports_missing_dependencies_or_works():
    """Или зависимости есть и бэкенд доступен, или он честно сообщает об их отсутствии."""
    backend = get_backend("hf", model="cointegrated/rubert-tiny2")
    available = backend.available()
    if not available:
        with pytest.raises(BackendUnavailable):
            backend.process(["тест"], "тест")
        message = backend.dependencies()[1]
        assert "transformers" in message


def test_unknown_backend_raises_value_error():
    with pytest.raises(ValueError, match="неизвестный бэкенд"):
        get_backend("magic")
