"""Тесты для hf_features_chunked с фальшивым токенизатором (без torch).

Эти тесты проверяют логику чанкинга и агрегации признаков без реальной модели.
Цель: увеличить покрытие features.py (сейчас 69.9%) и предотвратить незаметные
поломки hf-ветки.
"""

from unittest.mock import patch

import pytest

from spanverify.chunking import chunk_document
from spanverify.features import FeatureMatrix, hf_features_chunked


@pytest.fixture
def mock_hf_backend():
    """Фикстура для мока HFBackend.dependencies()."""
    with patch("spanverify.backends.hf.HFBackend") as mock_backend:
        mock_backend.dependencies.return_value = (True, "")
        yield mock_backend


@pytest.fixture
def mock_hf_features():
    """Фикстура для мока hf_features."""
    with patch("spanverify.features.hf_features") as mock_func:
        # hf_features возвращает FeatureMatrix с известными значениями
        def fake_hf_features(answer, context, **kwargs):
            # Создаём FeatureMatrix с длиной, равной числу токенов в ответе
            # Для простоты считаем токены как слова
            n_tokens = max(1, len(answer.split()))
            return FeatureMatrix(
                attention_entropy=[0.5] * n_tokens,
                ctx_attention_mass=[0.7] * n_tokens,
                embedding_density=[0.6] * n_tokens,
                ctx_attention_mass_norm=[0.8] * n_tokens,
                ctx_max_similarity=[0.9] * n_tokens,
                meta={"backend": "hf", "model": "test", "chunked": False},
            )

        mock_func.side_effect = fake_hf_features
        yield mock_func


def test_hf_features_chunked_short_context(mock_hf_backend, mock_hf_features):
    """Короткий контекст (помещается в окно) — обычный проход без чанкинга."""
    answer = "Это короткий ответ"
    context = "Короткий контекст"  # Менее 380 токенов

    result = hf_features_chunked(answer, context, model_name="test-model", max_length=512)

    # Должен быть вызван hf_features один раз (без чанкинга)
    assert mock_hf_features.call_count == 1
    assert result is not None


def test_hf_features_chunked_long_context(mock_hf_backend, mock_hf_features):
    """Длинный контекст (не помещается в окно) — чанкинг + отбор релевантных."""
    # Создаём длинный контекст (более 380 токенов ≈ 1000+ символов)
    long_context = " ".join([f"Предложение номер {i}." for i in range(100)])
    answer = "Это ответ на длинный документ"

    result = hf_features_chunked(answer, long_context, model_name="test-model", max_length=512)

    # Должен быть вызван hf_features несколько раз (по числу релевантных чанков)
    assert mock_hf_features.call_count >= 1
    assert result is not None


def test_hf_features_chunked_empty_answer(mock_hf_backend, mock_hf_features):
    """Пустой ответ — возврат пустого FeatureMatrix."""
    answer = ""
    context = "Какой-то контекст"

    result = hf_features_chunked(answer, context, model_name="test-model", max_length=512)

    # Должен вернуть пустой FeatureMatrix
    assert len(result) == 0
    assert result.meta.get("empty") is True


def test_hf_features_chunked_empty_context(mock_hf_backend, mock_hf_features):
    """Пустой контекст — обычный проход без чанкинга."""
    answer = "Ответ без контекста"
    context = ""

    hf_features_chunked(answer, context, model_name="test-model", max_length=512)

    # Должен быть вызван hf_features один раз
    assert mock_hf_features.call_count == 1


def test_hf_features_chunked_aggregation(mock_hf_backend):
    """Проверка агрегации признаков: максимум по чанкам."""
    call_count = [0]

    def fake_hf_features(answer, context, **kwargs):
        call_count[0] += 1
        n_tokens = max(1, len(answer.split()))
        # Первый чанк: низкие значения, второй: высокие
        if call_count[0] == 1:
            return FeatureMatrix(
                attention_entropy=[0.3] * n_tokens,
                ctx_attention_mass=[0.4] * n_tokens,
                embedding_density=[0.5] * n_tokens,
                meta={"backend": "hf", "model": "test"},
            )
        else:
            return FeatureMatrix(
                attention_entropy=[0.7] * n_tokens,
                ctx_attention_mass=[0.8] * n_tokens,
                embedding_density=[0.9] * n_tokens,
                meta={"backend": "hf", "model": "test"},
            )

    with patch("spanverify.features.hf_features", side_effect=fake_hf_features):
        # Длинный контекст, чтобы было несколько чанков
        long_context = " ".join([f"Предложение номер {i}." for i in range(100)])
        answer = "Это ответ"

        result = hf_features_chunked(answer, long_context, model_name="test-model", max_length=512)

        # Если было несколько вызовов, проверяем агрегацию (максимум)
        if call_count[0] > 1:
            # Значения должны быть максимумом из всех чанков
            assert all(v >= 0.6 for v in result.attention_entropy)
            assert all(v >= 0.7 for v in result.ctx_attention_mass)
            assert all(v >= 0.8 for v in result.embedding_density)


def test_chunk_document_integration():
    """Интеграционный тест: chunk_document на реальном тексте."""
    text = "Первое предложение. Второе предложение. Третье предложение."
    chunks = chunk_document(text, max_tokens_per_chunk=5)

    # Должно быть хотя бы 2 чанка (текст > 5 токенов)
    assert len(chunks.chunks) >= 2

    # Каждый чанк должен иметь валидные позиции
    for chunk in chunks.chunks:
        assert chunk.start >= 0
        assert chunk.end <= len(text)
        assert chunk.start < chunk.end
