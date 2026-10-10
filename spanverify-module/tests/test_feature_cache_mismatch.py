"""Явная отверга кеша признаков, не соответствующего модели или формату (задача 9).

Опасность здесь не в падении, а в тишине: если ключ кеша посчитан для одной
модели, а запрошена другая, признаки молча пересчитываются этой другой моделью
(часы вместо секунд, и посреди кросс-валидации), а кеш старого формата отдаёт
голову нулями вместо признаков, которых в него не писали. Поэтому расхождение
обязано превращаться в :class:`~spanverify.backends.base.BackendUnavailable`
или :class:`ValueError` с текстом ``model mismatch`` и командой на исправление.
"""

from __future__ import annotations

import pytest

from spanverify.backends.base import BackendUnavailable
from spanverify.features import (
    CataloguedFeatureCache,
    FeatureMatrix,
    extract_features,
    feature_cache_key,
    validate_feature_cache,
)

MODEL_CACHED = "ai-forever/rugpt3small_based_on_gpt2"
ANSWER = "Срок хранения первичных документов составляет 10 лет."
CONTEXT = "Регламент 343: срок хранения первичных документов составляет 10 лет."


def _matrix() -> FeatureMatrix:
    return FeatureMatrix(
        attention_entropy=[0.1, 0.2],
        ctx_attention_mass=[0.9, 0.8],
        embedding_density=[0.3, 0.3],
        meta={"model": MODEL_CACHED},
    )


def _cache(model: str | None = MODEL_CACHED, fields: tuple[str, ...] | None = None) -> CataloguedFeatureCache:
    """Кеш с реестром происхождения: ровно то, что пишет precompute_features."""
    cache = CataloguedFeatureCache({feature_cache_key(ANSWER, CONTEXT, "hf", model or ""): _matrix()})
    if model:
        cache.info["models"] = {model}
    cache.info["modes"] = {"hf"}
    cache.info["formats"] = {"2"}
    cache.info["fields"] = set(fields or ("attention_entropy", "ctx_attention_mass", "embedding_density"))
    return cache


def test_model_mismatch_raises_with_explicit_text() -> None:
    """Другая модель — BackendUnavailable, и в тексте есть «model mismatch»."""
    cache = _cache()
    with pytest.raises(BackendUnavailable) as excinfo:
        validate_feature_cache(cache, "hf", "some/other-model")
    message = str(excinfo.value)
    assert "model mismatch" in message
    assert MODEL_CACHED in message and "some/other-model" in message
    # Текст обязан говорить, что делать, — иначе следующий прогон повторит ошибку.
    assert "precompute_features" in message


def test_matching_model_is_accepted_without_recount() -> None:
    """Та же модель — проверка проходит, признаки берутся из кеша как есть."""
    cache = _cache()
    info = validate_feature_cache(cache, "hf", MODEL_CACHED)
    assert info["checked"] is True
    assert cache.info["models"] == {MODEL_CACHED}
    result = extract_features(ANSWER, CONTEXT, mode="hf", cache=cache, model_name=MODEL_CACHED)
    assert result.ctx_attention_mass == [0.9, 0.8]


def test_cache_without_model_name_is_rejected_for_hf() -> None:
    """Кеш формата v1 без имени модели для hf отвергается: проверить нечего."""
    cache = CataloguedFeatureCache({feature_cache_key(ANSWER, CONTEXT, "hf", ""): _matrix()})
    cache.info["fields"] = {"attention_entropy", "ctx_attention_mass", "embedding_density"}
    with pytest.raises(BackendUnavailable) as excinfo:
        validate_feature_cache(cache, "hf", MODEL_CACHED)
    assert "model mismatch" in str(excinfo.value)


def test_missing_head_features_raise_value_error() -> None:
    """В кеше нет строк новой диагностики — голова не должна молча есть нули."""
    cache = _cache()
    with pytest.raises(ValueError) as excinfo:
        validate_feature_cache(cache, "hf", MODEL_CACHED, head_features=("ctx_sim_contrast", "ctx_sim_margin"))
    message = str(excinfo.value)
    assert "ctx_sim_contrast" in message and "ctx_sim_margin" in message
    assert "precompute_features" in message


def test_empty_cache_skips_validation() -> None:
    """Пустой кеш — не ошибка: это «кеш выключен», а не «кеш плохой»."""
    assert validate_feature_cache(None, "hf", MODEL_CACHED)["checked"] is False
    assert validate_feature_cache(CataloguedFeatureCache(), "hf", MODEL_CACHED)["checked"] is False


def test_verifier_rejects_mismatched_cache_at_construction() -> None:
    """Верификатор в режиме hf падает на старте, а не на середине обучения."""
    from spanverify.engine import Verifier

    with pytest.raises(BackendUnavailable) as excinfo:
        Verifier(mode="hf", model_name="some/other-model", weights_path=None, features_cache=_cache())
    assert "model mismatch" in str(excinfo.value)


def test_demo_mode_ignores_model_registry() -> None:
    """Режим demo не зависит от модели: расхождение имён в нём не ошибка."""
    info = validate_feature_cache(_cache(), "demo", "some/other-model")
    assert isinstance(info, dict)
