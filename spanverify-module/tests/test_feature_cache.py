"""Кеш признаков: ключ, попадание, чтение шардов.

Тесты идут в режиме ``demo`` — веса модели для них не нужны. Проверяется то, от
чего зависит корректность эксперимента на реальной модели: что ключ однозначно
соответствует паре, что попадание возвращает ровно то, что было посчитано, что
промах считает заново, и что шарды склеиваются.
"""

from __future__ import annotations

import json

import pytest

from spanverify.features import (
    FeatureMatrix,
    extract_features,
    feature_cache_key,
    get_feature_cache,
    load_feature_cache,
    set_feature_cache,
)


@pytest.fixture(autouse=True)
def _no_global_cache():
    """Глобальный кеш — общий на весь процесс: тест обязан его чистить."""
    set_feature_cache(None)
    yield
    set_feature_cache(None)


def test_key_is_stable_for_the_same_input() -> None:
    """Одинаковая пара даёт одинаковый ключ — иначе кеш не сработает."""
    first = feature_cache_key("Срок хранения — пять лет", "Документ: приказ № 1", "hf", "model-a")
    second = feature_cache_key("Срок хранения — пять лет", "Документ: приказ № 1", "hf", "model-a")
    assert first == second


def test_key_differs_for_every_significant_part() -> None:
    """Ответ, контекст, режим и модель — всё входит в ключ."""
    answer, context = "Срок хранения — пять лет", "Документ: приказ № 1"
    base = feature_cache_key(answer, context, "hf", "model-a")
    assert base != feature_cache_key(answer + " ", context, "hf", "model-a")
    assert base != feature_cache_key(answer, context + "!", "hf", "model-a")
    assert base != feature_cache_key(answer, context, "demo", "model-a")
    assert base != feature_cache_key(answer, context, "hf", "model-b")


def test_list_context_matches_joined_string() -> None:
    """Контекст списком и тот же контекст строкой — один ключ.

    Верификатор передаёт контекст как есть (строкой или списком фрагментов), и
    от формы записи результат зависеть не должен.
    """
    chunks = ["Документ: приказ № 1", "Раздел 2. Сроки"]
    assert feature_cache_key("ответ", chunks, "hf", "m") == feature_cache_key("ответ", "\n".join(chunks), "hf", "m")


def test_cache_hit_returns_the_cached_matrix() -> None:
    """Попадание в кеш отдаёт ровно то, что туда положили, без пересчёта."""
    answer, context = "Срок хранения — пять лет", "Приказ № 1. Срок хранения первичных документов — пять лет."
    key = feature_cache_key(answer, context, "demo", "")
    planted = FeatureMatrix(
        attention_entropy=[0.1, 0.2],
        ctx_attention_mass=[0.9, 0.8],
        embedding_density=[0.3, 0.3],
        meta={"planted": True},
    )
    result = extract_features(answer, context, mode="demo", cache={key: planted})
    assert result.ctx_attention_mass == [0.9, 0.8]
    assert result.meta.get("planted") is True


def test_cache_miss_computes_normally() -> None:
    """Промах считает заново: пустой кеш не ломает конвейер."""
    answer = "Срок хранения — пять лет"
    context = "Приказ № 1. Срок хранения первичных документов — пять лет."
    cached = extract_features(answer, context, mode="demo", cache={})
    fresh = extract_features(answer, context, mode="demo")
    assert cached.attention_entropy == fresh.attention_entropy


def test_key_that_is_not_in_cache_is_ignored() -> None:
    """Чужой ключ в кеше не подменяет результат."""
    answer, context = "Срок хранения — пять лет", "Приказ № 1"
    planted = FeatureMatrix(attention_entropy=[0.0], ctx_attention_mass=[0.0], embedding_density=[0.0])
    result = extract_features(answer, context, mode="demo", cache={"чужой-ключ": planted})
    assert result.ctx_attention_mass != [0.0]


def test_global_cache_registry_round_trip() -> None:
    """Реестр нужен обучению, которое создаёт верификатор внутри себя."""
    assert get_feature_cache() is None
    mapping = {"k": FeatureMatrix(attention_entropy=[0.5], ctx_attention_mass=[0.5], embedding_density=[0.5])}
    set_feature_cache(mapping)
    assert get_feature_cache() == mapping
    set_feature_cache(None)
    assert get_feature_cache() is None


def test_global_cache_is_used_without_explicit_argument() -> None:
    """Явный параметр недоступен из train(), поэтому работает реестр."""
    answer, context = "Срок хранения — пять лет", "Приказ № 1. Срок хранения — пять лет."
    key = feature_cache_key(answer, context, "demo", "")
    set_feature_cache(
        {key: FeatureMatrix(attention_entropy=[0.42], ctx_attention_mass=[0.42], embedding_density=[0.42])}
    )
    result = extract_features(answer, context, mode="demo")
    assert result.attention_entropy == [0.42]


def test_load_feature_cache_merges_shards(tmp_path) -> None:
    """Шарды, посчитанные разными job'ами, склеиваются в один кеш."""
    for index, (key, value) in enumerate((("key-a", 0.1), ("key-b", 0.2), ("key-c", 0.3))):
        shard = tmp_path / f"features_shard{index}of3.jsonl"
        shard.write_text(
            json.dumps(
                {
                    "key": key,
                    "pair_id": f"p{index}",
                    "attention_entropy": [value],
                    "ctx_attention_mass": [value],
                    "embedding_density": [value],
                }
            )
            + "\n",
            encoding="utf-8",
        )
    cache = load_feature_cache([str(tmp_path)])
    assert sorted(cache) == ["key-a", "key-b", "key-c"]
    assert cache["key-b"].ctx_attention_mass == [0.2]
    assert cache["key-a"].meta["pair_id"] == "p0"


def test_load_feature_cache_ignores_blank_lines_and_missing(tmp_path) -> None:
    """Пустые строки и отсутствующие пути не роняют загрузку."""
    shard = tmp_path / "features_all.jsonl"
    shard.write_text('{"key": "k1", "ctx_attention_mass": [0.5]}\n\n{"key": "k2"}\n', encoding="utf-8")
    cache = load_feature_cache([str(shard), str(tmp_path / "нет такого.jsonl")])
    assert sorted(cache) == ["k1", "k2"]
    assert cache["k2"].ctx_attention_mass == []


def test_load_feature_cache_of_a_single_file(tmp_path) -> None:
    """Один файл — тоже допустимый источник."""
    path = tmp_path / "cache.jsonl"
    path.write_text('{"key": "only", "ctx_attention_mass": [0.7]}\n', encoding="utf-8")
    assert load_feature_cache(str(path))["only"].ctx_attention_mass == [0.7]


def test_counting_cache_counts_hits_and_misses() -> None:
    """Счётчик делает расхождение ключей видимым, а не тихим."""
    from spanverify.features import CountingFeatureCache

    cache = CountingFeatureCache({"k": FeatureMatrix()})
    assert cache.get("k") is not None
    assert cache.get("нет-такого-ключа") is None
    assert cache.get("нет-такого-ключа") is None
    assert (cache.hits, cache.misses) == (1, 2)
    assert "попаданий 1" in repr(cache)


def test_load_feature_cache_can_count(tmp_path) -> None:
    """Загрузчик умеет вернуть считающий кеш — им пользуется эксперимент."""
    from spanverify.features import CountingFeatureCache

    path = tmp_path / "features_all.jsonl"
    path.write_text('{"key": "k", "ctx_attention_mass": [0.4]}\n', encoding="utf-8")
    cache = load_feature_cache([str(path)], counting=True)
    assert isinstance(cache, CountingFeatureCache)
    assert cache["k"].ctx_attention_mass == [0.4]
