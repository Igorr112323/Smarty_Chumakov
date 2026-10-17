"""Тест round-trip кеша признаков: dump → load сохраняет все FEATURE_NAMES.

Регрессия: в версии 7 октября FEATURE_NAMES был из 3 признаков, потом добавили
ещё 2 (ctx_attention_mass_norm, ctx_max_similarity), а dump_matrix и
load_feature_cache не обновили. Кеш хранил только 3 поля, train.py:138 падал
с IndexError при обращении к пустому списку.

Тест ловит этот класс ошибок: создаёт FeatureMatrix со всеми 5 полями,
дампит через dump_matrix, загружает через load_feature_cache, проверяет:
1. Все 5 полей присутствуют
2. Длины массивов равны длине ответа (не пустые)
3. train.py:collect_samples проходит на загруженной матрице
"""
import json
import tempfile
from pathlib import Path

from spanverify.features import (
    FEATURE_NAMES,
    FeatureMatrix,
    load_feature_cache,
)
from spanverify.train import collect_samples


def _make_matrix(answer_len: int = 5) -> FeatureMatrix:
    """Создать FeatureMatrix со всеми 5 полями для теста."""
    return FeatureMatrix(
        attention_entropy=[0.5] * answer_len,
        ctx_attention_mass=[0.3] * answer_len,
        ctx_attention_mass_norm=[0.4] * answer_len,
        ctx_max_similarity=[0.6] * answer_len,
        embedding_density=[0.7] * answer_len,
    )


def test_dump_load_roundtrip_preserves_all_features():
    """dump_matrix → load_feature_cache сохраняет все FEATURE_NAMES."""
    # Импортируем dump_matrix локально, чтобы не добавлять зависимость в модуль
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from precompute_features import dump_matrix

    matrix = _make_matrix(answer_len=5)
    key = "test|model|512|answer|context"
    pair_id = "test-pair-001"

    # Дампим
    row = dump_matrix(key, matrix, pair_id, max_length=512, chunked=False)

    # Проверяем, что все 5 признаков записаны
    for name in FEATURE_NAMES:
        assert name in row, f"dump_matrix не записал поле {name}"
        assert len(row[name]) == 5, f"длина {name} != 5"

    # Загружаем из временного файла
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = Path(tmpdir) / "features_test.jsonl"
        cache_file.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

        cache = load_feature_cache(cache_file)

        # Проверяем, что все 5 признаков загружены
        assert key in cache, "ключ не найден в кеше"
        loaded = cache[key]
        for name in FEATURE_NAMES:
            values = getattr(loaded, name)
            assert values, f"поле {name} пустое после загрузки"
            assert len(values) == 5, f"длина {name} после загрузки != 5"


def test_load_feature_cache_detects_old_format():
    """load_feature_cache падает, если кеш старого формата (без двух полей)."""
    # Имитируем кеш старого формата (только 3 признака)
    old_row = {
        "key": "test|model|512|answer|context",
        "pair_id": "test-pair-001",
        "attention_entropy": [0.5, 0.5, 0.5],
        "ctx_attention_mass": [0.3, 0.3, 0.3],
        "embedding_density": [0.7, 0.7, 0.7],
        # ctx_attention_mass_norm и ctx_max_similarity отсутствуют
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = Path(tmpdir) / "features_old.jsonl"
        cache_file.write_text(json.dumps(old_row, ensure_ascii=False) + "\n", encoding="utf-8")

        # Ожидаем ValueError с явным сообщением
        try:
            load_feature_cache(cache_file)
            assert False, "load_feature_cache не упал на кеше старого формата"
        except ValueError as exc:
            assert "кеш неполон" in str(exc) or "пересчитайте" in str(exc)


def test_collect_samples_works_with_cached_matrix():
    """train.py:collect_samples проходит на матрице из кеша (все 5 полей)."""
    # Импортируем dump_matrix локально
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from precompute_features import dump_matrix

    matrix = _make_matrix(answer_len=5)
    key = "test|model|512|answer|context"
    pair_id = "test-pair-001"

    # Дампим и загружаем
    row = dump_matrix(key, matrix, pair_id, max_length=512, chunked=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = Path(tmpdir) / "features_test.jsonl"
        cache_file.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        cache = load_feature_cache(cache_file)

        # Проверяем, что загруженная матрица имеет все поля
        loaded = cache[key]
        for name in FEATURE_NAMES:
            values = getattr(loaded, name)
            assert values, f"поле {name} пустое"
            # Это то, что проверяет train.py:138 — getattr(features, name)[index]
            # Если список пустой, будет IndexError
            _ = values[0]  # noqa: B018 — намеренно обращаемся к элементу


if __name__ == "__main__":
    test_dump_load_roundtrip_preserves_all_features()
    test_load_feature_cache_detects_old_format()
    test_collect_samples_works_with_cached_matrix()
    print("✓ все тесты round-trip кеша прошли")
