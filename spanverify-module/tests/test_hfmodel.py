"""Тесты модуля работы с реальной моделью (пункт 2.1 промта).

Часть проверок не требует torch: это чистая логика окна, выбора слоя и
распознавания ошибки нехватки памяти. Проверки с самой моделью включаются
переменной окружения ``SPANVERIFY_TEST_MODEL`` (путь к локальной папке с весами
или имя модели) — так тест не ходит в сеть без явного разрешения.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from spanverify.hfmodel import (
    _feature_cache_path,
    _oom_error,
    clear_model_cache,
    model_features,
    resolve_layer,
    window_ranges,
)

MODEL = os.environ.get("SPANVERIFY_TEST_MODEL")
needs_model = pytest.mark.skipif(not MODEL, reason="нет SPANVERIFY_TEST_MODEL: тест с реальными весами пропущен")


def test_window_ranges_covers_tail() -> None:
    """Скользящее окно: последнее окно доходит до конца, окна перекрываются."""
    ranges = window_ranges(300, 128)
    assert ranges[0] == (0, 128)
    assert ranges[-1][1] == 300
    for (start, end), (next_start, _next_end) in zip(ranges, ranges[1:], strict=False):
        assert next_start < end, "окна должны перекрываться"
        assert next_start > start, "окно должно двигаться вперёд"


def test_window_ranges_short_text_single_window() -> None:
    """Короткий текст обрабатывается одним окном без нарезки."""
    assert window_ranges(50, 128) == [(0, 50)]


@pytest.mark.parametrize(
    ("spec", "total", "expected"),
    [
        ("first", 6, 0),
        ("middle", 6, 2),
        ("last", 6, 5),
        (-1, 6, 5),
        (-4, 6, 2),
        (0, 6, 0),
        (99, 6, 5),
        ("нет-такого-слоя", 6, 5),
        (True, 6, 5),
        ("last", 0, 0),
    ],
)
def test_resolve_layer(spec: object, total: int, expected: int) -> None:
    """Выбор слоя: имя, номер, отрицательный индекс, вне диапазона."""
    assert resolve_layer(spec, total) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("CUDA out of memory. Tried to allocate 2.00 GiB", True),
        ("RuntimeError: std::bad_alloc", True),
        ("cannot allocate memory", True),
        ("Segmentation fault", False),
        ("KeyError: 'input_ids'", False),
    ],
)
def test_oom_error_detection(message: str, expected: bool) -> None:
    """Распознавание нехватки памяти: окно уменьшается только в этом случае."""
    assert _oom_error(RuntimeError(message)) is expected


def test_feature_cache_path_depends_on_text() -> None:
    """Кэш признаков различает тексты и выключается пустым каталогом."""
    assert _feature_cache_path(None, "m", "текст", 0, 128, 5) is None
    first = _feature_cache_path("/tmp/cache", "m", "текст", 0, 128, 5)
    second = _feature_cache_path("/tmp/cache", "m", "другой текст", 0, 128, 5)
    assert first is not None and second is not None
    assert first != second
    assert first.parent == Path("/tmp/cache")


@needs_model
def test_model_features_on_real_model(tmp_path: Path) -> None:
    """Признаки реальной модели: длины совпадают, окно и кэш работают."""
    clear_model_cache()
    context = "Срок хранения первичных учётных документов составляет 5 лет. " * 40
    answer = "Срок хранения первичных учётных документов составляет 5 лет."
    result = model_features(
        answer,
        context,
        str(MODEL),
        max_tokens=128,
        feature_cache=str(tmp_path / "features"),
    )
    assert result.meta["backend"] == "hf"
    assert result.meta["windows"] >= 2, "длинный текст должен идти скользящим окном"
    assert len(result.entropy) == len(result.tokens)
    assert len(result.mass) == len(result.tokens)
    assert all(0.0 <= value <= 1.0 for value in result.entropy)
    assert all(0.0 <= value <= 1.0 for value in result.mass)

    cached = model_features(
        answer,
        context,
        str(MODEL),
        max_tokens=128,
        feature_cache=str(tmp_path / "features"),
    )
    assert cached.meta["cached"] is True
    assert cached.entropy == result.entropy
    files = list((tmp_path / "features").glob("features-*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["meta"]["model"] == str(MODEL)


@needs_model
def test_model_features_marks_missing_context() -> None:
    """Ответ без контекста: масса внимания на контекст равна нулю, ошибок нет."""
    result = model_features("Срок хранения составляет 5 лет.", None, str(MODEL), max_tokens=128)
    assert result.meta["context_tokens"] == 0
    assert all(value == 0.0 for value in result.mass)
