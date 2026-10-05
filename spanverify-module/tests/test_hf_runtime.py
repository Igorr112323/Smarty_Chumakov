"""Арифметика признаков режима ``hf`` (пункты 2.1 и 2.2 реестра).

Эти функции считают признаки из карт внимания реальной модели. Они вынесены
из torch-части специально, чтобы их можно было проверить обычными тестами на
настоящих числах: сама по себе загрузка весов проверяется отдельно и требует
модели, а формулы — нет.
"""

from __future__ import annotations

import math

import pytest

from spanverify.hf_runtime import (
    aggregate_layers,
    batch_plan,
    clear_model_cache,
    context_mass_of_row,
    entropy_of_row,
    hf_offline_hint,
    model_cache_info,
    normalize_mass,
    resolve_heads,
    support_distance,
    words_from_subwords,
)


def test_entropy_uniform_is_one() -> None:
    """Равномерное внимание — максимальная неопределённость (1,0)."""
    assert entropy_of_row([0.25] * 4) == pytest.approx(1.0)


def test_entropy_point_mass_is_zero() -> None:
    """Внимание в одну точку — нулевая энтропия."""
    assert entropy_of_row([1.0, 0.0, 0.0, 0.0]) == pytest.approx(0.0)


def test_entropy_is_normalised_by_window_length() -> None:
    """Нормировка на log(n) делает окна разной длины сравнимыми.

    Без неё равномерное внимание на 64 токена выглядело бы «неувереннее»,
    чем равномерное внимание на 8, хотя степень неопределённости одна.
    """
    assert entropy_of_row([1 / 8] * 8) == pytest.approx(entropy_of_row([1 / 64] * 64))


def test_entropy_raw_matches_formula() -> None:
    """Без нормировки значение совпадает с формулой Шеннона."""
    assert entropy_of_row([0.5, 0.5], normalize=False) == pytest.approx(math.log(2))


def test_entropy_of_empty_row() -> None:
    """Пустая строка внимания не роняет расчёт."""
    assert entropy_of_row([]) == 0.0
    assert entropy_of_row([0.0, 0.0]) == 0.0


def test_context_mass_counts_only_context_positions() -> None:
    """Масса — доля внимания, ушедшая на токены документа."""
    assert context_mass_of_row([0.1, 0.2, 0.3, 0.4], [0, 1]) == pytest.approx(0.3)
    assert context_mass_of_row([0.1, 0.2, 0.3, 0.4], []) == 0.0


def test_normalize_mass_is_half_at_random_level() -> None:
    """Масса «как при случайном внимании» даёт 0,5 — это точка отсчёта.

    Сырая масса растёт просто от длины контекста: если документ занимает
    половину последовательности, то и 0,5 массы — это «ничего не значит».
    """
    assert normalize_mass(0.5, 5, 10) == pytest.approx(0.5)
    assert normalize_mass(0.9, 9, 10) == pytest.approx(0.5)


def test_normalize_mass_rewards_above_random_attention() -> None:
    """Внимание выше случайного даёт значение выше 0,5, ниже — ниже."""
    assert normalize_mass(0.9, 5, 10) > 0.5
    assert normalize_mass(0.1, 5, 10) < 0.5


def test_normalize_mass_edge_cases() -> None:
    """Пустой контекст — нулевая нормированная масса, без деления на ноль."""
    assert normalize_mass(0.5, 0, 10) == 0.0
    assert normalize_mass(0.5, 5, 0) == 0.0


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("all", [0, 1, 2, 3]),
        (None, [0, 1, 2, 3]),
        ("first", [0]),
        ("last", [3]),
        ("half", [0, 1]),
        ("0,2", [0, 2]),
        ([0, -1], [0, 3]),
    ],
)
def test_resolve_heads(spec, expected) -> None:
    """Головы внимания выбираются по описанию (итерация 2.2 «в»)."""
    assert resolve_heads(spec, 4) == expected


def test_resolve_heads_no_heads() -> None:
    """Модель без голов внимания не ломает выбор."""
    assert resolve_heads("all", 0) == []


def test_aggregate_layers() -> None:
    """Слои сводятся средним, максимумом или берётся последний."""
    values = [[1.0, 2.0], [3.0, 6.0]]
    assert aggregate_layers(values, "mean") == [2.0, 4.0]
    assert aggregate_layers(values, "max") == [3.0, 6.0]
    assert aggregate_layers(values, "last") == [3.0, 6.0]
    assert aggregate_layers([]) == []


def test_support_distance_zero_at_anchor_and_one_without_anchors() -> None:
    """Расстояние до ближайшей опоры: 0 у подтверждённого, 1 — если опор нет."""
    assert support_distance([0, 1, 2, 3], [True, False, False, False])[0] == 0.0
    assert support_distance([0, 1, 2], [False, False, False]) == [1.0, 1.0, 1.0]
    assert support_distance([], []) == []


def test_support_distance_grows_with_distance() -> None:
    """Чем дальше токен от подтверждённого места, тем больше признак."""
    values = support_distance(list(range(6)), [True] + [False] * 5)
    assert values == sorted(values)
    assert values[-1] >= values[1]


def test_words_from_subwords_averages() -> None:
    """Значение слова — среднее по его сабтокенам; слово без них берёт default."""
    text = "Срок хранения"
    word_spans = [(0, 4), (5, 13)]
    offsets = [(0, 4), (4, 9), (9, 13)]
    values = [1.0, 0.0, 1.0]
    assert words_from_subwords(text, word_spans, offsets, values) == [1.0, 0.5]
    assert words_from_subwords(text, word_spans, [], [], default=0.25) == [0.25, 0.25]


def test_model_cache_is_empty_and_clearable() -> None:
    """Кэш весов виден снаружи и очищается (без него прогон читал веса заново)."""
    clear_model_cache()
    info = model_cache_info()
    assert info["size"] == 0 and info["keys"] == []


def test_offline_hint_reports_environment() -> None:
    """Подсказка об офлайн-режиме перечисляет ровно три переменные окружения."""
    hint = hf_offline_hint()
    assert "HF_HUB_OFFLINE" in hint and "HF_HOME" in hint
    assert hint.count(";") == 2


def test_batch_plan() -> None:
    """Окна режутся на пакеты — это путь обработки при нехватке памяти."""
    assert batch_plan(5, 2) == [(0, 2), (2, 4), (4, 5)]
