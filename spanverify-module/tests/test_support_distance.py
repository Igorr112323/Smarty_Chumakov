"""Признаки итерации 2: расстояние до подтверждающего фрагмента контекста.

Проверяются чистые функции — тензоров и весов модели для этого не нужно.
Содержание: расстояние нормируется на длину текста (иначе «200 токенов» в
коротком и длинном акте означали бы разное), а штраф за удалённость сохраняет
шкалу похожести.
"""

from __future__ import annotations

import math

import pytest

from spanverify.features import distance_decay, normalised_support_distance


def test_distance_is_normalised_by_sequence_length() -> None:
    """Расстояние измеряется в долях длины текста, а не в токенах."""
    assert normalised_support_distance(0, 100) == 0.0
    assert normalised_support_distance(99, 100) == 1.0
    assert normalised_support_distance(50, 101) == 0.5


def test_distance_degenerate_cases() -> None:
    """Короткие последовательности не дают деления на ноль."""
    assert normalised_support_distance(0, 1) == 0.0
    assert normalised_support_distance(5, 0) == 0.0
    assert normalised_support_distance(-10, 100) == 0.0
    assert normalised_support_distance(10_000, 100) == 1.0


def test_same_relative_distance_is_the_same_feature_value() -> None:
    """Одинаковое относительное удаление в длинном и коротком акте — одно число."""
    long_text = normalised_support_distance(200, 1001)  # 0.2
    short_text = normalised_support_distance(20, 101)  # 0.2
    assert long_text == short_text == pytest.approx(0.2)


def test_decay_keeps_similarity_scale_at_zero_distance() -> None:
    """На нулевом расстоянии штрафа нет: значение равно похожести."""
    assert distance_decay(0.8, 0, 10.0) == pytest.approx(0.8)
    assert distance_decay(0.0, 0, 10.0) == pytest.approx(0.0)


def test_decay_is_exponential_and_monotone() -> None:
    """Чем дальше опора, тем слабее она учитывается; убывание монотонное."""
    values = [distance_decay(1.0, distance, 10.0) for distance in (0, 5, 10, 40)]
    assert values[0] > values[1] > values[2] > values[3]
    assert values[1] == pytest.approx(math.exp(-0.5))
    assert values[2] == pytest.approx(math.exp(-1.0))


def test_decay_scale_is_stable() -> None:
    """Нулевой или отрицательный масштаб не приводит к делению на ноль."""
    assert distance_decay(0.7, 3, 0.0) == 0.0
    assert distance_decay(0.7, 3, -5.0) == 0.0


def test_decay_prefers_close_support_over_distant() -> None:
    """Близкая посредственная опора весомее далёкой точной.

    Смысл признака: подтверждение должно лежать рядом с утверждением, иначе это
    не опора для него, а просто похожий фрагмент где-то в документе.
    """
    close_mediocre = distance_decay(0.60, 2, 10.0)
    distant_strong = distance_decay(0.95, 45, 10.0)
    assert close_mediocre > distant_strong
