"""Нормировка массы внимания на длину контекста (итерация 1 доработки признаков).

Проверяются только чистые функции: тензоров и весов модели для этого не нужно.
Содержание проверки — что нормализованная масса убирает артефакт длины
контекста и позиции токена, из-за которого сырая масса несравнима между
документами.
"""

from __future__ import annotations

import pytest

from spanverify.features import context_mass_lift, expected_context_mass, normalised_context_mass


def test_expected_mass_is_share_of_visible_positions() -> None:
    """Ожидаемая масса = доля контекстных позиций в causal-префиксе."""
    assert expected_context_mass(100, 199) == 0.5  # 100 из 200 видимых
    assert expected_context_mass(100, 99) == 1.0  # весь префикс — контекст
    assert expected_context_mass(10, 999) == 0.01


def test_expected_mass_degenerate_cases() -> None:
    """Нулевой контекст и нулевая позиция не дают деления на ноль."""
    assert expected_context_mass(0, 50) == 0.0
    assert expected_context_mass(-5, 50) == 0.0
    assert expected_context_mass(10, -1) == 0.0


def test_normalised_mass_is_one_when_attention_is_uniform() -> None:
    """Равномерное внимание → ровно 1,0, независимо от длины контекста."""
    long_context = expected_context_mass(900, 999)  # 0.9
    short_context = expected_context_mass(90, 999)  # 0.09
    # В обоих случаях наблюдаемая масса равна ожидаемой: «как при случайном
    # внимании». Сырые массы 0.9 и 0.09 выглядят совершенно по-разному.
    assert normalised_context_mass(long_context, long_context) == 1.0
    assert normalised_context_mass(short_context, short_context) == 1.0


def test_normalised_mass_removes_context_length_artifact() -> None:
    """Одинаковая «сила опоры» в длинном и коротком документе → близкий norm.

    Сырые массы при этом различаются на порядок, поэтому сравнивать их между
    документами было нельзя: документ с длинным контекстом казался «более
    поддержанным» просто за счёт размера.
    """
    long_ctx_tokens, long_position = 900, 999
    short_ctx_tokens, short_position = 90, 99
    # Доля контекста в доступном префиксе одинакова (0.9), но абсолютные длины
    # разные: 900 из 1000 против 90 из 100.
    long_expected = expected_context_mass(long_ctx_tokens, long_position)
    short_expected = expected_context_mass(short_ctx_tokens, short_position)
    assert long_expected == short_expected == 0.9

    long_observed, short_observed = 0.90, 0.90
    assert abs(long_observed - short_observed) < 1e-12  # сырые совпадают
    assert normalised_context_mass(long_observed, long_expected) == normalised_context_mass(
        short_observed, short_expected
    )


def test_normalised_mass_orders_by_support_not_by_length() -> None:
    """Длинный контекст со слабой опорой не должен выглядеть поддержанным."""
    # Короткий контекст: равномерная масса ничтожна (0.09), наблюдаемая 0.18 —
    # это вдвое больше случайного. Длинный: равномерная 0.9, наблюдаемая 0.85 —
    # меньше случайного, хотя сырая величина в 4,7 раза больше первой.
    strong_short = normalised_context_mass(0.18, expected_context_mass(90, 999))
    weak_long = normalised_context_mass(0.85, expected_context_mass(900, 999))
    assert strong_short > 1.0 > weak_long
    assert 0.18 < 0.85  # сырые величины дают обратный порядок


def test_normalised_mass_zero_expected_gives_zero() -> None:
    """Без контекста нормированная масса не определена → 0, а не Infinity."""
    assert normalised_context_mass(0.0, 0.0) == 0.0
    assert normalised_context_mass(0.5, 0.0) == 0.0


def test_lift_is_bounded_and_signed() -> None:
    """lift = наблюдаемая − ожидаемая, знак сохраняет смысл, границы [-1, 1]."""
    assert context_mass_lift(0.8, 0.5) == pytest.approx(0.3)
    assert context_mass_lift(0.2, 0.5) == pytest.approx(-0.3)
    assert context_mass_lift(0.5, 0.5) == pytest.approx(0.0)
    assert context_mass_lift(5.0, 0.0) == 1.0
    assert context_mass_lift(0.0, 5.0) == -1.0


def test_position_monotonicity_of_expected_mass() -> None:
    """Чем дальше токен ответа от контекста, тем меньше равномерная масса."""
    values = [expected_context_mass(100, position) for position in (105, 150, 300, 900)]
    assert values[0] > values[1] > values[2] > values[3]
    assert values[0] == 100 / 106
