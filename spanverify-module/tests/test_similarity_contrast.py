"""Признаки итерации 3: выделенность максимума похожести над фоном.

Чистые функции — без тензоров и весов модели. Содержание проверки: один и тот
же максимум должен означать разное в «гладком» документе (где всё похоже на всё)
и в «рваном» (где есть один явный источник).
"""

from __future__ import annotations

import pytest

from spanverify.features import similarity_contrast, similarity_margin


def test_contrast_is_zero_when_all_sources_are_equal() -> None:
    """Нет выделяющегося источника — нет и опоры, даже при высоком максимуме."""
    assert similarity_contrast(0.9, 0.9) == pytest.approx(0.0)
    assert similarity_contrast(0.2, 0.2) == pytest.approx(0.0)


def test_contrast_separates_peak_from_smooth_document() -> None:
    """Одинаковый максимум 0,8: в гладком документе это фон, в рваном — опора."""
    smooth = similarity_contrast(0.80, 0.78)  # всё похоже на всё
    peaked = similarity_contrast(0.80, 0.10)  # один явный источник
    assert peaked > smooth
    assert smooth == pytest.approx(0.02)
    assert peaked == pytest.approx(0.70)


def test_margin_is_zero_with_a_single_candidate() -> None:
    """Один источник в контексте: отрыв от «второго» равен нулю."""
    assert similarity_margin(0.7, 0.7) == pytest.approx(0.0)


def test_margin_grows_with_uniqueness_of_the_source() -> None:
    """Чем больше отрыв от второго, тем определённее опора."""
    ambiguous = similarity_margin(0.62, 0.60)
    confident = similarity_margin(0.62, 0.15)
    assert confident > ambiguous
    assert confident == pytest.approx(0.47)
    assert ambiguous == pytest.approx(0.02)


def test_contrast_and_margin_are_signed() -> None:
    """Отрицательные значения допустимы и означают «хуже фона»."""
    assert similarity_contrast(0.1, 0.4) == pytest.approx(-0.3)
    assert similarity_margin(0.1, 0.4) == pytest.approx(-0.3)
