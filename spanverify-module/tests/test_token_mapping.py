"""Регрессия на сопоставление токенов и подслов модели (баг «одинаковых значений»).

Симптом дефекта: признаки на «факт-токене» в пилоте совпадали у групп
«с опорой» и «без опоры» до шестого знака (0.544094 против 0.544094), потому что
признак брался по индексу нашего токена в списке позиций модели, а модель режет
слова на подслова и индекс попадал на соседнее слово. Исправление — сопоставление
по перекрытию символов с усреднением по подсловам.
"""

from __future__ import annotations

from spanverify.features import _span_overlap, map_token_positions


def test_map_token_positions_matches_subwords_by_characters() -> None:
    """Токену сопоставляются те позиции модели, которые реально перекрываются."""
    token_spans = [(0, 4), (4, 13), (13, 15)]
    model_spans = [(3, 11, 15), (4, 15, 24), (5, 24, 26)]
    assert map_token_positions(token_spans, model_spans, 11) == [[3], [4], [5]]


def test_map_token_positions_averages_over_split_subwords() -> None:
    """Если слово модели разрезано на два подслова, токен получает обе позиции."""
    assert map_token_positions([(0, 10)], [(0, 0, 6), (1, 6, 10)], 0) == [[0, 1]]


def test_map_token_positions_without_overlap_is_empty() -> None:
    """Без пересечения по символам позиция не подставляется (раньше бралась «соседняя»)."""
    assert map_token_positions([(0, 2)], [(0, 50, 60)], 100) == [[]]


def test_span_overlap_counts_characters() -> None:
    """Перекрытие измеряется в символах и не путает касание с пересечением."""
    assert _span_overlap((10, 20), (15, 25)) == 5
    assert _span_overlap((10, 20), (20, 30)) == 0
    assert _span_overlap((10, 20), (0, 40)) == 10
