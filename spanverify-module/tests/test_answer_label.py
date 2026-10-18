"""Тесты для функции answer_label (задача 2.1)."""

import pytest

from spanverify.oof import answer_label


def test_answer_label_positive():
    """Ответ с размеченным недостоверным фрагментом должен давать label=1."""
    pair = {"id": "test-1", "labels": [[10, 20, 1], [30, 40, 0]]}  # Один фрагмент с label=1
    assert answer_label(pair) == 1


def test_answer_label_negative():
    """Ответ без размеченных недостоверных фрагментов должен давать label=0."""
    pair = {"id": "test-2", "labels": [[10, 20, 0], [30, 40, 0]]}  # Все фрагменты с label=0
    assert answer_label(pair) == 0


def test_answer_label_empty_labels():
    """Ответ без labels должен давать label=0."""
    pair = {"id": "test-3", "labels": []}
    assert answer_label(pair) == 0


def test_answer_label_missing_labels():
    """Ответ без поля labels должен давать label=0."""
    pair = {"id": "test-4"}
    assert answer_label(pair) == 0


def test_answer_label_single_char_number():
    """Односимвольное число в размеченном фрагменте должно давать label=1.

    Это критичный тест для задачи 2.1: односимвольные числа не проходят
    scored_token фильтр, но ответ всё равно должен считаться недостоверным.
    """
    pair = {"id": "test-5", "labels": [[5, 6, 1]]}  # Фрагмент "2" (односимвольное число)
    assert answer_label(pair) == 1


def test_answer_label_string_vs_int_labels():
    """Функция должна работать с метками как строками и как числами."""
    pair_str = {"id": "test-6", "labels": [[10, 20, "1"]]}  # Метка как строка
    assert answer_label(pair_str) == 1

    pair_int = {"id": "test-7", "labels": [[10, 20, 1]]}  # Метка как число
    assert answer_label(pair_int) == 1


def test_answer_label_multiple_fragments():
    """Ответ с несколькими недостоверными фрагментами должен давать label=1."""
    pair = {"id": "test-8", "labels": [[10, 20, 1], [30, 40, 1], [50, 60, 0]]}  # Два фрагмента с label=1
    assert answer_label(pair) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
