"""Тесты нормализации текста, чисел и единиц измерения (исправление C3).

Проверяется то, из-за чего метод рассыпался на внешних наборах: требование
почти дословного совпадения. Числа прописью, разные падежи единиц и разные
формы дат обязаны давать один и тот же нормализованный вид — иначе пересказ
документа воспринимается как расхождение.
"""

from __future__ import annotations

from spanverify.normalize import (
    canonical_units,
    normalize_dates,
    normalize_for_match,
    normalize_numbers,
    numbers_to_digits,
    stem_text,
    stem_word,
)


def test_numbers_to_digits_simple() -> None:
    """Число прописью превращается в цифру."""
    assert numbers_to_digits("срок хранения пять лет") == "срок хранения 5 лет"


def test_numbers_to_digits_compound() -> None:
    """Составное числительное собирается в одно число."""
    assert numbers_to_digits("двадцать пять лет") == "25 лет"
    assert numbers_to_digits("сто двадцать") == "120"


def test_numbers_to_digits_keeps_existing_digits() -> None:
    """Уже написанные цифрами числа не портятся."""
    assert numbers_to_digits("срок 10 лет") == "срок 10 лет"


def test_canonical_units_reduces_cases() -> None:
    """Разные падежи одной единицы приводятся к одной форме."""
    assert canonical_units("5 лет") == "5 год"
    assert canonical_units("5 года") == "5 год"
    assert canonical_units("5 году") == "5 год"
    assert canonical_units("10 процентов") == "10 процент"
    assert canonical_units("10 %") == "10 процент"


def test_normalize_dates_words_and_digits() -> None:
    """Словесная и числовая запись даты дают один вид."""
    assert normalize_dates("24 июня 2025 г.") == "24.06.2025"
    assert normalize_dates("24.06.2025") == "24.06.2025"
    assert normalize_dates("1 декабря 2024") == "01.12.2024"


def test_number_words_equal_to_digits() -> None:
    """Главное свойство: «пять лет» и «5 лет» совпадают после нормализации."""
    assert normalize_numbers("пять лет") == normalize_numbers("5 лет")
    assert normalize_for_match("срок составляет десять лет") == normalize_for_match("срок составляет 10 лет")


def test_digits_are_not_lost_in_match_form() -> None:
    """Числа обязаны сохраняться: иначе «10 лет» и «3 года» станут одинаковыми.

    Это регрессионная проверка: при первой реализации стеммер отбрасывал цифры,
    и подмена числа перестала отличаться от правильного значения.
    """
    ten = normalize_for_match("срок хранения 10 лет")
    three = normalize_for_match("срок хранения 3 года")
    assert ten != three
    assert "10" in ten
    assert "3" in three


def test_stem_reduces_word_forms() -> None:
    """Разные формы одного слова дают один стемм."""
    assert stem_word("хранения") == stem_word("хранение")
    assert stem_word("документов") == stem_word("документа")


def test_stem_keeps_short_words() -> None:
    """Короткие слова не обрезаются до неузнаваемости."""
    assert stem_word("год") == "год"
    assert stem_word("") == ""


def test_stem_handles_yo() -> None:
    """«Ё» и «Е» не должны разделять одно и то же слово."""
    assert stem_word("учёт") == stem_word("учет")


def test_stem_text_returns_all_words() -> None:
    """Из текста извлекаются все слова, а не только первое."""
    assert stem_text("срок хранения") == ["срок", "хранен"]


def test_empty_input_is_safe() -> None:
    """Пустой ввод не приводит к исключению."""
    assert numbers_to_digits("") == ""
    assert canonical_units("") == ""
    assert normalize_dates("") == ""
    assert normalize_numbers("") == ""
    assert normalize_for_match("") == ""
    assert stem_text("") == []
