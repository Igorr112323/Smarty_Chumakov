"""Нормализация чисел, единиц и словоформ (пункт 2.5 реестра).

Проверяется то, на чём метод рассыпался раньше: числа прописью, падежные
формы, разные написания единиц измерения.
"""

from __future__ import annotations

import pytest

from spanverify.numnorm import (
    canonical_unit,
    measurements,
    normalize_numbers,
    normalize_text,
    stem,
    words_to_number,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ноль", 0.0),
        ("пять", 5.0),
        ("пяти", 5.0),
        ("десять", 10.0),
        ("двадцать пять", 25.0),
        ("сорок пять", 45.0),
        ("сто двадцать", 120.0),
        ("двести", 200.0),
        ("тысяча двести тридцать четыре", 1234.0),
        ("полтора", 1.5),
        ("трое", 3.0),
    ],
)
def test_words_to_number(text: str, expected: float) -> None:
    """Числительные прописью разбираются в число, включая составные и дробные."""
    assert words_to_number(text) == expected


@pytest.mark.parametrize("text", ["договор", "срок хранения", "", "документы"])
def test_words_to_number_rejects_non_numbers(text: str) -> None:
    """Обычные слова числом не считаются: иначе любая фраза стала бы величиной."""
    assert words_to_number(text) is None


def test_normalize_numbers_keeps_source_offsets() -> None:
    """Замена прописи на цифры возвращает смещения в ИСХОДНОМ тексте.

    Это обязательное условие: человеку показывается фрагмент исходного ответа,
    а сравнение идёт по нормализованному виду.
    """
    text = "Срок хранения составляет двадцать пять лет"
    normalized, replacements = normalize_numbers(text)
    assert normalized == "Срок хранения составляет 25 лет"
    assert len(replacements) == 1
    start, end, value = replacements[0]
    assert text[start:end] == "двадцать пять"
    assert value == "25"


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("мегабайт", "MB"),
        ("МБ", "MB"),
        ("Мб", "MB"),
        ("лет", "year"),
        ("годами", "year"),
        ("рабочих дней", "day_work"),
        ("суток", "day"),
        ("часа", "hour"),
        ("процентов", "percent"),
    ],
)
def test_canonical_unit(text: str, code: str) -> None:
    """Разные написания одной единицы сводятся к одному коду."""
    assert canonical_unit(text) == code


def test_canonical_unit_unknown() -> None:
    """Для слова, которое единицей не является, возвращается None."""
    assert canonical_unit("договором") is None


def test_measurements_match_across_spelling() -> None:
    """«пять лет» и «5 лет» дают одинаковый ключ — это и есть устойчивость."""
    words = measurements("срок хранения пять лет")
    digits = measurements("срок хранения 5 лет")
    assert [m.key for m in words] == ["5|year"]
    assert [m.key for m in digits] == ["5|year"]


def test_measurements_do_not_double_count() -> None:
    """Число, уже распознанное прописью, повторно как цифра не берётся."""
    found = measurements("десять мегабайт и 20 МБ")
    assert [m.key for m in found] == ["10|MB", "20|MB"]


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("документами", "документ"),
        ("документов", "документ"),
        ("документ", "документ"),
        ("учёта", "учет"),
        ("учета", "учет"),
        ("хранения", "хранени"),
    ],
)
def test_stem(word: str, expected: str) -> None:
    """Падежные формы и «ё» сводятся к одной основе."""
    assert stem(word) == expected


def test_normalize_text_combines_both() -> None:
    """Канонический вид строки: цифры вместо прописи, основы вместо словоформ."""
    assert normalize_text("Документами учёта за пять лет") == "документ учет за 5 лет"
    assert normalize_text("Документы учета за 5 лет") == "документ учет за 5 лет"
