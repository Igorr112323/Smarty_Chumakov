"""Нормализация текста: «ё», леммы, единицы измерения, числа словами.

Тесты проверяют именно поведение, на которое опирается устойчивость метода
(пункт 2.5 промта): дословное совпадение слов и записей числа больше не требуется.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spanverify.normalize import (  # noqa: E402
    canonical_unit,
    has_number_words,
    iter_windows,
    lemma_sequence,
    lemmatize,
    normalize_text,
    numbers_compatible,
    numbers_in_text,
    numbers_values,
    token_lemmas,
)


def test_normalize_text_folds_yo_and_case() -> None:
    """«ё» и регистр не должны влиять на сравнение: «Всё» и «все» — одно слово."""
    assert normalize_text("Всё Согласно ПРИКАЗУ") == "все согласно приказу"
    assert normalize_text("5\u00a0лет — срок") == "5 лет - срок"
    assert normalize_text("«Срок»") == '"срок"'


def test_lemmatize_uses_dictionary_for_legal_words() -> None:
    """Формы частых юридических слов сводятся к одной лемме."""
    for word in ("документы", "документов", "документами", "документах"):
        assert lemmatize(word) == "документ"
    for word in ("годы", "годах", "годов", "лет"):
        assert lemmatize(word) == "год"
    assert lemmatize("хранения") == lemmatize("хранение")


def test_lemmatize_trims_endings_for_unknown_words() -> None:
    """Незнакомые слова усекаются по окончаниям, а не остаются как есть."""
    assert lemmatize("организациями").startswith("организаци")
    assert lemmatize("проверками") == lemmatize("проверка")
    assert lemmatize("123") == "123"


def test_lemma_sequence_and_token_lemmas_skip_punctuation() -> None:
    """Пунктуация не попадает в леммы, порядок слов сохраняется."""
    assert lemma_sequence(["Срок", ",", "хранения"]) == ("срок", "хранен")
    assert token_lemmas("Срок хранения — 5 лет.") == ("срок", "хранен", "5", "год")


def test_canonical_unit_maps_forms_to_kinds() -> None:
    """Единицы приводятся к каноническому виду, включая сокращения и символы."""
    assert canonical_unit("лет") == "год"
    assert canonical_unit("рабочих") == "рабочий день"
    assert canonical_unit("календарных") == "календарный день"
    assert canonical_unit("суток") == "сутки"
    assert canonical_unit("%") == "процент"
    assert canonical_unit("процентов") == "процент"
    assert canonical_unit("привет") is None


def test_numbers_in_text_parses_digits_words_and_mixed() -> None:
    """Числа распознаются во всех трёх записях: цифрами, словами, «5 (пять)»."""
    text = "Срок — двадцать пять лет, не более 30 календарных дней и 5 (пяти) процентов."
    mentions = numbers_in_text(text)
    values = [mention.value for mention in mentions]
    assert values == [25.0, 30.0, 5.0]
    assert mentions[0].unit == "год"
    assert mentions[1].unit == "календарный день"
    assert mentions[2].unit == "процент"
    assert mentions[2].source == "mixed"
    assert has_number_words(text)


def test_numbers_in_text_handles_ordinals_and_units_after() -> None:
    """Порядковые числительные и единица измерения в 2–3 словах от числа."""
    mentions = numbers_in_text("Третий абзац вступает в силу через тридцать календарных дней")
    values = [mention.value for mention in mentions]
    assert 3.0 in values and 30.0 in values
    thirty = next(mention for mention in mentions if mention.value == 30.0)
    assert thirty.unit == "календарный день"


def test_numbers_compatible_respects_units() -> None:
    """Числа считаются совпадающими только при согласованных единицах."""
    five_years = numbers_in_text("5 лет")[0]
    five_days = numbers_in_text("5 рабочих дней")[0]
    five_plain = numbers_in_text("5")[0]
    assert numbers_compatible(five_years, five_plain)
    assert not numbers_compatible(five_years, five_days)
    assert numbers_values("от 3 до 5 лет") == [3.0, 5.0]


def test_iter_windows_covers_tail_with_overlap() -> None:
    """Скользящее окно не теряет хвост и перекрывается (для длинных текстов)."""
    windows = list(iter_windows(10, window=4, step=3))
    assert windows[0] == (0, 4)
    assert windows[-1][1] == 10
    for (_start, end), (next_start, _next_end) in zip(windows, windows[1:]):
        assert next_start < end
