"""Число прописью в ответе не считается расхождением (дефект C3 реестра).

Смысл проверки: если документ содержит «25 мегабайт», а ответ написан
«двадцать пять мегабайт», ответ корректен. Пока значение числа считалось
по одному токену, «двадцать» превращалось в «20», «пять» — в «5», ни того
ни другого в документе не было, и корректный ответ обвинялся в подмене.

Здесь проверяется не только сама нормализация, но и то, что она доходит
до вердикта: признак может быть посчитан верно, а вердикт — нет.
"""

from __future__ import annotations

from spanverify.core import tokenize_with_offsets
from spanverify.engine import Verifier
from spanverify.features import _group_number_values
from spanverify.normalize import numbers_to_digits, words_to_number

CONTEXT = (
    "Регламент № 1. Для счета-фактуры предельный объём одного вложения "
    "составляет 25 мегабайт. Срок восстановления после сбоя составляет сутки."
)


def test_words_to_number_joins_group() -> None:
    """«двадцать пять» — одно число, а не два."""
    assert words_to_number(["двадцать", "пять"]) == 25


def test_numbers_to_digits_converts_words() -> None:
    """Пропись в тексте превращается в цифры, единица измерения сохраняется."""
    assert numbers_to_digits("двадцать пять мегабайт") == "25 мегабайт"


def test_group_value_covers_both_tokens() -> None:
    """Оба токена группы получают каноническое значение целиковой группы."""
    tokens = tokenize_with_offsets("объём составляет двадцать пять мегабайт")
    values = _group_number_values(tokens)
    assert "25" in values
    # значение относится к обоим токенам прописи, а не к одному из них
    assert values.count("25") == 2


def test_answer_with_number_in_words_is_grounded() -> None:
    """Число прописью не меняет вердикт: как «25», так и «двадцать пять»."""
    verifier = Verifier()
    digits = verifier.verify(
        "Предельный объём вложения составляет 25 мегабайт.",
        CONTEXT,
    )
    words = verifier.verify(
        "Предельный объём вложения составляет двадцать пять мегабайт.",
        CONTEXT,
    )
    assert digits.verdict == "grounded"
    assert words.verdict == "grounded", "число прописью получено как расхождение"


def test_substituted_number_still_detected() -> None:
    """Исправление не должно слепить: подставленное число по-прежнему ловится."""
    result = Verifier().verify(
        "Предельный объём вложения составляет 50 мегабайт.",
        CONTEXT,
    )
    assert result.verdict == "likely_hallucination"
    assert result.spans, "подмена обнаружена, но границы фрагмента не выданы"


def test_words_do_not_create_spans() -> None:
    """Корректная запись прописью не порождает спорных фрагментов."""
    result = Verifier().verify(
        "Предельный объём вложения составляет двадцать пять мегабайт.",
        CONTEXT,
    )
    assert result.spans == []
