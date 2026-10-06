"""Тесты покрытия фактов документа ответом (исправление B2/B3: missing и partial).

Смысл проверок: конвейер раньше отвечал только на вопрос «есть ли в ответе
лишнее», поэтому пропуск сведения не ловился вообще (``missing.recall = 0.0``).
Эти тесты фиксируют, что пропуск и отброшенное условие теперь обнаруживаются, а
на корректном ответе замечаний не появляется.
"""

from __future__ import annotations

from spanverify.coverage import (
    STATUS_DISTORTED,
    STATUS_IRRELEVANT,
    STATUS_MENTIONED,
    STATUS_OMITTED,
    STATUS_PARTIAL,
    cover_facts,
    extract_facts,
    missing_coverages,
    partial_coverages,
)

CONTEXT = (
    "Регламент: срок хранения первичных документов составляет 10 лет. "
    "Срок хранения вторичных документов составляет 5 лет."
)


def _statuses(answer: str, context: str = CONTEXT) -> list[str]:
    return [item.status for item in cover_facts(answer, context)]


def test_extract_facts_finds_values() -> None:
    """Из документа извлекаются факты со значениями."""
    facts = extract_facts(CONTEXT)
    values = [fact.value for fact in facts]
    assert any("10" in value for value in values), values
    assert any("5" in value for value in values), values


def test_fact_has_subject() -> None:
    """У факта есть предмет: иначе нельзя понять, про что именно ответ молчит."""
    facts = extract_facts(CONTEXT)
    assert any("хранения" in fact.subject for fact in facts), [fact.subject for fact in facts]


def test_mentioned_when_value_present() -> None:
    """Если значение на месте — замечаний нет, а чужой факт не упоминается вовсе."""
    statuses = _statuses("Срок хранения первичных документов составляет 10 лет.")
    assert statuses[0] == STATUS_MENTIONED
    assert STATUS_OMITTED not in statuses, statuses
    assert STATUS_DISTORTED not in statuses, statuses


def test_number_in_words_is_mentioned() -> None:
    """Число прописью не считается пропуском (связка с нормализацией)."""
    statuses = _statuses("Срок хранения первичных документов составляет десять лет.")
    assert statuses[0] == STATUS_MENTIONED


def test_omitted_when_value_dropped() -> None:
    """Пропуск значения — это missing, а не «всё подтверждено»."""
    statuses = _statuses("Срок хранения первичных документов составляет.")
    assert STATUS_OMITTED in statuses


def test_omitted_is_detected_by_missing_coverages() -> None:
    """Пропущенные факты выделяются отдельным списком (для метрики missing)."""
    items = missing_coverages(cover_facts("Срок хранения первичных документов составляет.", CONTEXT))
    assert len(items) >= 1
    assert all(item.status == STATUS_OMITTED for item in items)


def test_distorted_when_other_number_used() -> None:
    """Чужое число — подмена, а не пропуск: иначе missing и contradiction путаются."""
    statuses = _statuses("Срок хранения первичных документов составляет 3 года.")
    assert STATUS_DISTORTED in statuses


def test_partial_when_condition_dropped() -> None:
    """Отброшенное условие даёт partial."""
    context = "Регламент: срок хранения первичных документов составляет 10 лет, " "если документ не передан в архив."
    answer = "Срок хранения первичных документов составляет 10 лет."
    statuses = [item.status for item in cover_facts(answer, context)]
    assert STATUS_PARTIAL in statuses


def test_partial_is_detected_by_partial_coverages() -> None:
    """Отброшенные условия выделяются отдельным списком (для метрики partial)."""
    context = "Регламент: срок хранения первичных документов составляет 10 лет, " "если документ не передан в архив."
    items = partial_coverages(cover_facts("Срок хранения первичных документов составляет 10 лет.", context))
    assert len(items) >= 1
    assert all(item.status == STATUS_PARTIAL for item in items)


def test_irrelevant_when_answer_about_other_thing() -> None:
    """Ответ про другое не получает замечаний по чужим фактам (защита от ложных срабатываний)."""
    statuses = _statuses("Кадровый учёт ведётся в отделе кадров.")
    assert statuses == [STATUS_IRRELEVANT] * 2


def test_mentioning_one_object_does_not_blame_another() -> None:
    """Ответ про первичные документы не требует упоминания вторичных.

    Регрессионная проверка: при мягком пороге совпадения предмета механизм
    считал пропуском то, что ответ просто не про этот объект, и корректная
    пара получала ложное замечание.
    """
    statuses = _statuses("Срок хранения первичных документов составляет 10 лет.")
    assert STATUS_IRRELEVANT in statuses, statuses


def test_empty_input_gives_no_coverages() -> None:
    """Пустой ответ или пустой документ не порождают замечаний."""
    assert cover_facts("", CONTEXT) == []
    assert cover_facts("Ответ.", "") == []


def test_coverage_points_into_answer() -> None:
    """У найденного пропуска есть смещения в ответе: иначе фрагмент нельзя показать."""
    answer = "Срок хранения первичных документов составляет."
    items = missing_coverages(cover_facts(answer, CONTEXT))
    assert items, "пропуск должен быть найден"
    item = items[0]
    assert 0 <= item.answer_start < item.answer_end <= len(answer)


def test_extract_facts_respects_min_words() -> None:
    """Короткие служебные предложения не становятся фактами."""
    facts = extract_facts("Утвердить. Срок 10 лет для документов организации.", min_words=5)
    assert all(len(fact.sentence.split()) >= 5 for fact in facts)
