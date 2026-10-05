"""Атомарные факты документа и покрытие их ответом (пункт 2.4 реестра).

До появления этого механизма тип расхождения ``missing`` не обнаруживался
вообще (recall 0,0), а ``partial`` — 0,2667. Тесты фиксируют, что оба типа
теперь находятся и что чистый ответ при этом не помечается.
"""

from __future__ import annotations

from spanverify.facts import (
    ALL_KINDS,
    STATUS_OK,
    STATUS_OMITTED,
    STATUS_PARTIAL,
    coverage,
    coverage_spans,
    extract_facts,
    fact_kinds,
    split_document_sentences,
)

DOCUMENT = (
    "Регламент учёта № 2. Настоящий документ определяет правила учёта.\n\n"
    "Для счета-фактуры срок хранения составляет пять если контрагент подтвердил продление.\n\n"
    "Для счета-фактуры срок ответа на запрос контрагента составляет десять рабочих дней.\n\n"
    "Для акты выполненных работ срок хранения составляет семь если срок не продлён решением комиссии.\n\n"
    "Передача документов в архив не допускается без описи вложений.\n"
)


def test_split_document_sentences_handles_paragraphs() -> None:
    """Абзац — тоже граница предложения: иначе весь акт был бы одним фактом."""
    bounds = split_document_sentences(DOCUMENT)
    assert len(bounds) >= 5
    texts = [DOCUMENT[s:e].strip() for s, e in bounds]
    assert any(text.startswith("Для счета-фактуры срок хранения") for text in texts)


def test_split_document_sentences_keeps_abbreviations() -> None:
    """«ст. 5» и «п. 2» не разрывают предложение пополам."""
    text = "Согласно ст. 5 настоящего Положения срок составляет 30 дней."
    assert len(split_document_sentences(text)) == 1


def test_extract_facts_finds_subject_feature_value_condition() -> None:
    """Из предложения документа извлекаются все четыре части факта."""
    facts = extract_facts(DOCUMENT)
    target = [f for f in facts if "счета-фактуры" in f.subject and "хранения" in f.feature]
    assert target, [f.subject for f in facts]
    fact = target[0]
    assert fact.value.startswith("пять")
    assert fact.condition == "если контрагент подтвердил продление"
    assert fact.measures == ("5|-",)
    assert DOCUMENT[fact.start : fact.end].strip().startswith("Для счета-фактуры")


def test_extract_facts_detects_prohibition() -> None:
    """Запрет — отдельный вид факта (п. 3.2 задания)."""
    facts = extract_facts(DOCUMENT)
    kinds = {fact.kind for fact in facts}
    assert "prohibition" in kinds


def test_fact_kinds_covers_all_keys() -> None:
    """Сводка по видам фактов содержит все виды, даже нулевые."""
    summary = fact_kinds(extract_facts(DOCUMENT))
    assert set(summary) == set(ALL_KINDS)
    assert sum(summary.values()) == len(extract_facts(DOCUMENT))


def test_coverage_marks_faithful_answer_as_covered() -> None:
    """Полный ответ со значением и условием считается покрывающим факт."""
    answer = "Срок хранения для счета-фактуры — пять, если контрагент подтвердил продление."
    statuses = {item.status for item in coverage(answer, extract_facts(DOCUMENT)) if item.status != "absent"}
    assert statuses == {STATUS_OK}


def test_coverage_detects_missing_value() -> None:
    """Значение заменено формулировкой-заглушкой → пропуск сведения."""
    answer = "Срок хранения для счета-фактуры установлен."
    found = [item for item in coverage(answer, extract_facts(DOCUMENT)) if item.status == STATUS_OMITTED]
    assert found
    spans = coverage_spans(answer, extract_facts(DOCUMENT))
    assert [(span["kind"], answer[span["start"] : span["end"]]) for span in spans] == [("missing", "установлен")]


def test_coverage_detects_dropped_condition() -> None:
    """Значение есть, условия документа нет → усечение (partial)."""
    answer = "Срок хранения для счета-фактуры — пять."
    found = [item for item in coverage(answer, extract_facts(DOCUMENT)) if item.status == STATUS_PARTIAL]
    assert found
    spans = coverage_spans(answer, extract_facts(DOCUMENT))
    assert [(span["kind"], answer[span["start"] : span["end"]]) for span in spans] == [("partial", "пять")]


def test_coverage_detects_oversight() -> None:
    """Предмет назван, значения потеряны целиком → oversight."""
    answer = "Для акты выполненных работ правила учёта определены, значения приведены в регламенте."
    spans = coverage_spans(answer, extract_facts(DOCUMENT))
    assert [(span["kind"], answer[span["start"] : span["end"]]) for span in spans] == [
        ("oversight", "значения приведены в регламенте")
    ]


def test_other_subject_does_not_trigger_missing() -> None:
    """Ответ про другой объект не сверяется с фактом «похожего» признака.

    Это источник ложных замечаний: «срок хранения» есть у нескольких объектов,
    и совпадения одного лишь признака недостаточно.
    """
    answer = "Срок хранения для акты выполненных работ — семь, если срок не продлён решением комиссии."
    assert coverage_spans(answer, extract_facts(DOCUMENT)) == []


def test_numbers_in_words_are_matched() -> None:
    """Ответ цифрами против документа прописью — расхождением не считается."""
    answer = "Срок хранения для счета-фактуры — 5, если контрагент подтвердил продление."
    assert coverage_spans(answer, extract_facts(DOCUMENT)) == []


def test_coverage_spans_are_narrow() -> None:
    """Границы фрагмента — по значению/клаузе, а не по всему предложению."""
    answer = "Срок хранения для счета-фактуры — пять."
    span = coverage_spans(answer, extract_facts(DOCUMENT))[0]
    assert span["end"] - span["start"] <= len("пять") + 2
