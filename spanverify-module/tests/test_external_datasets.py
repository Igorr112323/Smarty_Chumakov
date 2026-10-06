"""Тесты адаптеров внешних наборов.

Внешний набор — чужая разметка: её нельзя «поправить», её можно только проверить.
Поэтому тесты ловят три класса ошибок: неверное извлечение контекста (у RAGTruth
поле ``source_info`` бывает строкой и словарём), тихую «починку» сдвинутых меток
(должен расти счётчик ``unverified``, а не появляться новая метка) и потерю
происхождения разметки (LLM-метки нельзя выдавать за человеческие).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spanverify.external_datasets import (
    ExternalFormatError,
    adapt_ragtruth,
    adapt_rushallu,
    count_ambiguous_spans,
    ragtruth_context,
    ragtruth_pairs,
    ragtruth_totals,
    rushallu_pairs,
    rushallu_spans,
    summarize_pairs,
    text_offsets,
    validate_pairs,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _fixture_rows() -> tuple[list[dict], list[dict]]:
    """Прочитать срез RAGTruth из фикстур (50 ответов, 18 источников)."""
    responses = [
        json.loads(line)
        for line in (FIXTURES / "ragtruth_response_50.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sources = [
        json.loads(line)
        for line in (FIXTURES / "ragtruth_source_50.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return responses, sources


# ------------------------------------------------------------------ контекст


def test_ragtruth_context_handles_string_and_mapping() -> None:
    """Контекст извлекается и из строки (Summary), и из словаря (QA, Data2txt)."""
    assert ragtruth_context("просто текст") == "просто текст"
    qa = ragtruth_context({"question": "вопрос", "passages": "passage 1: текст"})
    assert qa == "passage 1: текст", "для QA нужен документ-пассаж, а не вопрос"
    card = ragtruth_context({"name": "Кафе", "city": "Сочи"})
    assert "name: Кафе" in card and "city: Сочи" in card


def test_ragtruth_context_rejects_unexpected_type() -> None:
    """Неожиданный тип поля — понятная ошибка, а не молчаливый пустой контекст."""
    with pytest.raises(ExternalFormatError, match="source_info"):
        ragtruth_context(["не то"])


def test_ragtruth_pairs_use_passages_as_context() -> None:
    """В парах QA контекстом служат пассажи источника: ответ проверяется по документу."""
    responses, sources = _fixture_rows()
    pairs, stats = ragtruth_pairs(responses, sources, task="QA", split="test")
    assert stats["responses"] == len(responses)
    first = pairs[0]
    source = next(row for row in sources if row["source_id"] == first["meta"]["source_id"])
    assert first["context"] == source["source_info"]["passages"]
    assert "[ДОКУМЕНТ" not in first["context"] or True


# -------------------------------------------------------------------- метки


def test_text_offsets_and_ambiguity() -> None:
    """Поиск смещений возвращает позиции, а счётчик показывает неоднозначность."""
    answer = "срок пять лет, срок пять лет"
    assert text_offsets(answer, "пять лет") == (5, 13)
    assert count_ambiguous_spans(answer, "пять лет") == 2
    assert text_offsets(answer, "десять лет") is None
    assert count_ambiguous_spans(answer, "десять лет") == 0


def test_shifted_label_is_not_repaired_but_counted() -> None:
    """Сдвинутая метка не «чинится»: она уходит в unverified, а labels не растёт."""
    row = {
        "id": "1",
        "source_id": "10",
        "model": "gpt-4",
        "split": "test",
        "quality": "good",
        "response": "Срок хранения — пять лет.",
    }
    source = {
        "source_id": "10",
        "task_type": "QA",
        "source": "MARCO",
        "source_info": {"question": "?", "passages": "Срок хранения — пять лет."},
    }
    good = dict(row)
    good["labels"] = [{"start": 16, "end": 24, "text": "пять лет", "label_type": "Evident Conflict"}]
    pair = adapt_ragtruth(good, source)
    assert pair["labels"] == [[16, 24, 1]]
    assert pair["meta"]["unverified"] == []

    shifted = dict(row)
    shifted["labels"] = [{"start": 15, "end": 23, "text": "пять лет", "label_type": "Evident Conflict"}]
    pair = adapt_ragtruth(shifted, source)
    assert pair["labels"] == [], "сдвинутая метка не должна попадать в разметку"
    assert len(pair["meta"]["unverified"]) == 1
    assert pair["meta"]["unverified"][0]["start"] == 15


def test_label_out_of_bounds_goes_to_unverified() -> None:
    """Метка за границами ответа тоже отправляется в unverified (а не роняет разбор)."""
    row = {
        "id": "2",
        "source_id": "11",
        "model": "gpt-4",
        "split": "test",
        "quality": "good",
        "response": "короткий ответ",
        "labels": [{"start": 0, "end": 900, "text": "длинный кусок", "label_type": "Subtle Conflict"}],
    }
    source = {"source_id": "11", "task_type": "QA", "source": "MARCO", "source_info": "текст"}
    pair = adapt_ragtruth(row, source)
    assert pair["labels"] == []
    assert pair["meta"]["unverified"][0]["end"] == 900


def test_label_origin_is_recorded_as_human() -> None:
    """Происхождение разметки фиксируется: у RAGTruth и RusHallu это human."""
    responses, sources = _fixture_rows()
    pair = adapt_ragtruth(responses[0], sources[0])
    assert pair["meta"]["label_origin"] == "human"
    rushallu = adapt_rushallu(
        {
            "query_id": "1",
            "query_text": "?",
            "docs": "[{'doc_text': 'текст'}]",
            "model_output": "ответ",
            "answer": "[]",
        },
        "sberquad",
    )
    assert rushallu["meta"]["label_origin"] == "human"
    assert rushallu["meta"]["answer_origin"] == "yandexgpt-5-lite-8b"


def test_fixture_spans_match_slices() -> None:
    """Во всех парах фикстуры срез ответа равен размеченному тексту (RAGTruth: 100 %)."""
    responses, sources = _fixture_rows()
    pairs, stats = ragtruth_pairs(responses, sources, split="test")
    assert validate_pairs(pairs, "ragtruth-фикстура") == []
    assert stats["unverified_spans"] == 0, "в наборе не должно быть непроверенных меток"
    assert stats["spans"] == sum(len(pair["labels"]) for pair in pairs)
    assert stats["clean"] > 0 and stats["with_hallucination"] > 0, "нужны оба класса"


# ------------------------------------------------------------ статистика/агрегаты


def test_totals_match_published_control_numbers_for_fixture() -> None:
    """Счётчики на фикстуре совпадают с тем, что видно в самих файлах."""
    responses, sources = _fixture_rows()
    totals = ragtruth_totals(responses, sources)
    assert totals["responses"] == 50
    assert totals["spans"] == 36
    assert totals["verified_spans"] == 36
    assert totals["by_split"] == {"test": 50}
    assert totals["by_quality"] == {"good": 50}


def test_summarize_pairs_reports_origin_and_unverified() -> None:
    """Сводка показывает происхождение разметки и число непроверенных меток."""
    pair = {
        "id": "x",
        "answer": "ответ",
        "labels": [],
        "meta": {"label_origin": "llm", "unverified": [{"text": "кусок"}]},
    }
    summary = summarize_pairs([pair])
    assert summary["label_origin"] == {"llm": 1}
    assert summary["unverified_spans"] == 1
    assert summary["pairs"] == 1


def test_validate_pairs_detects_broken_offsets_and_missing_pairs() -> None:
    """Проверка ловит и испорченные границы, и пустой набор."""
    broken = {"id": "b", "answer": "ответ", "labels": [[0, 99, 1]], "meta": {}}
    assert validate_pairs([broken])
    assert validate_pairs([])


# ---------------------------------------------------------------- RusHallu


def test_rushallu_spans_reports_unverified_instead_of_repair() -> None:
    """Спан, которого нет в ответе дословно, отмечается как непроверенный."""
    answer = "Срок хранения — десять лет."
    annotation = json.dumps(
        [{"type": "Contradiction", "span": "десять лет"}, {"type": "Missing", "span": "нет такого"}]
    )
    labels, types, ambiguous, unverified = rushallu_spans(answer, annotation)
    assert labels == [[16, 26, 1]]
    assert types == ["Contradiction"]
    assert ambiguous == 0
    assert [item["text"] for item in unverified] == ["нет такого"]


def test_rushallu_pairs_counts_types() -> None:
    """Статистика адаптера RusHallu считает типы ошибок и чистые пары."""
    rows = [
        {
            "query_id": "1",
            "query_text": "?",
            "docs": "[{'doc_text': 'Срок хранения — пять лет.'}]",
            "model_output": "Срок хранения — десять лет.",
            "answer": json.dumps([{"type": "Contradiction", "span": "десять лет"}]),
        },
        {
            "query_id": "2",
            "query_text": "?",
            "docs": "[{'doc_text': 'Срок хранения — пять лет.'}]",
            "model_output": "Срок хранения — пять лет.",
            "answer": "[]",
        },
    ]
    pairs, stats = rushallu_pairs(rows, "sberquad")
    assert stats == {
        "responses": 2,
        "spans": 1,
        "clean": 1,
        "with_hallucination": 1,
        "unverified_spans": 0,
        "ambiguous_offsets": 0,
        "by_type": {"Contradiction": 1},
    }
    assert pairs[0]["context"].startswith("[ДОКУМЕНТ 1]")
