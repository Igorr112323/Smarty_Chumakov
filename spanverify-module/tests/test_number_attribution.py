"""Дефект D: число ответа привязывается к своему объекту, а не «находится где-то».

Проверяющий воспроизвёл на релизе v1.2.0: контекст с двумя сроками (первичные —
пять лет, вторичные — десять лет), ответ про **первичные** говорит «десять лет» —
и получал «ОПОРА НА КОНТЕКСТ ЕСТЬ», потому что слово «десять» действительно есть
в контексте, но в предложении про другой объект.

Эти тесты падают на коде v1.2.0 (где привязки нет) и проходят после исправления.
"""

from __future__ import annotations

import pytest

from spanverify.engine import Verifier


def _attribution(answer: str, context: str):
    """Отчёт привязки чисел (импорт внутри функции: на v1.2.0 модуль его не имеет)."""
    from spanverify.features import number_attribution

    return number_attribution(answer, context)


def _measurements(context: str):
    """Измерения контекста (импорт внутри функции по той же причине)."""
    from spanverify.features import context_measurements

    return context_measurements(context)


# Контекст с двумя измерениями: у каждого объекта свой срок.
CONTEXT_TWO = (
    "Согласно регламенту, срок хранения первичных документов составляет пять лет. "
    "Срок хранения вторичных документов составляет десять лет."
)
ANSWER_BORROWED = "Срок хранения первичных документов составляет десять лет."
ANSWER_OWN = "Срок хранения первичных документов составляет пять лет."

# Контекст с тремя измерениями: подмена берётся из первого, а ответ — про второй.
CONTEXT_THREE = (
    "Срок хранения первичных документов составляет пять лет. "
    "Срок хранения вторичных документов составляет десять лет. "
    "Срок хранения третичных документов составляет пятнадцать лет."
)
ANSWER_ABOUT_SECOND = "Срок хранения вторичных документов составляет пять лет."


def test_measurements_split_by_subject() -> None:
    """В контексте с двумя объектами находятся ровно два измерения с разными числами."""
    measurements = _measurements(CONTEXT_TWO)
    values = {item.value for item in measurements}
    assert values == {"5", "10"}, [item.as_dict() for item in measurements]
    subjects = [" ".join(item.subject) for item in measurements]
    assert any("первичных" in subject for subject in subjects)
    assert any("вторичных" in subject for subject in subjects)


def test_number_from_other_object_is_not_grounded() -> None:
    """Ответ берёт число из чужого объекта — вердикт не «опора есть», есть фрагмент."""
    result = Verifier(mode="demo").verify(ANSWER_BORROWED, CONTEXT_TWO)
    assert result.verdict != "grounded", result.verdict
    assert result.spans, "подменённое число должно дать спорный фрагмент"
    assert any("десять" in span.text for span in result.spans), [span.text for span in result.spans]


def test_number_from_own_object_is_grounded() -> None:
    """Контроль ложных срабатываний: число из своего объекта — строго подтверждено."""
    result = Verifier(mode="demo").verify(ANSWER_OWN, CONTEXT_TWO)
    assert result.verdict == "grounded", result.verdict
    assert result.spans == [], [span.text for span in result.spans]


def test_attribution_finds_correct_subject_among_three() -> None:
    """Три объекта в контексте: для ответа про второй объект подобран именно он."""
    report = _attribution(ANSWER_ABOUT_SECOND, CONTEXT_THREE)
    assert len(report) == 1, report
    item = report[0]
    assert item["value"] == "5"
    assert item["matched_value"] == "10", item
    assert "втор" in " ".join(item["matched_subject"]), item
    assert item["ok"] is False


def test_substitution_of_sentence_in_three_facts_is_flagged() -> None:
    """Подмена в середине трёхфактного документа находится, а не «тонет» в контексте."""
    result = Verifier(mode="demo").verify(ANSWER_ABOUT_SECOND, CONTEXT_THREE)
    assert result.verdict != "grounded", result.verdict
    assert any("пять" in span.text for span in result.spans), [span.text for span in result.spans]


def test_attribution_requires_two_measurements() -> None:
    """Один факт в контексте — привязка не срабатывает: решений нет и вердикт подтверждён."""
    context = "Срок хранения первичных документов составляет пять лет."
    answer = "Срок хранения первичных документов составляет пять лет."
    assert _attribution(answer, context) == []
    assert Verifier(mode="demo").verify(answer, context).verdict == "grounded"


def test_attribution_is_quiet_without_subject() -> None:
    """Если у числа в ответе нет слов-субъекта — привязка молчит (не обвиняет вслепую)."""
    context = "Пять лет. Десять лет."
    report = _attribution("Пять.", context)
    assert report == [] or report[0]["ok"] is True


@pytest.mark.parametrize("mode", ["demo"])
def test_demo_corpus_metrics_do_not_degrade(mode: str) -> None:
    """Регресс: после привязки метрики демо-корпуса не падают ниже допуска 0.94."""
    from spanverify.dataset import read_pairs
    from spanverify.train import train

    pairs = list(read_pairs("data/demo_pairs.jsonl"))
    report = train(pairs, mode=mode, seed=42, dataset_name="data/demo_pairs.jsonl")
    tokens = report.validation["end_to_end"]["tokens"]
    assert tokens["f1"] >= 0.94, tokens
    assert tokens["fpr"] <= 0.10, tokens
