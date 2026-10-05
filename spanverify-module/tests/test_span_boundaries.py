"""Тесты точности границ фрагментов (исправление B1).

Раньше найденный токен расширялся до границ всего предложения: фрагмент был
в ~24 раза шире истинной ошибки, строгий span-F1 держался на 0,098–0,283.
Здесь фиксируется, что фрагмент ограничивается клаузой, в которой стоит
спорное значение, и не захватывает соседние части предложения.
"""

from __future__ import annotations

from spanverify.engine import Verifier, _expand_to_sentence, _shrink_to_clause


def test_shrink_keeps_flagged_token() -> None:
    """Найденный токен обязан остаться внутри фрагмента при любом сужении."""
    answer = "Первая часть, вторая часть с числом 5 лет, третья часть."
    start = answer.index("5")
    end = start + len("5 лет")
    left, right = _shrink_to_clause(answer, start, end, start, end)
    assert left <= start, (left, start)
    assert right >= end, (right, end)


def test_shrink_does_not_capture_neighbour_clauses() -> None:
    """Соседние клаузы не попадают в фрагмент."""
    answer = "Первая часть, вторая часть с числом 5 лет, третья часть."
    start = answer.index("5")
    end = start + len("5 лет")
    left, right = _shrink_to_clause(answer, start, end, start, end)
    fragment = answer[left:right]
    assert "Первая" not in fragment, fragment
    assert "третья" not in fragment, fragment
    assert "5 лет" in fragment, fragment


def test_shrink_is_narrower_than_sentence() -> None:
    """Сужение даёт фрагмент не шире предложения."""
    answer = "Первая часть, вторая часть с числом 5 лет, третья часть."
    start = answer.index("5")
    end = start + len("5 лет")
    sent_left, sent_right = _expand_to_sentence(answer, start, end)
    left, right = _shrink_to_clause(answer, start, end, start, end)
    assert (right - left) <= (sent_right - sent_left)


def test_shrink_single_clause_keeps_sentence() -> None:
    """Если клауза одна, фрагмент остаётся предложением (не обрезается до слова)."""
    answer = "Срок хранения документов составляет 5 лет."
    start = answer.index("5")
    end = start + len("5 лет")
    left, right = _shrink_to_clause(answer, start, end, start, end)
    assert answer[left:right].strip() == "Срок хранения документов составляет 5 лет."


def test_span_on_real_substitution_is_local() -> None:
    """Сквозная проверка: подмена числа даёт локальный фрагмент, а не весь абзац."""
    context = "Регламент: срок хранения первичных документов составляет 10 лет."
    answer = (
        "Общий порядок установлен регламентом, срок хранения первичных документов "
        "составляет 3 года, порядок передачи определяется отдельно."
    )
    result = Verifier().verify(answer, context)
    assert result.spans, "подмена числа должна быть найдена"
    widths = [span.end - span.start for span in result.spans]
    assert min(widths) < len(answer) // 2, (widths, answer)


def test_span_boundaries_are_valid() -> None:
    """Границы фрагмента всегда лежат внутри ответа и не перевёрнуты."""
    context = "Регламент: срок хранения первичных документов составляет 10 лет."
    answer = "Срок хранения первичных документов составляет 3 года, а вторичных — 5 лет."
    result = Verifier().verify(answer, context)
    for span in result.spans:
        assert 0 <= span.start < span.end <= len(answer), (span.start, span.end, len(answer))
        assert span.text == answer[span.start : span.end].strip() or span.text in answer


def test_missing_fact_gives_span_when_coverage_enabled() -> None:
    """Пропуск значения ловится механизмом покрытия, если он включён явно.

    Механизм выключен по умолчанию: на отложенной части корпуса A он давал
    ложных замечаний больше, чем находил пропусков (см. комментарий в
    ``Verifier.__init__``). Тест удерживает работоспособность механизма, чтобы
    доработка не начиналась с нуля.
    """
    context = "Регламент: срок хранения первичных документов составляет 10 лет."
    answer = "Срок хранения первичных документов составляет."
    result = Verifier(coverage=True).verify(answer, context)
    assert result.spans, "пропуск значения должен быть найден"
    assert any(span.label == "doubtful" for span in result.spans)
