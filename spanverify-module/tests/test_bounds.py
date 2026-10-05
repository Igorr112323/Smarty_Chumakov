"""Точные границы фрагментов: сужение до проверяемого участка.

Проверяется пункт 2.3 промта: фрагмент не должен раздуваться до целого предложения
(ширина ×24 в старом отчёте), а обязан указывать на проверяемое значение.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spanverify.bounds import clause_bounds, narrow_bounds, span_variants  # noqa: E402

ANSWER = (
    "Срок хранения первичных учётных документов составляет 5 лет, "
    "а срок хранения вторичных документов — 30 лет и не подлежит продлению."
)


def _fragment(start: int, end: int) -> str:
    return ANSWER[start:end]


def test_narrow_bounds_keeps_only_value_and_unit() -> None:
    """Если в участке есть число, фрагмент сужается до «число + единица»."""
    start = ANSWER.index("30 лет")
    end = start + len("30 лет и не подлежит продлению.")
    narrowed = narrow_bounds(ANSWER, start, end)
    fragment = _fragment(*narrowed).lower()
    assert "30 лет" in fragment
    assert "не подлежит" not in fragment
    assert len(fragment) <= len("срок хранения ... 30 лет")


def test_clause_bounds_cut_at_comma() -> None:
    """Клауза ограничивается запятой и не захватывает соседнюю."""
    start = ANSWER.index("5 лет")
    end = start + len("5 лет")
    left, right = clause_bounds(ANSWER, start, end)
    fragment = _fragment(left, right)
    assert "5 лет" in fragment
    assert "вторичных" not in fragment


def test_narrow_bounds_without_number_uses_clause() -> None:
    """Без числа границей становится клауза, а не всё предложение."""
    text = "Документ подлежит хранению, порядок уничтожения определяется комиссией организации."
    start = text.index("порядок")
    end = start + len("порядок")
    narrowed = narrow_bounds(text, start, end)
    fragment = text[narrowed[0] : narrowed[1]]
    assert "порядок" in fragment
    assert "подлежит хранению" not in fragment


def test_narrow_bounds_expands_single_word() -> None:
    """Слишком короткий фрагмент доводится до читаемого (не одиночное слово)."""
    text = "Ответственность за хранение возложена на руководителя организации."
    start = text.index("хранение")
    narrowed = narrow_bounds(text, start, start + len("хранение"))
    fragment = text[narrowed[0] : narrowed[1]]
    assert len(fragment.split()) >= 2


def test_span_variants_returns_both_markups() -> None:
    """Одна и та же находка даёт узкую и расширенную границы."""
    start = ANSWER.index("5 лет")
    variants = span_variants(ANSWER, start, start + len("5 лет"))
    narrow_text = _fragment(*variants.narrow)
    expanded_text = _fragment(*variants.expanded)
    assert len(narrow_text) < len(expanded_text)
    assert "5 лет" in narrow_text
    assert expanded_text.startswith("Срок хранения первичных")


def test_narrow_bounds_is_safe_on_boundaries() -> None:
    """Границы не выходят за пределы текста и не теряют участок."""
    text = "5 лет."
    narrowed = narrow_bounds(text, 0, len(text))
    assert 0 <= narrowed[0] <= narrowed[1] <= len(text)
    empty = narrow_bounds("", 0, 0)
    assert empty == (0, 0)
