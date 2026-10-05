"""Точные границы фрагментов: сужение найденного места до проверяемого участка.

Проблема, которую решает модуль (пункт 2.3 промта): раньше найденный токен
расширялся до границ предложения, из-за чего ширина фрагмента превышала разметку в
24 раза (`reports/METRICS.json → demo/in_corpus/spans/width_ratio = 23.63`), а
строгий span-F1 при IoU ≥ 0.5 падал до 0.098–0.283.

Правила сужения, по которым теперь строятся границы:

1. **Число и единица измерения** — если в найденном участке есть число, фрагмент
   сужается ровно до «число + единица» («5 лет», «не позднее 30 календарных дней»
   сокращается до «30 календарных дней»), потому что именно значение проверяется.
2. **Клауза** — иначе фрагмент ограничивается клаузой (по запятым, тире, точке с
   запятой и союзам «и», «или», «либо», «но», «а»), а не целым предложением.
3. **Минимальная длина** — слишком короткие куски (одно слово, предлог, частица)
   расширяются до ближайших содержательных слов, чтобы фрагмент читался.

Дополнительно модуль даёт две разметки для метрик (узкую и расширенную): считается,
что «правильный» фрагмент — тот, что несёт проверяемое значение; расширенный —
предложение целиком. Оба числа пишутся в отчёт, а не заменяют друг друга.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .core import Token, split_sentences
from .normalize import NumberMention, numbers_in_text

__all__ = [
    "SpanVariants",
    "clause_bounds",
    "narrow_bounds",
    "span_variants",
]

# Минимальное число слов в узком фрагменте и максимальная длина (символы).
MIN_WORDS = 2
MAX_NARROW_CHARS = 160

_CLAUSE_BREAKS = (",", ";", "—", " - ", ":")
_CONJUNCTIONS = (" и ", " или ", " либо ", " но ", " а ", " что ", " который ", " которая ", " которые ")


@dataclass(frozen=True)
class SpanVariants:
    """Две границы одного и того же замечания: узкая и расширенная."""

    narrow: tuple[int, int]
    expanded: tuple[int, int]

    def as_dict(self) -> dict[str, list[int]]:
        return {"narrow": list(self.narrow), "expanded": list(self.expanded)}


def _word_spans(text: str, start: int, end: int) -> list[tuple[str, int, int]]:
    """Слова внутри отрезка: ``(текст, начало, конец)``."""
    words: list[tuple[str, int, int]] = []
    for match in _iter_words(text, start, end):
        words.append(match)
    return words


def _iter_words(text: str, start: int, end: int):
    import re

    for match in re.finditer(r"[A-Za-zА-Яа-яЁё0-9]+(?:-[A-Za-zА-Яа-яЁё0-9]+)*", text[start:end]):
        yield match.group(0), start + match.start(), start + match.end()


def clause_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Границы клаузы, содержащей отрезок ``[start, end)``."""
    if not text:
        return start, end
    # Ищем ближайшую границу слева и справа, не выходя за предложение.
    sentence_start, sentence_end = start, end
    for s_start, s_end in split_sentences(text):
        if s_start <= start < s_end:
            sentence_start, sentence_end = s_start, s_end
            break
    left = sentence_start
    for marker in _CLAUSE_BREAKS:
        index = text.rfind(marker, sentence_start, start)
        if index >= 0:
            left = max(left, index + len(marker))
    right = sentence_end
    for marker in _CLAUSE_BREAKS:
        index = text.find(marker, end, sentence_end)
        if index >= 0:
            right = min(right, index)
    for conjunction in _CONJUNCTIONS:
        index = text.rfind(conjunction, sentence_start, start)
        if index >= 0:
            left = max(left, index + 1)
        index = text.find(conjunction, end, sentence_end)
        if index >= 0:
            right = min(right, index)
    # Обрезаем ведущие и замыкающие пробелы и знаки.
    while left < right and text[left] in " \t\n,;:—-\"'«»":
        left += 1
    while right > left and text[right - 1] in " \t\n,;:—-\"'«»":
        right -= 1
    if right <= left:
        return start, end
    return left, right


def _number_mention_in(text: str, start: int, end: int) -> NumberMention | None:
    """Первое число (с единицей) внутри отрезка — по нему сужается фрагмент."""
    fragment = text[start:end]
    for mention in numbers_in_text(fragment):
        absolute_start = start + mention.start
        absolute_end = start + mention.end
        if mention.unit:
            # Единица измерения уже распознана парсером чисел (normalize.numbers_in_text):
            # здесь она не пересчитывается повторно, чтобы не расходиться с ним.
            return NumberMention(
                value=mention.value,
                text=text[absolute_start:absolute_end],
                start=absolute_start,
                end=absolute_end,
                unit=mention.unit,
                source=mention.source,
            )
        return NumberMention(
            value=mention.value,
            text=text[absolute_start:absolute_end],
            start=absolute_start,
            end=absolute_end,
            unit=mention.unit,
            source=mention.source,
        )
    return None


def _extend_with_unit(text: str, start: int, end: int) -> int:
    """Расширить конец числа на следующее слово, если это единица измерения."""
    from .normalize import canonical_unit

    for word, word_start, word_end in _iter_words(text, end, min(len(text), end + 32)):
        if canonical_unit(word):
            return word_end
        if word_start > end + 2:
            break
        break
    return end


def _expand_to_content_words(text: str, start: int, end: int) -> tuple[int, int]:
    """Довести отрезок до соседних содержательных слов (минимум ``MIN_WORDS``)."""
    words = _word_spans(text, 0, len(text))
    inside = [word for word in words if word[1] >= start and word[2] <= end]
    if len(inside) >= MIN_WORDS:
        return start, end
    # Расширяем вправо, затем влево, пока не наберём минимум.
    left, right = start, end
    for _ in range(MIN_WORDS * 3):
        right_word = next((word for word in words if word[1] >= right), None)
        left_word = next((word for word in reversed(words) if word[2] <= left), None)
        grew = False
        if len([word for word in words if word[1] >= left and word[2] <= right]) < MIN_WORDS:
            if right_word is not None and right_word[1] - right < 40:
                right = right_word[2]
                grew = True
            elif left_word is not None and left - left_word[2] < 40:
                left = left_word[1]
                grew = True
        if not grew:
            break
    return left, right


def narrow_bounds(
    text: str,
    start: int,
    end: int,
    *,
    prefer_number: bool = True,
    max_chars: int = MAX_NARROW_CHARS,
) -> tuple[int, int]:
    """Сузить отрезок до проверяемого участка (число с единицей или клауза).

    ``prefer_number=False`` оставляет только правило клаузы — используется в
    тестах и там, где значение не проверяется по числу.
    """
    if not text:
        return start, end
    start = max(0, min(start, len(text)))
    end = max(start, min(end, len(text)))
    clause_start, clause_end = clause_bounds(text, start, end)
    chosen = (clause_start, clause_end)
    if prefer_number:
        mention = _number_mention_in(text, clause_start, clause_end)
        if mention is not None:
            number_end = mention.end
            if mention.unit:
                number_end = _extend_with_unit(text, mention.start, mention.end)
            candidate = (mention.start, number_end)
            if candidate[1] - candidate[0] <= max_chars:
                chosen = candidate
    narrowed = _expand_to_content_words(text, *chosen)
    if narrowed[1] - narrowed[0] > max_chars:
        # Слишком длинная клауза: берём окно вокруг центра проверяемого участка.
        center = (start + end) // 2
        half = max_chars // 2
        narrowed = (max(0, center - half), min(len(text), center + half))
    return narrowed


def span_variants(
    text: str,
    start: int,
    end: int,
    *,
    tokens: Sequence[Token] | None = None,
) -> SpanVariants:
    """Обе границы замечания: узкая (проверяемый участок) и расширенная (предложение)."""
    expanded = (start, end)
    for sentence_start, sentence_end in split_sentences(text):
        if sentence_start <= start < sentence_end:
            expanded = (sentence_start, max(sentence_end, end))
            break
    narrow = narrow_bounds(text, start, end)
    return SpanVariants(narrow=narrow, expanded=expanded)
