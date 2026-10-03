"""Токенизация и разбиение на предложения (без внешних зависимостей).

Токенизатор возвращает *непрерывное* покрытие исходного текста: любой
символ принадлежит ровно одному токену. Это позволяет считать долю
участия ИИ в символах и рисовать разметку без потерь на пробелах.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

# Слово: буквы (кириллица/латиница), цифры, дефисы и апострофы внутри.
WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+(?:[-'’][A-Za-zА-Яа-яЁё0-9]+)*")

# Границы предложений: знак конца + пробел + заглавная буква/цифра, либо конец строки.
_SENT_BOUNDARY = re.compile(r"(?<=[.!?…])[ \t]+(?=[«\"(]?[A-ZА-ЯЁ0-9])")

ABBREV = {
    "т",
    "е",
    "д",
    "п",
    "г",
    "гг",
    "рис",
    "табл",
    "см",
    "др",
    "проф",
    "акад",
    "тыс",
    "млн",
    "млрд",
    "руб",
    "коп",
    "с",
    "стр",
    "напр",
    "им",
    "обл",
    "р",
    "ул",
    "кв",
    "оф",
    "mr",
    "dr",
    "etc",
    "vs",
    "i.e",
    "e.g",
}


@dataclass(frozen=True)
class Token:
    """Токен с абсолютными смещениями в исходном тексте."""

    text: str
    start: int
    end: int
    kind: str  # "word" | "gap"

    @property
    def is_word(self) -> bool:
        return self.kind == "word"

    @property
    def length(self) -> int:
        return self.end - self.start

    def __repr__(self) -> str:  # pragma: no cover - отладочное
        return f"Token({self.text!r}, {self.start}, {self.end}, {self.kind})"


def tokenize(text: str) -> list[Token]:
    """Разбить текст на непрерывную последовательность токенов."""
    tokens: list[Token] = []
    pos = 0
    for match in WORD_RE.finditer(text):
        if match.start() > pos:
            tokens.append(Token(text[pos : match.start()], pos, match.start(), "gap"))
        tokens.append(Token(match.group(0), match.start(), match.end(), "word"))
        pos = match.end()
    if pos < len(text):
        tokens.append(Token(text[pos:], pos, len(text), "gap"))
    return tokens


def word_tokens(tokens: Iterable[Token]) -> list[Token]:
    return [t for t in tokens if t.is_word]


def word_indices(tokens: Iterable[Token]) -> list[int]:
    return [i for i, t in enumerate(tokens) if t.is_word]


def sentences(text: str) -> list[tuple[int, int]]:
    """Список (start, end) предложений с учётом частых сокращений."""
    bounds: list[int] = []
    for match in _SENT_BOUNDARY.finditer(text):
        prefix = text[: match.start()].rstrip()
        last_word = re.split(r"[\s«\"(]", prefix)[-1]
        last_word = last_word.strip(".,!?…").lower()
        if last_word and len(last_word) <= 4 and last_word in ABBREV:
            continue
        bounds.append(match.end())
    result: list[tuple[int, int]] = []
    start = 0
    for boundary in bounds:
        result.append((start, boundary))
        start = boundary
    result.append((start, len(text)))
    return [(s, e) for s, e in result if text[s:e].strip()]


def sentence_spans(token_list: list[Token]) -> list[tuple[int, int]]:
    """Индексы токенов, сгруппированные по предложениям."""
    if not token_list:
        return []
    text = "".join(t.text for t in token_list)
    spans: list[tuple[int, int]] = []
    for s_start, s_end in sentences(text):
        first = last = None
        for i, tok in enumerate(token_list):
            if tok.end > s_start and tok.start < s_end:
                if first is None:
                    first = i
                last = i
        if first is not None and last is not None:
            spans.append((first, last + 1))
    return spans


def normalize_word(word: str) -> str:
    return word.lower().replace("ё", "е")


def char_ngrams(word: str, n: int = 3) -> Iterator[str]:
    """Символьные n-граммы с граничными маркерами."""
    padded = "^" + normalize_word(word) + "$"
    if len(padded) <= n:
        yield padded
        return
    for i in range(len(padded) - n + 1):
        yield padded[i : i + n]


def split_paragraphs(text: str) -> list[tuple[int, int]]:
    """Абзацы (start, end) по пустым строкам."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for block in re.split(r"\n\s*\n", text):
        start = text.index(block, pos) if block else pos
        if block.strip():
            spans.append((start, start + len(block)))
        pos = start + len(block)
    return spans
