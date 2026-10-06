"""Ядро контекстной верификации: типы и сборка входа «контекст + ответ».

Модуль отвечает на два вопроса по паре «контекст (источник) — ответ модели»:

* содержит ли ответ сведения, не подтверждаемые контекстом (недостоверность,
  она же галлюцинация), и где именно в ответе эти места;
* какая доля ответа похожа на машинную генерацию.

Токенизация — с непрерывным покрытием: объединение токенов через пробел
восстанавливает исходный текст, поэтому смещения фрагментов точны, а доля
участия ИИ считается в символах корректно.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "TOKEN_RE",
    "Token",
    "ContextChunks",
    "tokenize_with_offsets",
    "split_sentences",
    "split_chunks",
    "numbers_in",
    "SpanResult",
    "VerificationResult",
]

# Токен: слово (буквы/цифры с внутренними дефисами) либо одиночный не-пробельный знак.
TOKEN_RE = re.compile(r"\w+(?:[-'’]\w+)*|[^\w\s]", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?…])[ \t]+(?=[«\"(]?[A-ZА-ЯЁ0-9])")
_NUMBER_WORDS = {
    "ноль": 0,
    "один": 1,
    "одна": 1,
    "два": 2,
    "две": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
    "десять": 10,
    "одиннадцать": 11,
    "двенадцать": 12,
    "пятнадцать": 15,
    "двадцать": 20,
    "тридцать": 30,
    "сорок": 40,
    "пятьдесят": 50,
    "сто": 100,
}


PUNCT = ".,;:!?()«»\"'—–-"


@dataclass(frozen=True)
class Token:
    """Токен с абсолютными смещениями в тексте (пробелы приклеены к токену)."""

    text: str
    start: int
    end: int

    @property
    def lower(self) -> str:
        return self.text.lower().replace("ё", "е")

    @property
    def word(self) -> str:
        """Слово без пробелов и обрамляющей пунктуации."""
        return self.text.strip().strip(PUNCT)


@dataclass(frozen=True)
class ContextChunks:
    """Контекст, разбитый на чанки (предложения), для оценки близости."""

    text: str
    chunks: tuple[str, ...]
    positions: tuple[int, ...]  # смещение чанка в контексте


def configure_stdio(encoding: str = "utf-8") -> None:
    """Сделать вывод в консоль устойчивым к кодировкам Windows.

    На Windows консоль может быть в cp866/cp1251, и печать русских сообщений
    падает с ``UnicodeEncodeError`` (это ломало запуск собранного ``.exe`` в CI).
    Поток переводится в UTF-8 с заменой непереводимых символов; там, где
    переключение недоступно (тесты, перехваченный вывод), вызов молча пропускается.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding=encoding, errors="replace")
        except (ValueError, OSError):  # pragma: no cover - зависит от окружения
            continue


def tokenize_with_offsets(text: str) -> list[Token]:
    """Токены с непрерывным покрытием: пробелы приклеиваются к предыдущему токену.

    Инвариант: ``" ".join(t.text for t in tokens)`` восстанавливает текст
    с точностью до количества пробелов внутри, а последовательность
    ``[text[t.start:t.end]]`` совпадает с текстом посимвольно.
    """
    if not text or not text.strip():
        return []
    spans: list[Token] = []
    cursor = 0
    for match in TOKEN_RE.finditer(text):
        start, end = match.start(), match.end()
        while cursor < start and text[cursor].isspace():
            cursor += 1
        if cursor < start:
            # Редкий случай: не-пробельный разделитель, не попавший в шаблон.
            spans.append(Token(text[start:end], start, end))
            cursor = end
            continue
        spans.append(Token(text[start:end], start, end))
        cursor = end
        # приклеить следующие пробелы к текущему токену (непрерывность покрытия)
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor > end:
            spans[-1] = Token(text[start:cursor], start, cursor)
    if not spans or spans[-1].end < len(text):
        start = spans[-1].end if spans else 0
        if text[start:]:
            spans.append(Token(text[start:], start, len(text)))
    return spans


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Границы предложений (start, end) с учётом пробелов после знаков."""
    if not text or not text.strip():
        return []
    bounds: list[int] = []
    for match in _SENTENCE_RE.finditer(text):
        bounds.append(match.end())
    result: list[tuple[int, int]] = []
    start = 0
    for boundary in bounds:
        result.append((start, boundary))
        start = boundary
    if start < len(text):
        result.append((start, len(text)))
    return [(s, e) for s, e in result if text[s:e].strip()]


def split_chunks(context: str | Sequence[str] | None) -> ContextChunks:
    """Привести контекст (строку или список чанков) к единому виду."""
    if context is None:
        return ContextChunks("", (), ())
    if isinstance(context, str):
        chunks = [context[s:e] for s, e in split_sentences(context)] or ([context] if context else [])
        positions: list[int] = []
        cursor = 0
        for chunk in chunks:
            index = context.find(chunk, cursor)
            positions.append(index if index >= 0 else cursor)
            cursor = index + len(chunk) if index >= 0 else cursor
        return ContextChunks(context, tuple(chunks), tuple(positions))
    chunks_list = [str(item) for item in context if str(item).strip()]
    joined = "\n".join(chunks_list)
    positions = []
    cursor = 0
    for chunk in chunks_list:
        index = joined.find(chunk, cursor)
        positions.append(index if index >= 0 else cursor)
        cursor = (index + len(chunk)) if index >= 0 else cursor
    return ContextChunks(joined, tuple(chunks_list), tuple(positions))


def number_value(token: Token | str) -> str | None:
    """Каноническое значение числа: «5 год» и «пять лет» → ``"5"``."""
    word = token.word if isinstance(token, Token) else str(token).strip().strip(PUNCT)
    word = word.lower().replace("ё", "е")
    if not word:
        return None
    if word.isdigit():
        return str(int(word))
    return str(_NUMBER_WORDS[word]) if word in _NUMBER_WORDS else None


def numbers_in(text: str) -> set[str]:
    """Числовые значения в тексте (цифрами и словами) — для контроля фактов."""
    found: set[str] = set()
    for token in tokenize_with_offsets(text):
        value = number_value(token)
        if value is not None:
            found.add(value)
    return found


@dataclass
class SpanResult:
    """Спорный фрагмент ответа."""

    start: int
    end: int
    text: str
    risk: float
    label: str  # "likely_hallucination" | "doubtful"
    n_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "risk": round(self.risk, 4),
            "label": self.label,
            "n_tokens": self.n_tokens,
        }


@dataclass
class VerificationResult:
    """Результат проверки пары «контекст — ответ»."""

    score: float
    is_hallucination: bool
    ai_share: float
    ai_share_hard: float
    ai_participation: float = 0.0
    threshold: float = 0.0
    spans: list[SpanResult] = field(default_factory=list)
    tokens: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    mode: str = "demo"
    warning: str = ""
    verdict: str = ""

    def to_dict(self, with_tokens: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "score": round(self.score, 4),
            "is_hallucination": bool(self.is_hallucination),
            "ai_share": round(self.ai_share, 4),
            "ai_share_soft": round(self.ai_share, 4),
            "ai_share_hard": round(self.ai_share_hard, 4),
            "ai_participation": round(self.ai_participation, 4),
            "threshold": round(self.threshold, 4),
            "verdict": self.verdict,
            "spans": [span.to_dict() for span in self.spans],
            "latency_ms": round(self.latency_ms, 2),
            "mode": self.mode,
            "stats": self.stats,
        }
        if with_tokens:
            payload["tokens"] = self.tokens
        if (with_tokens or self.tokens) and "tokens" not in payload:
            payload["tokens"] = self.tokens
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerificationResult:
        """Восстановить результат из JSON (для тестов и офлайн-разбора)."""
        return cls(
            score=float(data.get("score", 0.0)),
            is_hallucination=bool(data.get("is_hallucination", False)),
            ai_share=float(data.get("ai_share", 0.0)),
            ai_share_hard=float(data.get("ai_share_hard", 0.0)),
            ai_participation=float(data.get("ai_participation", 0.0)),
            threshold=float(data.get("threshold", 0.0)),
            spans=[
                SpanResult(
                    start=int(item["start"]),
                    end=int(item["end"]),
                    text=str(item.get("text", "")),
                    risk=float(item.get("risk", 0.0)),
                    label=str(item.get("label", "doubtful")),
                    n_tokens=int(item.get("n_tokens", 0)),
                )
                for item in data.get("spans", [])
            ],
            tokens=list(data.get("tokens", [])),
            stats=dict(data.get("stats", {})),
            latency_ms=float(data.get("latency_ms", 0.0)),
            mode=str(data.get("mode", "demo")),
            verdict=str(data.get("verdict", "")),
        )


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0
