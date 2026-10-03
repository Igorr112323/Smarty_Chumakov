"""Суррогатный бэкенд (демо-режим, только стандартная библиотека).

Признак «предсказуемость токена» собирается из стилометрических сигналов,
которые коррелируют с машинной генерацией:

* **повторяемость содержательных слов** — у LLM узкий словарь ключевых
  терминов, который воспроизводится в документе многократно;
* **шаблонная лексика и связки** — канцелярско-академические обороты;
* **локальная повторяемость контекста** — насыщенность повторами в окне;
* **единообразие длины слов и предложений** — ровный «машинный» ритм.

Служебные слова из оценки исключаются: предлоги и союзы повторяются в
любом тексте и только размывают границу.

Это НЕ энтропия внимания и не перплексия языковой модели. Режим проверяет
конвейер и интерфейс на машинах без GPU и без доступа к весам модели.
Числа, полученные в этом режиме, не переносятся на реальные документы.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from ..lexicon import (
    BOILERPLATE_BIGRAMS,
    BOILERPLATE_WORDS,
    STOPWORDS,
    find_phrase_hits,
)
from ..text import normalize_word, sentence_spans, tokenize
from ..vectors import vectorize
from .base import Backend, BackendResult

WINDOW = 12
NEUTRAL_STOPWORD_SCORE = 0.35  # служебные слова не дают стилевого сигнала

WEIGHTS = {
    "phrase_membership": 0.20,  # токен входит в шаблонный оборот
    "phrase_density": 0.20,  # плотность оборотов в окне (работает на коротких текстах)
    "boilerplate": 0.14,  # отдельное шаблонное слово
    "content_reuse": 0.16,  # повторяемость содержательных слов в документе
    "context_repetition": 0.10,
    "length_uniformity": 0.08,
    "sentence_uniformity": 0.12,
}


class SurrogateBackend(Backend):
    name = "surrogate"
    description = "стилометрический суррогат (демо, без внешних зависимостей)"

    def __init__(self, **_: object) -> None:
        """Принимает и игнорирует параметры реальной модели (model, max_tokens…)."""

    def process(self, words: Sequence[str], text: str, dim: int = 4096) -> BackendResult:
        if not words:
            return BackendResult(predictability=[], vectors=[], meta={"backend": self.name})

        normal = [normalize_word(w) for w in words]
        n = len(words)

        # --- содержательные слова: частоты и повторы ---
        content = [w for w in normal if w not in STOPWORDS and len(w) > 2]
        content_counts = Counter(content)
        content_density = sum(1 for w in normal if w not in STOPWORDS) / n

        # --- шаблонные обороты ---
        phrase_hits = find_phrase_hits(normal)

        # --- длины и предложения ---
        lengths = [len(w) for w in words]
        mean_len = sum(lengths) / n

        tokens = tokenize(text)
        sent_spans = sentence_spans(tokens)
        sent_sizes = [max(1, e - s) for s, e in sent_spans]
        mean_sent = sum(sent_sizes) / len(sent_sizes) if sent_sizes else 1.0

        word_positions = [i for i, t in enumerate(tokens) if t.is_word]
        sent_len_by_word: dict[int, int] = {}
        for s_start, s_end in sent_spans:
            for pos in range(s_start, min(s_end, len(word_positions))):
                sent_len_by_word[pos] = s_end - s_start

        boilerplate_hits = 0
        predictability: list[float] = []
        informative: list[bool] = []
        for i, _word in enumerate(words):
            norm = normal[i]
            is_service = (norm in STOPWORDS or len(norm) <= 2) and i not in phrase_hits

            if is_service:
                # Служебные слова не различают стили: помечаем их как
                # неинформативные — детектор восстановит их оценку по окружению.
                informative.append(False)
                predictability.append(NEUTRAL_STOPWORD_SCORE)
                continue

            informative.append(True)

            # 1. Повторяемость содержательного слова внутри документа.
            count = content_counts.get(norm, 1)
            content_reuse = min(1.0, (count - 1) / 2.0)

            # 2. Шаблонная лексика и связки.
            boilerplate = 0.0
            if norm in BOILERPLATE_WORDS:
                boilerplate = 1.0
                boilerplate_hits += 1
            else:
                nxt = normal[i + 1] if i + 1 < n else ""
                prv = normal[i - 1] if i > 0 else ""
                if (norm, nxt) in BOILERPLATE_BIGRAMS or (prv, norm) in BOILERPLATE_BIGRAMS:
                    boilerplate = 0.8

            # 3. Шаблонные обороты: вхождение токена и плотность в окне.
            phrase_membership = 1.0 if i in phrase_hits else 0.0

            # 4. Локальные характеристики окна.
            lo, hi = max(0, i - WINDOW), min(n, i + WINDOW + 1)
            window = [w for w in normal[lo:hi] if w not in STOPWORDS and len(w) > 2]
            repeated = sum(1 for w in window if content_counts.get(w, 0) > 1)
            context_repetition = repeated / max(1, len(window))
            phrases_in_window = sum(1 for j in range(lo, hi) if j in phrase_hits and normal[j] not in STOPWORDS)
            content_in_window = sum(1 for w in window)
            phrase_density = phrases_in_window / max(1, content_in_window)

            # 5. Ритм: однородность длины слова и предложения.
            length_uniformity = 1.0 - min(1.0, abs(lengths[i] - mean_len) / 5.0)
            token_pos = word_positions[i] if i < len(word_positions) else 0
            local_sent = sent_len_by_word.get(token_pos, mean_sent)
            sentence_uniformity = 1.0 - min(1.0, abs(local_sent - mean_sent) / max(3.0, mean_sent))

            score = (
                WEIGHTS["phrase_membership"] * phrase_membership
                + WEIGHTS["phrase_density"] * phrase_density
                + WEIGHTS["content_reuse"] * content_reuse
                + WEIGHTS["boilerplate"] * boilerplate
                + WEIGHTS["context_repetition"] * context_repetition
                + WEIGHTS["length_uniformity"] * length_uniformity
                + WEIGHTS["sentence_uniformity"] * sentence_uniformity
            )
            predictability.append(_squash(score))

        vectors = vectorize(words, dim=dim)
        return BackendResult(
            predictability=predictability,
            vectors=vectors,
            informative=informative,
            meta={
                "backend": self.name,
                "mean_word_length": round(mean_len, 3),
                "mean_sentence_chars": round(mean_sent, 1),
                "content_word_ratio": round(content_density, 4),
                "boilerplate_rate": round(boilerplate_hits / n, 4),
                "type_token_ratio": round(len(set(normal)) / n, 4),
            },
        )


def _squash(x: float, midpoint: float = 0.42, steepness: float = 6.0) -> float:
    """Логистическое сжатие в (0, 1) вокруг рабочей точки."""
    return 1.0 / (1.0 + math.exp(-steepness * (x - midpoint)))
