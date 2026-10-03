"""Векторизация токенов и контекстная плотность (kNN).

Реализация намеренно обходится стандартной библиотекой: разреженные
хешированные векторы в словарях + инвертированный индекс для поиска
соседей. Это даёт пригодную скорость на документах до ~10^4 токенов
и нулевые зависимости при сборке в один исполняемый файл.
"""

from __future__ import annotations

from hashlib import blake2b
from typing import Iterable, Sequence

__all__ = [
    "hash_index",
    "token_vector",
    "vectorize",
    "normalize",
    "cosine",
    "knn_density",
]

# Веса компонент вектора токена.
W_TOKEN = 1.0       # сам токен (лемма не нужна — берём нормализованную форму)
W_CHARGRAM = 0.35   # символьные 3-граммы: устойчивость к словоформам
W_CONTEXT = 0.25    # соседние слова: контекст употребления


def hash_index(feature: str, dim: int) -> int:
    """Стабильный (не зависящий от PYTHONHASHSEED) индекс признака."""
    digest = blake2b(feature.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


def token_vector(
    word: str,
    dim: int = 4096,
    prev_word: str | None = None,
    next_word: str | None = None,
    chargrams: Iterable[str] = (),
) -> dict[int, float]:
    """Разреженный L2-нормированный вектор токена."""
    from .text import normalize_word

    vec: dict[int, float] = {}
    norm_word = normalize_word(word)
    vec[hash_index("w:" + norm_word, dim)] = vec.get(hash_index("w:" + norm_word, dim), 0.0) + W_TOKEN

    grams = list(chargrams)
    if grams:
        weight = W_CHARGRAM / max(1, len(grams)) ** 0.5
        for gram in grams:
            idx = hash_index("g:" + gram, dim)
            vec[idx] = vec.get(idx, 0.0) + weight

    for label, neighbour in (("p:", prev_word), ("n:", next_word)):
        if neighbour:
            idx = hash_index(label + normalize_word(neighbour), dim)
            vec[idx] = vec.get(idx, 0.0) + W_CONTEXT

    return normalize(vec)


def vectorize(
    words: Sequence[str],
    dim: int = 4096,
    gram_n: int = 3,
) -> list[dict[int, float]]:
    """Векторы для последовательности словоформ с учётом соседей."""
    from .text import char_ngrams

    vectors: list[dict[int, float]] = []
    for i, word in enumerate(words):
        prev_word = words[i - 1] if i > 0 else None
        next_word = words[i + 1] if i + 1 < len(words) else None
        vectors.append(
            token_vector(
                word,
                dim=dim,
                prev_word=prev_word,
                next_word=next_word,
                chargrams=char_ngrams(word, gram_n),
            )
        )
    return vectors


def normalize(vec: dict[int, float]) -> dict[int, float]:
    norm = sum(v * v for v in vec.values()) ** 0.5
    if norm <= 0:
        return {}
    return {k: v / norm for k, v in vec.items()}


def cosine(a: dict[int, float], b: dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(value * b.get(idx, 0.0) for idx, value in a.items())


def knn_density(
    vectors: Sequence[dict[int, float]],
    k: int = 5,
    max_postings: int = 512,
    mask: Sequence[bool] | None = None,
) -> list[float]:
    """Средняя косинусная близость к k ближайшим соседям по документу.

    Чем выше значение, тем «типичнее» токен для документа: машинный текст
    воспроизводит устойчивые обороты и даёт высокую плотность окружения.
    Значения в диапазоне [0, 1] (косинус неотрицательных векторов).

    ``mask`` позволяет исключить из подсчёта токены без стилевого сигнала
    (служебные слова): они повторяются в любом тексте и создают ложную
    плотность.
    """
    n = len(vectors)
    if n <= 1:
        return [0.0] * n

    index: dict[int, list[int]] = {}
    for i, vec in enumerate(vectors):
        for feature in vec:
            index.setdefault(feature, []).append(i)

    density = [0.0] * n
    for i, vec in enumerate(vectors):
        acc: dict[int, float] = {}
        for feature, weight in vec.items():
            postings = index.get(feature, ())
            if len(postings) > max_postings:  # слишком частый признак — шум
                continue
            for j in postings:
                if j != i and (mask is None or mask[j]):
                    acc[j] = acc.get(j, 0.0) + weight * vectors[j][feature]
        if not acc:
            continue
        top = sorted(acc.values(), reverse=True)[: max(1, k)]
        density[i] = sum(top) / len(top)
    return density
