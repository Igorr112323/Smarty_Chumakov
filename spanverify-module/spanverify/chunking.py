"""Чанкинг документа и отбор релевантных фрагментов для режима hf.

Решает проблему обрезки входа: при max_length=1024 длинные документы
обрезаются, и модель физически не видит доказательство. Вместо передачи
всего документа — отбираем 1–3 релевантных фрагмента, которые вместе
с ответом помещаются в окно модели.

Этап 1 плана работ:
1. Документ режется на фрагменты по предложениям/абзацам (перекрытие 1 предложение).
2. Каждый фрагмент ≤ ~380 токенов (чтобы с ответом уложиться в 512).
3. Для каждого клайма ответа отбираются 1–3 фрагмента документа (BM25).
4. Признаки считаются только против отобранных фрагментов.
5. Агрегация по фрагментам — максимум риска (не среднее).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .core import split_sentences
from .normalize import numbers_to_digits

__all__ = [
    "Chunk",
    "DocumentChunks",
    "chunk_document",
    "select_relevant_chunks",
    "normalize_for_retrieval",
    "estimate_tokens",
]

# Оценка токенов: ~2.7 знака на токен для русского текста (ruGPT3-small).
# Это приблизительная оценка для планирования; реальная токенизация — в модели.
CHARS_PER_TOKEN_ESTIMATE = 2.7


def estimate_tokens(text: str) -> int:
    """Приблизительная оценка числа токенов (без реального токенизатора)."""
    if not text:
        return 0
    return max(1, int(len(text) / CHARS_PER_TOKEN_ESTIMATE))


def normalize_for_retrieval(text: str) -> str:
    """Нормализация текста для retrieval: числа цифрами, lowercase, уборка пробелов.

    Числа приводятся к единому формату (цифры), чтобы «3 года» и «3» совпадали.
    Даты нормализуются отдельно (не здесь, а в number_attribution).
    """
    text = numbers_to_digits(text)
    text = text.lower()
    # Нормализация пробелов
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass
class Chunk:
    """Один фрагмент документа."""

    text: str
    start: int  # смещение в исходном документе
    end: int  # смещение в исходном документе
    sentences: tuple[str, ...] = ()  # предложения внутри чанка
    sentence_indices: tuple[int, ...] = ()  # индексы предложений в документе
    token_estimate: int = 0

    def __post_init__(self) -> None:
        if self.token_estimate == 0:
            self.token_estimate = estimate_tokens(self.text)


@dataclass
class DocumentChunks:
    """Документ, разбитый на чанки."""

    original_text: str
    chunks: list[Chunk] = field(default_factory=list)
    # Для каждого чанка — нормализованный текст (для retrieval)
    normalized: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.normalized and self.chunks:
            self.normalized = [normalize_for_retrieval(c.text) for c in self.chunks]


def _split_paragraphs(text: str) -> list[tuple[int, int]]:
    """Разбить текст на абзацы (по \\n\\n или \\n)."""
    paragraphs: list[tuple[int, int]] = []
    start = 0
    for m in re.finditer(r"\n\s*\n|\n", text):
        end = m.start()
        if end > start:
            paragraphs.append((start, end))
        start = m.end()
    if start < len(text):
        paragraphs.append((start, len(text)))
    return [(s, e) for s, e in paragraphs if text[s:e].strip()]


def _split_long_segment(text: str, max_chars: int, overlap_chars: int = 100) -> list[tuple[int, int]]:
    """Разбить длинный сегмент на части по max_chars с перекрытием.

    Старается резать по границам предложений внутри сегмента.
    Если и предложение длиннее max_chars — режет по словам.
    """
    if len(text) <= max_chars:
        return [(0, len(text))]

    parts: list[tuple[int, int]] = []
    pos = 0
    while pos < len(text):
        end = min(pos + max_chars, len(text))
        # Попробовать обрезать по предложению
        if end < len(text):
            # Найти последнее предложение в пределах max_chars
            seg = text[pos:end]
            sent_bounds = split_sentences(seg)
            if sent_bounds and sent_bounds[-1][1] > pos * 0.3:  # не слишком коротко
                end = pos + sent_bounds[-1][1]
            else:
                # Попробовать по пробелу
                space = text.rfind(" ", pos + max_chars // 2, end)
                if space > pos:
                    end = space
        parts.append((pos, end))
        if end >= len(text):
            break
        pos = max(end - overlap_chars, pos + 1)
    return parts


def chunk_document(
    text: str,
    max_tokens_per_chunk: int = 380,
    overlap_sentences: int = 1,
    max_chars_per_chunk: int | None = None,
) -> DocumentChunks:
    """Разбить документ на фрагменты с учётом абзацев, предложений и лимита токенов.

    Алгоритм:
    1. Разбить на абзацы (\\n\\n или \\n).
    2. Внутри абзаца — по предложениям.
    3. Группировать предложения в чанки до max_tokens_per_chunk.
    4. Если предложение длиннее max_tokens_per_chunk — разбить по символам.
    5. Перекрытие — overlap_sentences предложений между соседними чанками.

    max_chars_per_chunk — если None, вычисляется из max_tokens_per_chunk.
    """
    if max_chars_per_chunk is None:
        max_chars_per_chunk = int(max_tokens_per_chunk * CHARS_PER_TOKEN_ESTIMATE)

    if not text or not text.strip():
        return DocumentChunks(original_text=text)

    # Шаг 1: Разбить на абзацы
    paragraph_bounds = _split_paragraphs(text)
    if not paragraph_bounds:
        return DocumentChunks(
            original_text=text,
            chunks=[
                Chunk(
                    text=text,
                    start=0,
                    end=len(text),
                    sentences=(text,),
                    sentence_indices=(0,),
                )
            ],
        )

    # Шаг 2: Внутри каждого абзаца — предложения
    all_sentences: list[tuple[int, int, int]] = []  # (start, end, paragraph_idx)
    for p_idx, (p_start, p_end) in enumerate(paragraph_bounds):
        para_text = text[p_start:p_end]
        sent_bounds = split_sentences(para_text)
        if sent_bounds:
            for s, e in sent_bounds:
                abs_s = p_start + s
                abs_e = p_start + e
                # Если предложение длиннее лимита — разбить
                if abs_e - abs_s > max_chars_per_chunk:
                    sub_parts = _split_long_segment(
                        text[abs_s:abs_e],
                        max_chars_per_chunk,
                        overlap_chars=min(100, max_chars_per_chunk // 4),
                    )
                    for sub_s, sub_e in sub_parts:
                        all_sentences.append((abs_s + sub_s, abs_s + sub_e, p_idx))
                else:
                    all_sentences.append((abs_s, abs_e, p_idx))
        else:
            # Абзац без предложений (нет точек) — разбить по символам если длинный
            if p_end - p_start > max_chars_per_chunk:
                sub_parts = _split_long_segment(
                    text[p_start:p_end],
                    max_chars_per_chunk,
                    overlap_chars=min(100, max_chars_per_chunk // 4),
                )
                for sub_s, sub_e in sub_parts:
                    all_sentences.append((p_start + sub_s, p_start + sub_e, p_idx))
            else:
                all_sentences.append((p_start, p_end, p_idx))

    if not all_sentences:
        return DocumentChunks(
            original_text=text,
            chunks=[
                Chunk(
                    text=text,
                    start=0,
                    end=len(text),
                    sentences=(text,),
                    sentence_indices=(0,),
                )
            ],
        )

    # Шаг 3: Группировать в чанки по max_tokens_per_chunk
    chunks: list[Chunk] = []
    i = 0

    while i < len(all_sentences):
        chunk_texts: list[str] = []
        chunk_sent_indices: list[int] = []
        chunk_start = all_sentences[i][0]
        current_tokens = 0
        j = i

        while j < len(all_sentences):
            s, e, _ = all_sentences[j]
            sent_text = text[s:e]
            sent_tokens = estimate_tokens(sent_text)
            if current_tokens + sent_tokens > max_tokens_per_chunk and chunk_texts:
                break
            chunk_texts.append(sent_text)
            chunk_sent_indices.append(j)
            current_tokens += sent_tokens
            j += 1

        if not chunk_texts:
            # Одно предложение длиннее лимита — берём его целиком
            s, e, _ = all_sentences[i]
            chunk_texts = [text[s:e]]
            chunk_sent_indices = [i]
            j = i + 1

        chunk_end = all_sentences[chunk_sent_indices[-1]][1]
        chunk_text = text[chunk_start:chunk_end]

        chunks.append(
            Chunk(
                text=chunk_text,
                start=chunk_start,
                end=chunk_end,
                sentences=tuple(chunk_texts),
                sentence_indices=tuple(chunk_sent_indices),
            )
        )

        # Следующий чанк начинается с учётом перекрытия
        next_start = max(i + 1, j - overlap_sentences)
        if next_start <= i:
            next_start = i + 1
        i = next_start

    normalized = [normalize_for_retrieval(c.text) for c in chunks]
    return DocumentChunks(original_text=text, chunks=chunks, normalized=normalized)


# ────────────────────────────────────────────────────────────────────
# BM25-подобный отбор релевантных фрагментов
# ────────────────────────────────────────────────────────────────────

# Простая токенизация для retrieval (не модельная!)
_RETRIEVAL_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.UNICODE | re.IGNORECASE)


def _tokenize_for_bm25(text: str) -> list[str]:
    """Токенизация для BM25: слова и числа."""
    return _RETRIEVAL_TOKEN_RE.findall(text.lower())


_STOPWORDS_RU = frozenset(
    {
        "и",
        "в",
        "во",
        "не",
        "что",
        "он",
        "на",
        "я",
        "быть",
        "с",
        "а",
        "это",
        "все",
        "она",
        "так",
        "его",
        "но",
        "да",
        "ты",
        "к",
        "у",
        "же",
        "вы",
        "за",
        "мы",
        "из",
        "по",
        "о",
        "от",
        "до",
        "для",
        "или",
        "если",
        "уже",
        "был",
        "была",
        "было",
        "были",
        "бы",
        "как",
        "то",
        "только",
        "их",
        "её",
        "тем",
        "также",
        "при",
    }
)


def _remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in _STOPWORDS_RU and len(t) > 1]


@dataclass
class _BM25Index:
    """Простой BM25-индекс по чанкам документа."""

    docs: list[list[str]]  # токены каждого чанка
    doc_lens: list[int]
    avg_dl: float
    n_docs: int
    df: dict[str, int]  # document frequency
    k1: float = 1.5
    b: float = 0.75

    def score(self, query: list[str]) -> list[float]:
        """BM25-скор для каждого чанка по запросу."""
        scores = [0.0] * self.n_docs
        idf = {}
        for term in set(query):
            if term in self.df:
                # IDF с сглаживанием
                idf[term] = math.log(1 + (self.n_docs - self.df[term] + 0.5) / (self.df[term] + 0.5))

        query_tf = Counter(query)
        for term, _qtf in query_tf.items():
            if term not in idf:
                continue
            term_idf = idf[term]
            for i, doc_tokens in enumerate(self.docs):
                tf = doc_tokens.count(term)
                if tf == 0:
                    continue
                dl = self.doc_lens[i]
                norm_tf = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl))
                scores[i] += term_idf * norm_tf

        return scores


def _build_bm25_index(chunks_normalized: list[str]) -> _BM25Index:
    docs = [_remove_stopwords(_tokenize_for_bm25(text)) for text in chunks_normalized]
    doc_lens = [len(d) for d in docs]
    avg_dl = sum(doc_lens) / max(1, len(docs))
    df: dict[str, int] = {}
    for doc in docs:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1
    return _BM25Index(docs=docs, doc_lens=doc_lens, avg_dl=avg_dl, n_docs=len(docs), df=df)


def select_relevant_chunks(
    answer: str,
    doc_chunks: DocumentChunks,
    top_k: int = 3,
    min_score: float = 0.0,
) -> list[int]:
    """Отобрать top-k релевантных чанков документа для данного ответа.

    Использует BM25 по нормализованным токенам.
    Возвращает индексы чанков в порядке убывания релевантности.

    Если ответ короткий и все чанки имеют score=0 (нет совпадений),
    возвращается хотя бы один чанк (ближайший к началу документа).
    """
    if not doc_chunks.chunks:
        return []

    if len(doc_chunks.chunks) <= top_k:
        return list(range(len(doc_chunks.chunks)))

    query_tokens = _remove_stopwords(_tokenize_for_bm25(normalize_for_retrieval(answer)))
    if not query_tokens:
        # Нет содержательных токенов — вернуть первый чанк
        return [0]

    index = _build_bm25_index(doc_chunks.normalized)
    scores = index.score(query_tokens)

    # Отсортировать по убыванию score
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    # Взять top_k с ненулевым score, или хотя бы один
    result = [i for i in ranked[:top_k] if scores[i] > min_score]
    if not result:
        result = [ranked[0]]

    # Вернуть в порядке следования в документе (для согласованности)
    result.sort()
    return result


def select_relevant_chunks_for_claims(
    claims: list[str],
    doc_chunks: DocumentChunks,
    top_k_per_claim: int = 3,
) -> list[int]:
    """Отобрать релевантные чанки для нескольких клаймов ответа.

    Объединяет результаты для всех клаймов (dedup, сортировка по позиции).
    Это нужно для случая, когда ответ длинный и разбирается на клаймы.
    """
    if not claims:
        return select_relevant_chunks("", doc_chunks, top_k=top_k_per_claim)

    all_indices: set[int] = set()
    for claim in claims:
        indices = select_relevant_chunks(claim, doc_chunks, top_k=top_k_per_claim)
        all_indices.update(indices)

    return sorted(all_indices)
