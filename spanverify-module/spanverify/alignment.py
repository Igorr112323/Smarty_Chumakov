"""Сопоставление сабтокенов модели с символами ответа и окно длинного текста.

Это «слабое место пилота» из реестра (пункт 2.1). Три задачи решаются здесь и
решаются **без torch**, чтобы их можно было проверить тестами на настоящем
BPE-токенизаторе, а не на заглушке:

1. :func:`align_subwords` — какие сабтокены модели относятся к каждому слову
   ответа. Byte-level BPE (ruGPT-3, Qwen) включает ведущий пробел в начало
   сабтокена, из-за чего наивное сравнение смещений сдвигает всё на один
   токен и признаки «съезжают» на соседнее слово.
2. :func:`plan_windows` — как разложить последовательность длиннее окна модели
   на перекрывающиеся окна и как потом склеить значения
   (:func:`merge_window_values`), чтобы длинные акты обрабатывались целиком, а
   не обрезались по 512 токенам.
3. :func:`batch_sizes` — как разбить окна на пакеты при нехватке памяти
   (используется в :mod:`spanverify.backends.hf` при ``MemoryError`` /
   ``torch.cuda.OutOfMemoryError``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "Window",
    "align_subwords",
    "batch_sizes",
    "merge_window_values",
    "plan_windows",
    "trim_offsets",
]


def trim_offsets(text: str, offsets: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Убрать из смещений ведущие пробелы и спецтокены.

    Byte-level BPE отдаёт для « хранения» смещение, начинающееся на пробеле.
    Для привязки к словам пробел надо отбросить, иначе сабтокен пересекается
    сразу с двумя словами. Спецтокены (``end <= start``) превращаются в
    ``(-1, -1)`` и далее игнорируются.
    """
    trimmed: list[tuple[int, int]] = []
    for start, end in offsets:
        if end <= start:
            trimmed.append((-1, -1))
            continue
        start = max(0, min(start, len(text)))
        end = max(0, min(end, len(text)))
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        trimmed.append((start, end) if end > start else (-1, -1))
    return trimmed


def align_subwords(
    text: str,
    word_spans: Sequence[tuple[int, int]],
    offsets: Sequence[tuple[int, int]],
) -> list[list[int]]:
    """Сабтокены для каждого слова: ``[[i, i+1], [i+2], …]``.

    ``word_spans`` — символьные границы слов ответа (из
    :func:`spanverify.core.tokenize_with_offsets`), ``offsets`` — смещения
    сабтокенов токенизатора модели. Возвращается список той же длины, что
    ``word_spans``; пустой список означает, что слову не досталось ни одного
    сабтокена (например, оно обрезано окном модели).

    Правило: сабтокен относится к слову, если их символьные отрезки
    пересекаются **после** удаления пробелов. Если один сабтокен покрывает
    несколько слов (редко, но бывает на знаках), он достаётся каждому из них:
    это честнее, чем терять признак.
    """
    clean = trim_offsets(text, offsets)
    result: list[list[int]] = [[] for _ in word_spans]
    if not word_spans:
        return result
    # Оба списка отсортированы, поэтому хватает одного прохода «встречным
    # курсором» вместо квадратичного перебора (важно для длинных актов).
    sub_index = 0
    for word_index, (w_start, w_end) in enumerate(word_spans):
        scan = sub_index
        while scan < len(clean) and clean[scan][1] <= w_start:
            if clean[scan][0] >= 0:
                sub_index = scan + 1
            scan += 1
        probe = scan
        while probe < len(clean):
            s, e = clean[probe]
            if s < 0:
                probe += 1
                continue
            if s >= w_end:
                break
            if s < w_end and e > w_start:
                result[word_index].append(probe)
            probe += 1
    return result


@dataclass(frozen=True)
class Window:
    """Окно по сабтокенам: ``[start, end)`` и зона, за которую оно отвечает."""

    start: int
    end: int
    trusted_start: int
    trusted_end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def plan_windows(total: int, window: int, overlap: int) -> list[Window]:
    """Разложить ``total`` сабтокенов на перекрывающиеся окна длиной ``window``.

    У каждого окна есть «зона доверия»: часть, значения которой берутся именно
    из него. Края окна (кроме самого первого и последнего) отбрасываются —
    там у токена нет полноценного левого контекста, и признаки внимания
    искажены. Перекрытие ``overlap`` делится пополам между соседями.

    >>> [ (w.start, w.end, w.trusted_start, w.trusted_end) for w in plan_windows(10, 6, 2) ]
    [(0, 6, 0, 5), (4, 10, 5, 10)]
    """
    if total <= 0:
        return []
    window = max(2, int(window))
    overlap = max(0, min(int(overlap), window - 1))
    if total <= window:
        return [Window(0, total, 0, total)]
    step = max(1, window - overlap)
    starts: list[int] = []
    start = 0
    while True:
        starts.append(start)
        if start + window >= total:
            break
        start += step
    # Последнее окно прижимаем к концу, чтобы не осталось хвоста.
    if starts[-1] + window > total:
        starts[-1] = total - window
    windows: list[Window] = []
    for index, start in enumerate(starts):
        end = min(total, start + window)
        trusted_start = start if index == 0 else (starts[index - 1] + window + start) // 2
        if index + 1 < len(starts):
            next_start = starts[index + 1]
            trusted_end = (end + next_start) // 2
        else:
            trusted_end = total
        trusted_start = max(start, min(trusted_start, end))
        trusted_end = max(trusted_start, min(trusted_end, end))
        windows.append(Window(start, end, trusted_start, trusted_end))
    return windows


def merge_window_values(
    total: int,
    windows: Sequence[Window],
    values: Sequence[Sequence[float]],
    default: float = 0.0,
) -> list[float]:
    """Склеить значения по окнам в один ряд длиной ``total``.

    Для каждого сабтокена берётся значение того окна, в чью зону доверия он
    попал; если зоны пересеклись — среднее. Так длинный текст обрабатывается
    без шва на границе окон.
    """
    sums = [0.0] * total
    counts = [0] * total
    for window, row in zip(windows, values, strict=False):
        for position in range(window.trusted_start, window.trusted_end):
            local = position - window.start
            if 0 <= local < len(row) and position < total:
                sums[position] += float(row[local])
                counts[position] += 1
    out: list[float] = []
    for index in range(total):
        out.append(sums[index] / counts[index] if counts[index] else default)
    return out


def batch_sizes(count: int, batch: int) -> list[tuple[int, int]]:
    """Диапазоны пакетов ``[(start, end), …]`` для обработки окон порциями."""
    batch = max(1, int(batch))
    return [(start, min(count, start + batch)) for start in range(0, max(0, count), batch)]
