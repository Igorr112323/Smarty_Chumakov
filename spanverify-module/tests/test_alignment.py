"""Сопоставление сабтокенов с символами ответа и окна длинного текста (п. 2.1).

Фикстура ``tests/fixtures/bpe_offsets.json`` получена НАСТОЯЩИМ byte-level BPE
(обучен на текстах реальных актов, см. ``scripts/make_alignment_fixture.py``),
поэтому тест проверяет именно ту ловушку, на которой «съезжал» пилот: ведущий
пробел входит в сабтокен, и наивное сравнение смещений сдвигает признаки на
соседнее слово.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spanverify.alignment import (
    align_subwords,
    batch_sizes,
    merge_window_values,
    plan_windows,
    trim_offsets,
)
from spanverify.core import tokenize_with_offsets

FIXTURE = Path(__file__).parent / "fixtures" / "bpe_offsets.json"


def _samples() -> list[dict]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return payload["samples"]


def test_fixture_is_real_bpe() -> None:
    """Фикстура описывает свой источник: это не придуманные числа."""
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert payload["generator"] == "scripts/make_alignment_fixture.py"
    assert "BPE" in payload["tokenizer"]
    assert payload["corpus_files"], "фикстура должна знать, на чём обучался токенизатор"


def test_trim_offsets_removes_leading_space() -> None:
    """Ведущий пробел сабтокена отбрасывается, спецтокены помечаются (-1, -1)."""
    text = "Срок хранения"
    assert trim_offsets(text, [(0, 4), (4, 13), (0, 0)]) == [(0, 4), (5, 13), (-1, -1)]


@pytest.mark.parametrize("index", range(5))
def test_every_word_gets_its_own_subwords(index: int) -> None:
    """Каждое слово ответа получает сабтокены, покрывающие ровно это слово."""
    sample = _samples()[index]
    text = sample["text"]
    offsets = [(int(a), int(b)) for a, b in sample["offsets"]]
    words = [(t.start, t.end) for t in tokenize_with_offsets(text) if t.word]
    aligned = align_subwords(text, words, offsets)
    assert len(aligned) == len(words)
    for (raw_start, raw_end), indices in zip(words, aligned, strict=True):
        # Токенизатор продукта приклеивает пробелы к токену — для сравнения
        # берём границы самого слова.
        start, end = raw_start, raw_end
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        word = text[start:end]
        if not word or not any(ch.isalnum() for ch in word):
            continue
        assert indices, f"слово {word!r} осталось без сабтокенов"
        # Смещения byte-level BPE на многобайтовых символах («ё») могут
        # перекрываться, поэтому сравнивается покрытый ДИАПАЗОН символов,
        # а не склейка кусочков текста.
        covered_start = min(offsets[i][0] for i in indices)
        covered_end = max(offsets[i][1] for i in indices)
        assert covered_start <= start and covered_end >= end


def test_alignment_is_not_shifted_by_one() -> None:
    """Контрольный случай: признак не уезжает на соседнее слово.

    Берём слово «пять» и убеждаемся, что его сабтокены не пересекают слово
    «составляет» слева и «лет» справа.
    """
    sample = _samples()[0]
    text = sample["text"]
    offsets = [(int(a), int(b)) for a, b in sample["offsets"]]
    words = [(t.start, t.end) for t in tokenize_with_offsets(text) if t.word]
    aligned = align_subwords(text, words, offsets)
    target = next(i for i, (s, e) in enumerate(words) if text[s:e].strip() == "пять")
    start, end = words[target]
    covered_start = min(offsets[i][0] for i in aligned[target])
    covered_end = max(offsets[i][1] for i in aligned[target])
    # Сабтокены слова не залезают на соседние слова больше чем на пробел.
    assert covered_start >= start - 1
    assert covered_end <= end + 1


def test_plan_windows_short_text_is_single_window() -> None:
    """Текст короче окна обрабатывается целиком, без разрезания."""
    windows = plan_windows(100, 512, 64)
    assert len(windows) == 1
    assert (windows[0].start, windows[0].end) == (0, 100)


def test_plan_windows_covers_everything_without_gaps() -> None:
    """Зоны доверия окон покрывают все позиции ровно один раз."""
    total = 1500
    windows = plan_windows(total, 512, 64)
    assert len(windows) > 1
    covered: list[int] = []
    for window in windows:
        covered.extend(range(window.trusted_start, window.trusted_end))
    assert covered == list(range(total))


def test_plan_windows_long_text_respects_window_size() -> None:
    """Ни одно окно не длиннее окна модели — иначе падение на forward."""
    for window in plan_windows(5000, 512, 128):
        assert window.length <= 512


def test_merge_window_values_restores_series() -> None:
    """Склейка окон возвращает исходный ряд значений без шва."""
    total = 20
    source = [float(i) for i in range(total)]
    windows = plan_windows(total, 8, 4)
    rows = [[source[i] for i in range(w.start, w.end)] for w in windows]
    assert merge_window_values(total, windows, rows) == source


def test_merge_window_values_defaults_for_uncovered() -> None:
    """Позиция без окна получает значение по умолчанию, а не исключение."""
    assert merge_window_values(3, [], [], default=0.5) == [0.5, 0.5, 0.5]


def test_batch_sizes() -> None:
    """Окна режутся на пакеты фиксированного размера (обработка при нехватке памяти)."""
    assert batch_sizes(5, 2) == [(0, 2), (2, 4), (4, 5)]
    assert batch_sizes(0, 4) == []


def test_long_sample_is_windowed_and_aligned() -> None:
    """Длинный реальный текст: окна строятся, и слова по-прежнему находятся."""
    sample = _samples()[-1]
    text = sample["text"]
    offsets = [(int(a), int(b)) for a, b in sample["offsets"]]
    assert len(offsets) > 512, "последний образец фикстуры должен быть длиннее окна"
    windows = plan_windows(len(offsets), 512, 64)
    assert len(windows) >= 2
    words = [(t.start, t.end) for t in tokenize_with_offsets(text) if t.word]
    aligned = align_subwords(text, words, offsets)
    assert sum(1 for item in aligned if item) == len(words)
