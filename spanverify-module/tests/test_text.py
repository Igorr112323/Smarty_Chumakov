"""Тесты токенизации и разбиения на предложения."""

from __future__ import annotations

import pytest

from spanverify.text import (
    char_ngrams,
    normalize_word,
    sentence_spans,
    sentences,
    split_paragraphs,
    tokenize,
    word_tokens,
)


def test_tokenization_covers_text_without_gaps():
    text = "Привет, мир! Как дела — 42 раза?"
    tokens = tokenize(text)
    assert "".join(t.text for t in tokens) == text
    for previous, current in zip(tokens, tokens[1:], strict=False):
        assert previous.end == current.start


def test_tokenization_word_offsets_match_source():
    text = "Раз, два, три."
    for token in tokenize(text):
        if token.is_word:
            assert text[token.start : token.end] == token.text


def test_word_tokens_only_returns_words():
    tokens = word_tokens(tokenize("a, b; c"))
    assert [t.text for t in tokens] == ["a", "b", "c"]


def test_empty_text_gives_no_tokens():
    assert tokenize("") == []
    assert word_tokens(tokenize("   ")) == []


def test_sentences_split_and_keep_text():
    text = "Первое предложение. Второе предложение! Третье?"
    parts = sentences(text)
    assert len(parts) == 3
    assert parts[0][0] == 0
    assert "".join(text[s:e] for s, e in parts).replace(" ", "") == text.replace(" ", "")


def test_sentence_span_indices_point_to_word_tokens():
    text = "Первое предложение. Второе предложение!"
    tokens = tokenize(text)
    spans = sentence_spans(tokens)
    assert len(spans) == 2
    for start, end in spans:
        assert start < end <= len(tokens)
        assert any(tokens[i].is_word for i in range(start, end))


def test_hyphenated_and_numeric_words_are_single_tokens():
    tokens = [t.text for t in word_tokens(tokenize("экс-чемпион 42-й раз"))]
    assert tokens == ["экс-чемпион", "42-й", "раз"]


def test_char_ngrams_have_borders():
    grams = list(char_ngrams("кот", n=3))
    assert grams[0].startswith("^")
    assert grams[-1].endswith("$")
    assert len(grams) == len("кот") + 2 - 3 + 1


def test_normalize_word_folds_case_and_yo():
    assert normalize_word("ЁЖИК") == "ежик"


@pytest.mark.parametrize("n", [2, 3, 4])
def test_char_ngrams_length_matches_formula(n):
    word = "проверка"
    assert len(list(char_ngrams(word, n=n))) == len(word) + 2 - n + 1


def test_split_paragraphs_finds_blocks():
    text = "Абзац один.\n\nАбзац два.\n\nАбзац три."
    spans = split_paragraphs(text)
    assert len(spans) == 3
    assert text[spans[1][0] : spans[1][1]] == "Абзац два."
