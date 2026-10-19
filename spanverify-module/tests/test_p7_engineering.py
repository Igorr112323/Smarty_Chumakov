"""P7 exact arithmetic/offset regressions, independent of HF availability."""

from __future__ import annotations

import random

from spanverify.core import tokenize_with_offsets
from spanverify.features import _is_content, _normalised_support, _span_overlap, map_token_positions


def test_interval_index_and_membership_match_legacy_byte_exact():
    import json

    rng = random.Random(731)
    for _ in range(300):
        spans = [(i, rng.randrange(-20, 100), rng.randrange(-20, 100)) for i in range(50)]
        tokens = [(rng.randrange(-20, 100), rng.randrange(-20, 100)) for _ in range(30)]
        shift = rng.randrange(-10, 10)
        expected = []
        for start, end in tokens:
            hits = [(p, _span_overlap((start + shift, end + shift), (a, b))) for p, a, b in spans]
            hits = [(p, n) for p, n in hits if n > 0]
            cutoff = max(1, max((n for _, n in hits), default=0) // 2)
            expected.append(sorted(p for p, n in hits if n >= cutoff))
        assert map_token_positions(tokens, spans, shift) == expected
    assert map_token_positions([], [], 0) == []
    assert map_token_positions([(0, 1)], [(3, 0, 1)], 0) == [[3]]
    for text in ["", "Один", ", !", "Ёлка 10 лет. " * 1000]:
        tokens = tokenize_with_offsets(text)
        mass = [rng.random() for _ in tokens]
        content = [i for i, t in enumerate(tokens) if _is_content(t)]
        background = sum(mass[i] for i in content) / len(content) if content else 0
        expected = (
            [
                min(1.0, max(0.0, v / max(background, 1e-9))) if i in content else min(1.0, max(0.0, v))
                for i, v in enumerate(mass)
            ]
            if content
            else [0.0] * len(mass)
        )
        assert json.dumps(_normalised_support(mass, tokens)) == json.dumps(expected)


def test_window_partition_preserves_unicode_boundaries_and_all_characters():
    from spanverify.hf_windows import character_windows

    assert character_windows("", [], 1) == []
    text = "ёжик  конец "
    offsets = [(0, 1), (0, 1), (1, 4), (6, 11)]
    spans = character_windows(text, offsets, 1)
    assert "".join(text[a:b] for a, b in spans) == text
    assert all(a < b for a, b in spans)
    assert all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1))


def test_full_answer_stitches_every_token_once_without_zero_fill():
    from dataclasses import fields

    from spanverify.features import FeatureMatrix
    from spanverify.hf_windows import full_answer_features

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}

    def scorer(answer, context, **kwargs):
        count = len(kwargs["answer_tokens"])
        assert answer and kwargs["_windowed"]
        return FeatureMatrix(
            **{f.name: [0.75] * count for f in fields(FeatureMatrix) if f.name not in ("meta", "token_risk")},
            meta={"all_answer_tokens_mapped": True},
        )

    for answer in ["Ё" * 1300, "начало конец " * 150]:
        tokens = tokenize_with_offsets(answer)
        result = full_answer_features(
            answer, "контекст " * 100, tokens, Tokenizer(), None, scorer, {"max_length": 512, "model_name": "fixture"}
        )
        assert result.meta["scored_answer_tokens"] == len(tokens)
        assert result.meta["answer_last_end"] == tokens[-1].end
        assert result.attention_entropy == [0.75] * len(tokens)
        assert result.meta["answer_windows"] > 1


def test_cached_cosine_norms_are_float_exact():
    from spanverify.features import _cosine_dense, _cosine_dense_normed, _dense_cosine_scores_exact

    rng = random.Random(19)
    for a, b in [([], []), ([0.0], [0.0]), ([1.0], [-1.0])] + [
        ([rng.random() - 0.5 for _ in range(768)], [rng.random() - 0.5 for _ in range(768)]) for _ in range(50)
    ]:
        na, nb = sum(x * x for x in a) ** 0.5, sum(x * x for x in b) ** 0.5
        assert _cosine_dense(a, b).hex() == _cosine_dense_normed(a, b, na, nb).hex()

    vectors = [[rng.random() - 0.5 for _ in range(768)] for _ in range(100)]
    query = vectors[0]
    norm = sum(x * x for x in query) ** 0.5
    norms = [sum(x * x for x in row) ** 0.5 for row in vectors]
    actual = _dense_cosine_scores_exact(query, vectors, norm, norms)
    assert [x.hex() for x in actual] == [_cosine_dense(query, row).hex() for row in vectors]
