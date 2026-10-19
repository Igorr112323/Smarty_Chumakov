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
