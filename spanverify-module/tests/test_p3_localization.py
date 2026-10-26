"""Functional P3 tests; synthetic states are unit fixtures, never quality evidence."""

import copy
from dataclasses import replace
from pathlib import Path

import pytest

from spanverify.core import SpanResult, Token, tokenize_with_offsets
from spanverify.engine import Verifier, WeightsBundle
from spanverify.features import FEATURE_NAMES, FeatureMatrix, is_scored_token
from spanverify.localization import (
    decoder_config,
    localize_state,
    merge_localized_spans,
)


def state(risks, scored=None, config=None):
    text = " ".join(chr(65 + i) for i in range(len(risks)))
    tokens = [Token(text=text[i * 2 : i * 2 + 1], start=i * 2, end=i * 2 + 1) for i in range(len(risks))]
    return localize_state(
        text,
        tokens,
        risks,
        scored or [True] * len(risks),
        config or decoder_config(0.4, 0.8),
    )


def test_growth_and_weak_component():
    risks = [0.1, 0.5, 0.9, 0.6, 0.1, 0.5, 0.6]
    original = risks[:]
    kept, mask, _ = state(risks)
    assert kept == {1, 2, 3} and mask[0].token_indices == (1, 2, 3)
    assert mask[0].text == "B C D" and risks == original


def test_support_count():
    assert state([0.5, 0.9, 0.6], config=decoder_config(0.4, 0.8, seeds=2))[0] == set()
    assert state([0.9, 0.85, 0.6], config=decoder_config(0.4, 0.8, seeds=2))[0] == {
        0,
        1,
        2,
    }


def test_bridge_unscored_not_scored_gap():
    assert state(
        [0.9, 0.0, 0.5],
        scored=[True, False, True],
        config=decoder_config(0.4, 0.8, bridge=1),
    )[
        0
    ] == {0, 2}
    assert state(
        [0.9, 0.0, 0.5],
        scored=[True, False, True],
        config=decoder_config(0.4, 0.8, bridge=0),
    )[
        0
    ] == {0}
    assert state(
        [0.9, 0.0, 0.5],
        scored=[True, True, True],
        config=decoder_config(0.4, 0.8, bridge=1),
    )[
        0
    ] == {0}
    assert state(
        [0.9, 0.0, 0.0, 0.5],
        scored=[True, False, False, True],
        config=decoder_config(0.4, 0.8, bridge=1),
    )[
        0
    ] == {0}


def test_empty_single_number_edges_and_punctuation():
    text = "  17, рубля. "
    tokens = tokenize_with_offsets(text)
    scored = [is_scored_token(t.text) for t in tokens]
    risk = [0.9 if t.text.strip() == "17" else 0.1 for t in tokens]
    kept, mask, spans = localize_state(text, tokens, risk, scored, decoder_config(0.4, 0.8))
    assert len(kept) == 1 and len(spans) == 1 and spans[0].text == "17"
    assert all(s.text == text[s.start : s.end] for s in spans)
    assert localize_state(text, tokens, risk, scored, decoder_config(1.01, 1.01))[0] == set()
    assert localize_state("", [], [], [], decoder_config(0.4, 0.8)) == (set(), [], [])


@pytest.mark.parametrize(
    "change",
    [
        {"type": "unknown"},
        {"schema_version": 2},
        {"tau_low": float("nan")},
        {"tau_high": float("inf")},
        {"tau_high": 0.1},
        {"bridge_unscored_tokens": 2},
        {"min_seed_tokens": 0},
        {"min_seed_tokens": True},
        {"extra": "ignored"},
        {"attribution_boundary": "clause"},
    ],
)
def test_malformed_fails_closed(change):
    bad = {**decoder_config(0.4, 0.8), **change}
    with pytest.raises(ValueError):
        WeightsBundle(weights={}, threshold=0.5, localization_decoder=bad)
    with pytest.raises(ValueError):
        WeightsBundle.from_dict({"weights": {}, "threshold": 0.5, "localization_decoder": bad})
    with pytest.raises(ValueError):
        state([0.9], config=bad)


def test_nan_risk_length_fail():
    with pytest.raises(ValueError):
        state([float("nan")])
    with pytest.raises(ValueError):
        localize_state("A", [Token("A", 0, 1)], [], [True], decoder_config(0.4, 0.8))


def frozen_fixture():
    root = Path(__file__).resolve().parents[1]
    b = WeightsBundle.load(root / "reports/experiments/p2_hf/run39155368906/combined_bundle.json")
    return b


def test_anchor_exact_and_merge_metadata(monkeypatch):
    import spanverify.engine as engine

    answer = "Сумма договора 17 рублей."
    start = answer.index("17")
    end = start + 2
    monkeypatch.setattr(engine, "number_attribution", lambda a, c: [{"start": start, "end": end}])
    b = frozen_fixture()
    tokens = tokenize_with_offsets(answer)
    risk = [0.6] * len(tokens)
    v = Verifier(weights=b, mode="hf", span_expansion="mask")
    p = Verifier(
        weights=replace(b, localization_decoder=decoder_config(0.4, 0.8)),
        mode="hf",
        span_expansion="mask",
    )
    old = v._attribution_spans(answer, "context", tokens, risk, b.token_threshold)
    new = p._attribution_spans(answer, "context", tokens, risk, b.token_threshold)
    assert (new[0].start, new[0].end) == (start, end)
    assert old[0].end - old[0].start > 2 and new[0].source == "attribution"
    assert new[0].risk == old[0].risk
    masks = [
        SpanResult(
            start,
            end + 6,
            answer[start : end + 6],
            0.9,
            "likely_hallucination",
            source="mask",
            token_indices=(3,),
        )
    ]
    merged = merge_localized_spans(answer, tokens, [*masks, *new])
    assert len(merged) == 1 and merged[0].text == answer[merged[0].start : merged[0].end]
    assert set(merged[0].sources) == {"mask", "attribution"} and merged[0].token_indices == (3,)


def test_public_pure_invariance_roundtrip_empty_mask(monkeypatch, tmp_path):
    import spanverify.engine as engine

    monkeypatch.setattr(engine, "number_attribution", lambda a, c: [])
    baseline = frozen_fixture()
    original = copy.deepcopy(baseline.to_dict())
    answer = "Альфа бета гамма."
    tokens = tokenize_with_offsets(answer)
    matrix = FeatureMatrix(**{name: [0.1 + i * 0.1 for i in range(len(tokens))] for name in FEATURE_NAMES})
    saved = copy.deepcopy(matrix)
    config = decoder_config(0.2, 0.3, bridge=1)
    candidate = replace(baseline, localization_decoder=config)
    candidate.save(tmp_path / "p3.json")
    loaded = WeightsBundle.load(tmp_path / "p3.json")

    def result(bundle):
        v = Verifier(weights=bundle, mode="hf", span_expansion="mask")
        monkeypatch.setattr(v, "_features", lambda a, c, t: matrix)
        # Controlled tree-output unit fixture, shared unchanged by P2 and P3.
        monkeypatch.setattr(
            v._token_head_model,
            "predict_many",
            lambda X: [0.8] + [0.4] * (len(X) - 2) + [0.0],
        )
        r = v.verify(answer, "context", with_tokens=True)
        return v, r

    _, base = result(baseline)
    v, p3 = result(candidate)
    _, reload = result(loaded)
    assert baseline.to_dict() == original and matrix == saved
    assert (base.score, base.threshold, base.stats["raw_score_full"]) == (
        p3.score,
        p3.threshold,
        p3.stats["raw_score_full"],
    )
    assert [t["raw_risk"] for t in base.tokens] == [t["raw_risk"] for t in p3.tokens]
    assert p3.tokens == reload.tokens and p3.spans == reload.spans and p3.score == reload.score
    risks = [t["raw_risk"] for t in p3.tokens]
    scored = [is_scored_token(t.text) for t in tokens]
    kept, _, spans = localize_state(answer, tokens, risks, scored, config)
    assert kept == {i for i, t in enumerate(p3.tokens) if t["flagged"]} and spans == p3.spans
    assert p3.ai_share_hard == len(kept) / sum(scored)
    # Growth below legacy tau is not intersected away.
    assert any(r < baseline.token_threshold for i, r in enumerate(risks) if i in kept)
    _, empty = result(replace(candidate, localization_decoder=decoder_config(1.01, 1.01)))
    assert not any(t["flagged"] for t in empty.tokens) and not empty.spans and empty.ai_share_hard == 0
    assert [t["raw_risk"] for t in empty.tokens] == risks
