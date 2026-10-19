"""P5 pure candidate/filter, strict schemas and frozen runtime integration."""

import copy
import math
from dataclasses import replace
from pathlib import Path

import pytest

from spanverify.core import SpanResult, tokenize_with_offsets
from spanverify.engine import Verifier, WeightsBundle
from spanverify.features import FEATURE_NAMES, FeatureMatrix, is_scored_token
from spanverify.span_filter import FEATURES, build_candidates, feature_vector, filter_candidates, validate_filter


def config(bridge=1, seed=1, threshold=0):
    return dict(
        schema="span_filter_schema_v1",
        feature_order=list(FEATURES),
        generator=dict(tau_low=0.2, tau_high=0.6, bridge=bridge, min_seed=seed),
        head=dict(
            type="standardized_logreg_v1",
            mean=[0.0] * len(FEATURES),
            scale=[1.0] * len(FEATURES),
            coef=[0.0] * len(FEATURES),
            intercept=0.0,
        ),
        keep_threshold=threshold,
    )


@pytest.mark.parametrize("text,risks", [("", []), ("слово", [0.9]), ("слово", [0.1]), ("слово, тест", [0.8, 0.0, 0.7])])
@pytest.mark.parametrize("bridge", [0, 1, 2])
def test_candidate_edges_finite_deterministic(text, risks, bridge):
    ts = tokenize_with_offsets(text)
    scored = [is_scored_token(t.text) for t in ts]
    before = risks[:]
    a = build_candidates(text, ts, risks, scored, config(bridge)["generator"], 0.4)
    assert a == build_candidates(text, ts, risks, scored, config(bridge)["generator"], 0.4)
    assert risks == before
    assert all(
        s.text == text[s.start : s.end] and len(f) == len(FEATURES) and all(math.isfinite(v) for v in f) for s, f in a
    )


def test_no_gold_features():
    x = dict.fromkeys(FEATURES, 0.0)
    for forbidden in ("clean", "dirty", "gold", "label", "truth", "doc_id"):
        with pytest.raises(ValueError):
            feature_vector(x | {forbidden: 1.0})
    with pytest.raises(ValueError):
        feature_vector(x | {"mean_risk": float("nan")})


@pytest.mark.parametrize(
    "change",
    [
        {"schema": "unknown"},
        {"keep_threshold": float("nan")},
        {"feature_order": ["gold"]},
        {"generator": {}},
        {"head": {}},
    ],
)
def test_schema_fail_closed(change):
    with pytest.raises(ValueError):
        validate_filter(config() | change)


def test_empty_is_authoritative_and_rules_unfiltered():
    text = "слово"
    ts = tokenize_with_offsets(text)
    c = config(threshold=math.nextafter(1.0, 2.0))
    candidates = build_candidates(text, ts, [0.9], [True], c["generator"], 0.5)
    assert filter_candidates(text, ts, candidates, c) == (set(), [])
    rule = SpanResult(
        start=0, end=5, text=text, risk=0.9, label="doubtful", source="attribution", sources=("attribution",)
    )
    flags, spans = filter_candidates(text, ts, candidates, c, [rule])
    assert flags == {0} and spans[0].source == "attribution"
    anchor = build_candidates(text, ts, [0.9], [True], c["generator"], 0.5, [rule])[0][1]
    assert anchor[FEATURES.index("exact_anchor")] == 1


def test_doc_cross_fit_guard():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from p4_context_head import fold_partition

    with pytest.raises(ValueError):
        fold_partition([{"doc": "same", "fold": 0}, {"doc": "same", "fold": 1}], 0)


def test_roundtrip_legacy_and_public_pure(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    bundle = WeightsBundle.load(root / "reports/experiments/p4_context_head/run39164044941/combined_bundle.json")
    legacy = Verifier(weights=bundle, mode="demo", span_expansion="mask")
    text = "Сумма составляет 250 рублей."
    context = "Сумма составляет 250 рублей."
    ts = tokenize_with_offsets(text)
    matrix = FeatureMatrix(**{n: [0.4] * len(ts) for n in FEATURE_NAMES}, meta={})
    untouched = copy.deepcopy(matrix)
    monkeypatch.setattr(Verifier, "_features", lambda *args: matrix)
    before = legacy.verify(text, context, with_tokens=True)
    candidate = replace(bundle, span_filter=config(threshold=math.nextafter(1.0, 2.0)))
    candidate.save(tmp_path / "b.json")
    loaded = WeightsBundle.load(tmp_path / "b.json")
    pub = Verifier(weights=loaded, mode="demo", span_expansion="mask")
    after = pub.verify(text, context, with_tokens=True)
    assert before.score == after.score and before.is_hallucination == after.is_hallucination
    assert [t["raw_risk"] for t in before.tokens] == [t["raw_risk"] for t in after.tokens]
    assert not after.spans and not any(t["flagged"] for t in after.tokens)
    assert after.ai_share_hard == 0
    built = build_candidates(
        text,
        ts,
        [t["raw_risk"] for t in after.tokens],
        [is_scored_token(t.text) for t in ts],
        loaded.span_filter["generator"],
        after.score,
    )
    flags, spans = filter_candidates(text, ts, built, loaded.span_filter)
    assert spans == after.spans and flags == set()
    assert matrix == untouched
    bad = candidate.to_dict()
    bad["span_filter"]["head"]["coef"][0] = float("nan")
    with pytest.raises(ValueError):
        WeightsBundle.from_dict(bad)
