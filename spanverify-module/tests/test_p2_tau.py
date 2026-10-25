"""Behavior, not field-presence tests for global cutoff and frozen P1."""

import math
from dataclasses import replace
from pathlib import Path

import pytest
from test_p2_public_influence import _make_p2_model

from spanverify.engine import Verifier, WeightsBundle


def test_tau_changes_public_mask_and_roundtrip(tmp_path):
    base = WeightsBundle(weights={}, threshold=0.5, mode="demo", token_head=_make_p2_model())
    low = replace(base, token_threshold=0.1)
    high = replace(base, token_threshold=0.95)
    a = Verifier(weights=low, mode="demo", span_expansion="mask").verify("Альфа бета.", "Альфа бета.", with_tokens=True)
    b = Verifier(weights=high, mode="demo", span_expansion="mask").verify(
        "Альфа бета.", "Альфа бета.", with_tokens=True
    )
    assert [t["raw_risk"] for t in a.tokens] == [t["raw_risk"] for t in b.tokens]
    assert [t["flagged"] for t in a.tokens] != [t["flagged"] for t in b.tokens]
    assert a.spans != b.spans
    assert a.stats["span_threshold"] == 0.1 and b.stats["span_threshold"] == 0.95
    high.save(tmp_path / "bundle.json")
    c = Verifier(
        weights=WeightsBundle.load(tmp_path / "bundle.json"),
        mode="demo",
        span_expansion="mask",
    ).verify("Альфа бета.", "Альфа бета.", with_tokens=True)
    assert b.tokens == c.tokens and b.spans == c.spans
    assert c.stats["span_threshold"] == 0.95


def test_legacy_and_sentinel():
    from spanverify.engine import span_threshold_for

    b = WeightsBundle(weights={}, threshold=0.5, mode="demo")
    v = Verifier(weights=b, mode="demo")
    assert v._span_threshold([0.1, 0.3]) == span_threshold_for([0.1, 0.3], b.span_z, b.span_floor, b.span_cap)
    v.bundle = replace(b, token_threshold=math.nextafter(1.0, math.inf))
    assert v._span_threshold([1.0]) > 1.0
    for bad in [float("nan"), float("inf"), -0.1]:
        v.bundle = replace(b, token_threshold=bad)
        with pytest.raises(ValueError):
            v._span_threshold([0.5])


def test_frozen_bundle_full_roundtrip(tmp_path):
    p = (
        Path(__file__).resolve().parents[1]
        / "reports/experiments/p1_public_replay/run39144872719/p1_candidate_bundle.json"
    )
    b = WeightsBundle.load(p)
    original = b.to_dict()
    c = replace(b, token_head=_make_p2_model(), token_threshold=0.9876543210123456)
    c.save(tmp_path / "combined.json")
    r = WeightsBundle.load(tmp_path / "combined.json")
    assert b.to_dict() == original
    rd = r.to_dict()
    rd["token_head"] = original["token_head"]
    rd["token_threshold"] = original["token_threshold"]
    assert rd == original
    assert r.token_threshold == c.token_threshold


def test_fit_receipt_uses_only_requested_documents(monkeypatch):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import p2_experiment as exp

    calls = []
    monkeypatch.setattr(exp, "train_token_gbdt", lambda X, y: calls.append((X, y)) or "model")
    states = [
        {
            "doc": "fit",
            "X": [[1] * 8, [2] * 8],
            "gold": [1, 0],
            "scored": [True, False],
        },
        {"doc": "held", "X": [[9] * 8], "gold": [0], "scored": [True]},
    ]
    model, receipt = exp.fit_documents(states, {"fit"})
    assert calls == [([[1] * 8], [1])] and model == "model" and receipt["pairs"] == 1
