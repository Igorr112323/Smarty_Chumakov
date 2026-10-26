"""P4 pure context windows, export/runtime, frozen P1 and fold isolation."""

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from spanverify.context_head import (
    ContextGBDTModel,
    build_context_features,
    export_context_gbdt,
    feature_spec,
    validate_spec,
)
from spanverify.core import tokenize_with_offsets
from spanverify.engine import Verifier, WeightsBundle
from spanverify.features import FEATURE_NAMES, FeatureMatrix
from spanverify.localization import decoder_config


@pytest.mark.parametrize("k", [1, 2, 3])
def test_scored_clamp_and_pair_boundary(k):
    base = [[float(i)] * 8 for i in (1, 999, 3)]
    frozen = copy.deepcopy(base)
    result = build_context_features(base, [True, False, True], feature_spec(k))
    assert result[0][8] == 1 and result[0][9] == 3
    assert result[2][8] == 1 and result[2][9] == 3
    # Clamped ranges repeat endpoints; an unscored 999 is never a neighbour.
    assert result[0][10:14] == [1, 3, 2, -1]
    assert result[2][10:14] == [1, 3, 2, 1]
    assert result[1][8:14] == [999, 999, 999, 999, 999, 0]
    assert base == frozen
    assert result == build_context_features(base, [True, False, True], feature_spec(k))
    other = build_context_features([[777.0] * 8], [True], feature_spec(k))
    assert other[0][8:14] == [777, 777, 777, 777, 777, 0]
    assert build_context_features([], [], feature_spec(k)) == []


@pytest.mark.parametrize(
    "change",
    [
        {"schema": "unknown"},
        {"k": 0},
        {"k": True},
        {"n_features": 8},
        {"edge": "zero"},
        {"feature_order": ["gold"]},
        {"extra": 1},
    ],
)
def test_spec_fail_closed(change):
    bad = {**feature_spec(1), **change}
    with pytest.raises(ValueError):
        validate_spec(bad)
    with pytest.raises(ValueError):
        WeightsBundle(weights={}, threshold=0.5, context_features_spec=bad)
    with pytest.raises(ValueError):
        build_context_features([[1.0] * 8], [True], bad)


def test_bad_inputs():
    with pytest.raises(ValueError):
        build_context_features([[float("nan")] * 8], [True], feature_spec(1))
    with pytest.raises(ValueError):
        build_context_features([[1.0] * 8], [], feature_spec(1))
    with pytest.raises(ValueError):
        build_context_features([[1.0] * 7], [True], feature_spec(1))


def test_folds_use_source_membership_and_leakage_guard():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from p4_context_head import fold_partition

    states = [
        {"id": "a", "doc": "one", "fold": 4},
        {"id": "b", "doc": "two", "fold": 1},
    ]
    fit, held, r = fold_partition(states, 4)
    assert fit == [states[1]] and held == [states[0]] and r["shared_documents"] == 0
    with pytest.raises(ValueError):
        fold_partition([*states, {"id": "c", "doc": "one", "fold": 1}], 4)


def test_export_runtime_same_sklearn_object(tmp_path):
    from sklearn.ensemble import GradientBoostingClassifier

    rng = np.random.RandomState(42)
    X = rng.rand(100, 56)
    y = (X[:, 20] > 0.5).astype(int)
    clf = GradientBoostingClassifier(
        n_estimators=80,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=20,
        subsample=1,
        init="zero",
        random_state=42,
    ).fit(X, y)
    model = export_context_gbdt(clf, feature_spec(2))
    assert (
        max(abs(a - b) for a, b in zip(model.predict_many(X.tolist()), clf.predict_proba(X)[:, 1], strict=True))
        <= 1e-12
    )
    model.save(tmp_path / "head.json")
    loaded = ContextGBDTModel.load(tmp_path / "head.json")
    assert loaded.predict_many(X.tolist()) == model.predict_many(X.tolist())
    bad = model.to_dict()
    bad["n_features"] = 8
    with pytest.raises(ValueError):
        ContextGBDTModel.from_dict(bad)
    bad = model.to_dict()
    bad["trees"][0] = {
        "feature": 56,
        "threshold": 0.2,
        "left": {"value": 0},
        "right": {"value": 1},
    }
    with pytest.raises(ValueError):
        ContextGBDTModel.from_dict(bad)


def test_public_context_roundtrip_p1_and_unmodified_matrix(monkeypatch, tmp_path):
    from sklearn.ensemble import GradientBoostingClassifier

    import spanverify.engine as engine
    from spanverify.features import is_scored_token
    from spanverify.token_head import extract_token_features

    root = Path(__file__).resolve().parents[1]
    b = WeightsBundle.load(root / "reports/experiments/p3_postprocess/run39160817890/combined_bundle.json")
    original = copy.deepcopy(b.to_dict())
    rng = np.random.RandomState(5)
    X = rng.rand(100, 56)
    y = (X[:, 15] > 0.5).astype(int)
    m = export_context_gbdt(
        GradientBoostingClassifier(n_estimators=3, init="zero", random_state=42).fit(X, y),
        feature_spec(1),
    )
    candidate = replace(
        b,
        token_head=m.to_dict(),
        context_features_spec=feature_spec(1),
        localization_decoder=decoder_config(0.2, 0.5),
    )
    candidate.save(tmp_path / "bundle.json")
    loaded = WeightsBundle.load(tmp_path / "bundle.json")
    text = "Альфа бета гамма."
    tokens = tokenize_with_offsets(text)
    f = FeatureMatrix(**{name: [0.2] * len(tokens) for name in FEATURE_NAMES})
    saved = copy.deepcopy(f)

    def run(bundle):
        v = Verifier(weights=bundle, mode="hf", span_expansion="mask")
        monkeypatch.setattr(v, "_features", lambda a, c, t: f)
        monkeypatch.setattr(engine, "number_attribution", lambda a, c: [])
        return v, v.verify(text, "context", with_tokens=True)

    _, before = run(b)
    v, after = run(candidate)
    _, reloaded = run(loaded)
    assert before.score == after.score and before.stats["raw_score_full"] == after.stats["raw_score_full"]
    assert after.tokens == reloaded.tokens and after.spans == reloaded.spans
    base = [
        extract_token_features(i, len(tokens), t.text, {n: getattr(f, n)[i] for n in FEATURE_NAMES})
        for i, t in enumerate(tokens)
    ]
    pure = engine.smooth_token_risks(
        m.predict_many(build_context_features(base, [is_scored_token(t.text) for t in tokens], feature_spec(1))),
        [is_scored_token(t.text) for t in tokens],
    )
    assert pure == [t["raw_risk"] for t in after.tokens]
    assert b.to_dict() == original and f == saved
    assert WeightsBundle.from_dict(original).context_features_spec is None
    with pytest.raises(ValueError):
        Verifier(weights=replace(candidate, context_features_spec=feature_spec(2)), mode="hf")
