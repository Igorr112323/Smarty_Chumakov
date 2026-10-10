"""Offline invariants for acceptance infrastructure, not HF quality measurements."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from spanverify.backends.base import BackendUnavailable
from spanverify.core import tokenize_with_offsets
from spanverify.engine import Verifier, WeightsBundle
from spanverify.features import FeatureMatrix, extract_features, feature_cache_key, load_feature_cache

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pipeline = load_script("train_hf_a3")
precompute = load_script("precompute_features")


def pairs():
    return [
        {
            "id": str(i),
            "answer": "Верный текст.",
            "context": "Верный текст.",
            "labels": [],
            "meta": {"doc_id": f"doc-{i // 2}", "question": str(i), "template": str(i)},
        }
        for i in range(12)
    ]


def cache_row(pair, model="model-a"):
    size = len(tokenize_with_offsets(pair["answer"]))
    key = feature_cache_key(pair["answer"], pair["context"], "hf", model)
    matrix = FeatureMatrix(
        attention_entropy=[0.1] * size,
        ctx_attention_mass=[0.8] * size,
        embedding_density=[0.2] * size,
        ctx_sim_contrast=[0.3] * size,
        ctx_sim_margin=[0.2] * size,
        ctx_support_distance=[0.4] * size,
        ctx_similarity_decay=[0.5] * size,
        meta={"model": model, "mode": "hf", "cache_schema": 2},
    )
    return key, matrix


def test_cache_roundtrip_preserves_diagnostics_and_provenance(tmp_path):
    pair = pairs()[0]
    key, matrix = cache_row(pair)
    path = tmp_path / "features_all.jsonl"
    path.write_text(json.dumps(precompute.dump_matrix(key, matrix, pair["id"])) + "\n")
    loaded = load_feature_cache(path)
    for name in pipeline.CANDIDATES:
        assert getattr(loaded[key], name) == getattr(matrix, name)
    assert loaded[key].meta["model"] == "model-a"
    pipeline.validate_cache(loaded, [pair], "model-a", diagnostics=True)


def test_old_cache_remains_readable_but_cannot_be_acceptance_evidence(tmp_path):
    pair = pairs()[0]
    key, matrix = cache_row(pair)
    row = precompute.dump_matrix(key, matrix, pair["id"])
    row.pop("meta")
    path = tmp_path / "features_all.jsonl"
    path.write_text(json.dumps(row) + "\n")
    loaded = load_feature_cache(path)
    assert loaded[key].attention_entropy == matrix.attention_entropy
    with pytest.raises(ValueError, match="provenance missing"):
        pipeline.validate_cache(loaded, [pair], "model-a")


@pytest.mark.parametrize("entrypoint", ["extract", "verifier"])
def test_mismatch_is_explicit_before_dependencies_or_model_loading(entrypoint):
    pair = pairs()[0]
    key, matrix = cache_row(pair)
    with pytest.raises(BackendUnavailable, match="model mismatch"):
        if entrypoint == "extract":
            extract_features(pair["answer"], pair["context"], "hf", cache={key: matrix}, model_name="model-b")
        else:
            Verifier(mode="hf", model_name="model-b", features_cache={key: matrix})


def test_hit_with_corrupt_model_metadata_is_rejected():
    pair = pairs()[0]
    key, matrix = cache_row(pair)
    matrix.meta["model"] = "different-model"
    with pytest.raises(BackendUnavailable, match="model mismatch"):
        extract_features(pair["answer"], pair["context"], "hf", cache={key: matrix}, model_name="model-a")


def test_complete_cache_hit_needs_no_hf_dependencies():
    pair = pairs()[0]
    key, matrix = cache_row(pair)
    assert extract_features(pair["answer"], pair["context"], "hf", cache={key: matrix}, model_name="model-a") is matrix


@pytest.mark.parametrize("corruption", ["missing", "short", "nan", "diagnostic"])
def test_incomplete_or_malformed_cache_is_rejected(corruption):
    pair = pairs()[0]
    key, matrix = cache_row(pair)
    cache = {key: matrix}
    if corruption == "missing":
        cache = {}
    elif corruption == "short":
        matrix.ctx_attention_mass = []
    elif corruption == "nan":
        matrix.attention_entropy[0] = float("nan")
    else:
        matrix.ctx_sim_margin = []
    with pytest.raises(ValueError):
        pipeline.validate_cache(cache, [pair], "model-a", diagnostics=True)


def test_duplicate_cache_key_is_not_silently_overwritten(tmp_path):
    path = tmp_path / "features_all.jsonl"
    path.write_text('{"key":"k", "ctx_attention_mass":[0.1]}\n{"key":"k", "ctx_attention_mass":[0.2]}\n')
    with pytest.raises(ValueError, match="duplicate"):
        load_feature_cache(path)


def test_adapter_never_changes_corpus_and_keeps_document_variants_together():
    original = pairs()
    snapshot = copy.deepcopy(original)
    adapted = pipeline.training_adapter(original)
    from spanverify.train import _split_pairs

    train, val = _split_pairs(adapted, test_size=0.3, seed=42)
    assert original == snapshot
    assert train and val
    assert not ({pipeline.document_id(p) for p in train} & {pipeline.document_id(p) for p in val})


def test_saved_hf_model_identity_is_used_without_changing_config():
    bundle = WeightsBundle(weights={}, threshold=0.5, mode="hf", meta={"hf_model": "model-a"})
    verifier = Verifier(mode="hf", weights=bundle)
    assert verifier.model_name == "model-a"
    assert Verifier(mode="hf", weights=bundle, model_name="model-b").model_name == "model-b"


def test_none_meta_bundle_is_still_compatible():
    verifier = Verifier(mode="hf", weights=WeightsBundle(weights={}, threshold=0.5))
    assert verifier.model_name


def write_dataset(tmp_path, overlap=False):
    records = pairs()
    splits = {"train": records[:4], "dev": records[4:8], "test": records[8:]}
    if overlap:
        splits["test"][0]["meta"]["doc_id"] = "doc-0"
    dataset = tmp_path / "pairs.jsonl"
    dataset.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in records))
    (tmp_path / "splits").mkdir()
    for name, part in splits.items():
        (tmp_path / "splits" / f"{name}.jsonl").write_text(
            "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in part)
        )
    return dataset


def test_official_split_overlap_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        pipeline.load_splits(write_dataset(tmp_path, overlap=True))


def test_official_split_integrity(tmp_path):
    dataset = write_dataset(tmp_path)
    records, splits = pipeline.load_splits(dataset)
    assert sum(map(len, splits.values())) == len(records)
    path = tmp_path / "splits" / "dev.jsonl"
    path.write_text(path.read_text().replace("Верный текст", "Другой текст"))
    with pytest.raises(ValueError, match="not identical"):
        pipeline.load_splits(dataset)


def test_blocked_run_reports_null_and_never_creates_trained_weights(tmp_path, monkeypatch):
    dataset = write_dataset(tmp_path)

    def unavailable(_args):
        raise BackendUnavailable("no local HF weights")

    monkeypatch.setattr(pipeline, "run", unavailable)
    out = tmp_path / "config" / "weights_hf.json"
    reports = tmp_path / "reports"
    assert pipeline.main(["--dataset", str(dataset), "--out", str(out), "--reports-dir", str(reports)]) == 2
    result = json.loads((reports / "hf_a3_run.json").read_text())
    assert result["status"] == "blocked" and result["evaluations"] is None
    assert "no local HF weights" in result["error"]
    assert not out.exists()
    assert "null" in (reports / "hf_a3_eval.md").read_text()


def test_legacy_weights_output_is_forbidden(tmp_path):
    with pytest.raises(SystemExit):
        pipeline.main(["--out", str(pipeline.ROOT / "config" / "weights.json"), "--reports-dir", str(tmp_path)])


def test_safe_json_never_fabricates_undefined_auc():
    assert pipeline.safe_json({"auc": float("nan"), "time": float("inf")}) == {"auc": None, "time": None}


def test_hf_artifacts_are_isolated_and_reloadable(tmp_path):
    from spanverify.train import TrainReport

    config = tmp_path / "config"
    config.mkdir()
    for name in ("weights.json", "head.json", "participation.json", "config.json"):
        (config / name).write_text("legacy contents")
    bundle = WeightsBundle(weights={}, threshold=0.5, mode="hf", head={"type": "none", "file": None})
    report = TrainReport(bundle=bundle, participation={"calibrated_on": "not measured"})
    paths = pipeline.save_hf_artifacts(report, config / "weights_hf.json")
    assert set(paths) == {"weights", "head", "participation"}
    assert json.loads(paths["head"].read_text())["type"] == "none"
    assert json.loads(paths["participation"].read_text())["type"] == "none"
    assert WeightsBundle.load(paths["weights"]).mode == "hf"
    for name in ("weights.json", "head.json", "participation.json", "config.json"):
        assert (config / name).read_text() == "legacy contents"


def test_ablation_document_folds_are_reproducible_and_disjoint(monkeypatch):
    monkeypatch.setitem(sys.modules, "train_hf_a3", pipeline)
    ablation = load_script("hf_feature_ablation")
    folds = ablation.grouped_folds(pairs(), 3, 42)
    assert folds == ablation.grouped_folds(pairs(), 3, 42)
    all_validation = []
    by_id = {p["id"]: pipeline.document_id(p) for p in pairs()}
    for fold in folds:
        assert not {by_id[i] for i in fold["train"]} & {by_id[i] for i in fold["validation"]}
        all_validation.extend(fold["validation"])
    assert sorted(all_validation) == sorted(by_id)
    with pytest.raises(ValueError):
        ablation.grouped_folds(pairs(), 20, 42)


def test_auc_ablation_uses_fold_training_only_scaling(monkeypatch):
    monkeypatch.setitem(sys.modules, "train_hf_a3", pipeline)
    ablation = load_script("hf_feature_ablation")
    train = [[1.0], [3.0]]
    observed = []
    monkeypatch.setattr(ablation, "train_logreg", lambda rows, labels: {"weights": [1.0], "bias": 0.0})

    def predict(_model, rows):
        observed.extend(rows)
        return [0.5] * len(rows)

    monkeypatch.setattr(ablation, "probabilities", predict)
    ablation.fit_predict(train, [0, 1], [[101.0]], [0])
    assert observed == [[99.0]]


def test_auc_ablation_has_all_arms_and_never_promotes_without_followup(monkeypatch):
    monkeypatch.setitem(sys.modules, "train_hf_a3", pipeline)
    ablation = load_script("hf_feature_ablation")
    train_pairs, dev_pairs = pairs()[:8], pairs()[8:]

    def candidate_rows(_verifier, part):
        rows, labels, ids = [], [], []
        for pair in part:
            for label in (0, 1):
                rows.append([float(label)] * 10)
                labels.append(label)
                ids.append(pair["id"])
        return rows, labels, ids

    monkeypatch.setattr(ablation, "candidate_rows", candidate_rows)
    monkeypatch.setattr(ablation, "fit_predict", lambda tr, y, val, cols: [row[0] for row in val])
    result = ablation.ablate(train_pairs, dev_pairs, None, 2, 42, 10)
    assert set(result["arms"]) == {"baseline", "all_candidates", *pipeline.CANDIDATES}
    assert not result["promoted_features"]
    assert not result["test_used_for_selection"]
    assert not result["thresholds_changed"]
    assert all(arm["delta_auc_oof"] == 0 for arm in result["arms"].values())
    assert all(not arm["eligible_for_followup"] for arm in result["arms"].values())


def test_identical_cache_duplicates_across_shards_are_valid(tmp_path):
    path = tmp_path / "features_all.jsonl"
    path.write_text('{"key":"k", "pair_id":"a"}\n{"key":"k", "pair_id":"b"}\n')
    assert len(load_feature_cache(path)) == 1
