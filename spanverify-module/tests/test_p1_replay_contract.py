"""Real entrypoint/schema/parity checks. Fixture HF caches are NOT HF quality evidence."""

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from spanverify.answer_head import AnswerHeadModel, extract_answer_features
from spanverify.calibration import IsotonicCalibrator
from spanverify.core import tokenize_with_offsets
from spanverify.engine import Verifier, WeightsBundle
from spanverify.features import DEFAULT_WEIGHTS, FEATURE_NAMES, feature_cache_key, is_scored_token

ROOT = Path(__file__).resolve().parents[1]


def _module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


builder = _module("tested_p1_builder", "build_p1_candidate.py")
replay = _module("tested_p1_replay", "replay_p1_dev.py")


def test_real_frozen_candidate_cli_and_calibration_schema(tmp_path):
    source = ROOT / "reports/experiments/p1_raw/run39116955685"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    out = tmp_path / "candidate"
    command = [sys.executable, str(ROOT / "scripts/build_p1_candidate.py"), str(source), "--output-dir", str(out)]
    run = subprocess.run(command, cwd=tmp_path, env=env, text=True, capture_output=True)
    assert run.returncode == 0, run.stdout + run.stderr
    bundle = WeightsBundle.load(out / "p1_candidate_bundle.json")
    reference = json.loads((source / "calibrator_p1.json").read_text(encoding="utf-8"))
    assert bundle.isotonic is not None
    assert bundle.isotonic.thresholds == reference["thresholds"]
    assert bundle.isotonic.values == reference["values"]
    assert bundle.threshold == json.loads((source / "thresholds.json").read_text(encoding="utf-8"))["p1"]
    for name, expected in builder.EXPECTED_HASHES.items():
        assert hashlib.sha256((source / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == expected

    empty_cache = tmp_path / "empty-cache"
    empty_cache.mkdir()
    command = [
        sys.executable,
        str(ROOT / "scripts/replay_p1_dev.py"),
        str(source),
        str(empty_cache),
        "--candidate",
        str(out / "p1_candidate_bundle.json"),
        "--runtime-config",
        str(out / "runtime_config.json"),
        "--output-dir",
        str(out),
    ]
    run = subprocess.run(command, cwd=tmp_path, env=env, text=True, capture_output=True)
    assert run.returncode != 0
    assert "No module named 'spanverify'" not in run.stderr
    assert "Feature cache is empty" in run.stderr


def test_frozen_hash_allows_git_crlf_but_not_parameter_changes(tmp_path):
    source = ROOT / "reports/experiments/p1_raw/run39116955685"
    checkout = tmp_path / "windows-checkout"
    checkout.mkdir()
    for name in builder.EXPECTED_HASHES:
        data = (source / name).read_bytes().replace(b"\r\n", b"\n")
        (checkout / name).write_bytes(data.replace(b"\n", b"\r\n"))
    candidate = builder.build_candidate(checkout, tmp_path / "candidate")
    assert WeightsBundle.load(candidate).isotonic is not None
    model = json.loads((checkout / "p1_model.json").read_text(encoding="utf-8"))
    model["weights"][0] += 0.01
    (checkout / "p1_model.json").write_text(json.dumps(model), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        builder.build_candidate(checkout, tmp_path / "changed")


def _constant_head(bias=0.0):
    return AnswerHeadModel(scaler_means=[0.0] * 37, scaler_scales=[1.0] * 37, weights=[0.0] * 37, bias=bias)


def test_public_raw_is_not_calibrated_alias():
    bundle = WeightsBundle(
        weights=dict(DEFAULT_WEIGHTS),
        threshold=0.5,
        mode="demo",
        answer_head=_constant_head().to_dict(),
        isotonic=IsotonicCalibrator([1.0], [0.7]),
    )
    result = Verifier(weights=bundle).verify("Альфа бета.", "Альфа бета.")
    assert result.stats["raw_score_full"] == pytest.approx(0.5)
    assert result.score == pytest.approx(0.7)


def test_whole_bundle_near_threshold_decision_roundtrip(tmp_path):
    value = 1 / 3
    bundle = WeightsBundle(
        weights=dict(DEFAULT_WEIGHTS),
        threshold=value + 1e-9,
        mode="demo",
        answer_head=_constant_head().to_dict(),
        isotonic=IsotonicCalibrator([1.0], [value]),
    )
    before = Verifier(weights=bundle).verify("Альфа бета.", "Альфа бета.")
    path = tmp_path / "bundle.json"
    bundle.save(path)
    restored = WeightsBundle.load(path)
    after = Verifier(weights=restored).verify("Альфа бета.", "Альфа бета.")
    assert before.score == after.score
    assert bundle.threshold == restored.threshold
    assert before.is_hallucination is after.is_hallucination is False


def test_present_corrupted_p1_does_not_fall_back_to_legacy():
    bundle = WeightsBundle(weights=dict(DEFAULT_WEIGHTS), threshold=0.5, answer_head={"schema_version": "unsupported"})
    with pytest.raises(ValueError, match="Invalid P1 answer_head"):
        Verifier(weights=bundle)


@pytest.mark.parametrize(
    "field,value", [("raw_score", 0.2), ("calibrated_score", 0.2), ("prediction", 0), ("gold", 0), ("threshold", 0.8)]
)
def test_parity_detects_per_id_errors(field, value):
    expected = [
        {"pair_id": "a", "gold": 1, "prediction": 1, "raw_score": 0.4, "calibrated_score": 0.6, "threshold": 0.5}
    ]
    actual = copy.deepcopy(expected)
    actual[0][field] = value
    assert replay.compare_records(actual, expected)["matched"] is False


def test_parity_rejects_duplicate_ids():
    row = {"pair_id": "a", "gold": 0, "prediction": 0, "raw_score": 0.1, "calibrated_score": 0.1, "threshold": 0.5}
    with pytest.raises(ValueError, match="Duplicate"):
        replay.compare_records([row, row], [row, row])


def test_auc_constant_scores_is_half():
    assert replay.compute_auc([0, 1, 0, 1], [0.5] * 4) == 0.5


def test_cached_hf_fixture_full_replay_no_backend(tmp_path):
    """Run the actual Verifier path twice from a synthetic cache, without training."""
    source = tmp_path / "fixture-source"
    source.mkdir()
    model = AnswerHeadModel(
        scaler_means=[0.0] * 37, scaler_scales=[1.0] * 37, weights=[0.01 * ((i % 3) - 1) for i in range(37)], bias=0.1
    )
    calibrator = IsotonicCalibrator([0.2, 0.4, 0.6, 1.0], [0.1, 0.3, 0.7, 0.9])
    threshold = 0.5
    pairs = []
    reference = []
    cache_rows = []
    texts = ["Альфа бета.", "Пример документа: 25 рублей.", "Ответ с числом 2б.", "Третий вариант утверждения."]
    for i, text in enumerate(texts):
        label = i % 2
        pair = {
            "id": f"fixture-{i}",
            "context": text,
            "answer": text,
            "labels": [[0, 1, 1]] if label else [],
            "meta": {"doc_id": f"fixture-doc-{i}"},
        }
        pairs.append(pair)
        tokens = tokenize_with_offsets(text)
        columns = {
            name: [0.1 + ((j + k + i) % 6) / 10.0 for j in range(len(tokens))] for k, name in enumerate(FEATURE_NAMES)
        }
        matrix = [{name: columns[name][j] for name in FEATURE_NAMES} for j in range(len(tokens))]
        raw = model.predict(extract_answer_features(matrix, [is_scored_token(t.text) for t in tokens]))
        calibrated = calibrator.transform_one(raw)
        reference.append(
            {
                "pair_id": pair["id"],
                "gold": label,
                "p1_raw": raw,
                "p1_calibrated": calibrated,
                "p1_prediction": int(calibrated >= threshold),
            }
        )
        key = feature_cache_key(text, text, "hf", "ai-forever/rugpt3small_based_on_gpt2", chunking=True, max_length=512)
        cache_rows.append({"key": key, "pair_id": pair["id"], **columns})
    base = WeightsBundle(weights=dict(DEFAULT_WEIGHTS), threshold=0.5, mode="hf").to_dict()
    data = {
        "baseline_bundle.json": base,
        "p1_model.json": model.to_dict(),
        "calibrator_p1.json": calibrator.to_dict(),
        "thresholds.json": {"p1": threshold},
    }
    checks = {}
    for name, obj in data.items():
        path = source / name
        path.write_text(json.dumps(obj), encoding="utf-8")
        checks[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (source / "dev_predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reference), encoding="utf-8")
    (source / "p1_comparison.json").write_text(
        json.dumps({"dev_pairs": 4, "p1": {"fixture_only": True}}), encoding="utf-8"
    )
    dev = tmp_path / "dev.jsonl"
    dev.write_text("".join(json.dumps(p) + "\n" for p in pairs), encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "features_fixture.jsonl").write_text("".join(json.dumps(r) + "\n" for r in cache_rows), encoding="utf-8")
    output = tmp_path / "output"
    candidate = builder.build_candidate(source, output, expected_hashes=checks)
    assert replay.run_replay(source, candidate, output / "runtime_config.json", cache, dev, output) == 0
    parity = json.loads((output / "score_parity_report.json").read_text(encoding="utf-8"))
    assert parity["reference_parity"]["matched"] is True
    assert parity["roundtrip_parity"]["matched"] is True
    manifest = json.loads((output / "runtime_manifest.json").read_text(encoding="utf-8"))
    assert manifest["cache_misses"] == 0
    assert manifest["fresh_hf_inference"] is False
    assert (output / "all_family_metrics.json").is_file()
