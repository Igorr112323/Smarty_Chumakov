#!/usr/bin/env python3
"""Fail-closed public replay of frozen P1 on existing real-HF feature caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from spanverify.config import Config  # noqa: E402
from spanverify.dataset import read_pairs  # noqa: E402
from spanverify.engine import Verifier, WeightsBundle  # noqa: E402
from spanverify.features import (
    feature_cache_key,
    load_feature_cache,
    set_feature_cache,
)  # noqa: E402

TOLERANCE = 1e-12


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path, rows):
    Path(path).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def compute_auc(labels, scores):
    positive = [s for y, s in zip(labels, scores, strict=True) if y == 1]
    negative = [s for y, s in zip(labels, scores, strict=True) if y == 0]
    if not positive or not negative:
        return None
    return sum(float(p > n) + 0.5 * float(p == n) for p in positive for n in negative) / (len(positive) * len(negative))


def metrics(rows):
    tp = sum(row["gold"] == 1 and row["prediction"] == 1 for row in rows)
    fp = sum(row["gold"] == 0 and row["prediction"] == 1 for row in rows)
    fn = sum(row["gold"] == 1 and row["prediction"] == 0 for row in rows)
    tn = sum(row["gold"] == 0 and row["prediction"] == 0 for row in rows)
    labels = [row["gold"] for row in rows]
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "raw_auc": compute_auc(labels, [row["raw_score"] for row in rows]),
        "calibrated_auc": compute_auc(labels, [row["calibrated_score"] for row in rows]),
        "positive_predictions": tp + fp,
    }


def compare_records(actual, expected):
    """Compare identities, full scores and decisions; aggregate counts are insufficient."""
    if len(actual) != len(expected):
        raise ValueError("Record count mismatch")
    actual_ids = [row["pair_id"] for row in actual]
    expected_ids = [row["pair_id"] for row in expected]
    if len(set(actual_ids)) != len(actual_ids) or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Duplicate pair IDs")
    if set(actual_ids) != set(expected_ids):
        raise ValueError("Pair ID sets differ")
    reference = {row["pair_id"]: row for row in expected}
    errors = []
    max_delta = {"raw_score": 0.0, "calibrated_score": 0.0, "threshold": 0.0}
    for row in actual:
        ref = reference[row["pair_id"]]
        for key in ("gold", "prediction"):
            if row[key] != ref[key]:
                errors.append(
                    {
                        "pair_id": row["pair_id"],
                        "field": key,
                        "actual": row[key],
                        "expected": ref[key],
                    }
                )
        for key in max_delta:
            value, target = float(row[key]), float(ref[key])
            if not math.isfinite(value) or not math.isfinite(target):
                raise ValueError(f"Non-finite {key}")
            delta = abs(value - target)
            max_delta[key] = max(max_delta[key], delta)
            if delta > TOLERANCE:
                errors.append(
                    {
                        "pair_id": row["pair_id"],
                        "field": key,
                        "actual": value,
                        "expected": target,
                    }
                )
    return {
        "matched": not errors,
        "mismatches": errors,
        "max_abs_delta": max_delta,
        "pairs": len(actual),
    }


def make_verifier(bundle, runtime, cache):
    config = Config.load().with_overrides(
        backend="hf",
        hf_model=runtime["model_name"],
        hf_max_tokens=int(runtime["hf_max_tokens"]),
    )
    result = Verifier(
        config=config,
        mode="hf",
        model_name=runtime["model_name"],
        weights=bundle,
        features_cache=cache,
        span_expansion=runtime["span_expansion"],
    )
    if result.answer_head_model is None or bundle.isotonic is None:
        raise ValueError("Public loader lost the answer head or calibrator")
    return result


def collect_public_rows(verifier, pairs):
    result = []
    for pair in pairs:
        prediction = verifier.verify(pair["answer"], pair["context"])
        # Full value produced by the public forward pass, never calibrated-as-raw.
        raw_score = float(prediction.stats["raw_score_full"])
        result.append(
            {
                "pair_id": str(pair["id"]),
                "gold": int(any(int(span[2]) == 1 for span in pair.get("labels", []))),
                "raw_score": raw_score,
                "calibrated_score": float(prediction.score),
                "threshold": float(prediction.threshold),
                "prediction": int(prediction.score >= prediction.threshold),
                "verdict": prediction.verdict,
            }
        )
    return result


def run_replay(source_dir, candidate, runtime_path, cache_path, dev_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = read_json(runtime_path)
    if runtime.get("mode") != "hf" or runtime.get("chunking") is not True:
        raise ValueError("Expected frozen HF/chunked runtime configuration")
    references = read_jsonl(source_dir / "dev_predictions.jsonl")
    source_comparison = read_json(source_dir / "p1_comparison.json")
    pairs = list(read_pairs(dev_path))
    if len(pairs) != int(source_comparison["dev_pairs"]):
        raise ValueError("Dev size differs from source measurement")
    pair_ids = [str(pair["id"]) for pair in pairs]
    if len(pair_ids) != len(set(pair_ids)) or set(pair_ids) != {row["pair_id"] for row in references}:
        raise ValueError("Dev identity mismatch")
    threshold = float(read_json(source_dir / "thresholds.json")["p1"])
    expected = [
        {
            "pair_id": row["pair_id"],
            "gold": row["gold"],
            "raw_score": row["p1_raw"],
            "calibrated_score": row["p1_calibrated"],
            "threshold": threshold,
            "prediction": row["p1_prediction"],
        }
        for row in references
    ]
    cache = load_feature_cache([str(cache_path)], counting=True, strict=True)
    if not cache:
        raise ValueError("Feature cache is empty; fresh HF inference is forbidden in this replay")
    set_feature_cache(cache)
    # Same helper/arguments as Verifier._features, not the old broken require-cache preflight.
    missing = []
    for pair in pairs:
        key = feature_cache_key(
            pair["answer"],
            pair["context"],
            "hf",
            runtime["model_name"],
            chunking=True,
            max_length=int(runtime["hf_max_tokens"]),
        )
        if key not in cache:
            missing.append(str(pair["id"]))
    if missing:
        write_json(output_dir / "missing_cache_ids.json", missing)
        raise ValueError(f"Feature cache misses {len(missing)} required dev keys; no model fallback")

    bundle = WeightsBundle.load(candidate)
    if bundle.mode != "hf" or bundle.threshold != threshold:
        raise ValueError("Candidate mode/threshold differs from frozen source")
    calibration = read_json(source_dir / "calibrator_p1.json")
    if (
        bundle.isotonic is None
        or bundle.isotonic.thresholds != calibration["thresholds"]
        or bundle.isotonic.values != calibration["values"]
    ):
        raise ValueError("Candidate calibrator is missing or differs from frozen source")
    if bundle.answer_head != read_json(source_dir / "p1_model.json"):
        raise ValueError("Candidate answer head differs from frozen source")
    before = collect_public_rows(make_verifier(bundle, runtime, cache), pairs)
    roundtrip_path = output_dir / "p1_candidate_roundtrip.json"
    bundle.save(roundtrip_path)
    after_verifier = make_verifier(WeightsBundle.load(roundtrip_path), runtime, cache)
    after = collect_public_rows(after_verifier, pairs)
    reference_parity = compare_records(after, expected)
    roundtrip_parity = compare_records(after, before)
    all_families = after_verifier.evaluate(pairs)
    write_jsonl(output_dir / "public_replay.jsonl", after)
    write_json(output_dir / "all_family_metrics.json", all_families)
    measured_metrics = metrics(after)
    expected_metrics = metrics(expected)
    metric_mismatches = []
    for key, value in expected_metrics.items():
        actual = measured_metrics[key]
        if value is None or actual is None:
            matches = value is actual
        else:
            matches = math.isclose(float(value), float(actual), rel_tol=0.0, abs_tol=TOLERANCE)
        if not matches:
            metric_mismatches.append({"field": key, "actual": actual, "expected": value})
    passed = reference_parity["matched"] and roundtrip_parity["matched"] and not metric_mismatches
    report = {
        "status": "passed" if passed else "failed",
        "metric_mismatches": metric_mismatches,
        "expected_metrics_from_source_evidence": expected_metrics,
        "source_measurement_run_id": 39116955685,
        "source_measurement_commit": "b87850781b73d8630832cb972037158d0ec88927",
        "reference_parity": reference_parity,
        "roundtrip_parity": roundtrip_parity,
        "replay_metrics": measured_metrics,
        "reference_metrics": source_comparison["p1"],
        "token_base_note": "Known historical source token bundle; token/span metrics are measured anew here.",
    }
    write_json(output_dir / "score_parity_report.json", report)
    manifest = {
        "replay_run_id": os.environ.get("GITHUB_RUN_ID", "local-unit-check"),
        "replay_code_commit": os.environ.get("GITHUB_SHA", "local-unit-check"),
        "runtime": runtime,
        "python": sys.version,
        "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "dev_sha256": hashlib.sha256(dev_path.read_bytes()).hexdigest(),
        "pairs": len(pairs),
        "cache_keys": len(cache),
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
        "fresh_hf_inference": False,
        "official_test_evaluated": False,
        "parity_passed": report["status"] == "passed",
        "whole_contract_closed": False,
    }
    write_json(output_dir / "runtime_manifest.json", manifest)
    if cache.misses:
        raise ValueError("Unexpected cache misses during public replay")
    print(json.dumps(report["replay_metrics"], ensure_ascii=False, indent=2))
    print(
        f"Reference mismatches: {len(reference_parity['mismatches'])}; roundtrip mismatches: {len(roundtrip_parity['mismatches'])}"
    )
    return 0 if report["status"] == "passed" else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("features_cache", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--dev", type=Path, default=PACKAGE_ROOT / "data/corpus_a3/splits/dev.jsonl")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    candidate = args.candidate or args.source_dir / "p1_candidate_bundle.json"
    runtime = args.runtime_config or candidate.parent / "runtime_config.json"
    return run_replay(
        args.source_dir,
        candidate,
        runtime,
        args.features_cache,
        args.dev,
        args.output_dir or candidate.parent,
    )


if __name__ == "__main__":
    sys.exit(main())
