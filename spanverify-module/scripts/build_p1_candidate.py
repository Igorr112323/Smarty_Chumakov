#!/usr/bin/env python3
"""Assemble the frozen P1 candidate without fitting or changing its parameters."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from spanverify.answer_head import AnswerHeadModel  # noqa: E402
from spanverify.calibration import IsotonicCalibrator  # noqa: E402
from spanverify.engine import WeightsBundle  # noqa: E402

EXPECTED_HASHES = {
    "p1_model.json": "3292a55433095d6e5611f1a1cb5278d20d751707ef17d0899d681f154960221b",
    "calibrator_p1.json": "65c43c036488cdadcc4d67a5add5408f0ef6d429248907d22537deea9be49183",
    "thresholds.json": "067775a9573f67bd404e2b4308347272c4bc6523ebcd3de4abd0b75db709ebce",
    "baseline_bundle.json": "1dc509c304742b12df77ae9134044247714bc9fd87ca9ac34c4d45ffd95b1637",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_candidate(source_dir: Path, output_dir: Path, expected_hashes=None):
    """The CLI pins real artifact hashes; the optional map supports unit fixtures."""
    expected = EXPECTED_HASHES if expected_hashes is None else expected_hashes
    hashes = {}
    checkout_hashes = {}
    for name, wanted in expected.items():
        path = source_dir / name
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        # Git on Windows may materialize LF JSON as CRLF; no numeric data change.
        source_lf_hash = hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()
        if actual != wanted and source_lf_hash != wanted:
            raise ValueError(f"Frozen source checksum mismatch: {name}")
        hashes[name] = wanted
        checkout_hashes[name] = actual
    base = read_json(source_dir / "baseline_bundle.json")
    if base.get("mode") != "hf" or (base.get("meta") or {}).get("synthetic") is True:
        raise ValueError("A real HF baseline bundle is required")
    answer_payload = read_json(source_dir / "p1_model.json")
    answer_model = AnswerHeadModel.from_dict(answer_payload)
    if answer_model.n_features != 37:
        raise ValueError("Expected the frozen 37-feature P1 schema")
    calibration_payload = read_json(source_dir / "calibrator_p1.json")
    calibrator = IsotonicCalibrator.from_dict(calibration_payload)
    if not calibrator.thresholds or not calibrator.values:
        raise ValueError("Frozen P1 calibrator is empty")
    threshold = float(read_json(source_dir / "thresholds.json")["p1"])
    if not math.isfinite(threshold):
        raise ValueError("Non-finite threshold")
    runtime = {
        "schema": "frozen-p1-runtime-v1",
        "mode": "hf",
        "model_name": "ai-forever/rugpt3small_based_on_gpt2",
        "hf_max_tokens": 512,
        "chunking": True,
        "span_expansion": "clause_capped",
        "cache_source_run_id": 39060670054,
        "answer_target_fpr": 0.30,
        "source_measurement_run_id": 39116955685,
        "source_measurement_commit": "b87850781b73d8630832cb972037158d0ec88927",
    }
    bundle = WeightsBundle.from_dict(base)
    bundle.answer_head = answer_payload
    bundle.isotonic = calibrator
    bundle.threshold = threshold
    historical_meta = dict(bundle.meta or {})
    if "threshold_report" in historical_meta:
        historical_meta["historical_token_base_answer_threshold_report"] = historical_meta.pop("threshold_report")
    bundle.meta = {
        **historical_meta,
        "p1_provenance": {
            **runtime,
            "source_component_sha256": hashes,
            "checkout_component_sha256": checkout_hashes,
            "checksum_newline_rule": "source LF or Git-materialized CRLF; numeric JSON parameters unchanged",
        },
        "token_head_note": "Historical token base from source artifact, not the lost final refit of its script.",
        "runtime_config": runtime,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "p1_candidate_bundle.json"
    bundle.save(bundle_path)
    loaded = WeightsBundle.load(bundle_path)
    if loaded.isotonic is None:
        raise ValueError("WeightsBundle loader lost the answer calibrator")
    if (
        loaded.threshold != threshold
        or loaded.isotonic.thresholds != calibrator.thresholds
        or loaded.isotonic.values != calibrator.values
    ):
        raise ValueError("Candidate serialization changed frozen calibration/threshold")
    if loaded.answer_head != answer_payload:
        raise ValueError("Candidate serialization changed the answer head")
    (output_dir / "runtime_config.json").write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Candidate: {bundle_path}")
    print(f"Calibrator loaded: {len(loaded.isotonic.thresholds)} points; threshold={loaded.threshold!r}")
    return bundle_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    build_candidate(args.source_dir, args.output_dir or args.source_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
