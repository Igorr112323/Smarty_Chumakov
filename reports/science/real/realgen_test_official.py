#!/usr/bin/env python3
"""One-shot REALGEN official evaluation; sealed inputs are fail-closed.

No test_realgen file is opened unless all three gates pass: explicit --run-now,
FIRE-REALGEN-1 in the process environment, and a READY pre-deployment record
with pinned model artifacts. The workflow adds a fourth gate: a unique
workflow_dispatch on main with no prior official attempt. This module never
fits, tunes, calibrates, or changes a model.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
REAL = ROOT / "reports" / "science" / "real"
PREDEPLOY = REAL / "REALGEN_TEST_PREDEPLOY.json"
FROZEN = REAL / "REALGEN_FROZEN.json"
TEST_ANSWERS = REAL / "REALGEN_ANSWERS_TEST_REALGEN.jsonl"
MODEL_DIR = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "REALGEN_TEST_MODEL_BUNDLE"
EXPECTED_SHIFT_CODE = "FIRE-REALGEN-1"
OFFICIAL_SHIFT_ENV = "REALGEN_OFFICIAL_SHIFT"
OFFICIAL_SHIFT_VALUE = "1"
TRAINED_ON = "realgen_dev_300_only"
MODEL_NAMES = ("composite", "rubert_tiny2_ft", "r2_only")
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20261026
THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
ECE_BINS = 10


class Refusal(RuntimeError):
    """The official run is not authorized or is not safely frozen."""


class ShiftRefusal(Refusal):
    """Missing or wrong shift authorization: exit 3, sealed inputs untouched."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def authorize(run_now: bool, shift_code: str | None) -> dict[str, Any]:
    """Validate only control-plane metadata; does not open sealed test inputs."""
    if not run_now:
        raise Refusal("refusing: explicit --run-now is required")
    # Curator shift gate. This is the FIRST check and it precedes every file
    # read: without REALGEN_OFFICIAL_SHIFT=1 the sealed answers are not opened.
    if not hmac.compare_digest(os.environ.get(OFFICIAL_SHIFT_ENV, ""), OFFICIAL_SHIFT_VALUE):
        raise ShiftRefusal(f"refusing: {OFFICIAL_SHIFT_ENV}=1 is absent; sealed test_realgen file not opened")
    supplied = shift_code if shift_code is not None else os.environ.get("REALGEN_SHIFT_CODE", "")
    if not hmac.compare_digest(supplied, EXPECTED_SHIFT_CODE):
        raise ShiftRefusal("refusing: exact release code is absent or incorrect")
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise Refusal("refusing: official scoring is restricted to GitHub Actions")
    if os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch":
        raise Refusal("refusing: only an explicit workflow_dispatch may score the split")
    if os.environ.get("GITHUB_REF") != "refs/heads/main":
        raise Refusal("refusing: official scoring must run from main")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    if not run_id.isdecimal() or int(run_id) <= 0:
        raise Refusal("refusing: no real GitHub Actions run ID")

    # These are control/provenance files, never the test answers/labels/spans.
    predeploy = json.loads(PREDEPLOY.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    if predeploy.get("status") != "READY_FOR_ONE_OFFICIAL_RUN":
        raise Refusal("refusing: predeployment/model-freeze record is not READY")
    script_sha = _sha256(Path(__file__))
    if script_sha != predeploy.get("scripts", {}).get(Path(__file__).name, {}).get("sha256"):
        raise Refusal("refusing: evaluator SHA256 differs from preregistration")
    expected_test_sha = frozen["files"][TEST_ANSWERS.relative_to(ROOT).as_posix()]["sha256"]
    registered_test = predeploy.get("inputs", {}).get("test_realgen_answers", {})
    if registered_test.get("sha256") != expected_test_sha:
        raise Refusal("refusing: test input hash differs from sealed manifest")
    for relative, info in predeploy.get("inputs", {}).get("dev_and_protocol", {}).items():
        path = ROOT / relative
        if not path.is_file() or _sha256(path) != info["sha256"]:
            raise Refusal(f"refusing: pre-registered input hash mismatch: {relative}")
    frozen_bundle = predeploy.get("frozen_bundle", {})
    if frozen_bundle.get("status") != "FROZEN" or frozen_bundle.get("trained_on") != TRAINED_ON:
        raise Refusal("refusing: frozen_bundle is not a full-dev dev-only REALGEN bundle")
    if frozen_bundle.get("token_threshold") != THRESHOLD:
        raise Refusal("refusing: frozen token threshold differs from the preregistered 0.5")
    if not str(frozen_bundle.get("train_run", {}).get("run_id", "")).isdecimal():
        raise Refusal("refusing: frozen bundle has no real CI train run ID")
    for key in ("seed", "hyperparameters", "models", "bundle", "upstream_revisions"):
        if key not in frozen_bundle:
            raise Refusal(f"refusing: frozen_bundle.{key} is missing")
    if set(("composite", "r2_only", "rubert_tiny2_ft")) - set(frozen_bundle.get("models", {})):
        raise Refusal("refusing: frozen_bundle does not pin all three preregistered models")

    bundle = predeploy.get("model_bundle", {})
    bundle_sha = bundle.get("sha256")
    if frozen_bundle.get("bundle", {}).get("sha256") != bundle_sha:
        raise Refusal("refusing: model_bundle SHA256 differs from frozen_bundle")
    if bundle.get("artifact_name") != frozen_bundle.get("bundle", {}).get("artifact_name"):
        raise Refusal("refusing: Actions artifact provenance differs between bundle records")
    archive = Path(os.environ.get("REALGEN_MODEL_BUNDLE_PATH", bundle.get("path", "")))
    if not bundle_sha or not str(archive) or not archive.is_file() or _sha256(archive) != bundle_sha:
        raise Refusal("refusing: frozen model bundle is missing or its SHA256 differs")
    return {"predeploy": predeploy, "frozen": frozen, "bundle_path": archive, "run_id": run_id}


def _safe_read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _merge_intervals(intervals: list[tuple[int, int]], join_gap: int = 0) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for start, end in sorted((int(a), int(b)) for a, b in intervals if b > a):
        if out and start <= out[-1][1] + join_gap:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return [(a, b) for a, b in out]


def _iou(left: tuple[int, int], right: tuple[int, int]) -> float:
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union else 0.0


def _predicted_spans(tokens: list[dict[str, Any]], probabilities: list[float]) -> list[tuple[int, int]]:
    return _merge_intervals(
        [(token["start"], token["end"]) for token, p in zip(tokens, probabilities, strict=True) if p >= THRESHOLD],
        join_gap=1,
    )


def _span_counts(predicted: list[tuple[int, int]], gold: list[tuple[int, int]]) -> tuple[int, int, int]:
    candidates = sorted(
        ((_iou(p, g), pi, gi) for pi, p in enumerate(predicted) for gi, g in enumerate(gold)),
        reverse=True,
    )
    used_p: set[int] = set()
    used_g: set[int] = set()
    matches = 0
    for score, pi, gi in candidates:
        if score < IOU_THRESHOLD:
            break
        if pi not in used_p and gi not in used_g:
            used_p.add(pi)
            used_g.add(gi)
            matches += 1
    return matches, len(predicted) - matches, len(gold) - matches


def _auc(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return 0.5
    ordered = sorted(zip(scores, labels, strict=True), key=lambda item: item[0])
    rank_sum = 0.0
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        average_rank = ((i + 1) + j) / 2.0
        rank_sum += average_rank * sum(label for _, label in ordered[i:j])
        i = j
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _ece(labels: list[int], scores: list[float]) -> float:
    if not labels:
        return 0.0
    bins = [[] for _ in range(ECE_BINS)]
    for label, score in zip(labels, scores, strict=True):
        index = min(ECE_BINS - 1, max(0, int(score * ECE_BINS)))
        bins[index].append((label, score))
    return sum(
        len(bucket)
        / len(labels)
        * abs(sum(label for label, _ in bucket) / len(bucket) - sum(score for _, score in bucket) / len(bucket))
        for bucket in bins
        if bucket
    )


def _binary_metrics(labels: list[int], scores: list[float]) -> dict[str, float]:
    pred = [int(score >= THRESHOLD) for score in scores]
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, pred, strict=True))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, pred, strict=True))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, pred, strict=True))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, pred, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "auc": _auc(labels, scores),
        "brier": sum((y - p) ** 2 for y, p in zip(labels, scores, strict=True)) / len(labels) if labels else 0.0,
        "ece_10_equal_width": _ece(labels, scores),
    }


def _one_model(
    records: list[dict[str, Any]],
    spans_by_answer: dict[str, list[tuple[int, int]]],
    token_scores: list[list[float]],
    token_offsets: list[list[tuple[int, int]]],
) -> dict[str, Any]:
    token_y: list[int] = []
    token_p: list[float] = []
    span_tp = span_fp = span_fn = 0
    answer_y: list[int] = []
    answer_p: list[float] = []
    per_answer: list[dict[str, Any]] = []

    for rec, scores, offsets in zip(records, token_scores, token_offsets, strict=True):
        aid = rec["answer_id"]
        gold = _merge_intervals(spans_by_answer.get(aid, []))
        if len(scores) != len(offsets):
            raise Refusal(f"predictor token count mismatch for answer id {aid}")
        labels = [int(any(a < end and start < b for a, b in gold)) for start, end in offsets]
        token_y.extend(labels)
        token_p.extend(scores)
        predicted = _predicted_spans([{"start": a, "end": b} for a, b in offsets], scores)
        tp, fp, fn = _span_counts(predicted, gold)
        span_tp += tp
        span_fp += fp
        span_fn += fn
        answer_y.append(int(bool(gold)))
        answer_p.append(max(scores, default=0.0))
        per_answer.append({"token_y": labels, "token_p": scores, "gold": gold, "predicted": predicted})

    token_metrics = _binary_metrics(token_y, token_p)
    answer_metrics = _binary_metrics(answer_y, answer_p)
    span_precision = span_tp / (span_tp + span_fp) if span_tp + span_fp else 0.0
    span_recall = span_tp / (span_tp + span_fn) if span_tp + span_fn else 0.0
    span_metrics = {
        "precision_iou_0_5": span_precision,
        "recall_iou_0_5": span_recall,
        "f1_iou_0_5": (
            2 * span_precision * span_recall / (span_precision + span_recall) if span_precision + span_recall else 0.0
        ),
        "true_positive_spans": span_tp,
        "false_positive_spans": span_fp,
        "false_negative_spans": span_fn,
    }
    return {"token": token_metrics, "span": span_metrics, "answer": answer_metrics, "_per_answer": per_answer}


def _grouped_binary_bootstrap(
    labels: Any, scores: Any, groups: Any, counts: Any, n_groups: int, n_bins: int = ECE_BINS
) -> dict[str, float]:
    """Fast weighted cluster-bootstrap metrics for one binary level."""
    import numpy as np

    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.int32)
    counts = np.asarray(counts, dtype=np.float64)
    predicted = scores >= THRESHOLD
    tp = np.bincount(groups, weights=(labels == 1) & predicted, minlength=n_groups)
    fp = np.bincount(groups, weights=(labels == 0) & predicted, minlength=n_groups)
    fn = np.bincount(groups, weights=(labels == 1) & ~predicted, minlength=n_groups)
    tn = np.bincount(groups, weights=(labels == 0) & ~predicted, minlength=n_groups)
    brier = np.bincount(groups, weights=(labels - scores) ** 2, minlength=n_groups)
    total = (tp + fp + fn + tn) @ counts
    tp_n, fp_n, fn_n, tn_n = tp @ counts, fp @ counts, fn @ counts, tn @ counts
    precision = tp_n / (tp_n + fp_n) if tp_n + fp_n else 0.0
    recall = tp_n / (tp_n + fn_n) if tp_n + fn_n else 0.0

    bin_id = np.minimum(n_bins - 1, np.maximum(0, (scores * n_bins).astype(np.int32)))
    group_bin = groups * n_bins + bin_id
    bin_count = np.bincount(group_bin, minlength=n_groups * n_bins).reshape(n_groups, n_bins)
    bin_pos = np.bincount(group_bin, weights=labels, minlength=n_groups * n_bins).reshape(n_groups, n_bins)
    bin_score = np.bincount(group_bin, weights=scores, minlength=n_groups * n_bins).reshape(n_groups, n_bins)
    bc, by, bs = counts @ bin_count, counts @ bin_pos, counts @ bin_score
    nonempty = bc > 0
    ece = float(
        np.sum(bc[nonempty] / max(total, 1.0) * np.abs(by[nonempty] / bc[nonempty] - bs[nonempty] / bc[nonempty]))
    )
    # The AUC matrix is attached once per data level and injected by the caller.
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "fpr": float(fp_n / (fp_n + tn_n)) if fp_n + tn_n else 0.0,
        "brier": float((brier @ counts) / total) if total else 0.0,
        "ece_10_equal_width": ece,
        "positive_weight": float((tp + fn) @ counts),
        "negative_weight": float((fp + tn) @ counts),
    }


def _auc_pair_matrix(labels: Any, scores: Any, groups: Any, n_groups: int) -> Any:
    """Precompute pairwise positive-vs-negative counts for weighted draws."""
    import numpy as np

    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.int32)
    negative_scores = [np.sort(scores[(labels == 0) & (groups == group)]) for group in range(n_groups)]
    positive_idx = np.flatnonzero(labels == 1)
    matrix = np.zeros((n_groups, n_groups), dtype=np.float64)
    for index in positive_idx:
        score = scores[index]
        positive_group = groups[index]
        for negative_group, values in enumerate(negative_scores):
            left = np.searchsorted(values, score, side="left")
            right = np.searchsorted(values, score, side="right")
            matrix[positive_group, negative_group] += left + 0.5 * (right - left)
    return matrix


def _bootstrap_ci(
    per_answer: list[dict[str, Any]], n_boot: int = BOOTSTRAP_REPLICATES
) -> dict[str, dict[str, list[float]]]:
    """Percentile intervals with 5000 answer/document-cluster resamples."""
    import random

    import numpy as np

    rng = random.Random(BOOTSTRAP_SEED)
    n_groups = len(per_answer)
    token_y = np.asarray([y for answer in per_answer for y in answer["token_y"]], dtype=np.int8)
    token_p = np.asarray([p for answer in per_answer for p in answer["token_p"]], dtype=np.float64)
    token_group = np.concatenate(
        [np.full(len(answer["token_y"]), i, dtype=np.int32) for i, answer in enumerate(per_answer)]
    )
    answer_y = np.asarray([int(bool(answer["gold"])) for answer in per_answer], dtype=np.int8)
    answer_p = np.asarray([max(answer["token_p"], default=0.0) for answer in per_answer], dtype=np.float64)
    answer_group = np.arange(n_groups, dtype=np.int32)
    token_auc_matrix = _auc_pair_matrix(token_y, token_p, token_group, n_groups)
    answer_auc_matrix = _auc_pair_matrix(answer_y, answer_p, answer_group, n_groups)
    span_counts = np.asarray(
        [_span_counts(answer["predicted"], answer["gold"]) for answer in per_answer], dtype=np.float64
    )
    metric_names = {
        "token": ("precision", "recall", "f1", "fpr", "auc", "brier", "ece_10_equal_width"),
        "span": ("precision_iou_0_5", "recall_iou_0_5", "f1_iou_0_5"),
        "answer": ("precision", "recall", "f1", "fpr", "auc", "brier", "ece_10_equal_width"),
    }
    samples: dict[str, dict[str, list[float]]] = {
        level: {name: [] for name in names} for level, names in metric_names.items()
    }
    for _ in range(n_boot):
        counts = np.bincount([rng.randrange(n_groups) for _ in range(n_groups)], minlength=n_groups).astype(np.float64)
        token = _grouped_binary_bootstrap(token_y, token_p, token_group, counts, n_groups)
        answer = _grouped_binary_bootstrap(answer_y, answer_p, answer_group, counts, n_groups)
        for level, result, auc_matrix in (("token", token, token_auc_matrix), ("answer", answer, answer_auc_matrix)):
            denominator = result["positive_weight"] * result["negative_weight"]
            result["auc"] = float((counts @ auc_matrix @ counts) / denominator) if denominator else 0.5
            for name in metric_names[level]:
                samples[level][name].append(result[name])
        tp, fp, fn = (span_counts * counts[:, None]).sum(axis=0)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        span = {
            "precision_iou_0_5": precision,
            "recall_iou_0_5": recall,
            "f1_iou_0_5": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        }
        for name in metric_names["span"]:
            samples["span"][name].append(span[name])
    return {
        level: {
            name: [
                round(float(np.percentile(values, 2.5)), 6),
                round(float(np.percentile(values, 97.5)), 6),
            ]
            for name, values in level_values.items()
        }
        for level, level_values in samples.items()
    }


def run_evaluation(context: dict[str, Any]) -> dict[str, Any]:
    """Load sealed inputs only after authorize() returned a ready, pinned run."""
    import importlib.util
    import zipfile

    # This is the first location in the program permitted to open sealed payloads.
    expected_inputs = context["predeploy"]["inputs"]
    for key, path in (
        ("test_realgen_answers", TEST_ANSWERS),
        ("test_realgen_labels_and_spans", REAL / "REALGEN_SPANS.jsonl"),
        ("test_answer_context_source", REAL / "REALGEN_DOCS.jsonl"),
    ):
        expected = expected_inputs[key]["sha256"]
        if _sha256(path) != expected:
            raise Refusal(f"sealed input hash mismatch for {key}")

    with zipfile.ZipFile(context["bundle_path"]) as bundle_zip:
        for member in bundle_zip.infolist():
            target = (MODEL_DIR / member.filename).resolve()
            if target != MODEL_DIR.resolve() and MODEL_DIR.resolve() not in target.parents:
                raise Refusal("frozen bundle contains an unsafe archive path")
        bundle_zip.extractall(MODEL_DIR)
    records = _safe_read_jsonl(TEST_ANSWERS)
    docs_rows = _safe_read_jsonl(REAL / "REALGEN_DOCS.jsonl")
    docs = {row["doc_id"]: row for row in docs_rows}
    all_spans = _safe_read_jsonl(REAL / "REALGEN_SPANS.jsonl")
    test_ids = {row["answer_id"] for row in records}
    spans_by_answer: dict[str, list[tuple[int, int]]] = {}
    for span in all_spans:
        if span["answer_id"] in test_ids:
            spans_by_answer.setdefault(span["answer_id"], []).append((span["start"], span["end"]))

    predictor_path = MODEL_DIR / "realgen_test_predict.py"
    if not predictor_path.is_file():
        raise Refusal("frozen bundle lacks the registered prediction backend")
    spec = importlib.util.spec_from_file_location("realgen_frozen_predictor", predictor_path)
    if spec is None or spec.loader is None:
        raise Refusal("cannot load the frozen prediction backend")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    predictions, offsets = module.predict_all(records, docs, MODEL_DIR)
    if set(predictions) != set(MODEL_NAMES) or set(offsets) != set(MODEL_NAMES):
        raise Refusal("predictor did not return exactly the three preregistered models")

    from spanverify.core import tokenize_with_offsets

    token_offsets = [
        [(token.start, token.end) for token in tokenize_with_offsets(record["answer"])] for record in records
    ]
    for name in MODEL_NAMES:
        observed = [[(int(a), int(b)) for a, b in row] for row in offsets[name]]
        if observed != token_offsets:
            raise Refusal(f"predictor token offsets differ from the frozen tokenizer for {name}")
    output: dict[str, Any] = {
        "kind": "REALGEN_TEST_OFFICIAL_RESULT",
        "version": "v1",
        "campaign": "REALGEN-R1",
        "run_id": context["run_id"],
        "head_sha": os.environ["GITHUB_SHA"],
        "predeploy_sha256": _sha256(PREDEPLOY),
        "n_answers": len(records),
        "threshold": THRESHOLD,
        "span_iou_threshold": IOU_THRESHOLD,
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "unit": "answer/document cluster", "seed": BOOTSTRAP_SEED},
        "model_definitions": {
            "composite": "frozen 27-feature composite; no test-based fit or recalibration",
            "rubert_tiny2_ft": "frozen rubert-tiny2 fine-tune; mean subtoken-to-word probability",
            "r2_only": "frozen R2-only estimator; explicitly excludes E1 and old5",
        },
        "limitations": [
            "REALGEN labels are rule-generated; lexical feature/label circularity can inflate apparent performance.",
            "A single synthetic generator and 100 held-out answers do not establish transfer to free-form or external generation.",
            "test_realgen metrics do not authorize any use of the permanently locked real_test 1200 split.",
        ],
        "models": {},
    }
    if len(records) != 100:
        raise Refusal(f"sealed split cardinality mismatch: expected 100, got {len(records)}")
    for name in MODEL_NAMES:
        result = _one_model(records, spans_by_answer, predictions[name], token_offsets)
        result["ci95"] = _bootstrap_ci(records, result.pop("_per_answer"))
        output["models"][name] = result

    output["provenance"] = {
        "test_answers_sha256": _sha256(TEST_ANSWERS),
        "test_spans_sha256": _sha256(REAL / "REALGEN_SPANS.jsonl"),
        "real_test_1200_read": False,
        "test_realgen_read_only_after_authorization": True,
    }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-now", action="store_true", help="explicit one-off execution gate")
    args = parser.parse_args()
    try:
        context = authorize(args.run_now, None)
        result = run_evaluation(context)
    except ShiftRefusal as exc:
        # No shift authorization: exit 3, no sealed byte was opened.
        print(str(exc), file=sys.stderr)
        return 3
    except Refusal as exc:
        print(str(exc), file=sys.stderr)
        return 2
    out_dir = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "realgen-official-result"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "REALGEN_TEST_OFFICIAL_RESULT.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "kind": result["kind"],
        "run_id": result["run_id"],
        "n_answers": result["n_answers"],
        "models": result["models"],
    }
    summary_text = "REALGEN official evaluation (one authorized run)\n\n" + json.dumps(
        summary, ensure_ascii=False, indent=2
    )
    summary_path = out_dir / "GITHUB_STEP_SUMMARY.md"
    summary_path.write_text(
        "## REALGEN official evaluation\n\n```json\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as handle:
            handle.write(summary_path.read_text(encoding="utf-8"))
    # The workflow publishes summary_text through a check-run's output.text field.
    print(json.dumps({"result_path": str(result_path), "summary_text": summary_text}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
