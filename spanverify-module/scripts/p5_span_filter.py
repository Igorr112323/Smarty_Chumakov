#!/usr/bin/env python3
"""Preregistered P5: immutable P4 OOF risks, one pooled logreg, 5 doc folds.

All candidate cells share ONE head; no per-cell head hyperparameter search.
Diagnostics precede fitting. No dev JSON/rows are read until bundle freeze.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from dataclasses import asdict, replace
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import sklearn
from p3_postprocess import OLD_TAU, distribution, validate_states
from p4_context_head import evidence_metrics, fold_partition, readgz
from replay_p1_dev import compare_records, make_verifier, metrics, read_json, read_jsonl, write_json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import spanverify.engine as engine
from spanverify.answer_head import extract_answer_features
from spanverify.core import tokenize_with_offsets
from spanverify.dataset import read_pairs
from spanverify.engine import WeightsBundle, _iou, _span_f1
from spanverify.features import FEATURE_NAMES, feature_cache_key, load_feature_cache, set_feature_cache
from spanverify.localization import merge_localized_spans
from spanverify.span_filter import (
    FEATURES,
    HIGH_QUANTILES,
    KEEP_QUANTILES,
    LOW_QUANTILES,
    build_candidates,
    filter_candidates,
    probability,
    validate_filter,
)

P4 = Path("reports/experiments/p4_context_head/run39164044941")
P3 = Path("reports/experiments/p3_postprocess/run39160817890")
P2 = Path("reports/experiments/p2_hf/run39155368906")
RECIPE = dict(C=1.0, penalty="l2", class_weight="balanced", solver="lbfgs", max_iter=1000, tol=1e-6, random_state=42)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fit_head(X, y):
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(**RECIPE).fit(scaler.transform(X), y)
    if clf.n_iter_[0] >= RECIPE["max_iter"]:
        raise ValueError("Span head failed convergence")
    h = dict(
        type="standardized_logreg_v1",
        mean=scaler.mean_.tolist(),
        scale=scaler.scale_.tolist(),
        coef=clf.coef_[0].tolist(),
        intercept=float(clf.intercept_[0]),
    )
    # Same fitted object: complete fit-row portable probability parity, no refit.
    actual = np.array([probability(row, h) for row in X])
    delta = float(np.max(np.abs(actual - clf.predict_proba(scaler.transform(X))[:, 1])))
    if delta > 1e-12:
        raise ValueError("Span export drift")
    return h, dict(
        export_max_delta=delta,
        iterations=int(clf.n_iter_[0]),
        fit_rows=len(y),
        positive=int(sum(y)),
        negative=int(len(y) - sum(y)),
        class_weights={str(int(k)): float(len(y) / (2 * sum(y == k))) for k in (0, 1)},
    )


def binary(tp, fp, fn, tn):
    return dict(
        tp=int(tp),
        fp=int(fp),
        fn=int(fn),
        tn=int(tn),
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else 0.0,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        fpr=fp / (fp + tn) if fp + tn else 0.0,
    )


def label_candidates(candidates, truth):
    used = set()
    labels = []
    for s, _ in candidates:
        ranked = [(_iou((s.start, s.end), g), j) for j, g in enumerate(truth) if j not in used]
        best = max(ranked, key=lambda q: (q[0], -q[1]), default=(0.0, -1))
        positive = best[0] >= 0.5
        if positive:
            used.add(best[1])
        labels.append(int(positive))
    return labels


def cfg(generator, head, s):
    return dict(
        schema="span_filter_schema_v1",
        feature_order=list(FEATURES),
        generator=generator,
        head=head,
        keep_threshold=float(s),
    )


def score_cell(states, items, config, probs):
    tp = fp = fn = tn = 0
    pred = []
    for state, (candidates, ids) in zip(states, items, strict=True):
        flags, spans = filter_candidates(
            state["answer"], state["_tokens"], candidates, config, state["_rules"], probs[ids].tolist()
        )
        for i, (gold, scored) in enumerate(zip(state["gold"], state["scored"], strict=True)):
            if not scored:
                continue
            hit = i in flags
            tp += gold and hit
            fp += not gold and hit
            fn += gold and not hit
            tn += not gold and not hit
        pred.append([(s.start, s.end) for s in spans])
    tokens = binary(tp, fp, fn, tn)
    spans = _span_f1(pred, [s["truth"] for s in states])
    return dict(tokens=tokens, spans=spans, objective=min(tokens["f1"], spans["f1"]))


def run(cache_dir, out):
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        head="one pooled standardized logreg shared across all generator cells",
        recipe=RECIPE,
        low_quantiles=LOW_QUANTILES,
        high_quantiles=HIGH_QUANTILES,
        keep_quantiles=KEEP_QUANTILES,
        keep_controls=["all_keep=0", "all_drop=nextafter(1,2)"],
        criterion="max min(token F1,strict span F1), token FPR<=.30, tie smaller FPR then fixed iteration order",
        rule_policy="unfiltered evidence included in strict spans AND token flags",
        label="per-cell previous greedy IoU>=.5, no gold boundary snap",
        official_test_read=False,
        dev_opened=False,
    )
    write_json(out / "preregistered_protocol.json", protocol)
    inv = read_json(P4 / "published_files_sha256.json")["files"]
    names = ["combined_bundle.json", "k3_w1_oof.jsonl.gz", "source_oof_states.jsonl.gz", "source_dev_evidence.jsonl.gz"]
    hashes = {n: sha(P4 / n) for n in names}
    if (
        any(hashes[n] != inv[n]["sha256"] for n in names)
        or hashes["combined_bundle.json"] != "e6c96ea4390dbf124540c413594110180f84d9be9d5c13ad1577232b70ee9e48"
    ):
        raise ValueError("Frozen P4 SHA drift")
    manifest = read_json(Path("data/corpus_a3/manifest.json"))
    train_path = Path("data/corpus_a3/splits/train.jsonl")
    if sha(train_path) != manifest["sha256"]["splits/train.jsonl"]:
        raise ValueError("Train hash")
    if manifest["sha256"]["pairs.jsonl"] != "b9a4cf05a9078644dd539470481a1b8079aed2d9201e195c3ecd901276ff7c8e":
        raise ValueError("Corpus declaration")
    train = list(read_pairs(train_path))
    states = readgz(P4 / "source_oof_states.jsonl.gz")
    write_json(out / "oof_validation.json", validate_states(states, train, read_json(P2 / "fold_receipts.json")))
    risks = readgz(P4 / "k3_w1_oof.jsonl.gz")
    if len(risks) != 842 or {x["id"] for x in risks} != {s["id"] for s in states}:
        raise ValueError("P4 OOF identity")
    byid = {x["id"]: x for x in risks}
    for s in states:
        r = byid[s["id"]]
        if (
            (s["doc"], s["fold"]) != (r["doc"], r["fold"])
            or len(s["smoothed"]) != len(r["smoothed"])
            or any(not math.isfinite(v) or not 0 <= v <= 1 for v in r["smoothed"])
        ):
            raise ValueError("P4 OOF alignment")
        s["smoothed"] = r["smoothed"][:]
    original = {s["id"]: s["smoothed"][:] for s in states}
    shards = list(cache_dir.rglob("features_*.jsonl"))
    if len(shards) != 8 or len({sha(p) for p in shards}) != 8:
        raise ValueError("Need frozen eight shards")
    cache = load_feature_cache([str(p) for p in shards], counting=True, strict=True)
    set_feature_cache(cache)
    frozen = WeightsBundle.load(P4 / "combined_bundle.json")
    runtime = read_json(P4 / "runtime_config.json")
    v = make_verifier(frozen, runtime, cache)
    engine.cover_facts = lru_cache(maxsize=2048)(engine.cover_facts)
    engine.number_attribution = lru_cache(maxsize=2048)(engine.number_attribution)
    for s in states:
        if (
            feature_cache_key(s["answer"], s["context"], "hf", runtime["model_name"], chunking=True, max_length=512)
            not in cache
        ):
            raise ValueError("TRAIN miss")
        f = v._features(s["answer"], s["context"], s["_tokens"])
        rows = [{n: getattr(f, n)[i] for n in FEATURE_NAMES} for i in range(len(s["_tokens"]))]
        s["_p1"] = v._calibrate(v.answer_head_model.predict(extract_answer_features(rows, s["scored"])))
        s["_rules"] = [
            *v._attribution_spans(s["answer"], s["context"], s["_tokens"], s["smoothed"], OLD_TAU),
            *v._coverage_spans(s["answer"], s["context"], s["_tokens"], OLD_TAU),
        ]
    values = [r for s in states for r, m in zip(s["smoothed"], s["scored"], strict=True) if m]
    lows = sorted(set([*np.quantile(values, LOW_QUANTILES).tolist(), frozen.localization_decoder["tau_low"]]))
    highs = sorted(set([*np.quantile(values, HIGH_QUANTILES).tolist(), frozen.localization_decoder["tau_high"]]))
    generators = [
        dict(tau_low=lo, tau_high=hi, bridge=b, min_seed=m)
        for lo in lows
        for hi in highs
        if lo <= hi
        for b in (0, 1, 2)
        for m in (1, 2)
    ]
    write_json(out / "generator_cells.json", generators)
    X = []
    y = []
    folds = []
    cells = []
    diagnostics = []
    with gzip.open(out / "candidate_pool.jsonl.gz", "wt", encoding="utf-8") as pool:
        for cell, g in enumerate(generators):
            items = []
            counts = []
            clean_counts = []
            pergold = []
            any_hit = strict_hit = gold_total = 0
            oracle = []
            oracle_rules = []
            for s in states:
                candidates = build_candidates(
                    s["answer"], s["_tokens"], s["smoothed"], s["scored"], g, s["_p1"], s["_rules"]
                )
                labels = label_candidates(candidates, s["truth"])
                ids = []
                counts.append(len(candidates))
                if not s["truth"]:
                    clean_counts.append(len(candidates))
                for gold in s["truth"]:
                    overlaps = [_iou((c.start, c.end), gold) for c, _ in candidates]
                    any_hit += any(i > 0 for i in overlaps)
                    strict_hit += any(i >= 0.5 for i in overlaps)
                    gold_total += 1
                    pergold.append(sum(i > 0 for i in overlaps))
                oracle.append([(c.start, c.end) for (c, _), label in zip(candidates, labels, strict=True) if label])
                ideal = [c for (c, _), label in zip(candidates, labels, strict=True) if label]
                oracle_rules.append(
                    [(c.start, c.end) for c in merge_localized_spans(s["answer"], s["_tokens"], [*ideal, *s["_rules"]])]
                )
                for (c, row), label in zip(candidates, labels, strict=True):
                    index = len(X)
                    ids.append(index)
                    X.append(row)
                    y.append(label)
                    folds.append(s["fold"])
                    pool.write(
                        json.dumps(
                            dict(
                                index=index,
                                cell=cell,
                                pair_id=s["id"],
                                doc=s["doc"],
                                fold=s["fold"],
                                start=c.start,
                                end=c.end,
                                features=row,
                                label=label,
                            ),
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                items.append((candidates, np.array(ids, dtype=np.int64)))
            cells.append(items)
            diagnostics.append(
                dict(
                    cell=cell,
                    generator=g,
                    gold=gold_total,
                    any_overlap=any_hit,
                    iou_05_coverage=strict_hit,
                    uncovered_any=gold_total - any_hit,
                    uncovered_strict=gold_total - strict_hit,
                    candidates_per_pair=distribution(counts),
                    candidates_per_clean=distribution(clean_counts),
                    candidates_per_gold=distribution(pergold),
                    ideal_keep_diagnostic_only=_span_f1(oracle, [s["truth"] for s in states]),
                    ideal_keep_with_mandatory_rules=_span_f1(oracle_rules, [s["truth"] for s in states]),
                )
            )
            print("candidate cell", cell, "rows", len(X), flush=True)
    write_json(out / "candidate_diagnostics.json", diagnostics)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int32)
    folds = np.asarray(folds)
    if not np.isfinite(X).all() or X.shape[1] != len(FEATURES):
        raise ValueError("Nonfinite/gold feature contamination")
    write_json(
        out / "span_feature_spec.json",
        dict(
            schema="span_filter_schema_v1",
            features=FEATURES,
            no_gold_features=True,
            matched_vs_unmatched={
                str(label): {name: distribution(X[y == label, i].tolist()) for i, name in enumerate(FEATURES)}
                for label in (0, 1)
            },
        ),
    )
    write_json(
        out / "diagnostic_phase_complete.json",
        dict(
            before_fit=True,
            before_dev=True,
            cells=len(cells),
            candidate_rows=len(y),
            seconds=time.perf_counter() - started,
        ),
    )
    probs = np.full(len(y), np.nan)
    receipts = []
    for fold in range(5):
        _, _, receipt = fold_partition(states, fold)
        fit = folds != fold
        held = folds == fold
        head, detail = fit_head(X[fit], y[fit])
        probs[held] = [probability(row, head) for row in X[held]]
        receipt.update(detail)
        receipt["head"] = head
        receipts.append(receipt)
        write_json(out / "fit_receipts.json", receipts)
        print("fit fold", fold, detail, flush=True)
    if not np.isfinite(probs).all():
        raise ValueError("Missing OOF probabilities")
    with gzip.open(out / "oof_span_probabilities.jsonl.gz", "wt") as f:
        for i, p in enumerate(probs):
            f.write(json.dumps(dict(index=i, p=float(p))) + "\n")
    cuts = sorted(set([0.0, math.nextafter(1.0, 2.0), *np.quantile(probs, KEEP_QUANTILES).tolist()]))
    write_json(out / "keep_thresholds.json", cuts)
    dummy = receipts[0]["head"]
    best = None
    best_key = None
    grid_count = 0
    with gzip.open(out / "grid_scores.jsonl.gz", "wt") as grid:
        for cell, (g, items) in enumerate(zip(generators, cells, strict=True)):
            for threshold in cuts:
                config = cfg(g, dummy, threshold)
                score = score_cell(states, items, config, probs)
                row = dict(cell=cell, head="pooled_logreg_C1", keep_threshold=threshold, **score)
                grid.write(json.dumps(row) + "\n")
                grid_count += 1
                key = (score["objective"], -score["tokens"]["fpr"])
                if score["tokens"]["fpr"] <= 0.30 and (best_key is None or key > best_key):
                    best_key = key
                    best = row
            print("scored cell", cell, flush=True)
    if best is None:
        raise ValueError("No feasible configuration")
    write_json(
        out / "selected_config.json",
        dict(
            selected=best,
            generator=generators[best["cell"]],
            criterion=protocol["criterion"],
            grid_points=grid_count,
            source="TRAIN OOF ONLY; not independent acceptance",
        ),
    )
    head, finalfit = fit_head(X, y)
    write_json(out / "final_fit.json", finalfit)
    config = validate_filter(cfg(generators[best["cell"]], head, best["keep_threshold"]))
    candidate = replace(frozen, span_filter=config)
    candidate.save(out / "combined_bundle.json")
    if {k: a for k, a in frozen.to_dict().items() if k != "span_filter"} != {
        k: a for k, a in candidate.to_dict().items() if k != "span_filter"
    }:
        raise ValueError("Frozen P1/P4 fields drift")
    if any(s["smoothed"] != original[s["id"]] for s in states):
        raise ValueError("OOF risk mutated")
    write_json(
        out / "freeze_receipt.json",
        dict(bundle_sha256=sha(out / "combined_bundle.json"), dev_opened=False, seconds=time.perf_counter() - started),
    )
    # FIRST dev row parse occurs below, after all choices and final model frozen.
    dev_path = Path("data/corpus_a3/splits/dev.jsonl")
    if sha(dev_path) != manifest["sha256"]["splits/dev.jsonl"]:
        raise ValueError("Dev source drift")
    dev = list(read_pairs(dev_path))
    corpus = {p["id"]: p for p in dev}
    if len(corpus) != 181 or {p["meta"]["doc_id"] for p in train} & {p["meta"]["doc_id"] for p in dev}:
        raise ValueError("Dev identity/doc leakage")
    reference = {r["id"]: r for r in readgz(P4 / "dev_evidence.jsonl.gz")}
    for p in dev:
        if (
            feature_cache_key(p["answer"], p["context"], "hf", runtime["model_name"], chunking=True, max_length=512)
            not in cache
        ):
            raise ValueError("DEV miss")
    public = make_verifier(candidate, runtime, cache)
    reload = make_verifier(WeightsBundle.load(out / "combined_bundle.json"), runtime, cache)
    evidence = []
    answers = []
    errors = []
    delta = 0.0
    risk_count = 0
    for p in dev:
        result = public.verify(p["answer"], p["context"], with_tokens=True)
        again = reload.verify(p["answer"], p["context"], with_tokens=True)
        if result.tokens != again.tokens or result.spans != again.spans or result.verdict != again.verdict:
            errors.append([p["id"], "reload"])
        expected = reference[p["id"]]["tokens"]
        if len(expected) != len(result.tokens):
            raise ValueError("Full risk length drift")
        delta = max(
            delta, max(abs(a["raw_risk"] - b["raw_risk"]) for a, b in zip(result.tokens, expected, strict=True))
        )
        risk_count += len(expected)
        if any(not math.isfinite(t["raw_risk"]) for t in result.tokens):
            raise ValueError("NaN risk")
        tokens = tokenize_with_offsets(p["answer"])
        risks = [t["raw_risk"] for t in expected]
        scored = [engine.is_scored_token(t.text) for t in tokens]
        rules = [
            *public._attribution_spans(p["answer"], p["context"], tokens, risks, OLD_TAU),
            *public._coverage_spans(p["answer"], p["context"], tokens, OLD_TAU),
        ]
        built = build_candidates(p["answer"], tokens, risks, scored, config["generator"], result.score, rules)
        flags, spans = filter_candidates(p["answer"], tokens, built, config, rules)
        if spans != result.spans or flags != {i for i, t in enumerate(result.tokens) if t["flagged"]}:
            errors.append([p["id"], "pure/public"])
        if any(s.text != p["answer"][s.start : s.end] for s in result.spans):
            errors.append([p["id"], "slice"])
        evidence.append(
            dict(id=p["id"], spans=[asdict(s) for s in result.spans], tokens=result.tokens, verdict=result.verdict)
        )
        answers.append(
            dict(
                pair_id=p["id"],
                gold=int(any(y == 1 for _, _, y in p["labels"])),
                raw_score=result.stats["raw_score_full"],
                calibrated_score=result.score,
                threshold=result.threshold,
                prediction=int(result.is_hallucination),
                verdict=result.verdict,
            )
        )
    p1 = compare_records(answers, read_jsonl(P4 / "dev_answers.jsonl"))
    answer = metrics(answers)
    write_json(
        out / "parity.json",
        dict(
            p1=p1,
            reload_and_pure_mismatches=errors,
            p4_risk_max_delta=delta,
            full_token_risks=risk_count,
            frozen_fields_unchanged=True,
        ),
    )
    if (
        errors
        or delta != 0
        or risk_count != 13150
        or not p1["matched"]
        or [answer[k] for k in ("tp", "fp", "fn", "tn")] != [49, 14, 47, 71]
    ):
        raise ValueError("Dev parity failure")
    with gzip.open(out / "dev_evidence.jsonl.gz", "wt") as f:
        for r in evidence:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_json(out / "dev_answers.json", answers)
    allmetrics = {}
    baselines = {}
    for name, folder, ev, answerfile in [
        ("p2", P2, readgz(P4 / "source_dev_evidence.jsonl.gz"), "dev_answers.jsonl"),
        ("p3", P3, readgz(P3 / "dev_evidence.jsonl.gz"), "dev_answers.jsonl"),
        ("p4", P4, readgz(P4 / "dev_evidence.jsonl.gz"), "dev_answers.jsonl"),
    ]:
        recorded = read_json(folder / "all_family_metrics.json")[name]
        calc = evidence_metrics(ev, corpus, read_jsonl(folder / answerfile), v)
        mismatch = []
        for family in ("tokens", "spans", "answers", "verdicts"):
            for k, val in calc[family].items():
                if k in recorded[family] and isinstance(val, (int, float)) and abs(val - recorded[family][k]) > 1e-12:
                    mismatch.append([family, k])
        if abs(calc["answers"]["calibrated_auc"] - recorded["answers"]["auc"]) > 1e-12:
            mismatch.append(["answers", "auc"])
        baselines[name] = dict(matched=not mismatch, mismatches=mismatch)
        allmetrics[name] = recorded
    write_json(out / "baseline_parity.json", baselines)
    if any(not a["matched"] for a in baselines.values()):
        raise ValueError("Baseline regression")
    actual = evidence_metrics(evidence, corpus, answers, public)
    actual["answers"] = {k: answer[k] for k in ("tp", "fp", "fn", "tn", "precision", "recall", "f1", "fpr")} | {
        "auc": answer["calibrated_auc"],
        "threshold": answers[0]["threshold"],
    }
    allmetrics["p5"] = actual
    write_json(out / "all_family_metrics.json", allmetrics)
    decompose(evidence, corpus, out)
    misses = getattr(cache, "misses", 0)
    if misses:
        raise ValueError("Cache miss")
    write_json(
        out / "manifest.json",
        dict(
            run_id=os.getenv("GITHUB_RUN_ID", "local"),
            code_sha=os.getenv("GITHUB_SHA", "local"),
            input_hashes=hashes,
            cache_shards={p.name: sha(p) for p in shards},
            cache_keys=len(cache),
            cache_hits=getattr(cache, "hits", 0),
            cache_misses=misses,
            span_classifiers_fit=6,
            token_classifiers_fit=0,
            fresh_hf_inference=False,
            official_test_read=False,
            python=platform.python_version(),
            numpy=np.__version__,
            sklearn=sklearn.__version__,
            candidate_rows=len(y),
            grid_points=grid_count,
            total_seconds=time.perf_counter() - started,
            bundle_sha256=sha(out / "combined_bundle.json"),
            joint_gate=actual["tokens"]["f1"] >= 0.5
            and actual["tokens"]["fpr"] <= 0.4
            and actual["spans"]["f1"] >= 0.5,
            verdict_watch_passed=actual["verdicts"]["fpr"] <= 0.30,
        ),
    )
    print("COMPLETE", allmetrics["p5"], flush=True)


def decompose(evidence, corpus, out):
    counts = dict(fp_clean=0, fp_dirty=0, no_overlap=0, overlap_below_05=0, gold_without_overlap=0)
    for r in evidence:
        truth = [(a, b) for a, b, y in corpus[r["id"]]["labels"] if y == 1]
        pred = [(s["start"], s["end"]) for s in r["spans"]]
        counts["fp_dirty" if truth else "fp_clean"] += _span_f1([pred], [truth])["fp"]
        for s in pred:
            best = max((_iou(s, g) for g in truth), default=0.0)
            counts["no_overlap"] += best == 0
            counts["overlap_below_05"] += 0 < best < 0.5
        counts["gold_without_overlap"] += sum(not any(_iou(s, g) > 0 for s in pred) for g in truth)
    write_json(out / "dev_error_decomposition.json", counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        run(args.cache, args.output)
    except Exception as exc:
        write_json(
            args.output / "failure.json",
            dict(error=str(exc), traceback=traceback.format_exc(), official_test_read=False),
        )
        raise
    finally:
        write_json(
            args.output / "file_inventory.json",
            {
                f.name: dict(bytes=f.stat().st_size, sha256=sha(f))
                for f in args.output.iterdir()
                if f.is_file() and f.name != "file_inventory.json"
            },
        )
