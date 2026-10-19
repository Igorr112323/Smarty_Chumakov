#!/usr/bin/env python3
"""Fixed P4 CPU experiment: ten models, frozen folds, train-only selection."""

from __future__ import annotations

import gzip
import json
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
from p3_postprocess import (
    BUNDLE_HASH,
    MODEL_HASH,
    OLD_TAU,
    distribution,
    obj_sha,
    score_config,
    search,
    sha,
    validate_states,
)
from replay_p1_dev import (
    collect_public_rows,
    compare_records,
    make_verifier,
    metrics,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

import spanverify.engine as engine
from spanverify.context_head import build_context_features, export_context_gbdt, feature_spec
from spanverify.core import tokenize_with_offsets
from spanverify.dataset import read_pairs
from spanverify.engine import (
    WeightsBundle,
    _containment_recall,
    _mean_width_ratio,
    _span_f1,
    _token_metrics,
    smooth_token_risks,
)
from spanverify.features import FEATURE_NAMES, feature_cache_key, is_scored_token, load_feature_cache, set_feature_cache
from spanverify.localization import localize_state
from spanverify.token_head import extract_token_features, train_token_gbdt

P2 = Path("reports/experiments/p2_hf/run39155368906")
P3 = Path("reports/experiments/p3_postprocess/run39160817890")
RECIPE = dict(
    loss="log_loss",
    n_estimators=80,
    learning_rate=0.05,
    max_depth=3,
    min_samples_leaf=20,
    subsample=1.0,
    init="zero",
    random_state=42,
)
EXPECTED_OOF = "5ca0ff5121707e950a4a710c8d733c092aeca477857ca146847c5de0ef98959b"
EXPECTED_DEV = "e54d290b4894246e61a256af5568af500bc979997d7ab72e7f6ea237129d31ce"


def gzrows(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def readgz(path):
    return [json.loads(x) for x in gzip.decompress(path.read_bytes()).decode("utf-8").splitlines() if x.strip()]


def fold_partition(states, fold):
    fit = [s for s in states if s["fold"] != fold]
    held = [s for s in states if s["fold"] == fold]
    fit_docs = {s["doc"] for s in fit}
    held_docs = {s["doc"] for s in held}
    if not fit or not held or fit_docs & held_docs:
        raise ValueError("P4 document leakage/empty fold")
    return (
        fit,
        held,
        {
            "fold": fold,
            "fit_documents": sorted(fit_docs),
            "held_documents": sorted(held_docs),
            "shared_documents": 0,
            "fit_sha256": obj_sha(sorted(fit_docs)),
            "held_sha256": obj_sha(sorted(held_docs)),
        },
    )


def fit_model(states, k, w):
    X = [row for s in states for row, m in zip(s["_matrix"], s["scored"], strict=True) if m]
    y = [g for s in states for g, m in zip(s["gold"], s["scored"], strict=True) if m]
    if k is None:
        model = train_token_gbdt(X, y)
        return model, {
            "sklearn_export_max_delta": None,
            "fit_tokens": len(y),
            "positive": sum(y),
            "negative": len(y) - sum(y),
            "positive_multiplier": w,
        }
    arr = np.array(X, dtype=np.float32)
    labels = np.array(y, dtype=np.int32)
    weights = compute_sample_weight("balanced", labels) * np.where(labels == 1, float(w), 1.0)
    clf = GradientBoostingClassifier(**RECIPE)
    clf.fit(arr, labels, sample_weight=weights)
    model = export_context_gbdt(clf, feature_spec(k))
    # Same fitted object, every fit input checked, not a separately trained estimator.
    delta = float(np.max(np.abs(clf.predict_proba(arr)[:, 1] - np.array(model.predict_many(X)))))
    if delta > 1e-12:
        raise ValueError(f"Export parity delta {delta}")
    return model, {
        "sklearn_export_max_delta": delta,
        "fit_tokens": len(y),
        "positive": sum(y),
        "negative": len(y) - sum(y),
        "positive_multiplier": w,
    }


def context_diagnostics(states):
    groups = {name: {str(k): {} for k in (1, 2, 3)} for name in ("TP", "FN", "FP_clean", "FP_dirty")}
    errors = {
        "fn_edge_tokens": 0,
        "fn_interior_tokens": 0,
        "gold_wholly_missed": 0,
        "gold_fragmented": 0,
        "fp_clean_tokens": 0,
        "fp_dirty_tokens": 0,
    }
    for s in states:
        indices = [i for i, m in enumerate(s["scored"]) if m]
        r = s["smoothed"]
        flag = {i for i in indices if r[i] >= OLD_TAU}
        edge = set()
        for a, b in s["truth"]:
            covered = [i for i in indices if s["_tokens"][i].start < b and s["_tokens"][i].end > a]
            if covered:
                edge.update((covered[0], covered[-1]))
                errors["gold_wholly_missed"] += not any(i in flag for i in covered)
                parts = 0
                prev = None
                for i in covered:
                    if i in flag:
                        parts += prev is None or i != prev + 1
                        prev = i
                errors["gold_fragmented"] += parts > 1
        for j, i in enumerate(indices):
            gold = s["gold"][i]
            pred = i in flag
            category = (
                "TP"
                if gold and pred
                else "FN" if gold else ("FP_dirty" if s["truth"] else "FP_clean") if pred else None
            )
            if category is None:
                continue
            if category == "FN":
                errors["fn_edge_tokens" if i in edge else "fn_interior_tokens"] += 1
            if category.startswith("FP_"):
                errors["fp_dirty_tokens" if s["truth"] else "fp_clean_tokens"] += 1
            for k in (1, 2, 3):
                neighbours = [indices[max(0, min(len(indices) - 1, j + d))] for d in range(-k, k + 1) if d]
                values = {
                    "risk": [r[t] for t in neighbours],
                    **{
                        name: [s["_base"][t][f] for t in neighbours]
                        for f, name in enumerate(feature_spec(1)["base_order"])
                    },
                }
                for name, v in values.items():
                    for op, value in [("max", max(v)), ("mean", sum(v) / len(v))]:
                        groups[category][str(k)].setdefault(name + ":" + op, []).append(value)
    return {
        "source": "P2 train OOF only before fit/dev; neighbours are scored, same pair, clamp",
        "error_counts": errors,
        "statistics": {
            g: {k: {name: distribution(v) for name, v in d.items()} for k, d in ks.items()} for g, ks in groups.items()
        },
    }


def oracle(states):
    # Exact monotone threshold optimum, labels used ONLY in this diagnostic.
    rows = sorted(
        [(r, g) for s in states for r, g, m in zip(s["smoothed"], s["gold"], s["scored"], strict=True) if m],
        reverse=True,
    )
    positive = sum(g for _, g in rows)
    tp = fp = 0
    best = {"f1": 0.0}
    i = 0
    while i < len(rows):
        threshold = rows[i][0]
        while i < len(rows) and rows[i][0] == threshold:
            tp += rows[i][1]
            fp += 1 - rows[i][1]
            i += 1
        f1 = 2 * tp / (positive + tp + fp)
        if f1 > best["f1"]:
            best = {"tau": threshold, "f1": f1, "tp": tp, "fp": fp, "fn": positive - tp}
    # Ideal merge diagnostic removes false fragments and snaps any hit gold to gold.
    hit = total = 0
    for s in states:
        for a, b in s["truth"]:
            total += 1
            hit += any(
                m and r >= best["tau"] and t.start < b and t.end > a
                for t, r, m in zip(s["_tokens"], s["smoothed"], s["scored"], strict=True)
            )
    return {
        "diagnostic_only": True,
        "not_a_candidate": True,
        "monotone_token_oracle": best,
        "ideal_merge_gold_snap": {
            "definition": "oracle removes false spans, snaps every touched gold to its true boundary; not deployable",
            "tp": hit,
            "fp": 0,
            "fn": total - hit,
            "f1": 2 * hit / (total + hit) if total + hit else 0,
        },
    }


def evidence_metrics(evidence, corpus, answers, verifier):
    labels = []
    flags = []
    risks = []
    pred = []
    truth = []
    verdict_gold = []
    verdict_flags = []
    amap = {r["pair_id"]: r for r in answers}
    for r in evidence:
        pair = corpus[r["id"]]
        tokens = tokenize_with_offsets(pair["answer"])
        gold = [(a, b) for a, b, y in pair["labels"] if y == 1]
        for t, row in zip(tokens, r["tokens"], strict=True):
            if is_scored_token(t.text):
                labels.append(int(any(t.start < b and t.end > a for a, b in gold)))
                flags.append(row["flagged"])
                risks.append(row["raw_risk"])
        pred.append([(s["start"], s["end"]) for s in r["spans"]])
        truth.append(gold)
        verdict_gold.append(bool(gold))
        verdict_flags.append(amap[r["id"]]["verdict"] not in ("grounded", "empty"))
    return {
        "tokens": _token_metrics(labels, flags, risks),
        "spans": {
            **_span_f1(pred, truth),
            "recall_containment": _containment_recall(pred, truth),
            "mean_width_ratio": _mean_width_ratio(pred, truth),
        },
        "answers": metrics(answers),
        "verdicts": engine._binary_metrics(verdict_gold, verdict_flags),
    }


def run(source, cache_dir, out):
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    timings = {}
    provenance = {
        "p2_run": 39155368906,
        "p3_run": 39160817890,
        "p3_code": "36d92c5524c96a7ea7e05ce8490aff94e507d69a",
        "sources": {},
    }
    for name, expected in [
        ("source_oof_states.jsonl.gz", EXPECTED_OOF),
        ("source_dev_evidence.jsonl.gz", EXPECTED_DEV),
    ]:
        file = source / name
        raw = gzip.decompress(file.read_bytes())
        if sha(raw) != expected:
            raise ValueError("Source gzip hash drift")
        (out / name).write_bytes(file.read_bytes())
        provenance["sources"][name] = {"raw_sha256": sha(raw), "gzip_sha256": sha(file.read_bytes()), "bytes": len(raw)}
    write_json(out / "source_provenance.json", provenance)
    if (
        sha((P2 / "p2_token_head.json").read_bytes()) != MODEL_HASH
        or sha((P2 / "combined_bundle.json").read_bytes()) != BUNDLE_HASH
    ):
        raise ValueError("Frozen model hash mismatch")
    manifest = read_json(Path("data/corpus_a3/manifest.json"))
    if manifest["sha256"]["pairs.jsonl"] != "b9a4cf05a9078644dd539470481a1b8079aed2d9201e195c3ecd901276ff7c8e":
        raise ValueError("Corpus declaration mismatch")
    train_path = Path("data/corpus_a3/splits/train.jsonl")
    if sha(train_path.read_bytes()) != manifest["sha256"]["splits/train.jsonl"]:
        raise ValueError("Train hash mismatch")
    train = list(read_pairs(train_path))
    states = readgz(source / "source_oof_states.jsonl.gz")
    write_json(out / "oof_validation.json", validate_states(states, train, read_json(P2 / "fold_receipts.json")))
    shards = list(cache_dir.rglob("features_*.jsonl"))
    if len(shards) != 8 or len({sha(p.read_bytes()) for p in shards}) != 8:
        raise ValueError("Need 8 distinct cache shards")
    cache = load_feature_cache([str(p) for p in shards], counting=True, strict=True)
    set_feature_cache(cache)
    p3bundle = WeightsBundle.load(P3 / "combined_bundle.json")
    runtime = read_json(P3 / "runtime_config.json")
    v = make_verifier(p3bundle, runtime, cache)
    engine.cover_facts = lru_cache(maxsize=2048)(engine.cover_facts)
    engine.number_attribution = lru_cache(maxsize=2048)(engine.number_attribution)
    for s in states:
        key = feature_cache_key(s["answer"], s["context"], "hf", runtime["model_name"], chunking=True, max_length=512)
        if key not in cache:
            raise ValueError("Missing TRAIN cache " + s["id"])
        f = v._features(s["answer"], s["context"], s["_tokens"])
        if any(len(getattr(f, n)) != len(s["_tokens"]) for n in FEATURE_NAMES):
            raise ValueError("Incomplete HF token sequence")
        s["_base"] = [
            extract_token_features(i, len(s["_tokens"]), t.text, {n: getattr(f, n)[i] for n in FEATURE_NAMES})
            for i, t in enumerate(s["_tokens"])
        ]
        s["_attr"] = v._attribution_spans(s["answer"], s["context"], s["_tokens"], s["smoothed"], OLD_TAU)
        s["_coverage"] = v._coverage_spans(s["answer"], s["context"], s["_tokens"], OLD_TAU)
    write_json(out / "context_diagnostics.json", context_diagnostics(states))
    timings["validation_and_pre_fit_diagnostics"] = time.perf_counter() - started
    write_json(
        out / "feature_specs.json",
        {
            "base_five_HF_columns": list(FEATURE_NAMES),
            "contract_note": "56 inputs are deterministic derivatives of the same 5 HF columns plus unchanged position/length/digit; not new HF measurements",
            "variants": [feature_spec(k) for k in (1, 2, 3)],
        },
    )
    original_risks = {s["id"]: s["smoothed"][:] for s in states}
    scores = []
    receipts = []
    best = None
    best_risks = None
    candidates = [(None, 1)] + [(k, w) for k in (1, 2, 3) for w in (1, 2, 4)]
    write_json(
        out / "selection_protocol.json",
        {
            "candidates": candidates,
            "class_weights": "fit-only balanced sample weights times positive multiplier w; control matches P2",
            "recipe": RECIPE,
            "model_selection": "max min(token F1, strict span F1), FPR<=.30, tie lower FPR under FROZEN P3 decoder; then registered 1300-point procedure on winner only",
            "dev_tuning": False,
        },
    )
    for k, w in candidates:
        tick = time.perf_counter()
        name = f"k{k}_w{w}"
        for s in states:
            s["_matrix"] = s["_base"] if k is None else build_context_features(s["_base"], s["scored"], feature_spec(k))
        predicted = {}
        for fold in range(5):
            fit, held, receipt = fold_partition(states, fold)
            model, details = fit_model(fit, k, w)
            receipt.update(candidate=name, model_sha256=obj_sha(model.to_dict()), **details)
            receipts.append(receipt)
            for s in held:
                if s["id"] in predicted:
                    raise ValueError("Duplicate held token state")
                s["smoothed"] = smooth_token_risks(model.predict_many(s["_matrix"]), s["scored"])
                predicted[s["id"]] = s["smoothed"][:]
                # Refresh rule confidence on new risks, not its condition/anchors.
                s["_attr"] = v._attribution_spans(s["answer"], s["context"], s["_tokens"], s["smoothed"], OLD_TAU)
            write_json(out / "fit_receipts.json", receipts)
            print(name, "fold", fold, "complete", flush=True)
        if len(predicted) != 842:
            raise ValueError("OOF coverage gap")
        if k is None:
            delta = max(abs(a - b) for s in states for a, b in zip(s["smoothed"], original_risks[s["id"]], strict=True))
            p2v = make_verifier(
                WeightsBundle.load(P2 / "combined_bundle.json"), read_json(P2 / "runtime_config.json"), cache
            )
            from p2_experiment import selection

            control = selection(p2v, states, OLD_TAU)
            expected = read_json(P2 / "tau_selection.json")["selected"]
            checks = {
                key: abs(control["tokens"][key] - expected["tokens"][key]) <= 1e-12
                for key in ("f1", "fpr", "tp", "fp", "fn", "tn")
            }
            checks["span_f1"] = abs(control["spans"]["f1"] - expected["spans"]["f1"]) <= 1e-12
            write_json(
                out / "p2_control_parity.json",
                {"max_risk_delta": delta, "checks": checks, "actual": control, "expected": expected},
            )
            if delta > 1e-12 or not all(checks.values()):
                raise ValueError("P2 control mismatch: stop, no retuning")
        score = score_config(states, p3bundle.localization_decoder)
        score.update(k=k, w=w, candidate=name, seconds=time.perf_counter() - tick)
        scores.append(score)
        gzrows(
            out / (name + "_oof.jsonl.gz"),
            [{"id": s["id"], "doc": s["doc"], "fold": s["fold"], "smoothed": s["smoothed"]} for s in states],
        )
        write_json(out / "model_grid_scores.json", scores)
        if score["tokens"]["fpr"] <= 0.30 and (
            best is None or (score["objective"], -score["tokens"]["fpr"]) > (best["objective"], -best["tokens"]["fpr"])
        ):
            best = score
            best_risks = predicted
        print("Candidate", json.dumps(score), flush=True)
    write_json(out / "selected_model.json", best)
    timings["all_10_models_oof"] = sum(s["seconds"] for s in scores)
    for s in states:
        s["smoothed"] = best_risks[s["id"]]
        s["_attr"] = v._attribution_spans(s["answer"], s["context"], s["_tokens"], s["smoothed"], OLD_TAU)
    write_json(out / "oof_oracle_diagnostic.json", oracle(states))
    tick = time.perf_counter()
    selected_decoder = search(states, out)
    timings["registered_decoder_selection"] = time.perf_counter() - tick
    config = selected_decoder["config"]
    k, w = best["k"], best["w"]
    for s in states:
        s["_matrix"] = s["_base"] if k is None else build_context_features(s["_base"], s["scored"], feature_spec(k))
    tick = time.perf_counter()
    model, final_receipt = fit_model(states, k, w)
    timings["final_fit"] = time.perf_counter() - tick
    write_json(out / "final_fit.json", final_receipt)
    model.save(out / "p4_token_head.json")
    candidate = replace(
        p3bundle,
        token_head=model.to_dict(),
        context_features_spec=feature_spec(k) if k else None,
        localization_decoder=config,
    )
    candidate.save(out / "combined_bundle.json")
    p4runtime = {**runtime, "localization_decoder": config, "context_features_spec": candidate.context_features_spec}
    write_json(out / "runtime_config.json", p4runtime)
    orig = p3bundle.to_dict()
    changed = candidate.to_dict()
    allowed = {"token_head", "context_features_spec", "localization_decoder"}
    if {a: b for a, b in orig.items() if a not in allowed} != {a: b for a, b in changed.items() if a not in allowed}:
        raise ValueError("P1 fields mutated")
    # Dev remains closed until model, decoder and bundle are all written.
    dev_path = Path("data/corpus_a3/splits/dev.jsonl")
    if sha(dev_path.read_bytes()) != manifest["sha256"]["splits/dev.jsonl"]:
        raise ValueError("Dev hash mismatch")
    dev = list(read_pairs(dev_path))
    corpus = {p["id"]: p for p in dev}
    if len(dev) != 181 or len(corpus) != 181:
        raise ValueError("Need 181 unique dev pairs")
    for p in dev:
        if (
            feature_cache_key(p["answer"], p["context"], "hf", runtime["model_name"], chunking=True, max_length=512)
            not in cache
        ):
            raise ValueError("Missing DEV cache " + p["id"])
    historical = {}
    for label, evidence, answers, source_metrics in [
        (
            "p2",
            readgz(source / "source_dev_evidence.jsonl.gz"),
            read_jsonl(P2 / "dev_answers.jsonl"),
            read_json(P2 / "all_family_metrics.json")["p2"],
        ),
        (
            "p3",
            readgz(P3 / "dev_evidence.jsonl.gz"),
            read_jsonl(P3 / "dev_answers.jsonl"),
            read_json(P3 / "all_family_metrics.json")["p3"],
        ),
    ]:
        calc = evidence_metrics(evidence, corpus, answers, v)
        errors = []
        for family in ("tokens", "spans", "verdicts", "answers"):
            for field in (
                "tp",
                "fp",
                "fn",
                "tn",
                "precision",
                "recall",
                "f1",
                "fpr",
                "auc",
                "recall_containment",
                "mean_width_ratio",
            ):
                if (
                    field in source_metrics[family]
                    and field in calc[family]
                    and abs(calc[family][field] - source_metrics[family][field]) > 1e-12
                ):
                    errors.append([family, field])
        if abs(calc["answers"]["calibrated_auc"] - source_metrics["answers"]["auc"]) > 1e-12:
            errors.append(["answers", "auc"])
        historical[label] = {"matched": not errors, "mismatches": errors, "metrics": calc}
    write_json(out / "historical_evidence_parity.json", historical)
    if any(not h["matched"] for h in historical.values()):
        raise ValueError("Historical metric regression")
    pv = make_verifier(candidate, p4runtime, cache)
    rv = make_verifier(WeightsBundle.load(out / "combined_bundle.json"), p4runtime, cache)
    tick = time.perf_counter()
    before = collect_public_rows(pv, dev)
    after = collect_public_rows(rv, dev)
    parity = {
        "p1": compare_records(
            before, read_jsonl(Path("reports/experiments/p1_public_replay/run39144872719/public_replay.jsonl"))
        ),
        "reload": compare_records(after, before),
    }
    write_json(out / "p1_parity.json", parity)
    write_jsonl(out / "dev_answers.jsonl", before)
    answer = metrics(before)
    if any(not r["matched"] for r in parity.values()) or [answer[t] for t in ("tp", "fp", "fn", "tn")] != [
        49,
        14,
        47,
        71,
    ]:
        raise ValueError("P1 drift")
    evidence = []
    errors = []
    for p in dev:
        result = pv.verify(p["answer"], p["context"], with_tokens=True)
        reloaded = rv.verify(p["answer"], p["context"], with_tokens=True)
        if result.tokens != reloaded.tokens or result.spans != reloaded.spans or result.verdict != reloaded.verdict:
            errors.append(p["id"])
        tokens = tokenize_with_offsets(p["answer"])
        f = pv._features(p["answer"], p["context"], tokens)
        base = [
            extract_token_features(i, len(tokens), t.text, {n: getattr(f, n)[i] for n in FEATURE_NAMES})
            for i, t in enumerate(tokens)
        ]
        scored = [is_scored_token(t.text) for t in tokens]
        matrix = build_context_features(base, scored, candidate.context_features_spec) if k else base
        risks = smooth_token_risks(model.predict_many(matrix), scored)
        delta = max(abs(a - b["raw_risk"]) for a, b in zip(risks, result.tokens, strict=True))
        kept, _, spans = localize_state(
            p["answer"],
            tokens,
            risks,
            scored,
            config,
            pv._attribution_spans(p["answer"], p["context"], tokens, risks, OLD_TAU),
            pv._coverage_spans(p["answer"], p["context"], tokens, OLD_TAU),
        )
        if (
            delta > 1e-12
            or spans != result.spans
            or kept != {i for i, t in enumerate(result.tokens) if t["flagged"]}
            or not all(np.isfinite(risks))
        ):
            errors.append(p["id"])
        for span in result.spans:
            if span.text != p["answer"][span.start : span.end]:
                raise ValueError("Span slice mismatch")
        evidence.append(
            {
                "id": p["id"],
                "gold": p["labels"],
                "tokens": result.tokens,
                "spans": [asdict(s) for s in result.spans],
                "score": result.score,
                "threshold": result.threshold,
                "verdict": result.verdict,
            }
        )
    gzrows(out / "dev_evidence.jsonl.gz", evidence)
    write_json(
        out / "runtime_parity.json",
        {
            "pairs": 181,
            "mismatches": errors,
            "checks": ["reload", "pure context builder", "pure decoder", "finite raw risks", "span slices"],
        },
    )
    if errors:
        raise ValueError("P4 runtime parity failure")
    metrics4 = pv.evaluate(dev)
    allmetrics = {
        "p2": read_json(P2 / "all_family_metrics.json")["p2"],
        "p3": read_json(P3 / "all_family_metrics.json")["p3"],
        "p4": metrics4,
    }
    write_json(out / "all_family_metrics.json", allmetrics)
    timings["dev_replay_and_parity"] = time.perf_counter() - tick
    # Error decomposition after frozen dev evaluation, never used to select.
    decomposition = {"fp_clean": 0, "fp_dirty": 0, "fn_tokens": 0}
    for r in evidence:
        p = corpus[r["id"]]
        truth = [(a, b) for a, b, y in p["labels"] if y == 1]
        for t in r["tokens"]:
            if not is_scored_token(t["text"]):
                continue
            gold = any(t["start"] < b and t["end"] > a for a, b in truth)
            if t["flagged"] and not gold:
                decomposition["fp_dirty" if truth else "fp_clean"] += 1
            if gold and not t["flagged"]:
                decomposition["fn_tokens"] += 1
    write_json(out / "dev_error_decomposition.json", decomposition)
    if cache.misses:
        raise ValueError("Unexpected cache miss")
    write_json(
        out / "manifest.json",
        {
            "run_id": os.getenv("GITHUB_RUN_ID"),
            "code_sha": os.getenv("GITHUB_SHA"),
            "source": provenance,
            "recipe": RECIPE,
            "train_pairs": 842,
            "dev_pairs": 181,
            "official_test_read": False,
            "pairs_file_read": False,
            "corpus_hash_scope": "actual train/dev; pairs/test hashes verified as manifest declarations only to avoid reading test",
            "cache_keys": len(cache),
            "cache_hits": cache.hits,
            "cache_misses": cache.misses,
            "fresh_hf_inference": False,
            "classifiers_fit": 51,
            "timings": timings,
            "total_seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "candidate_sha256": sha((out / "combined_bundle.json").read_bytes()),
            "sizes": {f.name: f.stat().st_size for f in out.iterdir() if f.is_file()},
        },
    )
    print(
        json.dumps({"selected_model": best, "selected_decoder": selected_decoder, "dev": metrics4}, ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    try:
        run(args.source, args.cache, args.output)
    except Exception:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
