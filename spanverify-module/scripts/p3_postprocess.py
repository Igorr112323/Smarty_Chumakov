#!/usr/bin/env python3
"""P3: train-OOF-only postprocessor selection, then frozen public HF dev replay.

No classifier training, no tree/HF calls during search. Source artifacts are
archived verbatim as deterministic gzip before validation/selection.
"""

from __future__ import annotations

import gzip
import hashlib
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

import spanverify.engine as engine
from spanverify.core import tokenize_with_offsets
from spanverify.dataset import read_pairs
from spanverify.engine import WeightsBundle, _containment_recall, _iou, _mean_width_ratio, _span_f1
from spanverify.features import feature_cache_key, is_scored_token, load_feature_cache, set_feature_cache
from spanverify.localization import QUANTILES, decoder_config, localize_state

SOURCE = Path("reports/experiments/p2_hf/run39155368906")
MODEL_HASH = "33112af1997e12afe9aee07a6f84eb29760c9165e1a928938b2979110d822e9e"
BUNDLE_HASH = "87298b01ec157b20c6a06cee9244d75ce1d7882a30f20abddbb6e9781bbf46ab"
OLD_TAU = 0.5653513416423852


def sha(data):
    return hashlib.sha256(data).hexdigest()


def obj_sha(value):
    return sha(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())


def exact_binary(tp, fp, fn, tn):
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else 0.0,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        fpr=fp / (fp + tn) if fp + tn else 0.0,
    )


def distribution(values):
    return {
        "count": len(values),
        "quantiles": (
            dict(
                zip(
                    ["min", "p25", "p50", "p75", "p90", "p95", "p99", "max"],
                    map(float, np.quantile(values, [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1], method="linear")),
                    strict=False,
                )
            )
            if values
            else {}
        ),
    }


def coords(spans):
    return [(s.start, s.end) for s in spans]


def archive_sources(source_dir, out):
    provenance = {
        "run_id": 39155368906,
        "artifact_id": 12146627081,
        "code_sha": "7e0ed4454180c9c1ff0bf39ffbec5b993e41ade5",
        "files": {},
    }
    for name in ("oof_states.jsonl", "dev_evidence.jsonl"):
        matches = list(source_dir.rglob(name))
        if len(matches) != 1:
            raise ValueError(f"Expected one {name}, got {len(matches)}")
        raw = matches[0].read_bytes()
        packed = gzip.compress(raw, compresslevel=9, mtime=0)
        (out / ("source_" + name + ".gz")).write_bytes(packed)
        provenance["files"][name] = {
            "sha256": sha(raw),
            "bytes": len(raw),
            "gzip_sha256": sha(packed),
            "gzip_bytes": len(packed),
        }
    write_json(out / "source_provenance.json", provenance)
    return provenance


def validate_states(states, train, receipts):
    if len(states) != 842 or len({s["id"] for s in states}) != 842:
        raise ValueError("Expected 842 unique OOF pairs")
    corpus = {p["id"]: p for p in train}
    if set(corpus) != {s["id"] for s in states}:
        raise ValueError("OOF/corpus ID mismatch")
    docs = {p["meta"]["doc_id"] for p in train}
    held_counts = {d: 0 for d in docs}
    by_fold = {r["fold"]: r for r in receipts}
    if set(by_fold) != set(range(5)):
        raise ValueError("Expected five source folds")
    for r in receipts:
        fit, held = set(r["fit_documents"]), set(r["held_documents"])
        if fit & held or fit | held != docs or r["shared_documents"] != 0:
            raise ValueError("Source fold leakage")
        if obj_sha(sorted(fit)) != r["fit_sha256"] or obj_sha(sorted(held)) != r["held_sha256"]:
            raise ValueError("Source receipt hash mismatch")
        for doc in held:
            held_counts[doc] += 1
    if set(held_counts.values()) != {1}:
        raise ValueError("Document held coverage not exactly once")
    for s in states:
        p = corpus[s["id"]]
        tokens = tokenize_with_offsets(p["answer"])
        truth = [[int(a), int(b)] for a, b, y in p["labels"] if int(y) == 1]
        if s["answer"] != p["answer"] or s["context"] != p["context"] or s["doc"] != p["meta"]["doc_id"]:
            raise ValueError("OOF identity/content drift")
        if s["doc"] not in by_fold[s["fold"]]["held_documents"]:
            raise ValueError("Wrong OOF fold membership")
        if (
            s["offsets"] != [[t.start, t.end] for t in tokens]
            or s["scored"] != [is_scored_token(t.text) for t in tokens]
            or s["truth"] != truth
        ):
            raise ValueError("Offsets/scored/truth mismatch")
        gold = [int(any(t.start < b and t.end > a for a, b in truth)) for t in tokens]
        if s["gold"] != gold or len(s["smoothed"]) != len(tokens):
            raise ValueError("Gold/risk alignment mismatch")
        if not all(np.isfinite(r) and 0 <= r <= 1 for r in s["smoothed"]):
            raise ValueError("Invalid source risks")
        s["_tokens"] = tokens
    return {
        "pairs": len(states),
        "documents": len(docs),
        "folds": 5,
        "scored_tokens": sum(sum(s["scored"]) for s in states),
        "missing_states": 0,
        "shared_documents": 0,
        "coverage": "each pair/document held exactly once; no placeholders inserted",
    }


def source_diagnostics(verifier, exact_verifier, states):
    outputs = {k: [] for k in ("mask", "attribution", "coverage", "final", "exact_attribution", "final_exact_anchor")}
    truth = [s["truth"] for s in states]
    evidence = []
    merge_lost = 0
    risks_in = []
    risks_out = []
    gaps = []
    component_lengths = []
    for s in states:
        t = s["_tokens"]
        r = s["smoothed"]
        answer = s["answer"]
        context = s["context"]
        mask = verifier._build_spans(answer, t, r, OLD_TAU)
        attr = verifier._attribution_spans(answer, context, t, r, OLD_TAU)
        cov = verifier._coverage_spans(answer, context, t, OLD_TAU)
        exact = exact_verifier._attribution_spans(answer, context, t, r, OLD_TAU)
        final = engine._merge_spans([*mask, *attr, *cov])
        final_exact = engine._merge_spans([*mask, *exact, *cov])
        for key, spans in zip(outputs, (mask, attr, cov, final, exact, final_exact), strict=False):
            outputs[key].append(coords(spans))
        lost = []
        for j, g in enumerate(s["truth"]):
            if any(_iou(p, g) >= 0.5 for p in coords(mask)) and not any(_iou(p, g) >= 0.5 for p in coords(final)):
                lost.append(j)
        merge_lost += len(lost)
        eligible = [i for i, (v, m) in enumerate(zip(r, s["scored"], strict=False)) if m and v >= OLD_TAU]
        gaps.extend(b - a - 1 for a, b in zip(eligible, eligible[1:], strict=False))
        component_lengths.extend(p.n_tokens for p in mask)
        for risk, g, m in zip(r, s["gold"], s["scored"], strict=False):
            if m:
                (risks_in if g else risks_out).append(risk)
        evidence.append(
            {
                "id": s["id"],
                "doc": s["doc"],
                "fold": s["fold"],
                "truth": s["truth"],
                "mask": coords(mask),
                "attribution": coords(attr),
                "coverage": coords(cov),
                "final": coords(final),
                "exact_attribution": coords(exact),
                "final_exact_anchor": coords(final_exact),
                "mask_gold_matches_lost_after_merge": lost,
            }
        )
        # Cached pure rule outputs are also the actual selection inputs.
        s["_attr"] = exact
        s["_coverage"] = cov
    result = {}
    for source, predicted in outputs.items():
        clean_fp = dirty_fp = no_overlap = low_iou = fragmented_gold = 0
        widths = []
        numbers = []
        for spans, gold in zip(predicted, truth, strict=False):
            numbers.append(len(spans))
            widths.extend(b - a for a, b in spans)
            for span in spans:
                best = max((_iou(span, g) for g in gold), default=0)
                no_overlap += best == 0
                low_iou += 0 < best < 0.5
            fragmented_gold += sum(sum(_iou(span, g) > 0 for span in spans) > 1 for g in gold)
            per = _span_f1([spans], [gold])
            clean_fp += per["fp"] if not gold else 0
            dirty_fp += per["fp"] if gold else 0
        result[source] = {
            **_span_f1(predicted, truth),
            "fp_clean": clean_fp,
            "fp_dirty": dirty_fp,
            "no_overlap_spans": no_overlap,
            "overlap_below_iou_05_spans": low_iou,
            "gold_with_multiple_overlapping_predictions": fragmented_gold,
            "char_widths": distribution(widths),
            "components_per_pair": distribution(numbers),
            "containment": _containment_recall(predicted, truth),
            "mean_width_ratio_containment_only": _mean_width_ratio(predicted, truth),
        }
    result["merge_loss"] = {"gold_mask_matches_lost_after_final_merge": merge_lost}
    result["mask_component_tokens"] = distribution(component_lengths)
    result["eligible_token_gaps"] = distribution(gaps)
    result["scored_risk_inside_gold"] = distribution(risks_in)
    result["scored_risk_outside_gold"] = distribution(risks_out)
    return result, evidence


def score_config(states, config):
    tp = fp = fn = tn = 0
    predicted = []
    truth = []
    for s in states:
        kept, _, spans = localize_state(
            s["answer"], s["_tokens"], s["smoothed"], s["scored"], config, s["_attr"], s["_coverage"]
        )
        positives = sum(g for g, m in zip(s["gold"], s["scored"], strict=False) if m)
        pair_tp = sum(s["gold"][i] for i in kept)
        pair_fp = len(kept) - pair_tp
        tp += pair_tp
        fp += pair_fp
        fn += positives - pair_tp
        tn += sum(s["scored"]) - positives - pair_fp
        predicted.append(coords(spans))
        truth.append(s["truth"])
    token = exact_binary(tp, fp, fn, tn)
    span = _span_f1(predicted, truth)
    return {"config": config, "tokens": token, "spans": span, "objective": min(token["f1"], span["f1"])}


def search(states, out):
    pool = [r for s in states for r, m in zip(s["smoothed"], s["scored"], strict=False) if m]
    qvals = list(map(float, np.quantile(pool, QUANTILES, method="linear")))
    points = sorted(set([0.0, float(np.nextafter(max(pool), np.inf)), OLD_TAU, *qvals]))
    write_json(
        out / "numeric_candidates.json",
        {
            "quantiles": QUANTILES,
            "method": "numpy.quantile(method=linear)",
            "quantile_values": qvals,
            "cutoffs": points,
            "bridge": [0, 1],
            "min_seed_tokens": [1, 2],
            "source": "pooled train OOF scored smoothed risks only",
        },
    )
    rows = []
    best = None
    with gzip.open(out / "grid_scores.jsonl.gz", "wt", encoding="utf-8") as log:
        for low in points:
            for high in points:
                if high < low:
                    continue
                for bridge in (0, 1):
                    for seeds in (1, 2):
                        row = score_config(states, decoder_config(low, high, bridge, seeds))
                        log.write(json.dumps(row) + "\n")
                        rows.append(row)
                        if row["tokens"]["fpr"] <= 0.30 and (
                            best is None
                            or (row["objective"], -row["tokens"]["fpr"]) > (best["objective"], -best["tokens"]["fpr"])
                        ):
                            best = row
            print("Search low", low, "completed configs", len(rows), flush=True)
    write_json(
        out / "selected_decoder.json",
        {
            "selected": best,
            "points": len(rows),
            "criterion": "max min(token F1, strict span F1), token FPR<=0.30; tie smaller FPR, then fixed iteration order",
            "source": "train-OOF only; selection metrics are NOT independent acceptance",
        },
    )
    return best


def run(source_dir, cache_dir, out):
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    provenance = archive_sources(source_dir, out)
    for name, expected in [("p2_token_head.json", MODEL_HASH), ("combined_bundle.json", BUNDLE_HASH)]:
        if sha((SOURCE / name).read_bytes()) != expected:
            raise ValueError(f"Frozen {name} hash mismatch")
    states = read_jsonl(next(source_dir.rglob("oof_states.jsonl")))
    train = list(read_pairs("data/corpus_a3/splits/train.jsonl"))
    coverage = validate_states(states, train, read_json(SOURCE / "fold_receipts.json"))
    write_json(out / "oof_validation.json", coverage)
    baseline = WeightsBundle.load(SOURCE / "combined_bundle.json")
    original = baseline.to_dict()
    runtime = read_json(SOURCE / "runtime_config.json")
    if (
        runtime["model_name"] != "ai-forever/rugpt3small_based_on_gpt2"
        or runtime["hf_max_tokens"] != 512
        or runtime["chunking"] is not True
        or runtime["mode"] != "hf"
    ):
        raise ValueError("Frozen runtime mismatch")
    engine.cover_facts = lru_cache(maxsize=2048)(engine.cover_facts)
    engine.number_attribution = lru_cache(maxsize=2048)(engine.number_attribution)
    # No HF/tree calls here: only text-rule methods over existing states.
    v = make_verifier(baseline, runtime, {})
    anchored = make_verifier(replace(baseline, localization_decoder=decoder_config(OLD_TAU, OLD_TAU)), runtime, {})
    diagnostic, diagnostic_rows = source_diagnostics(v, anchored, states)
    write_json(out / "oof_diagnostics.json", diagnostic)
    write_jsonl(out / "oof_diagnostic_evidence.jsonl", diagnostic_rows)
    best = search(states, out)
    config = best["config"]
    candidate = replace(baseline, localization_decoder=config)
    if replace(candidate, localization_decoder=None).to_dict() != original:
        raise ValueError("Frozen bundle fields changed")
    (out / "p2_token_head.json").write_bytes((SOURCE / "p2_token_head.json").read_bytes())
    if baseline.to_dict() != original:
        raise ValueError("Baseline mutated")
    candidate.save(out / "combined_bundle.json")
    p3runtime = {**runtime, "localization_decoder": config}
    write_json(out / "runtime_config.json", p3runtime)
    # Only now, after writing the fixed winner, read dev and its reference evidence.
    dev = list(read_pairs("data/corpus_a3/splits/dev.jsonl"))
    reference_evidence = read_jsonl(next(source_dir.rglob("dev_evidence.jsonl")))
    if len(dev) != 181 or len(reference_evidence) != 181 or len({p["id"] for p in dev}) != 181:
        raise ValueError("Expected full dev")
    refs = {r["id"]: r for r in reference_evidence}
    if set(refs) != {p["id"] for p in dev}:
        raise ValueError("Dev reference ID mismatch")
    shards = list(cache_dir.rglob("features_*.jsonl"))
    if len(shards) != 8 or len({sha(p.read_bytes()) for p in shards}) != 8:
        raise ValueError("Need eight distinct A3 shards")
    cache = load_feature_cache([str(p) for p in shards], counting=True, strict=True)
    set_feature_cache(cache)
    missing = [
        p["id"]
        for p in dev
        if feature_cache_key(p["answer"], p["context"], "hf", runtime["model_name"], chunking=True, max_length=512)
        not in cache
    ]
    write_json(out / "cache_coverage.json", {"keys": len(cache), "missing": missing, "dev_pairs": len(dev)})
    if missing:
        raise ValueError("Missing cache; fresh inference forbidden")
    bv = make_verifier(baseline, runtime, cache)
    pv = make_verifier(candidate, p3runtime, cache)
    rv = make_verifier(WeightsBundle.load(out / "combined_bundle.json"), p3runtime, cache)
    before = collect_public_rows(pv, dev)
    after = collect_public_rows(rv, dev)
    baseline_answers = collect_public_rows(bv, dev)
    p1ref = read_jsonl(Path("reports/experiments/p1_public_replay/run39144872719/public_replay.jsonl"))
    p2ref = read_jsonl(SOURCE / "dev_answers.jsonl")
    parity = {
        "p1_reference": compare_records(before, p1ref),
        "p2_reference": compare_records(before, p2ref),
        "baseline_replay": compare_records(baseline_answers, p2ref),
        "reload": compare_records(after, before),
    }
    write_json(out / "answer_parity.json", parity)
    write_jsonl(out / "dev_answers.jsonl", before)
    risk_errors = []
    roundtrip_errors = []
    pure_errors = []
    rows = []
    max_delta = 0.0
    token_count = 0
    for pair in dev:
        result = pv.verify(pair["answer"], pair["context"], with_tokens=True)
        reread = rv.verify(pair["answer"], pair["context"], with_tokens=True)
        ref = refs[pair["id"]]
        token_count += len(result.tokens)
        if len(result.tokens) != len(ref["tokens"]):
            raise ValueError("Token reference length mismatch")
        for i, (t, r) in enumerate(zip(result.tokens, ref["tokens"], strict=False)):
            d = abs(t["raw_risk"] - r["raw_risk"])
            max_delta = max(max_delta, d)
            if d > 1e-12 or t["start"] != r["start"] or t["end"] != r["end"]:
                risk_errors.append({"id": pair["id"], "token": i, "delta": d})
        if result.tokens != reread.tokens or result.spans != reread.spans or result.verdict != reread.verdict:
            roundtrip_errors.append(pair["id"])
        tokens = tokenize_with_offsets(pair["answer"])
        risks = [t["raw_risk"] for t in result.tokens]
        scored = [is_scored_token(t.text) for t in tokens]
        kept, _, spans = localize_state(
            pair["answer"],
            tokens,
            risks,
            scored,
            config,
            pv._attribution_spans(pair["answer"], pair["context"], tokens, risks, OLD_TAU),
            pv._coverage_spans(pair["answer"], pair["context"], tokens, OLD_TAU),
        )
        if spans != result.spans or kept != {i for i, t in enumerate(result.tokens) if t["flagged"]}:
            pure_errors.append(pair["id"])
        for span in result.spans:
            if span.text != pair["answer"][span.start : span.end]:
                raise ValueError("Span text/offset mismatch")
        rows.append(
            {
                "id": pair["id"],
                "gold": pair["labels"],
                "tokens": result.tokens,
                "spans": [asdict(s) for s in result.spans],
                "score": result.score,
                "threshold": result.threshold,
                "verdict": result.verdict,
                "hard_share": result.ai_share_hard,
            }
        )
    with gzip.open(out / "dev_evidence.jsonl.gz", "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    risk_parity = {
        "pairs": 181,
        "tokens": token_count,
        "max_abs_delta": max_delta,
        "risk_mismatches": risk_errors,
        "reload_mismatches": roundtrip_errors,
        "pure_public_mismatches": pure_errors,
    }
    write_json(out / "risk_reload_parity.json", risk_parity)
    if any(not p["matched"] for p in parity.values()) or risk_errors or roundtrip_errors or pure_errors:
        raise ValueError("P1/P2 risk or runtime drift; P3 blocked")
    counts = metrics(before)
    if [counts[k] for k in ("tp", "fp", "fn", "tn")] != [49, 14, 47, 71]:
        raise ValueError("Answer counts changed")
    allmetrics = {"p2": bv.evaluate(dev), "p3": pv.evaluate(dev)}
    write_json(out / "all_family_metrics.json", allmetrics)
    if cache.misses:
        raise ValueError("Unexpected cache miss")
    status = (
        "ДОСТИГНУТО"
        if all(allmetrics["p3"][f]["f1"] >= 0.5 for f in ("tokens", "spans"))
        and allmetrics["p3"]["tokens"]["fpr"] <= 0.4
        else "НЕ ДОСТИГНУТО"
    )
    write_json(
        out / "manifest.json",
        {
            "run_id": os.getenv("GITHUB_RUN_ID"),
            "code_sha": os.getenv("GITHUB_SHA"),
            "source": provenance,
            "frozen_models": {"p2_token_head.json": MODEL_HASH, "combined_bundle.json": BUNDLE_HASH},
            "candidate_bundle_sha256": sha((out / "combined_bundle.json").read_bytes()),
            "train_pairs": 842,
            "dev_pairs": 181,
            "classifier_fits": 0,
            "fresh_hf_inference": False,
            "official_test": False,
            "cache_source": 39060670054,
            "cache_keys": len(cache),
            "cache_hits": cache.hits,
            "cache_misses": cache.misses,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "seconds": time.perf_counter() - started,
            "quality_status": status,
            "sizes": {p.name: p.stat().st_size for p in out.iterdir() if p.is_file()},
        },
    )
    print(
        json.dumps(
            {"quality_status": status, "selected": best, "metrics": allmetrics, "parity": risk_parity},
            ensure_ascii=False,
        ),
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
