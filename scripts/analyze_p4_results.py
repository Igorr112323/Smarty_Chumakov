#!/usr/bin/env python3
"""Post-hoc error accounting ONLY: no inference, fitting, search or config changes."""

import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spanverify-module"))
from spanverify.core import tokenize_with_offsets
from spanverify.dataset import read_pairs
from spanverify.engine import Verifier, WeightsBundle, _iou, _span_f1
from spanverify.features import is_scored_token
from spanverify.localization import localize_state

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "spanverify-module/reports/experiments/p4_context_head/run39164044941"


def loadgz(path):
    return [json.loads(x) for x in gzip.decompress(path.read_bytes()).decode().splitlines()]


def summarize(rows, corpus):
    predicted = []
    golds = []
    clean_fp = dirty_fp = no_overlap = low_iou = fragmented = 0
    for r in rows:
        pair = corpus[r["id"]]
        gold = [(a, b) for a, b, y in pair["labels"] if y == 1]
        pred = [(s["start"], s["end"]) for s in r["spans"]]
        m = _span_f1([pred], [gold])
        if gold:
            dirty_fp += m["fp"]
        else:
            clean_fp += m["fp"]
        for span in pred:
            iou = max((_iou(span, g) for g in gold), default=0)
            no_overlap += iou == 0
            low_iou += 0 < iou < 0.5
        fragmented += sum(sum(_iou(s, g) > 0 for s in pred) > 1 for g in gold)
        predicted.append(pred)
        golds.append(gold)
    return {
        **_span_f1(predicted, golds),
        "fp_clean_spans": clean_fp,
        "fp_dirty_spans": dirty_fp,
        "no_overlap_predictions": no_overlap,
        "overlap_below_05_predictions": low_iou,
        "gold_with_multiple_overlapping_predictions": fragmented,
    }


def main():
    corpus = {p["id"]: p for p in read_pairs(ROOT / "spanverify-module/data/corpus_a3/splits/dev.jsonl")}
    rows = loadgz(OUT / "dev_evidence.jsonl.gz")
    bundle = WeightsBundle.load(OUT / "combined_bundle.json")
    verifier = Verifier(weights=bundle, mode="hf", span_expansion="mask")
    stages = {name: [] for name in ("mask", "attribution", "coverage", "final")}
    miss = {
        "gold_without_eligible_scored_token": 0,
        "gold_with_eligible_but_no_retained_mask_overlap": 0,
        "gold_with_retained_overlap_but_no_final_strict_match": 0,
        "gold_strict_matched_by_final": 0,
    }
    parity = []
    token_errors = {"fn_edge_tokens": 0, "fn_interior_tokens": 0, "fp_clean_tokens": 0, "fp_dirty_tokens": 0}
    for r in rows:
        pair = corpus[r["id"]]
        text = pair["answer"]
        context = pair["context"]
        tokens = tokenize_with_offsets(text)
        risks = [t["raw_risk"] for t in r["tokens"]]
        scored = [is_scored_token(t.text) for t in tokens]
        attr = verifier._attribution_spans(text, context, tokens, risks, bundle.token_threshold)
        cov = verifier._coverage_spans(text, context, tokens, bundle.token_threshold)
        kept, masks, final = localize_state(text, tokens, risks, scored, bundle.localization_decoder, attr, cov)
        if [(s.start, s.end) for s in final] != [(s["start"], s["end"]) for s in r["spans"]] or kept != {
            i for i, t in enumerate(r["tokens"]) if t["flagged"]
        }:
            parity.append(r["id"])
        for name, spans in zip(stages, (masks, attr, cov, final), strict=True):
            stages[name].append({"id": r["id"], "spans": [{"start": s.start, "end": s.end} for s in spans]})
        gold = [(a, b) for a, b, y in pair["labels"] if y == 1]
        edge = set()
        for a, b in gold:
            indices = [i for i, t in enumerate(tokens) if scored[i] and t.start < b and t.end > a]
            if indices:
                edge.update((indices[0], indices[-1]))
            if any(_iou((s.start, s.end), (a, b)) >= 0.5 for s in final):
                miss["gold_strict_matched_by_final"] += 1
            elif not any(risks[i] >= bundle.localization_decoder["tau_low"] for i in indices):
                miss["gold_without_eligible_scored_token"] += 1
            elif not any(i in kept for i in indices):
                miss["gold_with_eligible_but_no_retained_mask_overlap"] += 1
            else:
                miss["gold_with_retained_overlap_but_no_final_strict_match"] += 1
        for i, t in enumerate(tokens):
            if not scored[i]:
                continue
            truth = any(t.start < b and t.end > a for a, b in gold)
            if truth and i not in kept:
                token_errors["fn_edge_tokens" if i in edge else "fn_interior_tokens"] += 1
            if not truth and i in kept:
                token_errors["fp_dirty_tokens" if gold else "fp_clean_tokens"] += 1
    # Geometric per-gold categories above are not greedy TP counts when duplicates exist.
    result = {
        "posthoc_only": True,
        "fit_or_search": False,
        "public_state_mismatches": parity,
        "p4_by_source": {k: summarize(v, corpus) for k, v in stages.items()},
        "p4_gold_geometry": miss,
        "p4_token_errors": token_errors,
    }
    p2 = loadgz(OUT / "source_dev_evidence.jsonl.gz")
    p3 = loadgz(ROOT / "spanverify-module/reports/experiments/p3_postprocess/run39160817890/dev_evidence.jsonl.gz")
    result["historical_final_errors"] = {"p2": summarize(p2, corpus), "p3": summarize(p3, corpus)}
    (OUT / "posthoc_plateau_diagnostics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
