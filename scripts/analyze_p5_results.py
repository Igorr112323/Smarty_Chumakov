#!/usr/bin/env python3
"""Supplemental immutable-evidence audit only; no fitting/search/HF inference."""

import gzip
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spanverify-module"))
from spanverify.core import tokenize_with_offsets
from spanverify.dataset import read_pairs
from spanverify.engine import Verifier, WeightsBundle, _iou, _span_f1
from spanverify.features import is_scored_token
from spanverify.localization import merge_localized_spans
from spanverify.span_filter import build_candidates, filter_candidates, probability

ROOT = Path(__file__).resolve().parents[1]
P4 = ROOT / "spanverify-module/reports/experiments/p4_context_head/run39164044941"
P5 = ROOT / "spanverify-module/reports/experiments/p5_span_filter/run39170981235"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def readgz(p):
    return [json.loads(x) for x in gzip.decompress(p.read_bytes()).splitlines()]


def main():
    inventory = json.loads((P4 / "published_files_sha256.json").read_text())["files"]
    checks = {n: sha(P4 / n) == r["sha256"] for n, r in inventory.items()}
    assert all(checks.values())
    old = json.loads((P4 / "combined_bundle.json").read_text())
    new = json.loads((P5 / "combined_bundle.json").read_text())
    assert old == {k: v for k, v in new.items() if k != "span_filter"}
    bundle = WeightsBundle.load(P5 / "combined_bundle.json")
    verifier = Verifier(weights=bundle, mode="hf", span_expansion="mask")
    corpus = {
        p["id"]: p
        for p in read_pairs(ROOT / "spanverify-module/data/corpus_a3/splits/dev.jsonl")
    }
    references = {r["id"]: r for r in readgz(P4 / "dev_evidence.jsonl.gz")}
    answers = {
        r["pair_id"]: r for r in json.loads((P5 / "dev_answers.json").read_text())
    }
    kept_masks = []
    rules_all = []
    final = []
    truths = []
    counts = dict(
        gold=0,
        any_candidate_overlap=0,
        strict_candidate_coverage=0,
        gold_without_candidate=0,
        fn_without_generator_and_final_overlap=0,
        gold_candidate_but_no_final_overlap=0,
        gold_final_overlap_below_05=0,
        fn_edge_tokens=0,
        fn_interior_tokens=0,
        fp_clean_tokens=0,
        fp_dirty_tokens=0,
    )
    details = []
    for r in readgz(P5 / "dev_evidence.jsonl.gz"):
        p = corpus[r["id"]]
        text = p["answer"]
        tokens = tokenize_with_offsets(text)
        truth = [(a, b) for a, b, y in p["labels"] if y == 1]
        scored = [is_scored_token(t.text) for t in tokens]
        risks = [t["raw_risk"] for t in references[r["id"]]["tokens"]]
        assert risks == [t["raw_risk"] for t in r["tokens"]]
        rules = [
            *verifier._attribution_spans(
                text, p["context"], tokens, risks, bundle.token_threshold
            ),
            *verifier._coverage_spans(
                text, p["context"], tokens, bundle.token_threshold
            ),
        ]
        config = bundle.span_filter
        candidates = build_candidates(
            text,
            tokens,
            risks,
            scored,
            config["generator"],
            answers[r["id"]]["calibrated_score"],
            rules,
        )
        flags, spans = filter_candidates(text, tokens, candidates, config, rules)
        assert [json.loads(json.dumps(asdict(s))) for s in spans] == r["spans"]
        assert flags == {i for i, t in enumerate(r["tokens"]) if t["flagged"]}
        candidate_coords = [(s.start, s.end) for s, _ in candidates]
        pred = [(s.start, s.end) for s in spans]
        mask = [
            s
            for s, row in candidates
            if probability(row, config["head"]) >= config["keep_threshold"]
        ]
        kept_masks.append(
            [(s.start, s.end) for s in merge_localized_spans(text, tokens, mask)]
        )
        rules_all.append(
            [(s.start, s.end) for s in merge_localized_spans(text, tokens, rules)]
        )
        final.append(pred)
        truths.append(truth)
        edge = set()
        for g in truth:
            counts["gold"] += 1
            any_candidate = any(_iou(s, g) > 0 for s in candidate_coords)
            counts["any_candidate_overlap"] += any_candidate
            counts["strict_candidate_coverage"] += any(
                _iou(s, g) >= 0.5 for s in candidate_coords
            )
            if not any_candidate:
                counts["gold_without_candidate"] += 1
                if not any(_iou(s, g) > 0 for s in pred):
                    counts["fn_without_generator_and_final_overlap"] += 1
            elif not any(_iou(s, g) > 0 for s in pred):
                counts["gold_candidate_but_no_final_overlap"] += 1
            if any(_iou(s, g) > 0 for s in pred) and not any(
                _iou(s, g) >= 0.5 for s in pred
            ):
                counts["gold_final_overlap_below_05"] += 1
            ids = [
                i
                for i, t in enumerate(tokens)
                if scored[i] and t.start < g[1] and t.end > g[0]
            ]
            if ids:
                edge.update((ids[0], ids[-1]))
        for i, t in enumerate(tokens):
            if not scored[i]:
                continue
            y = any(t.start < b and t.end > a for a, b in truth)
            if y and i not in flags:
                counts["fn_edge_tokens" if i in edge else "fn_interior_tokens"] += 1
            elif not y and i in flags:
                counts["fp_dirty_tokens" if truth else "fp_clean_tokens"] += 1
        details.append(
            dict(
                id=p["id"],
                candidate_count=len(candidates),
                kept_mask_count=len(mask),
                rule_count=len(rules),
            )
        )
    audit = dict(
        supplemental_posthoc_only=True,
        fit_or_search=False,
        source_p4_inventory_all_matched=checks,
        source_p4_dev_sha256=sha(P4 / "dev_evidence.jsonl.gz"),
        source_p4_oof_sha256=sha(P4 / "k3_w1_oof.jsonl.gz"),
        raw_bundle_fields_identical_except_span_filter=True,
        pure_replayed_pairs=len(details),
        pure_mismatches=[],
        counts=counts,
        strict_by_source=dict(
            kept_mask=_span_f1(kept_masks, truths),
            rules_only=_span_f1(rules_all, truths),
            final=_span_f1(final, truths),
        ),
        pairs=details,
    )
    (P5 / "posthoc_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in audit.items()
                if k not in ["pairs", "source_p4_inventory_all_matched"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
