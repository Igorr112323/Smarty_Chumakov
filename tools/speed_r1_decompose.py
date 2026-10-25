#!/usr/bin/env python3
"""
Detailed decomposition per §2.1: time per stage, plus fixed/marginal
Wraps Verifier.verify with timing hooks without modifying science code.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spanverify-module"))
# guard against test177
import os

RESERVED = {
    str((ROOT / f"spanverify-module/data/corpus_a3/{p}").resolve())
    for p in ("pairs.jsonl", "splits/test.jsonl")
}


def guard(event, args):
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        try:
            if os.path.realpath(os.fsdecode(args[0])) in RESERVED:
                raise PermissionError("reserved test177")
        except Exception:
            pass


import sys

sys.addaudithook(guard)

from spanverify.config import Config
from spanverify.core import tokenize_with_offsets
from spanverify.dataset import read_pairs
from spanverify.engine import Verifier, WeightsBundle

P5 = ROOT / "spanverify-module/reports/experiments/p5_span_filter/run39170981235"
BUNDLE = P5 / "combined_bundle.json"
SHA = "e0b13c693a058dfd66ed24a52fc14e9c70964e95885358e7b99cc1f328ad6892"
MODEL = "ai-forever/rugpt3small_based_on_gpt2"


def timed_stages(mode="demo", size=1000, scenario="answer-text", warm=False):
    """Return per-stage timings for one verification"""
    t_import_start = time.perf_counter()
    # already imported, but measure import time via subprocess separately
    t_import_end = time.perf_counter()
    import_s = t_import_end - t_import_start

    # Load bundle
    t0 = time.perf_counter()
    assert hashlib.sha256(BUNDLE.read_bytes()).hexdigest() == SHA
    config = Config.load().with_overrides(
        backend=mode, hf_model=MODEL, hf_max_tokens=512
    )
    bundle = WeightsBundle.load(BUNDLE)
    t1 = time.perf_counter()
    weights_load_s = t1 - t0

    # Verifier init
    t0 = time.perf_counter()
    v = Verifier(config=config, weights=bundle, mode=mode, model_name=MODEL)
    t1 = time.perf_counter()
    verifier_init_s = t1 - t0

    # Warmup if needed
    warm_s = None
    if warm:
        # use 1000 doc-context as warmup (P6 protocol)
        p = next(
            iter(
                read_pairs(ROOT / "spanverify-module/data/corpus_a3/splits/train.jsonl")
            )
        )
        base = p["context"] + "\n"
        txt = (base * ((1000 // len(base)) + 1))[:1000]
        aw, cw = (
            (p["answer"], txt)
            if scenario == "document-context"
            else (txt, p["context"])
        )
        # document-context scenario warmup uses opposite? Keep consistent:
        # P6 warmup is always document-context 1000
        aw2, cw2 = (p["answer"], txt) if False else (txt, p["context"])

        # Actually P6 warmup is document(1000,"document-context") -> (answer, context=1000 txt)
        # Let's replicate: (p["answer"], 1000ctx) if scenario generic? Simpler use generic:
        # local document function to avoid import loop
        def docfn(n, scenario):
            from spanverify.dataset import read_pairs as rp

            p = next(
                iter(rp(ROOT / "spanverify-module/data/corpus_a3/splits/train.jsonl"))
            )
            base = p["context"] + "\n"
            txt = (base * ((n // len(base)) + 1))[:n]
            assert len(txt) == n
            return (
                (p["answer"], txt)
                if scenario == "document-context"
                else (txt, p["context"])
            )

        aw_w, cw_w = docfn(1000, "document-context")
        t0 = time.perf_counter()
        try:
            v.verify(aw_w, cw_w)
        except Exception as e:
            return {"error_warm": str(e)}
        t1 = time.perf_counter()
        warm_s = t1 - t0

    # Prepare fixture
    def document_local(n, scenario):
        from spanverify.dataset import read_pairs as rp

        p = next(iter(rp(ROOT / "spanverify-module/data/corpus_a3/splits/train.jsonl")))
        base = p["context"] + "\n"
        txt = (base * ((n // len(base)) + 1))[:n]
        assert len(txt) == n
        return (
            (p["answer"], txt)
            if scenario == "document-context"
            else (txt, p["context"])
        )

    a, c = document_local(size, scenario)

    # Tokenization
    t0 = time.perf_counter()
    toks = tokenize_with_offsets(a)
    t1 = time.perf_counter()
    tokenization_s = t1 - t0

    # Monkey-patch internal functions to time stages
    stages = {}
    orig_hf_features = None
    orig_full = None
    orig_token_head = None
    orig_context_head = None
    orig_filter = None
    orig_build_candidates = None
    orig_isotonic = None

    # We will wrap Verifier._features, _localize, etc., by measuring inside verify
    # Simpler: time the whole verify and also extract breakdown from profile
    # For now, we do coarse breakdown: _features vs _localize vs _shares vs aggregation vs serialization

    # Patch to capture
    import spanverify.engine as eng_mod
    import spanverify.features as feat_mod

    # Keep originals
    orig_features = v._features
    orig_localize = v._localize
    orig_shares = v._shares
    orig_verdict = eng_mod._verdict
    orig_combine = feat_mod.combine

    timings = {}

    def wrapping_features(*args, **kwargs):
        t0 = time.perf_counter()
        res = orig_features(*args, **kwargs)
        t1 = time.perf_counter()
        timings["features_s"] = t1 - t0
        # If HF, features includes model forward; we can further split via inside
        return res

    def wrapping_localize(*args, **kwargs):
        t0 = time.perf_counter()
        res = orig_localize(*args, **kwargs)
        t1 = time.perf_counter()
        timings["localize_s"] = t1 - t0
        return res

    def wrapping_shares(*args, **kwargs):
        t0 = time.perf_counter()
        res = orig_shares(*args, **kwargs)
        t1 = time.perf_counter()
        timings["shares_s"] = t1 - t0
        return res

    v._features = wrapping_features
    v._localize = wrapping_localize
    v._shares = wrapping_shares

    # Now verify
    t0 = time.perf_counter()
    try:
        result = v.verify(a, c)
    except Exception as e:
        return {"status": "failed", "error": str(e), "weights_load_s": weights_load_s}
    t1 = time.perf_counter()
    verify_total_s = t1 - t0

    # Restore
    v._features = orig_features
    v._localize = orig_localize
    v._shares = orig_shares

    # Serialization
    t0 = time.perf_counter()
    payload = result.to_dict(with_tokens=False)
    t1 = time.perf_counter()
    serialization_s = t1 - t0

    # Torch init stage: if HF, we can measure model load time via _load_hf_runtime
    # It is lazy inside first _features; we already timed features includes it
    # To isolate, we can call _load_hf_runtime separately before
    torch_init_s = 0
    if mode == "hf":
        # Try to measure _load_hf_runtime if not yet cached
        try:
            from spanverify.features import _load_hf_runtime

            t0 = time.perf_counter()
            _load_hf_runtime(MODEL, None)
            t1 = time.perf_counter()
            torch_init_s = t1 - t0
        except Exception:
            torch_init_s = timings.get("features_s", 0)  # fallback

    # Aggregate
    # Estimate 5 features calc inside features_s, heads inside localize? Actually heads after features
    # For detailed, we split features_s into forward vs calc
    # But we provide at least these stages per spec:
    stages = {
        "process_startup/import_s": import_s,
        "weights_load_s": weights_load_s,
        "verifier_init_s": verifier_init_s,
        "torch_init_s": torch_init_s,
        "tokenization_s": tokenization_s,
        "features_forward_and_5feat_s": timings.get("features_s", 0),
        "candidate_build_span_filter_s": timings.get("localize_s", 0),
        "shares_aggregation_s": timings.get("shares_s", 0),
        "verify_total_s": verify_total_s,
        "serialization_s": serialization_s,
        "warmup_s": warm_s,
        "total_measured_s": weights_load_s
        + verifier_init_s
        + (warm_s or 0)
        + verify_total_s
        + serialization_s,
    }
    # Also compute overhead
    stages["other_overhead_s"] = max(
        0,
        verify_total_s
        - sum(
            [
                timings.get("features_s", 0),
                timings.get("localize_s", 0),
                timings.get("shares_s", 0),
                tokenization_s,
            ]
        ),
    )

    return {
        "mode": mode,
        "scenario": scenario,
        "chars": size,
        "warm": warm,
        "stages": stages,
        "features_meta": result.stats.get("features", {}),
        "token_count": result.stats.get("token_count"),
        "spans": len(result.spans),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="demo")
    ap.add_argument("--size", type=int, default=1000)
    ap.add_argument("--scenario", default="answer-text")
    ap.add_argument("--warm", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    res = timed_stages(args.mode, args.size, args.scenario, args.warm)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))
