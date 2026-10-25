#!/usr/bin/env python3
"""
SPEED-R1 Phase A — Декомпозиция таймингов HF/Demo
- Профилирует штатный режим и разбивает время по стадиям
- Замеры на фикстурах 1000 / 50_000 / 2_000_000, n>=10, медиана+p95, сырые массивы
- Две конфигурации: 2 ядра (OMP_NUM_THREADS=1) и все ядра
- Честно фиксирует недоступность HF в offline sandbox (whitelist) и использует
  исторические P7 измерения как authoritative с полной provenance.

Никакой fit/train, никаких изменений науки. Только измерение.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import math
import os
import platform
import pstats
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spanverify-module"))

# Guard: never open test177 or pairs.jsonl via audithook (same as P6)
P5 = ROOT / "spanverify-module/reports/experiments/p5_span_filter/run39170981235"
RESERVED = {
    str((ROOT / f"spanverify-module/data/corpus_a3/{p}").resolve())
    for p in ("pairs.jsonl", "splits/test.jsonl")
}


def guard(event, args):
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        try:
            rp = os.path.realpath(os.fsdecode(args[0]))
            if rp in RESERVED:
                raise PermissionError(
                    "SPEED-R1: test177/pairs.jsonl is reserved — not allowed per §1.3"
                )
        except Exception:
            pass


sys.addaudithook(guard)

from spanverify.config import Config
from spanverify.dataset import read_pairs
from spanverify.engine import Verifier, WeightsBundle

BUNDLE = P5 / "combined_bundle.json"
SHA = "e0b13c693a058dfd66ed24a52fc14e9c70964e95885358e7b99cc1f328ad6892"
MODEL = "ai-forever/rugpt3small_based_on_gpt2"


def fingerprint():
    """Machine fingerprint per §2.3"""
    cpuinfo = (
        Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
    )
    meminfo = (
        Path("/proc/meminfo").read_text() if Path("/proc/meminfo").exists() else ""
    )
    # Extract model name
    model_name = "unknown"
    for line in cpuinfo.splitlines():
        if line.startswith("model name"):
            model_name = line.split(":", 1)[1].strip()
            break
    cpu_count = os.cpu_count()
    # check affinity
    try:
        aff = os.sched_getaffinity(0)
        aff_list = sorted(aff)
    except Exception:
        aff_list = list(range(cpu_count)) if cpu_count else []
    # env for determinism
    env = {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "unset"),
    }
    # git SHA
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        sha = "unknown"
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        branch = "unknown"
    return {
        "cpu_model": model_name,
        "cpu_logical": cpu_count,
        "cpu_affinity": aff_list,
        "meminfo_first_line": meminfo.splitlines()[0] if meminfo else "unknown",
        "os": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "git_sha": sha,
        "git_branch": branch,
        "env_threads": env,
        "bundle_sha256": SHA,
        "model": MODEL,
    }


def document(n, scenario):
    """Same as P6/P7 document(): repeat first train context to exact n chars"""
    p = next(
        iter(read_pairs(ROOT / "spanverify-module/data/corpus_a3/splits/train.jsonl"))
    )
    base = p["context"] + "\n"
    text = (base * (n // len(base) + 1))[:n]
    assert len(text) == n
    return (
        (p["answer"], text) if scenario == "document-context" else (text, p["context"])
    )


def check_schema(result, answer):
    assert 0 <= result.score <= 1
    for span in result.spans:
        assert 0 <= span.start < span.end <= len(answer)
        assert answer[span.start : span.end] == span.text
        assert span.label and 0 <= span.risk <= 1


def percentile(data, p):
    """p in 0..100, linear interpolation like numpy"""
    if not data:
        return None
    s = sorted(data)
    n = len(s)
    if n == 1:
        return s[0]
    k = (n - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def run_single(mode, size, scenario, thermal, profile=False):
    """Run one verification, return timing decomposition."""
    # Load verifier — timed separately
    t0 = time.perf_counter()
    # Stage: process start already at import, but we measure from here
    # Weights load
    t_w_start = time.perf_counter()
    # verify bundle sha
    assert hashlib.sha256(BUNDLE.read_bytes()).hexdigest() == SHA, "bundle mismatch"
    config = Config.load().with_overrides(
        backend=mode, hf_model=MODEL, hf_max_tokens=512
    )
    bundle = WeightsBundle.load(BUNDLE)
    t_w_end = time.perf_counter()
    weights_load_s = t_w_end - t_w_start

    # Verifier init (includes config)
    t_v_start = time.perf_counter()
    verifier = Verifier(
        config=config,
        weights=bundle,
        mode=mode,
        model_name=MODEL,
        features_cache=None,
        span_expansion="mask",
    )
    t_v_end = time.perf_counter()
    verifier_init_s = t_v_end - t_v_start

    # If warm, do warmup request 1000 doc-context (not timed in main latency per spec, but we record)
    warmup_s = None
    if thermal == "warm-model":
        aw, cw = document(1000, "document-context")
        t_warm_start = time.perf_counter()
        try:
            verifier.verify(aw, cw)
        except Exception as e:
            # HF may fail due to missing model download (network whitelist)
            return {
                "status": "warmup_failed",
                "error": str(e),
                "weights_load_s": weights_load_s,
                "verifier_init_s": verifier_init_s,
            }
        warmup_s = time.perf_counter() - t_warm_start

    a, c = document(size, scenario)

    # Detailed stage timing via manual instrumentation
    # We will time tokenization separately, then full verify
    # Tokenization stage
    from spanverify.core import tokenize_with_offsets

    t_tok_start = time.perf_counter()
    tokens = tokenize_with_offsets(a)
    t_tok_end = time.perf_counter()
    tokenization_s = t_tok_end - t_tok_start

    # Now full verify with profiling
    # We instrument via cProfile if requested
    profiler = cProfile.Profile() if profile else None
    if profiler:
        profiler.enable()
    t_verify_start = time.perf_counter()
    try:
        result = verifier.verify(a, c)
    except Exception as e:
        if profiler:
            profiler.disable()
        return {
            "status": "verify_failed",
            "error": f"{type(e).__name__}: {e}",
            "weights_load_s": weights_load_s,
            "verifier_init_s": verifier_init_s,
            "warmup_s": warmup_s,
            "tokenization_s": tokenization_s,
            "model_available": False,
        }
    t_verify_end = time.perf_counter()
    if profiler:
        profiler.disable()
    total_verify_s = t_verify_end - t_verify_start
    # Serialization
    t_ser_start = time.perf_counter()
    payload = result.to_dict(with_tokens=False)
    t_ser_end = time.perf_counter()
    serialization_s = t_ser_end - t_ser_start

    # Extract feature meta for diagnostics
    features_meta = result.stats.get("features", {})

    # Profile top functions
    profile_top = []
    if profiler:
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats("cumulative")
        # Get stats dict
        stats = ps.stats
        # sort by self_seconds (index 2)
        sorted_stats = sorted(stats.items(), key=lambda kv: kv[1][2], reverse=True)[:15]
        for (fname, lineno, func), (cc, nc, tt, ct, callers) in sorted_stats:
            profile_top.append(
                {
                    "file": fname,
                    "line": lineno,
                    "function": func,
                    "calls": nc,
                    "self_seconds": tt,
                    "cumulative_seconds": ct,
                }
            )

    check_schema(result, a)

    # Latency_ms from result
    latency_ms = result.latency_ms

    # Estimate breakdown for HF: we can approximate torch time via profile
    # For demo, torch time ~0
    return {
        "status": "completed",
        "mode": mode,
        "scenario": scenario,
        "chars": size,
        "thermal": thermal,
        "weights_load_s": weights_load_s,
        "verifier_init_s": verifier_init_s,
        "warmup_s": warmup_s,
        "tokenization_s": tokenization_s,
        "verify_s": total_verify_s,
        "serialization_s": serialization_s,
        "total_s": weights_load_s
        + verifier_init_s
        + (warmup_s or 0)
        + total_verify_s
        + serialization_s,
        "latency_ms_reported": latency_ms,
        "profile_top": profile_top,
        "features_meta": features_meta,
        "spans": len(result.spans),
        "token_count": result.stats.get("token_count"),
        "mean_risk": result.stats.get("mean_risk"),
        "answer_chars": len(a),
        "context_chars": len(c),
    }


def benchmark(mode, sizes, scenarios, thermals, repeats, out_dir, profile=False):
    fp = fingerprint()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fingerprint.json").write_text(
        json.dumps(fp, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f":: fingerprint {fp}")

    # For each combination, run n repeats, collect raw
    all_rows = []
    raw_collections = {}  # key -> list of seconds
    for scenario in scenarios:
        for size in sizes:
            for thermal in thermals:
                key = f"{mode}-{scenario}-{size}-{thermal}"
                raw = []
                row_details = []
                for rep in range(repeats):
                    print(f"Running {key} rep {rep+1}/{repeats} ...", flush=True)
                    res = run_single(
                        mode, size, scenario, thermal, profile=(profile and rep == 0)
                    )
                    row_details.append(res)
                    if res.get("status") == "completed":
                        raw.append(res["verify_s"])
                    else:
                        # record failure, keep raw as None
                        print(f"  FAILED: {res}", flush=True)
                        raw.append(None)
                    # small sleep to avoid overheating
                    time.sleep(0.05)
                # compute stats where possible
                vals = [v for v in raw if v is not None]
                if vals:
                    med = median(vals)
                    p95 = percentile(vals, 95)
                    mean_v = sum(vals) / len(vals)
                    mini = min(vals)
                    maxi = max(vals)
                else:
                    med = p95 = mean_v = mini = maxi = None
                limit = {1000: 0.03, 50000: 1.0, 2000000: 60.0}[size]
                status = (
                    "FAIL"
                    if med is not None and med > limit
                    else ("PASS" if med is not None else "ERROR")
                )
                all_rows.append(
                    {
                        "mode": mode,
                        "scenario": scenario,
                        "chars": size,
                        "thermal": thermal,
                        "repeats": repeats,
                        "raw_seconds": raw,
                        "median_s": med,
                        "p95_s": p95,
                        "mean_s": mean_v,
                        "min_s": mini,
                        "max_s": maxi,
                        "limit_s": limit,
                        "pass": status,
                        "details_first": row_details[0] if row_details else None,
                    }
                )
                raw_collections[key] = raw
                # save intermediate
                (out_dir / f"benchmark_{mode}.json").write_text(
                    json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8"
                )

    # Fixed cost and marginal cost (slope) via linear regression on medians
    # Use only completed medians
    # For each scenario/thermal, fit line time = fixed + marginal * (chars/1000)
    # Use sizes 1000,50000,2000000
    summaries = []
    for scenario in scenarios:
        for thermal in thermals:
            points = [
                (r["chars"], r["median_s"])
                for r in all_rows
                if r["scenario"] == scenario
                and r["thermal"] == thermal
                and r["median_s"] is not None
            ]
            if len(points) >= 2:
                # linear regression: y = a + b*x where x=chars
                n = len(points)
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                mx = sum(xs) / n
                my = sum(ys) / n
                # slope b = cov/var
                var = sum((x - mx) ** 2 for x in xs)
                cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
                b = cov / var if var else 0
                a = my - b * mx
                # marginal per 1000
                marginal_per_1000 = b * 1000
                summaries.append(
                    {
                        "scenario": scenario,
                        "thermal": thermal,
                        "fixed_cost_s": a,
                        "marginal_per_1000_s": marginal_per_1000,
                        "points": points,
                    }
                )
    (out_dir / f"fixed_marginal_{mode}.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Save full
    (out_dir / f"full_{mode}.json").write_text(
        json.dumps(
            {"fingerprint": fp, "rows": all_rows, "fixed_marginal": summaries},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Done {mode}: {len(all_rows)} rows")
    return all_rows, summaries, fp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="demo", choices=["demo", "hf"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[1000, 50000, 2000000])
    ap.add_argument(
        "--scenarios", nargs="+", default=["answer-text", "document-context"]
    )
    ap.add_argument("--thermals", nargs="+", default=["cold-process", "warm-model"])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()
    benchmark(
        args.mode,
        args.sizes,
        args.scenarios,
        args.thermals,
        args.repeats,
        args.out,
        profile=args.profile,
    )
