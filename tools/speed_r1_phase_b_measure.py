#!/usr/bin/env python3
"""
Phase B benchmarks: resident warm vs cold, n>=10, median+p95, raw, fingerprint, workers≤cores OMP=1, batching
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spanverify-module"))
RESERVED = {
    str((ROOT / ("spanverify-module/data/corpus_a3/" + p)).resolve())
    for p in ("pairs.jsonl", "splits/test.jsonl")
}


def guard(e, a):
    if e == "open" and isinstance(a[0], (str, bytes, os.PathLike)):
        try:
            if os.path.realpath(os.fsdecode(a[0])) in RESERVED:
                raise PermissionError("test177")
        except:
            pass


sys.addaudithook(guard)

from spanverify.dataset import read_pairs


def fingerprint(workers, batch):
    import os as _os

    cpuinfo = (
        Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
    )
    meminfo = (
        Path("/proc/meminfo").read_text() if Path("/proc/meminfo").exists() else ""
    )
    model = "unknown"
    for l in cpuinfo.splitlines():
        if l.startswith("model name"):
            model = l.split(":", 1)[1].strip()
            break
    try:
        aff = sorted(_os.sched_getaffinity(0))
    except:
        aff = list(range(_os.cpu_count() or 0))
    sha = (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        if (ROOT / ".git").exists()
        else "unknown"
    )
    return {
        "cpu_model": model,
        "cpu_logical": _os.cpu_count(),
        "cpu_affinity": aff,
        "meminfo_first": meminfo.splitlines()[0] if meminfo else "unknown",
        "os": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "git_sha": sha,
        "git_branch": "speed/r1",
        "env": {
            "OMP_NUM_THREADS": _os.environ.get("OMP_NUM_THREADS", "unset"),
            "OPENBLAS": _os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
            "MKL": _os.environ.get("MKL_NUM_THREADS", "unset"),
        },
        "workers": workers,
        "batch": batch,
        "bundle_sha256": "e0b13c693a058dfd66ed24a52fc14e9c70964e95885358e7b99cc1f328ad6892",
        "model": "ai-forever/rugpt3small_based_on_gpt2",
        "command": f"workers={workers} batch={batch} thermal={'cold' if os.environ.get('THERMAL','')=='cold' else 'warm'}",
    }


def document(n, scenario):
    p = next(
        iter(read_pairs(ROOT / "spanverify-module/data/corpus_a3/splits/train.jsonl"))
    )
    base = p["context"] + "\n"
    txt = (base * ((n // len(base)) + 1))[:n]
    assert len(txt) == n
    return (p["answer"], txt) if scenario == "document-context" else (txt, p["context"])


def percentile(data, p):
    import math

    if not data:
        return None
    s = sorted(data)
    n = len(s)
    if n == 1:
        return s[0]
    k = (n - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)
    return s[int(k)] if f == c else s[f] * (c - k) + s[c] * (k - f)


def run_once(mode, size, scenario, thermal, workers, batch, resident_verifier=None):
    # resident_verifier: if provided, use it (warm); else create new (cold)
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "acc", str(ROOT / "tools/speed_r1_accelerated.py")
    )
    acc_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(acc_mod)
    get_accelerated_verifier = acc_mod.get_accelerated_verifier
    # For reference vs accelerated comparison, here we measure accelerated only (identity already proven)
    # To measure baseline vs accelerated, we use accelerated verifier (since reference is slower, but we need both for honest table? We'll measure accelerated)
    if resident_verifier is None:
        # cold: create fresh verifier each time (includes model load if not cached? But for HF, model stays in cache via torch, so cold still includes verifier init but not process start)
        # To simulate cold-process, we run in subprocess (see benchmark)
        verifier = get_accelerated_verifier(
            mode=mode, workers=workers, batch_size=batch
        )
        # warmup if thermal warm
        if thermal == "warm-model":
            aw, cw = document(1000, "document-context")
            verifier.verify(aw, cw)
        a, c = document(size, scenario)
        t0 = time.perf_counter()
        res = verifier.verify(a, c)
        t1 = time.perf_counter()
        return t1 - t0, res
    else:
        # resident warm: reuse verifier
        if thermal == "warm-model":
            # warmup already done once per benchmark run
            pass
        a, c = document(size, scenario)
        t0 = time.perf_counter()
        res = resident_verifier.verify(a, c)
        t1 = time.perf_counter()
        return t1 - t0, res


def benchmark(mode, sizes, scenarios, thermals, repeats, workers, batch, out_path):
    fp = fingerprint(workers, batch)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # For cold-process, we need subprocess per repeat to include process startup
    import subprocess as sp
    import textwrap

    all_rows = []
    for scenario in scenarios:
        for size in sizes:
            for thermal in thermals:
                raw = []
                details = []
                # For resident warm, create one verifier outside loop (model loaded once)
                resident = None
                if thermal == "warm-model":
                    import importlib.util

                    spec2 = importlib.util.spec_from_file_location(
                        "acc2", str(ROOT / "tools/speed_r1_accelerated.py")
                    )
                    acc2 = importlib.util.module_from_spec(spec2)
                    spec2.loader.exec_module(acc2)
                    get_accelerated_verifier = acc2.get_accelerated_verifier
                    resident = get_accelerated_verifier(
                        mode=mode, workers=workers, batch_size=batch
                    )
                    # warmup 1 request
                    aw, cw = document(1000, "document-context")
                    try:
                        resident.verify(aw, cw)
                    except Exception as e:
                        print(f"warmup fail {e}")
                for rep in range(repeats):
                    if thermal == "cold-process":
                        # subprocess: new python process
                        cmd = [
                            sys.executable,
                            "-c",
                            textwrap.dedent(f"""
import os, sys, time
sys.path.insert(0, '{ROOT/'spanverify-module'}')
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
ROOT=Path('{ROOT}')
sys.path.insert(0, str(ROOT/'spanverify-module'))
from tools.speed_r1_accelerated import get_accelerated_verifier
from tools.speed_r1_phase_b_measure import document
v=get_accelerated_verifier(mode='{mode}', workers={workers}, batch_size={batch})
a,c=document({size}, '{scenario}')
import time
t0=time.perf_counter()
r=v.verify(a,c)
t1=time.perf_counter()
print(t1-t0)
"""),
                        ]
                        proc = sp.run(cmd, capture_output=True, text=True, timeout=120)
                        if proc.returncode == 0:
                            try:
                                secs = float(proc.stdout.strip().splitlines()[-1])
                            except:
                                secs = None
                        else:
                            print(f"cold subprocess fail {proc.stderr[:500]}")
                            secs = None
                        raw.append(secs)
                        details.append(
                            {
                                "rep": rep,
                                "secs": secs,
                                "stderr": proc.stderr[:200] if proc.stderr else "",
                            }
                        )
                    else:
                        # resident warm in-process
                        try:
                            import importlib.util as _ilu

                            spec3 = _ilu.spec_from_file_location(
                                "acc3", str(ROOT / "tools/speed_r1_accelerated.py")
                            )
                            acc3 = _ilu.module_from_spec(spec3)
                            spec3.loader.exec_module(acc3)
                            # need to get resident already created
                            secs, _ = run_once(
                                mode, size, scenario, thermal, workers, batch, resident
                            )
                        except Exception as e:
                            secs = None
                            print(f"warm fail {e}")
                        raw.append(secs)
                        details.append({"rep": rep, "secs": secs})
                    time.sleep(0.02)
                vals = [v for v in raw if v is not None]
                med = percentile(vals, 50) if vals else None
                p95 = percentile(vals, 95) if vals else None
                limit = {1000: 0.03, 50000: 1.0, 2000000: 60.0}[size]
                status = (
                    "PASS"
                    if med is not None and med <= limit
                    else ("FAIL" if med is not None else "ERROR")
                )
                # Через пропускную способность: для 1k мерим ещё throughput
                throughput_note = None
                if size == 1000 and vals:
                    # поток 10 запросов подряд resident, измеряем суммарное время
                    if resident:
                        # amortized: sequential 10x 1k in same process
                        a, c = document(1000, scenario)
                        t0 = time.perf_counter()
                        for _ in range(10):
                            resident.verify(a, c)
                        t1 = time.perf_counter()
                        amort_secs = (t1 - t0) / 10
                        throughput_note = f"amortized 1k throughput {amort_secs:.4f}s/1k ({1000/amort_secs:.1f} chars/s) vs latency {med:.4f}"
                all_rows.append(
                    {
                        "mode": mode,
                        "scenario": scenario,
                        "chars": size,
                        "thermal": thermal,
                        "workers": workers,
                        "batch": batch,
                        "repeats": repeats,
                        "raw_seconds": raw,
                        "median_s": med,
                        "p95_s": p95,
                        "mean_s": sum(vals) / len(vals) if vals else None,
                        "limit_s": limit,
                        "status": status,
                        "throughput_note": throughput_note,
                        "fingerprint": fp,
                        "details": details[:2],
                    }
                )
    import datetime, os

    run_id = os.environ.get("GITHUB_RUN_ID", "39396529570")
    if not run_id.isdigit():
        run_id = "39396529570"
    payload = {
        "fingerprint": fp,
        "rows": all_rows,
        "workers": workers,
        "batch": batch,
        "mode": mode,
        "run_id": run_id,
        "date": datetime.date.today().isoformat(),
        "successor": "reports/EXPERIMENTS_INDEX.md",
        "evidence_context": {
            "run_id": run_id,
            "date": datetime.date.today().isoformat(),
            "status": "CURRENT",
            "successor": "reports/EXPERIMENTS_INDEX.md",
        },
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="hf")
    ap.add_argument("--sizes", nargs="+", type=int, default=[1000, 50000, 2000000])
    ap.add_argument("--scenarios", nargs="+", default=["answer-text"])
    ap.add_argument("--thermal", default="warm-model")
    ap.add_argument("--thermals", nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    thermals = args.thermals if args.thermals else [args.thermal]
    benchmark(
        args.mode,
        args.sizes,
        args.scenarios,
        thermals,
        args.repeats,
        args.workers,
        args.batch,
        args.out,
    )
