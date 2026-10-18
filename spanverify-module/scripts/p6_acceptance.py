"""P6: frozen-product acceptance only. No fit, threshold choice, or test177 access."""

from __future__ import annotations

import argparse
import cProfile
import gzip
import hashlib
import json
import os
import platform
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spanverify.config import Config  # noqa: E402
from spanverify.dataset import read_pairs  # noqa: E402
from spanverify.engine import Verifier, WeightsBundle  # noqa: E402

P5 = ROOT / "reports/experiments/p5_span_filter/run39170981235"
BUNDLE = P5 / "combined_bundle.json"
SHA = "e0b13c693a058dfd66ed24a52fc14e9c70964e95885358e7b99cc1f328ad6892"
MODEL = "ai-forever/rugpt3small_based_on_gpt2"
RESERVED = {str((ROOT / f"data/corpus_a3/{p}").resolve()) for p in ("pairs.jsonl", "splits/test.jsonl")}


def guard(event, args):
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        if os.path.realpath(os.fsdecode(args[0])) in RESERVED:
            raise PermissionError("P6 official test177 is reserved")


sys.addaudithook(guard)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def verifier(mode="hf", cache=None):
    assert hashlib.sha256(BUNDLE.read_bytes()).hexdigest() == SHA
    config = Config.load().with_overrides(backend=mode, hf_model=MODEL, hf_max_tokens=512)
    return Verifier(
        config=config,
        weights=WeightsBundle.load(BUNDLE),
        mode=mode,
        model_name=MODEL,
        features_cache=cache,
        span_expansion="mask",
    )


def environment():
    from importlib.metadata import distributions

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cpuinfo": Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else platform.processor(),
        "meminfo": Path("/proc/meminfo").read_text() if Path("/proc/meminfo").exists() else None,
        "versions": {d.metadata["Name"]: d.version for d in distributions()},
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "run_url": f"https://github.com/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/{os.getenv('GITHUB_RUN_ID')}",
        "bundle_sha256": SHA,
        "feature_cache": False,
    }


def parity(cache_path, out):
    from spanverify.features import feature_cache_key, load_feature_cache

    cache = load_feature_cache(list(Path(cache_path).rglob("features_*.jsonl")), strict=True, counting=True)
    v = verifier(cache=cache)
    ref = {r["id"]: r for r in gz(P5 / "dev_evidence.jsonl.gz")}
    answers = json.loads((P5 / "dev_answers.json").read_text())
    if isinstance(answers, dict):
        answers = answers.get("rows", answers.get("answers", answers))
    answer_ref = {r["pair_id"]: r for r in answers}
    mismatches = []
    delta = 0.0
    risks = 0
    for p in read_pairs(ROOT / "data/corpus_a3/splits/dev.jsonl"):
        key = feature_cache_key(p["answer"], p["context"], "hf", MODEL, chunking=True, max_length=512)
        assert key in cache, p["id"]  # never fall through to fresh inference
        result = v.verify(p["answer"], p["context"], with_tokens=True)
        expected = ref[p["id"]]
        actual = {
            "id": p["id"],
            "spans": [asdict(s) for s in result.spans],
            "tokens": result.tokens,
            "verdict": result.verdict,
        }
        if actual != expected:
            mismatches.append(p["id"])
        a = answer_ref[p["id"]]
        for val, target in [
            (result.score, a["calibrated_score"]),
            (result.threshold, a["threshold"]),
            (result.stats["raw_score_full"], a["raw_score"]),
        ]:
            delta = max(delta, abs(val - target))
        risks += len(result.tokens)
    report = {
        "pairs": len(ref),
        "full_token_risks": risks,
        "mismatches": mismatches,
        "p1_max_abs_delta": delta,
        "cache_misses": cache.misses,
        "bundle_sha256": SHA,
        "passed": not mismatches and delta == 0 and risks == 13150 and cache.misses == 0,
        "environment": environment(),
    }
    save(out / "parity.json", report)
    assert report["passed"], report


def document(n, scenario):
    p = next(iter(read_pairs(ROOT / "data/corpus_a3/splits/train.jsonl")))
    base = p["context"] + "\n"
    text = (base * (n // len(base) + 1))[:n]
    assert len(text) == n
    return (p["answer"], text) if scenario == "document-context" else (text, p["context"])


def check_schema(result, answer):
    assert 0 <= result.score <= 1
    for span in result.spans:
        assert 0 <= span.start < span.end <= len(answer)
        assert answer[span.start : span.end] == span.text
        assert span.label and 0 <= span.risk <= 1 and 0 <= span.confidence <= 1


def worker(args):
    a, c = document(args.size, args.scenario)
    v = verifier(args.mode)
    warmup_seconds = None
    if args.thermal == "warm-model":
        started = time.perf_counter()
        v.verify(*document(1000, "document-context"))
        warmup_seconds = time.perf_counter() - started
    profiler = cProfile.Profile()

    def interrupted(*_):
        profiler.disable()
        profiler.dump_stats(str(args.out) + ".prof")
        raise SystemExit(143)

    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, interrupted)
    started = time.perf_counter()
    profiler.enable()
    result = v.verify(a, c)
    profiler.disable()
    elapsed = time.perf_counter() - started
    profiler.dump_stats(str(args.out) + ".prof")
    check_schema(result, a)
    save(
        args.out,
        {
            "status": "completed",
            "seconds": elapsed,
            "warmup_seconds": warmup_seconds,
            "answer_chars": len(a),
            "context_chars": len(c),
            "answer_utf8_bytes": len(a.encode()),
            "context_utf8_bytes": len(c.encode()),
            "input_sha256": hashlib.sha256((a + c).encode()).hexdigest(),
            "schema_valid": True,
            "stats": result.stats,
            "spans": [asdict(s) for s in result.spans],
            "score": result.score,
            "verdict": result.verdict,
        },
    )


def benchmark(args):
    save(args.out / "environment.json", environment())
    rows = []
    for scenario in ("document-context", "answer-text"):
        for size in (1000, 50000, 2000000):
            for thermal in ("cold-process", "warm-model"):
                samples = []
                for repeat in range(3):
                    name = f"{args.mode}-{scenario}-{size}-{thermal}-{repeat}"
                    path = args.out / f"{name}.json"
                    command = [
                        sys.executable,
                        __file__,
                        "worker",
                        "--mode",
                        args.mode,
                        "--size",
                        str(size),
                        "--scenario",
                        scenario,
                        "--thermal",
                        thermal,
                        "--out",
                        str(path),
                    ]
                    with (args.out / f"{name}.log").open("w") as log:
                        proc = subprocess.Popen(command, stdout=log, stderr=log)
                        try:
                            code = proc.wait(timeout=75)
                        except subprocess.TimeoutExpired:
                            proc.terminate()
                            try:
                                proc.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                                proc.wait()
                            code = 124
                    if path.exists():
                        sample = json.loads(path.read_text())
                    else:
                        sample = {
                            "status": "timeout" if code == 124 else "error",
                            "returncode": code,
                            "seconds": None,
                            "process_lower_bound_seconds": 75 if code == 124 else None,
                        }
                        save(path, sample)
                    samples.append(sample)
                complete = all(s["status"] == "completed" for s in samples)
                values = [s["seconds"] for s in samples if s["seconds"] is not None]
                limit = {1000: 0.03, 50000: 1.0, 2000000: 60.0}[size]
                row = {
                    "mode": args.mode,
                    "scenario": scenario,
                    "chars": size,
                    "thermal": thermal,
                    "repeats": samples,
                    "median_seconds": statistics.median(values) if complete else None,
                    "max_seconds": max(values) if complete else None,
                    "limit_seconds": limit,
                    "status": "достигнут" if complete and max(values) <= limit else "не достигнут",
                }
                rows.append(row)
                save(args.out / "benchmark.json", rows)
                print(args.mode, scenario, size, thermal, row["status"], flush=True)


def shares(args):
    v = verifier(args.mode)
    pairs = list(read_pairs(ROOT / "data/corpus_a3/splits/train.jsonl"))
    rows = []
    for pair in pairs[:3]:
        human = (pair["context"] * (800 // len(pair["context"]) + 1))[:800]
        # Explicit programmatic generator, not falsely labeled real LLM authorship.
        machine = (
            "Таким образом, необходимо отметить, что комплексная оптимизация процессов "
            "является ключевым аспектом эффективного функционирования организации. " * 8
        )[:800]
        for q in (0.0, 0.25, 0.5, 0.75, 1.0):
            cut = int(800 * q)
            text = human[: 800 - cut] + machine[:cut]
            result = v.verify(text, pair["context"])
            rows.append(
                {
                    "train_pair_id": pair["id"],
                    "fraction": q,
                    "machine_chars": cut,
                    "total_chars": len(text),
                    "ai_share": result.ai_share,
                    "ai_participation": result.ai_participation,
                }
            )
    mae = sum(abs(r["fraction"] - r["ai_share"]) for r in rows) / len(rows)
    curves = [[r for r in rows if r["fraction"] == q] for q in (0.0, 0.25, 0.5, 0.75, 1.0)]
    mean_curve = [statistics.mean(r["ai_share"] for r in curve) for curve in curves]
    save(
        args.out / "ai-share.json",
        {
            "rows": rows,
            "mae": mae,
            "mean_curve": mean_curve,
            "monotone_mean": all(a <= b for a, b in zip(mean_curve, mean_curve[1:], strict=False)),
            "synthetic_generator_only": True,
            "calibration_fitted": False,
            "mode": args.mode,
        },
    )


def interfaces(args):
    import urllib.error
    import urllib.request

    env = dict(os.environ, SPANVERIFY_HF_MODEL=MODEL, SPANVERIFY_HF_MAX_TOKENS="512")
    a, c = document(1000, "document-context")
    ap, cp = args.out / "answer.txt", args.out / "context.txt"
    ap.write_text(a, encoding="utf-8")
    cp.write_text(c, encoding="utf-8")
    cli_command = [
        sys.executable,
        "-m",
        "spanverify",
        "verify",
        "--answer-file",
        str(ap),
        "--context-file",
        str(cp),
        "--json",
        "--mode",
        args.mode,
        "--weights",
        str(BUNDLE),
    ]
    cli = subprocess.run(cli_command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    (args.out / "cli.stdout.json").write_text(cli.stdout, encoding="utf-8")
    (args.out / "cli.stderr.log").write_text(cli.stderr, encoding="utf-8")
    assert cli.returncode in (0, 1), cli.stderr
    payload = json.loads(cli.stdout)
    assert cli.returncode == (0 if payload["verdict"] in ("grounded", "empty") else 1)
    request = {"answer": a, "context": c}
    save(args.out / "http-request.json", request)
    command = [
        sys.executable,
        "-m",
        "spanverify",
        "server",
        "--host",
        "0.0.0.0",
        "--port",
        "8765",
        "--no-browser",
        "--quiet",
        "--mode",
        args.mode,
        "--weights",
        str(BUNDLE),
    ]
    with (args.out / "server.log").open("w") as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 120
            while True:
                try:
                    with urllib.request.urlopen("http://127.0.0.1:8765/health", timeout=5) as response:
                        save(args.out / "health.json", json.load(response))
                    break
                except (OSError, TimeoutError):
                    if time.monotonic() > deadline or proc.poll() is not None:
                        raise RuntimeError("server failed startup") from None
                    time.sleep(0.2)
            with urllib.request.urlopen("http://127.0.0.1:8765/", timeout=10) as response:
                page = response.read()
                assert response.status == 200 and b"<html" in page.lower()
                (args.out / "web.html").write_bytes(page)
            req = urllib.request.Request(
                "http://127.0.0.1:8765/v1/verify", json.dumps(request).encode(), {"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=120) as response:
                api = json.load(response)
                save(args.out / "http-response.json", api)
            for key in ("score", "threshold", "verdict", "spans", "ai_share", "ai_participation"):
                assert payload[key] == api[key], key
            save(
                args.out / "interfaces.json",
                {
                    "mode": args.mode,
                    "cli_returncode": cli.returncode,
                    "cli_command": cli_command,
                    "server_command": command,
                    "web_status": 200,
                    "http_status": 200,
                    "parity_fields": ["score", "threshold", "verdict", "spans", "ai_share"],
                    "passed": True,
                    "environment": environment(),
                },
            )
        finally:
            proc.terminate()
            proc.wait(timeout=20)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["parity", "benchmark", "worker", "shares", "interfaces"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache", type=Path)
    p.add_argument("--mode", choices=["demo", "hf"], default="hf")
    p.add_argument("--size", type=int, default=1000)
    p.add_argument("--scenario", default="document-context")
    p.add_argument("--thermal", default="cold-process")
    args = p.parse_args()
    if args.action != "worker":
        args.out.mkdir(parents=True, exist_ok=True)
    if args.action == "parity":
        parity(args.cache, args.out)
    else:
        {"benchmark": benchmark, "worker": worker, "shares": shares, "interfaces": interfaces}[args.action](args)


if __name__ == "__main__":
    main()
