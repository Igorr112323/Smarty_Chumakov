"""P7 product-only measurements/receipts. Never inspect test177 or fit quality."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p6_acceptance import check_schema, document, environment, save, verifier


def rates(out):
    env = json.loads((out / "environment.json").read_text(encoding="utf-8"))
    cpu = next(
        (
            line.split(":", 1)[1].strip()
            for line in env.get("cpuinfo", "").splitlines()
            if line.startswith("model name")
        ),
        "unknown",
    )
    ram = next(
        (line for line in (env.get("meminfo") or "").splitlines() if line.startswith("MemTotal:")),
        "unknown",
    )
    summary = dict(
        phase=os.environ.get("P7_MEASUREMENT_PHASE", "after"),
        cpu=cpu,
        logical_cpu=env["cpu_count"],
        ram=ram,
        python=env["python"],
        platform=env["platform"],
        versions={
            k: v for k, v in env["versions"].items() if k.lower() in ("torch", "transformers", "numpy", "sentencepiece")
        },
        commit=env["commit"],
        run_url=env["run_url"],
    )
    print("::notice title=P7 benchmark environment::" + json.dumps(summary), flush=True)
    path = out / "benchmark.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    profiles = []
    for row in rows:
        n = row["chars"]
        row["seconds_per_1000_chars"] = row["median_seconds"] * 1000 / n if row["median_seconds"] is not None else None
        row["seconds_per_1000_chars_lower_bound"] = (
            row["median_lower_bound_seconds"] * 1000 / n if row.get("median_lower_bound_seconds") is not None else None
        )
        if row.get("diagnostic_profile"):
            profile = (out / row["diagnostic_profile"]).with_suffix(".summary.json")
            if profile.exists():
                row["profile_top"] = [
                    {k: v for k, v in r.items() if k in ("function", "calls", "self_seconds", "cumulative_seconds")}
                    for r in sorted(
                        json.loads(profile.read_text(encoding="utf-8")),
                        key=lambda r: r["self_seconds"],
                        reverse=True,
                    )[:5]
                ]
                profiles.append(
                    dict(
                        scenario=row["scenario"],
                        chars=n,
                        thermal=row["thermal"],
                        top=row["profile_top"],
                    )
                )
    # Actions preserves only ten notices per step: retain all twelve rows.
    columns = [
        "scenario",
        "chars",
        "thermal",
        "median_seconds",
        "max_seconds",
        "limit_seconds",
        "status",
        "seconds_per_1000_chars",
        "median_lower_bound_seconds",
        "seconds_per_1000_chars_lower_bound",
    ]
    print(
        "::notice title=P7 benchmark table::"
        + json.dumps(
            dict(
                mode=rows[0]["mode"],
                phase=os.environ.get("P7_MEASUREMENT_PHASE", "after"),
                columns=columns,
                rows=[[r.get(k) for k in columns] for r in rows],
            ),
            ensure_ascii=False,
        ),
        flush=True,
    )
    for i in range(0, len(profiles), 3):
        print(
            "::notice title=P7 benchmark profiles::"
            + json.dumps(
                dict(mode=rows[0]["mode"], cases=profiles[i : i + 3]),
                ensure_ascii=False,
            ),
            flush=True,
        )
    save(path, rows)


def functional(mode, out):
    a, c = document(2_000_000, "answer-text")
    comparison = {}

    def public_hash(result):
        payload = result.to_dict(with_tokens=True)
        payload.pop("latency_ms", None)
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    if mode == "hf" and os.environ.get("P7_COMPARE_SPAN_INDEX_BASELINE") == "1":
        from unittest.mock import patch

        from spanverify import engine, span_filter

        modules = []
        for name in ("engine", "span_filter"):
            module = types.ModuleType(f"spanverify._p7_preindex_{name}")
            module.__package__ = "spanverify"
            sys.modules[module.__name__] = module
            source = subprocess.check_output(
                ["git", "show", f"1c3aa13:spanverify-module/spanverify/{name}.py"],
                text=True,
            )
            exec(compile(source, "<accepted-preindex-P7>", "exec"), module.__dict__)
            modules.append(module)
        verifier(mode).verify(*document(1000, "document-context"))
        with (
            patch.object(engine.Verifier, "_shares", modules[0].Verifier._shares),
            patch.object(span_filter, "filter_candidates", modules[1].filter_candidates),
        ):
            began = time.perf_counter()
            before = verifier(mode).verify(a, c, with_tokens=True)
            comparison["before_seconds"] = time.perf_counter() - began
            comparison["before_public_sha256"] = public_hash(before)
            del before
        comparison.update(
            before_source="1c3aa13",
            thermal="both warm-model; same process",
            used_for_native_repeats=False,
        )
    start = time.perf_counter()
    result = verifier(mode).verify(a, c, with_tokens=True)
    seconds = time.perf_counter() - start
    check_schema(result, a)
    assert result.tokens and result.tokens[-1]["end"] == len(a)
    assert all(math.isfinite(t["raw_risk"]) and 0 <= t["raw_risk"] <= 1 for t in result.tokens)
    assert len(result.tokens) == result.stats["token_count"]
    if mode == "hf":
        assert result.stats["features"]["full_answer"] and result.stats["features"]["all_answer_tokens_mapped"]
    if comparison:
        comparison["after_public_sha256"] = public_hash(result)
        comparison["byte_equal"] = comparison["before_public_sha256"] == comparison["after_public_sha256"]
        assert comparison["byte_equal"], "Span index changed full 2M public JSON"
    report = dict(
        interval_index_comparison=comparison,
        mode=mode,
        seconds=seconds,
        seconds_per_1000_chars=seconds / 2000,
        answer_chars=len(a),
        utf8_bytes=len(a.encode()),
        token_records=len(result.tokens),
        last_end=result.tokens[-1]["end"],
        all_risks_finite_bounded=True,
        schema_valid=True,
        full_token_sha256=hashlib.sha256(
            json.dumps(result.tokens, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        features=result.stats["features"],
        environment=environment(),
        functional_passed=True,
        speed_60s_met=seconds <= 60,
    )
    save(out / "functional-2M.json", report)
    print(
        "::notice title=P7 full 2M::"
        + json.dumps({k: v for k, v in report.items() if k != "environment"}, ensure_ascii=False)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["rates", "functional"])
    p.add_argument("--mode", default="demo")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    rates(a.out) if a.action == "rates" else functional(a.mode, a.out)
