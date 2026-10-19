"""P7 product-only measurements/receipts. Never inspect test177 or fit quality."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p6_acceptance import check_schema, document, environment, save, verifier


def rates(out):
    path = out / "benchmark.json"
    rows = json.loads(path.read_text())
    for row in rows:
        n = row.get("chars", row.get("size"))
        row["seconds_per_1000_chars"] = (
            row["median_seconds"] * 1000 / n if row.get("median_seconds") is not None else None
        )
        brief = {k: v for k, v in row.items() if k not in ("samples", "repeats")}
        print("::notice title=P7 speed::" + json.dumps(brief, ensure_ascii=False), flush=True)
    save(path, rows)


def functional(mode, out):
    a, c = document(2_000_000, "answer-text")
    start = time.perf_counter()
    result = verifier(mode).verify(a, c, with_tokens=True)
    seconds = time.perf_counter() - start
    check_schema(result, a)
    assert result.tokens and result.tokens[-1]["end"] == len(a)
    assert all(math.isfinite(t["raw_risk"]) and 0 <= t["raw_risk"] <= 1 for t in result.tokens)
    assert len(result.tokens) == result.stats["token_count"]
    if mode == "hf":
        assert result.stats["features"]["full_answer"] and result.stats["features"]["all_answer_tokens_mapped"]
    report = dict(
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
