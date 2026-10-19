"""Complete 2M-answer functional request, separate from bounded speed trials.

No model changes, no cache, no fitting. Demo only: native HF long-answer
truncation is a separately recorded failure, not silently worked around.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p6_acceptance import check_schema, document, environment, save, verifier  # noqa: E402


def main(out):
    answer, context = document(2_000_000, "answer-text")
    started = time.perf_counter()
    result = verifier("demo").verify(answer, context, with_tokens=True)
    seconds = time.perf_counter() - started
    check_schema(result, answer)
    assert result.tokens and result.tokens[-1]["end"] == len(answer)
    assert len(result.tokens) == result.stats["token_count"]
    assert all(0 <= t["raw_risk"] <= 1 for t in result.tokens)
    canonical = json.dumps(result.tokens, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    save(
        out,
        {
            "mode": "demo",
            "functional_passed": True,
            "answer_chars": len(answer),
            "answer_utf8_bytes": len(answer.encode()),
            "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
            "context_chars": len(context),
            "token_count": len(result.tokens),
            "last_token_end": result.tokens[-1]["end"],
            "all_full_token_risks_finite_and_bounded": True,
            "all_spans_schema_and_slices_valid": True,
            "full_token_records_sha256": hashlib.sha256(canonical).hexdigest(),
            "seconds": seconds,
            "speed_limit_60s_met": seconds <= 60,
            "environment": environment(),
            "stats": result.stats,
            "score": result.score,
            "verdict": result.verdict,
            "note": "Real sandbox CPU product request; separate environment from GitHub benchmarks. No HF full-answer coverage claim.",
        },
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
