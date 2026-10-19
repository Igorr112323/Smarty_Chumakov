"""Byte-exact pre-P7 golden contract; synthetic/train-only input, no fitting."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p6_acceptance import verifier

from spanverify.core import tokenize_with_offsets
from spanverify.features import _normalised_support, map_token_positions


def payload():
    rng = random.Random(731)
    mappings = []
    for _ in range(300):
        tokens = [(rng.randrange(-20, 100), rng.randrange(-20, 100)) for _ in range(30)]
        spans = [(i, rng.randrange(-20, 100), rng.randrange(-20, 100)) for i in range(50)]
        start = rng.randrange(-10, 10)
        mappings.append(map_token_positions(tokens, spans, start))
    supports = []
    products = []
    v = verifier("demo")
    for answer in ["", "Один", "Один, два! 10 лет.", "ёжик ёлка\nдва десять — 123.", "Срок хранения 10 лет. " * 100]:
        tokens = tokenize_with_offsets(answer)
        supports.append(_normalised_support([rng.random() for _ in tokens], tokens))
        products.append(
            v.verify(answer, "Регламент: срок хранения 10 лет.", with_tokens=True).to_dict(with_tokens=True)
        )
    # Measured wall-clock telemetry cannot be byte-identical across executions.
    # Every predictive field, offset, text and full token record is retained.
    for product in products:
        product.pop("latency_ms", None)
    return dict(mappings=mappings, supports=supports, products=products)


def canonical(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


if __name__ == "__main__":
    path = Path(sys.argv[2])
    actual = canonical(payload())
    if sys.argv[1] == "capture":
        assert not path.exists(), "Never overwrite a pre-change golden"
        path.write_bytes(actual)
    else:
        assert actual == path.read_bytes(), "Golden I/O drift"
    print(json.dumps(dict(bytes=len(actual), sha256=hashlib.sha256(actual).hexdigest(), passed=True)))
