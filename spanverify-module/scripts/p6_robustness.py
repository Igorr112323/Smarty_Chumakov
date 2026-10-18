"""Saved-evidence P6 robustness: no inference, training, threshold choice or test177."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p6_acceptance import BUNDLE, P5, ROOT, gz, save, verifier  # noqa: E402

from spanverify.core import tokenize_with_offsets  # noqa: E402
from spanverify.engine import WeightsBundle, _span_f1  # noqa: E402
from spanverify.features import is_scored_token  # noqa: E402
from spanverify.span_filter import build_candidates, filter_candidates  # noqa: E402

P4 = ROOT / "reports/experiments/p4_context_head/run39164044941"


def binary(tp, fp, fn, tn):
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
        "fpr": fp / (fp + tn) if fp + tn else 0,
    }


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["mode"]].append(row)
        groups["ALL"].append(row)
    result = {}
    for mode, items in groups.items():
        counts = [0, 0, 0, 0]
        for r in items:
            for gold, flag in r["tokens"]:
                idx = 0 if gold and flag else 1 if flag else 2 if gold else 3
                counts[idx] += 1
        block = {
            "pairs": len(items),
            "tokens": binary(*counts),
            "spans": _span_f1([r["spans"] for r in items], [r["truth"] for r in items]),
        }
        if all("answer" in r for r in items):
            counts = [0, 0, 0, 0]
            for r in items:
                gold, pred = r["answer"]
                counts[0 if gold and pred else 1 if pred else 2 if gold else 3] += 1
            block["answers"] = binary(*counts)
        else:
            block["answers"] = {
                "status": "not measured",
                "reason": "P5 does not store complete OOF answer predictions; no replacement or new inference",
            }
        result[mode] = block
    return result


def main(out):
    bundle = WeightsBundle.load(BUNDLE)
    v = verifier()
    config = bundle.span_filter
    pools = defaultdict(list)
    probs = {r["index"]: r["p"] for r in gz(P5 / "oof_span_probabilities.jsonl.gz")}
    for row in gz(P5 / "candidate_pool.jsonl.gz"):
        if row["cell"] == 218:
            pools[row["pair_id"]].append(row)
    train = {p["id"]: p for p in map(json.loads, (ROOT / "data/corpus_a3/splits/train.jsonl").read_text().splitlines())}
    risks = {r["id"]: r["smoothed"] for r in gz(P4 / "k3_w1_oof.jsonl.gz")}
    rows = []
    for state in gz(P4 / "source_oof_states.jsonl.gz"):
        pair = train[state["id"]]
        tokens = tokenize_with_offsets(state["answer"])
        smoothed = risks[state["id"]]
        pool = pools[state["id"]]
        rules = [
            *v._attribution_spans(state["answer"], state["context"], tokens, smoothed, bundle.token_threshold),
            *v._coverage_spans(state["answer"], state["context"], tokens, bundle.token_threshold),
        ]
        p1_score = pool[0]["features"][-1] if pool else 0.0
        candidates = build_candidates(
            state["answer"], tokens, smoothed, state["scored"], config["generator"], p1_score, rules
        )
        assert len(candidates) == len(pool), state["id"]
        for (span, feature), ref in zip(candidates, pool, strict=True):
            assert (span.start, span.end) == (ref["start"], ref["end"])
            assert max(abs(a - b) for a, b in zip(feature, ref["features"], strict=True)) <= 1e-12
        flags, spans = filter_candidates(
            state["answer"], tokens, candidates, config, rules, [probs[r["index"]] for r in pool]
        )
        rows.append(
            {
                "mode": pair["meta"]["mode"],
                "tokens": [
                    (g, i in flags) for i, (g, m) in enumerate(zip(state["gold"], state["scored"], strict=True)) if m
                ],
                "spans": [(s.start, s.end) for s in spans],
                "truth": state["truth"],
            }
        )
    oof = summarize(rows)
    assert [oof["ALL"]["tokens"][k] for k in ("tp", "fp", "fn", "tn")] == [919, 422, 465, 44801]
    assert [oof["ALL"]["spans"][k] for k in ("tp", "fp", "fn")] == [214, 105, 253]
    dev = {p["id"]: p for p in map(json.loads, (ROOT / "data/corpus_a3/splits/dev.jsonl").read_text().splitlines())}
    answers = {a["pair_id"]: a for a in json.loads((P5 / "dev_answers.json").read_text())}
    rows = []
    for ref in gz(P5 / "dev_evidence.jsonl.gz"):
        p = dev[ref["id"]]
        truth = [(a, b) for a, b, y in p["labels"] if y == 1]
        labels = []
        for t in ref["tokens"]:
            if is_scored_token(t["text"]):
                gold = any(t["start"] < b and t["end"] > a for a, b in truth)
                labels.append((gold, t["flagged"]))
        answer = answers[ref["id"]]
        rows.append(
            {
                "mode": p["meta"]["mode"],
                "tokens": labels,
                "spans": [(s["start"], s["end"]) for s in ref["spans"]],
                "truth": truth,
                "answer": (answer["gold"], answer["prediction"]),
            }
        )
    dev_result = summarize(rows)
    assert [dev_result["ALL"]["tokens"][k] for k in ("tp", "fp", "fn", "tn")] == [192, 134, 95, 7548]
    assert [dev_result["ALL"]["spans"][k] for k in ("tp", "fp", "fn")] == [46, 32, 50]
    save(
        Path(out) / "robustness.json",
        {
            "mapping": "reports/acceptance/protocol.json (published before these numbers)",
            "train_oof": oof,
            "saved_dev": dev_result,
            "no_fitting": True,
            "caution": "Mode proxies, not new perturbation tests; OOF answer unavailable is not replaced by P1 or in-sample predictions.",
        },
    )


if __name__ == "__main__":
    main(sys.argv[1])
