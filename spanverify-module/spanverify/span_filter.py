"""P5 pure versioned candidate builder and portable keep/drop filter.

No labels, pair clean/dirty indicators, fitting libraries or mutable risk state.
P3 bridge validation is deliberately NOT changed; P5 owns bridge=2 semantics.
"""

from __future__ import annotations

import math
from dataclasses import replace

from .core import SpanResult
from .localization import merge_localized_spans

FEATURES = (
    "scored_count",
    "seed_count",
    "max_risk",
    "mean_risk",
    "min_risk",
    "risk_mass",
    "low_fraction",
    "high_fraction",
    "char_width",
    "token_width",
    "bridged_gaps",
    "bridged_tokens",
    "left_gradient",
    "right_gradient",
    "exact_anchor",
    "rule_tag",
    "relative_start",
    "relative_end",
    "p1_calibrated_score",
)
KEEP_QUANTILES = (0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99, 1)
LOW_QUANTILES = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90)
HIGH_QUANTILES = (0.90, 0.94, 0.96, 0.975, 0.985)


def validate_generator(c):
    if not isinstance(c, dict) or set(c) != {"tau_low", "tau_high", "bridge", "min_seed"}:
        raise ValueError("Malformed P5 generator")
    if any(
        type(c[k]) not in (int, float) or not math.isfinite(c[k]) or not 0 <= c[k] <= 1 for k in ("tau_low", "tau_high")
    ):
        raise ValueError("Invalid P5 cutoffs")
    if (
        c["tau_low"] > c["tau_high"]
        or type(c["bridge"]) is not int
        or c["bridge"] not in (0, 1, 2)
        or type(c["min_seed"]) is not int
        or c["min_seed"] not in (1, 2)
    ):
        raise ValueError("Invalid P5 generator cell")
    return c


def feature_vector(values):
    if not isinstance(values, dict) or set(values) != set(FEATURES):
        raise ValueError("Only registered gold-free span features are permitted")
    row = [values[n] for n in FEATURES]
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in row):
        raise ValueError("Nonfinite span feature")
    return row


def build_candidates(answer, tokens, risks, scored, generator, p1_score, rules=()):
    validate_generator(generator)
    if not (len(tokens) == len(risks) == len(scored)) or any(type(x) is not bool for x in scored):
        raise ValueError("P5 input alignment")
    if (
        any(not math.isfinite(r) or not 0 <= r <= 1 for r in risks)
        or not math.isfinite(p1_score)
        or not 0 <= p1_score <= 1
    ):
        raise ValueError("Nonfinite/out-of-range P5 input")
    low, high = generator["tau_low"], generator["tau_high"]
    groups, current, previous = [], [], None
    for i, (v, m) in enumerate(zip(risks, scored, strict=True)):
        if not m:
            continue
        if v < low:
            if current:
                groups.append(current)
            current, previous = [], None
            continue
        if previous is not None and i - previous - 1 > generator["bridge"]:
            groups.append(current)
            current = []
        current.append(i)
        previous = i
    if current:
        groups.append(current)
    result = []
    for ids in groups:
        vals = [risks[i] for i in ids]
        seeds = sum(v >= high for v in vals)
        if seeds < generator["min_seed"]:
            continue
        a, b = tokens[ids[0]].start, tokens[ids[-1]].end
        gaps = [j - i - 1 for i, j in zip(ids, ids[1:], strict=False)]
        left = next((risks[i] for i in range(ids[0] - 1, -1, -1) if scored[i]), vals[0])
        right = next((risks[i] for i in range(ids[-1] + 1, len(tokens)) if scored[i]), vals[-1])
        overlap = [r for r in rules if r.start < b and r.end > a]
        values = dict(
            zip(
                FEATURES,
                (
                    len(ids),
                    seeds,
                    max(vals),
                    sum(vals) / len(vals),
                    min(vals),
                    sum(vals),
                    sum(v >= low for v in vals) / len(vals),
                    seeds / len(vals),
                    b - a,
                    ids[-1] - ids[0] + 1,
                    sum(g > 0 for g in gaps),
                    sum(gaps),
                    vals[0] - left,
                    vals[-1] - right,
                    int(any("attribution" in (r.sources or (r.source,)) for r in overlap)),
                    int(bool(overlap)),
                    a / max(1, len(answer)),
                    b / max(1, len(answer)),
                    p1_score,
                ),
                strict=True,
            )
        )
        span = SpanResult(
            start=a,
            end=b,
            text=answer[a:b],
            risk=max(vals),
            label="likely_hallucination" if max(vals) >= 0.75 else "doubtful",
            n_tokens=len(ids),
            token_indices=tuple(ids),
            source="mask",
            sources=("mask",),
        )
        result.append((span, feature_vector(values)))
    return result


def validate_filter(c):
    required = {"schema", "feature_order", "generator", "head", "keep_threshold"}
    if (
        not isinstance(c, dict)
        or set(c) != required
        or c["schema"] != "span_filter_schema_v1"
        or c["feature_order"] != list(FEATURES)
    ):
        raise ValueError("Malformed span_filter_schema_v1")
    validate_generator(c["generator"])
    h = c["head"]
    if (
        not isinstance(h, dict)
        or set(h) != {"type", "mean", "scale", "coef", "intercept"}
        or h["type"] != "standardized_logreg_v1"
    ):
        raise ValueError("Unknown span head")
    for key in ("mean", "scale", "coef"):
        if (
            not isinstance(h[key], list)
            or len(h[key]) != len(FEATURES)
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in h[key])
        ):
            raise ValueError("Malformed span head vector")
    if any(x <= 0 for x in h["scale"]) or type(h["intercept"]) not in (int, float) or not math.isfinite(h["intercept"]):
        raise ValueError("Invalid span scale/intercept")
    s = c["keep_threshold"]
    if type(s) not in (int, float) or not math.isfinite(s) or not 0 <= s <= math.nextafter(1.0, 2.0):
        raise ValueError("Invalid keep threshold")
    return c


def probability(row, head):
    if len(row) != len(FEATURES) or any(not math.isfinite(v) for v in row):
        raise ValueError("Invalid span feature row")
    z = head["intercept"] + sum(
        (v - m) / s * w for v, m, s, w in zip(row, head["mean"], head["scale"], head["coef"], strict=True)
    )
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def filter_candidates(answer, tokens, candidates, config, rules=(), probabilities=None):
    """Exactly one canonical merge/keep implementation, also used by OOF search.

    Optional cross-fitted probabilities never replace raw token risks. Rules are
    mandatory evidence and are NEVER passed through keep/drop.
    """
    validate_filter(config)
    ps = [probability(row, config["head"]) for _, row in candidates] if probabilities is None else probabilities
    if len(ps) != len(candidates) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in ps):
        raise ValueError("Invalid span probabilities")
    kept = [replace(s) for (s, _), p in zip(candidates, ps, strict=True) if p >= config["keep_threshold"]]
    spans = merge_localized_spans(answer, tokens, [*kept, *rules])
    flags = {i for i, t in enumerate(tokens) if any(t.start < s.end and t.end > s.start for s in spans)}
    return flags, spans
