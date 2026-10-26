"""P3 seed/grow localization. Pure Python, no model fitting or mutable risk state."""

from __future__ import annotations

import math
from dataclasses import replace

from .core import SpanResult

DECODER_TYPE = "seed_grow_v1"
QUANTILES = (
    0.50,
    0.75,
    0.80,
    0.85,
    0.90,
    0.92,
    0.94,
    0.95,
    0.96,
    0.965,
    0.97,
    0.975,
    0.98,
    0.9825,
    0.985,
    0.9875,
    0.99,
    0.9925,
    0.995,
    0.9975,
    0.999,
    1.0,
)


def validate_decoder(config):
    required = {
        "type",
        "schema_version",
        "tau_low",
        "tau_high",
        "bridge_unscored_tokens",
        "min_seed_tokens",
        "attribution_boundary",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("Malformed localization decoder fields")
    if config["type"] != DECODER_TYPE or type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("Unknown localization decoder schema/type")
    for name in ("tau_low", "tau_high"):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Decoder cutoff must be finite and nonnegative; above-max sentinel is allowed")
    if config["tau_high"] < config["tau_low"]:
        raise ValueError("tau_high must be >= tau_low")
    for name, values in (
        ("bridge_unscored_tokens", (0, 1)),
        ("min_seed_tokens", (1, 2)),
    ):
        if type(config[name]) is not int or config[name] not in values:
            raise ValueError(f"Invalid {name}")
    if config["attribution_boundary"] != "exact":
        raise ValueError("P3 requires exact attribution anchors")
    return config


def decoder_config(low, high, bridge=0, seeds=1):
    return validate_decoder(
        dict(
            type=DECODER_TYPE,
            schema_version=1,
            tau_low=float(low),
            tau_high=float(high),
            bridge_unscored_tokens=bridge,
            min_seed_tokens=seeds,
            attribution_boundary="exact",
        )
    )


def merge_localized_spans(answer, tokens, spans):
    """Merge without stale text, duplicate token counts, or lost provenance."""
    merged = []
    for span in sorted(spans, key=lambda s: (s.start, s.end)):
        if not 0 <= span.start < span.end <= len(answer):
            raise ValueError("Invalid span coordinates")
        if merged and span.start <= merged[-1].end:
            prev = merged.pop()
            start, end = prev.start, max(prev.end, span.end)
            sources = tuple(sorted(set((prev.sources or (prev.source,)) + (span.sources or (span.source,)))))
            indices = tuple(sorted(set(prev.token_indices) | set(span.token_indices)))
            merged.append(
                SpanResult(
                    start=start,
                    end=end,
                    text=answer[start:end],
                    risk=max(prev.risk, span.risk),
                    label=(
                        "likely_hallucination" if "likely_hallucination" in (prev.label, span.label) else "doubtful"
                    ),
                    source=sources[0] if len(sources) == 1 else "rule",
                    sources=sources,
                    token_indices=indices,
                    n_tokens=sum(t.start < end and t.end > start for t in tokens),
                )
            )
        else:
            merged.append(
                replace(
                    span,
                    text=answer[span.start : span.end],
                    sources=span.sources or (span.source,),
                )
            )
    return merged


def localize_state(answer, tokens, risks, scored_mask, config, attribution_spans=(), coverage_spans=()):
    """One canonical decoder for OOF selection and public Verifier.

    Inputs are full sequences, never truncated to the last scored token. Only
    unscored gaps may be bridged; a low-risk scored word always breaks a component.
    Rule spans do not create token labels. Empty retained set is authoritative.
    """
    validate_decoder(config)
    if not (len(tokens) == len(risks) == len(scored_mask)):
        raise ValueError("Token/risk/scored-mask length mismatch")
    if any(not math.isfinite(r) for r in risks):
        raise ValueError("Nonfinite token risk")
    low, high = config["tau_low"], config["tau_high"]
    groups, current = [], []
    previous = None
    for i, (risk, scored) in enumerate(zip(risks, scored_mask, strict=False)):
        if not scored:
            continue
        if risk < low:
            if current:
                groups.append(current)
            current, previous = [], None
            continue
        if previous is not None and i - previous - 1 > config["bridge_unscored_tokens"]:
            groups.append(current)
            current = []
        current.append(i)
        previous = i
    if current:
        groups.append(current)
    retained, masks = set(), []
    for group in groups:
        if sum(risks[i] >= high for i in group) < config["min_seed_tokens"]:
            continue
        retained.update(group)
        start, end = tokens[group[0]].start, tokens[group[-1]].end
        # Identical predetermined whitespace trim for all components, no expansion.
        while start < end and answer[start].isspace():
            start += 1
        while end > start and answer[end - 1].isspace():
            end -= 1
        risk = max(risks[i] for i in group)
        masks.append(
            SpanResult(
                start=start,
                end=end,
                text=answer[start:end],
                risk=risk,
                label="likely_hallucination" if risk >= 0.75 else "doubtful",
                n_tokens=len(group),
                token_indices=tuple(group),
                source="mask",
                sources=("mask",),
            )
        )
    spans = merge_localized_spans(answer, tokens, [*masks, *attribution_spans, *coverage_spans])
    return retained, masks, spans
