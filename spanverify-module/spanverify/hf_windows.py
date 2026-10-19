"""Full-answer HF windows. No document selection, model fitting or quality thresholds."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import fields

from .core import Token


def character_windows(text, offsets, budget):
    """Partition all characters, keeping shared Unicode/BPE offsets together."""
    if budget < 1:
        raise ValueError("positive window budget required")
    offsets = [(int(a), int(b)) for a, b in offsets if b > a]
    if not offsets:
        return [(0, len(text))] if text else []
    out = []
    start, i = 0, 0
    while i < len(offsets):
        stop = min(len(offsets), i + budget)
        end = offsets[stop - 1][1]
        while stop < len(offsets) and offsets[stop][0] < end:
            end = max(end, offsets[stop][1])
            stop += 1
        end = offsets[stop][0] if stop < len(offsets) else len(text)
        if end <= start:
            raise ValueError("nonprogressing tokenizer offsets")
        out.append((start, end))
        start, i = end, stop
    assert start == len(text)
    return out


def full_answer_features(answer, context, tokens, tokenizer, runtime, scorer, options):
    """Score all answer windows against all subwindows of the selected context.

    Existing top-3 context selection is outside this function and unchanged.
    Long-only semantics: max support over context subwindows, character-weighted
    mean over fragments of a lexical token crossing answer-window boundaries.
    No unscored tail, duplicate output tokens or synthetic risk defaults.
    """
    from .features import FeatureMatrix

    limit = options["max_length"]
    if limit < 16:
        raise ValueError("HF full-answer windows require max_length >= 16")
    budget = max(1, limit // 4)
    offsets = tokenizer(answer, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    windows = character_windows(answer, offsets, budget)
    context_offsets = tokenizer(context, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    contexts = character_windows(context, context_offsets, budget) or [(0, 0)]
    names = [f.name for f in fields(FeatureMatrix) if f.name not in ("meta", "token_risk")]
    sums = {name: [0.0] * len(tokens) for name in names}
    weights = [0] * len(tokens)
    ends = [t.end for t in tokens]
    calls = 0
    for start, end in windows:
        indices = []
        local = []
        i = bisect_right(ends, start)
        while i < len(tokens) and tokens[i].start < end:
            t = tokens[i]
            a, b = max(start, t.start), min(end, t.end)
            if b > a:
                indices.append((i, b - a))
                local.append(Token(answer[a:b], a - start, b - start))
            i += 1
        if not local:
            continue
        maxima = None
        for a, b in contexts:
            part = scorer(
                answer[start:end], context[a:b], answer_tokens=local, _runtime=runtime, _windowed=True, **options
            )
            if not part.meta.get("all_answer_tokens_mapped"):
                raise ValueError("HF window has unscored answer tokens; refusing zero fill")
            if maxima is None:
                maxima = {name: list(getattr(part, name)) for name in names}
            else:
                for name in names:
                    maxima[name] = [max(x, y) for x, y in zip(maxima[name], getattr(part, name), strict=True)]
            calls += 1
        for j, (index, weight) in enumerate(indices):
            weights[index] += weight
            for name in names:
                sums[name][index] += maxima[name][j] * weight
    if any(w == 0 for w in weights):
        raise ValueError("HF answer coverage gap; refusing synthetic defaults")
    return FeatureMatrix(
        **{name: [v / w for v, w in zip(values, weights, strict=True)] for name, values in sums.items()},
        meta=dict(
            backend="hf",
            model=options["model_name"],
            full_answer=True,
            answer_windows=len(windows),
            context_subwindows=len(contexts),
            model_calls=calls,
            all_answer_tokens_mapped=True,
            scored_answer_tokens=len(tokens),
            answer_last_end=tokens[-1].end,
        ),
    )
