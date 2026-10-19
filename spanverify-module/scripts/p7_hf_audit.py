"""Actual HF short-path golden and train/dev truncation audit, never test177."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p6_acceptance import MODEL, P5, ROOT, environment, read_pairs, save

from spanverify import features


def main(out):
    tokenizer, _model = features._load_hf_runtime(MODEL, "cpu")
    audit = {}
    for split in ("train", "dev"):
        pairs = list(read_pairs(ROOT / f"data/corpus_a3/splits/{split}.jsonl"))
        counts = [len(tokenizer(p["answer"], add_special_tokens=False)["input_ids"]) for p in pairs]
        from spanverify.chunking import chunk_document, estimate_tokens, select_relevant_chunks
        from spanverify.core import tokenize_with_offsets

        coverage_rows = []
        for pair, n in zip(pairs, counts, strict=True):
            answer, context = pair["answer"], pair["context"]
            selected = [context]
            if context and estimate_tokens(context) > 380:
                chunks = chunk_document(context, max_tokens_per_chunk=380)
                if len(chunks.chunks) > 1:
                    selected = [chunks.chunks[i].text for i in (select_relevant_chunks(answer, chunks, top_k=3) or [0])]
            tokens = tokenize_with_offsets(answer)
            covered = [False] * len(tokens)
            lengths = []
            for ctx in selected:
                prompt = f"{ctx}\n{answer}" if ctx else answer
                shift = len(prompt) - len(answer)
                lengths.append(len(tokenizer(prompt)["input_ids"]))
                offsets = tokenizer(prompt, return_offsets_mapping=True, truncation=True, max_length=512)[
                    "offset_mapping"
                ]
                spans = [(i, a, b) for i, (a, b) in enumerate(offsets) if b > a and a >= shift and i < 512]
                positions = features.map_token_positions([(t.start, t.end) for t in tokens], spans, shift)
                covered = [x or bool(y) for x, y in zip(covered, positions, strict=True)]
            missing = [t.start for t, ok in zip(tokens, covered, strict=True) if not ok]
            coverage_rows.append(
                dict(
                    id=pair["id"],
                    answer_model_tokens=n,
                    native_tokens=len(tokens),
                    selected_contexts=len(selected),
                    max_prompt_model_tokens=max(lengths, default=0),
                    legacy_unmapped_native_tokens=len(missing),
                    first_unmapped_start=missing[0] if missing else None,
                )
            )
        save(out / f"legacy-coverage-{split}.json", coverage_rows)
        audit[split] = dict(
            answers=len(counts),
            answer_over_512=sum(n > 512 for n in counts),
            max_answer_model_tokens=max(counts),
            legacy_rows_with_unmapped_native_tokens=sum(r["legacy_unmapped_native_tokens"] > 0 for r in coverage_rows),
            legacy_short_rows_with_unmapped_native_tokens=sum(
                r["legacy_unmapped_native_tokens"] > 0 and r["answer_model_tokens"] <= 512 for r in coverage_rows
            ),
            legacy_unmapped_native_tokens=sum(r["legacy_unmapped_native_tokens"] for r in coverage_rows),
            answer_input_sha256=__import__("hashlib")
            .sha256(json.dumps([(p["id"], p["answer"]) for p in pairs], ensure_ascii=False).encode())
            .hexdigest(),
        )
    save(
        out / "truncation-audit.json",
        dict(
            splits=audit,
            official_test_read=False,
            long_contract="window512_full_answer_v1; user authorized >512 only",
            environment=environment(),
        ),
    )
    print("::notice title=P7 truncation audit::" + json.dumps(audit))
    assert audit["train"]["answer_over_512"] == 22 and audit["dev"]["answer_over_512"] == 3
    source = subprocess.check_output(["git", "show", "14a018a:spanverify-module/spanverify/features.py"], text=True)
    old = types.ModuleType("spanverify._p7_legacy_features")
    old.__package__ = "spanverify"
    sys.modules[old.__name__] = old
    exec(compile(source, "<frozen-P6-features>", "exec"), old.__dict__)
    rows = []
    train = list(read_pairs(ROOT / "data/corpus_a3/splits/train.jsonl"))
    for pair in train[:3]:
        before = dataclasses.asdict(old.hf_features(pair["answer"], pair["context"], model_name=MODEL))
        after = dataclasses.asdict(features.hf_features(pair["answer"], pair["context"], model_name=MODEL))
        assert json.dumps(before, sort_keys=True) == json.dumps(after, sort_keys=True), pair["id"]
        rows.append(dict(id=pair["id"], before=before, after=after))
    save(out / "hf-short-golden.json", dict(passed=True, rows=rows, environment=environment()))
    save(
        out / "truncation-audit.json",
        dict(
            splits=audit,
            policy="answer-only model tokens; short path unchanged",
            official_test_read=False,
            environment=environment(),
        ),
    )
    import hashlib
    import os
    from unittest.mock import patch

    from p6_acceptance import verifier

    if os.getenv("P7_FULL_DEV_REPLAY") == "1":
        short, changed = [], []
        reference = {r["pair_id"]: r for r in json.loads((P5 / "dev_answers.json").read_text())}
        legacy, current = verifier("hf"), verifier("hf")
        legacy._features = lambda answer, context, tokens: old.hf_features_chunked(
            answer, context, model_name=MODEL, answer_tokens=tokens, max_length=512
        )
        for split in ("dev", "train"):
            for pair in read_pairs(ROOT / f"data/corpus_a3/splits/{split}.jsonl"):
                length = len(tokenizer(pair["answer"], add_special_tokens=False)["input_ids"])
                if split == "train" and length <= 512:
                    continue
                # Reuse the identical eval weights in the baseline replay only;
                # this is not a timing trial and does not alter its arithmetic.
                with (
                    patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
                    patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=_model),
                ):
                    if split == "dev":
                        before_result = legacy.verify(pair["answer"], pair["context"], with_tokens=True)
                        after_result = current.verify(pair["answer"], pair["context"], with_tokens=True)
                        if length <= 512:
                            assert before_result.score == after_result.score
                            assert before_result.threshold == after_result.threshold
                        before = before_result.to_dict(with_tokens=True)
                        after = after_result.to_dict(with_tokens=True)
                        before.pop("latency_ms", None)
                        after.pop("latency_ms", None)
                        a = json.dumps(before, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
                        b = json.dumps(after, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
                        if length <= 512:
                            assert a == b, pair["id"]
                            short.append(dict(id=pair["id"], sha256=hashlib.sha256(a).hexdigest(), equal=True))
                        else:
                            changed.append(
                                dict(
                                    split=split,
                                    id=pair["id"],
                                    model_tokens=length,
                                    before_sha256=hashlib.sha256(a).hexdigest(),
                                    after_sha256=hashlib.sha256(b).hexdigest(),
                                    legacy_score=before_result.score,
                                    new_score=after_result.score,
                                    saved_P5_score=reference[pair["id"]]["calibrated_score"],
                                    changed_token_records=sum(
                                        x != y for x, y in zip(before["tokens"], after["tokens"], strict=True)
                                    ),
                                )
                            )
                    else:
                        before = old.hf_features_chunked(pair["answer"], pair["context"], model_name=MODEL)
                        after = features.hf_features_chunked(pair["answer"], pair["context"], model_name=MODEL)
                        deltas = {
                            name: max(
                                (abs(a - b) for a, b in zip(getattr(before, name), getattr(after, name), strict=True)),
                                default=0.0,
                            )
                            for name in features.FEATURE_NAMES
                        }
                        changed.append(dict(split=split, id=pair["id"], model_tokens=length, max_feature_deltas=deltas))
                save(
                    out / "fresh-feature-contract.json",
                    dict(
                        short_dev=short,
                        authorized_long_rows=changed,
                        contract="window512_full_answer_v1",
                        historical_metrics_overwritten=False,
                        quality_fitting=False,
                    ),
                )
        assert (
            len(short) == 178
            and sum(r["split"] == "dev" for r in changed) == 3
            and sum(r["split"] == "train" for r in changed) == 22
        )
        print(
            "::notice title=P7 fresh contract::"
            + json.dumps(
                dict(
                    short_dev_pairs=len(short),
                    short_byte_mismatches=0,
                    changed_dev_long_rows=3,
                    train_long_feature_rows=22,
                    contract="window512_full_answer_v1",
                )
            )
        )

        print("::notice title=P7 long dev deltas::" + json.dumps([r for r in changed if r["split"] == "dev"]))

    # Cache optimization golden against the previously accepted uncached P7
    # window contract, full public tokens/risks/texts/offsets included.
    previous_source = subprocess.check_output(
        ["git", "show", "2502a62:spanverify-module/spanverify/hf_windows.py"], text=True
    )
    previous = types.ModuleType("spanverify._p7_uncached_windows")
    previous.__package__ = "spanverify"
    sys.modules[previous.__name__] = previous
    exec(compile(previous_source, "<accepted-uncached-P7-window-contract>", "exec"), previous.__dict__)
    from spanverify import hf_windows

    long_text = "Срок хранения первичных документов десять лет. " * 120
    context = "Срок хранения первичных документов составляет 10 лет."
    with patch.object(hf_windows, "full_answer_features", previous.full_answer_features):
        baseline = verifier("hf").verify(long_text, context, with_tokens=True).to_dict(with_tokens=True)
    optimized = verifier("hf").verify(long_text, context, with_tokens=True).to_dict(with_tokens=True)
    for result in (baseline, optimized):
        result.pop("latency_ms", None)
        for key in ("model_calls", "identical_window_cache_hits", "logical_window_scores"):
            result["stats"]["features"].pop(key, None)
    left = json.dumps(baseline, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    right = json.dumps(optimized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    assert left == right, "Identical-window cache changed predictive output"
    save(
        out / "long-cache-golden.json",
        dict(
            passed=True,
            bytes=len(left),
            sha256=hashlib.sha256(left).hexdigest(),
            excluded_nonpredictive_fields=[
                "latency_ms",
                "model_calls",
                "identical_window_cache_hits",
                "logical_window_scores",
            ],
        ),
    )
    print(
        "::notice title=P7 exact long cache::" + json.dumps(dict(passed=True, sha256=hashlib.sha256(left).hexdigest()))
    )

    # Actual >512 model positions: every native token must map, no surrogate fallback.
    text = "Срок хранения первичных документов десять лет. " * 120
    matrix = features.hf_features(text, "", model_name=MODEL)
    assert matrix.meta["full_answer"] and matrix.meta["all_answer_tokens_mapped"]
    save(
        out / "hf-long-answer.json",
        dict(chars=len(text), meta=matrix.meta, token_features=dataclasses.asdict(matrix), environment=environment()),
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
