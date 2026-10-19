"""Actual HF short-path golden and train/dev truncation audit, never test177."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p6_acceptance import MODEL, ROOT, environment, read_pairs, save

from spanverify import features


def main(out):
    tokenizer, _model = features._load_hf_runtime(MODEL, "cpu")
    audit = {}
    for split in ("train", "dev"):
        pairs = list(read_pairs(ROOT / f"data/corpus_a3/splits/{split}.jsonl"))
        counts = [len(tokenizer(p["answer"], add_special_tokens=False)["input_ids"]) for p in pairs]
        audit[split] = dict(
            answers=len(counts),
            answer_over_512=sum(n > 512 for n in counts),
            max_answer_model_tokens=max(counts),
            answer_input_sha256=__import__("hashlib")
            .sha256(json.dumps([(p["id"], p["answer"]) for p in pairs], ensure_ascii=False).encode())
            .hexdigest(),
        )
    save(
        out / "truncation-audit.json",
        dict(
            splits=audit,
            official_test_read=False,
            blocked="Saved OOF/dev contain long answers; full-answer semantics need explicit authorization",
            environment=environment(),
        ),
    )
    print("::notice title=P7 truncation audit::" + json.dumps(audit))
    assert (
        audit["train"]["answer_over_512"] == 0 and audit["dev"]["answer_over_512"] == 0
    ), "Long-answer change would affect saved OOF/dev states; reject it"
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
