"""P8 one-shot frozen public inference. No fit, selection, or model modifications."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from replay_p1_dev import metrics  # noqa: E402

from spanverify import engine  # noqa: E402
from spanverify.config import Config  # noqa: E402
from spanverify.core import tokenize_with_offsets  # noqa: E402
from spanverify.engine import Verifier, WeightsBundle  # noqa: E402
from spanverify.features import CountingFeatureCache, is_scored_token  # noqa: E402
from spanverify.gates import gate_report  # noqa: E402

SHA = "e0b13c693a058dfd66ed24a52fc14e9c70964e95885358e7b99cc1f328ad6892"
BUNDLE = ROOT / "reports/experiments/p5_span_filter/run39170981235/combined_bundle.json"
MODEL = "ai-forever/rugpt3small_based_on_gpt2"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def frozen_metrics(evidence, corpus, answers):
    source = ROOT / "scripts/p4_context_head.py"
    node = next(
        n for n in ast.parse(source.read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef) and n.name == "evidence_metrics"
    )
    scope = dict(
        tokenize_with_offsets=tokenize_with_offsets, is_scored_token=is_scored_token, metrics=metrics, engine=engine
    )
    for name in ("_token_metrics", "_span_f1", "_containment_recall", "_mean_width_ratio"):
        scope[name] = getattr(engine, name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    result = scope["evidence_metrics"](evidence, corpus, answers, None)
    result["answers"]["auc"] = result["answers"]["calibrated_auc"]
    return result


def summarise(evidence, corpus, answers):
    result = frozen_metrics(evidence, corpus, answers)
    gates = {
        k: gate_report(v if k != "spans" else dict(v, fpr_tokens=result["tokens"]["fpr"]), k)
        for k, v in result.items()
        if k in ("tokens", "spans", "answers")
    }
    return dict(
        metrics=result,
        gates=gates,
        joint_gate=bool(
            result["tokens"]["f1"] >= 0.5
            and result["tokens"]["fpr"] <= 0.4
            and result["spans"]["f1"] >= 0.5
            and result["tokens"]["fpr"] <= 0.4
        ),
        span_fpr_convention="unchanged P5 token-FPR proxy, no invented span TN",
    )


def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    assert len(BUNDLE.read_bytes()) == 122874 and hashlib.sha256(BUNDLE.read_bytes()).hexdigest() == SHA
    assert not os.getenv("SPANVERIFY_AI_SHARE_CALIBRATOR")
    dataset = ROOT / (
        "data/corpus_a3/splits/test.jsonl" if args.kind == "official" else "data/external/rushallu/pairs.jsonl"
    )
    if args.kind == "aggregate":
        dataset = ROOT / "data/external/rushallu/pairs.jsonl"
    if args.kind == "aggregate":
        pairs = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
        records = []
        for file in Path(args.inputs).rglob("evidence.jsonl"):
            records.extend(json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip())
        assert len(records) == 1000 and len({r["id"] for r in records}) == 1000
        corpus = {p["id"]: p for p in pairs}
        report = summarise([r["evidence"] for r in records], corpus, [r["answer_metrics"] for r in records])
        report.update(
            kind="external validation, NOT contractual gate",
            pairs=1000,
            annotation_granularity="human exact spans:423 original spans/333positive answers; original adapter first occurrence for one ambiguity; all families supported, no fabricated spans",
            dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
            run_id=os.environ["GITHUB_RUN_ID"],
            commit=os.environ["GITHUB_SHA"],
        )
        save(out / "result.json", report)
        print("::notice title=P8 B result::" + json.dumps(report))
        return
    cache = CountingFeatureCache()
    config = Config.load()
    config = config.with_overrides(backend="hf", hf_model=MODEL, hf_max_tokens=512)
    verifier = Verifier(
        config=config,
        weights=WeightsBundle.load(BUNDLE),
        mode="hf",
        model_name=MODEL,
        features_cache=cache,
        span_expansion="mask",
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    # Audit selects metadata fields only; annotation event values are never retained/inspected.
    import ijson

    audit_rows = []
    current = {}
    with dataset.open("rb") as stream:
        for prefix, event, value in ijson.parse(stream, multiple_values=True):
            if prefix in ("id", "answer", "context") and event == "string":
                current[prefix] = value
            if prefix == "" and event == "end_map":
                audit_rows.append(
                    dict(id=current["id"], model_positions=len(tokenizer(current["answer"])["input_ids"]))
                )
                current = {}
    expected = 177 if args.kind == "official" else 1000
    assert len(audit_rows) == expected and len({p["id"] for p in audit_rows}) == expected
    audit = dict(
        pairs=expected,
        over512=sum(r["model_positions"] > 512 for r in audit_rows),
        max_positions=max(r["model_positions"] for r in audit_rows),
        labels_inspected=False,
        contract="window512_full_answer_v1",
        rows=audit_rows,
    )
    save(out / "pre-label-long-audit.json", audit)
    print("::notice title=P8 long audit::" + json.dumps({k: v for k, v in audit.items() if k != "rows"}), flush=True)
    pairs = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.kind == "external":
        pairs = [p for i, p in enumerate(pairs) if i % 4 == args.shard]
    fit_calls = []

    def no_fit(frame, event, arg):
        if event == "call" and frame.f_code.co_name in ("fit", "fit_transform", "train"):
            if (
                frame.f_code.co_name == "train"
                and frame.f_globals.get("__name__", "").startswith(("torch.", "transformers."))
                and frame.f_locals.get("mode") is False
            ):
                return
            fit_calls.append(dict(file=frame.f_code.co_filename, function=frame.f_code.co_name))
            raise RuntimeError("P8 forbidden fit/train call")

    sys.setprofile(no_fit)
    evidence = []
    answers = []
    started = time.perf_counter()
    file = out / "evidence.jsonl"
    if file.exists():
        raise RuntimeError("Refuse overwriting a consumed inference stream")
    with file.open("x") as handle:
        for i, p in enumerate(pairs):
            tick = time.perf_counter()
            result = verifier.verify(p["answer"], p["context"], with_tokens=True)
            elapsed = time.perf_counter() - tick
            public = result.to_dict(with_tokens=True)
            ev = dict(id=p["id"], spans=[asdict(s) for s in result.spans], tokens=result.tokens, verdict=result.verdict)
            answer = dict(
                pair_id=p["id"],
                gold=int(any(int(x[2]) == 1 for x in p["labels"])),
                prediction=int(result.score >= result.threshold),
                raw_score=result.stats["raw_score_full"],
                calibrated_score=result.score,
                threshold=result.threshold,
                verdict=result.verdict,
            )
            record = dict(
                id=p["id"],
                evidence=ev,
                answer_metrics=answer,
                public=public,
                inference_seconds=elapsed,
                public_sha256=hashlib.sha256(
                    json.dumps(public, ensure_ascii=False, sort_keys=True).encode()
                ).hexdigest(),
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            evidence.append(ev)
            answers.append(answer)
            print(f'PAIR {i+1}/{len(pairs)} {p["id"]} {elapsed:.6f}s', flush=True)
    sys.setprofile(None)
    report = summarise(evidence, {p["id"]: p for p in pairs}, answers)
    report.update(
        kind=args.kind,
        pairs=len(pairs),
        shard=args.shard,
        cache_hits=cache.hits,
        cache_misses=cache.misses,
        cache_policy="empty P8 feature cache; all product fresh inference, no stale legacy512 cache",
        fit_calls=fit_calls,
        seconds=time.perf_counter() - started,
        bundle_bytes=122874,
        bundle_sha256=SHA,
        dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        evidence_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
        long_over512=audit["over512"],
        run_id=os.environ["GITHUB_RUN_ID"],
        commit=os.environ["GITHUB_SHA"],
        product_source="b9cf9acdab1ca7bf082d03629ea7e599597f3fa8",
    )
    save(out / "result.json", report)
    print("::notice title=P8 result::" + json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["official", "external", "aggregate"], required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--inputs")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    try:
        main(args)
    except BaseException:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "INFRASTRUCTURE-TRACEBACK.txt").write_text(traceback.format_exc())
        print("::error title=P8 infrastructure::" + traceback.format_exc().replace("\n", "%0A"))
        raise
