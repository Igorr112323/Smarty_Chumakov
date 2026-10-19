"""Original five document-fold OOF calibration, train only; no quality fitting."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p6_acceptance import P5, ROOT, SHA, environment, read_pairs, save, verifier

from spanverify.share_calibration import AiShareCalibrator


def main(mode, out):
    docs = {}
    for pair in read_pairs(ROOT / "data/corpus_a3/splits/train.jsonl"):
        docs.setdefault(str(pair["meta"]["doc_id"]), pair)
    folds = json.loads((P5 / "fit_receipts.json").read_text())
    held = [str(d) for f in folds for d in f["held_documents"]]
    assert len(held) == len(set(held)) and set(held) == set(docs)
    v = verifier(mode)
    rows = []
    machine = (
        "Таким образом, необходимо отметить, что комплексная оптимизация процессов "
        "является ключевым аспектом эффективного функционирования организации. " * 8
    )[:800]
    for doc_id, pair in sorted(docs.items()):
        human = (pair["context"] * (800 // len(pair["context"]) + 1))[:800]
        for q in (0.0, 0.25, 0.5, 0.75, 1.0):
            cut = int(800 * q)
            text = human[: 800 - cut] + machine[:cut]
            raw = v.verify(text, pair["context"]).ai_share
            rows.append(dict(document=doc_id, fraction=q, raw=raw))
        save(out / "raw-train-scores.json", dict(mode=mode, rows=rows, raw_score="frozen ai_share", bundle_sha256=SHA))
    predictions, receipts = [], []
    for fold in folds:
        fit_docs, held_docs = set(map(str, fold["fit_documents"])), set(map(str, fold["held_documents"]))
        assert not fit_docs & held_docs
        fit = [r for r in rows if r["document"] in fit_docs]
        val = [r for r in rows if r["document"] in held_docs]
        cal = AiShareCalibrator.fit([r["raw"] for r in fit], [r["fraction"] for r in fit], mode=mode, bundle_sha256=SHA)
        predictions += [dict(r, fold=fold["fold"], calibrated=cal.transform(r["raw"])) for r in val]
        receipts.append(
            dict(
                fold=fold["fold"],
                fit_documents=sorted(fit_docs),
                held_documents=sorted(held_docs),
                shared_documents=[],
                calibrator=cal.to_dict(),
            )
        )

    def metrics(key):
        curve = [
            statistics.mean(r[key] for r in predictions if r["fraction"] == q) for q in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]
        return dict(
            mae=statistics.mean(abs(r[key] - r["fraction"]) for r in predictions),
            mean_curve=curve,
            monotone=all(a <= b for a, b in zip(curve, curve[1:], strict=False)),
        )

    fit_all = AiShareCalibrator.fit(
        [r["raw"] for r in rows], [r["fraction"] for r in rows], mode=mode, bundle_sha256=SHA
    )
    save(out / f"ai_share_calibrator_v1.{mode}.json", fit_all.to_dict())
    report = dict(
        mode=mode,
        documents=len(docs),
        mixtures=len(rows),
        original_document_folds=True,
        pooled_before=metrics("raw"),
        pooled_after=metrics("calibrated"),
        rows=predictions,
        fold_receipts=receipts,
        environment=environment(),
        status="частично",
        limitation="Synthetic train-only diagnostic, NOT real-LLM authorship calibration; no quality fitting.",
    )
    save(out / "oof-calibration.json", report)
    print(
        "::notice title=P7 AI-share::"
        + json.dumps(
            {k: v for k, v in report.items() if k not in ("rows", "fold_receipts", "environment")}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["demo", "hf"], required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    main(a.mode, a.out)
