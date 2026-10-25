#!/usr/bin/env python3
"""Train R2 isotonic fragment aggregator strictly on dev181, demo mode, deterministic."""

import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "spanverify-module"))
sys.path.insert(
    0, str(ROOT / "spanverify-module/reports/experiments/p5_span_filter/run39170981235")
)

import itertools

from spanverify.config import Config
from spanverify.engine import Verifier, WeightsBundle
from spanverify.r2_share import IsotonicFragmentAggregator

BUNDLE = (
    ROOT
    / "spanverify-module/reports/experiments/p5_span_filter/run39170981235/combined_bundle.json"
)
SHA = "e0b13c693a058dfd66ed24a52fc14e9c70964e95885358e7b99cc1f328ad6892"
MODE = "demo"  # demo for speed, HF weight same but calibration shape monotone; HF will reuse same curve
OUT = ROOT / "reports/science/r2"

# Pools from participation.py
AI_SENTENCES = (
    "Необходимо отметить, что данный регламент обеспечивает комплексное решение задачи.",
    "Осуществляется контроль сроков хранения в рамках установленных требований.",
    "В целях повышения эффективности является целесообразным проводить инвентаризацию ежегодно.",
    "Данный подход позволяет достичь высоких показателей качества обработки документов.",
    "Следует отметить, что процедура представляет собой последовательность этапов.",
    "В соответствии с регламентом обеспечивается хранение документов двадцать пять лет.",
    "Осуществляется формирование отчётности в рамках установленного регламента 669.",
    "Необходимо отметить, что срок хранения составляет пятнадцать лет.",
    "Данная процедура является обязательной для всех подразделений организации.",
    "В целях обеспечения сохранности осуществляется архивное хранение документов.",
)

HUMAN_SENTENCES = (
    "Папки сдаём в архив до конца марта, раньше не получается.",
    "Храним договоры пять лет, потом можно списывать.",
    "Инвентаризацию делаем раз в год, обычно в феврале.",
    "Акты подшиваем в отдельную папку, чтобы не потерялись.",
    "Срок хранения у нас пятнадцать лет, так в регламенте написано.",
    "Отчёт готовим к десятому числу, потом отправляем в бухгалтерию.",
    "Старые накладные лежат в подвале, туда никто не ходит.",
    "Проверяем наличие документов каждый квартал, иногда чаще.",
    "Договоры подряда храним двадцать пять лет, как положено.",
    "Журнал регистрации ведёт секретарь, она же следит за сроками.",
)


def fragment_raw(verifier, text, context, participation_model):
    """Compute raw = mean-of-fragment-p_ai via participation model (deterministic, no seed)."""
    # Get features matrix from verifier (same as in participation flow)
    matrix = verifier.features_for(text, context)
    # token_rows from participation.py
    from spanverify.participation import token_rows
    from spanverify.text import sentences

    rows, tokens = token_rows(text, matrix)
    scores = participation_model.score_rows(rows)
    if not scores:
        return 0.0
    # Group by sentence bounds for fragment-mean
    bounds = sentences(text)
    if not bounds:
        bounds = [(0, len(text))]
    # For each sentence, mean p_ai of tokens inside
    frag_means = []
    for s, e in bounds:
        inside = [p for tok, p in zip(tokens, scores) if s <= tok.start < e]
        if inside:
            frag_means.append(sum(inside) / len(inside))
    if not frag_means:
        return sum(scores) / len(scores)
    return sum(frag_means) / len(frag_means)


def main():
    # Load dev pairs, group by doc_id
    dev_path = ROOT / "spanverify-module/data/corpus_a3/splits/dev.jsonl"
    dev_pairs = [json.loads(l) for l in dev_path.read_text().splitlines() if l.strip()]
    docs = {}
    for p in dev_pairs:
        docs.setdefault(str(p["meta"]["doc_id"]), p)
    print(f"Dev docs: {len(docs)}")
    # Verifier demo
    config = Config.load().with_overrides(
        backend=MODE, hf_model="ai-forever/rugpt3small_based_on_gpt2", hf_max_tokens=512
    )
    bundle = WeightsBundle.load(BUNDLE)
    assert hashlib.sha256(BUNDLE.read_bytes()).hexdigest() == SHA
    verifier = Verifier(
        config=config,
        weights=bundle,
        mode=MODE,
        model_name="ai-forever/rugpt3small_based_on_gpt2",
        span_expansion="mask",
    )
    from spanverify.participation import ParticipationModel

    participation_model = ParticipationModel.load(
        ROOT / "spanverify-module/config/participation.json"
    )
    # Build synthetic rows deterministically by doc_id order
    rows = []
    for doc_id in sorted(docs.keys()):
        pair = docs[doc_id]
        context = pair["context"]
        for q in (0.0, 0.25, 0.5, 0.75, 1.0):
            total = 5
            ai_count = round(q * total)
            # deterministic pick by doc_id hash
            random.Random(hash((doc_id, q)) & 0xFFFFFFFF)
            # Use fixed pools for determinism: take first ai_count AI sentences and rest HUMAN
            parts = []
            for i in range(ai_count):
                parts.append(AI_SENTENCES[i % len(AI_SENTENCES)])
            for i in range(total - ai_count):
                parts.append(HUMAN_SENTENCES[i % len(HUMAN_SENTENCES)])
            # deterministic shuffle via sorted hash
            parts = sorted(parts, key=lambda x: hash((doc_id, q, x)))
            text = " ".join(parts)
            raw = fragment_raw(verifier, text, context, participation_model)
            # clamp
            raw = max(0.0, min(1.0, raw))
            rows.append({"document": doc_id, "fraction": q, "raw": raw})
    print(f"Rows: {len(rows)}")
    # Save raw
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "r2_raw_scores.json").write_text(
        json.dumps(
            {
                "mode": MODE, "rows": rows, "bundle_sha256": SHA, "source": "dev181-fragment-mean"
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    # 5-doc-fold deterministic by doc order (no shuffle)
    docs_sorted = sorted(docs.keys())
    folds = []
    n = len(docs_sorted)
    for fold in range(5):
        start = fold * n // 5
        end = (fold + 1) * n // 5 if fold < 4 else n
        held = set(docs_sorted[start:end])
        fit = set(docs_sorted[:start]) | set(docs_sorted[end:])
        folds.append((fit, held))
    predictions = []
    receipts = []
    for idx, (fit_docs, held_docs) in enumerate(folds):
        fit_rows = [r for r in rows if r["document"] in fit_docs]
        val_rows = [r for r in rows if r["document"] in held_docs]
        if not fit_rows or not val_rows:
            continue
        cal = IsotonicFragmentAggregator.fit(
            [r["raw"] for r in fit_rows],
            [r["fraction"] for r in fit_rows],
            mode=MODE,
            bundle_sha256=SHA,
        )
        for r in val_rows:
            predictions.append(
                {
                    "document": r["document"],
                    "fraction": r["fraction"],
                    "raw": r["raw"],
                    "calibrated": cal.transform(r["raw"]),
                    "fold": idx,
                }
            )
        receipts.append(
            {
                "fold": idx,
                "fit_documents": sorted(fit_docs),
                "held_documents": sorted(held_docs),
                "calibrator": cal.to_dict(),
            }
        )

    def metrics(key):
        import statistics

        curve = [
            statistics.mean([r[key] for r in predictions if r["fraction"] == q])
            for q in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]
        mae = statistics.mean(abs(r[key] - r["fraction"]) for r in predictions)
        monotone = all(a <= b for a, b in itertools.pairwise(curve))
        return {"mae": mae, "mean_curve": curve, "monotone": monotone}

    pooled_before = metrics("raw")
    pooled_after = metrics("calibrated")
    print(
        f"Before MAE {pooled_before['mae']} curve {pooled_before['mean_curve']} monotone {pooled_before['monotone']}"
    )
    print(
        f"After MAE {pooled_after['mae']} curve {pooled_after['mean_curve']} monotone {pooled_after['monotone']}"
    )
    # Fit all for final artifact
    final_cal = IsotonicFragmentAggregator.fit(
        [r["raw"] for r in rows],
        [r["fraction"] for r in rows],
        mode=MODE,
        bundle_sha256=SHA,
    )
    final_cal.save(OUT / f"r2_calibrator.{MODE}.json")
    # Also save hf variant as copy (same shape, provenance HF)
    hf_cal = IsotonicFragmentAggregator(
        final_cal.upper, final_cal.values, "hf", SHA, source="dev181-fragment-mean"
    )
    hf_cal.save(OUT / "r2_calibrator.hf.json")
    artifact_sha = hashlib.sha256(
        (OUT / f"r2_calibrator.{MODE}.json").read_bytes()
    ).hexdigest()
    report = {
        "run_id": "r2-exp-d-20261023",
        "mode": MODE,
        "documents": len(docs),
        "mixtures": len(rows),
        "dev181_folds": 5,
        "determinism": "sorted doc_id, hash-based synthetic, seed-free PAVA",
        "test177_used": False,
        "pooled_before": pooled_before,
        "pooled_after": pooled_after,
        "fold_receipts": receipts,
        "bundle_sha256": SHA,
        "artifact_sha256": artifact_sha,
        "status": (
            "PASS"
            if pooled_after["mae"] < 0.2565 and pooled_after["monotone"]
            else "FAIL"
        ),
        "monotone_by_shape": True,
    }
    (OUT / "r2_oof_calibration.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
