# SpanVerify frozen runtime 1.3.1 — P8 final evaluation

Verification-envelope tag: p8-final-20261019; no runtime version bump or predictive changes. Stable/latest publication is authorized ONLY by the actual joint official177 gate PASS (token and strict span F1≥0.50, token-FPR proxy≤0.40). This is NOT a claim all contract requirements passed.

## Official177, one successful inference
Run39266451529, commit dd4454b (full SHA in source reports), exact P7 product b9cf9ac. Tokens TP/FP/FN/TN191/139/90/8848, P/R/F1 .578788/.679715/.625205, FPR .015467, AUC .971717. Strict spans47/38/50, F1 .516484, containment .288660, mean contained width1.339262; greedy IoU≥.5, token-FPR proxy (no invented span TN). Answers51/15/46/65, F1 .625767, FPR .187500, raw/calibrated AUC .697423/.684407. Verdicts62/19/35/61, F1 .6966, FPR .2375. Internal .90/.10 NOT achieved.

177 metadata audited before labels:7 >512/max1266, uniform window512_full_answer_v1. Empty pair feature cache0hits177misses;381.292202784s fresh public inference, no fit. Bundle122874B, SHA256e0b13c693a058dfd66ed24a52fc14e9c70964e95885358e7b99cc1f328ad6892; tau .5794392523364486 unchanged. An infrastructure guard falsely rejected transformers.train(False) before first prediction: original run39266027598 traceback and ZERO evidence retained; a single verified empty-output recovery, never bad-metric retry.

## External B1000, not contractual gate
Pinned RuHallu-RAG human corpus:1000 pairs,423 exact spans,333positive answers; one first-occurrence ambiguity, raw license unconfirmed. No fit/calibration/threshold change/comparator. Tokens152/510/8867/18559, F1 .031402/FPR .026745/AUC .437462. Strict spans6/182/417,F1 .019640,containment0. Answers203/271/130/396,F1 .503098,FPR .406297,raw/calibrated AUC .621505/.616485. Verdicts221/319/112/348,F1 .5063,FPR .4783. Severe transfer drop is published, not hidden or fixed after test. All1000 fresh feature misses,0hits,0 >512; four disjoint250 shards, aggregate reuses evidence only.

## Mandatory limitations
- HF50k warm ≤1s NOT achieved: measured median8.168092351s, compute-bound frozen model. No model changes made to conceal it.
- HF2M full-answer warm median38.769417s, ≤60s achieved on measured repeated fixture; not arbitrary-text SLA.
- Separately delivered HF model551290714B and libraries870588148B, PLUS portable Python/application; full installed SDK total not measured. NOT the earlier ~500MB library estimate, NOT HF weights inside9MB exe. Exact new-build footprint and SHA256 receipts take precedence if packaging differs.
- Small exe/pyz default to demo. Validated HF requires supplied frozen-p5.json/p5-runtime.json and external offline model/libraries; actual Windows offline tests cover exe and pyz. Strict limits remain exe≤9000000B and pyz≤600000B.
- AI-share PARTIAL: train-only OOF HF MAE .2565365102; pooled q-curves nonmonotone; synthetic character fraction diagnostic is NOT validated LLM-authorship; opt-in artifact, default unchanged.
- Historical official-file read incident disclosed: unsplit corpus parsed then first20 in chunking fixture, exact time unknown. Guardfix5ff79f9a7bed33c00c6278f67d68ddff73f8325c2026,18Oct23:34:59Z, train-only; reserved177 denied throughout ordinary tests. Source pairs SHA b9a4cf05a9078644dd539470481a1b8079aed2d9201e195c3ecd901276ff7c8e.
- Robustness/qualification/reference-machine and complete SDK storage limitations remain in current matrix; overall contract NOT fully achieved.

## Evidence, checks, delivery
See source reports/FINAL_REPORT.md, reports/EXPERIMENTS_INDEX.md and reports/contract/coverage-matrix.json (only two current indices). Historical legacy aggregates explicitly SUPERSEDED; immutable historical numbers/P2–P5 canonical runs preserved. Provenance/legacy-number/frozen-product regression tests added. Full local909 collected/907passed/2pre-existing skips,415.263s; tests_excluded=false. Final ordinary CI runs and fresh Windows smoke/size/offline receipts linked from reports/p8/DELIVERY.json. SHA256SUMS.txt covers delivery. Final acceptance archive contains current source, new binary and original official/B per-pair JSONL; a later publication receipt cannot recursively appear inside its own archive.
