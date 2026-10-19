# P8 stable — closure of accepted code contour

Acceptance HEAD **9be619031a6cd0d47e826e7ea63208d87f42315e**. Stable envelope **p8-stable-20261019**; frozen runtime remains1.3.1. NO rebuild, scoring, training or recalibration: all8 assets copied byte-for-byte from RC **v1.3.1-rc.p8.20261019**, with file SHA256/digest equality before/after transfer. The original archive's embedded source is879cd14 (pre-publication documentation); acceptance9be6190 adds receipts/provenance only. The archive is deliberately NOT repackaged.

## Valid numerical acceptance
Official test177, [run39266451529](https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/39266451529), **joint gate PASS** (ТЗ4.1.6): tokens F1 **0.625205**, FPR **0.015467**; strict spans F1 **0.516484**; answers F1 **0.625767**, FPR **0.1875**. Frozen bundle122874B, SHA256 **e0b13c693a058dfd66ed24a52fc14e9c70964e95885358e7b99cc1f328ad6892**. All predictive files unchanged.

## Process exception explicitly accepted, NOT erased
Curator accepted the numerical result with the P8 exception: cancelled CI39266027509 performed a temporary synthetic ParticipationModel.fit before seal, in another job/process/machine. Test/B were not involved, bundle bytes/hash unchanged, inference workers fit_calls=[]; no influence on the measured product. The prohibition was violated historically; this is not relabelled as zero global fits. Disclosure complete; always-contract barrier fixed with regression. [Incident](https://github.com/Igorr112323/Smarty_Chumakov/blob/arena/c5f69767-smarty-chumakov/reports/p8/infra-freeze-barrier/incident.json), [curator decision](https://github.com/Igorr112323/Smarty_Chumakov/blob/arena/c5f69767-smarty-chumakov/reports/p8/closure/CURATOR_DECISION.json).

## Limitations remain unchanged
- HF50k warm **8.168092351s** vs **≤1s: NOT achieved**, compute-bound frozen model. HF1k warm0.763317488s vs≤0.03s: NOT achieved. HF2M warm38.769417043s≤60s on measured fixture, not arbitrary-text SLA.
- AI-share PARTIAL: OOF HF MAE **0.2565**, nonmonotone pooled curves; synthetic mixture is not real LLM-authorship validation.
- Libraries **870.6MB**, model directory **554.3MB** (weight file551.3MB); directory+libs1424865456B PLUS Python/app. Full installed SDK total unknown. Compressed637048763B is not installed footprint.
- Corpus B transfer limitation: answers F1 **0.503**, FPR **0.406**, token F1 **0.031** (span F1 .019640); **NOT a gate**.1000 human pairs,423 exact spans,333 positive answers; one first-match ambiguity/license unconfirmed. No comparator selection/tuning.
- Historical unsplit-test read→first20 remains disclosed; exact read instant unknown, train-only guard5ff79f9a7bed33c00c6278f67d68ddff73f8325c.

EXE **8981233B**, PYZ **232874B**, both within strict9000000/600000B limits. Offline HF exe/pyz HTTP200 evidence retained; known product registry keys unchanged, not a global OS trace. **SHA256SUMS.txt** included unchanged.

Archive **p8-final-acceptance.zip**,65248935B, SHA256:
`58143baac4a39d86aeaa46879c497d5195f0202f58174c91af746d94dfe2f158`.

[Expert passport](https://github.com/Igorr112323/Smarty_Chumakov/blob/arena/c5f69767-smarty-chumakov/reports/ACCEPTANCE_PASSPORT.md) · [Deviations](https://github.com/Igorr112323/Smarty_Chumakov/blob/arena/c5f69767-smarty-chumakov/reports/DEVIATIONS.md) · [Read-only verification](https://github.com/Igorr112323/Smarty_Chumakov/blob/arena/c5f69767-smarty-chumakov/reports/p8/VERIFICATION.json). These notes accept disclosed limits, not all-contract compliance. Any subsequent predictive/code change requires a new stage/protocol.
