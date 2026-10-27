# REALGEN-R1 official test pre-deployment review

**Status: READY_FOR_ONE_OFFICIAL_RUN — waiting only for the curator's shift code.**
`REALGEN_TEST_PREDEPLOY.json` now contains a complete frozen full-dev inference
bundle, its real Actions provenance and the fail-closed shift gate. No official
`test_realgen` evaluation has been dispatched, no official test metric exists,
and `official_run_authorized` remains `false` until the curator sends
`FIRE-REALGEN-1` in a separate message. Do not run the workflow before that.

## Sealed-data boundary

Preparation, training and ordinary CI never parsed the 100 sealed
`REALGEN_ANSWERS_TEST_REALGEN.jsonl` answers, test labels/spans, or test-specific
answer-context rows. Their SHA256 values in the preregistration come from the
existing freeze manifest. The permanently locked `real_test_pairs.jsonl` (1200)
was not read in any role. Integrity checks were digests only:

| Sealed file | SHA256 | Git blob at `dc907325` and at this head |
| --- | --- | --- |
| `REALGEN_ANSWERS_TEST_REALGEN.jsonl` (100) | `8aac288649bdb3cc6715f0918c0b83928f4462ccdbaf24ebf591c2c3d12cf111` | `a449e25dc62e67229df523b0be5509c23c94eeb3` (unchanged) |
| `real_test_pairs.jsonl` (1200) | `f40c737cf20db6730666620174982de5a7be1bda2a4f7fb359e376dc18324420` | `6cbbc3ba97f2853c92cc00442c0a00127a40dbb2` (unchanged) |

The dev-only fit used `REALGEN_ANSWERS_DEV.jsonl` (300), the 300 dev rows of
`REALGEN_DOCS.jsonl` and dev labels from the already published
`REALGEN_PHASE_B_OOF.jsonl` (51,472 tokens). `realgen_freeze_bundle.py`
installs a `Path.open` guard that refuses the sealed answers, all labels/spans,
test-only context rows and the 1200-pair file; `REALGEN_SPANS.jsonl` was not
needed at all. `trained_on` is asserted to be `realgen_dev_300_only`, and the
official evaluator re-checks that field before it opens anything.

## Frozen full-dev bundle (trained in CI, dev-300 only)

Verified Actions run [40408250130](https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/40408250130),
head `f62291ef462b4710f44a23ab5335c7e20ddcb47c`, workflow
`.github/workflows/realgen-freeze-bundle.yml`, job/step timeouts 40/38/20/5 min,
runtime 453.181 s. Artifact `realgen-test-model-bundle`
(id `12691713845`, 107,877,533 bytes, 90 days retention) contains
`REALGEN_TEST_MODEL_BUNDLE.zip` and its manifest. The ~108 MB zip is **not** a
git blob: the official workflow downloads it with `gh run download` inside
Actions. Extraction used the approved
`GITHUB_STEP_SUMMARY → check-runs output.text` bridge
(`REALGEN-FREEZE/manifest/meta` + `chunk-0000`, base64(gzip(JSON, mtime=0))),
followed by a hand commit of the 11,789-byte
`REALGEN_TEST_MODEL_BUNDLE_MANIFEST.json` and an `extraction` entry in
`RUN_ID_AUDIT.json` (seq 13/15). No workflow auto-commit, no blob logs.

| Item | Value |
| --- | --- |
| Bundle ZIP SHA256 | `9fbac0e6554fc597763d0ee989b66826068721a7fd7bf3eb52df77a8d5e17f87` |
| Bundle ZIP size | 107.840 MB (107,840,377 bytes), 6 files, 118.834 MB uncompressed |
| FT weights | `rubert_ft/model.safetensors`, SHA256 `09d6fe5ede551cbd4fdf8a30e5014c4f8f1528bf9eb07937f13d5a227d14cca5`, 116.393 MB (116,393,320 bytes) |
| Dev training data | 300 answers / 300 docs / 51,472 tokens / 1,828 positive tokens |
| FT protocol | `cointegrated/rubert-tiny2`, batch 8, lr 2e-5, 3 epochs, max_length 96, seed 42, `random.Random(42)` shuffle, 1,829 windows, 43,807 subtokens, 687 optimizer steps |
| Upstream pins | `ai-forever/rugpt3small_based_on_gpt2@a9307e696cd3c5b7f953ff4cb19d76a4d81821d5`; `cointegrated/rubert-tiny2@e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae` |
| Stack | Python 3.11.17, torch 2.14.1+cpu, transformers 5.20.1, numpy 2.4.6 |

Three systems are frozen with **no test-based fit, calibration or threshold
tuning**:

* **composite** — `StandardScaler` + `LogisticRegression(max_iter=1000,
  class_weight="balanced", random_state=42)` on the fixed 27-feature order
  (old5 + E1 + R2). All 27 coefficients, 27 scaler means, 27 scaler scales and
  the intercept are written into `frozen_bundle.models.composite`.
* **r2_only** — the same estimator on exactly the 16 deterministic R2 features
  (E1 and old5 excluded); coefficients/scaler/intercept in
  `frozen_bundle.models.r2_only`.
* **rubert_tiny2_ft** — one full-dev fine-tune; weights SHA and MB above; the
  bundle also contains `config.json`, `tokenizer.json`, `tokenizer_config.json`
  and the reviewed `realgen_test_predict.py` backend.

The selected token threshold is the preregistered fixed **0.5** for all three
systems (`policies.token_threshold`); it is recorded in the bundle and
re-checked by the predictor and the evaluator. No dev/test threshold scan was
applied. Inference recomputes old5/E1 features with the pinned upstream
revisions and applies the frozen numbers; the bundle's own SHA256 is checked
before extraction.

## Frozen protocol (unchanged)

Threshold 0.5 for tokens and answers. Subtoken probabilities are averaged into
word-token probabilities; thresholded token intervals separated by at most one
character are merged into character spans; span hits are one-to-one greedy
matches at IoU ≥ 0.5. Answer risk is the maximum word-token score. Brier and
10-bin equal-width ECE are reported at token and answer levels; FPR uses the
fixed 0.5 threshold; confidence intervals use 5,000 answer/document-cluster
bootstrap resamples with seed `20261026` and percentile 95% limits. All three
systems are reported separately; the R2-only baseline excludes E1 and old5.

## Fail-closed shift gate

`realgen_test_official.py` now requires `REALGEN_OFFICIAL_SHIFT=1` **before any
file read**. Without it (and without the exact `REALGEN_SHIFT_CODE`) it raises
`ShiftRefusal` and exits **3**; the sealed answers are never opened. The
official workflow sets the flag only in its single scoring step, after the
`FIRE-REALGEN-1` workflow-level gate, the one-shot no-prior-attempt guard, the
READY preregistration check and the bundle SHA/provenance check. The workflow
has no push/PR/schedule trigger, no auto-commit, and a 40-minute timeout.

`spanverify-module/tests/test_realgen_official_shift.py` proves the behaviour in
ordinary CI: (1) exit code 3 without the env flag, (2) zero `Path.open` calls on
any sealed path, (3) an explicit refusal naming `REALGEN_OFFICIAL_SHIFT`, plus
static ordering of the gate before metadata/sealed reads and pinning of the
dev-only `frozen_bundle`. The suite is now 980 collected nodes in 78 files
(972 + 5 parametrized workflow-limit nodes + 3 new tests); no matrix branch was
removed.

## CI evidence

* Training/freeze: [40408250130](https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/40408250130) — success, all steps including the zip self-check and artifact upload.
* Suite on the code-bearing head `f62291ef…`: [40408249857](https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/40408249857) — success: Ubuntu/Windows push matrix, lint/format, ≥85% coverage, contract, numbers check and the read-only P8 seal.
* Main post-merge baseline [40354903059](https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/40354903059) and the green PR #25 head [40398753669](https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/40398753669) remain the historical references.

`reports/CI_RUNS.json` is a frozen snapshot from 2026-10-21, so
`docs/ЗАДАНИЯ_CI.md` still shows "прогонов нет" for the two newest workflows;
the real run IDs and their `extraction` fields are in `RUN_ID_AUDIT.json`
(seq 13–15) rather than invented registry rows.

## Limitations to carry into interpretation

- REALGEN labels are rule-generated; lexical feature/label circularity can inflate apparent performance.
- A single generator and 100 held-out answers do not establish transfer to free-form or external generation.
- The earlier Phase B OOF metrics are development evidence only, not official test results; the frozen estimators were fitted on all 300 dev answers.
- The permanent `real_test` 1200 split remains out of scope, unread and unused; the 100 `test_realgen` answers stay sealed until the curator's shift code.
