# P7 preregistration

Baseline remote 14a018a1b90b51cc47cd8baf628147ac22e1d8b1; bundle 122874 bytes,
e0b13c693a058dfd66ed24a52fc14e9c70964e95885358e7b99cc1f328ad6892.
No model training/reselection, threshold/decoder/span-filter changes or test177 reads.
Existing temporary synthetic/demo unit/CI fit permission explicitly reconfirmed by user in P7.
Before each predictive change: capture golden, then byte equality; actual saved-P5 dev
replay must pass 181 pairs / 13150 token risks / zero mismatch / P1 delta=0.
If a gate fails, revert that predictive change before attempting another.
Long-answer fix must retain the short-answer branch and context top-3 policy.
Calibration only train document-fold held-out predictions using original five document folds;
PAVA on frozen raw ai_share, separate versioned artifact, no dev/test calibration.
Performance uses train/synthetic only; 3 warm/cold trials, same P6 definitions,
latency includes no download/import; timeout is censored failure, never a passing median.
Robustness is studied/documented, not a quality gate. Official test remains reserved.

First attempt rejected by byte gate because original to_dict includes wall-clock latency_ms. Predictive code reverted before continuing. Original golden retained; stable golden removes ONLY latency_ms, never scores/tokens/offsets/texts. Comparison confirms original predictive fields unchanged.

## Blocking audit finding
Real tokenizer audit on 36cfac9: train 22/842 >512 (max1923), dev3/181 >512 (max1046).
The prior short-answer assumption is false. Cached dev replay passes but bypasses
new feature extraction, so is NOT proof of fresh long-path parity. Automatic long
dispatch reverted immediately; no test177 reads. An explicit contract decision
is needed before enabling changed long-answer features on those saved inputs.

## User-authorized amendment
After observed22 train/3 dev >512, user explicitly selected authorize_versioned_long_path.
Long outputs may change only for >512 model positions, version window512_full_answer_v1.
Historical P5 metrics and all quality parameters unchanged.178 short dev rows require
fresh old/new byte parity; cached181 replay is separately labeled cached compatibility.
22 long train rows get feature deltas,3 long dev rows get actual output deltas.
Python3.12 compensated builtin sum is preserved by Python fallback (no false numpy exact claim).

Bounded exact-window memoization: per-request LRU256, key includes complete answer
window/context strings and every local token text/start/end; same immutable eval
model/options. Every logical window/token still participates in the same arithmetic.
Only repeated identical computations are reused. Cache has no train/dev identities.
Actual full public long output must be byte-identical to accepted uncached P7
2502a62, excluding ONLY timing/execution counters. This repeated-train benchmark
is best-case; no claim of universal latency for unrepeated documents.
