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

Bounded exact-window memoization: per-request LRU4096, key includes complete answer
window/context strings and every local token text/start/end; same immutable eval
model/options. Every logical window/token still participates in the same arithmetic.
Only repeated identical computations are reused. Cache has no train/dev identities.
Actual full public long output must be byte-identical to accepted uncached P7
2502a62, excluding ONLY timing/execution counters. This repeated-train benchmark
is best-case; no claim of universal latency for unrepeated documents.

The capacity is a memory/performance bound, not a quality parameter.256 can thrash
on repeated train windows when the exact BPE phase/context key cycle exceeds256;
4096 bounds retained matrices to roughly200MiB, with no change to any arithmetic.

Windows22ae68c reached offline /health but /v1/verify returned HTTP500.
No autonomy claim is made from a successful health response. The smoke remains
mandatory. The next build preserves the server JSON error body, and declares
setuptools in the isolated external runtime (it is no longer inherited from the
build interpreter's removed development site-packages).

Tokenizer-only audit also reconstructs exact legacy selected top3 contexts and
native-token coverage after combined context+answer512 truncation, not merely
answer-alone length. Any <=512-answer gaps are disclosed and remain on the
byte-frozen short path: authorization permits changes only >512. No scope is
silently expanded to repair those legacy short-path outputs.

Pre-index full HF 2M: actual uncached1738.99458161s vs identical-window cache
125.521545277s; both282830 token records, end2000000, SHA56c3028861b5ea021310a59f98272fa1bff6e33a43755af6fae0bc00f89655a4.
These are single functional observations, not three-repeat speed acceptance.
After removing3283->14 duplicate model computations, two remaining O(tokens*spans)
boolean queries in P5 flags and AI-share membership are replaced by an exact
sorted-start/prefix-max-end index. The predicates (strict overlap vs start-point
containment) remain distinct. No scores, ordering, spans, thresholds or features
are modified.100000 synthetic random interval queries matched reference, including
empty, duplicate, inverted and boundary coordinates. A fresh SAME-PROCESS full2M
public JSON comparison against1c3aa13 excludes only latency; native3-repeat
cold/warm timing remains separate. Historical P3/P5 artifacts remain untouched.

Windows failure1c3aa13: real /v1/verify500 body reports missing urllib.error.
The frozen parent urllib package was partial; adding Lib to sys.path alone is
insufficient. The launcher now extends already-loaded stdlib package search paths
with the matching portable Lib, and ships app-local VC++ runtime dependencies.
No install, source checkout, host-Python path or registry edits are introduced.

Hosted-runner CPU models vary (P6demo9V45, a P7demo run9V74). They are never
presented as the same physical VM. An additional paired benchmark materializes
ONLY the immutable14a018a application/config snapshot; no old tests or fitting.
It executes P6-before and P7-after on the SAME current runner with the same
measurement harness, pinned dependencies,4logical CPUs,16GB class RAM,2threads,
identical first-train inputs and3cold/warm repeats. New paired P6 controls do NOT
replace historical P6 numbers. HF-before50k was truncated and is explicitly not
a like-for-like full-answer speedup baseline. Timings remain separate from
quality/calibration and same-process full-public-byte proofs.

The interim60min HF-contract timeout was rejected by the unchanged40min
workflow guard. It is reverted to40, not granted an exception or test skip.
All prior completed fresh HF replay jobs fit40min. The additional paired
workflow adds5existing parametrized infrastructure tests: current collection888,
not883. No original test was deleted, relaxed, or newly skipped.

The7262 full-HF job failed before producing an artifact: its historical
`git show 1c3aa13:...` comparison was combined with default shallow checkout.
Full-HF checkout now fetches history. Independent benchmark/calibration/full2M
jobs depend on cached parity, not serial fresh-HF audit; the final archive still
requires EVERY job, including fresh HF audit. No check is removed or weakened.
The fresh HF7262 audit completed09:31:41→10:02:19UTC (30m38s), within40min.
