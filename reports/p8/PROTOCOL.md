# P8 preregistration: 2026-10-19
One authorized official177 inference; no fits (including test fits) before its completion.
Acceptance source b9cf9acdab1ca7bf082d03629ea7e599597f3fa8, byte-unchanged
production package/config/frozen P5 bundle122874B/e0b13c693a058dfd66ed24a52fc14e9c70964e95885358e7b99cc1f328ad6892.
No AI-share opt-in changes. window512_full_answer_v1 uniform >512 positions.
No existing legacy512 test cache is trusted: empty counting cache, fresh product
inference explicitly timed; exact request-local window memoization unchanged.
Pre-label long audit selects id/answer/context only from streaming events.
All177 public outputs persisted+fsynced before aggregate metrics. P4/P5 frozen
evidence_metrics function and engine/gates reused, span FPR token proxy unchanged.
An atomically created p8-official-test177-consumed Git tag rejects second dispatch
and run_attempt>1. Any infrastructure failure preserves traceback/partial evidence;
no quality-number retry and no subsequent parameter changes permitted.
Corpus B recovered from same pinned CSV hashes, not altered/relabeled. It has
423exact human spans across333positive answers, so all supported metric families
are measured; original one ambiguous substring first-match policy disclosed.
Four disjoint shards, union1000unique IDs, aggregation has no inference/fitting.
License unclear: raw B texts stay in workflow artifacts, not Git.
Before official completion, preregistration push uses [skip ci] solely to enforce
user's absolute no-test-fit barrier. Ordinary CI is not disabled/modified;
full suites and normal CI run only AFTER the official measurement is sealed.
Bad numbers are published, not tuned. Joint test gate alone permits latest release.

## Sole infrastructure recovery
Original39266027598 failed before first completed prediction: eval() calls
transformers.train(False), rejected by wrapper's too-narrow torch-only exception.
Traceback retained. Recovery changes only measurement guard to allow the same
False-mode transformers evaluation call, never train(True)/fit. Product unchanged.
Before recovery, download original artifact and assert empty evidence/no result;
retain original consumed tag, atomically create separate recovery-1 tag.
Long metadata audit177/7over512/max1266 already published before labels.
