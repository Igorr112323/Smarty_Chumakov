# P0 Status Report

**Date:** 2026-10-17  
**Branch:** arena/c5f69767-smarty-chumakov  
**Current commit:** f17ad95

## Executive Summary

P0 has made significant progress on calibration fixes and infrastructure, but **quality targets are NOT met**. The core issue is that while token-level detection works reasonably well (AUC=0.8489), answer-level aggregation fails to discriminate between clean and fabricated answers (AUC=0.5047), resulting in very high false positive rates.

## Completed Work

### P0-0: Infrastructure ✅
- Feature cache system for HF experiments
- Cross-run artifact download workflow
- Postprocessing pipeline for analysis

### P0-1: Span Expansion ✅
- Implemented 4 span expansion strategies (clause, mask, clause_capped, mask_padded)
- Compared on dev set
- **Result:** clause_capped selected (strict F1=0.1798)
- **Status:** Internal targets NOT met (target ≥0.45)

### P0-2: Answer Calibration (Partial) ✅

#### Calibration Fixes
- Fixed PAVA algorithm to correctly handle ties and upper bounds
- Fixed transform_one to use bisect_left correctly
- Added compress=False option for exact calibration
- **All unit tests pass**

#### HF Bundle Measurement
- Rewrote measure_aggregators.py to require explicit HF bundle
- Prevents accidental use of demo bundle
- Token control: **PASSED** ✅
  - TP=138, FP=636, FN=149, TN=7046 (exact match)
  - Token F1=0.2601, AUC=0.8489

#### Answer-Level Results
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Answer F1 | 0.6099 | ≥0.50 | ✅ |
| Answer FPR | **0.6941** | ≤0.40 | ❌ |
| Answer AUC | **0.5047** | >0.60 | ❌ |
| Answer Precision | 0.5354 | - | - |
| Answer Recall | 0.7083 | - | - |
| Threshold | 0.5951 | - | - |

**Key findings:**
1. Answer AUC=0.5047 is near random (0.5), indicating the model cannot discriminate between clean and fabricated answers at the answer level
2. Answer FPR=0.6941 means 69.4% of clean answers are incorrectly flagged
3. 88.4% of all answers are flagged as containing fabrications
4. Token-level detection works (AUC=0.8489) but aggregation to answer level fails

#### Comparison of Aggregators
**NOT COMPLETED** - Requires retraining answer calibration for each aggregator (current, max, mean_top3). This is the next step.

## Quality Gap Analysis

### Token Level (Works)
- Token AUC: 0.8489 (good discrimination)
- Token F1: 0.2601 (low due to class imbalance)
- Token FPR: 0.0828 (acceptable)

### Span Level (Poor)
- Span F1: 0.1798 (target ≥0.45)
- Span containment: 0.5313 (target ≥0.45)
- Span width ratio: 10.28 (target ≤2.0)

### Answer Level (Fails)
- Answer AUC: 0.5047 (near random)
- Answer FPR: 0.6941 (target ≤0.40)
- Most answers flagged (88.4%)

### Verdict Level (Fails)
- Verdict FPR: 0.8706 (target ≤0.40)

## Root Cause Analysis

The fundamental issue is that **answer-level aggregation does not work**:

1. **Token detection works** (AUC=0.8489) - the model can identify suspicious tokens
2. **Aggregation fails** (AUC=0.5047) - combining token scores into answer scores loses discriminative power
3. **Threshold selection fails** - cannot find a threshold that maintains acceptable FPR

Possible causes:
- Aggregation method (current: 0.5*mean + 0.5*mean_top20%) may not be optimal
- Answer-level calibration may not be learning the right mapping
- Class imbalance in answer labels
- Token-level signals may not be sufficient for answer-level decisions

## Next Steps

### Immediate (P0-2 completion)
1. **Compare three aggregators** on dev with correct HF bundle:
   - Current: 0.5*mean + 0.5*mean_top20%
   - Max: max token score
   - Mean top-3: mean of top 3 token scores
2. Select best aggregator by answer F1 with FPR ≤0.30
3. Save chosen aggregator in bundle

### P0-1: Verdict Calibration
1. Use chosen answer aggregator
2. Calibrate verdict threshold on dev
3. Target: verdict FPR ≤0.40

### P1: Quality Improvement (if P0 fails)
1. Investigate why answer aggregation fails
2. Consider alternative aggregation methods
3. Consider ensemble approaches
4. Consider different model architectures

## Contract Compliance

| Requirement | Target | Current | Status |
|-------------|--------|---------|--------|
| Token F1 | ≥0.50 | 0.2601 | ❌ |
| Token FPR | ≤0.40 | 0.0828 | ✅ |
| Answer F1 | ≥0.50 | 0.6099 | ✅ |
| Answer FPR | ≤0.40 | 0.6941 | ❌ |
| Verdict F1 | ≥0.50 | 0.6719 | ✅ |
| Verdict FPR | ≤0.40 | 0.8706 | ❌ |

**Overall: NOT COMPLIANT** - Multiple FPR targets exceeded

## Key Files

- Calibration fixes: `spanverify-module/spanverify/calibration.py`
- Measurement script: `spanverify-module/scripts/measure_aggregators.py`
- Results: `spanverify-module/reports/experiments/p0_answer/run39060670054/`
- Workflow: `.github/workflows/postprocess-p0-answer.yml`

## Runs

- **39060670054**: Original span expansion comparison (source cache)
- **39072235315**: P0-2 measurement with correct HF bundle ✅
- **Commit**: f17ad95 (report saved)

## Conclusion

P0 has successfully fixed calibration bugs and established proper measurement infrastructure. However, the core quality problem remains: **answer-level aggregation does not work**. The model can detect suspicious tokens but cannot reliably determine if an entire answer contains fabrications.

**Recommendation:** Complete P0-2 aggregator comparison, then proceed to P0-1 verdict calibration. If quality targets cannot be met, investigate fundamental issues with answer-level aggregation before proceeding to contract delivery.
