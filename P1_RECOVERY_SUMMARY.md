# P1 Recovery Summary

## Status: ✅ Successfully Completed

## Timeline

### Initial Problem (2026-10-18)
- Run 39114874149 failed with API compatibility errors
- Three critical issues found:
  1. `AttributeError: 'AnswerHeadModel' object has no attribute 'n_features'`
  2. `TypeError: unsupported operand type(s) for -: 'list' and 'float'` in `predict()`
  3. `AttributeError: 'AnswerHeadModel' object has no attribute 'save'`

### Root Cause Analysis
- Original implementation had 41 features (incorrect), should be 37
- Used `zip(strict=False)` which silently truncated mismatches
- Missing critical API methods for model persistence
- Cross-fitting OOF implementation was incomplete

### Fixes Implemented (Commit b878507)

#### 1. AnswerHeadModel API Completion
```python
# Added properties and methods
@property
def n_features(self) -> int:
    return len(self.weights)

def save(self, path: str | Path) -> None:
    """Save model to JSON file"""

@classmethod
def load(cls, path: str | Path) -> "AnswerHeadModel":
    """Load model from JSON file"""

def predict_many(self, matrix: list[list[float]]) -> list[float]:
    """Predict probabilities for batch of feature vectors"""
```

#### 2. Feature Count Correction
- Fixed from 41 to 37 features
- 5 HF features × 7 statistics (mean, std, min, q10, median, q90, max) + 2 metadata
- Removed incorrect `zip(strict=False)` usage

#### 3. Cross-Fitting OOF Implementation
```python
# Proper 5-fold cross-fitting with seed=42
folds, doc_id_by_pair = _build_doc_folds(train_pairs, n_folds=5, seed=42)

for fold_idx in range(5):
    # Train on 4 folds
    fold_model = train_answer_head(fit_X, fit_y)
    
    # Predict on held-out fold
    held_out_scores = fold_model.predict_many(held_out_X)
```

#### 4. Calibration & Threshold Selection
```python
# Isotonic calibration with proper FPR control
calibrator_p1 = IsotonicCalibrator.fit(p1_oof_scores, train_labels, compress=False)
report_p1 = choose_threshold_report(cal_p1_train, train_labels, max_fpr=0.30)
threshold_p1 = float(report_p1["threshold"])  # 0.5794392523364486
```

#### 5. Reproducibility Improvements
- SHA-256 hashes for fold receipts (replaced non-reproducible `hash()`)
- Full coverage validation (842 train + 181 dev pairs)
- No silent fallbacks with `dict.get(pid, 0.0)`

### Verification (Commit d59cd4e)
- Successfully recovered all 14 artifact files from Azure Blob Storage
- Export run 39125627036 published raw files to branch
- SHA-256 verification of all exported files

### Results

#### Dev Set Performance (181 pairs)
| Metric | Baseline (mean_top3) | P1 (answer-head) |
|--------|---------------------|------------------|
| Raw AUC | 0.5083333333 | **0.7321078431** |
| Calibrated AUC | 0.4838235294 | **0.7338235294** |
| Precision | 0.0000000000 | **0.7777777778** |
| Recall | 0.0000000000 | **0.5104166667** |
| **F1** | **0.0000000000** | **0.6163522013** |
| **FPR** | **0.0000000000** | **0.1647058824** |
| TP/FP/FN/TN | 0/0/96/85 | **49/14/47/71** |

#### Contract Compliance
- ✅ F1 ≥ 0.50 (achieved: 0.616)
- ✅ FPR ≤ 0.40 (achieved: 0.165)

## Technical Details

### Feature Engineering
```
37 features total:
- 5 HF features (attention_entropy, ctx_attention_mass, ctx_attention_mass_norm, 
                 ctx_max_similarity, embedding_density)
- 7 statistics per feature: mean, std, min, q10, median, q90, max
- 2 metadata: log1p(token_count), empty_flag
```

### Model Architecture
- **Algorithm**: L2-regularized logistic regression
- **Training**: Gradient descent, lr=0.1, 100 iterations, l2_penalty=1.0
- **Standardization**: Per-feature mean/std normalization
- **Calibration**: Isotonic regression with PAVA algorithm

### Experimental Protocol
- **Train set**: 842 pairs (5-fold cross-validation)
- **Dev set**: 181 pairs (held-out)
- **Seed**: 42 (reproducible folds)
- **Threshold selection**: max_fpr=0.30 on training set

## Files Modified

### Core Implementation
- `spanverify/answer_head.py` - Complete P1 answer-head implementation
- `spanverify/oof.py` - Cross-fitting OOF with answer feature extraction
- `spanverify/engine.py` - Public Verifier integration + performance optimization
- `spanverify/features.py` - Restored from b878507 for compatibility
- `spanverify/calibration.py` - Restored from b878507 for compatibility
- `spanverify/train.py` - Restored from b878507 for compatibility

### Scripts
- `scripts/measure_p1.py` - Measurement script with cross-fitting
- `scripts/build_p1_candidate.py` - P1 bundle assembly
- `scripts/replay_p1_dev.py` - Public API replay on dev set

### Workflows
- `.github/workflows/postprocess-p0-answer-p1.yml` - Original measurement
- `.github/workflows/recover-p1-results.yml` - Raw export from Azure
- `.github/workflows/p1-public-replay.yml` - Public API replay

### Tests
- `tests/test_answer_head.py` - 17 tests for answer-head
- `tests/test_p1_integration.py` - 6 tests for public API integration
- Total: 56+ tests passing

## Known Limitations

1. **Baseline anomaly**: mean_top3 aggregator produced 0 positive predictions (threshold too conservative)
2. **Token-head caveat**: baseline_bundle contains historical token-head, not final fine-tuned version
3. **Answer-head independence**: P1 uses label-free HF features, independent of token-head quality
4. **Replay needed**: Full token/span/verdict metrics not yet measured on current candidate

## Next Steps

1. Complete public replay through Verifier API (workflow ready)
2. Measure token/span/verdict metrics on current candidate
3. Analyze baseline aggregator failure (0 predictions)
4. Consider joint optimization if contract targets not met

## References

- Original measurement: https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/39116955685
- Export run: https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/39125627036
- Recovered files: `spanverify-module/reports/experiments/p1_raw/run39116955685/`
- P1 candidate README: `spanverify-module/reports/experiments/p1_raw/run39116955685/P1_CANDIDATE_README.md`

## Conclusion

P1 implementation successfully recovered and verified. Answer-head achieves F1=0.616 with FPR=0.165, exceeding contract targets. All API compatibility issues resolved, cross-fitting OOF properly implemented, reproducibility ensured through SHA-256 verification.
