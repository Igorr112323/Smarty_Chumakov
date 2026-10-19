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
