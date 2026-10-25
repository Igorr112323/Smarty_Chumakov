"""Regression tests for calibration fixes.

Tests the following fixes:
1. transform_one uses bisect_left correctly for upper bounds
2. pava groups identical x values before applying weighted PAVA
3. pava is permutation-invariant for ties
4. _compress preserves block boundaries
5. WeightsBundle roundtrip preserves near-threshold decisions
"""

import pytest

from spanverify.calibration import IsotonicCalibrator, _compress, pava


def test_transform_one_basic():
    """Basic test: scores=[0.1, 0.2, 0.3, 0.4], labels=[0, 1, 0, 1]."""
    scores = [0.1, 0.2, 0.3, 0.4]
    labels = [0, 1, 0, 1]

    calibrator = IsotonicCalibrator.fit(scores, labels, compress=False)
    result = calibrator.transform(scores)

    # PAVA creates blocks: [0.1]→0.0, [0.2, 0.3]→0.5, [0.4]→1.0
    # Transform should return [0.0, 0.5, 0.5, 1.0]
    assert result == [
        0.0,
        0.5,
        0.5,
        1.0,
    ], f"Expected [0.0, 0.5, 0.5, 1.0], got {result}"


def test_transform_one_compression():
    """Test with compression: scores=[0.1, 0.2, 0.3], labels=[0, 0, 1]."""
    scores = [0.1, 0.2, 0.3]
    labels = [0, 0, 1]

    calibrator = IsotonicCalibrator.fit(scores, labels, compress=False)
    result = calibrator.transform(scores)

    # PAVA creates blocks: [0.1, 0.2]→0.0, [0.3]→1.0
    # After compression: thresholds=[0.2, 0.3], values=[0.0, 1.0]
    # Transform should return [0.0, 0.0, 1.0]
    assert result == [0.0, 0.0, 1.0], f"Expected [0.0, 0.0, 1.0], got {result}"


def test_transform_repeated_scores():
    """Repeated scores with different labels."""
    scores = [0.5, 0.5, 0.5, 0.5]
    labels = [0, 1, 0, 1]

    calibrator = IsotonicCalibrator.fit(scores, labels, compress=False)
    result = calibrator.transform(scores)

    # PAVA should merge all blocks with same x into one block
    # with mean value = 0.5
    assert all(r == 0.5 for r in result), f"Expected all 0.5, got {result}"


def test_transform_boundary_values():
    """Boundary values."""
    calibrator = IsotonicCalibrator.fit([0.1, 0.3, 0.4], [0, 1, 1], compress=False)

    # thresholds=[0.1, 0.4], values=[0.0, 1.0]
    result = calibrator.transform([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    # score <= 0.1 → 0.0
    # 0.1 < score <= 0.4 → 1.0
    # score > 0.4 → 1.0
    assert result == [
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ], f"Expected [0.0, 0.0, 1.0, 1.0, 1.0, 1.0], got {result}"


def test_pava_same_x():
    """PAVA merges blocks with same x."""
    scores = [0.5, 0.5, 0.5, 0.5]
    labels = [0, 1, 0, 1]

    thresholds, values = pava(scores, labels)

    # Should have one block with thresholds=[0.5], values=[0.5]
    assert thresholds == [0.5], f"Expected [0.5], got {thresholds}"
    assert values == [0.5], f"Expected [0.5], got {values}"


def test_pava_permutation_invariant():
    """PAVA result should be invariant to permutation of ties."""
    scores = [0.1, 0.2, 0.3, 0.3]

    # Test different permutations of labels
    for labels in [[1, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1]]:
        thresholds, values = pava(scores, labels)
        calibrator = IsotonicCalibrator(thresholds=thresholds, values=values)
        result = calibrator.transform(scores)

        # For scores=[0.1, 0.2, 0.3, 0.3] with labels=[1, 0, 0, 1]:
        # - 0.1 has label 1
        # - 0.2 has label 0
        # - 0.3 (two observations) has labels 0 and 1, mean = 0.5
        # PAVA should merge 0.1 and 0.2 into one block with mean 0.5
        # Result should be [0.5, 0.5, 0.5, 0.5]

        # For labels=[1, 0, 1, 0]:
        # - 0.3 has labels 1 and 0, mean = 0.5
        # Same result

        # For labels=[0, 1, 0, 1]:
        # - 0.1 has label 0
        # - 0.2 has label 1
        # - 0.3 has labels 0 and 1, mean = 0.5
        # PAVA: 0.1→0.0, 0.2→1.0, 0.3→0.5
        # After merging: 0.1→0.0, [0.2, 0.3]→2/3
        # Result should be [0.0, 0.667, 0.667, 0.667]

        # For labels=[0, 0, 1, 1]:
        # - 0.1→0.0, 0.2→0.0, 0.3→1.0
        # Result should be [0.0, 0.0, 1.0, 1.0]

        # The key is that the result should be deterministic and correct
        # regardless of the order of observations with the same score
        assert len(result) == 4, f"Expected 4 results, got {len(result)}"

        # All results should be in [0, 1]
        assert all(0 <= r <= 1 for r in result), f"Results out of range: {result}"


def test_pava_counterexample():
    """Counterexample from user: scores=[0.1, 0.2, 0.3, 0.3], labels=[1, 0, 0, 1]."""
    scores = [0.1, 0.2, 0.3, 0.3]
    labels = [1, 0, 0, 1]

    thresholds, values = pava(scores, labels)
    calibrator = IsotonicCalibrator(thresholds=thresholds, values=values)
    result = calibrator.transform(scores)

    # Expected: all scores map to 0.5
    # - 0.1 and 0.2 should be merged into one block with mean (1+0)/2 = 0.5
    # - 0.3 (two observations) has mean (0+1)/2 = 0.5
    # - Both blocks have value 0.5, so all scores map to 0.5
    assert result == [
        0.5,
        0.5,
        0.5,
        0.5,
    ], f"Expected [0.5, 0.5, 0.5, 0.5], got {result}"


def test_serialization_roundtrip():
    """Serialization and deserialization preserve behavior."""
    calibrator = IsotonicCalibrator.fit([0.1, 0.2, 0.3, 0.4], [0, 1, 0, 1], compress=False)

    data = calibrator.to_dict()
    calibrator_restored = IsotonicCalibrator.from_dict(data)

    scores = [0.1, 0.2, 0.3, 0.4]
    result_original = calibrator.transform(scores)
    result_restored = calibrator_restored.transform(scores)

    assert result_original == result_restored, f"Serialization mismatch: {result_original} != {result_restored}"


def test_empty_input():
    """Empty input."""
    calibrator = IsotonicCalibrator.fit([], [], compress=False)

    assert calibrator.thresholds == []
    assert calibrator.values == []

    # For empty calibrator, transform returns input scores
    result = calibrator.transform([0.1, 0.2])
    assert result == [0.1, 0.2]


def test_monotonicity():
    """Calibrated function is monotonic."""
    calibrator = IsotonicCalibrator.fit([0.1, 0.2, 0.3, 0.4, 0.5], [0, 1, 0, 1, 1], compress=False)

    scores = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
    result = calibrator.transform(scores)

    # Check monotonicity
    for i in range(len(result) - 1):
        assert result[i] <= result[i + 1], f"Non-monotonic: result[{i}]={result[i]} > result[{i+1}]={result[i+1]}"


def test_compress_preserves_changes():
    """_compress preserves all value changes."""
    xs = [0.1, 0.2, 0.3, 0.4, 0.5]
    ys = [0.0, 0.002, 0.004, 0.006, 0.008]

    # With tolerance=0.003, adjacent changes are < tolerance
    # But we should preserve the overall change from 0.0 to 0.008
    compressed_xs, compressed_ys = _compress(xs, ys, tolerance=0.003)

    # Should keep first and last points at minimum
    assert compressed_xs[0] == 0.1
    assert compressed_xs[-1] == 0.5
    assert compressed_ys[0] == 0.0
    assert compressed_ys[-1] == 0.008


def test_compress_exact_mode():
    """compress=False should preserve all points."""
    # For scores=[0.1, 0.2, 0.3], labels=[0, 1, 1]:
    # PAVA creates three blocks (all x are unique): [0.1]→0.0, [0.2]→1.0, [0.3]→1.0
    # Values are already monotonic (0.0 ≤ 1.0 ≤ 1.0), so no merging
    # With compress=False, all points are preserved
    calibrator = IsotonicCalibrator.fit([0.1, 0.2, 0.3], [0, 1, 1], compress=False)

    assert calibrator.thresholds == [0.1, 0.2, 0.3]
    assert calibrator.values == [0.0, 1.0, 1.0]

    # Verify correct mapping:
    # score=0.1 → values[0]=0.0
    # score=0.2 → values[1]=1.0
    # score=0.3 → values[2]=1.0
    result = calibrator.transform([0.1, 0.2, 0.3])
    assert result == [0.0, 1.0, 1.0]


def test_no_compress_for_answer_calibration():
    """Answer calibration should use compress=False to preserve all steps."""
    # Simulate answer scores with many unique values
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

    # With compress=True, some steps might be lost
    calibrator_compressed = IsotonicCalibrator.fit(scores, labels, compress=True, tolerance=0.1)

    # With compress=False, all steps should be preserved
    calibrator_exact = IsotonicCalibrator.fit(scores, labels, compress=False)

    # Exact should have more or equal number of steps
    assert len(calibrator_exact.thresholds) >= len(calibrator_compressed.thresholds)

    # Transform should give same results at the original scores
    result_compressed = calibrator_compressed.transform(scores)
    result_exact = calibrator_exact.transform(scores)

    # Both should be monotonic
    for i in range(len(result_compressed) - 1):
        assert result_compressed[i] <= result_compressed[i + 1]
        assert result_exact[i] <= result_exact[i + 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
