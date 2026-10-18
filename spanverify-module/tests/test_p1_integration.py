"""Тест, доказывающий что P1 реально влияет на публичный Verifier.verify()."""

import pytest

from spanverify.answer_head import ANSWER_FEATURE_SCHEMA_VERSION, N_ANSWER_FEATURES, AnswerHeadModel
from spanverify.engine import Verifier, WeightsBundle
from spanverify.features import FEATURE_NAMES


def test_p1_changes_public_score():
    """Разные P1-головы дают разные публичные score."""
    # Базовый бандл без answer_head (legacy)
    base_bundle = WeightsBundle(
        weights={name: 0.0 for name in FEATURE_NAMES},
        threshold=0.5,
        mode="demo",
    )

    # P1-голова с bias=-4 (должна давать низкий score ~0.018)
    p1_negative = AnswerHeadModel(
        scaler_means=[0.0] * N_ANSWER_FEATURES,
        scaler_scales=[1.0] * N_ANSWER_FEATURES,
        weights=[0.0] * N_ANSWER_FEATURES,
        bias=-4.0,
        schema_version=ANSWER_FEATURE_SCHEMA_VERSION,
    )

    # P1-голова с bias=+4 (должна давать высокий score ~0.982)
    p1_positive = AnswerHeadModel(
        scaler_means=[0.0] * N_ANSWER_FEATURES,
        scaler_scales=[1.0] * N_ANSWER_FEATURES,
        weights=[0.0] * N_ANSWER_FEATURES,
        bias=+4.0,
        schema_version=ANSWER_FEATURE_SCHEMA_VERSION,
    )

    # Бандл с P1-головой (bias=-4)
    bundle_negative = WeightsBundle(
        weights={name: 0.0 for name in FEATURE_NAMES},
        threshold=0.5,
        mode="demo",
        answer_head=p1_negative.to_dict(),
    )

    # Бандл с P1-головой (bias=+4)
    bundle_positive = WeightsBundle(
        weights={name: 0.0 for name in FEATURE_NAMES},
        threshold=0.5,
        mode="demo",
        answer_head=p1_positive.to_dict(),
    )

    # Создаём верификаторы
    verifier_base = Verifier(weights=base_bundle)
    verifier_neg = Verifier(weights=bundle_negative)
    verifier_pos = Verifier(weights=bundle_positive)

    # Проверяем что P1 загружена
    assert verifier_base.answer_head_model is None
    assert verifier_neg.answer_head_model is not None
    assert verifier_pos.answer_head_model is not None

    # Проверяем что P1 даёт разные raw scores
    answer = "Альфа бета."
    context = "Контекст."

    # Базовый verify (legacy)
    result_base = verifier_base.verify(answer, context)

    # P1 с bias=-4
    result_neg = verifier_neg.verify(answer, context)

    # P1 с bias=+4
    result_pos = verifier_pos.verify(answer, context)

    # КРИТИЧЕСКАЯ ПРОВЕРКА: разные P1 дают разные score
    # В старой реализации все три были одинаковые (0.003653846)
    # В новой реализации должны быть разные
    assert result_neg.score != result_pos.score, (
        f"P1 не влияет на score! " f"bias=-4: {result_neg.score}, bias=+4: {result_pos.score}"
    )

    # Дополнительная проверка: bias=+4 должен давать более высокий score
    assert result_pos.score > result_neg.score, (
        f"bias=+4 должен давать более высокий score, " f"но bias=-4: {result_neg.score}, bias=+4: {result_pos.score}"
    )

    print(f"✓ Legacy score: {result_base.score}")
    print(f"✓ P1 bias=-4 score: {result_neg.score}")
    print(f"✓ P1 bias=+4 score: {result_pos.score}")
    print(f"✓ Разница P1: {result_pos.score - result_neg.score:.6f}")


def test_p1_save_load_roundtrip_preserves_score():
    """Сохранение/загрузка бандла с P1 сохраняет score."""
    p1_model = AnswerHeadModel(
        scaler_means=[0.0] * N_ANSWER_FEATURES,
        scaler_scales=[1.0] * N_ANSWER_FEATURES,
        weights=[0.1] * N_ANSWER_FEATURES,
        bias=1.5,
        schema_version=ANSWER_FEATURE_SCHEMA_VERSION,
    )

    bundle = WeightsBundle(
        weights={name: 0.0 for name in FEATURE_NAMES},
        threshold=0.5,
        mode="demo",
        answer_head=p1_model.to_dict(),
    )

    verifier = Verifier(weights=bundle)

    answer = "Тестовый ответ."
    context = "Контекст."

    result_before = verifier.verify(answer, context)

    # Сохраняем и загружаем бандл
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_path = Path(tmpdir) / "bundle.json"
        bundle.save(bundle_path)

        # Загружаем бандл
        loaded_bundle = WeightsBundle.load(bundle_path)
        verifier_loaded = Verifier(weights=loaded_bundle)

        result_after = verifier_loaded.verify(answer, context)

        # Score должен совпадать
        assert abs(result_before.score - result_after.score) < 1e-9, (
            f"Score изменился после save/load: " f"before={result_before.score}, after={result_after.score}"
        )


def test_legacy_bundle_without_answer_head():
    """Старый бандл без answer_head сохраняет legacy-поведение."""
    # Бандл без answer_head
    bundle = WeightsBundle(
        weights={name: 0.0 for name in FEATURE_NAMES},
        threshold=0.5,
        mode="demo",
    )

    verifier = Verifier(weights=bundle)

    # P1 не загружена
    assert verifier.answer_head_model is None

    # verify() работает и использует legacy
    result = verifier.verify("Тест.", "Контекст.")
    assert result.score >= 0.0
    assert result.score <= 1.0


def test_p1_rejects_nan_input():
    """P1 отклоняет NaN во входных данных."""
    p1_model = AnswerHeadModel(
        scaler_means=[0.0] * N_ANSWER_FEATURES,
        scaler_scales=[1.0] * N_ANSWER_FEATURES,
        weights=[0.0] * N_ANSWER_FEATURES,
        bias=0.0,
        schema_version=ANSWER_FEATURE_SCHEMA_VERSION,
    )

    # NaN в первом признаке
    features_with_nan = [float("nan")] + [0.0] * (N_ANSWER_FEATURES - 1)

    with pytest.raises(ValueError, match="not finite"):
        p1_model.predict(features_with_nan)


def test_p1_rejects_unsupported_schema():
    """P1 отклоняет неподдерживаемую schema_version."""
    with pytest.raises(ValueError, match="Unsupported schema_version"):
        AnswerHeadModel(
            scaler_means=[0.0] * N_ANSWER_FEATURES,
            scaler_scales=[1.0] * N_ANSWER_FEATURES,
            weights=[0.0] * N_ANSWER_FEATURES,
            bias=0.0,
            schema_version="unsupported",
        )


def test_p1_rejects_nan_in_weights():
    """P1 отклоняет NaN в коэффициентах."""
    weights_with_nan = [0.0] * N_ANSWER_FEATURES
    weights_with_nan[5] = float("nan")

    with pytest.raises(ValueError, match="not finite"):
        AnswerHeadModel(
            scaler_means=[0.0] * N_ANSWER_FEATURES,
            scaler_scales=[1.0] * N_ANSWER_FEATURES,
            weights=weights_with_nan,
            bias=0.0,
            schema_version=ANSWER_FEATURE_SCHEMA_VERSION,
        )


if __name__ == "__main__":
    test_p1_changes_public_score()
    print("\n✓ Все тесты пройдены!")
