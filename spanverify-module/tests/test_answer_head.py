"""Тесты для P1 answer-головы."""

import json
import tempfile
from pathlib import Path

import pytest

from spanverify.answer_head import (
    ANSWER_FEATURE_SCHEMA_VERSION,
    AnswerHeadModel,
    compute_percentile,
    extract_answer_features,
    train_answer_head,
)


class TestComputePercentile:
    """Тесты для compute_percentile."""

    def test_single_value(self):
        assert compute_percentile([5.0], 50) == 5.0

    def test_median_odd(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert compute_percentile(values, 50) == 3.0

    def test_q10_q90(self):
        values = list(range(1, 101))
        values = [float(v) for v in values]
        q10 = compute_percentile(values, 10)
        q90 = compute_percentile(values, 90)
        assert 10.0 <= q10 <= 12.0
        assert 90.0 <= q90 <= 92.0


class TestExtractAnswerFeatures:
    """Тесты для extract_answer_features."""

    def test_basic_extraction(self):
        """Проверяем базовое извлечение признаков."""
        features_list = [
            {
                "attention_entropy": 0.5,
                "ctx_attention_mass": 0.3,
                "ctx_attention_mass_norm": 0.4,
                "ctx_max_similarity": 0.6,
                "embedding_density": 0.7,
            },
            {
                "attention_entropy": 0.6,
                "ctx_attention_mass": 0.4,
                "ctx_attention_mass_norm": 0.5,
                "ctx_max_similarity": 0.7,
                "embedding_density": 0.8,
            },
        ]
        scored_mask = [True, True]

        result = extract_answer_features(features_list, scored_mask)
        assert len(result) == 37  # 5 * 7 + 2

    def test_empty_scored(self):
        """Пустой scored-набор: 35 нулевых сводок + log1p(0)=0 + empty=1."""
        features_list = [
            {
                "attention_entropy": 0.5,
                "ctx_attention_mass": 0.3,
                "ctx_attention_mass_norm": 0.4,
                "ctx_max_similarity": 0.6,
                "embedding_density": 0.7,
            },
        ]
        scored_mask = [False]

        result = extract_answer_features(features_list, scored_mask)
        assert len(result) == 37
        # Первые 35 — нули (нет scored токенов)
        assert result[:35] == [0.0] * 35
        # log1p(0) = 0
        assert result[35] == 0.0
        # empty flag = 1
        assert result[36] == 1.0

    def test_length_mismatch_raises(self):
        features_list = [{"attention_entropy": 0.5}]
        scored_mask = [True, False]
        with pytest.raises(ValueError, match="не совпадают"):
            extract_answer_features(features_list, scored_mask)

    def test_nan_raises(self):
        features_list = [
            {
                "attention_entropy": float("nan"),
                "ctx_attention_mass": 0.3,
                "ctx_attention_mass_norm": 0.4,
                "ctx_max_similarity": 0.6,
                "embedding_density": 0.7,
            },
        ]
        scored_mask = [True]
        with pytest.raises(ValueError, match="NaN/inf"):
            extract_answer_features(features_list, scored_mask)


class TestAnswerHeadModel:
    """Тесты для AnswerHeadModel."""

    def test_n_features_property(self):
        model = AnswerHeadModel(
            scaler_means=[0.0] * 37,
            scaler_scales=[1.0] * 37,
            weights=[0.1] * 37,
            bias=0.0,
        )
        assert model.n_features == 37

    def test_predict_scalar(self):
        model = AnswerHeadModel(
            scaler_means=[0.0] * 37,
            scaler_scales=[1.0] * 37,
            weights=[0.1] * 37,
            bias=0.0,
        )
        result = model.predict([0.0] * 37)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_predict_many(self):
        model = AnswerHeadModel(
            scaler_means=[0.0] * 37,
            scaler_scales=[1.0] * 37,
            weights=[0.1] * 37,
            bias=0.0,
        )
        matrix = [[float(i)] * 37 for i in range(5)]
        results = model.predict_many(matrix)
        assert len(results) == 5
        assert all(isinstance(r, float) for r in results)
        assert all(0.0 <= r <= 1.0 for r in results)

    def test_predict_wrong_size_raises(self):
        model = AnswerHeadModel(
            scaler_means=[0.0] * 37,
            scaler_scales=[1.0] * 37,
            weights=[0.1] * 37,
            bias=0.0,
        )
        with pytest.raises(ValueError, match="37"):
            model.predict([0.0] * 10)

    def test_save_load_roundtrip(self):
        """Сохранение/загрузка на 37-мерном fixture."""
        model = AnswerHeadModel(
            scaler_means=[0.1] * 37,
            scaler_scales=[1.5] * 37,
            weights=[0.2] * 37,
            bias=0.3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            model.save(path)

            # Проверяем что файл валидный JSON
            data = json.loads(path.read_text(encoding="utf-8"))
            assert len(data["weights"]) == 37
            assert len(data["scaler_means"]) == 37
            assert len(data["scaler_scales"]) == 37

            loaded = AnswerHeadModel.load(path)

            assert loaded.n_features == 37
            assert loaded.bias == 0.3
            assert loaded.weights == model.weights
            assert loaded.scaler_means == model.scaler_means
            assert loaded.scaler_scales == model.scaler_scales

    def test_save_load_predict_equality(self):
        """Предсказания после save/load должны совпадать."""
        import random

        random.seed(42)
        model = train_answer_head(
            [[random.gauss(0, 1) for _ in range(37)] for _ in range(10)],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        )

        test_matrix = [[random.gauss(0, 1) for _ in range(37)] for _ in range(5)]
        original_scores = model.predict_many(test_matrix)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            model.save(path)
            loaded = AnswerHeadModel.load(path)
            loaded_scores = loaded.predict_many(test_matrix)

            for a, b in zip(original_scores, loaded_scores, strict=False):
                assert abs(a - b) < 1e-12


class TestTrainAnswerHead:
    """Тесты для train_answer_head."""

    def test_basic_training(self):
        import random

        random.seed(42)
        X = [[random.gauss(0, 1) for _ in range(37)] for _ in range(20)]
        y = [0] * 10 + [1] * 10

        model = train_answer_head(X, y)
        assert model.n_features == 37
        assert model.schema_version == ANSWER_FEATURE_SCHEMA_VERSION

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            train_answer_head([[0.0] * 37], [0, 1])

    def test_nan_raises(self):
        X = [[float("nan")] * 37]
        y = [0]
        with pytest.raises(ValueError, match="NaN/inf"):
            train_answer_head(X, y)


class TestSmokeIntegration:
    """Сквозной smoke: train → n_features → predict → predict_many → save → load → равенство."""

    def test_full_pipeline(self):
        import random

        random.seed(123)

        # Обучаем на 4 искусственных векторах
        X = [[random.gauss(0, 1) for _ in range(37)] for _ in range(4)]
        y = [0, 1, 0, 1]

        model = train_answer_head(X, y)

        # n_features
        assert model.n_features == 37

        # predict (скаляр)
        score_one = model.predict(X[0])
        assert isinstance(score_one, float)
        assert 0.0 <= score_one <= 1.0

        # predict_many
        scores_many = model.predict_many(X)
        assert len(scores_many) == 4
        assert abs(scores_many[0] - score_one) < 1e-12

        # save → load
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            model.save(path)
            loaded = AnswerHeadModel.load(path)

            # Равенство scores
            loaded_scores = loaded.predict_many(X)
            for a, b in zip(scores_many, loaded_scores, strict=False):
                assert abs(a - b) < 1e-12

            # Равенство решений (threshold=0.5)
            threshold = 0.5
            decisions_original = [s >= threshold for s in scores_many]
            decisions_loaded = [s >= threshold for s in loaded_scores]
            assert decisions_original == decisions_loaded
