"""Тесты для P2 token-head (GradientBoosting с экспортом в JSON).

Критические проверки:
1. Точный паритет runtime vs sklearn (max delta ≤ 1e-12)
2. float32 конвертация для порогов деревьев
3. init="zero" → initial_prediction=0.0
4. Guards: NaN/inf, schema, feature indices, missing children
5. Save/load roundtrip без изменения предсказаний
"""

import json
import tempfile
from pathlib import Path

import pytest

from spanverify.token_head import (
    TOKEN_HEAD_SCHEMA_VERSION,
    DecisionTree,
    TokenGBDTModel,
    _to_float32,
    export_token_gbdt_from_sklearn,
    extract_token_features,
)


class TestExtractTokenFeatures:
    """Тесты извлечения 8 признаков для токена."""

    def test_basic_extraction(self):
        """Проверка базового извлечения признаков."""
        hf_features = {
            "attention_entropy": 0.5,
            "ctx_attention_mass": 0.3,
            "ctx_attention_mass_norm": 0.4,
            "ctx_max_similarity": 0.6,
            "embedding_density": 0.7,
        }

        features = extract_token_features(
            token_index=5,
            total_tokens=20,
            token_text="123test",
            hf_features=hf_features,
        )

        assert len(features) == 8
        assert features[0] == 0.5  # attention_entropy
        assert features[4] == 0.7  # embedding_density
        assert features[5] == 0.25  # position = 5/20
        assert features[6] == 0.35  # length = min(1, 7/20)
        assert features[7] == 1.0  # has_digit (есть '1', '2', '3')

    def test_no_digit(self):
        """Токен без цифр."""
        hf_features = {
            "attention_entropy": 0.1,
            "ctx_attention_mass": 0.2,
            "ctx_attention_mass_norm": 0.3,
            "ctx_max_similarity": 0.4,
            "embedding_density": 0.5,
        }

        features = extract_token_features(
            token_index=0,
            total_tokens=10,
            token_text="test",
            hf_features=hf_features,
        )

        assert features[7] == 0.0  # no digit

    def test_missing_feature_raises(self):
        """Отсутствующий HF-признак вызывает ошибку."""
        hf_features = {
            "attention_entropy": 0.1,
            # ctx_attention_mass отсутствует
            "ctx_attention_mass_norm": 0.3,
            "ctx_max_similarity": 0.4,
            "embedding_density": 0.5,
        }

        with pytest.raises(ValueError, match="Missing HF feature"):
            extract_token_features(0, 10, "test", hf_features)

    def test_nan_feature_raises(self):
        """NaN в HF-признаке вызывает ошибку."""
        hf_features = {
            "attention_entropy": float("nan"),
            "ctx_attention_mass": 0.2,
            "ctx_attention_mass_norm": 0.3,
            "ctx_max_similarity": 0.4,
            "embedding_density": 0.5,
        }

        with pytest.raises(ValueError, match="not finite"):
            extract_token_features(0, 10, "test", hf_features)


class TestToFloat32:
    """Тесты конвертации float64 → float32."""

    def test_exact_match(self):
        """Значения точно совпадают после конвертации."""
        assert _to_float32(1.0) == 1.0
        assert _to_float32(0.0) == 0.0
        assert _to_float32(-1.0) == -1.0

    def test_threshold_boundary(self):
        """Критический тест: 1.0 + 1e-9 после float32 становится ровно 1.0.

        Это демонстрирует почему нужна конвертация: sklearn деревья работают
        с float32, и пороговые значения должны сравниваться в float32.
        """
        val = 1.0 + 1e-9
        assert val > 1.0  # float64: больше
        assert _to_float32(val) == 1.0  # float32: ровно 1.0

    def test_precision_loss(self):
        """Float32 теряет точность."""
        val = 1.0000001
        f32_val = _to_float32(val)
        # Float32 имеет ~7 знаков точности
        assert abs(f32_val - val) < 1e-6


class TestTokenGBDTModel:
    """Тесты для TokenGBDTModel."""

    def _make_simple_model(self) -> TokenGBDTModel:
        """Создать простую модель для тестов."""
        # Простое дерево: если feature[0] <= 0.5, то 0.2, иначе 0.8
        tree = DecisionTree(
            threshold=0.5,
            feature_index=0,
            left=DecisionTree(threshold=0.0, feature_index=0, value=0.2),
            right=DecisionTree(threshold=0.0, feature_index=0, value=0.8),
        )

        return TokenGBDTModel(
            trees=[tree],
            learning_rate=0.1,
            initial_prediction=0.0,
        )

    def test_predict_one_basic(self):
        """Базовое предсказание."""
        model = self._make_simple_model()

        # feature[0] = 0.3 <= 0.5 -> tree value = 0.2
        # raw_score = 0.0 + 0.1 * 0.2 = 0.02
        # sigmoid(0.02) ≈ 0.505
        features = [0.3] + [0.0] * 7
        prob = model.predict_one(features)

        assert 0.0 <= prob <= 1.0

    def test_predict_one_right_branch(self):
        """Предсказание в правой ветке."""
        model = self._make_simple_model()

        # feature[0] = 0.7 > 0.5 -> tree value = 0.8
        # raw_score = 0.0 + 0.1 * 0.8 = 0.08
        # sigmoid(0.08) ≈ 0.520
        features = [0.7] + [0.0] * 7
        prob = model.predict_one(features)

        assert 0.0 <= prob <= 1.0
        assert prob > 0.5

    def test_predict_many(self):
        """Множественное предсказание."""
        model = self._make_simple_model()

        matrix = [
            [0.3] + [0.0] * 7,
            [0.7] + [0.0] * 7,
            [0.5] + [0.0] * 7,
        ]

        probs = model.predict_many(matrix)
        assert len(probs) == 3
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_wrong_feature_count_raises(self):
        """Неверное число признаков вызывает ошибку."""
        model = self._make_simple_model()

        with pytest.raises(ValueError, match="Expected 8 features"):
            model.predict_one([0.0] * 5)

    def test_nan_input_raises(self):
        """NaN во входных данных вызывает ошибку."""
        model = self._make_simple_model()

        features = [float("nan")] + [0.0] * 7
        with pytest.raises(ValueError, match="not finite"):
            model.predict_one(features)

    def test_inf_input_raises(self):
        """Inf во входных данных вызывает ошибку."""
        model = self._make_simple_model()

        features = [float("inf")] + [0.0] * 7
        with pytest.raises(ValueError, match="not finite"):
            model.predict_one(features)

    def test_save_load_roundtrip(self):
        """Сохранение и загрузка модели."""
        model = self._make_simple_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            model.save(path)

            # Проверяем что файл валидный JSON
            data = json.loads(path.read_text())
            assert data["type"] == "token_gbdt"
            assert data["schema_version"] == TOKEN_HEAD_SCHEMA_VERSION
            assert data["n_features"] == 8
            assert data["learning_rate"] == 0.1
            assert data["initial_prediction"] == 0.0

            # Загружаем и проверяем предсказания
            loaded = TokenGBDTModel.load(path)

            features = [0.3] + [0.0] * 7
            original_prob = model.predict_one(features)
            loaded_prob = loaded.predict_one(features)

            # Точное совпадение
            assert original_prob == loaded_prob

    def test_unknown_schema_raises(self):
        """Неизвестная схема вызывает ошибку."""
        data = {
            "type": "token_gbdt",
            "schema_version": "unknown_v2",
            "n_features": 8,
            "learning_rate": 0.1,
            "initial_prediction": 0.0,
            "trees": [],
        }

        with pytest.raises(ValueError, match="Unsupported schema version"):
            TokenGBDTModel.from_dict(data)

    def test_unknown_type_raises(self):
        """Неизвестный тип модели вызывает ошибки."""
        data = {
            "type": "unknown_model",
            "schema_version": TOKEN_HEAD_SCHEMA_VERSION,
            "n_features": 8,
            "learning_rate": 0.1,
            "initial_prediction": 0.0,
            "trees": [],
        }

        with pytest.raises(ValueError, match="Unknown token model type"):
            TokenGBDTModel.from_dict(data)

    def test_nan_in_saved_model_raises(self):
        """NaN в сохранённой модели вызывает ошибку при загрузке."""
        data = {
            "type": "token_gbdt",
            "schema_version": TOKEN_HEAD_SCHEMA_VERSION,
            "n_features": 8,
            "learning_rate": 0.1,
            "initial_prediction": float("nan"),
            "trees": [],
        }

        with pytest.raises(ValueError, match="not finite"):
            TokenGBDTModel.from_dict(data)

    def test_wrong_n_features_raises(self):
        """Неверное число признаков в модели вызывает ошибку."""
        data = {
            "type": "token_gbdt",
            "schema_version": TOKEN_HEAD_SCHEMA_VERSION,
            "n_features": 5,  # должно быть 8
            "learning_rate": 0.1,
            "initial_prediction": 0.0,
            "trees": [],
        }

        with pytest.raises(ValueError, match="Expected 8 features"):
            TokenGBDTModel.from_dict(data)


class TestDecisionTree:
    """Тесты для DecisionTree."""

    def test_leaf_predict(self):
        """Предсказание leaf-узла."""
        tree = DecisionTree(threshold=0.0, feature_index=0, value=0.75)
        assert tree.predict_one([0.0] * 8) == 0.75

    def test_internal_predict(self):
        """Предсказание internal-узла."""
        tree = DecisionTree(
            threshold=0.5,
            feature_index=0,
            left=DecisionTree(threshold=0.0, feature_index=0, value=0.2),
            right=DecisionTree(threshold=0.0, feature_index=0, value=0.8),
        )

        assert tree.predict_one([0.3] + [0.0] * 7) == 0.2
        assert tree.predict_one([0.7] + [0.0] * 7) == 0.8

    def test_internal_missing_children_raises(self):
        """Internal-узел без детей вызывает ошибку."""
        tree = DecisionTree(threshold=0.5, feature_index=0, left=None, right=None)

        with pytest.raises(ValueError, match="missing children"):
            tree.predict_one([0.3] + [0.0] * 7)

    def test_to_dict_internal_missing_children_raises(self):
        """Internal-узел без детей не сохраняется."""
        tree = DecisionTree(threshold=0.5, feature_index=0, left=None, right=None)

        with pytest.raises(ValueError, match="must have both children"):
            tree.to_dict()

    def test_to_dict_from_dict(self):
        """Конвертация в словарь и обратно."""
        tree = DecisionTree(
            threshold=0.5,
            feature_index=2,
            left=DecisionTree(threshold=0.0, feature_index=0, value=0.3),
            right=DecisionTree(threshold=0.0, feature_index=0, value=0.7),
        )

        data = tree.to_dict()
        loaded = DecisionTree.from_dict(data)

        features = [0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert tree.predict_one(features) == loaded.predict_one(features)


class TestTrainTokenGBDT:
    """Тесты обучения (требуют sklearn)."""

    def test_tiny_fixture(self):
        """Обучение на fixture."""
        pytest.importorskip("sklearn")

        # Данные побольше для min_samples_leaf=20
        import numpy as np

        from spanverify.token_head import train_token_gbdt

        np.random.seed(42)
        X = np.random.randn(100, 8).tolist()
        y = [1 if x[0] > 0 else 0 for x in X]

        model = train_token_gbdt(X, y)

        assert isinstance(model, TokenGBDTModel)
        assert model.n_features == 8
        assert len(model.trees) == 80  # n_estimators
        assert model.initial_prediction == 0.0  # init="zero"

        # Проверяем предсказания
        probs = model.predict_many(X)
        assert len(probs) == 100
        assert all(0.0 <= p <= 1.0 for p in probs)

        # Модель должна различать классы в среднем
        pos_probs = [probs[i] for i in range(100) if y[i] == 1]
        neg_probs = [probs[i] for i in range(100) if y[i] == 0]
        assert sum(pos_probs) / len(pos_probs) > sum(neg_probs) / len(neg_probs)

    def test_exact_parity_with_sklearn_balanced(self):
        """ТОЧНЫЙ паритет runtime vs sklearn с balanced weights.

        Используем тот же sklearn объект для обучения и экспорта.
        Проверяем max delta ≤ 1e-12.
        """
        pytest.importorskip("sklearn")

        import numpy as np
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.utils import class_weight

        # Обучаем данные
        np.random.seed(42)
        X = np.random.randn(100, 8).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int32)

        # Обучаем sklearn с balanced weights
        weights = class_weight.compute_sample_weight("balanced", y)
        gbdt = GradientBoostingClassifier(
            loss="log_loss",
            n_estimators=80,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=20,
            subsample=1.0,
            init="zero",
            random_state=42,
        )
        gbdt.fit(X, y, sample_weight=weights)

        # Экспортируем из того же sklearn объекта
        model = export_token_gbdt_from_sklearn(gbdt)

        # Сравниваем с sklearn
        sklearn_probs = gbdt.predict_proba(X)[:, 1]
        runtime_probs = model.predict_many(X.tolist())

        # Проверяем ТОЧНЫЙ паритет с допуском 1e-12
        max_delta = 0.0
        for i, (sk_prob, rt_prob) in enumerate(zip(sklearn_probs, runtime_probs, strict=False)):
            delta = abs(sk_prob - rt_prob)
            max_delta = max(max_delta, delta)
            assert delta < 1e-12, f"Row {i}: sklearn={sk_prob:.15f}, runtime={rt_prob:.15f}, " f"delta={delta:.2e}"

        print(f"Max delta: {max_delta:.2e}")

    def test_exact_parity_imbalanced(self):
        """Паритет на imbalanced данных."""
        pytest.importorskip("sklearn")

        import numpy as np
        from sklearn.ensemble import GradientBoostingClassifier

        # 500 строк, 25 positive / 475 negative
        np.random.seed(123)
        X = np.random.randn(500, 8).astype(np.float32)
        y = np.zeros(500, dtype=np.int32)
        y[:25] = 1  # 25 positive, 475 negative

        gbdt = GradientBoostingClassifier(
            loss="log_loss",
            n_estimators=80,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=20,
            subsample=1.0,
            init="zero",
            random_state=42,
        )
        gbdt.fit(X, y)

        model = export_token_gbdt_from_sklearn(gbdt)

        sklearn_probs = gbdt.predict_proba(X)[:, 1]
        runtime_probs = model.predict_many(X.tolist())

        max_delta = max(abs(sk - rt) for sk, rt in zip(sklearn_probs, runtime_probs, strict=False))
        assert max_delta < 1e-12, f"Max delta: {max_delta:.2e}"

    def test_exact_parity_custom_weights(self):
        """Паритет с custom sample weights."""
        pytest.importorskip("sklearn")

        import numpy as np
        from sklearn.ensemble import GradientBoostingClassifier

        np.random.seed(456)
        X = np.random.randn(80, 8).astype(np.float32)
        y = (X[:, 0] > 0.5).astype(np.int32)

        # Custom weights: первые 40 имеют вес 2.0
        weights = np.ones(80, dtype=np.float32)
        weights[:40] = 2.0

        gbdt = GradientBoostingClassifier(
            loss="log_loss",
            n_estimators=80,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=20,
            subsample=1.0,
            init="zero",
            random_state=42,
        )
        gbdt.fit(X, y, sample_weight=weights)

        model = export_token_gbdt_from_sklearn(gbdt)

        sklearn_probs = gbdt.predict_proba(X)[:, 1]
        runtime_probs = model.predict_many(X.tolist())

        max_delta = max(abs(sk - rt) for sk, rt in zip(sklearn_probs, runtime_probs, strict=False))
        assert max_delta < 1e-12, f"Max delta: {max_delta:.2e}"

    def test_exact_parity_held_out(self):
        """Паритет на held-out данных (не участвовавших в обучении)."""
        pytest.importorskip("sklearn")

        import numpy as np
        from sklearn.ensemble import GradientBoostingClassifier

        np.random.seed(789)
        X_train = np.random.randn(100, 8).astype(np.float32)
        y_train = (X_train[:, 0] > 0).astype(np.int32)

        X_test = np.random.randn(30, 8).astype(np.float32)

        gbdt = GradientBoostingClassifier(
            loss="log_loss",
            n_estimators=80,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=20,
            subsample=1.0,
            init="zero",
            random_state=42,
        )
        gbdt.fit(X_train, y_train)

        model = export_token_gbdt_from_sklearn(gbdt)

        sklearn_probs = gbdt.predict_proba(X_test)[:, 1]
        runtime_probs = model.predict_many(X_test.tolist())

        max_delta = max(abs(sk - rt) for sk, rt in zip(sklearn_probs, runtime_probs, strict=False))
        assert max_delta < 1e-12, f"Max delta: {max_delta:.2e}"

    def test_parity_after_save_load(self):
        """Паритет сохраняется после save/load."""
        pytest.importorskip("sklearn")

        import numpy as np
        from sklearn.ensemble import GradientBoostingClassifier

        np.random.seed(999)
        X = np.random.randn(60, 8).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int32)

        gbdt = GradientBoostingClassifier(
            loss="log_loss",
            n_estimators=80,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=20,
            subsample=1.0,
            init="zero",
            random_state=42,
        )
        gbdt.fit(X, y)

        model = export_token_gbdt_from_sklearn(gbdt)

        # Предсказания до save
        probs_before = model.predict_many(X.tolist())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            model.save(path)
            loaded = TokenGBDTModel.load(path)

        # Предсказания после load
        probs_after = loaded.predict_many(X.tolist())

        # Точное совпадение
        for i, (before, after) in enumerate(zip(probs_before, probs_after, strict=False)):
            assert before == after, f"Row {i}: {before} != {after}"

    def test_parity_at_threshold_boundary(self):
        """Паритет на threshold boundary — критический тест float32."""
        pytest.importorskip("sklearn")

        import numpy as np
        from sklearn.ensemble import GradientBoostingClassifier

        # Создаём данные где threshold дерева ≈ 1.0
        np.random.seed(111)
        X = np.random.randn(80, 8).astype(np.float32)
        X[:, 0] = 1.0 + np.random.randn(80).astype(np.float32) * 0.1
        y = (X[:, 0] > 1.0).astype(np.int32)

        gbdt = GradientBoostingClassifier(
            loss="log_loss",
            n_estimators=80,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=20,
            subsample=1.0,
            init="zero",
            random_state=42,
        )
        gbdt.fit(X, y)

        model = export_token_gbdt_from_sklearn(gbdt)

        sklearn_probs = gbdt.predict_proba(X)[:, 1]
        runtime_probs = model.predict_many(X.tolist())

        max_delta = max(abs(sk - rt) for sk, rt in zip(sklearn_probs, runtime_probs, strict=False))
        assert max_delta < 1e-12, f"Max delta: {max_delta:.2e}"
