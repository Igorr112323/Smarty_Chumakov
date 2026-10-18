"""Тесты для P2 token-head (GradientBoosting с экспортом в JSON)."""
import json
import math
import tempfile
from pathlib import Path

import pytest

from spanverify.token_head import (
    N_TOKEN_FEATURES,
    TOKEN_HEAD_SCHEMA_VERSION,
    DecisionTree,
    TokenGBDTModel,
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
        assert abs(prob - 0.505) < 0.01
    
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
            
            # Загружаем и проверяем предсказания
            loaded = TokenGBDTModel.load(path)
            
            features = [0.3] + [0.0] * 7
            original_prob = model.predict_one(features)
            loaded_prob = loaded.predict_one(features)
            
            assert abs(original_prob - loaded_prob) < 1e-12
    
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
        """Неизвестный тип модели вызывает ошибку."""
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
        """Обучение на маленьком fixture."""
        pytest.importorskip("sklearn")
        
        from spanverify.token_head import train_token_gbdt
        
        # Данные побольше для min_samples_leaf=20
        import numpy as np
        np.random.seed(42)
        X = np.random.randn(100, 8).tolist()
        y = [1 if x[0] > 0 else 0 for x in X]
        
        model = train_token_gbdt(X, y)
        
        assert isinstance(model, TokenGBDTModel)
        assert model.n_features == 8
        assert len(model.trees) == 80  # n_estimators
        
        # Проверяем предсказания
        probs = model.predict_many(X)
        assert len(probs) == 100
        assert all(0.0 <= p <= 1.0 for p in probs)
        
        # Модель должна различать классы в среднем
        pos_probs = [probs[i] for i in range(100) if y[i] == 1]
        neg_probs = [probs[i] for i in range(100) if y[i] == 0]
        assert sum(pos_probs) / len(pos_probs) > sum(neg_probs) / len(neg_probs)
    
    def test_sklearn_runtime_parity(self):
        """Паритет sklearn и runtime predictor.
        
        Проверяем что runtime predictor даёт близкие результаты к sklearn.
        Точный паритет невозможен из-за различий float32/float64 и конвертации деревьев.
        """
        pytest.importorskip("sklearn")
        
        from spanverify.token_head import train_token_gbdt
        from sklearn.ensemble import GradientBoostingClassifier
        import numpy as np
        
        # Обучаем данные
        np.random.seed(42)
        X = np.random.randn(50, 8).astype(np.float32)
        y = (X[:, 0] > 0).astype(int)
        
        model = train_token_gbdt(X.tolist(), y.tolist())
        
        # Сравниваем с sklearn
        gbdt = GradientBoostingClassifier(
            loss="log_loss",
            n_estimators=80,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=20,
            subsample=1.0,
            random_state=42,
        )
        gbdt.fit(X, y)
        
        sklearn_probs = gbdt.predict_proba(X)[:, 1]
        runtime_probs = model.predict_many(X.tolist())
        
        # Проверяем что предсказания близки (допуск 1e-2 из-за float32)
        max_delta = max(abs(sk - rt) for sk, rt in zip(sklearn_probs, runtime_probs))
        mean_delta = sum(abs(sk - rt) for sk, rt in zip(sklearn_probs, runtime_probs)) / len(sklearn_probs)
        
        assert max_delta < 0.01, f"Max delta too large: {max_delta}"
        assert mean_delta < 0.005, f"Mean delta too large: {mean_delta}"
        
        # Проверяем что бинарные предсказания совпадают
        sklearn_binary = (sklearn_probs >= 0.5).astype(int)
        runtime_binary = [1 if p >= 0.5 else 0 for p in runtime_probs]
        assert sklearn_binary.tolist() == runtime_binary
