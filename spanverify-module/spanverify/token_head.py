"""P2: Token-head на основе GradientBoosting с экспортом в JSON.

Этот модуль реализует:
1. Обучение GradientBoostingClassifier из sklearn (train-only)
2. Экспорт модели в переносимый JSON
3. Runtime predictor на pure Python (без sklearn)

Схема признаков (8):
1-5: Пять исходных FEATURE_NAMES из HF FeatureMatrix
6: position = token_index / max(1, total_token_count)
7: length = min(1, len(token.text.strip()) / 20)
8: has_digit = int(any(ch.isdigit() for ch in token.text))

Критические детали экспорта:
- init="zero" при обучении → initial_prediction=0.0
- Runtime конвертирует входы в float32 для паритета с sklearn
- Точный паритет с sklearn.predict_proba[:,1] при max_delta ≤ 1e-12
"""
from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from .features import FEATURE_NAMES


TOKEN_HEAD_SCHEMA_VERSION = "token_gbdt_v1"
N_TOKEN_FEATURES = 8  # 5 HF + position + length + has_digit


def _to_float32(value: float) -> float:
    """Конвертировать float64 в float32 и обратно для паритета с sklearn.
    
    sklearn деревья обрабатывают входы как float32. Для точного паритета
    runtime должен делать то же самое.
    """
    return struct.unpack('f', struct.pack('f', float(value)))[0]


def extract_token_features(
    token_index: int,
    total_tokens: int,
    token_text: str,
    hf_features: dict[str, float],
) -> list[float]:
    """Извлечь 8 признаков для одного токена.
    
    Args:
        token_index: индекс токена в последовательности
        total_tokens: общее число токенов
        token_text: текст токена
        hf_features: словарь {feature_name: value} для 5 HF-признаков
    
    Returns:
        Список из 8 float значений
    
    Raises:
        ValueError: если отсутствует HF-признак или значение не finite
    """
    features = []
    
    # 1-5: HF признаки (строго в порядке FEATURE_NAMES)
    for name in FEATURE_NAMES:
        if name not in hf_features:
            raise ValueError(f"Missing HF feature: {name}")
        val = hf_features[name]
        if not math.isfinite(val):
            raise ValueError(f"HF feature {name} is not finite: {val}")
        features.append(val)
    
    # 6: position
    position = token_index / max(1, total_tokens)
    features.append(position)
    
    # 7: length
    length = min(1.0, len(token_text.strip()) / 20.0)
    features.append(length)
    
    # 8: has_digit
    has_digit = 1.0 if any(ch.isdigit() for ch in token_text) else 0.0
    features.append(has_digit)
    
    return features


@dataclass
class DecisionTree:
    """Представление одного дерева решений в JSON-формате."""
    threshold: float
    feature_index: int
    left: Optional[DecisionTree] = None
    right: Optional[DecisionTree] = None
    value: Optional[float] = None  # leaf value
    
    def to_dict(self) -> dict[str, Any]:
        if self.value is not None:
            return {"value": self.value}
        if self.left is None or self.right is None:
            raise ValueError("Internal node must have both children")
        return {
            "threshold": self.threshold,
            "feature": self.feature_index,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionTree:
        if "value" in data:
            val = float(data["value"])
            if not math.isfinite(val):
                raise ValueError(f"Leaf value is not finite: {val}")
            return cls(threshold=0.0, feature_index=0, value=val)
        
        # Internal node
        if "left" not in data or "right" not in data:
            raise ValueError("Internal node must have left and right children")
        if "threshold" not in data or "feature" not in data:
            raise ValueError("Internal node must have threshold and feature")
        
        threshold = float(data["threshold"])
        feature_index = int(data["feature"])
        
        if not math.isfinite(threshold):
            raise ValueError(f"Threshold is not finite: {threshold}")
        if feature_index < 0 or feature_index >= N_TOKEN_FEATURES:
            raise ValueError(f"Feature index out of range: {feature_index}")
        
        return cls(
            threshold=threshold,
            feature_index=feature_index,
            left=cls.from_dict(data["left"]),
            right=cls.from_dict(data["right"]),
        )
    
    def predict_one(self, features: list[float]) -> float:
        """Предсказать значение для одного вектора признаков."""
        if self.value is not None:
            return self.value
        if self.left is None or self.right is None:
            raise ValueError("Internal node has missing children")
        if features[self.feature_index] <= self.threshold:
            return self.left.predict_one(features)
        else:
            return self.right.predict_one(features)


@dataclass
class TokenGBDTModel:
    """Переносимая модель GradientBoosting для token classification."""
    trees: List[DecisionTree]
    learning_rate: float
    initial_prediction: float
    schema_version: str = TOKEN_HEAD_SCHEMA_VERSION
    n_features: int = N_TOKEN_FEATURES
    
    def to_dict(self) -> dict[str, Any]:
        if not math.isfinite(self.learning_rate):
            raise ValueError(f"Learning rate is not finite: {self.learning_rate}")
        if not math.isfinite(self.initial_prediction):
            raise ValueError(f"Initial prediction is not finite: {self.initial_prediction}")
        return {
            "type": "token_gbdt",
            "schema_version": self.schema_version,
            "n_features": self.n_features,
            "learning_rate": self.learning_rate,
            "initial_prediction": self.initial_prediction,
            "trees": [tree.to_dict() for tree in self.trees],
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenGBDTModel:
        if data.get("type") != "token_gbdt":
            raise ValueError(f"Unknown token model type: {data.get('type')}")
        if data.get("schema_version") != TOKEN_HEAD_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema version: {data.get('schema_version')}"
            )
        
        n_features = int(data.get("n_features", N_TOKEN_FEATURES))
        if n_features != N_TOKEN_FEATURES:
            raise ValueError(
                f"Expected {N_TOKEN_FEATURES} features, got {n_features}"
            )
        
        learning_rate = float(data["learning_rate"])
        initial_prediction = float(data["initial_prediction"])
        
        if not math.isfinite(learning_rate):
            raise ValueError(f"Learning rate is not finite: {learning_rate}")
        if not math.isfinite(initial_prediction):
            raise ValueError(f"Initial prediction is not finite: {initial_prediction}")
        
        return cls(
            trees=[DecisionTree.from_dict(t) for t in data["trees"]],
            learning_rate=learning_rate,
            initial_prediction=initial_prediction,
            schema_version=data["schema_version"],
            n_features=n_features,
        )
    
    def save(self, path: str | Path) -> None:
        """Сохранить модель в JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
    
    @classmethod
    def load(cls, path: str | Path) -> TokenGBDTModel:
        """Загрузить модель из JSON."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)
    
    def predict_one(self, features: list[float]) -> float:
        """Предсказать вероятность для одного токена.
        
        Returns:
            Вероятность в [0, 1]
        
        Raises:
            ValueError: если неверное число признаков или NaN/inf
        """
        if len(features) != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {len(features)}"
            )
        
        # Проверка NaN/inf
        for i, val in enumerate(features):
            if not math.isfinite(val):
                raise ValueError(f"Feature {i} is not finite: {val}")
        
        # Конвертация в float32 для паритета с sklearn
        features_f32 = [_to_float32(f) for f in features]
        
        # Sum of tree predictions
        raw_score = self.initial_prediction
        for tree in self.trees:
            raw_score += self.learning_rate * tree.predict_one(features_f32)
        
        # Stable sigmoid с защитой от переполнения
        raw_score = max(-30.0, min(30.0, raw_score))
        return 1.0 / (1.0 + math.exp(-raw_score))
    
    def predict_many(self, features_matrix: list[list[float]]) -> list[float]:
        """Предсказать вероятности для матрицы признаков."""
        return [self.predict_one(features) for features in features_matrix]


def train_token_gbdt(
    X: list[list[float]],
    y: list[int],
    sample_weights: Optional[list[float]] = None,
) -> TokenGBDTModel:
    """Обучить GradientBoosting на токенах (train-only).
    
    Args:
        X: матрица признаков (n_samples x 8)
        y: бинарные метки (0 или 1)
        sample_weights: веса образцов (для балансировки классов)
    
    Returns:
        Обученная TokenGBDTModel
    
    Эта функция требует scikit-learn и используется только при обучении.
    
    Критические детали:
    - init="zero" → initial_prediction=0.0 всегда
    - Если заданы sample_weights, они используются для обучения
    - Экспорт использует тот же sklearn объект для паритета
    """
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.utils import class_weight
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for training. "
            "Install with: pip install scikit-learn"
        ) from e
    
    import numpy as np
    
    X_array = np.array(X, dtype=np.float32)
    y_array = np.array(y, dtype=np.int32)
    
    # Sample weights для балансировки
    if sample_weights is None:
        # Автоматическая балансировка по классам
        weights = class_weight.compute_sample_weight("balanced", y_array)
    else:
        weights = np.array(sample_weights, dtype=np.float32)
    
    # Обучение GradientBoosting с init="zero"
    gbdt = GradientBoostingClassifier(
        loss="log_loss",
        n_estimators=80,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=20,
        subsample=1.0,
        init="zero",  # КРИТИЧНО: zero init → initial_prediction=0.0
        random_state=42,
    )
    
    gbdt.fit(X_array, y_array, sample_weight=weights)
    
    # С init="zero" начальное предсказание всегда 0.0
    initial_pred = 0.0
    
    # Конвертация деревьев в наш формат
    trees = []
    for tree_ensemble in gbdt.estimators_[:, 0]:  # shape: (n_estimators,)
        # Каждый элемент - это DecisionTreeRegressor
        tree = _convert_sklearn_tree(tree_ensemble.tree_)
        trees.append(tree)
    
    return TokenGBDTModel(
        trees=trees,
        learning_rate=gbdt.learning_rate,
        initial_prediction=initial_pred,
    )


def export_token_gbdt_from_sklearn(gbdt) -> TokenGBDTModel:
    """Экспортировать уже обученный sklearn GradientBoosting в наш формат.
    
    Эта функция используется для точного паритета: берём тот же sklearn объект,
    который использовался для обучения, и конвертируем его в наш формат.
    
    Args:
        gbdt: обученный sklearn.ensemble.GradientBoostingClassifier
    
    Returns:
        TokenGBDTModel с теми же деревьями
    
    Raises:
        ValueError: если init не "zero"
    """
    # Проверяем что init="zero"
    if hasattr(gbdt, 'init_') and gbdt.init_ is not None:
        # sklearn >= 1.1 использует init_ атрибут
        if not isinstance(gbdt.init_, str) or gbdt.init_ != "zero":
            raise ValueError(
                f"Only init='zero' is supported, got {gbdt.init_}"
            )
    
    initial_pred = 0.0  # init="zero" всегда даёт 0.0
    
    # Конвертация деревьев
    trees = []
    for tree_ensemble in gbdt.estimators_[:, 0]:
        tree = _convert_sklearn_tree(tree_ensemble.tree_)
        trees.append(tree)
    
    return TokenGBDTModel(
        trees=trees,
        learning_rate=gbdt.learning_rate,
        initial_prediction=initial_pred,
    )


def _convert_sklearn_tree(sklearn_tree) -> DecisionTree:
    """Рекурсивно конвертировать sklearn tree в наш формат."""
    node_id = 0
    return _convert_node(sklearn_tree, node_id)


def _convert_node(sklearn_tree, node_id: int) -> DecisionTree:
    """Рекурсивная конвертация узла."""
    # Leaf node
    if sklearn_tree.children_left[node_id] == -1:
        # Value для regression tree
        value = float(sklearn_tree.value[node_id, 0, 0])
        if not math.isfinite(value):
            raise ValueError(f"Tree leaf value is not finite: {value}")
        return DecisionTree(threshold=0.0, feature_index=0, value=value)
    
    # Internal node
    feature = int(sklearn_tree.feature[node_id])
    threshold = float(sklearn_tree.threshold[node_id])
    left_child = int(sklearn_tree.children_left[node_id])
    right_child = int(sklearn_tree.children_right[node_id])
    
    if not math.isfinite(threshold):
        raise ValueError(f"Tree threshold is not finite: {threshold}")
    if feature < 0 or feature >= N_TOKEN_FEATURES:
        raise ValueError(f"Tree feature index out of range: {feature}")
    
    left = _convert_node(sklearn_tree, left_child)
    right = _convert_node(sklearn_tree, right_child)
    
    return DecisionTree(
        threshold=threshold,
        feature_index=feature,
        left=left,
        right=right,
    )
