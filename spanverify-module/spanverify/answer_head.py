"""P1: Обучаемая answer-голова на сводках HF-признаков.

Этот модуль реализует отдельную L2-logreg answer-голову, которая использует
сводки (mean, std, min, q10, median, q90, max) по пяти HF-признакам для
каждого scored-токена, а также log1p(scored_token_count) и empty-scored flag.

Используются полные исходные FeatureMatrix, до supervised token-head и
answer-калибровки, не округлённый payload.

Схема признаков (37):
- 5 FEATURE_NAMES × 7 сводок (mean, std, min, q10, median, q90, max) = 35
- log1p(scored_token_count) = 1
- empty-scored flag = 1
Итого: 37 признаков.

Порядок сводок: mean, std, min, q10, median, q90, max.
Метод квантилей: линейная интерполяция (как numpy.percentile method='linear').
std: выборочное стандартное отклонение (n-1), для n=1 → 0.0.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .features import FEATURE_NAMES

# Schema version для отслеживания изменений в структуре признаков
ANSWER_FEATURE_SCHEMA_VERSION = "1.0"

# Число признаков в схеме 1.0
N_ANSWER_FEATURES = len(FEATURE_NAMES) * 7 + 2  # 5*7+2 = 37

# Порядок сводок
STATISTIC_NAMES = ("mean", "std", "min", "q10", "median", "q90", "max")


def compute_percentile(values: list[float], percentile: float) -> float:
    """Вычислить перцентиль для списка значений.

    Использует линейную интерполяцию, как numpy.percentile с method='linear'.
    """
    if not values:
        return 0.0

    sorted_values = sorted(values)
    n = len(sorted_values)

    if n == 1:
        return sorted_values[0]

    # Вычисляем позицию
    k = (n - 1) * (percentile / 100.0)
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return sorted_values[int(k)]

    # Линейная интерполяция
    d0 = sorted_values[int(f)] * (c - k)
    d1 = sorted_values[int(c)] * (k - f)
    return d0 + d1


def extract_answer_features(
    features_list: list[dict[str, float]],
    scored_mask: list[bool],
) -> list[float]:
    """Извлечь сводки HF-признаков для answer-головы.

    Args:
        features_list: список словарей с HF-признаками для каждого токена.
            Каждый словарь содержит ключи из FEATURE_NAMES.
        scored_mask: список булевых масок, указывающих, является ли токен scored.

    Returns:
        Список из 37 признаков:
        - Для каждого из 5 FEATURE_NAMES по 7 сводок: mean, std, min, q10, median, q90, max
        - log1p(scored_token_count)
        - empty-scored flag

    Raises:
        ValueError: если lengths не совпадают, или в features_list есть
            NaN/inf, или отсутствует обязательный HF-признак.
    """
    if len(features_list) != len(scored_mask):
        raise ValueError(
            f"Длины features_list ({len(features_list)}) и scored_mask " f"({len(scored_mask)}) не совпадают"
        )

    # Извлекаем признаки только для scored токенов
    scored_features: dict[str, list[float]] = {name: [] for name in FEATURE_NAMES}

    for i, (features, is_scored) in enumerate(zip(features_list, scored_mask, strict=False)):
        if is_scored:
            for name in FEATURE_NAMES:
                if name not in features:
                    raise ValueError(f"Токен {i}: отсутствует обязательный HF-признак '{name}'")
                val = features[name]
                if not math.isfinite(val):
                    raise ValueError(f"Токен {i}: признак '{name}' содержит NaN/inf ({val})")
                scored_features[name].append(val)

    # Вычисляем сводки для каждого признака
    answer_features: list[float] = []

    for name in FEATURE_NAMES:
        values = scored_features[name]

        if not values:
            # Если нет scored токенов, используем нулевые значения для 7 сводок
            answer_features.extend([0.0] * 7)
        else:
            # 7 сводок: mean, std, min, q10, median, q90, max
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0.0
            min_val = min(values)
            q10_val = compute_percentile(values, 10)
            median_val = statistics.median(values)
            q90_val = compute_percentile(values, 90)
            max_val = max(values)

            answer_features.extend(
                [
                    mean_val,
                    std_val,
                    min_val,
                    q10_val,
                    median_val,
                    q90_val,
                    max_val,
                ]
            )

    # Добавляем log1p(scored_token_count)
    scored_token_count = sum(1 for is_scored in scored_mask if is_scored)
    answer_features.append(math.log1p(scored_token_count))

    # Добавляем empty-scored flag (1 если нет scored токенов, иначе 0)
    answer_features.append(1.0 if scored_token_count == 0 else 0.0)

    return answer_features


@dataclass
class AnswerHeadModel:
    """Модель answer-головы (L2-logreg).

    Атрибуты:
        scaler_means: средние значения для стандартизации признаков
        scaler_scales: стандартные отклонения для стандартизации признаков
        weights: веса логистической регрессии
        bias: смещение логистической регрессии
        schema_version: версия схемы признаков
        feature_order: порядок имён признаков
        statistic_order: порядок сводок
    """

    scaler_means: list[float]
    scaler_scales: list[float]
    weights: list[float]
    bias: float
    schema_version: str = ANSWER_FEATURE_SCHEMA_VERSION
    feature_order: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    statistic_order: list[str] = field(default_factory=lambda: list(STATISTIC_NAMES))

    def __post_init__(self) -> None:
        n = len(self.weights)
        if len(self.scaler_means) != n:
            raise ValueError(f"scaler_means length ({len(self.scaler_means)}) != weights length ({n})")
        if len(self.scaler_scales) != n:
            raise ValueError(f"scaler_scales length ({len(self.scaler_scales)}) != weights length ({n})")

        # Проверка schema_version
        if self.schema_version != ANSWER_FEATURE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version: {self.schema_version!r}. " f"Expected {ANSWER_FEATURE_SCHEMA_VERSION!r}"
            )

        # Проверка feature_order и statistic_order
        if self.feature_order != list(FEATURE_NAMES):
            raise ValueError(f"feature_order mismatch: expected {list(FEATURE_NAMES)}, got {self.feature_order}")
        if self.statistic_order != list(STATISTIC_NAMES):
            raise ValueError(f"statistic_order mismatch: expected {list(STATISTIC_NAMES)}, got {self.statistic_order}")

        # Проверка NaN/inf в коэффициентах
        for i, w in enumerate(self.weights):
            if not math.isfinite(w):
                raise ValueError(f"weights[{i}] is not finite: {w}")
        for i, m in enumerate(self.scaler_means):
            if not math.isfinite(m):
                raise ValueError(f"scaler_means[{i}] is not finite: {m}")
        for i, s in enumerate(self.scaler_scales):
            if not math.isfinite(s):
                raise ValueError(f"scaler_scales[{i}] is not finite: {s}")
            if s < 0:
                raise ValueError(f"scaler_scales[{i}] is negative: {s}")
        if not math.isfinite(self.bias):
            raise ValueError(f"bias is not finite: {self.bias}")

    @property
    def n_features(self) -> int:
        """Число признаков модели (размерность весов)."""
        return len(self.weights)

    def to_dict(self) -> dict[str, Any]:
        """Преобразовать модель в словарь для сохранения."""
        return {
            "type": "answer_logreg",
            "scaler_means": self.scaler_means,
            "scaler_scales": self.scaler_scales,
            "weights": self.weights,
            "bias": self.bias,
            "schema_version": self.schema_version,
            "feature_order": self.feature_order,
            "statistic_order": self.statistic_order,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnswerHeadModel:
        """Загрузить модель из словаря."""
        return cls(
            scaler_means=list(data["scaler_means"]),
            scaler_scales=list(data["scaler_scales"]),
            weights=list(data["weights"]),
            bias=float(data["bias"]),
            schema_version=data.get("schema_version", ANSWER_FEATURE_SCHEMA_VERSION),
            feature_order=list(data.get("feature_order", FEATURE_NAMES)),
            statistic_order=list(data.get("statistic_order", STATISTIC_NAMES)),
        )

    def save(self, path: str | Path) -> None:
        """Сохранить модель в JSON-файл (UTF-8)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> AnswerHeadModel:
        """Загрузить модель из JSON-файла."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def _standardize_one(self, answer_features: list[float]) -> list[float]:
        """Стандартизировать один вектор признаков."""
        if len(answer_features) != self.n_features:
            raise ValueError(
                f"Входной вектор имеет длину {len(answer_features)}, " f"но модель ожидает {self.n_features} признаков"
            )
        # Проверка NaN/inf во входных данных
        for i, x in enumerate(answer_features):
            if not math.isfinite(x):
                raise ValueError(f"answer_features[{i}] is not finite: {x}")
        return [
            (x - mean) / (scale if scale > 0 else 1.0)
            for x, mean, scale in zip(answer_features, self.scaler_means, self.scaler_scales, strict=False)
        ]

    def _score_standardized(self, standardized: list[float]) -> float:
        """Вычислить вероятность из стандартизованного вектора."""
        logit = self.bias + sum(w * x for w, x in zip(self.weights, standardized, strict=False))
        # Сигмоида с защитой от переполнения
        logit = max(-30.0, min(30.0, logit))
        return 1.0 / (1.0 + math.exp(-logit))

    def predict(self, answer_features: list[float]) -> float:
        """Предсказать вероятность недостоверности ответа для одного вектора.

        Args:
            answer_features: список из 37 признаков для answer-головы

        Returns:
            Вероятность недостоверности ответа (от 0 до 1)

        Raises:
            ValueError: если длина вектора не соответствует модели или содержит NaN/inf
        """
        standardized = self._standardize_one(answer_features)
        return self._score_standardized(standardized)

    def predict_many(self, matrix: list[list[float]]) -> list[float]:
        """Предсказать вероятности для матрицы признаков.

        Args:
            matrix: список строк, каждая — список из 37 признаков

        Returns:
            Список вероятностей (по одной на строку)
        """
        return [self.predict(row) for row in matrix]


def train_answer_head(
    answer_features_list: list[list[float]],
    labels: list[int],
    l2_penalty: float = 1.0,
    learning_rate: float = 0.1,
    n_iterations: int = 100,
) -> AnswerHeadModel:
    """Обучить answer-голову (L2-logreg).

    Args:
        answer_features_list: список списков признаков для каждого ответа
        labels: список меток (0 или 1) для каждого ответа
        l2_penalty: коэффициент регуляризации L2
        learning_rate: шаг градиентного спуска
        n_iterations: число итераций

    Returns:
        Обученная модель AnswerHeadModel

    Raises:
        ValueError: если lengths не совпадают или есть NaN/inf
    """
    if not answer_features_list or not labels:
        raise ValueError("answer_features_list and labels must not be empty")

    if len(answer_features_list) != len(labels):
        raise ValueError(f"Длины не совпадают: {len(answer_features_list)} vs {len(labels)}")

    n_samples = len(answer_features_list)
    n_features = len(answer_features_list[0])

    # Проверка размеров
    for i, row in enumerate(answer_features_list):
        if len(row) != n_features:
            raise ValueError(f"Строка {i} имеет длину {len(row)}, ожидалось {n_features}")
        for j, val in enumerate(row):
            if not math.isfinite(val):
                raise ValueError(f"Строка {i}, признак {j}: NaN/inf ({val})")

    # Стандартизация: вычисляем mean и std для каждого признака
    scaler_means: list[float] = []
    scaler_scales: list[float] = []

    for j in range(n_features):
        values = [answer_features_list[i][j] for i in range(n_samples)]
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if n_samples > 1 else 1.0
        scaler_means.append(mean_val)
        scaler_scales.append(std_val if std_val > 0 else 1.0)

    # Стандартизируем признаки
    standardized_features: list[list[float]] = []
    for i in range(n_samples):
        standardized = [
            (x - mean) / (scale if scale > 0 else 1.0)
            for x, mean, scale in zip(answer_features_list[i], scaler_means, scaler_scales, strict=False)
        ]
        standardized_features.append(standardized)

    # Обучаем логистическую регрессию с L2 регуляризацией
    weights = [0.0] * n_features
    bias = 0.0

    for _ in range(n_iterations):
        # Вычисляем градиенты
        grad_weights = [0.0] * n_features
        grad_bias = 0.0

        for i in range(n_samples):
            # Предсказание
            logit = bias + sum(w * x for w, x in zip(weights, standardized_features[i], strict=False))
            logit = max(-30.0, min(30.0, logit))
            prediction = 1.0 / (1.0 + math.exp(-logit))

            # Ошибка
            error = prediction - labels[i]

            # Градиенты
            for j in range(n_features):
                grad_weights[j] += error * standardized_features[i][j]
            grad_bias += error

        # Усредняем градиенты
        for j in range(n_features):
            grad_weights[j] /= n_samples
            # Добавляем L2 регуляризацию
            grad_weights[j] += l2_penalty * weights[j]
        grad_bias /= n_samples

        # Обновляем веса
        for j in range(n_features):
            weights[j] -= learning_rate * grad_weights[j]
        bias -= learning_rate * grad_bias

    return AnswerHeadModel(
        scaler_means=scaler_means,
        scaler_scales=scaler_scales,
        weights=weights,
        bias=bias,
        schema_version=ANSWER_FEATURE_SCHEMA_VERSION,
        feature_order=list(FEATURE_NAMES),
        statistic_order=list(STATISTIC_NAMES),
    )
