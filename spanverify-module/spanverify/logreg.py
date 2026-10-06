"""Логистическая регрессия и стандартизация без внешних зависимостей.

Модуль выделен из ``spanverify.train``, чтобы им могли пользоваться и обучение
конвейера (голова «достоверно/недостоверно»), и оценка доли участия ИИ
(``spanverify.participation``) — без циклических импортов ``engine → train``.

Реализация намеренно простая: один батч-градиентный спуск, детерминированный
при фиксированном порядке строк. Этого достаточно для корпусов проекта
(тысячи токенов) и не тянет numpy/sklearn в поставку.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

__all__ = ["standardize", "train_logreg", "probabilities"]


def standardize(rows: Sequence[Sequence[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    """Привести строки к нулевому среднему и единичной дисперсии.

    Возвращает ``(нормированные строки, средние, масштабы)``; масштаб нулевой
    дисперсии заменяется единицей, чтобы не делить на ноль.
    """
    if not rows:
        return [], [], []
    width = len(rows[0])
    means = [sum(row[j] for row in rows) / len(rows) for j in range(width)]
    scales = []
    for j in range(width):
        variance = sum((row[j] - means[j]) ** 2 for row in rows) / len(rows)
        scales.append(variance**0.5 or 1.0)
    normalized = [[(row[j] - means[j]) / scales[j] for j in range(width)] for row in rows]
    return normalized, means, scales


def train_logreg(
    rows: Sequence[Sequence[float]],
    labels: Sequence[int],
    epochs: int = 400,
    learning_rate: float = 0.5,
    l2: float = 1e-3,
) -> dict[str, Any]:
    """Обучить логистическую регрессию одним батчем (без внешних зависимостей)."""
    width = len(rows[0]) if rows else 0
    weights = [0.0] * width
    bias = 0.0
    count = max(1, len(rows))
    for _ in range(epochs):
        gradient = [0.0] * width
        bias_gradient = 0.0
        for row, label in zip(rows, labels, strict=False):
            score = bias + sum(w * x for w, x in zip(weights, row, strict=False))
            probability = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, score))))
            error = probability - label
            for j in range(width):
                gradient[j] += error * row[j]
            bias_gradient += error
        for j in range(width):
            weights[j] -= learning_rate * (gradient[j] / count + l2 * weights[j])
        bias -= learning_rate * bias_gradient / count
    return {"weights": weights, "bias": bias}


def probabilities(model: dict[str, Any], rows: Sequence[Sequence[float]]) -> list[float]:
    """Вероятности положительного класса для строк (в том же порядке)."""
    out: list[float] = []
    for row in rows:
        score = model["bias"] + sum(w * x for w, x in zip(model["weights"], row, strict=False))
        out.append(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, score)))))
    return out
