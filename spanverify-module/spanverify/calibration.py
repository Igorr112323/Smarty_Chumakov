"""Калибровка оценки: изотоническая регрессия (PAVA) и подбор порога.

Сырая оценка детектора не имеет вероятностного смысла. PAVA строит
монотонное отображение «сырая оценка -> вероятность ИИ» на размеченных
данных, после чего порог выбирается по ограничению на долю ложных
срабатываний (``max_fpr``).
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "pava",
    "IsotonicCalibrator",
    "choose_threshold",
    "choose_threshold_report",
    "metrics_at",
    "cross_validate",
]


def pava(scores: Sequence[float], labels: Sequence[float]) -> tuple[list[float], list[float]]:
    """Изотоническая регрессия методом смежных нарушителей (PAVA).

    Возвращает (значения_score, сглаженные_значения) — кусочно-постоянную
    неубывающую функцию от score.
    """
    if len(scores) != len(labels):
        raise ValueError("scores и labels должны быть одной длины")
    if not scores:
        return [], []

    order = sorted(range(len(scores)), key=lambda i: scores[i])
    xs = [float(scores[i]) for i in order]
    ys = [float(labels[i]) for i in order]

    # Блоки: [сумма y, количество, суммарный вес]
    blocks: list[list[float]] = []
    for x, y in zip(xs, ys, strict=False):
        blocks.append([y, 1.0, x])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            y2, n2, x2 = blocks.pop()
            y1, n1, x1 = blocks.pop()
            blocks.append([y1 + y2, n1 + n2, x2 if n2 >= n1 else x1])

    values: list[float] = []
    xs_out: list[float] = []
    for total, count, _ in blocks:
        values.append(total / count)
    # Границы блоков — по исходным score.
    bounds: list[float] = []
    idx = 0
    for block in blocks:
        size = int(block[1])
        bounds.append(xs[idx + size - 1])
        idx += size
    xs_out = bounds
    return xs_out, values


@dataclass
class IsotonicCalibrator:
    """Монотонное отображение сырой оценки в вероятность «текст от ИИ»."""

    thresholds: list[float]
    values: list[float]
    meta: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if len(self.thresholds) != len(self.values):
            raise ValueError("thresholds и values должны быть одной длины")

    # ---------- применение ----------

    def transform_one(self, score: float) -> float:
        if not self.values:
            return float(score)
        if score <= self.thresholds[0]:
            return float(self.values[0])
        if score >= self.thresholds[-1]:
            return float(self.values[-1])
        lo, hi = 0, len(self.thresholds) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if self.thresholds[mid] <= score:
                lo = mid
            else:
                hi = mid
        return float(self.values[lo])

    def transform(self, scores: Iterable[float]) -> list[float]:
        return [self.transform_one(s) for s in scores]

    # ---------- обучение ----------

    @classmethod
    def fit(
        cls,
        scores: Sequence[float],
        labels: Sequence[float],
        tolerance: float = 0.003,
        max_points: int = 2000,
        **meta: Any,
    ) -> IsotonicCalibrator:
        """Обучить калибратор и сжать отображение без потери монотонности.

        PAVA на десятках тысяч токенов даёт тысячи блоков; для хранения и
        быстрого применения оставляем только точки, где значение меняется
        заметно (``tolerance``), но не больше ``max_points``.
        """
        xs, ys = pava(scores, labels)
        xs, ys = _compress(xs, ys, tolerance=tolerance, max_points=max_points)
        return cls(thresholds=xs, values=ys, meta=dict(meta) if meta else None)

    # ---------- сериализация ----------

    def to_dict(self) -> dict[str, Any]:
        return {"thresholds": self.thresholds, "values": self.values, "meta": self.meta or {}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IsotonicCalibrator:
        return cls(
            thresholds=[float(x) for x in data.get("thresholds", [])],
            values=[float(v) for v in data.get("values", [])],
            meta=dict(data.get("meta", {})),
        )

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        return p

    @classmethod
    def load(cls, path: str | Path) -> IsotonicCalibrator | None:
        """Загрузить калибратор с диска или из бандла приложения."""
        p = Path(path)
        if p.is_file():
            with p.open("r", encoding="utf-8") as fh:
                return cls.from_dict(json.load(fh))
        if p.is_absolute():
            return None

        from .config import read_runtime_text

        embedded = read_runtime_text(str(p))
        return cls.from_dict(json.loads(embedded)) if embedded else None


def metrics_at(probs: Sequence[float], labels: Sequence[int], threshold: float) -> dict[str, float]:
    """Метрики бинарной классификации при заданном пороге."""
    tp = fp = tn = fn = 0
    for prob, label in zip(probs, labels, strict=False):
        predicted = 1 if prob >= threshold else 0
        if predicted and label:
            tp += 1
        elif predicted and not label:
            fp += 1
        elif not predicted and label:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    hdr = tn / (tn + fp) if tn + fp else 0.0  # доля верно опознанных «человеческих» токенов
    accuracy = (tp + tn) / max(1, len(labels))

    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "hdr": hdr,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def choose_threshold_report(
    probs: Sequence[float],
    labels: Sequence[int],
    max_fpr: float = 0.10,
    candidates: int = 200,
) -> dict[str, float | bool]:
    """Подобрать порог ответа: максимум F1 при ограничении FPR <= max_fpr.

    Возвращает порог и метрики на нём, включая признак ``constraint_met`` —
    достижимо ли ограничение. Признак нужен, потому что недостижимость не редкость,
    а нормальный исход на слабом разделении, и скрывать её нельзя.

    Историческая ошибка, которую здесь исправляет fallback: раньше при
    недостижимом ограничении брался порог с минимальным FPR, а это всегда
    максимальный порог — конвейер переставал помечать что-либо вообще. В прогоне
    37472951524 на корпусе A3 это дало порог 1,0, recall 0,036 и token F1 0,013
    при том, что кросс-валидация головы показывала AUC 0,85: признаки работали,
    а порог их гасил. Теперь берётся порог с максимальным F1, а фактический FPR
    честно попадает в отчёт.
    """
    empty: dict[str, float | bool] = {
        "threshold": 0.5,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "fpr": 0.0,
        "constraint_met": False,
    }
    if not probs:
        return empty
    uniq = sorted(set(float(p) for p in probs))
    if len(uniq) > candidates:
        step = len(uniq) / candidates
        uniq = [uniq[int(i * step)] for i in range(candidates)]
    grid = sorted(set(uniq + [max(uniq) + 1e-9]))

    best_constrained: tuple[float, float, dict] = (-1.0, -1.0, empty)
    best_free: tuple[float, float, dict] = (-1.0, -1.0, empty)
    for thr in grid:
        m = metrics_at(probs, labels, thr)
        f1, fpr = m["f1"], m["fpr"]
        if fpr <= max_fpr and (f1 > best_constrained[0] or (f1 == best_constrained[0] and -fpr > best_constrained[1])):
            best_constrained = (f1, -fpr, dict(m, constraint_met=True))
        if f1 > best_free[0] or (f1 == best_free[0] and -fpr > best_free[1]):
            best_free = (f1, -fpr, dict(m, constraint_met=False))

    if best_constrained[0] >= 0:
        return best_constrained[2]
    # Ограничение недостижимо: не молчим, а берём лучший F1 и фиксируем FPR.
    return dict(best_free[2])


def choose_threshold(
    probs: Sequence[float],
    labels: Sequence[int],
    max_fpr: float = 0.10,
    candidates: int = 200,
) -> float:
    """Порог ответа: максимум F1 при ограничении FPR, иначе максимум F1.

    Если ограничение недостижимо, возвращается порог с лучшим F1, а не «порог с
    минимальным FPR» (то есть максимум шкалы, при котором конвейер не помечает
    ничего). Подробности — в :func:`choose_threshold_report`.
    """
    return float(choose_threshold_report(probs, labels, max_fpr=max_fpr, candidates=candidates)["threshold"])


def cross_validate(
    scores: Sequence[float],
    labels: Sequence[int],
    folds: int = 5,
    max_fpr: float = 0.10,
    seed: int = 1312,
) -> dict[str, Any]:
    """k-fold кросс-валидация калибровки.

    На каждом фолде калибратор обучается на остальных фолдах, порог
    подбирается на обучающей части и применяется к отложенной.
    """
    if len(scores) != len(labels):
        raise ValueError("scores и labels должны быть одной длины")
    n = len(scores)
    if n < folds or folds < 2:
        probs = IsotonicCalibrator.fit(scores, labels).transform(scores)
        thr = choose_threshold(probs, labels, max_fpr=max_fpr)
        return {"folds": [], "mean": metrics_at(probs, labels, thr), "n": n}

    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    fold_size = n // folds
    fold_reports: list[dict[str, Any]] = []
    all_probs = [0.0] * n

    for fold in range(folds):
        start = fold * fold_size
        stop = n if fold == folds - 1 else start + fold_size
        test_idx = indices[start:stop]
        train_idx = indices[:start] + indices[stop:]
        calibrator = IsotonicCalibrator.fit([scores[i] for i in train_idx], [labels[i] for i in train_idx])
        train_probs = calibrator.transform([scores[i] for i in train_idx])
        thr = choose_threshold(train_probs, [labels[i] for i in train_idx], max_fpr=max_fpr)
        test_probs = calibrator.transform([scores[i] for i in test_idx])
        for i, prob in zip(test_idx, test_probs, strict=False):
            all_probs[i] = prob
        report = metrics_at(test_probs, [labels[i] for i in test_idx], thr)
        report["fold"] = fold
        report["n_test"] = len(test_idx)
        fold_reports.append(report)

    def mean_of(key: str) -> float:
        return sum(r[key] for r in fold_reports) / len(fold_reports)

    overall_thr = choose_threshold(all_probs, labels, max_fpr=max_fpr)
    mean = {k: mean_of(k) for k in ("precision", "recall", "f1", "fpr", "hdr", "accuracy")}
    mean["threshold"] = mean_of("threshold")
    return {
        "folds": fold_reports,
        "mean": mean,
        "overall": metrics_at(all_probs, labels, overall_thr),
        "n": n,
    }


def _compress(
    xs: Sequence[float],
    ys: Sequence[float],
    tolerance: float = 0.003,
    max_points: int = 2000,
) -> tuple[list[float], list[float]]:
    """Прореживание изотонной ступенчатой функции."""
    if len(xs) <= 2:
        return [float(x) for x in xs], [float(y) for y in ys]

    keep: list[int] = [0]
    for i in range(1, len(xs) - 1):
        if abs(ys[i] - ys[keep[-1]]) > tolerance:
            keep.append(i)
    keep.append(len(xs) - 1)

    if len(keep) > max_points:
        step = len(keep) / max_points
        sampled = [keep[int(i * step)] for i in range(max_points)]
        if sampled[-1] != keep[-1]:
            sampled.append(keep[-1])
        keep = sampled

    return [float(xs[i]) for i in keep], [float(ys[i]) for i in keep]
