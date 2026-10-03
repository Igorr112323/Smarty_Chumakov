"""Обучение калибратора и подбор порога на размеченном корпусе.

Единый путь для CLI (``spanverify calibrate``), скриптов экспериментов и
тестов: собираем по токенам сырые (сглаженные) оценки детектора, метки из
разметки, обучаем изотоническую регрессию и выбираем порог при ограничении
на долю ложных срабатываний.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .calibration import (
    IsotonicCalibrator,
    choose_threshold,
    cross_validate,
    metrics_at,
)
from .config import Config
from .detector import Detector
from .text import tokenize

__all__ = ["TrainingReport", "collect_scores", "token_labels", "train_calibrator"]


def token_labels(tokens: Sequence, labels: Iterable[Sequence[Any]]) -> list[int]:
    """Разметка по токенам из разметки по символам ``[[start, end, label], ...]``."""
    flags = [0] * len(tokens)
    for start, end, label in labels:
        if int(label) != 1:
            continue
        for i, token in enumerate(tokens):
            if token.start < int(end) and token.end > int(start):
                flags[i] = 1
    return flags


def collect_scores(
    detector: Detector,
    documents: Iterable[dict],
) -> tuple[list[float], list[int], dict[str, int]]:
    """Калиброванные вероятности по токенам и метки для корпуса документов.

    Оценки берутся тем же методом, что и при анализе, поэтому порог,
    подобранный по этим значениям, применим и в рабочем режиме.
    """
    scores: list[float] = []
    labels: list[int] = []
    stats = {"documents": 0, "skipped": 0, "tokens": 0}

    for doc in documents:
        text = doc.get("text", "")
        tokens = [t for t in tokenize(text) if t.is_word]
        if not tokens:
            stats["skipped"] += 1
            continue
        probs = detector.token_probabilities(text)
        flags = token_labels(tokens, doc.get("labels", []))
        if len(probs) != len(flags):
            stats["skipped"] += 1
            continue
        scores.extend(probs)
        labels.extend(flags)
        stats["documents"] += 1
        stats["tokens"] += len(flags)

    return scores, labels, stats


@dataclass
class TrainingReport:
    """Итог обучения: калибратор, порог, метрики, статистика корпуса."""

    calibrator: IsotonicCalibrator
    threshold: float
    metrics: dict[str, float]
    cross_validation: dict[str, Any]
    stats: dict[str, int] = field(default_factory=dict)
    dataset: str = ""

    def summary(self) -> str:
        mean = self.cross_validation.get("mean", {})
        return (
            f"документов={self.stats.get('documents')} токенов={self.stats.get('tokens')} "
            f"порог={self.threshold:.4f} "
            f"CV: P={mean.get('precision', 0):.3f} R={mean.get('recall', 0):.3f} "
            f"F1={mean.get('f1', 0):.3f} FPR={mean.get('fpr', 0):.3f}"
        )


def train_calibrator(
    detector: Detector,
    documents: Sequence[dict],
    config: Config | None = None,
    dataset_name: str = "",
) -> TrainingReport:
    """Обучить калибратор, подобрать порог и посчитать метрики.

    Оценки берутся без калибровки: иначе уже откалиброванные значения
    попали бы в изотоническую регрессию повторно. Бэкенд переиспользуется,
    чтобы не загружать модель дважды.
    """
    config = config or detector.config
    scoring = (
        Detector(config, backend=detector.backend, calibrator=False) if detector.calibrator is not None else detector
    )
    scores, labels, stats = collect_scores(scoring, documents)
    if not scores:
        raise ValueError("не удалось собрать оценки: проверьте разметку корпуса")

    cross = cross_validate(scores, labels, folds=config.folds, max_fpr=config.max_fpr, seed=config.seed)
    calibrator = IsotonicCalibrator.fit(scores, labels)
    probs = calibrator.transform(scores)
    threshold = choose_threshold(probs, labels, max_fpr=config.max_fpr)
    metrics = metrics_at(probs, labels, threshold)

    ai_share = sum(labels) / len(labels)
    calibrator.meta = {
        "trained_on": dataset_name or "inline",
        "documents": stats["documents"],
        "tokens": stats["tokens"],
        "ai_token_share": round(ai_share, 4),
        "threshold": round(threshold, 4),
        "metrics_in_sample": {k: round(v, 4) for k, v in metrics.items()},
        "cross_validation_mean": {k: round(v, 4) for k, v in cross.get("mean", {}).items()},
        "note": (
            "Калибровка на синтетическом корпусе проверяет конвейер, а не качество "
            "на реальных текстах. Боевой порог обучается на размеченных данных "
            "в режиме backend='hf'."
        ),
    }
    return TrainingReport(
        calibrator=calibrator,
        threshold=threshold,
        metrics=metrics,
        cross_validation=cross,
        stats=stats,
        dataset=dataset_name,
    )
