"""Обучение: подбор весов признаков, порог маски, голова и калибровка.

Три процедуры, все — на обучающей части корпуса, метрики считаются на
отложенной (группировка по парам, утечки между частями нет):

1. **Подбор весов признаков** (сетка с шагом 0.1, сумма = 1) по AUC на
   токенном уровне.
2. **Порог маски** ``T = median + z·1.4826·MAD`` в границах ``[floor, cap]``:
   перебор ``z`` и ``floor`` по F1 при ограничении FPR.
3. **Решающая голова** — логистическая регрессия по признакам токена
   [H, m, d, risk, позиция, длина] с 5-кратной кросс-валидацией. Если голова
   не превосходит пороговый метод на валидации — выбирается пороговый, и это
   фиксируется в отчёте (тоже результат).

Завершается изотонической калибровкой (PAVA) оценки ответа и подбором порога
при целевом FPR. Итог сохраняется в ``config/weights.json`` и, при победе
головы, в ``config/head.json``.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._version import __version__
from .calibration import IsotonicCalibrator, choose_threshold_report, metrics_at
from .core import tokenize_with_offsets
from .engine import (
    SMOOTH_WINDOW,
    Verifier,
    WeightsBundle,
    _answer_score,
    _smooth,
    span_threshold_for,
)
from .features import DEFAULT_WEIGHTS, combine, is_scored_token
from .logreg import (
    probabilities as _head_probabilities,
)
from .logreg import (
    standardize as _standardize,
)
from .logreg import (
    train_logreg as _train_logreg,
)
from .participation import ParticipationModel, build_participation_corpus, corpus_rows_and_labels

__all__ = [
    "TrainReport",
    "shared_groups",
    "TokenSample",
    "collect_samples",
    "auc_score",
    "train",
    "save_training_artifacts",
]

WEIGHT_GRID_STEP = 0.1
Z_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
FLOOR_GRID = (0.25, 0.3, 0.35, 0.4, 0.45, 0.5)
CAP_GRID = (0.5, 0.55, 0.6, 0.7, 0.8)
HEAD_FEATURES = ("attention_entropy", "ctx_attention_mass", "embedding_density", "risk", "position", "length")


@dataclass
class TokenSample:
    """Токен ответа с признаками и меткой."""

    pair_id: str
    index: int
    total: int
    text: str
    features: dict[str, float]
    label: int


def collect_samples(
    verifier: Verifier,
    pairs: Iterable[dict],
) -> tuple[list[TokenSample], dict[str, Any]]:
    """Собрать токенные признаки и метки по корпусу пар.

    Метка токена — 1, если он попадает в размеченный фрагмент недостоверности.
    """
    samples: list[TokenSample] = []
    stats = {"pairs": 0, "skipped": 0, "tokens": 0, "positive": 0}

    for pair in pairs:
        answer = pair.get("answer", "")
        context = pair.get("context", "")
        tokens = tokenize_with_offsets(answer)
        if not tokens:
            stats["skipped"] += 1
            continue

        features = verifier.features_for(answer, context, tokens)
        risk = combine(
            features.attention_entropy,
            features.ctx_attention_mass,
            features.embedding_density,
            verifier.bundle.weights,
            entropy_quantile=verifier.entropy_quantile,
        )

        labels = [0] * len(tokens)
        for start, end, label in pair.get("labels", []):
            if int(label) != 1:
                continue
            for index, token in enumerate(tokens):
                if token.start < int(end) and token.end > int(start):
                    labels[index] = 1

        for index, token in enumerate(tokens):
            if not is_scored_token(token.text):
                continue
            samples.append(
                TokenSample(
                    pair_id=str(pair.get("id", stats["pairs"])),
                    index=index,
                    total=len(tokens),
                    text=token.text.strip(),
                    features={
                        "attention_entropy": features.attention_entropy[index],
                        "ctx_attention_mass": features.ctx_attention_mass[index],
                        "embedding_density": features.embedding_density[index],
                        "risk": risk[index],
                    },
                    label=labels[index],
                )
            )
            stats["positive"] += labels[index]
        stats["pairs"] += 1
        stats["tokens"] += len(tokens)

    return samples, stats


def auc_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Площадь под ROC-кривой (ранговая формула Манна — Уитни)."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[position]]:
            end += 1
        average_rank = (position + end) / 2 + 1
        for k in range(position, end + 1):
            ranks[order[k]] = average_rank
        position = end + 1
    rank_sum = sum(ranks[i] for i, label in enumerate(labels) if label == 1)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _token_metrics(labels: Sequence[int], flags: Sequence[bool], risk: Sequence[float]) -> dict[str, float]:
    tp = sum(1 for label, flag in zip(labels, flags, strict=False) if label and flag)
    fp = sum(1 for label, flag in zip(labels, flags, strict=False) if not label and flag)
    fn = sum(1 for label, flag in zip(labels, flags, strict=False) if label and not flag)
    tn = sum(1 for label, flag in zip(labels, flags, strict=False) if not label and not flag)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "auc": auc_score(labels, risk),
    }


@dataclass
class TrainReport:
    """Итог обучения."""

    bundle: WeightsBundle
    validation: dict[str, Any] = field(default_factory=dict)
    folds: list[dict[str, Any]] = field(default_factory=list)
    head: dict[str, Any] = field(default_factory=dict)
    participation: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    dataset: str = ""
    seed: int = 42

    def summary(self) -> str:
        validation = self.validation
        end_to_end = validation.get("end_to_end", {})
        served = end_to_end.get("tokens", {})
        return (
            f"веса={ {k: round(v, 2) for k, v in self.bundle.weights.items()} } "
            f"T(span): z={self.bundle.span_z} floor={self.bundle.span_floor} cap={self.bundle.span_cap} "
            f"порог_ответа={self.bundle.threshold:.4f} "
            f"token F1={validation.get('f1', 0):.3f} FPR={validation.get('fpr', 0):.3f} "
            f"AUC={validation.get('auc', 0):.3f} сигнал={validation.get('selection', {}).get('signal')} "
            f"| сквозной F1={served.get('f1', 0):.3f} FPR={served.get('fpr', 0):.3f} "
            f"| участие ИИ: AUC={self.participation.get('auc_out_of_fold', 0):.3f} "
            f"(калибровка: {self.participation.get('calibrated_on', '—')})"
        )


def _weight_grid(step: float = WEIGHT_GRID_STEP) -> list[dict[str, float]]:
    grid: list[dict[str, float]] = []
    total_units = int(round(1 / step))
    for units_w1 in range(total_units + 1):
        for units_w2 in range(total_units + 1 - units_w1):
            units_w3 = total_units - units_w1 - units_w2
            weights = {
                "attention_entropy": units_w1 * step,
                "ctx_attention_mass": units_w2 * step,
                "embedding_density": units_w3 * step,
            }
            if abs(sum(weights.values()) - 1.0) < 1e-9:
                grid.append(weights)
    return grid


def _group_key(pair: dict) -> tuple[str, str]:
    """Ключ группы для честного разделения: субъект + шаблон/вопрос.

    Без группировки один и тот же субъект (и тот же шаблон вопроса) попадает и в
    обучение, и в «отложенную» часть, и метрика перестаёт быть отложенной.
    """
    meta = pair.get("meta") or {}
    subject = str(meta.get("subject") or "")
    template = str(meta.get("question") or meta.get("template") or "")
    return (subject, template)


def _split_pairs(pairs: Sequence[dict], test_size: float, seed: int, group: bool = True) -> tuple[list, list]:
    """Разделить пары на обучающую и отложенную части.

    По умолчанию — групповое разделение по ``(subject, шаблон)``: ни один субъект
    и ни один шаблон не встречаются в обеих частях. ``group=False`` оставлен для
    сравнения и старых сценариев (построчный шаффл).
    """
    ordered = list(pairs)
    rng = random.Random(seed)
    if group:
        groups: dict[tuple[str, str], list[dict]] = {}
        for pair in ordered:
            groups.setdefault(_group_key(pair), []).append(pair)
        keys = sorted(groups)
        rng.shuffle(keys)
        cut = max(1, int(round(len(keys) * (1 - test_size))))
        train_keys = set(keys[:cut])
        train = [pair for pair in ordered if _group_key(pair) in train_keys]
        test = [pair for pair in ordered if _group_key(pair) not in train_keys]
        if train and test:
            return train, test
    rng.shuffle(ordered)
    cut = max(1, int(len(ordered) * (1 - test_size)))
    return ordered[:cut], ordered[cut:]


def shared_groups(train_pairs: Sequence[dict], test_pairs: Sequence[dict]) -> int:
    """Сколько групп встречается и в обучении, и в отложенной части (для отчёта)."""
    return len({_group_key(pair) for pair in train_pairs} & {_group_key(pair) for pair in test_pairs})


def _risk_for(
    samples: Sequence[TokenSample],
    weights: dict[str, float],
    entropy_quantile: float = 1.0,
) -> list[float]:
    """Пересчитать риск по уже собранным признакам (без повторного расчёта).

    ``entropy_quantile`` обязан совпадать с тем, что использует ``Verifier``:
    иначе голова обучается на одном масштабе риска, а API считает другой.
    """
    entropy = [sample.features["attention_entropy"] for sample in samples]
    mass = [sample.features["ctx_attention_mass"] for sample in samples]
    density = [sample.features["embedding_density"] for sample in samples]
    return combine(entropy, mass, density, weights, entropy_quantile=entropy_quantile)


def _score_pairs(
    verifier: Verifier,
    pairs: Sequence[dict],
    weights: dict[str, float],
    span_z: float,
    span_floor: float,
    span_cap: float,
    samples_by_pair: dict[str, list[TokenSample]],
) -> dict[str, Any]:
    """Оценить качество правил на парах: токены, фрагменты, ответы."""
    labels: list[int] = []
    flags: list[bool] = []
    risks: list[float] = []
    raw_scores: list[float] = []
    answer_labels: list[int] = []
    span_predicted: list[tuple[int, int]] = []
    span_truth: list[tuple[int, int]] = []

    for pair in pairs:
        pair_id = str(pair.get("id"))
        samples = samples_by_pair.get(pair_id, [])
        if not samples:
            continue
        risks_by_index = {
            sample.index: risk for sample, risk in zip(samples, _risk_for(samples, weights), strict=False)
        }
        labels_by_index = {sample.index: sample.label for sample in samples}
        max_index = max(samples, key=lambda sample: sample.index).index
        risk_sequence = [risks_by_index.get(index, 0.0) for index in range(max_index + 1)]
        smoothed = _smooth(risk_sequence, SMOOTH_WINDOW)
        threshold = span_threshold_for(smoothed, span_z, span_floor, span_cap)

        for index in risks_by_index:
            flags.append(smoothed[index] >= threshold)
            labels.append(labels_by_index[index])
            risks.append(smoothed[index])

        raw = _answer_score(smoothed)
        raw_scores.append(raw)
        is_positive = any(sample.label == 1 for sample in samples)
        answer_labels.append(1 if is_positive else 0)

        # Фрагмент-уровень: объединяем соседние помеченные токены.
        flagged_indices = [index for index, value in enumerate(smoothed) if value >= threshold]
        if flagged_indices:
            span_predicted.append((min(flagged_indices), max(flagged_indices) + 1))
        truth_indices = [sample.index for sample in samples if sample.label == 1]
        if truth_indices:
            span_truth.append((min(truth_indices), max(truth_indices) + 1))

    metrics = _token_metrics(labels, flags, risks)
    metrics["answer_auc"] = auc_score(answer_labels, raw_scores)
    metrics["span_predicted"] = len(span_predicted)
    metrics["span_truth"] = len(span_truth)
    metrics["span_f1"] = _span_f1_from_indices(span_predicted, span_truth, iou_threshold=0.5)
    metrics["raw_scores"] = raw_scores
    metrics["answer_labels"] = answer_labels
    return metrics


def _span_f1_from_indices(
    predicted: Sequence[tuple[int, int]],
    truth: Sequence[tuple[int, int]],
    iou_threshold: float = 0.5,
) -> float:
    """F1 по фрагментам в токенных индексах (жадное сопоставление с IoU ≥ 0.5)."""
    tp = fp = 0
    unmatched = list(truth)
    for guess in predicted:
        best_index, best_iou = -1, 0.0
        for index, gold in enumerate(unmatched):
            intersection = max(0, min(guess[1], gold[1]) - max(guess[0], gold[0]))
            if not intersection:
                continue
            union = (guess[1] - guess[0]) + (gold[1] - gold[0]) - intersection
            overlap = intersection / union if union else 0.0
            if overlap > best_iou:
                best_index, best_iou = index, overlap
        if best_index >= 0 and best_iou >= iou_threshold:
            unmatched.pop(best_index)
            tp += 1
        else:
            fp += 1
    fn = len(unmatched)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


# ---------------------------------------------------------------- голова


def _head_rows(samples: Sequence[TokenSample], risks: Sequence[float], with_label: bool = True):
    rows: list[list[float]] = []
    labels: list[int] = []
    for sample, risk in zip(samples, risks, strict=False):
        rows.append(
            [
                sample.features["attention_entropy"],
                sample.features["ctx_attention_mass"],
                sample.features["embedding_density"],
                risk,
                sample.index / max(1, sample.total),
                min(1.0, len(sample.text) / 20),
            ]
        )
        labels.append(sample.label)
    return (rows, labels) if with_label else rows


# ---------------------------------------------------------------- основной вход


def _score_pairs(
    verifier: Verifier,
    pairs: Sequence[dict],
    risk_fn,
    span_params: Sequence[float],
    samples_by_pair: dict[str, list[TokenSample]],
) -> dict[str, Any]:
    """Оценить правила на парах: токены, фрагменты, ответы.

    ``risk_fn`` — функция «список токенных сэмплов → список рисков». Она
    передаётся снаружи, чтобы порог маски подбирался ровно на том сигнале,
    который будет отдавать API (правило или обученная голова).
    """
    labels: list[int] = []
    flags: list[bool] = []
    risks: list[float] = []
    raw_scores: list[float] = []
    answer_labels: list[int] = []
    span_predicted: list[tuple[int, int]] = []
    span_truth: list[tuple[int, int]] = []

    for pair in pairs:
        pair_id = str(pair.get("id"))
        samples = samples_by_pair.get(pair_id, [])
        if not samples:
            continue
        values = risk_fn(samples)
        risks_by_index = {sample.index: value for sample, value in zip(samples, values, strict=False)}
        max_index = max(sample.index for sample in samples)
        sequence = [0.0] * (max_index + 1)
        for sample in samples:
            sequence[sample.index] = risks_by_index[sample.index]
        smoothed = _smooth(sequence, SMOOTH_WINDOW)
        threshold = span_threshold_for(smoothed, *span_params)

        for sample in samples:
            labels.append(sample.label)
            flags.append(smoothed[sample.index] >= threshold)
            risks.append(smoothed[sample.index])

        raw_scores.append(_answer_score([smoothed[sample.index] for sample in samples]))
        answer_labels.append(1 if any(sample.label == 1 for sample in samples) else 0)

        flagged_indices = [index for index, value in enumerate(smoothed) if value >= threshold]
        if flagged_indices:
            span_predicted.append((min(flagged_indices), max(flagged_indices) + 1))
        truth_indices = [sample.index for sample in samples if sample.label == 1]
        if truth_indices:
            span_truth.append((min(truth_indices), max(truth_indices) + 1))

    metrics = _token_metrics(labels, flags, risks)
    metrics["answer_auc"] = auc_score(answer_labels, raw_scores)
    metrics["span_predicted"] = len(span_predicted)
    metrics["span_truth"] = len(span_truth)
    metrics["span_f1"] = _span_f1_from_indices(span_predicted, span_truth, iou_threshold=0.5)
    metrics["raw_scores"] = raw_scores
    metrics["answer_labels"] = answer_labels
    return metrics


def _select_mask(
    verifier: Verifier,
    pairs: Sequence[dict],
    risk_fn,
    samples_by_pair: dict[str, list[TokenSample]],
    target_fpr: float,
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    """Подобрать z/floor/cap маски: максимум F1 при FPR ≤ target (на обучающей части)."""
    best_params = (1.0, 0.35, 0.6)
    best_key: tuple[float, float, float] | None = None
    best_metrics: dict[str, Any] = {}
    for span_z in Z_GRID:
        for floor in FLOOR_GRID:
            for cap in CAP_GRID:
                if floor >= cap:
                    continue
                metrics = _score_pairs(verifier, pairs, risk_fn, (span_z, floor, cap), samples_by_pair)
                if metrics["fpr"] > target_fpr:
                    continue
                key = (metrics["f1"], -metrics["fpr"], metrics["recall"])
                if best_key is None or key > best_key:
                    best_key, best_params, best_metrics = key, (span_z, floor, cap), metrics
    return best_params, best_metrics


def _head_scores(samples: Sequence[TokenSample], payload: dict[str, Any], rule_risks: Sequence[float]) -> list[float]:
    """Вероятности головы по сэмплам (та же арифметика, что в движке)."""
    model = (payload or {}).get("model") or {}
    weights = model.get("weights")
    if not weights:
        return list(rule_risks)
    means = payload.get("scaler", {}).get("means", [])
    scales = payload.get("scaler", {}).get("scales", [])
    if len(weights) != len(HEAD_FEATURES) or len(means) != len(weights) or len(scales) != len(weights):
        # Артефакт не той размерности (например, обучен на другом наборе
        # признаков): честно откатываемся к правилу, а не падаем с IndexError.
        return list(rule_risks)

    out: list[float] = []
    for sample, rule_risk in zip(samples, rule_risks, strict=False):
        row = [
            sample.features["attention_entropy"],
            sample.features["ctx_attention_mass"],
            sample.features["embedding_density"],
            rule_risk,
            sample.index / max(1, sample.total),
            min(1.0, len(sample.text) / 20),
        ]
        normalized = [(value - means[j]) / (scales[j] or 1.0) for j, value in enumerate(row)]
        score = model.get("bias", 0.0) + sum(w * x for w, x in zip(weights, normalized, strict=False))
        out.append(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, score)))))
    return out


def train(
    pairs: Sequence[dict],
    mode: str = "demo",
    seed: int = 42,
    target_fpr: float = 0.1,
    test_size: float = 0.3,
    folds: int = 5,
    dataset_name: str = "",
    verifier: Verifier | None = None,
    version: str = __version__,
    group_split: bool = True,
) -> TrainReport:
    """Обучить веса, пороги, голову и калибровку; вернуть отчёт."""
    verifier = verifier or Verifier(
        mode=mode, weights=WeightsBundle(weights=dict(DEFAULT_WEIGHTS), threshold=0.5, mode=mode)
    )
    train_pairs, test_pairs = _split_pairs(pairs, test_size=test_size, seed=seed, group=group_split)

    train_samples, train_stats = collect_samples(verifier, train_pairs)
    test_samples, test_stats = collect_samples(verifier, test_pairs)
    if not train_samples or not test_samples:
        raise ValueError("недостаточно данных: проверьте корпус и разметку")

    train_by_pair: dict[str, list[TokenSample]] = {}
    for sample in train_samples:
        train_by_pair.setdefault(sample.pair_id, []).append(sample)
    test_by_pair: dict[str, list[TokenSample]] = {}
    for sample in test_samples:
        test_by_pair.setdefault(sample.pair_id, []).append(sample)

    # Квантиль энтропии берётся у верификатора: обучение и API обязаны
    # нормировать риск одинаково, иначе голова видит не тот признак, который
    # потом получит на входе.
    entropy_quantile = verifier.entropy_quantile

    # 1. Веса признаков — по AUC на обучающей части (перебор сетки, сумма = 1).
    best_weights = dict(DEFAULT_WEIGHTS)
    best_auc = -1.0
    weight_table: list[dict[str, float]] = []
    for weights in _weight_grid():
        risks = _risk_for(train_samples, weights, entropy_quantile=entropy_quantile)
        auc = auc_score([sample.label for sample in train_samples], risks)
        weight_table.append({**weights, "auc": auc})
        if not math.isnan(auc) and auc > best_auc:
            best_auc, best_weights = auc, weights

    def rule_risk(samples: Sequence[TokenSample]) -> list[float]:
        return _risk_for(samples, best_weights, entropy_quantile=entropy_quantile)

    # 2. Голова: логистическая регрессия, 5-кратная кросс-валидация по парам.
    head_report = _train_and_compare_head(
        train_samples,
        test_samples,
        best_weights,
        folds,
        seed,
        entropy_quantile=entropy_quantile,
    )
    head_model = head_report.get("payload") or {}

    def head_risk_fn(samples: Sequence[TokenSample]) -> list[float]:
        return _head_scores(samples, head_model, rule_risk(samples))

    # 3. Порог маски — на том сигнале, который пойдёт в API. Сравниваем правило и
    #    голову на обучающей части (тест не используется для выбора!), победитель
    #    определяется по F1 при ограничении FPR.
    rule_params, rule_metrics = _select_mask(verifier, train_pairs, rule_risk, train_by_pair, target_fpr)
    candidate = {
        "signal": "rule",
        "params": rule_params,
        "metrics": rule_metrics,
        "risk_fn": rule_risk,
    }
    if head_report.get("type") == "logreg":
        head_params, head_metrics = _select_mask(verifier, train_pairs, head_risk_fn, train_by_pair, target_fpr)
        head_report["head_best_f1"] = head_metrics.get("f1", 0.0)
        head_report["head_best_fpr"] = head_metrics.get("fpr", 1.0)
        if (head_metrics.get("f1", 0.0), -head_metrics.get("fpr", 1.0)) > (
            rule_metrics.get("f1", 0.0),
            -rule_metrics.get("fpr", 1.0),
        ):
            candidate = {
                "signal": "logreg",
                "params": head_params,
                "metrics": head_metrics,
                "risk_fn": head_risk_fn,
            }
    head_report["selected"] = candidate["signal"]
    head_report["rule_best_f1"] = rule_metrics.get("f1", 0.0)
    head_report["rule_best_fpr"] = rule_metrics.get("fpr", 1.0)
    best_span = candidate["params"]
    risk_fn = candidate["risk_fn"]

    # 4. Калибровка оценки ответа (PAVA) и порог по целевому FPR — на обучающей части.
    train_metrics = _score_pairs(verifier, train_pairs, risk_fn, best_span, train_by_pair)
    calibrator = IsotonicCalibrator.fit(train_metrics["raw_scores"], train_metrics["answer_labels"])
    calibrated = calibrator.transform(train_metrics["raw_scores"])
    threshold_report = choose_threshold_report(calibrated, train_metrics["answer_labels"], max_fpr=target_fpr)
    threshold = float(threshold_report["threshold"])
    if not threshold_report.get("constraint_met", True):
        # Молчаливый порог (ничего не помечать) почти всегда укладывается в FPR,
        # поэтому сюда попадаем только если ограничение невыполнимо совсем.
        # Тогда берём максимум F1 и пишем фактический FPR, а не притворяемся,
        # что целевой FPR достигнут.
        print(
            f"ВНИМАНИЕ: целевой FPR {target_fpr} на обучающей части недостижим; "
            f"порог выбран по максимуму F1 ({threshold_report['f1']:.3f}), "
            f"фактический FPR {threshold_report['fpr']:.3f}",
            flush=True,
        )

    validation = _score_pairs(verifier, test_pairs, risk_fn, best_span, test_by_pair)
    validation_at = metrics_at(calibrator.transform(validation["raw_scores"]), validation["answer_labels"], threshold)

    # Модель встраивается в бандл, а не только путь к файлу. Иначе оценка
    # читает config/head.json из поставки (демо-голова) и применяет её к
    # признакам другой модели: в прогоне 37472951524 из-за этого token F1
    # оказался 0,013 при AUC головы 0,85. Файл по-прежнему пишется — для
    # перезапуска без бандла в памяти.
    if candidate["signal"] == "logreg" and head_report.get("model"):
        saved_head = {
            "type": "logreg",
            "file": "config/head.json",
            "model": head_report["model"],
            "scaler": head_report.get("scaler") or {},
        }
    else:
        saved_head = {"type": "none", "file": None}
    bundle = WeightsBundle(
        weights=best_weights,
        threshold=float(threshold),
        target_fpr=target_fpr,
        span_z=best_span[0],
        span_floor=best_span[1],
        span_cap=best_span[2],
        isotonic=calibrator,
        head=saved_head,
        folds=head_report.get("folds", []),
        seed=seed,
        version=version,
        mode=mode,
        meta={
            "dataset": dataset_name or "inline",
            "synthetic": mode == "demo",
            "signal": candidate["signal"],
            # Чтобы недостигнутое ограничение по FPR было видно в отчёте, а не
            # только в логе: иначе порог 1,0 выглядит как «модель ничего не нашла».
            "threshold_report": {
                key: (float(value) if isinstance(value, (int, float)) else value)
                for key, value in threshold_report.items()
            },
            "note": (
                "Демонстрационный корпус: параметры проверяют конвейер, а не качество "
                "на реальных документах. Для боевой калибровки используйте размеченные "
                "реальные пары в режиме hf."
            ),
        },
    )

    # 5. Оценка доли участия ИИ (требование заявки): отдельная голова на том же
    #    сигнале. Корпус синтетический — это фиксируется в calibrated_on.
    participation = _fit_participation(verifier, seed=seed, version=version)
    bundle.meta["participation"] = {
        "type": "participation-logreg",
        "auc_out_of_fold": participation.get("auc_out_of_fold", 0.0),
        "calibrated_on": participation.get("calibrated_on", ""),
    }

    # 6. Сквозная проверка: те же параметры, но через публичный verify() — то, что
    #    реально увидит пользователь API. Числа берутся из настоящего пути.
    end_to_end = Verifier(mode=mode, weights=bundle).evaluate(test_pairs)

    report = TrainReport(
        bundle=bundle,
        validation={
            "tokens": _token_metrics_from_samples(test_by_pair, risk_fn, best_span),
            "precision": validation["precision"],
            "recall": validation["recall"],
            "f1": validation["f1"],
            "fpr": validation["fpr"],
            "auc": validation["auc"],
            "tp": validation["tp"],
            "fp": validation["fp"],
            "fn": validation["fn"],
            "tn": validation["tn"],
            "span_f1": validation["span_f1"],
            "answer_auc": validation["answer_auc"],
            "answer_threshold": threshold,
            "answer_precision": validation_at["precision"],
            "answer_recall": validation_at["recall"],
            "answer_f1": validation_at["f1"],
            "answer_fpr": validation_at["fpr"],
            "end_to_end": end_to_end,
            "selection": {
                "signal": candidate["signal"],
                "rule_f1_train": rule_metrics.get("f1", 0.0),
                "head_f1_train": head_report.get("head_best_f1", 0.0),
            },
        },
        folds=head_report.get("folds", []),
        head=head_report,
        participation=participation,
        stats={
            "train": train_stats,
            "test": test_stats,
            "split": {
                "grouped": group_split,
                "train_pairs": len(train_pairs),
                "test_pairs": len(test_pairs),
                "shared_groups": shared_groups(train_pairs, test_pairs),
            },
            "weights_table": sorted(weight_table, key=lambda row: -row["auc"])[:10],
        },
        dataset=dataset_name,
        seed=seed,
    )
    return report


def _token_metrics_from_samples(
    samples_by_pair: dict[str, list[TokenSample]], risk_fn, span_params: Sequence[float]
) -> dict[str, float]:
    """Метрики по токенам тестовой части ровно по правилам движка (без расширения)."""
    labels: list[int] = []
    flags: list[bool] = []
    risks: list[float] = []
    for samples in samples_by_pair.values():
        values = risk_fn(samples)
        sequence = [0.0] * (max(sample.index for sample in samples) + 1)
        for sample, value in zip(samples, values, strict=False):
            sequence[sample.index] = value
        smoothed = _smooth(sequence, SMOOTH_WINDOW)
        threshold = span_threshold_for(smoothed, *span_params)
        for sample in samples:
            labels.append(sample.label)
            flags.append(smoothed[sample.index] >= threshold)
            risks.append(smoothed[sample.index])
    return _token_metrics(labels, flags, risks)


def _train_and_compare_head(
    train_samples: Sequence[TokenSample],
    test_samples: Sequence[TokenSample],
    weights: dict[str, float],
    folds: int,
    seed: int,
    entropy_quantile: float = 1.0,
) -> dict[str, Any]:
    """Обучить голову с кросс-валидацией и сравнить её с пороговым правилом."""
    train_risks = _risk_for(train_samples, weights, entropy_quantile=entropy_quantile)
    test_risks = _risk_for(test_samples, weights, entropy_quantile=entropy_quantile)
    train_rows, train_labels = _head_rows(train_samples, train_risks)
    test_rows, test_labels = _head_rows(test_samples, test_risks)
    train_scaled, means, scales = _standardize(train_rows)
    test_scaled = [[(row[j] - means[j]) / (scales[j] or 1.0) for j in range(len(row))] for row in test_rows]

    fold_reports: list[dict[str, Any]] = []
    rng = random.Random(seed)
    order = list(range(len(train_scaled)))
    rng.shuffle(order)
    fold_count = max(2, folds)
    fold_size = max(1, len(order) // fold_count)
    out_of_fold = [0.0] * len(train_scaled)
    for fold in range(fold_count):
        start_index = fold * fold_size
        stop_index = len(order) if fold == fold_count - 1 else start_index + fold_size
        test_idx = order[start_index:stop_index]
        test_set = set(test_idx)
        train_idx = [index for index in order if index not in test_set]
        if not train_idx or not test_idx:
            continue
        model = _train_logreg([train_scaled[i] for i in train_idx], [train_labels[i] for i in train_idx])
        for index, probability in zip(
            test_idx, _head_probabilities(model, [train_scaled[i] for i in test_idx]), strict=False
        ):
            out_of_fold[index] = probability
        fold_reports.append(
            {
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "auc": auc_score([train_labels[i] for i in test_idx], [out_of_fold[i] for i in test_idx]),
            }
        )

    final_model = _train_logreg(train_scaled, train_labels)
    head_auc = auc_score(train_labels, out_of_fold)
    rule_auc = auc_score(train_labels, train_risks)

    payload = {
        "type": "logreg",
        "features": list(HEAD_FEATURES),
        "scaler": {"means": means, "scales": scales},
        "model": final_model,
        "auc_out_of_fold": head_auc,
        "auc_rule": rule_auc,
        "auc_test": auc_score(test_labels, _head_probabilities(final_model, test_scaled)),
        "folds": fold_reports,
    }
    return {
        "type": "logreg",
        "saved": {"type": "logreg", "file": "config/head.json"},
        "auc_out_of_fold": head_auc,
        "auc_rule": rule_auc,
        "auc_test": payload["auc_test"],
        "folds": fold_reports,
        "scaler": payload["scaler"],
        "model": final_model,
        "features": list(HEAD_FEATURES),
        "payload": payload,
    }


def _fit_participation(verifier: Verifier, seed: int, version: str) -> dict[str, Any]:
    """Обучить оценку доли участия ИИ на синтетическом корпусе двух стилей."""
    samples = build_participation_corpus(count=240, seed=seed + 1985)
    rows, labels = corpus_rows_and_labels(verifier, samples)
    if not rows or len(set(labels)) < 2:
        return {"type": "none", "payload": None, "auc_out_of_fold": 0.0, "calibrated_on": ""}
    model = ParticipationModel.fit(rows, labels, seed=seed, version=version)
    return {
        "type": "participation-logreg",
        "payload": model.to_dict(),
        "auc_out_of_fold": model.auc_out_of_fold,
        "calibrated_on": model.calibrated_on,
        "rows": len(rows),
    }


def save_training_artifacts(report: TrainReport, weights_path: str | Path, root: str | Path = ".") -> dict[str, Path]:
    """Сохранить weights.json (и head.json, если голова победила)."""
    weights_path = Path(weights_path)
    report.bundle.save(weights_path)

    written: dict[str, Path] = {"weights": weights_path}
    head = report.head
    if report.bundle.head.get("type") == "logreg" and head.get("payload"):
        head_path = Path(root) / "config" / "head.json"
        head_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **head["payload"],
            "version": report.bundle.version,
            "seed": report.seed,
        }
        head_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written["head"] = head_path
    participation = report.participation
    if participation.get("payload"):
        part_path = Path(root) / "config" / "participation.json"
        part_path.parent.mkdir(parents=True, exist_ok=True)
        part_payload = {
            **participation["payload"],
            "version": report.bundle.version,
            "seed": report.seed,
        }
        part_path.write_text(json.dumps(part_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written["participation"] = part_path
    return written


def load_head(path: str | Path) -> dict[str, Any] | None:
    target = Path(path)
    if not target.is_file():
        return None
    return json.loads(target.read_text(encoding="utf-8"))
