"""Out-of-fold (OOF) обучение модели для P0-2.

Реализует корректный OOF-протокол:
1. Разбиение на K фолдов по документам
2. Для каждого фолда: обучение модели (веса + голова) на остальных фолдах
3. Получение raw-рисков на held-out фолде
4. Сбор всех OOF-предсказаний
5. Обучение answer-калибратора на OOF-предсказаниях

Это гарантирует, что held-out документы не влияют на обучение модели.
"""
from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .calibration import IsotonicCalibrator, choose_threshold_report
from .engine import SMOOTH_WINDOW, Verifier, WeightsBundle, _answer_score, _smooth
from .features import DEFAULT_WEIGHTS, FEATURE_NAMES
from .logreg import probabilities as head_probabilities, standardize, train_logreg
from .train import (
    TokenSample,
    _head_rows,
    _head_scores,
    _risk_for,
    _weight_grid,
    auc_score,
    collect_samples,
)
from .answer_head import extract_answer_features


def answer_label(pair: dict) -> int:
    """Определить метку ответа из исходной разметки.
    
    Ответ считается недостоверным (label=1), если есть хотя бы один
    размеченный фрагмент с label==1 в исходной разметке.
    
    Использует исходные labels из pair, а не токены, чтобы избежать
    потерь из-за scored_token фильтра (например, односимвольные числа).
    """
    return int(any(int(row[2]) == 1 for row in pair.get("labels", [])))

@dataclass
class OOFResult:
    """Результат OOF-обучения."""
    
    oof_scores: list[float] = field(default_factory=list)
    oof_labels: list[int] = field(default_factory=list)
    oof_pair_ids: list[str] = field(default_factory=list)
    oof_doc_ids: list[str] = field(default_factory=list)
    oof_folds: list[int] = field(default_factory=list)
    
    fold_receipts: list[dict[str, Any]] = field(default_factory=list)
    
    final_weights: dict[str, float] = field(default_factory=dict)
    final_head: dict[str, Any] = field(default_factory=dict)
    final_head_hash: str = ""
    
    shared_documents: int = 0


def _compute_head_hash(head: dict[str, Any]) -> str:
    """Вычислить хеш головы для идентификации."""
    canonical = {
        "type": head.get("type"),
        "weights": head.get("model", {}).get("weights", []),
        "bias": head.get("model", {}).get("bias", 0.0),
        "scaler_means": head.get("scaler", {}).get("means", []),
        "scaler_scales": head.get("scaler", {}).get("scales", []),
    }
    serialized = str(sorted(canonical.items()))
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def _train_model_on_subset(
    samples: list[TokenSample],
    entropy_quantile: float = 1.0,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Обучить модель (веса + голову) на подмножестве samples.
    
    Returns:
        (best_weights, head_payload)
    """
    if not samples:
        raise ValueError("Cannot train on empty samples")
    
    # 1. Подбор весов признаков по AUC
    best_weights = dict(DEFAULT_WEIGHTS)
    best_auc = -1.0
    
    for weights in _weight_grid():
        risks = _risk_for(samples, weights, entropy_quantile=entropy_quantile)
        auc = auc_score([s.label for s in samples], risks)
        if not math.isnan(auc) and auc > best_auc:
            best_auc = auc
            best_weights = weights
    
    # 2. Обучение головы (logreg)
    risks = _risk_for(samples, best_weights, entropy_quantile=entropy_quantile)
    rows, labels = _head_rows(samples, risks)
    scaled, means, scales = standardize(rows)
    model = train_logreg(scaled, labels)
    
    head_payload = {
        "type": "logreg",
        "features": ["attention_entropy", "ctx_attention_mass", "embedding_density", "risk", "position", "length"],
        "scaler": {"means": means, "scales": scales},
        "model": model,
    }
    
    return best_weights, head_payload


def _get_risks_with_head(
    samples: list[TokenSample],
    weights: dict[str, float],
    head: dict[str, Any],
    entropy_quantile: float = 1.0,
) -> list[float]:
    """Получить риски с применением головы."""
    rule_risks = _risk_for(samples, weights, entropy_quantile=entropy_quantile)
    return _head_scores(samples, head, rule_risks)


def _aggregate_answer_scores(
    samples_by_pair: dict[str, list[TokenSample]],
    risks: list[float],
    sample_indices: list[int],
    aggregator: str = "current",
) -> list[float]:
    """Агрегировать токенные риски в answer scores.
    
    Args:
        samples_by_pair: samples, сгруппированные по pair_id
        risks: flat list рисков для всех samples
        sample_indices: индексы, указывающие, к какому sample относится каждый risk
        aggregator: тип агрегатора ("current", "max", "mean_top3")
    
    Returns:
        list of answer scores (один на пару)
    """
    # Группируем риски по парам
    risks_by_pair: dict[str, list[float]] = defaultdict(list)
    
    for risk, idx in zip(risks, sample_indices, strict=False):
        # Нужно найти pair_id для этого sample
        # Для этого нам нужно пройтись по samples_by_pair и сопоставить индексы
        pass
    
    # На самом деле, нам нужна другая структура данных
    # Давайте переделаем: принимаем dict[pair_id -> list[risk]]
    raise NotImplementedError("Use _aggregate_from_dict instead")


def _aggregate_from_dict(
    risks_by_pair: dict[str, list[float]],
    aggregator: str = "current",
) -> dict[str, float]:
    """Агрегировать токенные риски в answer scores из dict.
    
    Args:
        risks_by_pair: dict[pair_id -> list[smoothed_risk_for_scored_tokens]]
        aggregator: тип агрегатора
    
    Returns:
        dict[pair_id -> answer_score]
    """
    scores = {}
    
    for pair_id, risks in risks_by_pair.items():
        if not risks:
            scores[pair_id] = 0.0
            continue
        
        if aggregator == "current":
            # 0.5 * mean + 0.5 * mean(top 20%)
            ordered = sorted(risks, reverse=True)
            top_count = max(1, int(len(ordered) * 0.2))
            scores[pair_id] = 0.5 * (sum(risks) / len(risks)) + 0.5 * (sum(ordered[:top_count]) / len(ordered[:top_count]))
        
        elif aggregator == "max":
            scores[pair_id] = max(risks)
        
        elif aggregator == "mean_top3":
            ordered = sorted(risks, reverse=True)
            top3 = ordered[:3]
            scores[pair_id] = sum(top3) / len(top3)
        
        else:
            raise ValueError(f"Unknown aggregator: {aggregator}")
    
    return scores


def train_oof_model(
    verifier: Verifier,
    train_pairs: list[dict],
    n_folds: int = 5,
    seed: int = 42,
) -> OOFResult:
    """Обучить модель с OOF-протоколом.
    
    Args:
        verifier: верификатор с HF-бандлом
        train_pairs: пары для обучения (official train)
        n_folds: количество фолдов
        seed: seed для воспроизводимости
    
    Returns:
        OOFResult с OOF-предсказаниями и финальной моделью
    """
    # 1. Группируем пары по документам
    doc_groups: dict[str, list[int]] = defaultdict(list)
    doc_id_by_pair: dict[str, str] = {}
    
    for i, pair in enumerate(train_pairs):
        pair_id = str(pair.get("id", i))
        meta = pair.get("meta", {})
        doc_id = meta.get("doc_id") or meta.get("document_id") or meta.get("group")
        if not doc_id:
            raise ValueError(f"Pair {pair_id} missing doc_id in meta")
        doc_groups[doc_id].append(i)
        doc_id_by_pair[pair_id] = doc_id
    
    # 2. Создаём фолды по документам
    doc_ids = list(doc_groups.keys())
    rng = random.Random(seed)
    rng.shuffle(doc_ids)
    
    fold_size = len(doc_ids) // n_folds
    folds: list[list[int]] = []  # folds[fold_idx] = list of pair indices
    
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(doc_ids)
        fold_doc_ids = doc_ids[start:end]
        fold_indices = []
        for doc_id in fold_doc_ids:
            fold_indices.extend(doc_groups[doc_id])
        folds.append(fold_indices)
    
    # 3. OOF-обучение
    result = OOFResult()
    
    # Собираем samples для всех пар заранее (один раз)
    all_samples, _ = collect_samples(verifier, train_pairs)
    samples_by_pair: dict[str, list[TokenSample]] = {}
    for sample in all_samples:
        samples_by_pair.setdefault(sample.pair_id, []).append(sample)
    
    for fold_idx in range(n_folds):
        # Определяем train и test для этого фолда
        test_indices = set(folds[fold_idx])
        train_indices = []
        for i in range(n_folds):
            if i != fold_idx:
                train_indices.extend(folds[i])
        
        # Получаем samples для train и test
        train_samples = []
        for idx in train_indices:
            pair_id = str(train_pairs[idx].get("id", idx))
            train_samples.extend(samples_by_pair.get(pair_id, []))
        
        test_samples = []
        for idx in test_indices:
            pair_id = str(train_pairs[idx].get("id", idx))
            test_samples.extend(samples_by_pair.get(pair_id, []))
        
        # Обучаем модель на train
        weights, head = _train_model_on_subset(train_samples, entropy_quantile=verifier.entropy_quantile)
        
        # Получаем риски на test
        test_risks = _get_risks_with_head(test_samples, weights, head, entropy_quantile=verifier.entropy_quantile)
        
        # Группируем риски по парам и применяем сглаживание
        test_samples_by_pair: dict[str, list[TokenSample]] = {}
        for sample in test_samples:
            test_samples_by_pair.setdefault(sample.pair_id, []).append(sample)
        
        risks_by_pair: dict[str, list[float]] = {}
        risk_idx = 0
        for pair_id, samples in test_samples_by_pair.items():
            pair_risks = []
            for _ in samples:
                pair_risks.append(test_risks[risk_idx])
                risk_idx += 1
            
            # Сглаживание
            max_index = max(s.index for s in samples)
            risk_by_index = {s.index: r for s, r in zip(samples, pair_risks, strict=False)}
            risk_sequence = [risk_by_index.get(i, 0.0) for i in range(max_index + 1)]
            smoothed = _smooth(risk_sequence, SMOOTH_WINDOW)
            
            # Берём только scored токены
            scored_risks = [smoothed[s.index] for s in samples if s.index < len(smoothed)]
            risks_by_pair[pair_id] = scored_risks
        
        # Сохраняем OOF-предсказания
        for idx in test_indices:
            pair_id = str(train_pairs[idx].get("id", idx))
            doc_id = doc_id_by_pair[pair_id]
            label = answer_label(train_pairs[idx])
            # Для answer score нужно агрегировать, но пока сохраним сырые риски
            # Агрегация будет выполнена позже для каждого агрегатора
            result.oof_pair_ids.append(pair_id)
            result.oof_doc_ids.append(doc_id)
            result.oof_labels.append(label)
            result.oof_folds.append(fold_idx)
            # scores будут добавлены позже
        
        # Сохраняем receipt для этого фолда
        train_doc_ids = set()
        for idx in train_indices:
            pair_id = str(train_pairs[idx].get("id", idx))
            train_doc_ids.add(doc_id_by_pair[pair_id])
        
        test_doc_ids = set()
        for idx in test_indices:
            pair_id = str(train_pairs[idx].get("id", idx))
            test_doc_ids.add(doc_id_by_pair[pair_id])
        
        receipt = {
            "fold": fold_idx,
            "n_train_pairs": len(train_indices),
            "n_test_pairs": len(test_indices),
            "n_train_docs": len(train_doc_ids),
            "n_test_docs": len(test_doc_ids),
            "train_doc_ids_hash": hashlib.sha256(str(sorted(train_doc_ids)).encode()).hexdigest()[:16],
            "test_doc_ids_hash": hashlib.sha256(str(sorted(test_doc_ids)).encode()).hexdigest()[:16],
            "shared_documents": len(train_doc_ids & test_doc_ids),
            "weights_hash": hashlib.sha256(str(sorted(weights.items())).encode()).hexdigest()[:16],
            "head_hash": _compute_head_hash(head),
        }
        result.fold_receipts.append(receipt)
        
        # Проверяем, что нет пересечений
        assert receipt["shared_documents"] == 0, f"Fold {fold_idx} has shared documents!"
    
    # 4. Обучаем финальную модель на всём train
    final_weights, final_head = _train_model_on_subset(all_samples, entropy_quantile=verifier.entropy_quantile)
    result.final_weights = final_weights
    result.final_head = final_head
    result.final_head_hash = _compute_head_hash(final_head)
    
    # 5. Проверяем shared_documents = 0 между всеми фолдами
    all_test_doc_sets = []
    for fold_idx in range(n_folds):
        test_indices = set(folds[fold_idx])
        test_doc_ids = set()
        for idx in test_indices:
            pair_id = str(train_pairs[idx].get("id", idx))
            test_doc_ids.add(doc_id_by_pair[pair_id])
        all_test_doc_sets.append(test_doc_ids)
    
    # Проверяем попарно
    for i in range(len(all_test_doc_sets)):
        for j in range(i + 1, len(all_test_doc_sets)):
            shared = all_test_doc_sets[i] & all_test_doc_sets[j]
            result.shared_documents += len(shared)
    
    assert result.shared_documents == 0, "Found shared documents between folds!"
    
    return result


def compute_oof_risks(
    verifier: Verifier,
    train_pairs: list[dict],
    n_folds: int = 5,
    seed: int = 42,
) -> tuple[dict[str, list[float]], list[str], list[int], list[str], list[int]]:
    """Вычислить OOF raw token risks для всех пар.
    
    Обучает модель по фолдам один раз и возвращает OOF-риски для каждой пары.
    Эти риски потом можно агрегировать разными способами.
    
    Returns:
        (risks_by_pair, pair_ids, labels, doc_ids, folds)
    """
    # Группируем пары по документам
    doc_groups: dict[str, list[int]] = defaultdict(list)
    doc_id_by_pair: dict[str, str] = {}
    
    for i, pair in enumerate(train_pairs):
        pair_id = str(pair.get("id", i))
        meta = pair.get("meta", {})
        doc_id = meta.get("doc_id") or meta.get("document_id") or meta.get("group")
        doc_groups[doc_id].append(i)
        doc_id_by_pair[pair_id] = doc_id
    
    # Создаём фолды по документам
    doc_ids = list(doc_groups.keys())
    rng = random.Random(seed)
    rng.shuffle(doc_ids)
    
    fold_size = len(doc_ids) // n_folds
    folds: list[list[int]] = []
    
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(doc_ids)
        fold_doc_ids = doc_ids[start:end]
        fold_indices = []
        for doc_id in fold_doc_ids:
            fold_indices.extend(doc_groups[doc_id])
        folds.append(fold_indices)
    
    # Собираем samples для всех пар один раз
    all_samples, _ = collect_samples(verifier, train_pairs)
    samples_by_pair: dict[str, list[TokenSample]] = {}
    for sample in all_samples:
        samples_by_pair.setdefault(sample.pair_id, []).append(sample)
    
    # Для каждого фолда обучаем модель и получаем OOF-риски
    risks_by_pair: dict[str, list[float]] = {}
    pair_ids: list[str] = []
    labels: list[int] = []
    pair_doc_ids: list[str] = []
    pair_folds: list[int] = []
    
    import sys
    for fold_idx in range(n_folds):
        print(f"    Фолд {fold_idx + 1}/{n_folds}...", file=sys.stderr)
        sys.stderr.flush()
        
        # Определяем train и test для этого фолда
        test_indices = set(folds[fold_idx])
        train_indices = []
        for i in range(n_folds):
            if i != fold_idx:
                train_indices.extend(folds[i])
        
        # Обучаем модель на train
        train_samples = []
        for idx in train_indices:
            pair_id = str(train_pairs[idx].get("id", idx))
            train_samples.extend(samples_by_pair.get(pair_id, []))
        
        print(f"      Обучение модели на {len(train_samples)} samples...", file=sys.stderr)
        sys.stderr.flush()
        weights, head = _train_model_on_subset(train_samples, entropy_quantile=verifier.entropy_quantile)
        
        # Получаем риски на test
        test_samples = []
        for idx in test_indices:
            pair_id = str(train_pairs[idx].get("id", idx))
            test_samples.extend(samples_by_pair.get(pair_id, []))
        
        print(f"      Вычисление рисков на {len(test_samples)} test samples...", file=sys.stderr)
        sys.stderr.flush()
        test_risks = _get_risks_with_head(test_samples, weights, head, entropy_quantile=verifier.entropy_quantile)
        
        # Группируем риски по парам и применяем сглаживание
        test_samples_by_pair: dict[str, list[TokenSample]] = {}
        for sample in test_samples:
            test_samples_by_pair.setdefault(sample.pair_id, []).append(sample)
        
        # Создаём маппинг pair_id -> pair для доступа к исходным labels
        test_pairs_by_id = {
            str(train_pairs[idx].get("id", idx)): train_pairs[idx]
            for idx in test_indices
        }
        risk_idx = 0
        for pair_id, samples in test_samples_by_pair.items():
            pair_risks = []
            for _ in samples:
                pair_risks.append(test_risks[risk_idx])
                risk_idx += 1
            
            # Сглаживание
            max_index = max(s.index for s in samples)
            risk_by_index = {s.index: r for s, r in zip(samples, pair_risks, strict=False)}
            risk_sequence = [risk_by_index.get(i, 0.0) for i in range(max_index + 1)]
            smoothed = _smooth(risk_sequence, SMOOTH_WINDOW)
            
            # Берём только scored токены
            scored_risks = [smoothed[s.index] for s in samples if s.index < len(smoothed)]
            risks_by_pair[pair_id] = scored_risks
            
            # Сохраняем метаданные
            label = answer_label(test_pairs_by_id[pair_id])
            doc_id = doc_id_by_pair[pair_id]
            
            pair_ids.append(pair_id)
            labels.append(label)
            pair_doc_ids.append(doc_id)
            pair_folds.append(fold_idx)
        
        print(f"      ✓ Фолд {fold_idx + 1} завершен, обработано {len(test_samples_by_pair)} пар", file=sys.stderr)
        sys.stderr.flush()
    
    return risks_by_pair, pair_ids, labels, pair_doc_ids, pair_folds


def compute_oof_answer_scores(
    risks_by_pair: dict[str, list[float]],
    pair_ids: list[str],
    labels: list[int],
    aggregator: str = "current",
) -> tuple[list[float], list[int], list[str]]:
    """Вычислить OOF answer scores для данного агрегатора из предвычисленных рисков.
    
    Args:
        risks_by_pair: предвычисленные OOF token risks
        pair_ids: список pair_id в том же порядке, что и labels
        labels: лейблы для каждой пары
        aggregator: тип агрегатора
    
    Returns:
        (oof_scores, oof_labels, oof_pair_ids)
    """
    # Агрегируем риски в answer scores
    answer_scores = _aggregate_from_dict(risks_by_pair, aggregator)
    
    # Собираем результаты в том же порядке, что pair_ids
    oof_scores = [answer_scores.get(pair_id, 0.0) for pair_id in pair_ids]
    
    return oof_scores, labels, pair_ids


def fit_answer_calibration(
    raw_oof_scores: list[float],
    labels: list[int],
    target_fpr: float = 0.30,
) -> tuple[IsotonicCalibrator, float, dict[str, Any]]:
    """Обучить answer-калибратор на raw OOF scores.
    
    Args:
        raw_oof_scores: сырые OOF answer scores
        labels: лейблы для каждой пары
        target_fpr: целевой FPR для выбора порога
    
    Returns:
        (calibrator, threshold, threshold_report)
    """
    calibrator = IsotonicCalibrator.fit(raw_oof_scores, labels, compress=False)
    calibrated_scores = calibrator.transform(raw_oof_scores)
    threshold_report = choose_threshold_report(calibrated_scores, labels, max_fpr=target_fpr)
    threshold = float(threshold_report["threshold"])
    
    return calibrator, threshold, threshold_report


def compute_dev_answer_scores(
    verifier: Verifier,
    dev_pairs: list[dict],
    final_weights: dict[str, float],
    final_head: dict[str, Any],
    aggregator: str = "current",
) -> tuple[list[float], list[int], list[str]]:
    """Вычислить answer scores на dev с использованием финальной модели.
    
    Args:
        verifier: верификатор
        dev_pairs: пары для оценки
        final_weights: веса признаков из финальной модели
        final_head: голова из финальной модели
        aggregator: тип агрегатора
    
    Returns:
        (dev_scores, dev_labels, dev_pair_ids)
    """
    # Собираем samples для dev
    dev_samples, _ = collect_samples(verifier, dev_pairs)
    
    # Получаем риски с головой
    dev_risks = _get_risks_with_head(dev_samples, final_weights, final_head, entropy_quantile=verifier.entropy_quantile)
    
    # Группируем по парам и применяем сглаживание
    dev_samples_by_pair: dict[str, list[TokenSample]] = {}
    for sample in dev_samples:
        dev_samples_by_pair.setdefault(sample.pair_id, []).append(sample)
    
    risks_by_pair: dict[str, list[float]] = {}
    risk_idx = 0
    for pair_id, samples in dev_samples_by_pair.items():
        pair_risks = []
        for _ in samples:
            pair_risks.append(dev_risks[risk_idx])
            risk_idx += 1
        
        # Сглаживание
        max_index = max(s.index for s in samples)
        risk_by_index = {s.index: r for s, r in zip(samples, pair_risks, strict=False)}
        risk_sequence = [risk_by_index.get(i, 0.0) for i in range(max_index + 1)]
        smoothed = _smooth(risk_sequence, SMOOTH_WINDOW)
        
        # Берём только scored токены
        scored_risks = [smoothed[s.index] for s in samples if s.index < len(smoothed)]
        risks_by_pair[pair_id] = scored_risks
    
    # Агрегируем в answer scores
    answer_scores = _aggregate_from_dict(risks_by_pair, aggregator)
    
    # Собираем результаты
    dev_scores: list[float] = []
    dev_labels: list[int] = []
    dev_pair_ids: list[str] = []
    
    samples_by_pair: dict[str, list[TokenSample]] = {}
    for sample in dev_samples:
        samples_by_pair.setdefault(sample.pair_id, []).append(sample)
    
    for pair in dev_pairs:
        pair_id = str(pair.get("id", "unknown"))
        label = answer_label(pair)
        dev_pair_ids.append(pair_id)
        dev_labels.append(label)
        dev_scores.append(answer_scores.get(pair_id, 0.0))
    
    return dev_scores, dev_labels, dev_pair_ids


def extract_oof_answer_features(
    samples_by_pair: dict[str, list[TokenSample]],
    all_pair_ids: list[str] | None = None,
) -> dict[str, list[float]]:
    """Извлечь answer-признаки для P1 answer-головы из OOF samples.
    
    Args:
        samples_by_pair: словарь {pair_id: список TokenSample}
            (только scored токены, как из collect_samples)
        all_pair_ids: полный список pair_id для покрытия всех пар.
            Если None, то только те, что есть в samples_by_pair.
    
    Returns:
        Словарь {pair_id: список answer-признаков (37 признаков)}
        Для пар без scored токенов: 35 нулевых сводок + log1p(0)=0 + empty=1.
    
    Для каждого pair извлекаются признаки из полных FeatureMatrix
    (не из округлённого payload) для scored токенов.
    """
    n_features = len(FEATURE_NAMES) * 7 + 2  # 37
    answer_features_by_pair: dict[str, list[float]] = {}
    
    # Определяем полный список pair_id
    pair_ids_to_process = all_pair_ids if all_pair_ids is not None else list(samples_by_pair.keys())
    
    for pair_id in pair_ids_to_process:
        samples = samples_by_pair.get(pair_id, [])
        
        if not samples:
            # Нет scored токенов: 35 нулевых сводок + log1p(0)=0 + empty=1
            zero_features = [0.0] * (len(FEATURE_NAMES) * 7)
            zero_features.append(math.log1p(0))  # = 0.0
            zero_features.append(1.0)  # empty flag
            answer_features_by_pair[pair_id] = zero_features
            continue
        
        # Извлекаем признаки для каждого scored токена
        features_list: list[dict[str, float]] = []
        scored_mask: list[bool] = []
        
        for sample in samples:
            # Извлекаем признаки из sample.features
            features_dict = {name: sample.features[name] for name in FEATURE_NAMES}
            features_list.append(features_dict)
            # Все samples уже отфильтрованы по is_scored_token
            scored_mask.append(True)
        
        # Извлекаем answer-признаки
        answer_features = extract_answer_features(features_list, scored_mask)
        answer_features_by_pair[pair_id] = answer_features
    
    return answer_features_by_pair
