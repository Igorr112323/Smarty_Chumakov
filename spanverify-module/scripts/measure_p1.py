#!/usr/bin/env python3
"""P1: Измерение обучаемой answer-головы с настоящим cross-fitting OOF.

Сравнивает baseline (mean_top3 агрегатор) с P1 (обучаемая answer-голова на HF-признаках).
P1 OOF: 5 документных фолдов, seed=42, на каждом фолде train_answer_head на fit,
предсказание на held-out. Финальная модель — на всех train для dev.
"""
import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spanverify.dataset import read_pairs
from spanverify.engine import Verifier, WeightsBundle
from spanverify.features import load_feature_cache, set_feature_cache, FEATURE_NAMES
from spanverify.oof import (
    compute_oof_risks,
    compute_oof_answer_scores,
    compute_dev_answer_scores,
    extract_oof_answer_features,
    _train_model_on_subset,
    answer_label,
)
from spanverify.answer_head import train_answer_head, AnswerHeadModel, N_ANSWER_FEATURES
from spanverify.train import collect_samples, auc_score
from spanverify.calibration import IsotonicCalibrator, choose_threshold_report


def _build_doc_folds(train_pairs, n_folds=5, seed=42):
    """Построить 5 документных фолдов для OOF."""
    doc_groups = defaultdict(list)
    doc_id_by_pair = {}
    
    for i, pair in enumerate(train_pairs):
        pair_id = str(pair.get("id", i))
        meta = pair.get("meta", {})
        doc_id = meta.get("doc_id") or meta.get("document_id") or meta.get("group") or pair_id
        doc_groups[doc_id].append(i)
        doc_id_by_pair[pair_id] = doc_id
    
    doc_ids = sorted(doc_groups.keys())
    rng = random.Random(seed)
    rng.shuffle(doc_ids)
    
    fold_size = len(doc_ids) // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(doc_ids)
        fold_doc_ids = set(doc_ids[start:end])
        fold_indices = []
        for doc_id in fold_doc_ids:
            fold_indices.extend(doc_groups[doc_id])
        folds.append(fold_indices)
    
    return folds, doc_id_by_pair


def _write_jsonl(path, rows):
    """Записать JSONL файл."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256_list(items):
    """Вычислить SHA-256 хеш от списка строк (канонический порядок)."""
    canonical = json.dumps(sorted(items), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="P1: Измерение обучаемой answer-головы")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--mode", default="hf")
    parser.add_argument("--features-cache", required=True)
    parser.add_argument("--baseline-experiment", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-source-run-id", type=int, required=True)
    parser.add_argument("--measurement-run-id", type=int, required=True)
    parser.add_argument("--measurement-commit", required=True)
    
    args = parser.parse_args()
    start_time = time.time()
    
    print("=" * 60)
    print("P1: Измерение обучаемой answer-головы (cross-fitting OOF)")
    print("=" * 60)
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # [1/9] Загружаем бандл
    print("\n[1/9] Загрузка бандла...")
    baseline_path = Path(args.baseline_experiment)
    exp = json.loads(baseline_path.read_text())
    bundle = WeightsBundle.from_dict(exp["bundle"])
    print(f"✓ Бандл загружен")
    
    # [2/9] Загружаем данные
    print("\n[2/9] Загрузка данных...")
    splits_dir = Path(args.splits)
    train_pairs = list(read_pairs(splits_dir / "train.jsonl"))
    dev_pairs = list(read_pairs(splits_dir / "dev.jsonl"))
    print(f"✓ Train: {len(train_pairs)} pairs")
    print(f"✓ Dev: {len(dev_pairs)} pairs")
    
    # [3/9] Загружаем кеш
    print("\n[3/9] Загрузка кеша...")
    cache = load_feature_cache([args.features_cache], counting=True, strict=True)
    set_feature_cache(cache)
    print(f"✓ Кеш: {len(cache)} ключей")
    
    # [4/9] Создаём верификатор
    print("\n[4/9] Создание верификатора...")
    verifier = Verifier(
        mode="hf",
        model_name=exp["model"],
        weights=bundle,
        features_cache=cache,
        span_expansion="clause_capped",
    )
    print(f"✓ Verifier создан")
    
    # [5/9] Собираем samples и извлекаем answer-признаки
    print("\n[5/9] Сбор samples и извлечение answer-признаков...")
    
    # Train samples
    print("  Train samples...")
    sys.stdout.flush()
    train_samples, train_stats = collect_samples(verifier, train_pairs)
    train_samples_by_pair = {}
    for sample in train_samples:
        train_samples_by_pair.setdefault(sample.pair_id, []).append(sample)
    print(f"  ✓ Train: {len(train_samples)} scored tokens, {len(train_samples_by_pair)} пар")
    
    # Dev samples
    print("  Dev samples...")
    sys.stdout.flush()
    dev_samples, dev_stats = collect_samples(verifier, dev_pairs)
    dev_samples_by_pair = {}
    for sample in dev_samples:
        dev_samples_by_pair.setdefault(sample.pair_id, []).append(sample)
    print(f"  ✓ Dev: {len(dev_samples)} scored tokens, {len(dev_samples_by_pair)} пар")
    
    # Все pair_ids
    train_pair_ids = [str(p.get("id", i)) for i, p in enumerate(train_pairs)]
    dev_pair_ids = [str(p.get("id", i)) for i, p in enumerate(dev_pairs)]
    
    # Извлекаем answer-признаки (один раз, label-free)
    print("  Извлечение answer-признаков...")
    train_features_by_pair = extract_oof_answer_features(train_samples_by_pair, train_pair_ids)
    dev_features_by_pair = extract_oof_answer_features(dev_samples_by_pair, dev_pair_ids)
    
    assert len(train_features_by_pair) == len(train_pairs), \
        f"Train features coverage: {len(train_features_by_pair)} != {len(train_pairs)}"
    assert len(dev_features_by_pair) == len(dev_pairs), \
        f"Dev features coverage: {len(dev_features_by_pair)} != {len(dev_pairs)}"
    
    print(f"  ✓ Answer-признаки: train={len(train_features_by_pair)}, dev={len(dev_features_by_pair)}")
    
    # Gold labels
    train_labels = [answer_label(p) for p in train_pairs]
    dev_labels = [answer_label(p) for p in dev_pairs]
    
    print(f"  Train positive: {sum(train_labels)}/{len(train_labels)}")
    print(f"  Dev positive: {sum(dev_labels)}/{len(dev_labels)}")
    
    # [6/9] P1 Cross-fitting OOF (5 фолдов)
    print("\n[6/9] P1 cross-fitting OOF (5 фолдов)...")
    
    folds, doc_id_by_pair = _build_doc_folds(train_pairs, n_folds=5, seed=42)
    
    # Массив для OOF предсказаний
    p1_oof_scores = [0.0] * len(train_pairs)
    pair_id_to_idx = {pid: i for i, pid in enumerate(train_pair_ids)}
    
    fold_receipts = []
    
    for fold_idx in range(5):
        held_out_indices = set(folds[fold_idx])
        fit_indices = []
        for fi in range(5):
            if fi != fold_idx:
                fit_indices.extend(folds[fi])
        
        # Проверяем отсутствие пересечений
        assert len(held_out_indices & set(fit_indices)) == 0
        
        # Готовим fit данные
        fit_X = []
        fit_y = []
        fit_doc_ids = set()
        for idx in fit_indices:
            pid = train_pair_ids[idx]
            fit_X.append(train_features_by_pair[pid])
            fit_y.append(train_labels[idx])
            fit_doc_ids.add(doc_id_by_pair[pid])
        
        # Обучаем P1 на fit
        fold_model = train_answer_head(fit_X, fit_y)
        
        # Предсказываем на held-out
        held_out_pids = []
        held_out_X = []
        held_out_y = []
        held_out_doc_ids = set()
        for idx in held_out_indices:
            pid = train_pair_ids[idx]
            held_out_pids.append(pid)
            held_out_X.append(train_features_by_pair[pid])
            held_out_y.append(train_labels[idx])
            held_out_doc_ids.add(doc_id_by_pair[pid])
        
        held_out_scores = fold_model.predict_many(held_out_X)
        
        # Записываем OOF предсказания
        for pid, score in zip(held_out_pids, held_out_scores):
            idx = pair_id_to_idx[pid]
            p1_oof_scores[idx] = score
        
        # Receipt
        receipt = {
            "fold": fold_idx,
            "fit_doc_count": len(fit_doc_ids),
            "fit_sample_count": len(fit_X),
            "held_out_doc_count": len(held_out_doc_ids),
            "held_out_sample_count": len(held_out_X),
            "held_out_positive": sum(held_out_y),
            "held_out_negative": len(held_out_y) - sum(held_out_y),
            "fit_doc_ids_sha256": _sha256_list(sorted(fit_doc_ids)),
            "held_out_doc_ids_sha256": _sha256_list(sorted(held_out_doc_ids)),
            "fit_doc_ids": sorted(fit_doc_ids),
            "held_out_doc_ids": sorted(held_out_doc_ids),
        }
        fold_receipts.append(receipt)
        
        print(f"  Фолд {fold_idx}: fit={len(fit_X)}, held_out={len(held_out_X)}, "
              f"pos={sum(held_out_y)}, neg={len(held_out_y) - sum(held_out_y)}")
    
    # Проверка покрытия P1 OOF
    assert len(p1_oof_scores) == len(train_pairs), \
        f"P1 OOF coverage: {len(p1_oof_scores)} != {len(train_pairs)}"
    
    # Проверяем что каждая пара была предсказана ровно один раз
    p1_oof_seen = set()
    for fold_idx in range(5):
        held_out_indices = set(folds[fold_idx])
        for idx in held_out_indices:
            pid = train_pair_ids[idx]
            if pid in p1_oof_seen:
                raise ValueError(f"P1 OOF: pair_id {pid} предсказана более одного раза")
            p1_oof_seen.add(pid)
    
    if len(p1_oof_seen) != len(train_pairs):
        raise ValueError(
            f"P1 OOF coverage: предсказано {len(p1_oof_seen)} пар, ожидалось {len(train_pairs)}"
        )
    
    print(f"  ✓ P1 OOF: {len(p1_oof_scores)} предсказаний, покрытие 100%")
    
    # Baseline OOF scores (mean_top3)
    print("\n  Baseline OOF (mean_top3)...")
    risks_by_pair, oof_pair_ids, oof_labels, oof_doc_ids, oof_folds_base = compute_oof_risks(
        verifier, train_pairs, n_folds=5, seed=42
    )
    baseline_oof_scores, _, _ = compute_oof_answer_scores(
        risks_by_pair, oof_pair_ids, oof_labels, aggregator="mean_top3"
    )
    
    # Маппинг baseline OOF к train_pair_ids порядку
    oof_pid_to_score = dict(zip(oof_pair_ids, baseline_oof_scores))
    baseline_oof_aligned = []
    for pid in train_pair_ids:
        if pid not in oof_pid_to_score:
            raise ValueError(f"Baseline OOF: pair_id {pid} отсутствует в предсказаниях")
        baseline_oof_aligned.append(oof_pid_to_score[pid])
    
    print(f"  ✓ Baseline OOF: {len(baseline_oof_aligned)} предсказаний")
    
    # [7/9] Обучение финальных моделей
    print("\n[7/9] Обучение финальных моделей...")
    
    # Финальная P1 модель на всех train
    final_train_X = [train_features_by_pair[pid] for pid in train_pair_ids]
    final_p1_model = train_answer_head(final_train_X, train_labels)
    print(f"  ✓ Финальная P1 модель: {final_p1_model.n_features} признаков")
    
    # Финальная token-head модель для dev
    all_train_samples, _ = collect_samples(verifier, train_pairs)
    final_weights, final_head = _train_model_on_subset(
        all_train_samples, entropy_quantile=verifier.entropy_quantile
    )
    print(f"  ✓ Финальная token-head модель")
    
    # Dev scores
    print("  Dev scores...")
    
    # P1 dev scores (из уже извлечённых признаков)
    p1_dev_X = [dev_features_by_pair[pid] for pid in dev_pair_ids]
    p1_dev_raw_scores = final_p1_model.predict_many(p1_dev_X)
    
    # Baseline dev scores
    dev_scores_baseline, dev_labels_baseline, dev_pair_ids_baseline = compute_dev_answer_scores(
        verifier, dev_pairs, final_weights, final_head, aggregator="mean_top3"
    )
    
    # Маппинг baseline dev к dev_pair_ids порядку
    dev_pid_to_score = dict(zip(dev_pair_ids_baseline, dev_scores_baseline))
    baseline_dev_aligned = []
    for pid in dev_pair_ids:
        if pid not in dev_pid_to_score:
            raise ValueError(f"Baseline dev: pair_id {pid} отсутствует в предсказаниях")
        baseline_dev_aligned.append(dev_pid_to_score[pid])
    
    print(f"  ✓ Dev scores: P1={len(p1_dev_raw_scores)}, baseline={len(baseline_dev_aligned)}")
    
    # [8/9] Калибровка и пороги
    print("\n[8/9] Калибровка и выбор порогов...")
    
    # Baseline калибровка
    calibrator_baseline = IsotonicCalibrator.fit(baseline_oof_aligned, train_labels, compress=False)
    cal_baseline_train = calibrator_baseline.transform(baseline_oof_aligned)
    report_baseline = choose_threshold_report(cal_baseline_train, train_labels, max_fpr=0.30)
    threshold_baseline = float(report_baseline["threshold"])
    
    # P1 калибровка
    calibrator_p1 = IsotonicCalibrator.fit(p1_oof_scores, train_labels, compress=False)
    cal_p1_train = calibrator_p1.transform(p1_oof_scores)
    report_p1 = choose_threshold_report(cal_p1_train, train_labels, max_fpr=0.30)
    threshold_p1 = float(report_p1["threshold"])
    
    print(f"  Baseline threshold: {threshold_baseline:.6f}")
    print(f"  P1 threshold: {threshold_p1:.6f}")
    
    # Dev калиброванные scores
    cal_baseline_dev = calibrator_baseline.transform(baseline_dev_aligned)
    cal_p1_dev = calibrator_p1.transform(p1_dev_raw_scores)
    
    # [9/9] Метрики на dev
    print("\n[9/9] Вычисление метрик на dev...")
    
    def compute_confusion(labels, calibrated_scores, threshold):
        tp = sum(1 for s, l in zip(calibrated_scores, labels) if s >= threshold and l == 1)
        fp = sum(1 for s, l in zip(calibrated_scores, labels) if s >= threshold and l == 0)
        fn = sum(1 for s, l in zip(calibrated_scores, labels) if s < threshold and l == 1)
        tn = sum(1 for s, l in zip(calibrated_scores, labels) if s < threshold and l == 0)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        fpr = fp / (fp + tn) if fp + tn > 0 else 0.0
        return {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
            "positive_predictions": tp + fp,
        }
    
    baseline_metrics = compute_confusion(dev_labels, cal_baseline_dev, threshold_baseline)
    p1_metrics = compute_confusion(dev_labels, cal_p1_dev, threshold_p1)
    
    raw_auc_baseline = auc_score(dev_labels, baseline_dev_aligned)
    cal_auc_baseline = auc_score(dev_labels, cal_baseline_dev)
    raw_auc_p1 = auc_score(dev_labels, p1_dev_raw_scores)
    cal_auc_p1 = auc_score(dev_labels, cal_p1_dev)
    
    print("\n" + "=" * 60)
    print("Результаты сравнения baseline vs P1")
    print("=" * 60)
    print(f"\nBaseline (mean_top3):")
    print(f"  Dev Raw AUC: {raw_auc_baseline:.4f}")
    print(f"  Dev Cal AUC: {cal_auc_baseline:.4f}")
    print(f"  Dev F1: {baseline_metrics['f1']:.4f}")
    print(f"  Dev FPR: {baseline_metrics['fpr']:.4f}")
    print(f"  Dev Precision: {baseline_metrics['precision']:.4f}")
    print(f"  Dev Recall: {baseline_metrics['recall']:.4f}")
    print(f"  TP/FP/FN/TN: {baseline_metrics['tp']}/{baseline_metrics['fp']}/{baseline_metrics['fn']}/{baseline_metrics['tn']}")
    print(f"  Threshold: {threshold_baseline:.6f}")
    print(f"  Positive predictions: {baseline_metrics['positive_predictions']}")
    
    print(f"\nP1 (answer-head):")
    print(f"  Dev Raw AUC: {raw_auc_p1:.4f}")
    print(f"  Dev Cal AUC: {cal_auc_p1:.4f}")
    print(f"  Dev F1: {p1_metrics['f1']:.4f}")
    print(f"  Dev FPR: {p1_metrics['fpr']:.4f}")
    print(f"  Dev Precision: {p1_metrics['precision']:.4f}")
    print(f"  Dev Recall: {p1_metrics['recall']:.4f}")
    print(f"  TP/FP/FN/TN: {p1_metrics['tp']}/{p1_metrics['fp']}/{p1_metrics['fn']}/{p1_metrics['tn']}")
    print(f"  Threshold: {threshold_p1:.6f}")
    print(f"  Positive predictions: {p1_metrics['positive_predictions']}")
    
    print(f"\nImprovement:")
    print(f"  F1 delta: {p1_metrics['f1'] - baseline_metrics['f1']:+.4f}")
    print(f"  FPR delta: {p1_metrics['fpr'] - baseline_metrics['fpr']:+.4f}")
    
    # Сохранение артефактов
    print("\nСохранение артефактов...")
    duration = time.time() - start_time
    
    # P1 модель
    p1_model_path = out_dir / "p1_model.json"
    final_p1_model.save(p1_model_path)
    
    # Baseline bundle - СОЗДАЁМ ПРАВИЛЬНЫЙ БАНДЛ с финальными весами
    # (не исходный, а обученный на всех train данных)
    final_baseline_bundle = WeightsBundle(
        weights=final_weights,
        threshold=bundle.threshold,
        target_fpr=bundle.target_fpr,
        span_z=bundle.span_z,
        span_floor=bundle.span_floor,
        span_cap=bundle.span_cap,
        isotonic=bundle.isotonic,
        head=final_head,
        answer_head=None,  # baseline не использует P1
        folds=bundle.folds,
        seed=bundle.seed,
        version=bundle.version,
        mode=bundle.mode,
        meta=bundle.meta,
    )
    baseline_bundle_path = out_dir / "baseline_bundle.json"
    baseline_bundle_path.write_text(json.dumps(final_baseline_bundle.to_dict(), indent=2), encoding="utf-8")
    
    # Калибраторы
    calibrator_baseline_path = out_dir / "calibrator_baseline.json"
    calibrator_baseline_path.write_text(json.dumps(calibrator_baseline.to_dict(), indent=2), encoding="utf-8")
    
    calibrator_p1_path = out_dir / "calibrator_p1.json"
    calibrator_p1_path.write_text(json.dumps(calibrator_p1.to_dict(), indent=2), encoding="utf-8")
    
    # Thresholds
    thresholds_path = out_dir / "thresholds.json"
    thresholds_path.write_text(json.dumps({
        "baseline": threshold_baseline,
        "p1": threshold_p1,
        "max_fpr": 0.30,
    }, indent=2), encoding="utf-8")
    
    # Fold receipts
    fold_receipts_path = out_dir / "fold_receipts.json"
    fold_receipts_path.write_text(json.dumps(fold_receipts, indent=2), encoding="utf-8")
    
    # OOF JSONL (842 строки)
    oof_jsonl_path = out_dir / "oof_predictions.jsonl"
    oof_rows = []
    for i, pid in enumerate(train_pair_ids):
        oof_rows.append({
            "pair_id": pid,
            "doc_id": doc_id_by_pair[pid],
            "fold": oof_folds_base[oof_pair_ids.index(pid)] if pid in oof_pair_ids else -1,
            "gold": train_labels[i],
            "baseline_raw": baseline_oof_aligned[i],
            "baseline_calibrated": cal_baseline_train[i],
            "baseline_prediction": 1 if cal_baseline_train[i] >= threshold_baseline else 0,
            "p1_raw": p1_oof_scores[i],
            "p1_calibrated": cal_p1_train[i],
            "p1_prediction": 1 if cal_p1_train[i] >= threshold_p1 else 0,
        })
    _write_jsonl(oof_jsonl_path, oof_rows)
    
    # Dev JSONL (181 строка)
    dev_jsonl_path = out_dir / "dev_predictions.jsonl"
    dev_rows = []
    for i, pid in enumerate(dev_pair_ids):
        dev_rows.append({
            "pair_id": pid,
            "gold": dev_labels[i],
            "baseline_raw": baseline_dev_aligned[i],
            "baseline_calibrated": cal_baseline_dev[i],
            "baseline_prediction": 1 if cal_baseline_dev[i] >= threshold_baseline else 0,
            "p1_raw": p1_dev_raw_scores[i],
            "p1_calibrated": cal_p1_dev[i],
            "p1_prediction": 1 if cal_p1_dev[i] >= threshold_p1 else 0,
        })
    _write_jsonl(dev_jsonl_path, dev_rows)
    
    # Runtime manifest
    manifest = {
        "schema_version": "p1-measurement-v2",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cache_source_run_id": args.cache_source_run_id,
        "measurement_run_id": args.measurement_run_id,
        "measurement_commit": args.measurement_commit,
        "train_count": len(train_pairs),
        "dev_count": len(dev_pairs),
        "p1_features_count": final_p1_model.n_features,
        "p1_schema_version": final_p1_model.schema_version,
        "n_folds": 5,
        "seed": 42,
        "oof_method": "cross-fitting",
        "threshold_method": "choose_threshold_report(max_fpr=0.30)",
    }
    
    manifest_path = out_dir / "runtime_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    
    # Comparison JSON
    comparison = {
        "status": "success",
        "schema_version": "p1-comparison-v2",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cache_source_run_id": args.cache_source_run_id,
        "measurement_run_id": args.measurement_run_id,
        "measurement_commit": args.measurement_commit,
        "duration_s": round(duration, 2),
        "train_pairs": len(train_pairs),
        "dev_pairs": len(dev_pairs),
        "baseline": {
            "aggregator": "mean_top3",
            "threshold": threshold_baseline,
            "dev_raw_auc": raw_auc_baseline,
            "dev_calibrated_auc": cal_auc_baseline,
            **baseline_metrics,
        },
        "p1": {
            "model": "answer-head",
            "n_features": final_p1_model.n_features,
            "schema_version": final_p1_model.schema_version,
            "threshold": threshold_p1,
            "dev_raw_auc": raw_auc_p1,
            "dev_calibrated_auc": cal_auc_p1,
            **p1_metrics,
        },
        "improvement": {
            "f1_delta": p1_metrics["f1"] - baseline_metrics["f1"],
            "fpr_delta": p1_metrics["fpr"] - baseline_metrics["fpr"],
        },
    }
    
    comparison_path = out_dir / "p1_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    
    # Evidence
    evidence = {
        "schema_version": "p1-evidence-v2",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train_pairs_count": len(train_pairs),
        "dev_pairs_count": len(dev_pairs),
        "train_positive_count": sum(train_labels),
        "dev_positive_count": sum(dev_labels),
        "oof_lines": len(oof_rows),
        "dev_lines": len(dev_rows),
        "fold_receipts": fold_receipts,
        "artifacts": {
            "p1_model": str(p1_model_path),
            "baseline_bundle": str(baseline_bundle_path),
            "calibrator_baseline": str(calibrator_baseline_path),
            "calibrator_p1": str(calibrator_p1_path),
            "thresholds": str(thresholds_path),
            "oof_predictions": str(oof_jsonl_path),
            "dev_predictions": str(dev_jsonl_path),
            "comparison": str(comparison_path),
        },
    }
    
    evidence_path = out_dir / "evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    
    # Markdown отчёт
    md_lines = [
        "# P1: Сравнение baseline vs answer-head (cross-fitting OOF)",
        "",
        f"**Train pairs:** {len(train_pairs)}",
        f"**Dev pairs:** {len(dev_pairs)}",
        f"**P1 features:** {final_p1_model.n_features} (schema {final_p1_model.schema_version})",
        f"**OOF method:** 5-fold cross-fitting, seed=42",
        f"**Threshold method:** choose_threshold_report(max_fpr=0.30)",
        "",
        "## Результаты",
        "",
        "| Метрика | Baseline (mean_top3) | P1 (answer-head) | Delta |",
        "|---------|---------------------|-----------------|-------|",
        f"| Dev Raw AUC | {raw_auc_baseline:.4f} | {raw_auc_p1:.4f} | {raw_auc_p1 - raw_auc_baseline:+.4f} |",
        f"| Dev Cal AUC | {cal_auc_baseline:.4f} | {cal_auc_p1:.4f} | {cal_auc_p1 - cal_auc_baseline:+.4f} |",
        f"| Dev F1 | {baseline_metrics['f1']:.4f} | {p1_metrics['f1']:.4f} | {p1_metrics['f1'] - baseline_metrics['f1']:+.4f} |",
        f"| Dev FPR | {baseline_metrics['fpr']:.4f} | {p1_metrics['fpr']:.4f} | {p1_metrics['fpr'] - baseline_metrics['fpr']:+.4f} |",
        f"| Dev Precision | {baseline_metrics['precision']:.4f} | {p1_metrics['precision']:.4f} | {p1_metrics['precision'] - baseline_metrics['precision']:+.4f} |",
        f"| Dev Recall | {baseline_metrics['recall']:.4f} | {p1_metrics['recall']:.4f} | {p1_metrics['recall'] - baseline_metrics['recall']:+.4f} |",
        f"| TP/FP/FN/TN | {baseline_metrics['tp']}/{baseline_metrics['fp']}/{baseline_metrics['fn']}/{baseline_metrics['tn']} | {p1_metrics['tp']}/{p1_metrics['fp']}/{p1_metrics['fn']}/{p1_metrics['tn']} | |",
        f"| Threshold | {threshold_baseline:.6f} | {threshold_p1:.6f} | |",
        f"| Pos. predictions | {baseline_metrics['positive_predictions']} | {p1_metrics['positive_predictions']} | |",
        "",
        "## Fold receipts",
        "",
        "| Fold | Fit docs | Fit samples | Held-out docs | Held-out samples | Held-out pos | Held-out neg |",
        "|------|----------|-------------|---------------|------------------|--------------|--------------|",
    ]
    
    for r in fold_receipts:
        md_lines.append(
            f"| {r['fold']} | {r['fit_doc_count']} | {r['fit_sample_count']} | "
            f"{r['held_out_doc_count']} | {r['held_out_sample_count']} | "
            f"{r['held_out_positive']} | {r['held_out_negative']} |"
        )
    
    md_lines.extend([
        "",
        f"**Длительность:** {duration:.1f}s",
    ])
    
    md_path = out_dir / "p1_comparison.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    
    # Проверка комплектности
    print(f"\nПроверка комплектности артефактов...")
    required_files = [
        "p1_model.json",
        "baseline_bundle.json",
        "calibrator_baseline.json",
        "calibrator_p1.json",
        "thresholds.json",
        "fold_receipts.json",
        "oof_predictions.jsonl",
        "dev_predictions.jsonl",
        "runtime_manifest.json",
        "p1_comparison.json",
        "p1_comparison.md",
        "evidence.json",
    ]
    for fname in required_files:
        fpath = out_dir / fname
        assert fpath.exists(), f"Отсутствует файл: {fname}"
        print(f"  ✓ {fname} ({fpath.stat().st_size} bytes)")
    
    # Проверка числа строк
    oof_line_count = len(oof_jsonl_path.read_text().strip().split("\n"))
    dev_line_count = len(dev_jsonl_path.read_text().strip().split("\n"))
    assert oof_line_count == len(train_pairs), f"OOF lines: {oof_line_count} != {len(train_pairs)}"
    assert dev_line_count == len(dev_pairs), f"Dev lines: {dev_line_count} != {len(dev_pairs)}"
    print(f"  ✓ OOF JSONL: {oof_line_count} строк")
    print(f"  ✓ Dev JSONL: {dev_line_count} строк")
    
    print(f"\n✓ Все артефакты сохранены в {out_dir}")
    
    print("\n" + "=" * 60)
    print(f"✓ Завершено за {duration:.1f}s")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        import traceback
        print("\n" + "=" * 60, file=sys.stderr)
        print("❌ КРИТИЧЕСКАЯ ОШИБКА", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(f"Тип ошибки: {type(e).__name__}", file=sys.stderr)
        print(f"Сообщение: {e}", file=sys.stderr)
        print("\nПолный traceback:", file=sys.stderr)
        traceback.print_exc()
        print("=" * 60, file=sys.stderr)
        sys.exit(1)
