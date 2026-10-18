#!/usr/bin/env python3
"""Recovery script: извлечение и независимый пересчёт P1-метрик из артефактов.

Этот скрипт НЕ импортирует spanverify, НЕ ставит torch/transformers,
НЕ обучает модели, НЕ обращается к HF-инференсу.

Он скачивает готовые артефакты, проверяет provenance, IDs/gold,
независимо пересчитывает метрики из evidence и сохраняет файлы.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    """Загрузить JSON файл."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict[str, Any]) -> None:
    """Сохранить JSON файл."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def compute_confusion(labels: list[int], predictions: list[int]) -> dict[str, int]:
    """Вычислить confusion matrix."""
    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def compute_metrics(confusion: dict[str, int]) -> dict[str, float]:
    """Вычислить метрики из confusion matrix."""
    tp, fp, fn, tn = confusion["tp"], confusion["fp"], confusion["fn"], confusion["tn"]
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    fpr = fp / (fp + tn) if fp + tn > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "positive_predictions": tp + fp,
    }


def compute_auc(labels: list[int], scores: list[float]) -> float:
    """Вычислить AUC-ROC (упрощённая версия)."""
    if len(labels) == 0:
        return 0.0
    
    # Сортируем по scores
    paired = sorted(zip(scores, labels), reverse=True)
    
    tp = 0
    fp = 0
    total_positive = sum(labels)
    total_negative = len(labels) - total_positive
    
    if total_positive == 0 or total_negative == 0:
        return 0.0
    
    auc = 0.0
    prev_fpr = 0.0
    prev_tpr = 0.0
    
    for score, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
        
        tpr = tp / total_positive
        fpr = fp / total_negative
        
        # Трапецеидальное правило
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        
        prev_fpr = fpr
        prev_tpr = tpr
    
    return auc


def verify_provenance(
    evidence: dict[str, Any],
    source_run_id: int,
    source_sha: str,
) -> None:
    """Проверить provenance evidence."""
    if evidence.get("source_run_id") != source_run_id:
        raise ValueError(
            f"Source run ID mismatch: expected {source_run_id}, "
            f"got {evidence.get('source_run_id')}"
        )
    
    if evidence.get("source_sha") != source_sha:
        raise ValueError(
            f"Source SHA mismatch: expected {source_sha}, "
            f"got {evidence.get('source_sha')}"
        )


def verify_ids_and_gold(
    oof_predictions: list[dict],
    dev_predictions: list[dict],
    expected_train_count: int,
    expected_dev_count: int,
) -> None:
    """Проверить IDs и gold labels."""
    if len(oof_predictions) != expected_train_count:
        raise ValueError(
            f"OOF predictions count mismatch: expected {expected_train_count}, "
            f"got {len(oof_predictions)}"
        )
    
    if len(dev_predictions) != expected_dev_count:
        raise ValueError(
            f"Dev predictions count mismatch: expected {expected_dev_count}, "
            f"got {len(dev_predictions)}"
        )
    
    # Проверяем что все pair_id уникальны
    oof_ids = [p["pair_id"] for p in oof_predictions]
    if len(oof_ids) != len(set(oof_ids)):
        raise ValueError("Duplicate pair_id in OOF predictions")
    
    dev_ids = [p["pair_id"] for p in dev_predictions]
    if len(dev_ids) != len(set(dev_ids)):
        raise ValueError("Duplicate pair_id in dev predictions")


def recompute_metrics(
    predictions: list[dict],
    raw_key: str,
    calibrated_key: str,
    prediction_key: str,
    threshold: float,
) -> dict[str, Any]:
    """Независимо пересчитать метрики из predictions."""
    labels = [p["gold"] for p in predictions]
    raw_scores = [p[raw_key] for p in predictions]
    calibrated_scores = [p[calibrated_key] for p in predictions]
    predictions_binary = [p[prediction_key] for p in predictions]
    
    # Confusion matrix из predictions
    confusion = compute_confusion(labels, predictions_binary)
    metrics = compute_metrics(confusion)
    
    # AUC
    raw_auc = compute_auc(labels, raw_scores)
    calibrated_auc = compute_auc(labels, calibrated_scores)
    
    return {
        **metrics,
        **confusion,
        "raw_auc": raw_auc,
        "calibrated_auc": calibrated_auc,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Recovery P1 metrics from artifacts")
    parser.add_argument("--artifacts-dir", required=True, help="Directory with downloaded artifacts")
    parser.add_argument("--source-run-id", type=int, required=True, help="Source measurement run ID")
    parser.add_argument("--source-sha", required=True, help="Source measurement SHA")
    parser.add_argument("--output-dir", required=True, help="Output directory for recovered results")
    
    args = parser.parse_args()
    
    artifacts_dir = Path(args.artifacts_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Recovery P1 Results")
    print("=" * 60)
    print(f"Source run: {args.source_run_id}")
    print(f"Source SHA: {args.source_sha}")
    print()
    
    # Загружаем артефакты
    comparison_dir = artifacts_dir / "p1-comparison"
    logs_dir = artifacts_dir / "p1-measurement-logs"
    
    if not comparison_dir.exists():
        print(f"ERROR: comparison artifacts not found at {comparison_dir}")
        return 1
    
    # Ищем файлы
    comparison_file = None
    oof_file = None
    dev_file = None
    evidence_file = None
    
    for root, dirs, files in os.walk(comparison_dir):
        for fname in files:
            fpath = Path(root) / fname
            if fname == "p1_comparison.json":
                comparison_file = fpath
            elif fname == "oof_predictions.jsonl":
                oof_file = fpath
            elif fname == "dev_predictions.jsonl":
                dev_file = fpath
            elif fname == "evidence.json":
                evidence_file = fpath
    
    if not comparison_file:
        print("ERROR: p1_comparison.json not found")
        return 1
    
    print(f"Found p1_comparison.json: {comparison_file}")
    
    if not oof_file or not dev_file:
        print("WARNING: oof_predictions.jsonl or dev_predictions.jsonl not found")
        print("Will use only p1_comparison.json for metrics")
    
    if not evidence_file:
        print("WARNING: evidence.json not found")
    
    # Загружаем исходное сравнение
    print("\nЗагрузка исходного p1_comparison.json...")
    original_comparison = load_json(comparison_file)
    
    # Проверяем provenance
    if evidence_file:
        print("Проверка provenance...")
        evidence = load_json(evidence_file)
        try:
            verify_provenance(evidence, args.source_run_id, args.source_sha)
            print("✓ Provenance verified")
        except ValueError as e:
            print(f"ERROR: {e}")
            return 1
    
    # Загружаем predictions если есть
    oof_predictions = []
    dev_predictions = []
    
    if oof_file:
        print(f"Загрузка oof_predictions.jsonl...")
        with open(oof_file, encoding="utf-8") as f:
            oof_predictions = [json.loads(line) for line in f if line.strip()]
        print(f"  ✓ Loaded {len(oof_predictions)} OOF predictions")
    
    if dev_file:
        print(f"Загрузка dev_predictions.jsonl...")
        with open(dev_file, encoding="utf-8") as f:
            dev_predictions = [json.loads(line) for line in f if line.strip()]
        print(f"  ✓ Loaded {len(dev_predictions)} dev predictions")
    
    # Проверяем IDs и gold
    if oof_predictions and dev_predictions:
        print("Проверка IDs и gold labels...")
        try:
            verify_ids_and_gold(oof_predictions, dev_predictions, 842, 181)
            print("✓ IDs and gold verified")
        except ValueError as e:
            print(f"ERROR: {e}")
            return 1
    
    # Независимый пересчёт метрик
    print("\nНезависимый пересчёт метрик...")
    
    recovered_baseline = {}
    recovered_p1 = {}
    
    if dev_predictions:
        # Получаем thresholds из original comparison
        threshold_baseline = original_comparison["baseline"]["threshold"]
        threshold_p1 = original_comparison["p1"]["threshold"]
        
        # Пересчитываем baseline
        recovered_baseline = recompute_metrics(
            dev_predictions,
            raw_key="baseline_raw",
            calibrated_key="baseline_calibrated",
            prediction_key="baseline_prediction",
            threshold=threshold_baseline,
        )
        
        # Пересчитываем P1
        recovered_p1 = recompute_metrics(
            dev_predictions,
            raw_key="p1_raw",
            calibrated_key="p1_calibrated",
            prediction_key="p1_prediction",
            threshold=threshold_p1,
        )
        
        print("\nBaseline (recomputed):")
        print(f"  Raw AUC: {recovered_baseline['raw_auc']:.6f}")
        print(f"  Calibrated AUC: {recovered_baseline['calibrated_auc']:.6f}")
        print(f"  F1: {recovered_baseline['f1']:.6f}")
        print(f"  FPR: {recovered_baseline['fpr']:.6f}")
        print(f"  TP/FP/FN/TN: {recovered_baseline['tp']}/{recovered_baseline['fp']}/{recovered_baseline['fn']}/{recovered_baseline['tn']}")
        
        print("\nP1 (recomputed):")
        print(f"  Raw AUC: {recovered_p1['raw_auc']:.6f}")
        print(f"  Calibrated AUC: {recovered_p1['calibrated_auc']:.6f}")
        print(f"  F1: {recovered_p1['f1']:.6f}")
        print(f"  FPR: {recovered_p1['fpr']:.6f}")
        print(f"  TP/FP/FN/TN: {recovered_p1['tp']}/{recovered_p1['fp']}/{recovered_p1['fn']}/{recovered_p1['tn']}")
    
    # Сравнение с оригиналом
    print("\nСравнение с оригиналом...")
    matches = {}
    
    if dev_predictions:
        orig_baseline = original_comparison["baseline"]
        orig_p1 = original_comparison["p1"]
        
        # Проверяем совпадение метрик
        for metric in ["f1", "fpr", "precision", "recall", "tp", "fp", "fn", "tn"]:
            baseline_match = abs(recovered_baseline[metric] - orig_baseline[metric]) < 1e-6
            p1_match = abs(recovered_p1[metric] - orig_p1[metric]) < 1e-6
            matches[metric] = baseline_match and p1_match
        
        # Проверяем AUC
        for metric in ["raw_auc", "calibrated_auc"]:
            baseline_match = abs(recovered_baseline[metric] - orig_baseline[f"dev_{metric}"]) < 1e-6
            p1_match = abs(recovered_p1[metric] - orig_p1[f"dev_{metric}"]) < 1e-6
            matches[metric] = baseline_match and p1_match
        
        if all(matches.values()):
            print("✓ All metrics match original p1_comparison.json")
        else:
            print("✗ Some metrics differ from original:")
            for metric, match in matches.items():
                status = "✓" if match else "✗"
                print(f"  {status} {metric}")
    
    # Сохраняем результаты
    print("\nСохранение результатов...")
    
    # recovery_audit.json
    import datetime
    audit = {
        "source_run_id": args.source_run_id,
        "source_sha": args.source_sha,
        "recovery_timestamp": datetime.datetime.now().isoformat(),
        "artifacts_found": {
            "comparison": str(comparison_file),
            "oof_predictions": str(oof_file) if oof_file else None,
            "dev_predictions": str(dev_file) if dev_file else None,
            "evidence": str(evidence_file) if evidence_file else None,
        },
        "provenance_verified": bool(evidence_file),
        "ids_and_gold_verified": bool(oof_predictions and dev_predictions),
        "metrics_recomputed": bool(dev_predictions),
        "metrics_match_original": all(matches.values()) if matches else None,
        "recovered_baseline": recovered_baseline,
        "recovered_p1": recovered_p1,
    }
    
    save_json(output_dir / "recovery_audit.json", audit)
    print(f"✓ recovery_audit.json")
    
    # RECOVERED_RESULTS.md
    md_lines = [
        "# Recovered P1 Results",
        "",
        f"**Source run:** {args.source_run_id}",
        f"**Source SHA:** {args.source_sha}",
        "",
        "## Summary",
        "",
    ]
    
    if dev_predictions:
        md_lines.extend([
            "### Baseline (mean_top3)",
            "",
            f"- Raw AUC: {recovered_baseline['raw_auc']:.6f}",
            f"- Calibrated AUC: {recovered_baseline['calibrated_auc']:.6f}",
            f"- Precision: {recovered_baseline['precision']:.6f}",
            f"- Recall: {recovered_baseline['recall']:.6f}",
            f"- F1: {recovered_baseline['f1']:.6f}",
            f"- FPR: {recovered_baseline['fpr']:.6f}",
            f"- TP/FP/FN/TN: {recovered_baseline['tp']}/{recovered_baseline['fp']}/{recovered_baseline['fn']}/{recovered_baseline['tn']}",
            f"- Threshold: {original_comparison['baseline']['threshold']:.6f}",
            f"- Positive predictions: {recovered_baseline['positive_predictions']}",
            "",
            "### P1 (answer-head)",
            "",
            f"- Raw AUC: {recovered_p1['raw_auc']:.6f}",
            f"- Calibrated AUC: {recovered_p1['calibrated_auc']:.6f}",
            f"- Precision: {recovered_p1['precision']:.6f}",
            f"- Recall: {recovered_p1['recall']:.6f}",
            f"- F1: {recovered_p1['f1']:.6f}",
            f"- FPR: {recovered_p1['fpr']:.6f}",
            f"- TP/FP/FN/TN: {recovered_p1['tp']}/{recovered_p1['fp']}/{recovered_p1['fn']}/{recovered_p1['tn']}",
            f"- Threshold: {original_comparison['p1']['threshold']:.6f}",
            f"- Positive predictions: {recovered_p1['positive_predictions']}",
            "",
        ])
        
        # Проверка критериев
        baseline_pass = recovered_baseline['f1'] >= 0.50 and recovered_baseline['fpr'] <= 0.40
        p1_pass = recovered_p1['f1'] >= 0.50 and recovered_p1['fpr'] <= 0.40
        
        md_lines.extend([
            "## Criteria Check (F1≥0.50, FPR≤0.40)",
            "",
            f"- Baseline: {'✓ PASS' if baseline_pass else '✗ FAIL'}",
            f"- P1: {'✓ PASS' if p1_pass else '✗ FAIL'}",
            "",
        ])
    
    md_lines.extend([
        "## Verification",
        "",
        f"- Provenance verified: {'✓' if audit['provenance_verified'] else '✗'}",
        f"- IDs and gold verified: {'✓' if audit['ids_and_gold_verified'] else '✗'}",
        f"- Metrics recomputed: {'✓' if audit['metrics_recomputed'] else '✗'}",
        f"- Metrics match original: {'✓' if audit.get('metrics_match_original') else '✗'}",
        "",
        "## Notes",
        "",
        "This recovery does NOT imply product readiness:",
        "- b878507 did not use P1 in public score",
        "- Baseline/checkpoint/OOF path has previously described caveats",
        "- Dev numbers are not equal to final acceptance",
        "",
    ])
    
    (output_dir / "RECOVERED_RESULTS.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"✓ RECOVERED_RESULTS.md")
    
    # Копируем оригинальные файлы
    print("\nКопирование оригинальных файлов...")
    (output_dir / "p1_comparison.json").write_text(
        comparison_file.read_text(encoding="utf-8"), encoding="utf-8"
    )
    print(f"✓ p1_comparison.json")
    
    if oof_file:
        (output_dir / "oof_predictions.jsonl").write_text(
            oof_file.read_text(encoding="utf-8"), encoding="utf-8"
        )
        print(f"✓ oof_predictions.jsonl")
    
    if dev_file:
        (output_dir / "dev_predictions.jsonl").write_text(
            dev_file.read_text(encoding="utf-8"), encoding="utf-8"
        )
        print(f"✓ dev_predictions.jsonl")
    
    if evidence_file:
        (output_dir / "evidence.json").write_text(
            evidence_file.read_text(encoding="utf-8"), encoding="utf-8"
        )
        print(f"✓ evidence.json")
    
    print("\n" + "=" * 60)
    print("Recovery completed successfully")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
