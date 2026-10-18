#!/usr/bin/env python3
"""P0-2: Полное сравнение трёх агрегаторов с корректным OOF-протоколом."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spanverify.dataset import read_pairs
from spanverify.engine import Verifier, WeightsBundle
from spanverify.features import load_feature_cache, set_feature_cache
from spanverify.oof import (
    _train_model_on_subset,
    compute_dev_answer_scores,
    compute_oof_answer_scores,
    compute_oof_risks,
    fit_answer_calibration,
)
from spanverify.train import auc_score, collect_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--mode", default="hf")
    parser.add_argument("--features-cache", required=True)
    parser.add_argument("--baseline-experiment", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-source-run-id")
    parser.add_argument("--measurement-run-id")
    parser.add_argument("--measurement-commit")

    args = parser.parse_args()
    start_time = time.time()

    print("=" * 60)
    print("P0-2: Полное сравнение трёх агрегаторов с OOF")
    print("=" * 60)

    # Загружаем бандл
    print("\n[1/6] Загрузка бандла...")
    baseline_path = Path(args.baseline_experiment)
    exp = json.loads(baseline_path.read_text())
    bundle = WeightsBundle.from_dict(exp["bundle"])
    print("✓ Бандл загружен")

    # Загружаем данные
    print("\n[2/6] Загрузка данных...")
    splits_dir = Path(args.splits)
    train_pairs = list(read_pairs(splits_dir / "train.jsonl"))
    dev_pairs = list(read_pairs(splits_dir / "dev.jsonl"))
    print(f"✓ Train: {len(train_pairs)} pairs")
    print(f"✓ Dev: {len(dev_pairs)} pairs")

    # Загружаем кеш
    print("\n[3/6] Загрузка кеша...")
    cache = load_feature_cache([args.features_cache], counting=True, strict=True)
    set_feature_cache(cache)
    print(f"✓ Кеш: {len(cache)} ключей")

    # Создаём верификатор
    print("\n[4/6] Создание верификатора...")
    verifier = Verifier(
        mode="hf",
        model_name=exp["model"],
        weights=bundle,
        features_cache=cache,
        span_expansion="clause_capped",
    )
    print("✓ Verifier создан")

    # OOF-обучение
    print("\n[5/6] OOF-обучение с 2 фолдами...")
    print("  Вычисление OOF-рисков...")
    sys.stdout.flush()

    risks_by_pair, oof_pair_ids, oof_labels, oof_doc_ids, oof_folds = compute_oof_risks(
        verifier, train_pairs, n_folds=2, seed=42
    )

    print("✓ OOF-риски вычислены")
    print(f"  Пар с OOF-рисками: {len(risks_by_pair)}")

    # Обучаем финальную модель
    print("\n  Обучение финальной модели на всём train...")
    all_samples, _ = collect_samples(verifier, train_pairs)
    final_weights, final_head = _train_model_on_subset(all_samples, entropy_quantile=verifier.entropy_quantile)
    print("✓ Финальная модель обучена")

    # Сравнение трёх агрегаторов
    print("\n[6/6] Сравнение трёх агрегаторов...")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregators = ["current", "max", "mean_top3"]
    results = {}

    for agg_name in aggregators:
        print(f"\n  Агрегатор: {agg_name}")

        # OOF scores
        oof_scores, oof_labels_agg, oof_pair_ids_agg = compute_oof_answer_scores(
            risks_by_pair, oof_pair_ids, oof_labels, aggregator=agg_name
        )

        print(f"    OOF AUC: {auc_score(oof_labels_agg, oof_scores):.4f}")

        # Обучение калибратора
        calibrator, threshold, threshold_report = fit_answer_calibration(oof_scores, oof_labels_agg, target_fpr=0.30)

        print(f"    Порог: {threshold:.4f}")
        print(f"    FPR на OOF: {threshold_report['fpr']:.4f}")

        # Оценка на dev
        dev_scores, dev_labels, dev_pair_ids = compute_dev_answer_scores(
            verifier, dev_pairs, final_weights, final_head, aggregator=agg_name
        )

        calibrated_dev = calibrator.transform(dev_scores)

        # Метрики на dev
        tp = fp = fn = tn = 0
        for cal_score, label in zip(calibrated_dev, dev_labels, strict=False):
            predicted = 1 if cal_score >= threshold else 0
            if predicted == 1 and label == 1:
                tp += 1
            elif predicted == 1 and label == 0:
                fp += 1
            elif predicted == 0 and label == 1:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        fpr = fp / (fp + tn) if fp + tn > 0 else 0.0

        raw_auc = auc_score(dev_labels, dev_scores)
        calibrated_auc = auc_score(dev_labels, calibrated_dev)

        print(f"    Dev F1: {f1:.4f}")
        print(f"    Dev FPR: {fpr:.4f}")
        print(f"    Dev Raw AUC: {raw_auc:.4f}")
        print(f"    Dev Calibrated AUC: {calibrated_auc:.4f}")

        results[agg_name] = {
            "oof_auc": auc_score(oof_labels_agg, oof_scores),
            "threshold": threshold,
            "threshold_report": threshold_report,
            "dev_tp": tp,
            "dev_fp": fp,
            "dev_fn": fn,
            "dev_tn": tn,
            "dev_precision": precision,
            "dev_recall": recall,
            "dev_f1": f1,
            "dev_fpr": fpr,
            "dev_raw_auc": raw_auc,
            "dev_calibrated_auc": calibrated_auc,
        }

    # Сохраняем результат
    print("\nСохранение результата...")
    duration = time.time() - start_time

    output = {
        "status": "success",
        "train_pairs": len(train_pairs),
        "dev_pairs": len(dev_pairs),
        "cache_keys": len(cache),
        "oof_pairs": len(oof_pair_ids),
        "aggregators": results,
        "duration_s": round(duration, 2),
    }

    result_path = out_dir / "p0_answer_comparison.json"
    result_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"✓ Результат сохранен: {result_path}")

    # Markdown отчёт
    md_lines = [
        "# P0-2: Сравнение трёх агрегаторов с OOF",
        "",
        f"**Train pairs:** {len(train_pairs)}",
        f"**Dev pairs:** {len(dev_pairs)}",
        f"**Cache keys:** {len(cache)}",
        f"**OOF pairs:** {len(oof_pair_ids)}",
        "",
        "## Результаты",
        "",
        "| Агрегатор | OOF AUC | Dev Raw AUC | Dev Cal AUC | F1 | FPR | TP/FP/FN/TN | Порог |",
        "|-----------|---------|-------------|-------------|----|-----|-------------|-------|",
    ]

    for agg_name in aggregators:
        r = results[agg_name]
        md_lines.append(
            f"| {agg_name} | {r['oof_auc']:.4f} | {r['dev_raw_auc']:.4f} | {r['dev_calibrated_auc']:.4f} | "
            f"{r['dev_f1']:.4f} | {r['dev_fpr']:.4f} | {r['dev_tp']}/{r['dev_fp']}/{r['dev_fn']}/{r['dev_tn']} | "
            f"{r['threshold']:.4f} |"
        )

    md_lines.extend(
        [
            "",
            f"**Длительность:** {duration:.1f}s",
        ]
    )

    md_path = out_dir / "p0_answer_comparison.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"✓ Markdown отчёт: {md_path}")

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
