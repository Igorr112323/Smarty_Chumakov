"""Аудит знаков признаков и влияния знака плотности на качество правила.

Зачем скрипт появился. В формуле риска признаки опоры должны входить с обратным
знаком: у подтверждённого токена опора высокая, а риск низкий. В предыдущей
редакции ``combine`` плотность ``embedding_density`` входила со знаком «плюс»,
то есть высокая близость к документу УВЕЛИЧИВАЛА риск. Проверка одиночных
признаков это подтверждает прямым измерением:

* ``ctx_attention_mass``: AUC = 0.215 как риск против 0.785 как опора (1 − x);
* ``embedding_density``: AUC = 0.127 как риск против 0.873 как опора.

Скрипт ничего не меняет: он измеряет оба варианта формулы на одних и тех же
признаках одного корпуса, пишет отчёт ``reports/SIGN_AUDIT.json`` и печатает
разницу. Так исправление знака подтверждается прогоном, а не рассуждением.

Запуск::

    python scripts/sign_audit.py --dataset data/corpus_a/splits/train.jsonl \\
        --out reports/SIGN_AUDIT.json --seed 42
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spanverify.dataset import read_pairs  # noqa: E402
from spanverify.engine import Verifier  # noqa: E402
from spanverify.features import (
    DEFAULT_WEIGHTS,
    FEATURE_NAMES,
    scale_robust,
)  # noqa: E402
from spanverify.train import auc_score, collect_samples  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

#: Веса «до исправления знака» — те, с которыми считались числа прежних
#: релизов (масса 0.9, плотность 0.0/0.1). Нужны только для сравнения формул.
LEGACY_WEIGHTS = {
    "attention_entropy": 0.1,
    "ctx_attention_mass": 0.9,
    "embedding_density": 0.0,
}


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        return "unknown"


def _token_metrics(labels: list[int], risks: list[float], threshold: float) -> dict:
    flags = [risk >= threshold for risk in risks]
    tp = sum(1 for label, flag in zip(labels, flags, strict=False) if label and flag)
    fp = sum(1 for label, flag in zip(labels, flags, strict=False) if not label and flag)
    fn = sum(1 for label, flag in zip(labels, flags, strict=False) if label and not flag)
    tn = sum(1 for label, flag in zip(labels, flags, strict=False) if not label and not flag)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": round(threshold, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fp / (fp + tn), 4) if fp + tn else 0.0,
        "auc": round(auc_score(labels, risks), 4),
    }


def measure(dataset: str, seed: int) -> dict:
    pairs = list(read_pairs(dataset))
    verifier = Verifier(mode="demo")
    samples, stats = collect_samples(verifier, pairs)
    labels = [sample.label for sample in samples]

    per_feature: dict[str, dict] = {}
    for name in FEATURE_NAMES:
        values = [sample.features.get(name, 0.0) for sample in samples]
        per_feature[name] = {
            "auc_as_risk": round(auc_score(labels, values), 4),
            "auc_as_support": round(auc_score(labels, [1.0 - value for value in values]), 4),
        }
    for name in ("ctx_attention_mass_lift", "ctx_support_distance", "ctx_sim_contrast"):
        values = [sample.features.get(name) for sample in samples]
        if any(value is not None for value in values):
            clean = [float(value or 0.0) for value in values]
            per_feature[name] = {
                "auc_as_risk": round(auc_score(labels, clean), 4),
                "auc_as_support": round(auc_score(labels, [1.0 - value for value in clean]), 4),
            }

    # Две формулы правила на одних и тех же признаках: так измеряется вклад
    # исправления знака, а не «стало лучше» словами.
    entropy = [sample.features["attention_entropy"] for sample in samples]
    mass = [sample.features["ctx_attention_mass"] for sample in samples]
    density = [sample.features["embedding_density"] for sample in samples]
    e_hat = scale_robust(entropy, 1.0)
    legacy_total = sum(LEGACY_WEIGHTS.values()) or 1.0
    legacy_risks = [
        min(
            1.0,
            max(
                0.0,
                (
                    LEGACY_WEIGHTS["attention_entropy"] * e_hat[index]
                    + LEGACY_WEIGHTS["ctx_attention_mass"] * (1.0 - min(1.0, max(0.0, mass[index])))
                    + LEGACY_WEIGHTS["embedding_density"] * min(1.0, max(0.0, density[index]))
                )
                / legacy_total,
            ),
        )
        for index in range(len(samples))
    ]
    corrected_risks = [sample.features["risk"] for sample in samples]

    threshold = 0.5
    return {
        "pairs": stats["pairs"],
        "tokens": len(samples),
        "positive_tokens": stats["positive"],
        "per_feature": per_feature,
        "legacy_formula": {
            "weights": LEGACY_WEIGHTS,
            "note": "плотность входит со знаком «плюс» — как в редакции до исправления",
            "metrics": _token_metrics(labels, legacy_risks, threshold),
        },
        "corrected_formula": {
            "weights": {name: DEFAULT_WEIGHTS[name] for name in FEATURE_NAMES},
            "note": "все признаки опоры входят как 1 − x",
            "metrics": _token_metrics(labels, corrected_risks, threshold),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Аудит знаков признаков")
    parser.add_argument("--dataset", default="data/corpus_a/splits/train.jsonl")
    parser.add_argument("--out", default="reports/SIGN_AUDIT.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.time()
    measured = measure(args.dataset, args.seed)
    payload = {
        "kind": "sign_audit",
        "commit": git_commit(),
        "seed": args.seed,
        "mode": "demo",
        "dataset": args.dataset,
        "command": f"python scripts/sign_audit.py --dataset {args.dataset} --out {args.out} --seed {args.seed}",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_s": round(time.time() - started, 2),
        "note": (
            "замер на обучающей части; порог 0.5 фиксирован для сравнения формул, "
            "приёмка считается на отложенной части отдельной командой"
        ),
        "metrics": measured,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{out}: пар {measured['pairs']}, токенов {measured['tokens']}, положительных {measured['positive_tokens']}")
    for name, values in measured["per_feature"].items():
        print(f"  {name:26s} AUC(риск)={values['auc_as_risk']:.4f}  AUC(опора)={values['auc_as_support']:.4f}")
    legacy = measured["legacy_formula"]["metrics"]
    corrected = measured["corrected_formula"]["metrics"]
    print(f"  формула до исправления: F1={legacy['f1']} FPR={legacy['fpr']} AUC={legacy['auc']}")
    print(f"  формула после исправления: F1={corrected['f1']} FPR={corrected['fpr']} AUC={corrected['auc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
