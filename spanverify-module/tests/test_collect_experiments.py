"""Единый файл чисел читает прогонные файлы: эксперименты и внешние срезы hf.

Смысл набора: числа из ``reports/experiments/*/experiment.json`` и внешних
срезов ``reports/ext_*_hf.json`` обязаны попадать в ``METRICS.json`` тем же
скриптом ``collect_metrics.py``, иначе документы нечем сверять с прогонными
файлами. Пока числа не в METRICS — в документы они не попадают.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = MODULE_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


def test_hf_experiments_block_reads_run_files() -> None:
    """Все ``experiment.json`` и ``pilot.json`` из прогонных каталогов читаются."""
    collect = _load_script("collect_metrics")
    block = collect._hf_experiments_block()
    experiments = MODULE_ROOT / "reports" / "experiments"
    if not experiments.is_dir() or not list(experiments.rglob("*.json")):
        import pytest

        pytest.skip("прогонных файлов нет")
    assert block and block["available"] is True
    runs = block["runs"]
    assert "a3" in runs, "прогон по корпусу A3 обязан быть в едином файле чисел"
    a3 = runs["a3"]
    assert a3["kind"] == "experiment"
    assert a3["mode"] == "hf"
    assert a3["seed"] == 42
    assert isinstance(a3["tokens"]["f1"], float)
    assert a3["pairs"] == 1200


def test_experiment_block_maps_official_test_fields(tmp_path: Path) -> None:
    """Новые поля прогона (официальный тест, кривая маски) проходят в блок."""
    collect = _load_script("collect_metrics")
    payload = {
        "dataset": "data/corpus_a3/pairs.jsonl",
        "mode": "hf",
        "model": "m",
        "seed": 42,
        "run_id": "123",
        "duration_s": 1.0,
        "corpus": {"pairs": 10},
        "gate": "НЕ достигнут",
        "metrics": {
            "tokens": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "fpr": 0.1, "auc": 0.8},
            "spans": {"f1": 0.2, "recall_containment": 0.4, "mean_width_ratio": 5.0},
            "answers": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "fpr": 0.0, "auc": 0.5, "threshold": 0.7},
            "verdicts": {"tp": 1, "fp": 1, "fn": 1, "tn": 1, "precision": 0.5, "recall": 0.5, "f1": 0.5, "fpr": 0.5},
        },
        "splits": {"sizes": {"train": 5, "test": 5}, "rule": "только на официальной обучающей части"},
        "test_official": {
            "pairs": 5,
            "tokens": {"f1": 0.4, "fpr": 0.05, "auc": 0.7, "precision": 0.4, "recall": 0.4},
            "spans": {"strict_f1_iou_0_5": 0.1},
            "answers": {"f1": 0.0},
            "verdicts": {"f1": 0.6, "fpr": 0.4},
        },
        "gate_test_official": "НЕ достигнут",
        "by_type_test": {"missing": {"token_f1": 0.3, "token_recall": 0.2, "pairs": 2}},
        "clean_pairs": {"pairs": 3, "token_fpr": 0.01, "verdict_fpr": 0.33, "flagged_share": 0.33},
        "mask_scan": {
            "signal": "logreg",
            "target_fpr": 0.1,
            "points_total": 10,
            "points_within_target": 4,
            "selected": {"span_z": 2.0, "token_f1": 0.2, "token_fpr": 0.05, "token_recall": 0.3},
            "best_within_target": {"span_z": 2.0, "token_f1": 0.2, "token_fpr": 0.05, "token_recall": 0.3},
            "adopted": False,
            "statement": "Точка лучше текущей при FPR ≤ 0.1 не найдена.",
        },
        "verdict_rule_comparison": {
            "test_pairs": 5,
            "rule_any_span": {"f1": 0.66, "fpr": 0.89, "precision": 0.5, "recall": 0.8},
            "rule_min_two_tokens": {"f1": 0.6, "fpr": 0.4, "precision": 0.5, "recall": 0.7},
        },
        "bundle": {"threshold": 0.7, "span_z": 2.0, "span_floor": 0.05, "span_cap": 0.5, "weights": {}, "meta": {}},
    }
    block = collect._experiment_block(payload)
    assert block["test_official"]["tokens"]["f1"] == 0.4
    assert block["gate_test_official"] == "НЕ достигнут"
    assert block["by_type_test"]["missing"]["token_f1"] == 0.3
    assert block["clean_pairs"]["verdict_fpr"] == 0.33
    assert block["mask_scan"]["adopted"] is False
    assert block["verdict_rule_comparison"]["rule_min_two_tokens"]["fpr"] == 0.4
    assert block["splits"]["sizes"]["test"] == 5


def test_hf_external_slices_block_reads_committed_slice() -> None:
    """Срез ``ext_ragtruth_qa_hf.json`` читается вместе с оговоркой про срез."""
    collect = _load_script("collect_metrics")
    block = collect._hf_external_slices_block()
    slice_path = MODULE_ROOT / "reports" / "ext_ragtruth_qa_hf.json"
    if not slice_path.is_file():
        import pytest

        pytest.skip("среза нет в репозитории")
    assert block and block["slices"]
    item = block["slices"]["ext_ragtruth_qa_hf.json"]
    raw = json.loads(slice_path.read_text(encoding="utf-8"))
    assert item["mode"] == "hf"
    assert item["pairs"] == raw["pairs"]
    assert item["note"] and "СРЕЗ" in item["note"], "оговорка «это срез» обязательна"


def test_check_numbers_allows_experiment_and_slice_values() -> None:
    """Числа прогонных блоков законны для сверки документов."""
    checker = _load_script("check_numbers")
    metrics = {
        "meta": {"version": "1.3.0"},
        "demo": {
            "in_corpus": {
                "tokens": {"f1": 0.95, "fpr": 0.01, "auc": 0.99, "n": 100},
                "answers": {"auc": 1.0},
            },
            "validation": {"f1": 0.94, "fpr": 0.01, "auc": 0.99},
            "participation": {"auc_out_of_fold": 0.9, "rows": 10},
        },
        "tests": {},
        "hf_experiments": {
            "runs": {
                "a3": {
                    "kind": "experiment",
                    "tokens": {"f1": 0.2031, "fpr": 0.0701, "auc": 0.8446},
                    "answers": {"f1": 0.0, "fpr": 0.0, "auc": 0.5},
                    "verdicts": {"f1": 0.662, "fpr": 0.8926, "precision": 0.5396, "recall": 0.8561},
                    "spans": {"strict_f1": 0.0942, "coverage": 0.4742},
                    "test_official": {"tokens": {"f1": 0.21, "fpr": 0.06, "auc": 0.85}},
                    "by_type_test": {"missing": {"token_f1": 0.3, "token_recall": 0.4, "verdict_fpr": 0.2}},
                    "clean_pairs": {"token_fpr": 0.015, "verdict_fpr": 0.3, "flagged_share": 0.3},
                    "mask_scan": {
                        "selected": {"token_f1": 0.2, "token_fpr": 0.05, "token_recall": 0.36},
                        "best_within_target": {"token_f1": 0.2, "token_fpr": 0.05, "token_recall": 0.36},
                    },
                    "verdict_rule_comparison": {
                        "rule_any_span": {"f1": 0.66, "fpr": 0.89, "precision": 0.54, "recall": 0.86},
                        "rule_min_two_tokens": {"f1": 0.61, "fpr": 0.55, "precision": 0.55, "recall": 0.7},
                    },
                }
            }
        },
        "hf_external_slices": {
            "slices": {
                "ext_ragtruth_qa_hf.json": {
                    "tokens": {"f1": 0.0, "fpr": 0.0, "auc": 0.7975},
                    "verdicts": {"f1": 0.2727, "fpr": 0.4167},
                    "their": {"rouge1": 0.5442, "accuracy": 0.525},
                }
            }
        },
    }
    allowed = checker._allowed(metrics)
    assert "0.2031" in allowed["token_f1"]
    assert "0.662" in allowed["token_f1"]
    assert "0.8926" in allowed["fpr"]
    assert "0.7975" in allowed["auc"]
    assert "0.5442" in allowed["token_f1"]
