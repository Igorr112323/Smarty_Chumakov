"""Единый файл чисел и его сверка с документами (исправление P0-3 аудита).

Смысл набора: числа в документах больше не берутся из памяти. Есть один файл
``reports/METRICS.json``, собранный прогоном кода, и есть проверка, которая
падает, если документ разошёлся с ним. Тесты ниже защищают это правило и
честные оговорки, которые нельзя потерять при правках.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parent
METRICS_PATH = MODULE_ROOT / "reports" / "METRICS.json"


def _load_script(name: str):
    """Загрузить скрипт из ``scripts/`` по пути (там нет пакета)."""
    path = MODULE_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


def _metrics() -> dict:
    """Прочитать единый файл чисел; без него проверки бессмысленны — пропуск."""
    if not METRICS_PATH.is_file():
        pytest.skip("reports/METRICS.json отсутствует: сначала scripts/collect_metrics.py")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def test_metrics_json_labels_demo_corpus_as_synthetic() -> None:
    """Оговорка о синтетике и требование реальных данных обязаны быть в файле чисел."""
    metrics = _metrics()
    disclaimer = metrics["disclaimer"]
    assert "синтетическ" in disclaimer.lower()
    assert "1200" in disclaimer, "нужно число пар для научного подтверждения"
    assert metrics["demo"]["demo_answer_example"]["verdict"] == "likely_hallucination"


def test_metrics_json_records_group_split_without_shared_subjects() -> None:
    """Сплит в отчётных числах обязан быть групповым и без общих субъектов (P1-3)."""
    metrics = _metrics()
    split = metrics["demo"]["split"]
    assert split["grouped"] is True
    assert split["shared_groups"] == 0
    assert split["train_pairs"] + split["test_pairs"] == 240


def test_metrics_json_keeps_cross_corpus_drop() -> None:
    """Падение качества на чужом генераторе нельзя спрятать (P0-2).

    Если кто-то удалит раздел или подменит числа, тест упадёт: перенос обязан
    показывать и падение F1, и выход FPR за порог 0.10.
    """
    metrics = _metrics()
    cross = metrics.get("cross_corpus")
    assert cross is not None, "кросс-корпусный тест обязан быть в файле чисел"
    assert cross["cross_corpus_f1"] < cross["in_corpus_f1"] - 0.05
    assert cross["cross_fpr"] > 0.10, "FPR на чужом корпусе обязан быть показан честно"


def test_metrics_json_exposes_strict_span_quality() -> None:
    """Строгая метрика фрагментов (узкая разметка) обязана публиковаться рядом с мягкой (P1-4)."""
    metrics = _metrics()
    for name in ("in_corpus", "whole_corpus"):
        spans = metrics["demo"][name]["spans"]
        assert 0.0 <= spans["strict_f1"] < spans["soft_f1"] <= 1.0, name
        assert spans["coverage"] >= 0.9, name


def test_check_numbers_reports_stale_value() -> None:
    """Проверка чисел действительно ловит расхождение, а не «всегда зелёная»."""
    checker = _load_script("check_numbers")
    metrics = _make_fake_metrics()
    stale = Path("FAKE_DOC.md")
    stale.write_text("token F1 0.959 при FPR 0.002\n", encoding="utf-8")
    try:
        problems = checker.check_docs([stale], metrics)
    finally:
        stale.unlink(missing_ok=True)
    assert any("token_f1" in problem for problem in problems)


def test_check_numbers_requires_itog_sections(tmp_path: Path) -> None:
    """ИТОГ.md обязан содержать разделы «заявлено/факт», «не проверено» и «3 шага» (приёмка E)."""
    checker = _load_script("check_numbers")
    (tmp_path / "ИТОГ.md").write_text("# Итог\n\nБез обязательных разделов.\n", encoding="utf-8")
    problems = " ".join(checker.check_required_sections(tmp_path))
    assert "заявлено / факт" in problems
    assert "не проверено" in problems
    assert "трёх шагов" in problems


def test_documents_match_metrics_file() -> None:
    """Сквозная проверка: README, ИТОГ и docs/ сходятся с reports/METRICS.json."""
    checker = _load_script("check_numbers")
    metrics = _metrics()
    docs = [REPO_ROOT / name for name in checker.DEFAULT_DOCS if (REPO_ROOT / name).is_file()]
    assert len(docs) >= 6, "проверять нужно все основные документы"
    assert checker.check_docs(docs, metrics) == []


def test_pilot_publishes_control_thresholds_and_draws_png(tmp_path: Path) -> None:
    """Пилот обязан публиковать контроль корпуса (0.8/0.3) и рисовать PNG без зависимостей."""
    pilot = _load_script("pilot_rugpt3small")
    source = (MODULE_ROOT / "scripts" / "pilot_rugpt3small.py").read_text(encoding="utf-8")
    assert "grounded_copy_min" in source and "unsupported_copy_max" in source
    assert '"wording"' in source, "нужна формулировка «модель реальная, корпус синтетический»"

    payload = {
        "auc": {
            "last": {
                "attention_entropy": {"auc_oriented": 0.52},
                "ctx_attention_mass": {"auc_oriented": 0.48},
                "embedding_density": {"auc_oriented": 0.5},
            }
        }
    }
    png = pilot.write_auc_png(payload, tmp_path / "pilot.png")
    data = png.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(data) > 500, "картинка не должна быть пустышкой"


def _make_fake_metrics() -> dict:
    """Минимальный файл чисел для проверки самого чекера (значения нарочно «правильные»)."""
    return {
        "meta": {"version": "1.2.0"},
        "tests": {"collected": 226, "coverage_percent": 86.0},
        "demo": {
            "split": {"grouped": True, "shared_groups": 0, "train_pairs": 168, "test_pairs": 72},
            "in_corpus": {
                "tokens": {"f1": 0.953, "fpr": 0.002, "auc": 0.996, "n": 594},
                "spans": {"strict_f1": 0.28, "soft_f1": 1.0, "coverage": 1.0},
                "answers": {"f1": 1.0, "fpr": 0.0, "auc": 1.0},
            },
            "whole_corpus": {
                "tokens": {"f1": 0.951, "fpr": 0.005, "auc": 0.997, "n": 2001},
                "spans": {"strict_f1": 0.315, "soft_f1": 1.0, "coverage": 1.0},
                "answers": {"f1": 0.996, "fpr": 0.0, "auc": 0.996},
            },
            "validation": {"f1": 0.953, "fpr": 0.002, "auc": 0.996},
            "participation": {"auc_out_of_fold": 1.0, "rows": 8179},
        },
        "cross_corpus": {"in_corpus_f1": 0.953, "cross_corpus_f1": 0.731, "cross_fpr": 0.144},
        "pilot": None,
    }
