"""Тесты журнала состояния и итогового отчёта (шаг 4-1).

Проверяется то, что легко сломать незаметно: цель/факт сравниваются числами, а не
строками; отсутствие измерения — это `null` с причиной, а не «выполнено»; итоговый
отчёт собирается из файлов и содержит ровно двенадцать разделов; при отсутствии
файла печатается «нет данных», а не выдуманное число.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import collect_status  # noqa: E402
import make_final_report  # noqa: E402


def _write(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def test_targets_report_compares_numbers_and_keeps_null(tmp_path: Path) -> None:
    """Цель/факт: сравнение числовое, отсутствие измерения — null и причина."""
    source = _write(
        tmp_path / "targets_source.json",
        {
            "items": [
                {
                    "name": "Качество по токенам",
                    "target": "F1 >= 0.90",
                    "target_value": 0.9,
                    "compare": ">=",
                    "metric_path": "corpus_a.by_split.test.tokens.f1",
                },
                {
                    "name": "Покрытие автотестами",
                    "target": ">= 85 %",
                    "target_value": 85,
                    "compare": ">=",
                    "metric_path": None,
                },
            ]
        },
    )
    metrics = {"corpus_a": {"by_split": {"test": {"tokens": {"f1": 0.7338}}}}}
    report = collect_status.targets_report(source, metrics, {"coverage_percent": 88.0})
    quality, coverage = report["items"]
    assert quality["fact"] == 0.7338 and quality["achieved"] is False
    assert coverage["fact"] == 88.0 and coverage["achieved"] is True
    assert "reason" in coverage


def test_targets_report_without_fact_has_reason(tmp_path: Path) -> None:
    """Нет измерения — нет «достигнуто»: в отчёте null и объяснение."""
    source = _write(
        tmp_path / "targets_source.json",
        {
            "items": [
                {
                    "name": "Корпус реальных актов",
                    "target": ">= 120 актов",
                    "target_value": 120,
                    "compare": ">=",
                    "metric_path": "corpus_a3_real.pairs",
                }
            ]
        },
    )
    report = collect_status.targets_report(source, {"corpus_a3_real": {"available": False}}, {})
    item = report["items"][0]
    assert item["fact"] is None
    assert item["achieved"] is None
    assert item["reason"], "нужна причина, почему числа нет"


def test_review_queue_counts_only_marked_entries(tmp_path: Path) -> None:
    """Очередь ручной проверки: число отмеченных считает человек, а не скрипт."""
    queue = tmp_path / "data" / "review_queue"
    queue.mkdir(parents=True)
    rows = [
        {"id": "a", "answer": "1"},
        {"id": "b", "answer": "2", "reviewed": True},
        {"id": "c", "answer": "3"},
    ]
    (queue / "corpus_a_10pct.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    report = collect_status.review_queue_report(tmp_path)
    assert report["total"] == 3
    assert report["checked"] == 1


def test_final_report_has_twelve_sections_and_no_invented_numbers(tmp_path: Path) -> None:
    """Итоговый отчёт: 12 разделов; нет файла — «нет данных» с именем файла."""
    sections = [
        make_final_report.section_1(None),
        make_final_report.section_2(None),
        make_final_report.section_3(None),
        make_final_report.section_4(None, None),
        make_final_report.section_5(None),
        make_final_report.section_6(None, None),
        make_final_report.section_7(None),
        make_final_report.section_8(None, None),
        make_final_report.section_9(None, None),
        make_final_report.section_10(None),
        make_final_report.section_11(None, None),
        make_final_report.section_12(),
    ]
    joined = "\n".join("\n".join(part) for part in sections)
    assert joined.count("нет данных") >= 8, "каждый раздел без источника обязан это сказать"
    for number in ("0.953", "0.7338", "0.4115"):
        assert number not in joined, "числа не выдумываются из воздуха"
    assert "нет файла" in joined
    headers = [line for part in sections for line in part if line.startswith("## ")]
    assert len(headers) == 12, headers


def test_train_hf_bundle_requires_model() -> None:
    """Без --model скрипт обучения hf-параметров завершается понятной ошибкой."""
    spec = importlib.util.spec_from_file_location("train_hf_bundle_under_test", ROOT / "scripts" / "train_hf_bundle.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    try:
        module.main([])
    except SystemExit as exit_error:  # argparse: обязательный --model
        assert exit_error.code not in (0, None)
    else:  # pragma: no cover - защита от изменения контракта
        raise AssertionError("ожидалась ошибка о обязательном аргументе")
