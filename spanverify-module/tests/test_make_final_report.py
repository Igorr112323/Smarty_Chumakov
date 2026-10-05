"""Тесты сборки итогового файла результатов прогона.

Отчёт опасен ровно одним: в нём может появиться правдоподобное число, которого никто
не измерял. Поэтому тесты проверяют не вёрстку, а два свойства:

1. отсутствующее измерение превращается в ``null``, а не в величину;
2. отметка «достигнут» ставится сравнением с целью, а не переписыванием цели.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.make_final_report import (
    NO,
    TARGETS,
    collect_values,
    deliverable_rows,
    get,
    load_context,
    num,
    read_json,
    render,
)


def _context(**parts: object) -> dict:
    """Контекст отчёта: по умолчанию ничего не измерено."""
    base: dict = {
        "root": Path("."),
        "metrics": None,
        "audit": None,
        "a3": None,
        "a3_manifest": None,
        "ocr_bench": None,
        "availability": None,
        "a3_hf": None,
        "a3_demo": None,
        "pilot": None,
        "ext_qa_hf": None,
        "ext_rushallu_hf": None,
        "robustness": None,
        "functional": None,
    }
    base.update(parts)
    return base


def test_num_never_invents_a_value() -> None:
    """Отсутствие числа печатается как null, а не как ноль."""
    assert num(None) == NO
    assert num(0) == "0"
    assert num(0.0) == "0"
    assert num(0.12345) == "0.1235"
    assert num(1.0) == "1"


def test_empty_repository_produces_report_full_of_nulls(tmp_path: Path) -> None:
    """Если не измерено ничего, в отчёте нет ни одного выдуманного числа."""
    context = load_context(tmp_path)
    context["root"] = tmp_path
    text = render(context)

    assert "# Результаты прогона" in text
    assert "Достигнуто показателей: **0 из" in text
    # Все целевые показатели обязаны быть перечислены даже без измерений.
    for target in TARGETS:
        assert target["name"] in text
    assert "null" in text


def test_targets_are_compared_not_restated() -> None:
    """Отметка «достигнут» зависит от значения, а не от названия показателя."""
    context = _context(
        a3={
            "documents_total": 130,
            "facts_found_total": 1300,
            "by_level": {"федеральный": 70, "региональный": 60},
        },
        a3_manifest={
            "pairs": 1378,
            "shared_groups": 0,
            "balance": {"clean_share": 0.4572, "counts": {"faithful": 630, "contradiction": 168, "missing": 84}},
        },
        metrics={
            "tests": {"coverage_percent": 87.75},
            "corpus_a": {
                "by_mode": {"faithful": {"verdict_fpr": 0.0247}},
                "by_split": {"test": {"spans": {"narrow": {"f1_iou_0_5": 0.7387, "width_ratio": 1.0}}}},
            },
            "external_tests": {"hf_runs": 0},
        },
    )
    values = collect_values(context)
    assert values["a3_documents"] == 130
    assert values["a3_min_per_mode"] == 84  # faithful в минимум не входит: это чистые пары
    assert values["hf_runs"] == 0

    by_name = {target["name"]: target for target in TARGETS}
    assert by_name["Документов реальных опубликованных актов"]["cmp"](values["a3_documents"]) is True
    assert by_name["Прогонов в режиме hf"]["cmp"](values["hf_runs"]) is False
    assert by_name["Доля чистых пар"]["cmp"](values["a3_clean_share"]) is True
    # Доля вне коридора 0.40–0.50 не засчитывается, даже если она «хорошая».
    assert by_name["Доля чистых пар"]["cmp"](0.9) is False
    # Пересечение документов между частями недопустимо ни при каком размере корпуса.
    assert by_name["Общих документов между частями train/dev/test"]["cmp"](1) is False


def test_mode_column_is_present_for_every_metric_table() -> None:
    """В таблице метрик обязателен столбец «Режим»: demo нельзя выдать за hf."""
    context = _context(
        metrics={
            "corpus_a": {"by_split": {"test": {"tokens": {"f1": 0.73}, "verdicts": {"f1": 0.96}}}},
            "external_tests": {"runs": {"набор": {"mode": "demo", "tokens": {"f1": 0.2}}}},
        }
    )
    text = render(context)
    header = next(line for line in text.splitlines() if line.startswith("| Корпус | Часть | Режим"))
    assert "Режим" in header
    assert "никогда не выдаются за" in text


def test_hf_row_is_null_until_the_run_exists() -> None:
    """Строка режима hf остаётся null, пока прогон не выполнен, и заполняется после."""
    without = render(_context(metrics={"corpus_a": {}}))
    hf_line = next(line for line in without.splitlines() if "| **hf** |" in line)
    assert hf_line.count("null") >= 5

    with_run = render(
        _context(
            metrics={"corpus_a": {}},
            a3_hf={
                "tokens": {"precision": 0.9, "recall": 0.8, "f1": 0.85, "fpr": 0.01, "auc": 0.95},
                "verdicts": {"f1": 0.9, "fpr": 0.02},
            },
        )
    )
    hf_line = next(line for line in with_run.splitlines() if "| **hf** |" in line)
    assert "0.85" in hf_line
    assert "null" not in hf_line


def test_deliverables_separate_ready_from_needs_human() -> None:
    """Список документов различает «готово» и «нужны данные человека»."""
    rows = deliverable_rows(Path(__file__).resolve().parents[2])
    statuses = {row["status"] for row in rows}
    assert statuses <= {"готово", "не готово", "готово, но нужны данные человека", "нужны данные человека"}
    # Документы с реквизитами договора не могут быть помечены просто «готово».
    by_path = {row["path"]: row for row in rows}
    assert "человека" in by_path["docs/РИД_программа/"]["status"]
    assert "человека" in by_path["docs/РИД_полезная_модель/"]["status"]


def test_legal_basis_and_robots_decision_are_stated() -> None:
    """Правовое основание и спорное место в robots.txt обязаны быть в отчёте."""
    text = render(_context())
    assert "п. 6 ст. 1259 ГК РФ" in text
    assert "RFC 9309" in text
    assert "вынесено человеку на" in text
    assert "/Search" in text


def test_reproduction_command_is_a_single_block() -> None:
    """В отчёте есть ровно один блок команды воспроизведения, и он исполним по виду."""
    text = render(_context())
    assert text.count("```bash") == 1
    assert "python scripts/make_final_report.py" in text
    assert "python scripts/check_numbers.py" in text


def test_read_json_and_get_are_tolerant(tmp_path: Path) -> None:
    """Битый или отсутствующий файл не роняет сборку отчёта."""
    assert read_json(tmp_path / "нет.json") is None
    broken = tmp_path / "broken.json"
    broken.write_text("{не json", encoding="utf-8")
    assert read_json(broken) is None

    good = tmp_path / "ok.json"
    good.write_text(json.dumps({"a": {"b": 1}}), encoding="utf-8")
    assert get(read_json(good), "a.b") == 1
    assert get(read_json(good), "a.нет", "по умолчанию") == "по умолчанию"


def test_project_report_matches_measured_numbers() -> None:
    """Отчёт проекта собирается на настоящих файлах и совпадает с единым файлом чисел."""
    from scripts.make_final_report import ROOT

    context = load_context(ROOT)
    if not context["metrics"]:
        return
    text = render(context)
    coverage = get(context["metrics"], "tests.coverage_percent")
    assert num(coverage, 2) in text
    documents = get(context["a3"] or {}, "documents_total")
    if documents is not None:
        assert str(documents) in text
