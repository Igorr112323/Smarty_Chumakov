"""Тесты реестра незакрытых работ.

Главное свойство реестра: отметка «закрыт» ставится только числом. Поэтому тесты
проверяют не вёрстку, а решающие правила — и в обе стороны: на числах, которые
критерий удовлетворяют, и на числах, которые не удовлетворяют. Отдельно проверяется,
что отсутствие числа даёт «нет данных», а не «закрыт».
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.make_todo_audit import (
    CLOSED,
    NO_DATA,
    OPEN,
    REGISTRY,
    check_baseline,
    check_corpus_a3,
    check_false_flags,
    check_hf_runs,
    check_missing,
    check_partial,
    check_pilot_auc,
    check_review_queue,
    check_span_quality,
    evaluate,
    load_context,
    render,
)


def _context(metrics: dict, **extra: object) -> dict:
    """Минимальный контекст проверки (как его собирает load_context)."""
    base: dict = {
        "root": Path("."),
        "metrics": metrics,
        "a3": {},
        "a3_pairs": 0,
        "ocr_bench": {},
        "availability": {},
        "review": {},
    }
    base.update(extra)
    return base


def test_hf_runs_rule_reacts_to_number() -> None:
    """Пункт о режиме hf закрывается только при ненулевом числе прогонов."""
    zero = _context({"external_tests": {"hf_runs": 0, "demo_runs": 4}})
    assert check_hf_runs(zero)[0] == OPEN

    one = _context({"external_tests": {"hf_runs": 2, "demo_runs": 4}})
    status, detail = check_hf_runs(one)
    assert status == CLOSED
    assert "2" in detail

    assert check_hf_runs(_context({}))[0] == NO_DATA


def test_pilot_rule_requires_confidence_interval_above_chance() -> None:
    """Пилот закрывается, только если нижняя граница доверительного интервала выше 0.5."""
    chance = _context({"pilot": {"auc": {"8": {"attention": {"auc_oriented": 0.51, "ci_low": 0.44, "ci_high": 0.58}}}}})
    assert check_pilot_auc(chance)[0] == OPEN

    good = _context({"pilot": {"auc": {"8": {"attention": {"auc_oriented": 0.72, "ci_low": 0.61, "ci_high": 0.83}}}}})
    assert check_pilot_auc(good)[0] == CLOSED

    assert check_pilot_auc(_context({}))[0] == NO_DATA


def test_span_rule_needs_both_f1_and_width() -> None:
    """Границы фрагментов: нужен и высокий F1, и узкая разметка — одного мало."""
    wide = _context(
        {"corpus_a": {"by_split": {"test": {"spans": {"narrow": {"f1_iou_0_5": 0.8, "width_ratio": 24.0}}}}}}
    )
    assert check_span_quality(wide)[0] == OPEN

    weak = _context(
        {"corpus_a": {"by_split": {"test": {"spans": {"narrow": {"f1_iou_0_5": 0.1, "width_ratio": 1.0}}}}}}
    )
    assert check_span_quality(weak)[0] == OPEN

    good = _context(
        {"corpus_a": {"by_split": {"test": {"spans": {"narrow": {"f1_iou_0_5": 0.74, "width_ratio": 1.0}}}}}}
    )
    assert check_span_quality(good)[0] == CLOSED


def test_fact_coverage_rules_use_measured_recall() -> None:
    """Пропуск и частичное подтверждение оцениваются по полноте вердикта."""
    bad = _context(
        {
            "corpus_a": {
                "by_mode": {
                    "missing": {"verdict_recall": 0.0, "pairs": 10},
                    "partial": {"verdict_recall": 0.2667, "pairs": 15},
                }
            }
        }
    )
    assert check_missing(bad)[0] == OPEN
    assert check_partial(bad)[0] == OPEN

    good = _context(
        {
            "corpus_a": {
                "by_mode": {
                    "missing": {"verdict_recall": 0.9, "pairs": 10},
                    "partial": {"verdict_recall": 1.0, "pairs": 15},
                }
            }
        }
    )
    assert check_missing(good)[0] == CLOSED
    assert check_partial(good)[0] == CLOSED


def test_false_flag_rule_threshold_is_five_percent() -> None:
    """Ложные срабатывания на чистых парах: порог 5 %, 14 % — не закрыт."""
    bad = _context({"corpus_a": {"by_mode": {"faithful": {"verdict_fpr": 0.14, "pairs": 81}}}})
    assert check_false_flags(bad)[0] == OPEN

    good = _context({"corpus_a": {"by_mode": {"faithful": {"verdict_fpr": 0.0247, "pairs": 81}}}})
    assert check_false_flags(good)[0] == CLOSED


def test_corpus_a3_rule_checks_every_target_separately() -> None:
    """Корпус A3 закрывается только при выполнении всех целевых чисел сразу."""
    documents = {f"eo-{index}": {"facts_found": 12} for index in range(130)}

    enough = _context(
        {}, a3={"documents": documents, "by_level": {"федеральный": 70, "региональный": 60}}, a3_pairs=1300
    )
    status, detail = check_corpus_a3(enough)
    assert status == CLOSED
    assert "130" in detail

    few_regional = _context(
        {}, a3={"documents": documents, "by_level": {"федеральный": 120, "региональный": 10}}, a3_pairs=1300
    )
    assert check_corpus_a3(few_regional)[0] == OPEN

    few_pairs = _context(
        {}, a3={"documents": documents, "by_level": {"федеральный": 70, "региональный": 60}}, a3_pairs=100
    )
    assert check_corpus_a3(few_pairs)[0] == OPEN

    assert check_corpus_a3(_context({}))[0] == OPEN


def test_baseline_and_review_rules_do_not_close_on_absence() -> None:
    """Отсутствие данных не закрывает пункт: null — это «не закрыт» или «нет данных»."""
    assert (
        check_baseline(_context({"corpus_b": {"baseline_comparison": None, "baseline_note": "не извлечено"}}))[0]
        == OPEN
    )
    assert check_baseline(_context({"corpus_b": {"baseline_comparison": {"f1": 0.6}}}))[0] == CLOSED

    assert check_review_queue(_context({}))[0] == NO_DATA
    assert check_review_queue(_context({}, review={"total": 18, "checked": 4}))[0] == OPEN
    assert check_review_queue(_context({}, review={"total": 18, "checked": 18}))[0] == CLOSED


def test_registry_keys_are_unique_and_prioritised() -> None:
    """Реестр состоит из уникальных пунктов с приоритетами P0/P1/P2."""
    keys = [item.key for item in REGISTRY]
    assert len(keys) == len(set(keys))
    assert {item.priority for item in REGISTRY} <= {"P0", "P1", "P2"}
    assert all(item.command for item in REGISTRY)
    assert all(item.evidence for item in REGISTRY)


def test_render_marks_every_item_and_counts_match(tmp_path: Path) -> None:
    """В отчёте каждый пункт помечен, а сводка совпадает с числом пунктов."""
    context = _context({"external_tests": {"hf_runs": 0, "demo_runs": 4}})
    context["root"] = tmp_path
    rows = evaluate(context)
    text = render(rows, context)

    assert len(rows) == len(REGISTRY)
    for row in rows:
        assert row["key"] in text
        assert row["status"] in {CLOSED, OPEN, NO_DATA}
    assert f"| **Всего** | **{len(rows)}** |" in text
    assert "не правится руками" in text


def test_load_context_survives_missing_files(tmp_path: Path) -> None:
    """Отсутствие файлов отчётов не роняет сборку реестра."""
    context = load_context(tmp_path)
    assert context["metrics"] == {}
    assert context["a3"] == {}
    assert context["a3_pairs"] == 0
    rows = evaluate(context)
    assert all(row["status"] in {OPEN, NO_DATA} for row in rows)


def test_project_audit_is_reproducible() -> None:
    """Реестр проекта собирается на настоящих файлах и совпадает с JSON-версией."""
    from scripts.make_todo_audit import ROOT

    report = ROOT / "reports" / "TODO_AUDIT.json"
    if not report.is_file():
        return
    saved = json.loads(report.read_text(encoding="utf-8"))
    assert {item["key"] for item in saved["items"]} == {item.key for item in REGISTRY}
