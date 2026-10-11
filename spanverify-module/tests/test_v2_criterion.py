"""Гейт критерия цикла v2: он обязан краснеть по-настоящему, а не «вежливо».

Правила, из которых вырос этот файл:
* невыполненный критерий показывается падением с текстом NOT MET, а не `skip`
  или `xfail` (ослаблять гейты запретил заказчик);
* ратчет включается с шага 0: пока `reports/v2/freeze.json` нет, цикл v2 ещё не
  начинал измерять — отдельный тест проверяет, что чисел v2 в репозитории нет нигде;
* если шаг 0 сделан, а измерения на тесте нет — тест падает и не даёт «забыть»
  про цикл на середине (в том числе после срабатывания стоп-правила).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "v2"
F1_MIN = 0.70
FPR_MAX = 0.40


def _read(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _comma(value: float) -> str:
    return f"{value:.6f}".replace(".", ",")


def test_v2_measurement_is_not_cherry_picked_before_step_zero() -> None:
    """До шага 0 в репозитории не может быть чисел цикла v2 — проверяем это прямо."""
    if (REPORTS / "freeze.json").is_file():
        return  # цикл начался: дальше говорит test_v2_criterion_on_held_out_test
    for name in ("manifest.json", "sweep.json", "val_curve.json", "seedcheck.json", "predictions_test.jsonl.gz"):
        assert not (REPORTS / name).is_file(), f"артефакт {name} появился до заморозки шага 0"
    doc = (ROOT / "docs" / "EXPERIMENTS_v2.md").read_text(encoding="utf-8")
    assert "не измерено" in doc, "в журнале v2 до артефактов обязаны быть слова «не измерено»"


def test_v2_criterion_on_held_out_test() -> None:
    freeze = _read(REPORTS / "freeze.json")
    if freeze is None:
        # Цикл ещё не начинал измерять: пока корпуса и шага 0 нет, требовать числа
        # бессмысленно — за тем, чтобы «результат» не появился раньше данных, следит
        # test_v2_measurement_is_not_cherry_picked_before_step_zero. С момента шага 0
        # этот тест краснеет и не гаснет, пока цикл не доведён до результата.
        return
    sweep = _read(REPORTS / "sweep.json")
    if sweep is None:
        pytest.fail("шаг 0 сделан, но шага 2 нет (reports/v2/sweep.json): отбор конфигурации не выполнен")
    if (sweep.get("stop_rule") or {}).get("triggered"):
        best = max(
            (float((item.get("e1_val_chosen") or {}).get("f1") or 0.0) for item in sweep.get("results") or []),
            default=0.0,
        )
        pytest.fail(
            f"ИЗМЕРЕНИЕ НА ТЕСТЕ НЕ ВЫПОЛНЯЛОСЬ: стоп-правило шага 2, F1_val = {_comma(best)} "
            f"при требовании ≥ {_comma(F1_MIN)}. Тестовая выборка v2 не открывалась "
            "(решение заказчика от 2026-10-11: «не достигнуто на val» — цикл закрывается)."
        )
    manifest = _read(REPORTS / "manifest.json")
    if manifest is None:
        pytest.fail(
            "стоп-правило шага 2 не сработало, но шаг 3 не выполнен: нет reports/v2/manifest.json. "
            "После отбора на val прогон на тесте обязателен — ровно один."
        )
    aggregate = manifest.get("metrics_mean_std") or {}
    f1 = float(aggregate.get("f1") or 0.0)
    fpr = float(aggregate.get("fpr") if aggregate.get("fpr") is not None else 1.0)
    goal = manifest.get("goal_internal") or {}
    if abs(float(goal.get("token_f1") or 0.0) - f1) > 1e-6 or abs(float(goal.get("token_fpr") or 0.0) - fpr) > 1e-6:
        pytest.fail(
            f"goal_internal разошёлся с метриками: goal {goal.get('token_f1')}/{goal.get('token_fpr')}, "
            f"metrics_mean_std {f1}/{fpr}"
        )
    assert f1 >= F1_MIN and fpr <= FPR_MAX, (
        f"NOT MET: F1(токены, test v2) = {_comma(f1)} при требовании ≥ {_comma(F1_MIN)}; "
        f"FPR = {_comma(fpr)} при требовании ≤ {_comma(FPR_MAX)}. "
        "Записи экспериментов и причины — docs/EXPERIMENTS_v2.md (шаги 0–3)."
    )


def test_v2_contract_minimum_is_reported_as_a_separate_line() -> None:
    """Контрактный минимум 0,60 — отдельная строка отчёта, а не то же самое, что цель 0,70."""
    manifest = _read(REPORTS / "manifest.json")
    readme = (ROOT.parent / "README.md").read_text(encoding="utf-8")
    if manifest is None:
        assert "<!-- HF-CRITERION-V2:BEGIN -->" not in readme, "блока v2 в README быть не должно до измерения"
        return
    contract = manifest.get("goal_contract_min") or {}
    aggregate = manifest.get("metrics_mean_std") or {}
    assert contract, "в манифесте нет отдельной строки контрактного минимума"
    expected = (
        "PASS"
        if float(aggregate.get("f1") or 0.0) >= 0.60 and float(aggregate.get("fpr") or 1.0) <= FPR_MAX
        else "NOT MET"
    )
    assert contract.get("status") == expected
    assert float(contract.get("token_f1") or 0.0) == pytest.approx(float(aggregate.get("f1") or 0.0), abs=1e-6)
