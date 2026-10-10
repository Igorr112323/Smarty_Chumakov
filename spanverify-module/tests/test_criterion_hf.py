"""Критерий ТЗ по режиму `hf`: PASS только если он действительно достигнут.

Ориентир (ТЗ, Приложение №3 к договору 0117812, п. 4.1): **F1 ≥ 0,60 при
FPR ≤ 0,40** на отложенной по документам части `test`, первичный уровень —
токены (определения заморожены в ``docs/METRIC_SPEC.md``).

Правила этого теста, и они сознательно жёсткие:

* ни ``skip``, ни ``xfail``: невыполненный критерий показывается как **NOT MET**
  с числами, а «измерение не выполнено» — как падение с командой прогона;
* статус пересчитывается из ``predictions_test.jsonl.gz``, а не берётся на веру
  из поля ``criterion``;
* проверяется и то, что прогон был честным по форме: 5 seed'ов, средний по всем,
  обучение без test, зафиксированный порог, железо и версии в манифесте.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "reports" / "hf_final" / "manifest.json"
PREDICTIONS = ROOT / "reports" / "hf_final" / "predictions_test.jsonl.gz"
GOLD = ROOT / "reports" / "hf_final" / "gold_test.jsonl.gz"
EXPECTED_SEEDS = [42, 43, 44, 45, 46]
COMMAND = (
    "python scripts/hf_protocol.py final --config config/hf_final_config.json "
    "--grid-cache reports/hf-grid --splits data/corpus_a3/splits --seeds 42,43,44,45,46 "
    "--include-val --out reports/hf_final"
)


def _harness():
    script = ROOT / "scripts" / "hf_protocol.py"
    spec = importlib.util.spec_from_file_location("hf_protocol_for_criterion", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _manifest() -> dict:
    if not (MANIFEST.is_file() and PREDICTIONS.is_file()):
        pytest.fail(
            "ИЗМЕРЕНИЕ НЕ ВЫПОЛНЕНО: нет reports/hf_final/manifest.json (или предсказаний). "
            f"Команда шага 3: {COMMAND}. Прогон — workflow «Протокол hf (критерий ТЗ)»: "
            "run_final=true при dispatch либо пуш с меткой [hf-final] (см. docs/EXPERIMENTS.md)."
        )
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_criterion_on_held_out_test() -> None:
    """F1 ≥ 0,60 при FPR ≤ 0,40 по токенам на test. Иначе — NOT MET с числами."""
    manifest = _manifest()
    harness = _harness()
    numbers = manifest.get("metrics_mean_std") or {}
    status = harness.criterion_status(numbers)
    if status["status"] != "PASS":
        pytest.fail(
            "NOT MET: "
            f"F1(токены, test) = {status['token_f1']} при требовании ≥ 0,60; "
            f"FPR = {status['token_fpr']} при требовании ≤ 0,40. "
            f"Записи экспериментов и причины — docs/EXPERIMENTS.md (шаги 1–3)."
        )
    assert status["f1_gap"] <= 0.0 and status["fpr_gap"] <= 0.0
    recorded = str((manifest.get("criterion") or {}).get("status") or "")
    assert recorded == "PASS", f"манифест заявляет «{recorded}», пересчёт даёт PASS — числа разошлись"


def test_criterion_status_is_recomputed_from_predictions() -> None:
    """Числа, на которых стоит статус, восстанавливаются из предсказаний."""
    manifest = _manifest()
    harness = _harness()
    problems, numbers = harness.compare_manifest(MANIFEST, PREDICTIONS, GOLD)
    assert not problems, "манифест не сходится с предсказаниями:\n  " + "\n  ".join(problems)
    assert str(numbers["criterion"]) == str((manifest.get("criterion") or {}).get("status"))


def test_run_shape_is_the_frozen_one() -> None:
    """Форма прогона: 5 seed'ов, mean ± std по всем, обучение без test, порог из train."""
    manifest = _manifest()
    assert list(manifest.get("seeds") or []) == EXPECTED_SEEDS, manifest.get("seeds")
    per_seed = manifest.get("per_seed") or []
    assert len(per_seed) == len(EXPECTED_SEEDS), "число записей per_seed не равно числу seed'ов"
    assert str(manifest.get("fit_on")) in {"train", "train+val"}, "обучение зашло в test"
    assert str(manifest.get("protocol")) == "hf-criterion-v1"
    assert manifest.get("threshold_source"), "не указано, откуда взялся порог"
    assert manifest.get("metric_spec_sha256"), "нет хеша замороженной спецификации метрик"
    aggregate = manifest.get("metrics_mean_std") or {}
    assert "f1_per_seed" in aggregate and len(aggregate["f1_per_seed"]) == len(EXPECTED_SEEDS)
    # Лучший seed не выбирается: среднее обязано лежать между минимумом и максимумом.
    values = [float(item["tokens"]["f1"]) for item in per_seed]
    assert min(values) - 1e-9 <= float(aggregate["f1"]) <= max(values) + 1e-9


def test_provenance_is_recorded() -> None:
    """Манифест обязан содержать все входные данные прогона: модель, железо, время, хеши."""
    manifest = _manifest()
    for key in (
        "code_commit",
        "config_sha256",
        "splits_sha256",
        "hardware",
        "duration_s",
        "started_at",
        "predictions",
        "prediction_rows",
        "deviations",
    ):
        assert key in manifest, f"в манифесте нет поля {key}"
    splits = manifest.get("splits_sha256") or {}
    assert len(splits.get("test") or "") == 64, "sha256 test должен быть полным"
    hardware = manifest.get("hardware") or {}
    for key in ("cpu_model", "ram_total_gb", "torch_version", "transformers_version", "python"):
        assert hardware.get(key), f"в манифесте нет данных железа: {key}"
    assert float(manifest.get("duration_s") or 0.0) > 0.0, "время прогона не записано"
    assert int(manifest.get("prediction_rows") or 0) > 0


def test_test_split_is_not_used_for_selection() -> None:
    """В отборе конфигурации test не участвовал: sweep говорит только про val.

    Проверка по артефакту шага 2: если в ``sweep.json`` появятся численные поля,
    помеченные как test, или отбор пойдёт не по val — это подгонка по тесту, и
    результат по договору недействителен.
    """
    sweep = ROOT / "reports" / "hf_protocol" / "sweep.json"
    if not MANIFEST.is_file():
        # Шага 3 ещё не было. Проверять нечего, но и молчать нельзя: фиксируем,
        # что несуществующих чисел в документах нет — стоит отметка «не измерено».
        for path in (ROOT.parent / "README.md", ROOT / "docs" / "EXPERIMENTS.md"):
            assert "не измерено" in path.read_text(
                encoding="utf-8"
            ), f"{path.name}: измерения нет, а отметки «не измерено» тоже нет — откуда числа?"
        return
    assert sweep.is_file(), "манифест финала есть, а артефакта шага 2 (sweep.json) нет: откуда взялась конфигурация?"
    text = sweep.read_text(encoding="utf-8")
    payload = json.loads(text)
    assert payload.get("run_count", 0) <= 10, "нарушен лимит десяти экспериментов"
    best = payload.get("best")
    journal = (ROOT / "docs" / "EXPERIMENTS.md").read_text(encoding="utf-8")
    for item in payload.get("experiments", []):
        name = str(item.get("name") or "")
        errors = {
            corpus: numbers.get("error")
            for corpus, numbers in (item.get("per_corpus") or {}).items()
            if numbers.get("error")
        }
        for corpus in item.get("per_corpus") or {}:
            assert "test" not in str(corpus).lower(), f"отбор по test: {corpus}"
        if name == best:
            # Ошибки в ВЫБРАННОЙ конфигурации не прощаются: по ней считается финал.
            assert not errors, f"финальная конфигурация {name} получена с ошибками: {errors}"
        elif errors:
            # Упавший и отклонённый эксперимент обязан остаться в журнале — иначе
            # неудобный прогон исчезает из отчётности вместе с причиной.
            assert name and name in journal, f"{name}: ошибка {errors} и нет записи в docs/EXPERIMENTS.md"
            continue
        assert (item.get("selection") or {}).get("token_f1_val") is not None, "отбор не по val"
    assert "val" in str(payload.get("rule", "")).lower(), "в правиле отбора нет val"
