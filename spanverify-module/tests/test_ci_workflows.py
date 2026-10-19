"""Проверки документа о заданиях CI (`docs/ЗАДАНИЯ_CI.md`, `scripts/collect_ci_workflows.py`).

Зачем это нужно. Заданий много, и по файлам не видно, какие обязаны быть зелёными.
Документ собирается скриптом из двух источников фактов, и проверки здесь
удерживают его от расхождений с ними:

1. каждое задание из `.github/workflows/*.yml` попадает в документ (иначе о
   задании молчат, и «все зелёные» будет неправдой);
2. у каждого задания в документе есть запуск и статус, а не пустая строка;
3. у задания с числами приёмки (замер объёмов режима hf, размеры поставки) в
   документе назван источник этих чисел — файл, откуда число попадает в реестр;
4. скрипт не выдумывает статусы: если прогонов нет, так и написано.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parent
SCRIPT = REPO_ROOT / "scripts" / "collect_ci_workflows.py"
DOC = REPO_ROOT / "docs" / "ЗАДАНИЯ_CI.md"
PAYLOAD = REPO_ROOT / "reports" / "CI_WORKFLOWS.json"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"


def _load_script():
    """Загрузить скрипт сбора по пути (в ``scripts/`` нет пакета)."""
    spec = importlib.util.spec_from_file_location("collect_ci_workflows", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("collect_ci_workflows", module)
    spec.loader.exec_module(module)
    return module


def test_document_lists_every_workflow() -> None:
    """В документе есть каждое задание из .github/workflows."""
    assert DOC.is_file(), "нет docs/ЗАДАНИЯ_CI.md (собрать: scripts/collect_ci_workflows.py)"
    payload = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    doc = DOC.read_text(encoding="utf-8")
    names = [item["name"] for item in payload["workflows"]]
    assert names, "в реестре заданий пусто"
    assert len(names) == len(list(WORKFLOWS.glob("*.yml"))), "не все файлы заданий попали в реестр"
    for name in names:
        assert f"`{name}`" in doc, f"{name}: задания нет в документе"


def test_document_has_trigger_and_status_for_each_job() -> None:
    """У каждого задания в документе назван запуск и последний прогон."""
    payload = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    doc_lines = DOC.read_text(encoding="utf-8").splitlines()
    for item in payload["workflows"]:
        row = next((line for line in doc_lines if line.startswith(f"| `{item['name']}`")), None)
        assert row, f"{item['name']}: строки нет в таблице"
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        assert len(cells) >= 5, f"{item['name']}: в строке {len(cells)} столбцов"
        assert cells[1], f"{item['name']}: не указано, как задание запускается"
        status = cells[2]
        assert status.startswith(
            ("успех", "прогонов нет", "идёт", "failure", "cancelled", "пропущен")
        ), f"{item['name']}: непонятный статус {status!r} — статус берётся из reports/CI_RUNS.json"


def test_acceptance_numbers_have_a_named_registry_file() -> None:
    """У заданий, чьи числа идут в приёмку, назван файл-источник числа."""
    payload = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    registry = payload["registry"]
    for name in ("hf-footprint", "build-and-release-exe", "ci"):
        assert name in registry, f"{name}: нет записи о том, куда попадают его числа"
    assert "4.2.5-hf-footprint.json" in registry["hf-footprint"]
    assert "release_sizes.json" in registry["build-and-release-exe"]


def test_script_declares_sources_and_does_not_invent_statuses() -> None:
    """Скрипт называет источники и умеет честно писать «прогонов нет»."""
    module = _load_script()
    assert module._status(None) == "прогонов нет"
    assert "успех" in module._status({"id": 1, "conclusion": "success"})
    assert "идёт" in module._status({"id": 2, "conclusion": None})
    payload = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    assert payload["source"] == ".github/workflows/*.yml + reports/CI_RUNS.json"
    assert payload["runs_considered"] >= 1, "не записано, сколько прогонов просмотрено"


def test_windows_rc_keeps_strict_size_and_frozen_packaging_contract() -> None:
    """P6: оба size limits обязательны; fit-зависимости не попадают в exe-среду."""
    import ast

    import yaml

    workflow = yaml.safe_load((WORKFLOWS / "build-and-release-exe.yml").read_text(encoding="utf-8"))
    jobs = workflow["jobs"]
    tests = "\n".join(str(s.get("run", "")) for s in jobs["test"]["steps"])
    build = "\n".join(str(s.get("run", "")) for s in jobs["windows-exe"]["steps"])
    assert "requirements-token-head-tests.txt" in tests
    assert "requirements-token-head-tests.txt" not in build
    assert 'if ((-not $sizes["exe_ok"]) -or (-not $sizes["pyz_ok"]))' in build
    assert "spanverify train" not in build and "git push" not in build
    assert "--prerelease" in build and "--latest=false" in build
    assert "spanverify-module/reports/experiments/** -text" in (REPO_ROOT / ".gitattributes").read_text()
    tree = ast.parse((MODULE_ROOT / "spanverify.spec").read_text(encoding="utf-8"))
    excluded = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "excludes" for t in n.targets)
    )
    assert {"numpy", "sklearn", "scipy", "torch", "transformers"} <= set(excluded)
