"""Контрактные тесты договора: проверки, реестр чисел, документы и CI.

Здесь проверяется не поведение движка (это делает ``test_contract_surface.py``),
а сама система доказательств:

* ``scripts/contract_check.py`` покрывает все пункты ТЗ пп. 4.1 и 4.2, плана и
  отчётности, и пишет артефакт по каждому пункту;
* ``reports/NUMBERS.json`` — реестр чисел с происхождением (коммит, seed, режим,
  набор данных, источник);
* ``docs/СООТВЕТСТВИЕ_ДОГОВОРУ.md`` собирается из ``reports/CONTRACT.json``;
* документы договора (календарный план, структура отчёта, руководство) содержат
  обязательные разделы;
* CI содержит обязательную job ``contract``.

Тесты быстрые: тяжёлые прогоны (предельный текст 2 000 000 знаков, корпуса,
релиз) выполняет ``contract_check.py``, а не тестовый прогон.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]  # корень репозитория
MODULE = ROOT / "spanverify-module"
CONTRACT_JSON = ROOT / "reports" / "CONTRACT.json"
NUMBERS_JSON = ROOT / "reports" / "NUMBERS.json"
COMPLIANCE_DOC = ROOT / "docs" / "СООТВЕТСТВИЕ_ДОГОВОРУ.md"
PLAN_DOC = ROOT / "docs" / "ТЗ_и_календарный_план.md"
REPORT_DOC = ROOT / "docs" / "ОТЧЁТ_о_НИР_шаблон.md"
GUIDE_DOC = ROOT / "docs" / "РУКОВОДСТВО.md"
ACCEPTANCE_DOC = ROOT / "docs" / "ПРОГРАММА_И_МЕТОДИКА_ПРИЁМКИ.md"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
CONTRACT_SCRIPT = ROOT / "scripts" / "contract_check.py"
NUMBERS_SCRIPT = ROOT / "scripts" / "build_numbers.py"
CHECKSUM_SCRIPT = ROOT / "scripts" / "verify_checksum.py"

STATUSES = {"соответствует", "не соответствует", "не измерено"}


def _run(args: list[str], cwd: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=900,
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONPATH": str(MODULE)},
    )


def _load_harness() -> dict:
    """Прочитать реестр проверок из scripts/contract_check.py без его запуска."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib

    import contract_check  # noqa: PLC0415

    importlib.reload(contract_check)
    return {item.id: item for item in contract_check.CHECKS}


# ------------------------------------------------------------------ покрытие пунктов


def test_harness_covers_every_numbered_requirement() -> None:
    """У каждого пункта ТЗ пп. 4.1 и 4.2 есть проверка в контрактном прогоне."""
    checks = _load_harness()
    for index in range(1, 11):
        for prefix in (f"4.1.{index}-", f"4.2.{index}-"):
            assert any(item.startswith(prefix) for item in checks), f"нет проверки пункта {prefix[:-1]}"


def test_harness_covers_plan_reporting_and_rules() -> None:
    """План, отчётность, структура отчёта, реестр чисел и CI тоже проверяются."""
    checks = _load_harness()
    for required in (
        "plan.stages",
        "plan.reporting",
        "report.structure",
        "report.abstract",
        "report.originality",
        "numbers.registry",
        "numbers.check",
        "docs.compliance",
        "docs.acceptance",
        "ci.contract-job",
        "dod.tests",
    ):
        assert required in checks, f"нет проверки {required}"


def test_every_check_has_command_and_explanation() -> None:
    """У каждой проверки есть команда воспроизведения и описание способа."""
    for check in _load_harness().values():
        assert check.command.strip(), f"{check.id}: нет команды"
        assert check.how.strip(), f"{check.id}: не сказано, как проверяется"
        assert check.point.strip(), f"{check.id}: не указан пункт договора"
        assert len(check.requirement) > 15, f"{check.id}: требование описано слишком кратко"


# ------------------------------------------------------------------ артефакты прогона


@pytest.mark.skipif(not CONTRACT_JSON.is_file(), reason="контрактный прогон ещё не выполнялся")
def test_contract_json_is_complete_and_traceable() -> None:
    """CONTRACT.json: статус по каждому пункту, артефакт, коммит и дата у каждой записи."""
    payload = json.loads(CONTRACT_JSON.read_text(encoding="utf-8"))
    registry = _load_harness()
    assert payload["kind"] == "contract_check"
    assert payload["commit"], "в прогоне не зафиксирован коммит"
    assert payload["checks"], "в прогоне нет ни одной проверки"
    by_id = {item["id"]: item for item in payload["checks"]}
    assert set(by_id) == set(registry), "состав проверок в CONTRACT.json и в скрипте разошёлся"
    for item in payload["checks"]:
        assert item["status"] in STATUSES, f"{item['id']}: неизвестный статус {item['status']}"
        assert item["commit"] and item["date"], f"{item['id']}: нет коммита или даты"
        if item["status"] == "не измерено":
            assert item["note"].strip(), f"{item['id']}: «не измерено» без обоснования"
    # Сводка обязана сходиться с содержимым, иначе документ врёт.
    summary = payload["summary"]
    assert summary["соответствует"] == sum(1 for i in payload["checks"] if i["status"] == "соответствует")
    assert summary["не соответствует"] == sum(1 for i in payload["checks"] if i["status"] == "не соответствует")
    assert summary["не измерено"] == sum(1 for i in payload["checks"] if i["status"] == "не измерено")


@pytest.mark.skipif(not CONTRACT_JSON.is_file(), reason="контрактный прогон ещё не выполнялся")
def test_measured_checks_have_artifacts_with_provenance() -> None:
    """У измеренных проверок есть файл-артефакт с коммитом, seed и режимом."""
    payload = json.loads(CONTRACT_JSON.read_text(encoding="utf-8"))
    for item in payload["checks"]:
        if item["status"] == "не измерено":
            continue
        assert item["artifact"], f"{item['id']}: измерение без файла-артефакта"
        artifact = json.loads((ROOT / item["artifact"]).read_text(encoding="utf-8"))
        for key in ("commit", "seed", "mode", "dataset", "metrics"):
            assert key in artifact, f"{item['artifact']}: нет поля {key}"


def test_compliance_doc_is_generated_from_contract_json() -> None:
    """Документ соответствия собирается из CONTRACT.json и содержит таблицу статусов."""
    assert COMPLIANCE_DOC.is_file(), "нет docs/СООТВЕТСТВИЕ_ДОГОВОРУ.md"
    text = COMPLIANCE_DOC.read_text(encoding="utf-8")
    assert "reports/CONTRACT.json" in text
    assert "| Пункт |" in text, "в документе нет таблицы «пункт → факт → статус»"
    assert "scripts/contract_check.py" in text
    assert any(status in text for status in ("соответствует", "не соответствует", "не измерено"))


def test_agent_in_script_and_generated_doc_agree() -> None:
    """Скрипт умеет и запускать проверки, и генерировать документ — это один источник."""
    source = CONTRACT_SCRIPT.read_text(encoding="utf-8")
    assert "def render_compliance_doc(" in source
    assert "СООТВЕТСТВИЕ_ДОГОВОРУ.md" in source
    assert "--only" in source and "--fast" in source


# ------------------------------------------------------------------ реестр чисел


def test_numbers_registry_exists_with_provenance() -> None:
    """Реестр чисел: у измерения есть источник, у «не измерено» — причина."""
    assert NUMBERS_JSON.is_file(), "нет reports/NUMBERS.json"
    payload = json.loads(NUMBERS_JSON.read_text(encoding="utf-8"))
    assert payload["kind"] == "numbers_registry"
    assert payload["entries"], "реестр пуст"
    for entry in payload["entries"]:
        # "contract" — требование договора, "measured" — измерение прогона,
        # "source" — версия источника требований (SHA-256 и размер подписанного
        # договора): ни требование, ни измерение, но и то, и другое к нему
        # ссылается, поэтому реестр хранит его рядом.
        assert entry["kind"] in {"contract", "measured", "source"}, entry["id"]
        assert entry["unit"], f"{entry['id']}: не указана единица измерения"
        if entry["kind"] == "contract":
            assert entry["note"], f"{entry['id']}: требование без пояснения"
            continue
        if entry["kind"] == "source":
            # Факт об источнике требований (хеш и размер файла договора): набора
            # данных у него нет — он сам описывает, откуда взяты требования.
            assert entry["source"], f"{entry['id']}: нет файла-источника"
            assert entry["note"], f"{entry['id']}: источник без пояснения"
            continue
        assert entry["source"], f"{entry['id']}: нет файла-источника"
        assert entry["dataset"], f"{entry['id']}: нет метки набора данных"
        if entry["value"] is None:
            assert entry["note"], f"{entry['id']}: «не измерено» без причины"


def test_numbers_registry_matches_artifacts() -> None:
    """``build_numbers.py --check`` подтверждает, что реестр не устарел."""
    proc = _run([sys.executable, str(NUMBERS_SCRIPT), "--check"])
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_check_numbers_accepts_documents() -> None:
    """``check_numbers.py`` не находит чисел без источника в документах."""
    proc = _run(
        [
            sys.executable,
            str(MODULE / "scripts" / "check_numbers.py"),
            "--metrics",
            "reports/CURRENT_METRICS.json",
            "--numbers",
            str(NUMBERS_JSON),
        ],
        cwd=MODULE,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_check_numbers_rejects_invented_value(tmp_path: Path) -> None:
    """Подложенное число документ-проверка обязана поймать (проверка проверки)."""
    fake = tmp_path / "FAKE.md"
    fake.write_text("token F1 0.999 при FPR 0.001\n", encoding="utf-8")
    proc = _run(
        [
            sys.executable,
            str(MODULE / "scripts" / "check_numbers.py"),
            "--metrics",
            "reports/CURRENT_METRICS.json",
            "--numbers",
            str(NUMBERS_JSON),
            "--docs",
            str(fake),
        ],
        cwd=MODULE,
    )
    assert proc.returncode == 1, "выдуманное число принято как законное"
    assert "0.999" in proc.stdout


# ------------------------------------------------------------------ поставка и сумма


def test_checksum_script_detects_substitution() -> None:
    """Скрипт сверки суммы различает исправный файл и подменённый."""
    proc = _run([sys.executable, str(CHECKSUM_SCRIPT), "--self-test"])
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_zipapp_fits_contract_package_limit(tmp_path: Path) -> None:
    """ТЗ п. 4.2: переносимый пакет укладывается в 0,6 МБ при любой трактовке."""
    out = tmp_path / "release"
    proc = _run([sys.executable, "scripts/make_zipapp.py", "--out", str(out)], cwd=MODULE)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    target = out / "spanverify.pyz"
    assert target.is_file(), "пакет не собран"
    size = target.stat().st_size
    assert size <= 0.6 * 1_048_576, f"{size} байт > 0,6 МиБ"
    assert size <= 600_000, f"{size} байт > 0,6 МБ в десятичной трактовке"


# ------------------------------------------------------------------ документы


def test_calendar_plan_follows_contract() -> None:
    """Календарный план: два этапа по 6 месяцев, 200 000 и 300 000 руб., АС «Фонд-М»."""
    text = PLAN_DOC.read_text(encoding="utf-8")
    for marker in ("1-й этап", "2-й этап", "6 месяц", "200 000", "300 000", "АС «Фонд-М»"):
        assert marker in text, f"в календарном плане нет «{marker}»"


def test_reporting_list_is_complete() -> None:
    """Перечень отчётности обоих этапов и срок 20 дней указаны."""
    text = PLAN_DOC.read_text(encoding="utf-8")
    for marker in (
        "промежуточный отчёт",
        "заключительный отчёт",
        "финансов",
        "РИД",
        "предускорения",
        "дорожная карта",
        "20 календарных дней",
    ):
        assert marker.lower() in text.lower(), f"в перечне отчётности нет «{marker}»"


def test_report_template_follows_appendix_10() -> None:
    """Структура отчёта: титульный лист, реферат, содержание, термины, сокращения и т. д."""
    text = REPORT_DOC.read_text(encoding="utf-8")
    for marker in (
        "Титульный лист",
        "Реферат",
        "Содержание",
        "Термины",
        "Сокращения",
        "Введение",
        "Основная часть",
        "Заключение",
        "Источники",
        "Приложения",
    ):
        assert marker.lower() in text.lower(), f"в шаблоне отчёта нет раздела «{marker}»"


def test_report_abstract_has_keywords_and_originality() -> None:
    """Реферат: поля об объёме и 5–15 ключевых слов; оригинальность ≥ 80 %."""
    text = REPORT_DOC.read_text(encoding="utf-8")
    for marker in ("Ключевые слова", "Объём", "иллюстрац", "таблиц", "источник", "приложен", "80 %"):
        assert marker.lower() in text.lower(), f"в реферате нет «{marker}»"
    lines = text.splitlines()
    index = next(i for i, line in enumerate(lines) if "Ключевые слова" in line)
    keywords = [item.strip(" .;*") for item in lines[index + 1].split(",") if item.strip(" .;*")]
    assert 5 <= len(keywords) <= 15, f"ключевых слов {len(keywords)}, требуется 5–15"


def test_guide_has_install_update_and_remove() -> None:
    """Руководство описывает установку копированием, обновление заменой и удаление."""
    text = GUIDE_DOC.read_text(encoding="utf-8")
    for marker in ("Установка", "Обновление", "замен", "предыдущ", "Удалить", "localhost:8765"):
        assert marker.lower() in text.lower(), f"в руководстве нет «{marker}»"


def test_acceptance_program_uses_contract_harness() -> None:
    """Программа приёмки построена на тех же проверках, что и CI."""
    text = ACCEPTANCE_DOC.read_text(encoding="utf-8")
    for marker in ("contract_check.py", "CONTRACT.json", "check_numbers.py", "NUMBERS.json"):
        assert marker in text, f"в программе приёмки нет ссылки на {marker}"


# ------------------------------------------------------------------ CI


def test_ci_has_required_contract_job() -> None:
    """В CI есть обязательная job contract с прогоном проверок и сверкой чисел."""
    workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    jobs = workflow["jobs"]
    assert "contract" in jobs, "нет job contract"
    job = jobs["contract"]
    assert job.get("timeout-minutes"), "у job contract нет ограничения времени"
    steps = " \n".join(str(step.get("run", "")) for step in job["steps"])
    assert "scripts/contract_check.py" in steps, "job contract не запускает контрактные проверки"
    assert "scripts/check_numbers.py" in steps, "job contract не сверяет числа документов"
    assert "build_numbers.py" in steps, "job contract не собирает реестр чисел"


def test_ci_keeps_coverage_gate() -> None:
    """Порог покрытия 85 % остаётся обязательным в CI."""
    workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    text = json.dumps(workflow, ensure_ascii=False)
    assert "--cov-fail-under=85" in text
