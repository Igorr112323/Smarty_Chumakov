"""Правила оформления workflow: таймауты и кавычки в `name:`.

Оба правила выучены на собственных ошибках, и обе ошибки стоили прогона:

* job без ``timeout-minutes`` висел до общего лимита в 6 часов (диагноз
  «почему CI висит», причина 1: job загрузки корпуса не успевал закончиться);
* ``name:`` с двоеточием без кавычек ломает YAML целиком — прогон завершался
  ``failure`` без единого job'а (прогон 37465518131), что выглядит как загадка,
  а не как опечатка.

Тест дешёвый (читает несколько файлов) и стоит в общем прогоне, поэтому
нарушение будет поймано до пуша, а не после.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
MAX_JOB_MINUTES = 40


def _workflow_paths() -> list[Path]:
    return sorted(WORKFLOWS_DIR.glob("*.yml"))


def test_there_are_workflows_to_check() -> None:
    """Страховка от того, что проверка молча пройдёт на пустом каталоге."""
    assert _workflow_paths(), f"не найдено workflow в {WORKFLOWS_DIR}"


@pytest.mark.parametrize("path", _workflow_paths(), ids=lambda p: p.name)
def test_workflow_is_valid_yaml(path: Path) -> None:
    """Файл разбирается: опечатка в YAML хоронит весь прогон без диагностики."""
    try:
        yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - сообщение важнее типа
        pytest.fail(f"{path.name}: не разбирается как YAML — {type(exc).__name__}: {exc}")


@pytest.mark.parametrize("path", _workflow_paths(), ids=lambda p: p.name)
def test_every_job_has_timeout(path: Path) -> None:
    """Без timeout-minutes job висит до общего лимита и занимает слот раннера."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    jobs = data.get("jobs") or {}
    assert jobs, f"{path.name}: нет ни одного job'а"
    missing = sorted(name for name, job in jobs.items() if not (job or {}).get("timeout-minutes"))
    assert not missing, f"{path.name}: job'ы без timeout-minutes: {missing}"


@pytest.mark.parametrize("path", _workflow_paths(), ids=lambda p: p.name)
def test_no_job_is_longer_than_the_limit(path: Path) -> None:
    """Лимит 40 минут: «висение» должно заканчиваться за минуты, а не за часы."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    over = {
        name: job["timeout-minutes"]
        for name, job in (data.get("jobs") or {}).items()
        if (job or {}).get("timeout-minutes", 0) > MAX_JOB_MINUTES
    }
    assert not over, f"{path.name}: job'ы дольше {MAX_JOB_MINUTES} минут: {over}"


@pytest.mark.parametrize("path", _workflow_paths(), ids=lambda p: p.name)
def test_step_timeouts_do_not_exceed_job_timeout(path: Path) -> None:
    """Шаг с лимитом больше лимита job'а — тот же «висящий» job, но незаметный."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    problems: list[str] = []
    for name, job in (data.get("jobs") or {}).items():
        job_limit = (job or {}).get("timeout-minutes")
        if not job_limit:
            continue
        for step in job.get("steps") or []:
            step_limit = (step or {}).get("timeout-minutes")
            if step_limit and step_limit > job_limit:
                problems.append(f"{name}: шаг {step_limit} мин > job {job_limit} мин")
    assert not problems, f"{path.name}: {problems}"


@pytest.mark.parametrize("path", _workflow_paths(), ids=lambda p: p.name)
def test_names_with_colon_are_quoted(path: Path) -> None:
    """`- name: Текст: с двоеточием` ломает YAML — такие имена надо кавычить."""
    offenders: list[str] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = re.match(r"^(\s*)- name: (.+?)\s*$", line)
        if not match:
            continue
        value = match.group(2)
        if value.startswith(('"', "'")):
            continue
        if ":" in value and not value.startswith("${{"):
            offenders.append(f"строка {number}: {value}")
    assert not offenders, f"{path.name}: name с двоеточием без кавычек — {offenders}"


def test_ci_has_concurrency() -> None:
    """Без concurrency каждый пуш плодит полный прогон, и они толпятся."""
    data = yaml.safe_load((WORKFLOWS_DIR / "ci.yml").read_text(encoding="utf-8"))
    assert data.get("concurrency"), "в ci.yml нет блока concurrency"
