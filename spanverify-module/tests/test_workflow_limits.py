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
# Тяжёлый прогон hf — единственное исключение: лимиты заданы явно (пилот 90,
# эксперименты 180, метрики 45), у каждого job'а есть timeout-minutes. Правило
# 40 минут остаётся для ci.yml и остальных workflow: без него job висел часами.
# Тяжёлые прогоны на реальном hf-раннере: предпосчёт признаков и обучение
# с отбором признаков занимают часы, потому что прямой проход по модели стоит
# секунды на пару. Исключение разрешено только этим файлам и только с
# обоснованием в шапке workflow — иначе «висящий» job снова съедает очередь.
HEAVY_JOB_MINUTES = {
    "hf-runs.yml": 180,
    "hf-a3-head.yml": 150,
    "hf-protocol.yml": 150,
}


def _load_strict(path: Path) -> object:
    """Загрузить YAML с запретом дублирующихся ключей.

    Обычный ``yaml.safe_load`` повторяющийся ключ молча «схлопывает» (последнее
    значение выигрывает) — так же молча GitHub отклоняет workflow, и job вообще
    не стартует. Прогон `38042202834` упал ровно на этом: у шага оказалось два
    ``uses``. Проверка делает класс ошибок локальным.
    """

    class Strict(yaml.SafeLoader):
        pass

    def no_duplicates(loader, node, deep=False):
        mapping: dict = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"дублирующийся ключ {key!r} (строка {key_node.start_mark.line + 1}) в {path.name}")
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    Strict.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, no_duplicates)
    return yaml.load(path.read_text(encoding="utf-8"), Loader=Strict)


def test_workflow_yaml_has_no_duplicate_keys() -> None:
    """Каждый workflow читается строго: два одинаковых ключа в映射 — это битый CI."""
    for path in _workflow_paths():
        _load_strict(path)  # падает с указанием файла и строки


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
    limit = HEAVY_JOB_MINUTES.get(path.name, MAX_JOB_MINUTES)
    over = {
        name: job["timeout-minutes"]
        for name, job in (data.get("jobs") or {}).items()
        if (job or {}).get("timeout-minutes", 0) > limit
    }
    assert not over, f"{path.name}: job'ы дольше {limit} минут: {over}"


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
