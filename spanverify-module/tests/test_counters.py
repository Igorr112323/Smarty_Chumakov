"""Тесты единого источника счётчиков (`scripts/counters.py`).

Смысл проверки не в том, что «763» — правильное число, а в том, что семь мест в документах не могут
разойтись между собой и с `reports/METRICS.json`: счётчик считается скриптом, документы правит
скрипт по явным шаблонам, а ручная (или глобальная) правка чисел — ровно тот режим, который уже
ломал отчёт. Поэтому проверяются три вещи: совпадение `COUNTERS.json` с `METRICS.json` и с pytest,
согласованность блока в отчёте и то, что подстановка по шаблону не трогает лишнего.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for _path in (str(ROOT), str(SCRIPTS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import counters  # noqa: E402

REPO_ROOT = ROOT.parent
COUNTERS = ROOT / "reports" / "COUNTERS.json"
METRICS = ROOT / "reports" / "METRICS.json"
REPORT = REPO_ROOT / "docs" / "ОТЧЁТ_по_договору.md"


def test_counters_artifact_exists_and_agrees_with_metrics_file() -> None:
    assert (
        COUNTERS.is_file()
    ), "нет `reports/COUNTERS.json`: выполните scripts/counters.py --json-out reports/COUNTERS.json"
    payload = json.loads(COUNTERS.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    assert payload["tests"]["collected"] == metrics["tests"]["collected"], (
        "счётчик в `METRICS.json` (scripts/collect_metrics.py) разошёлся с scripts/counters.py — "
        "нужен пересчёт, а не правка документа"
    )
    assert payload["tests"]["files"] >= 1 and payload["formatting"]["black_files"] >= 1
    assert payload["documents"]["checked"] >= 1
    for section in ("tests", "formatting", "documents"):
        assert payload[section]["command"], f"у счётчика {section} нет команды — число непроверяемо"


def test_docs_match_the_counter_block_without_manual_edits() -> None:
    """`counters.py --check` обязан проходить на реальном репозитории (тот же код, что в CI)."""
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "counters.py"), "--check"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-1200:]
    assert "счётчики согласованы" in result.stdout


def test_report_block_is_generated_not_handwritten() -> None:
    text = REPORT.read_text(encoding="utf-8")
    assert counters.MARK_START in text and counters.MARK_END in text, "в отчёте нет маркеров блока счётчиков"
    start = text.index(counters.MARK_START)
    end = text.index(counters.MARK_END, start)
    block = text[start:end]
    payload = json.loads(COUNTERS.read_text(encoding="utf-8"))
    assert str(payload["tests"]["collected"]) in block, "блок отчёта не совпадает с COUNTERS.json"
    assert "counters.py" in block


def test_substitution_touches_only_the_matched_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Подстановка точечная: соседняя строка с похожим числом не меняется."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "ОТЧЁТ_по_договору.md").write_text(
        "предыстория: было 734 теста\n"
        "| «752 собрано» | `python3 -m pytest --collect-only -q` (сумма по файлам) | "
        "сумма id по 63 файлам = 752; `tests.collected = 758` |\n"
        "`-m pytest -q` | 752 тестов, `FAILED` ровно один\n",
        encoding="utf-8",
    )
    module_docs = tmp_path / "spanverify-module"
    module_docs.mkdir()
    (module_docs / "README.md").write_text("├── tests/ 752 тестов (pytest)\n", encoding="utf-8")
    monkeypatch.setattr(counters, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        counters, "DOC_PATTERNS", [("docs/ОТЧЁТ_по_договору.md", r"\| «(\d+) собрано» \|", {"tests": "{tests}"})]
    )
    fake = {"tests": {"collected": 777, "files": 64}, "formatting": {"black_files": 161}, "documents": {"checked": 14}}
    touched = counters._apply(fake, write=True)
    assert touched == ["docs/ОТЧЁТ_по_договору.md:2"], touched
    text = (docs / "ОТЧЁТ_по_договору.md").read_text(encoding="utf-8")
    assert "было 734 теста" in text, "историческая строка задета быть не может"
    assert "| «777 собрано» |" in text
    assert "tests.collected = 758" in text, "вне шаблона числа менять нельзя"


def test_missing_black_does_not_break_the_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Без пакета `black` (job «Тесты») сверка счётчиков всё равно проходит и не врёт нулём."""
    # Патчим ПОСЛЕ reload: reload пересоздаёт модуль и его COMMANDS, — патч до reload терялся,
    # и тест «про отсутствие black» незаметно проверял присутствие.
    counters_module = importlib.reload(counters)
    monkeypatch.setitem(counters_module.COMMANDS, "black", [sys.executable, "-c", "raise SystemExit(3)"])
    payload = counters_module.collect()
    assert payload["formatting"]["available"] is False
    assert payload["formatting"]["black_files"] is None
    assert "недоступен" in "\n".join(payload and counters_module.block(payload))


def test_ambiguous_pattern_is_fatal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Если шаблон встречается дважды — скрипт обязан остановиться, а не выбрать первый."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "ОТЧЁТ_по_договору.md").write_text("| «1 собрано» |\n| «2 собрано» |\n", encoding="utf-8")
    monkeypatch.setattr(counters, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        counters, "DOC_PATTERNS", [("docs/ОТЧЁТ_по_договору.md", r"\| «(\d+) собрано» \|", {"tests": "{tests}"})]
    )
    fake = {"tests": {"collected": 777, "files": 1}, "formatting": {"black_files": 1}, "documents": {"checked": 1}}
    with pytest.raises(SystemExit):
        counters._apply(fake, write=True)
