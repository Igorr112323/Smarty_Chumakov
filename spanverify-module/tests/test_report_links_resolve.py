"""Ссылки договорного отчёта обязаны разрешаться в репозитории.

Почему это тест, а не заметка: в отчётах уже встречались ссылки на артефакты, которых в
дереве нет (несуществующая ветка данных, «неработающая цепочка»). Такой дефект не ловится
ни линтером, ни тестами кода — только явной сверкой путей, которые обещаны читателю.

Правила, закреплённые тестом:

* каждый оборотный путь вида ```каталог/…/файл` `` (первый сегмент — каталог репозитория
  или один из каталогов модуля) обязан существовать — как файл или каталог;
* путь с ``*`` трактуется как glob и требует хотя бы одного совпадения;
* путь, которого сознательно ещё нет (артефакт будущего прогона), разрешено оставлять
  только в строке с оправданием — «создаётся прогоном», «не проверено», «файла пока нет»,
  «не выполнено», «ожидает», «в CI»; иначе тест падает.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "spanverify-module"
REPORT = ROOT / "docs" / "ОТЧЁТ_по_договору.md"

# Первый сегмент, по которому токен считается путём репозитория.
MODULE_DIRS = {"data", "reports", "tests", "scripts", "config", "spanverify", "docs"}
JUSTIFICATIONS = (
    "создаётся прогоном",
    "не проверено",
    "файла пока нет",
    "не выполнено",
    "не выполнена",
    "ожидает",
    "в CI",
    "не измерено",
)
PATH_SHAPE = re.compile(r"^[A-Za-z0-9_\-./\u0400-\u04FF*]+$")


def _repo_dirs() -> set[str]:
    return {path.name for path in ROOT.iterdir()} | MODULE_DIRS


def _candidates(text: str) -> list[tuple[str, int, str]]:
    """(путь, номер строки, контекст) — путь и следующая строка вместе.

    Markdown в отчёте переносится вручную, и оправдание («создаётся прогоном») может стоять
    на строке после пути.
    """
    known = _repo_dirs()
    lines = text.splitlines()
    found: list[tuple[str, int, str]] = []
    in_fence = False
    for number, line in enumerate(lines, start=1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for token in re.findall(r"`([^`]+)`", line):
            token = token.strip().strip(",.;:")
            if not token or "://" in token or "::" in token:
                continue
            if "/" not in token or not PATH_SHAPE.fullmatch(token):
                continue
            if token.split("/")[0] not in known:
                continue
            context = " ".join([line] + ([lines[number]] if number < len(lines) else []))
            found.append((token, number, context))
    return found


def _resolves(path: str) -> bool:
    if "*" in path:
        return any(list(base.glob(path)) for base in (ROOT, MODULE))
    return any((base / path).exists() for base in (ROOT, MODULE))


def test_report_exists() -> None:
    assert REPORT.is_file(), "нет договорного отчёта"


def test_every_cited_path_resolves_or_is_justified() -> None:
    text = REPORT.read_text(encoding="utf-8")
    broken: list[str] = []
    checked = 0
    for path, number, line in _candidates(text):
        checked += 1
        if _resolves(path):
            continue
        lowered = line.lower()
        if any(marker.lower() in lowered for marker in JUSTIFICATIONS):
            continue
        broken.append(f"строка {number}: путь «{path}» отсутствует и не обоснован")
    assert checked >= 25, f"слишком мало разобранных ссылок ({checked}): сломался разбор"
    assert not broken, "отчёт ссылается на несуществующие пути: " + "; ".join(broken)


def test_promised_future_artifacts_are_still_missing() -> None:
    """Пометка «создаётся прогоном» обязана соответствовать дереву: файла под `reports/` нет.

    Обратная сторона правила: пометка не должна пережить появление артефакта, иначе
    читателю показывают «артефакта ещё нет» там, где он уже лежит в репозитории.
    """
    text = REPORT.read_text(encoding="utf-8")
    for path, number, context in _candidates(text):
        if "создаётся прогоном" not in context.lower():
            continue
        if not (path.startswith("reports/") or "/reports/" in path):
            continue
        assert not _resolves(path), (
            f"строка {number}: путь «{path}» уже есть в репозитории, а отчёт называет его"
            " артефактом будущего прогона"
        )


def test_statuses_use_allowed_vocabulary() -> None:
    """Колонка статуса в таблице работ — только из закрытого словаря.

    «Почти сделано» и «в целом готово» — ровно то, из-за чего отчёты приходилось
    переписывать: статус либо «сделано», либо «частично» (и незакрытая часть названа), либо
    «не выполняется» / «ожидает» с блокером, либо «не проверено».
    """
    text = REPORT.read_text(encoding="utf-8")
    table = text.split("## 1. Инвентаризация работ", 1)[1].split("## 2.", 1)[0]
    allowed = ("сделано", "частично", "не выполняется", "ожидает", "не проверено")
    rows = [line for line in table.splitlines() if line.startswith("| ") and line.count("|") >= 6]
    works = [line for line in rows if re.match(r"\|\s*\d+\s*\|", line)]
    assert len(works) == 10, f"в таблице работ {len(works)} строк, ожидалось 10"
    for line in works:
        status = [cell.strip() for cell in line.strip().strip("|").split("|")][1].lower()
        assert any(word in status for word in allowed), f"статус вне словаря: «{status}»"
        assert "почти" not in status, f"размытая формулировка статуса: «{status}»"
        assert "готово" not in status, f"размытая формулировка статуса: «{status}»"
