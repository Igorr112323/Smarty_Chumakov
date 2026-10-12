#!/usr/bin/env python3
"""Единый источник счётчиков: число тестов, файлов тестов, файлов `black`, документов `check_numbers`.

Проблема, из-за которой скрипт появился: счётчикиliving в семи документах и раньше обновлялись
вручную (или глобальной заменой — был случай, когда замена `734 → 752` задела метрику
`token F1 0,734`). Скрипт делает единственное, что должно делаться автоматически: считает три
числа одной командой, пишет `reports/COUNTERS.json`, генерирует таблицу в отчёте между маркерами
``<!-- counters:start -->`` / ``<!-- counters:end -->`` и — в режиме ``--fix-docs`` — подставляет
число тестов в те строки документов, где оно уже стоит, по явному перечню шаблонов.

Шаблоны (а не «заменить везде число»): каждый обязан совпасть ровно один раз, иначе скрипт падает.
`PROGRESS.md` намеренно не правится: это журнал, исторические строки в нём — часть истории.

Команды:

    python3 scripts/counters.py                 # напечатать
    python3 scripts/counters.py --json-out reports/COUNTERS.json
    python3 scripts/counters.py --write-report --fix-docs
    python3 scripts/counters.py --check         # для CI и тестов
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parent

COMMANDS = {
    "tests": [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
    "black": [sys.executable, "-m", "black", "--check", "."],
    "check_numbers": [sys.executable, "scripts/check_numbers.py"],
    "metrics": [
        sys.executable,
        "-c",
        "import json,sys; print(json.load(open('reports/METRICS.json'))['tests']['collected'])",
    ],
}

# (файл, шаблон с группами-числами, шаблон замены). Группы заполняются по порядку из values.
DOC_PATTERNS: list[tuple[str, str, dict[str, str]]] = [
    ("README.md", r"# (\d+) тестов", {"tests": "{tests}"}),
    ("ИТОГ.md", r"\*\*Тесты и покрытие\*\*: (\d+) тестов проходят", {"tests": "{tests}"}),
    ("ИТОГ.md", r"\| (\d+) тестов, покрытие (\d+) % \|", {"tests": "{tests}"}),
    (
        "docs/ОТЧЁТ_о_НИР_шаблон.md",
        r"\| Автоматические тесты \| (\d+) тестов",
        {"tests": "{tests}"},
    ),
    ("spanverify-module/README.md", r"реализован, (\d+) тестов", {"tests": "{tests}"}),
    ("spanverify-module/README.md", r"├── tests/ (\d+) тестов \(pytest\)", {"tests": "{tests}"}),
    ("docs/ОТЧЁТ_по_договору.md", r"\| «(\d+) собрано» \|", {"tests": "{tests}"}),
    (
        "docs/ОТЧЁТ_по_договору.md",
        r"сумма id по (\d+) файлам = (\d+)",
        {"files": "{files}", "tests": "{tests}"},
    ),
    ("docs/ОТЧЁТ_по_договору.md", r"`tests\.collected = (\d+)`", {"tests": "{tests}"}),
    (
        "docs/ОТЧЁТ_по_договору.md",
        r"-m pytest -q` \| (\d+) тестов, `FAILED` ровно один",
        {"tests": "{tests}"},
    ),
]

MARK_START = "<!-- counters:start -->"
MARK_END = "<!-- counters:end -->"
REPORT = "docs/ОТЧЁТ_по_договору.md"


def _run(argv: list[str]) -> str | None:
    """Вывод команды целиком; `None` — если исполняемого файла нет в этой среде.

    Второй случай не ошибка: job «Тесты» ставит только `pytest` и `pyyaml` (см. `ci.yml`), пакета
    `black` в нём нет, а счётчик форматирования должен работать там, где black есть (job «Линтер»).
    Падать из-за этого в CI — значит сделать сверку документов зависимой от окружения.
    """
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, cwd=MODULE_ROOT, check=False)
    except (FileNotFoundError, OSError):
        return None
    return (completed.stdout or "") + (completed.stderr or "")


def collect() -> dict[str, Any]:
    """Три счётчика одной командой; каждый — с текстом команды, которой получен."""
    out = _run(COMMANDS["tests"])
    tests = 0
    files = 0
    for line in out.splitlines():
        match = re.fullmatch(r"(?:\.\./)?[\w/\-\.]+\.py: (\d+)", line.strip())
        if match:
            tests += int(match.group(1))
            files += 1
    black_out = _run(COMMANDS["black"])
    black_match = re.search(r"(\d+) files? would be left unchanged", black_out or "")
    # «Сколько документов сверяется» — свойство самого чекера (список DEFAULT_DOCS), а не результат
    # прогона: иначе правка документов не могла бы начаться, пока check_numbers красный, —
    # и получался бы замкнутый круг «сначала почини числа, чтобы узнать, сколько чисел чинить».
    import check_numbers

    docs_out = _run(COMMANDS["check_numbers"]) or ""
    documents_checked = sum(1 for name in check_numbers.DEFAULT_DOCS if (REPO_ROOT / name).is_file())
    numbers_consistent = "проверено документов" in docs_out
    metrics_out = _run(COMMANDS["metrics"]).strip()
    return {
        "tests": {
            "collected": tests,
            "files": files,
            "command": " ".join(COMMANDS["tests"][:-2]) + " (сумма по строкам «файл: число»)",
            "metrics_json": int(metrics_out) if metrics_out.isdigit() else None,
        },
        "formatting": {
            "black_files": int(black_match.group(1)) if black_match else None,
            "available": black_match is not None,
            "command": " ".join(COMMANDS["black"]),
        },
        "documents": {
            "checked": documents_checked,
            "consistent": numbers_consistent,
            "command": " ".join(COMMANDS["check_numbers"]),
        },
        "note": (
            "число тестов в `tests.collected` того же файла `reports/METRICS.json` обязано совпасть "
            "с `collected` здесь: METRICS.json обновляется `scripts/collect_metrics.py`"
        ),
    }


def block(counters: dict[str, Any]) -> list[str]:
    """Таблица для отчёта: строки генерируются, а не переписываются от руки."""
    return [
        "| Счётчик | Значение | Команда |",
        "|---|---|---|",
        f"| Тестов собрано | {counters['tests']['collected']} | `{counters['tests']['command']}` |",
        f"| Файлов тестов | {counters['tests']['files']} | там же (число строк «файл: N») |",
        f"| `tests.collected` в `reports/METRICS.json` | {counters['tests']['metrics_json']} | `python3 scripts/collect_metrics.py` |",
        "| Файлов по `black --check .` | {} | `{}` |".format(
            (
                counters["formatting"]["black_files"]
                if counters["formatting"]["available"]
                else "пакета `black` в этой среде нет (job «Тесты») — счётчик недоступен, снимается в job «Линтер»"
            ),
            counters["formatting"]["command"],
        ),
        f"| Документов сверил `check_numbers` | {counters['documents']['checked']} | `{counters['documents']['command']}` |",
    ]


def render(counters: dict[str, Any]) -> str:
    return "\n".join(
        [
            MARK_START,
            "",
            "Счётчики берутся из `reports/COUNTERS.json`, который пишет",
            "`python3 scripts/counters.py --json-out reports/COUNTERS.json`; правка чисел в",
            "документах вручную не выполняется (и глобальная замена запрещена: одна такая",
            "задела когда-то метрику `token F1 0,734`).",
            "",
            *block(counters),
            MARK_END,
        ]
    )


def _apply(counters: dict[str, Any], *, write: bool) -> list[str]:
    """Подстановка чисел по шаблонам; возврат списка затронутых строк «файл:номер».

    Каждый шаблон обязан встретиться ровно один раз — это и есть защита от «глобальной замены»:
    если строка уехала или её стало две, скрипт падает, а не правит наугад.
    """
    values = {
        "tests": str(counters["tests"]["collected"]),
        "files": str(counters["tests"]["files"]),
        "black": str(counters["formatting"]["black_files"]),
        "docs": str(counters["documents"]["checked"]),
    }
    touched: list[str] = []
    for rel, pattern, mapping in DOC_PATTERNS:
        path = REPO_ROOT / rel
        text = path.read_text(encoding="utf-8")
        matches = list(re.finditer(pattern, text))
        if len(matches) != 1:
            raise SystemExit(f"{rel}: шаблон `{pattern}` дал {len(matches)} совпадений, ожидалось ровно 1")
        found = matches[0]
        groups = [str(group) for group in found.groups()]
        rebuilt = found.group(0)
        for index, key in enumerate(mapping):
            if index >= len(groups):
                break
            current, wanted = groups[index], values[key]
            if current == wanted:
                continue
            rebuilt = rebuilt.replace(current, wanted, 1)
            groups[index] = wanted
        if rebuilt != found.group(0):
            line = text[: found.start()].count("\n") + 1
            touched.append(f"{rel}:{line}")
            if write:
                path.write_text(text.replace(found.group(0), rebuilt, 1), encoding="utf-8")
    return touched


def _differs(current: str, wanted: str, *, ignore_black: bool) -> bool:
    """Нужно ли обновлять блок.

    Строка `black` — единственная, которая зависит от окружения прогона: в job «Тесты» пакета
    `black` нет, и там счётчик физически нечем снять. Сравнивать блок целиком в такой среде —
    значит получить «блок устарел» там, где устареть нечему: значения тестов и документов совпадают,
    а отличается лишь строка, помеченная как недоступная. Поэтому при отсутствии black эта строка
    из сравнения исключается — но только она.
    """
    if current == wanted:
        return False
    if not ignore_black:
        return True

    def strip(text: str) -> str:
        return "\n".join(line for line in text.splitlines() if "Файлов по" not in line)

    return strip(current) != strip(wanted)


def _replace_report_block(counters: dict[str, Any], *, write: bool) -> bool:
    path = REPO_ROOT / REPORT
    text = path.read_text(encoding="utf-8")
    if MARK_START not in text or MARK_END not in text:
        raise SystemExit(f"в {REPORT} нет маркеров {MARK_START} / {MARK_END} — блок счётчиков не вставлен")
    start = text.index(MARK_START)
    end = text.index(MARK_END, start) + len(MARK_END)
    current = text[start:end]
    wanted = render(counters)
    if not _differs(current, wanted, ignore_black=not counters["formatting"]["available"]):
        return False
    if write:
        path.write_text(text[:start] + wanted + text[end:], encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="единый источник счётчиков документов")
    parser.add_argument("--json-out", default=None, help="куда положить reports/COUNTERS.json")
    parser.add_argument("--write-report", action="store_true", help="заменить блок счётчиков в отчёте")
    parser.add_argument("--fix-docs", action="store_true", help="подставить числа в документы по шаблонам")
    parser.add_argument("--check", action="store_true", help="проверить согласованность (для CI), без записи")
    args = parser.parse_args(argv)

    counters = collect()
    if args.json_out:
        out = Path(args.json_out)
        if not out.is_absolute():
            out = MODULE_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(counters, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"записано: {out}")
    if args.check:
        problems: list[str] = []
        if counters["tests"]["metrics_json"] != counters["tests"]["collected"]:
            problems.append(
                f"`reports/METRICS.json` → tests.collected = {counters['tests']['metrics_json']}, "
                f"а по pytest — {counters['tests']['collected']}: выполните scripts/collect_metrics.py"
            )
        if counters["tests"]["metrics_json"] is None:
            problems.append("`reports/METRICS.json` не читается — счётчик тестов не с чем сверить")
        if not counters["documents"]["consistent"]:
            problems.append(
                "`scripts/check_numbers.py` красный: числа в документах не сходятся с METRICS.json — "
                "выполните scripts/collect_metrics.py и scripts/counters.py --fix-docs"
            )
        if _replace_report_block(counters, write=False):
            problems.append(f"блок счётчиков в {REPORT} устарел: python3 scripts/counters.py --write-report")
        stale = _apply(counters, write=False)
        if stale:
            problems.append("числа тестов в документах расходятся со счётчиком: " + ", ".join(stale))
        if problems:
            for problem in problems:
                print(f"РАСХОЖДЕНИЕ: {problem}")
            return 1
        print(
            f"счётчики согласованы: тестов {counters['tests']['collected']} в {counters['tests']['files']} файлах, "
            f"black — "
            + (
                f"{counters['formatting']['black_files']} файлов"
                if counters["formatting"]["available"]
                else "недоступен в этой среде"
            )
            + f", check_numbers — {counters['documents']['checked']} документов"
        )
        return 0
    if args.write_report:
        changed = _replace_report_block(counters, write=True)
        print(f"блок счётчиков в {REPORT}: {'обновлён' if changed else 'уже актуален'}")
    if args.fix_docs:
        touched = _apply(counters, write=True)
        print("подставлены числа в строки: " + (", ".join(touched) if touched else "все уже верны"))
    for line in block(counters):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
