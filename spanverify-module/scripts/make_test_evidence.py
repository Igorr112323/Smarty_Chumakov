"""Отчёт о прогоне тестов из журнала pytest (файл создаётся скриптом, не руками).

Зачем: «тесты проходят» должно подтверждаться файлом, а не словами. Скрипт берёт
журнал уже выполненного прогона, добавляет команду, окружение, число собранных
тестов и код возврата и пишет короткий отчёт в Markdown.

Запуск::

    python -m pytest -q > /tmp/pytest_full.txt 2>&1; echo "exit=$?" >> /tmp/pytest_full.txt
    python scripts/make_test_evidence.py --log /tmp/pytest_full.txt --out reports/TEST_RUN.md
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def collected_count() -> int | None:
    """Сколько тестов собирает pytest (не запуская их)."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    per_file = re.findall(r":\s*(\d+)\s*$", result.stdout, flags=re.MULTILINE)
    if per_file:
        return sum(int(value) for value in per_file)
    match = re.search(r"(\d+)\s+tests?\s+collected", result.stdout)
    return int(match.group(1)) if match else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Отчёт о прогоне тестов")
    parser.add_argument("--log", type=Path, required=True, help="журнал прогона pytest")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "TEST_RUN.md")
    parser.add_argument("--command", default="python -m pytest -q")
    parser.add_argument("--exit", type=int, default=None, help="код возврата (если не указан — берётся из журнала)")
    args = parser.parse_args(argv)

    text = args.log.read_text(encoding="utf-8", errors="replace") if args.log.is_file() else ""
    exit_match = re.findall(r"exit=(-?\d+)", text)
    exit_code = args.exit if args.exit is not None else (int(exit_match[-1]) if exit_match else None)

    progress = [line for line in text.splitlines() if re.search(r"\[\s*\d+%\]", line)]
    skipped = sum(len(re.findall(r"s", line)) for line in progress)
    failures = sorted({match.group(1) for match in re.finditer(r"^FAILED\s+(\S+)", text, flags=re.MULTILINE)})

    collected = collected_count()
    lines = [
        "# Прогон тестов",
        "",
        f"Отчёт собран {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} из журнала `{args.log}`.",
        "",
        f"* команда: `{args.command}`",
        f"* код возврата: {exit_code if exit_code is not None else 'нет данных'}",
        f"* собрано тестов: {collected if collected is not None else 'нет данных'}",
        f"* отмечено пропусков в ходе прогона: {skipped}",
        f"* окружение: {platform.platform()}, Python {platform.python_version()}, CPU {platform.processor() or platform.machine()}",
        "",
    ]
    if exit_code == 0 and not failures:
        lines.append("Вывод: прогон завершился успешно, проваленных тестов в журнале нет.")
    else:
        lines.append("Вывод: прогон **не** завершился успешно; проваленные тесты перечислены ниже.")
        if failures:
            lines += ["", "| Проваленный тест |", "|---|"]
            lines += [f"| `{name}` |" for name in failures]
    lines += [
        "",
        "Журнал целиком — в файле, указанном выше; здесь только сводка, чтобы её нельзя",
        "было прочитать неоднозначно.",
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"отчёт: {args.out}")
    print(
        json.dumps(
            {"exit_code": exit_code, "collected": collected, "skipped": skipped, "failures": failures},
            ensure_ascii=False,
        )
    )
    return 0 if exit_code == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
