"""Сверка чисел документов с реестром — запуск из корня репозитория.

Рабочая реализация живёт в модуле (``spanverify-module/scripts/check_numbers.py``):
она знает документы, шаблоны чисел и правила чтения строк. Этот файл — тонкая
обёртка, чтобы команда из корня репозитория была такой же короткой, как остальные
контрактные команды::

    python scripts/check_numbers.py

Пути по умолчанию: ``spanverify-module/reports/METRICS.json`` и
``reports/NUMBERS.json``. Любые дополнительные аргументы передаются модулю как
есть (например ``--docs README.md``).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "spanverify-module"


def main() -> int:
    script = MODULE / "scripts" / "check_numbers.py"
    if not script.is_file():
        print(f"нет скрипта сверки чисел: {script}", file=sys.stderr)
        return 2
    command = [
        sys.executable,
        str(script),
        "--metrics",
        str(MODULE / "reports" / "METRICS.json"),
        "--numbers",
        str(ROOT / "reports" / "NUMBERS.json"),
        *sys.argv[1:],
    ]
    return subprocess.call(command, cwd=MODULE)


if __name__ == "__main__":
    raise SystemExit(main())
