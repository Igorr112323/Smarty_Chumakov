"""Точка входа исполняемого файла.

Двойной клик по spanverify.exe (или запуск ``python app.py``) поднимает
локальный сервис и открывает веб-интерфейс. Аргументы командной строки
пробрасываются в CLI:

    spanverify.exe                  # сервис + браузер на порту 8000
    spanverify.exe --port 9000
    spanverify.exe analyze --file document.txt --json
"""

from __future__ import annotations

import multiprocessing
import sys

from spanverify.cli import main


def run() -> int:
    multiprocessing.freeze_support()  # корректная работа в собранном .exe
    argv = sys.argv[1:]
    if not argv:
        argv = ["serve", "--open-browser"]
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(run())
