"""Точка входа собранного .exe: запускает сервис и открывает браузер.

Аргументы:

    spanverify.exe                # порт 8765, браузер открывается сам
    spanverify.exe --port 9000    # другой порт
    spanverify.exe --no-browser   # не открывать браузер (для тестов и CI)

Если порт занят, выбирается следующий свободный — приложение не падает из-за
занятого порта, а сообщает фактический адрес.
"""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from pathlib import Path

# Запуск возможен двумя способами: из собранного .exe (пакет внутри бандла) и
# напрямую как скрипт (`python scripts/exe_launcher.py`) — во втором случае
# Python кладёт в sys.path только папку скрипта, поэтому корень проекта
# добавляем сами. В собранном .exe эта ветка просто ничего не меняет.
if __package__ in (None, "") and not getattr(sys, "frozen", False):
    _root = Path(__file__).resolve().parent.parent
    if (_root / "spanverify").is_dir():
        sys.path.insert(0, str(_root))
    os.chdir(_root) if (_root / "config").is_dir() else None

from spanverify import __version__  # noqa: E402
from spanverify.server import DEFAULT_PORT, free_port, serve  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Разобрать аргументы и запустить сервис (блокирующий вызов)."""
    parser = argparse.ArgumentParser(prog="spanverify", description=f"SpanVerify {__version__}")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--mode", choices=["demo", "hf"], default="demo")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--port-file",
        default=None,
        help="записать фактический порт в файл (для автотестов: если 8765 занят, порт сдвигается)",
    )
    parser.add_argument("--version", action="version", version=f"SpanVerify {__version__}")
    args = parser.parse_args(argv)

    port = free_port(args.port)
    if not args.quiet or args.port_file:
        print(f"[spanverify] сервис будет запущен на http://127.0.0.1:{port}", flush=True)
    if port != args.port:
        print(f"[spanverify] порт {args.port} занят, использую {port}", flush=True)
    if args.port_file:
        try:
            Path(args.port_file).write_text(str(port), encoding="utf-8")
        except OSError as error:  # pragma: no cover - защита от неверного пути
            print(f"[spanverify] не удалось записать файл порта: {error}", flush=True)
    if not args.no_browser:
        try:
            webbrowser.open(f"http://127.0.0.1:{port}")
        except Exception:  # noqa: BLE001 - в среде без браузера это не ошибка
            pass
    serve(host=args.host, port=port, mode=args.mode, quiet=args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
