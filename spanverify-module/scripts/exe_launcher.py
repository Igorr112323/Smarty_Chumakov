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
import sys
import webbrowser

from spanverify import __version__
from spanverify.server import DEFAULT_PORT, free_port, serve


def main(argv: list[str] | None = None) -> int:
    """Разобрать аргументы и запустить сервис (блокирующий вызов)."""
    parser = argparse.ArgumentParser(prog="spanverify", description=f"SpanVerify {__version__}")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--mode", choices=["demo", "hf"], default="demo")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--version", action="version", version=f"SpanVerify {__version__}")
    args = parser.parse_args(argv)

    port = free_port(args.port)
    if port != args.port:
        print(f"[spanverify] порт {args.port} занят, использую {port}", flush=True)
    if not args.no_browser:
        try:
            webbrowser.open(f"http://127.0.0.1:{port}")
        except Exception:  # noqa: BLE001 - в среде без браузера это не ошибка
            pass
    serve(host=args.host, port=port, mode=args.mode, quiet=args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
