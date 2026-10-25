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
from spanverify.core import configure_stdio  # noqa: E402
from spanverify.server import DEFAULT_PORT, free_port, serve  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Разобрать аргументы и запустить сервис (блокирующий вызов)."""
    configure_stdio()
    parser = argparse.ArgumentParser(prog="spanverify", description=f"SpanVerify {__version__}")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--mode", choices=["demo", "hf"], default="demo")
    parser.add_argument("--weights", default="config/weights.json")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--runtime-dir",
        default=None,
        help="Portable external Python libraries directory (HF only)",
    )
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--port-file",
        default=None,
        help="записать фактический порт в файл (для автотестов: если 8765 занят, порт сдвигается)",
    )
    parser.add_argument("--version", action="version", version=f"SpanVerify {__version__}")
    args = parser.parse_args(argv)

    dll_handles = []  # Keep Windows loader search cookies alive throughout serve().
    if args.runtime_dir:
        libraries = Path(args.runtime_dir).resolve()
        # HF's external runtime includes optional stdlib modules deliberately
        # absent from the small frozen demo executable (e.g. unittest/pydoc).
        # These come from the portable matching Python, never host registry.
        for path in (
            libraries.parent / "python" / "DLLs",
            libraries.parent / "python" / "Lib",
            libraries,
        ):
            if path.is_dir():
                sys.path.insert(0, str(path))
        # Native abi3 extensions (tokenizers) need portable python3.dll/MSVC,
        # not just sys.path. Python 3.8+ uses restricted DLL search directories.
        if hasattr(os, "add_dll_directory"):
            for path in (
                libraries.parent / "python",
                libraries.parent / "python" / "DLLs",
                libraries,
                *libraries.glob("*.libs"),
            ):
                if path.is_dir():
                    dll_handles.append(os.add_dll_directory(str(path)))
        # A frozen stdlib package may already exist but contain only a subset
        # of modules (urllib.parse without urllib.error, etc.). Extend package
        # search paths with the matching portable stdlib, not host Python.
        stdlib = libraries.parent / "python" / "Lib"
        for name, module in list(sys.modules.items()):
            path = stdlib.joinpath(*name.split("."))
            if getattr(module, "__path__", None) is not None and path.is_dir():
                module.__path__ = [str(path), *module.__path__]
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
    serve(
        host=args.host,
        port=port,
        mode=args.mode,
        weights_path=args.weights,
        model_name=args.model,
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
