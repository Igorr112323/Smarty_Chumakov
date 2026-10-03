"""HTTP-сервис SpanVerify: проверка ответа относительно контекста.

Маршруты контракта (Приложение Г):

    GET  /              — веб-интерфейс (один HTML, без сборки)
    GET  /health        — работоспособность, режим, порог, версия
    GET  /v1/config     — действующие параметры (содержимое weights.json)
    GET  /v1/model      — сведения о модели/режиме и обученной голове
    POST /v1/verify     — {"answer": "...", "context": "...", ...} → вердикт

Сервер построен на стандартной библиотеке (``http.server``): он запускается и
из исходников, и из собранного .exe, не требуя uvicorn/fastapi. Это осознанный
выбор: меньше зависимостей — меньше риск, что собранный бинарник не стартует
(и сам бинарник меньше). Ответы JSON, заголовки CORS разрешают вызов со
страницы предпросмотра и из внешних скриптов.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from . import __version__
from .engine import WEIGHTS_FILENAME, Verifier
from .webui import INDEX_HTML

MAX_BODY = 4 * 1024 * 1024  # 4 МБ на запрос
DEFAULT_PORT = 8765
DEFAULT_MODE = "demo"


class Service:
    """Ленивая потокобезопасная инициализация движка."""

    def __init__(
        self,
        mode: str = DEFAULT_MODE,
        weights_path: str | Path | None = WEIGHTS_FILENAME,
        model_name: str | None = None,
    ) -> None:
        self.mode = mode
        self.weights_path = weights_path
        self.model_name = model_name
        self.started_at = time.time()
        self._verifier: Verifier | None = None
        self._lock = threading.Lock()

    @property
    def verifier(self) -> Verifier:
        if self._verifier is None:
            with self._lock:
                if self._verifier is None:
                    self._verifier = Verifier(
                        mode=self.mode,
                        weights_path=self.weights_path,
                        model_name=self.model_name,
                    )
        return self._verifier

    # ------------------------------------------------------------ справка

    def health(self) -> dict[str, Any]:
        verifier = self.verifier
        return {
            "status": "ok",
            "version": __version__,
            "mode": verifier.mode,
            "backend": verifier.mode,
            "calibrated": verifier.bundle.isotonic is not None,
            "threshold": round(verifier.bundle.threshold, 6),
            "weights": {name: round(value, 4) for name, value in verifier.bundle.weights.items()},
            "span_threshold": {
                "z": verifier.bundle.span_z,
                "floor": verifier.bundle.span_floor,
                "cap": verifier.bundle.span_cap,
            },
            "head": (verifier.bundle.head or {}).get("type", "none"),
            "synthetic": bool((verifier.bundle.meta or {}).get("synthetic")),
            "uptime_s": round(time.time() - self.started_at, 1),
            "warning": verifier.warning,
        }

    def config(self) -> dict[str, Any]:
        """Действующая конфигурация: то, чем реально считается ответ."""
        verifier = self.verifier
        payload = verifier.bundle.to_dict()
        payload["runtime"] = {
            "weights_path": str(self.weights_path) if self.weights_path else None,
            "weights_found": bool(getattr(verifier, "weights_loaded", False)),
            "weights_source": verifier.bundle.source,
            "dataset": (verifier.bundle.meta or {}).get("dataset"),
            "mode": verifier.mode,
            "version": __version__,
        }
        return payload

    def model(self) -> dict[str, Any]:
        verifier = self.verifier
        head = verifier.bundle.head or {"type": "none", "file": None}
        return {
            "mode": verifier.mode,
            "model_name": verifier.model_name,
            "features": sorted(verifier.bundle.weights),
            "head": head,
            "head_auc_out_of_fold": (verifier.bundle.meta or {}).get("head_auc_out_of_fold"),
            "folds": verifier.bundle.folds or [],
            "seed": verifier.bundle.seed,
            "version": verifier.bundle.version,
            "synthetic": bool((verifier.bundle.meta or {}).get("synthetic")),
            "warning": verifier.warning,
        }

    # ------------------------------------------------------------ проверка

    def verify(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Разобрать запрос контракта и вернуть ответ контракта."""
        if not isinstance(payload, dict):
            raise ValueError("тело запроса должно быть объектом JSON")
        answer = payload.get("answer")
        if answer is None:
            answer = payload.get("text")  # совместимость с прежним полем
        if not isinstance(answer, str):
            raise ValueError("поле 'answer' обязательно и должно быть строкой")
        context = payload.get("context")
        if context is None:
            context = payload.get("prompt")
        if isinstance(context, list):
            context = [str(part) for part in context]
        elif context is not None:
            context = str(context)

        result = self.verifier.verify(
            answer=answer,
            context=context,
            with_tokens=bool(payload.get("with_tokens", False)),
        )
        body = result.to_dict(with_tokens=bool(payload.get("with_tokens", False)))
        body["mode"] = self.mode
        return body


def make_handler(service: Service):
    """Собрать обработчик HTTP для сервиса (нужен для тестов и встраивания)."""

    class Handler(BaseHTTPRequestHandler):
        server_version = f"SpanVerify/{__version__}"
        protocol_version = "HTTP/1.1"

        # -------------------------------------------------------- утилиты

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            if getattr(self.server, "quiet", False):
                return
            print(f"[spanverify] {self.address_string()} {fmt % args}", flush=True)

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            # CORS: страница может открываться с другого источника (предпросмотр).
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            # X-Frame-Options намеренно НЕ выставляется: интерфейс должен
            # открываться во фрейме предпросмотра.
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def _error(self, status: int, message: str) -> None:
            self._json(status, {"error": message, "status": status})

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            if length > MAX_BODY:
                raise ValueError(f"тело запроса больше {MAX_BODY} байт")
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as error:
                raise ValueError(f"некорректный JSON: {error}") from error
            return payload if isinstance(payload, dict) else {"value": payload}

        # -------------------------------------------------------- маршруты

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._send(204, b"", "text/plain; charset=utf-8")

        def do_HEAD(self) -> None:  # noqa: N802
            self.do_GET()

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path.rstrip("/") or "/"
            if path in ("/", "/index.html"):
                self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/health":
                self._json(200, service.health())
            elif path == "/v1/config":
                self._json(200, service.config())
            elif path == "/v1/model":
                self._json(200, service.model())
            elif path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
            else:
                self._error(404, f"маршрут {path} не найден")

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path.rstrip("/")
            if path not in ("/v1/verify", "/verify"):
                self._error(404, f"маршрут {path} не найден")
                return
            try:
                payload = self._read_json()
            except ValueError as error:
                self._error(400, str(error))
                return
            try:
                self._json(200, service.verify(payload))
            except ValueError as error:
                self._error(400, str(error))
            except Exception as error:  # noqa: BLE001 - сервер не должен падать
                self._error(500, f"внутренняя ошибка: {type(error).__name__}: {error}")

    return Handler


def serve(
    host: str = "0.0.0.0",
    port: int = DEFAULT_PORT,
    mode: str = DEFAULT_MODE,
    weights_path: str | Path | None = WEIGHTS_FILENAME,
    model_name: str | None = None,
    quiet: bool = False,
    warmup: bool = True,
) -> None:
    """Запустить сервис (блокирующий вызов)."""
    service = Service(mode=mode, weights_path=weights_path, model_name=model_name)
    if warmup:
        try:
            service.verifier.verify("Прогрев конвейера.", "Прогрев конвейера.")
        except Exception as error:  # noqa: BLE001 - прогрев не должен мешать старту
            print(f"[spanverify] предупреждение: прогрев не удался: {error}", flush=True)
    handler = make_handler(service)
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    httpd.quiet = quiet  # type: ignore[attr-defined]
    print(
        f"SpanVerify {__version__} запущен на http://{host}:{port} "
        f"(режим: {service.verifier.mode}, порог: {service.verifier.bundle.threshold:.4f})",
        flush=True,
    )
    if service.verifier.warning:
        print(f"[spanverify] {service.verifier.warning}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[spanverify] остановка по Ctrl+C", flush=True)
    finally:
        httpd.server_close()


def free_port(preferred: int = DEFAULT_PORT, host: str = "127.0.0.1", attempts: int = 20) -> int:
    """Подобрать свободный порт начиная с ``preferred`` (для .exe-лаунчера)."""
    for offset in range(attempts):
        candidate = preferred + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, candidate))
            except OSError:
                continue
            return candidate
    return preferred
