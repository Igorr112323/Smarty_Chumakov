"""HTTP API и встроенный веб-интерфейс (стандартная библиотека).

Маршруты:

    GET  /health        — состояние сервиса и бэкенда
    GET  /v1/config     — действующая конфигурация
    POST /v1/verify     — {"text": "...", "threshold": 0.5, "explain": false}
    GET  /              — веб-интерфейс (один HTML-файл, без сборки)

Сервер потоковый, без внешних зависимостей: запускается и из собранного
.exe, и из исходников.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from . import __version__
from .config import Config
from .detector import Detector
from .webui import INDEX_HTML

MAX_BODY = 4 * 1024 * 1024  # 4 МБ текста на запрос


class Service:
    """Ленивая инициализация детектора + потокобезопасный доступ."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config.load()
        self._detector: Detector | None = None
        self._lock = threading.Lock()

    @property
    def detector(self) -> Detector:
        if self._detector is None:
            with self._lock:
                if self._detector is None:
                    self._detector = Detector(self.config)
        return self._detector

    def warmup(self) -> None:
        self.detector.analyze("Прогрев модели.")

    def health(self) -> dict[str, Any]:
        backend = getattr(self.detector.backend, "name", "unknown")
        info: dict[str, Any] = {
            "status": "ok",
            "version": __version__,
            "backend": backend,
            "calibrated": self.detector.calibrator is not None,
            "threshold": self.config.threshold,
        }
        if backend == "hf":
            from .backends.hf import HFBackend

            ok, reason = HFBackend.dependencies()
            info["hf_ready"] = ok
            if not ok:
                info["hf_error"] = reason
        else:
            info["notice"] = (
                "ДЕМО-РЕЖИМ: оценивается работоспособность конвейера, "
                "а не достоверность текста. Метрики — только в режиме 'hf'."
            )
        return info


def make_handler(service: Service):
    class Handler(BaseHTTPRequestHandler):
        server_version = f"SpanVerify/{__version__}"
        protocol_version = "HTTP/1.1"

        # ---------- утилиты ----------

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            if "--quiet" not in self.server.args:  # type: ignore[attr-defined]
                super().log_message(fmt, *args)

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(code, body, "application/json; charset=utf-8")

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            if length > MAX_BODY:
                raise ValueError(f"слишком большой запрос: {length} байт (лимит {MAX_BODY})")
            raw = self.rfile.read(length)
            try:
                data = json.loads(raw.decode("utf-8"))
            except UnicodeDecodeError as exc:
                raise ValueError("тело запроса должно быть в UTF-8") from exc
            if not isinstance(data, dict):
                raise ValueError("ожидается JSON-объект")
            return data

        # ---------- маршруты ----------

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._send(204, b"", "text/plain; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path.rstrip("/") or "/"
            if path == "/":
                self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/health":
                self._json(200, service.health())
                return
            if path == "/v1/config":
                self._json(200, service.config.to_dict())
                return
            if path == "/v1/model":
                det = service.detector
                self._json(
                    200,
                    {
                        "backend": det.backend_name,
                        "description": getattr(det.backend, "description", ""),
                        "calibrated": det.calibrator is not None,
                        "calibration": det.calibrator.meta if det.calibrator else None,
                    },
                )
                return
            if path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
                return
            self._json(404, {"error": "not_found", "path": path})

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path.rstrip("/")
            if path not in {"/v1/verify", "/verify"}:
                self._json(404, {"error": "not_found", "path": path})
                return
            try:
                data = self._read_json()
            except ValueError as exc:
                self._json(400, {"error": "bad_request", "message": str(exc)})
                return

            text = data.get("text")
            if not isinstance(text, str) or not text.strip():
                self._json(400, {"error": "bad_request", "message": "поле 'text' обязательно"})
                return

            threshold = data.get("threshold")
            if threshold is not None:
                try:
                    threshold = float(threshold)
                except (TypeError, ValueError):
                    self._json(400, {"error": "bad_request", "message": "'threshold' — число"})
                    return
                if not 0.0 <= threshold <= 1.0:
                    self._json(400, {"error": "bad_request", "message": "'threshold' в [0, 1]"})
                    return

            try:
                if data.get("explain"):
                    payload = service.detector.explain(text, threshold=threshold)
                else:
                    payload = {
                        "result": service.detector.analyze(text, threshold=threshold).to_dict()
                    }
            except Exception as exc:  # noqa: BLE001 - сервис не должен падать
                self._json(
                    500,
                    {
                        "error": "internal_error",
                        "message": f"{type(exc).__name__}: {exc}",
                        "hint": "для режима 'hf' проверьте зависимости и доступ к весам модели",
                    },
                )
                return
            self._json(200, payload)

    return Handler


def create_server(config: Config | None = None, quiet: bool = False) -> ThreadingHTTPServer:
    config = config or Config.load()
    service = Service(config)
    handler = make_handler(service)
    server = ThreadingHTTPServer((config.host, config.port), handler)
    server.daemon_threads = True
    server.args = ["--quiet"] if quiet else []  # type: ignore[attr-defined]
    return server


def serve(config: Config | None = None, quiet: bool = False) -> None:
    config = config or Config.load()
    server = create_server(config, quiet=quiet)
    host, port = server.server_address[0], server.server_address[1]
    print(f"SpanVerify {__version__} — интерфейс: http://localhost:{port}")
    print(f"API: POST http://localhost:{port}/v1/verify | GET /health")
    print(f"Режим: {config.backend} | {config.describe()}")
    if config.backend == "surrogate":
        print("ВНИМАНИЕ: демо-режим. Достоверность текста здесь не оценивается.")
    print("Остановка: Ctrl+C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановлено.")
    finally:
        server.server_close()
