"""HTTP-сервис SpanVerify: проверка ответа относительно контекста.

Маршруты контракта (Приложение Г):

    GET  /              — веб-интерфейс (один HTML, без сборки)
    GET  /health        — работоспособность, режим, порог, версия
    GET  /v1/config     — действующие параметры (содержимое weights.json)
    GET  /v1/model      — сведения о модели/режиме и обученной голове
    POST /v1/verify     — {"answer": "...", "context": "...", ...} → вердикт
    GET  /metrics       — метрики Prometheus (текстовая экспозиция, задача 4)

Промышленный контур (задача 4): при переданном файле ключей (``--api-keys`` или
``SPANVERIFY_API_KEYS``) все маршруты, кроме ``GET /health`` и веб-интерфейса,
требуют заголовок ``X-API-Key``; права по ролям — в :mod:`spanverify.auth`.
Каждый запрос попадает в метрики (длительность, код, вердикт), а каждая проверка
и каждый отказ — в JSONL-журнал аудита (``--audit-log``), где вместо текста ответа
лежит его ``sha256``.

Сервер построен на стандартной библиотеке (``http.server``): он запускается и
из исходников, и из собранного .exe, не требуя uvicorn/fastapi. Это осознанный
выбор: меньше зависимостей — меньше риск, что собранный бинарник не стартует
(и сам бинарник меньше). Ответы JSON, заголовки CORS разрешают вызов со
страницы предпросмотра и из внешних скриптов.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from . import __version__
from .auth import AuditLog, AuthError, KeyStore, Principal, load_keystore
from .engine import WEIGHTS_FILENAME, Verifier
from .metrics import global_metrics
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
        auth: KeyStore | None = None,
        audit: AuditLog | Path | str | None = None,
    ) -> None:
        self.mode = mode
        self.weights_path = weights_path
        self.model_name = model_name
        self.started_at = time.time()
        self.auth = auth if auth is not None else KeyStore()
        self.audit = audit if isinstance(audit, AuditLog) else (AuditLog(audit) if audit else AuditLog(None))
        self.metrics = global_metrics()
        self.metrics.mode = mode
        self.metrics.version = __version__
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
            "weights_source": self.verifier.bundle.source,
            "warning": verifier.warning,
            # Аддитивные поля промышленного контура (задача 4): видны в /health,
            # чтобы пробва и человек видели, включён ли доступ и журнал.
            "auth": {"enabled": self.auth.enabled, "source": self.auth.source, "keys": len(self.auth)},
            "audit": {"enabled": self.audit.enabled, "path": str(self.audit.path) if self.audit.path else None},
        }

    def metrics_text(self) -> str:
        """Текст для ``GET /metrics`` (Prometheus text format 1.0.0)."""
        self.metrics.mode = self.verifier.mode
        return self.metrics.render()

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
            "measures": {
                "ai_participation": (
                    "оценка доли участия ИИ [0..1]: средняя вероятность машинного стиля "
                    "по содержательным токенам; калибровка на синтетическом корпусе"
                ),
                "ai_share": "доля спорного (недостоверного) текста, мягкая оценка",
                "ai_share_hard": "доля содержательных токенов выше порога маски",
                "score": "оценка недостоверности ответа по документу-контексту",
            },
            "participation": {
                "loaded": verifier.participation is not None,
                "auc_out_of_fold": getattr(verifier.participation, "auc_out_of_fold", None),
                "calibrated_on": getattr(verifier.participation, "calibrated_on", None),
                "features": list(getattr(verifier.participation, "features", [])),
            },
            "warning": verifier.warning,
        }

    # ------------------------------------------------------------ проверка

    def verify(self, payload: dict[str, Any], principal: Principal | None = None) -> dict[str, Any]:
        """Разобрать запрос контракта и вернуть ответ контракта.

        Считает длительность, обновляет метрики и пишет событие аудита. Поле
        ``latency_ms`` в ответе — аддитивное: старые клиенты его игнорируют.
        """
        started = time.perf_counter()
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
        latency_ms = (time.perf_counter() - started) * 1000.0
        body["latency_ms"] = round(latency_ms, 3)
        self.metrics.observe("/v1/verify", 200, latency_ms / 1000.0, str(body.get("verdict", "")), self.verifier.mode)
        if self.audit.enabled:
            self.audit.verify(
                answer=answer,
                verdict=str(body.get("verdict", "")),
                score=float(body.get("score", 0.0)),
                threshold=float(body.get("threshold", 0.0)),
                latency_ms=latency_ms,
                principal=principal,
                route="/v1/verify",
                mode=self.verifier.mode,
            )
        return body


def make_handler(service: Service):
    """Собрать обработчик HTTP для сервиса (нужен для тестов и встраивания)."""

    class Handler(BaseHTTPRequestHandler):
        server_version = f"SpanVerify/{__version__}"
        protocol_version = "HTTP/1.1"
        # Код последнего отправленного ответа — нужен, чтобы записать запрос в
        # метрики по тому же коду, который увидел клиент (в /metrics и 404).
        _status_of_last_response = 200

        # -------------------------------------------------------- утилиты

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            if getattr(self.server, "quiet", False):
                return
            print(f"[spanverify] {self.address_string()} {fmt % args}", flush=True)

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self._status_of_last_response = status
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

        # --------------------------------------------------- доступ и метрики

        def _presented_key(self) -> str | None:
            """Ключ из X-API-Key или из Authorization: Bearer … (что удобнее клиенту)."""
            return self.headers.get("X-API-Key") or self.headers.get("Authorization")

        def _guard(self, permission: str, path: str) -> Principal | None:
            """Проверить право; None — доступ включён не был (роль не требуется).

            При отказе ответ и событие журнала уже отправлены, вызывающий код
            обязан остановиться. Открытым остаётся только ``/health`` и
            веб-интерфейс: пробы живости Kubernetes не умеют заголовки.
            """
            if not service.auth.enabled:
                return None
            if permission == "metrics" and os.environ.get("SPANVERIFY_METRICS_PUBLIC") == "1":
                return None
            try:
                result = service.auth.require(self._presented_key(), permission)
            except AuthError as error:
                service.audit.denial(
                    reason=error.message, status=error.status, route=path, presented_key=self._presented_key()
                )
                service.metrics.observe(path, error.status, 0.0, "", service.verifier.mode)
                self._json(error.status, error.to_dict())
                return None
            return result.principal

        def _finish(self, path: str, status: int, started: float) -> None:
            service.metrics.observe(path, status, (time.perf_counter() - started), "", service.verifier.mode)

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._send(204, b"", "text/plain; charset=utf-8")

        def do_HEAD(self) -> None:  # noqa: N802
            self.do_GET()

        def do_GET(self) -> None:  # noqa: N802
            started = time.perf_counter()
            path = urlparse(self.path).path.rstrip("/") or "/"
            if path in ("/", "/index.html"):
                self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/health":
                self._json(200, service.health())
            elif path == "/metrics":
                if self._guard("metrics", path) is None and service.auth.enabled:
                    return
                body = service.metrics_text().encode("utf-8")
                self._send(200, body, "text/plain; version=0.0.4; charset=utf-8")
            elif path == "/v1/config":
                if self._guard("config", path) is None and service.auth.enabled:
                    return
                self._json(200, service.config())
            elif path == "/v1/model":
                if self._guard("model", path) is None and service.auth.enabled:
                    return
                self._json(200, service.model())
            elif path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
            else:
                self._error(404, f"маршрут {path} не найден")
            self._finish(path, self._status_of_last_response, started)

        def do_POST(self) -> None:  # noqa: N802
            started = time.perf_counter()
            path = urlparse(self.path).path.rstrip("/")
            if path not in ("/v1/verify", "/verify"):
                self._error(404, f"маршрут {path} не найден")
                self._finish(path, 404, started)
                return
            principal = self._guard("verify", path)
            if service.auth.enabled and principal is None:
                return
            try:
                payload = self._read_json()
            except ValueError as error:
                self._error(400, str(error))
                self._finish(path, 400, started)
                return
            try:
                self._json(200, service.verify(payload, principal=principal))
            except ValueError as error:
                self._error(400, str(error))
                self._finish(path, 400, started)
            except Exception as error:  # noqa: BLE001 - сервер не должен падать
                self._error(500, f"внутренняя ошибка: {type(error).__name__}: {error}")
                self._finish(path, 500, started)

    return Handler


def serve(
    host: str = "0.0.0.0",
    port: int = DEFAULT_PORT,
    mode: str = DEFAULT_MODE,
    weights_path: str | Path | None = WEIGHTS_FILENAME,
    model_name: str | None = None,
    quiet: bool = False,
    warmup: bool = True,
    api_keys: str | Path | None = None,
    audit_log: str | Path | None = None,
) -> None:
    """Запустить сервис (блокирующий вызов).

    ``api_keys`` — путь к ``keys.json`` (или ``None``: тогда только переменная
    ``SPANVERIFY_API_KEYS``), ``audit_log`` — путь JSONL-журнала. Оба параметра
    аддитивны: без них поведение ровно прежнее — доступ открыт, журнал не пишется.
    """
    auth = load_keystore(api_keys)
    service = Service(mode=mode, weights_path=weights_path, model_name=model_name, auth=auth, audit=audit_log)
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
    if auth.enabled:
        print(f"[spanverify] доступ по ключам включён: {auth.source}, ключей {len(auth)}", flush=True)
    else:
        print("[spanverify] доступ по ключам выключен (keys.json/SPANVERIFY_API_KEYS не заданы)", flush=True)
    if service.audit.enabled:
        print(f"[spanverify] журнал аудита: {service.audit.path}", flush=True)
    for notice in auth.warnings:
        print(f"[spanverify] keys: {notice}", flush=True)
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
