"""Ключи доступа и журнал аудита HTTP-сервиса (задача 4, промышленный контур).

Модель простая и намеренно не изобретает JWT/OAuth: у внутреннего сервиса есть
файл ключей, у каждого ключа — субъект (кто вызвал) и роль. Роль даёт право на
действие, а не на «страницу»:

=============  ==============================================
роль           права
=============  ==============================================
``admin``      verify, config, model, metrics, shutdown
``operator``   verify, config, model
``viewer``     config, model, metrics
=============  ==============================================

``GET /health`` не требует ключа в любом случае: пробы живости в Kubernetes
(Kubelet) не отправляют заголовков, а закрытая проба убивает pod. ``/metrics``
под ключом ``viewer`` и выше — сборщик метрик обязан передать
``Authorization``/``X-API-Key``; если это мешает, включается
``SPANVERIFY_METRICS_PUBLIC=1`` (только для кластерной сети, см.
``docs/ДЕПЛОЙ_ENTERPRISE.md``).

Ключи в файле хранятся хешами (``key_hash``), а не текстом: украденный
``keys.json`` не должен отдавать рабочие ключи. Сырой ключ принимается только для
локальной разработки и помечается в ``warnings`` файла. Постоянное сравнение —
``hmac.compare_digest``: по времени отклика перебирать ключи нельзя.

Журнал аудита — JSONL (одна строка = одно событие), в нём нет текстов ответа и
контекста: только ``sha256`` ответа, длины, вердикт, длительность, субъект и
маршрут. Так журнал пригоден для разбора инцидентов, но не становится
второй копией проверяемых документов (которые могут содержать
персональные данные).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "AUDIT_EVENT_FIELDS",
    "ROLE_ADMIN",
    "ROLE_OPERATOR",
    "ROLE_VIEWER",
    "ROLES",
    "AuthError",
    "AuthResult",
    "AuditLog",
    "KeyStore",
    "Principal",
    "hash_answer",
    "hash_key",
    "load_keystore",
]

ROLE_ADMIN = "admin"
ROLE_OPERATOR = "operator"
ROLE_VIEWER = "viewer"
ROLES = (ROLE_ADMIN, ROLE_OPERATOR, ROLE_VIEWER)

#: Право → роли, которым оно разрешено. Метрики и проверка конфигурации — read-only,
#: поэтому доступны оператору и наблюдателю; обучение и перезапуск — admin.
PERMISSION_ROLES: dict[str, tuple[str, ...]] = {
    "verify": (ROLE_ADMIN, ROLE_OPERATOR),
    "config": ROLES,
    "model": ROLES,
    "metrics": (ROLE_ADMIN, ROLE_VIEWER),
    "shutdown": (ROLE_ADMIN,),
}

_KEY_RE = re.compile(r"^[A-Za-z0-9._~+-]{16,128}$")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def hash_key(key: str) -> str:
    """SHA-256 ключа в hex — то, что хранится в файле ключей."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def hash_answer(text: str) -> str:
    """Отпечаток ответа для журнала: ``sha256:<hex 64>``.

    Хеш ставится на нормализованный по краям текст: перенос строки в конце поля
    JSON не должен менять отпечаток, иначе сравнение двух записей об одном
    ответе не работает.
    """
    return "sha256:" + hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


class AuthError(Exception):
    """Отказ в доступе. ``status`` — HTTP-код, который отдаёт сервис."""

    def __init__(self, message: str, status: int = 401) -> None:
        super().__init__(message)
        self.status = status
        self.message = message

    def to_dict(self) -> dict[str, Any]:
        return {"error": self.message, "status": self.status}


@dataclass(frozen=True)
class Principal:
    """Кто обращается к сервису: субъект из файла ключей и его роль."""

    subject: str
    role: str
    key_id: str = ""

    def allows(self, permission: str) -> bool:
        return self.role in PERMISSION_ROLES.get(permission, ())

    def to_dict(self) -> dict[str, str]:
        return {"subject": self.subject, "role": self.role, "key_id": self.key_id}


@dataclass
class KeyStore:
    """Набор ключей с ролями. Загружается из JSON-файла, переопределяется окружением."""

    principals: dict[str, Principal] = field(default_factory=dict)  # key_hash → principal
    warnings: tuple[str, ...] = ()
    source: str = ""

    def __len__(self) -> int:
        return len(self.principals)

    @property
    def enabled(self) -> bool:
        return bool(self.principals)

    def authenticate(self, key: str | None) -> Principal:
        """По строке заголовка → субъект, либо :class:`AuthError` (401/403)."""
        if not self.principals:
            # Ключей нет: доступ включён только если сервис явно запущен с
            # файлом ключей. Без него authenticate не вызывается (см. server).
            raise AuthError("хранилище ключей пусто — доступ не настроен", status=503)
        candidate = (key or "").strip()
        if candidate.lower().startswith("bearer "):
            candidate = candidate[7:].strip()
        if not candidate:
            raise AuthError("нет заголовка X-API-Key (или Authorization: Bearer …)", status=401)
        if not _KEY_RE.match(candidate):
            # Формат проверяем до поиска: так «пустой»/«мусорный» заголовок
            # не отличается по времени от «неверного, но похожего».
            raise AuthError("ключ не проходит формат (16–128 символов A-Za-z0-9._~+-)", status=401)
        digest = hash_key(candidate)
        # Постоянное сравнение по всем записям: поиск по словарю тоже не
        # зависит от содержимого попытки, но перечисление сохраняет порядок
        # файла и не даёт отличить «нет такого префикса» от «нет такого ключа».
        for stored, principal in self.principals.items():
            if hmac.compare_digest(stored, digest):
                return principal
        raise AuthError("ключ не найден", status=401)

    def require(self, key: str | None, permission: str) -> AuthResult:
        """Проверка ключа + права на действие; результат годится для журнала."""
        principal = self.authenticate(key)
        if not principal.allows(permission):
            raise AuthError(f"роли {principal.role!r} не разрешено действие {permission!r}", status=403)
        return AuthResult(principal=principal, permission=permission)


@dataclass(frozen=True)
class AuthResult:
    """Успешная проверка: субъект и запрашиваемое действие."""

    principal: Principal
    permission: str

    def to_dict(self) -> dict[str, str]:
        data = self.principal.to_dict()
        data["permission"] = self.permission
        return data


def _record(entry: Mapping[str, Any], index: int, warnings: list[str]) -> tuple[str, Principal] | None:
    subject = str(entry.get("subject") or f"key-{index}")
    role = str(entry.get("role") or ROLE_VIEWER)
    if role not in ROLES:
        warnings.append(f"запись {index}: роль {role!r} неизвестна, пропущена")
        return None
    key_id = str(entry.get("id") or f"k{index}")
    digest = entry.get("key_hash")
    if not digest and entry.get("key"):
        warnings.append(f"запись {index} ({subject}): ключ хранится открытым текстом — допустимо только в разработке")
        digest = hash_key(str(entry["key"]))
    if not digest or not _HASH_RE.match(str(digest)):
        warnings.append(f"запись {index} ({subject}): нет корректного key_hash (64 hex), пропущена")
        return None
    return str(digest).lower(), Principal(subject=subject, role=role, key_id=key_id)


def load_keystore(path: str | Path | None = None, env: Mapping[str, str] | None = None) -> KeyStore:
    """Собрать хранилище: файл ``keys.json`` + переменная ``SPANVERIFY_API_KEYS``.

    Переменная окружения — «subject:hash,subject:hash» (или «subject:роль:ключ»
    для одноразового ключа в контейнере). Так секреты попадают в pod из Secret,
    не через файл в образе.
    """
    environ = os.environ if env is None else env
    warnings: list[str] = []
    principals: dict[str, Principal] = {}
    source = ""

    if path:
        file_path = Path(path)
        if not file_path.is_file():
            raise AuthError(f"файл ключей не найден: {file_path}", status=503)
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        entries = payload.get("keys") if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            raise AuthError("в файле ключей ожидался список keys", status=503)
        for index, entry in enumerate(entries, start=1):
            if not isinstance(entry, Mapping):
                warnings.append(f"запись {index}: не объект, пропущена")
                continue
            record = _record(entry, index, warnings)
            if record is None:
                continue
            digest, principal = record
            if digest in principals:
                warnings.append(f"запись {index}: ключ {principal.key_id} повторяется, вторая проигнорирована")
                continue
            principals[digest] = principal
        source = str(file_path)

    raw = str(environ.get("SPANVERIFY_API_KEYS") or "").strip()
    if raw:
        for index, item in enumerate(part.strip() for part in raw.split(",")):
            if not item:
                continue
            parts = item.split(":")
            if len(parts) == 2:
                subject, value = parts
                digest = value.lower() if _HASH_RE.match(value) else hash_key(value)
                role = ROLE_ADMIN if index == 0 else ROLE_OPERATOR
            elif len(parts) == 3:
                subject, role, value = parts
                digest = value.lower() if _HASH_RE.match(value) else hash_key(value)
            else:
                warnings.append(f"SPANVERIFY_API_KEYS: элемент {item!r} не разобран")
                continue
            if role not in ROLES:
                warnings.append(f"SPANVERIFY_API_KEYS: роль {role!r} неизвестна, элемент пропущен")
                continue
            principals[digest] = Principal(subject=subject, role=role, key_id="env")
        source = f"{source} + SPANVERIFY_API_KEYS" if source else "SPANVERIFY_API_KEYS"

    return KeyStore(principals=principals, warnings=tuple(warnings), source=source or "не настроено")


class AuditLog:
    """JSONL-журнал событий доступа и проверок.

    Запись — одна строка JSON: ``ts`` (unix, с мс), ``event``, поля события,
    ``answer_sha256`` (отпечаток ответа вместо текста) и, при отказе,
    ``reason``. Файл открывается на дописывание, буферизация выключена
    ``flush()`` на каждую запись: процесс в контейнере может быть убит по
    SIGKILL, и последние события не должны пропасть.
    """

    def __init__(self, path: str | Path | None, *, echo: bool = False) -> None:
        self.path = Path(path) if path else None
        self.echo = echo
        self._lock = threading.Lock()
        self._handle = None
        self.written = 0
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def write(self, event: str, **fields: Any) -> dict[str, Any]:
        """Записать событие; возвращает то, что записано (для тестов)."""
        record: dict[str, Any] = {"ts": round(time.time(), 3), "event": event}
        for name, value in fields.items():
            if value is not None:
                record[name] = value
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self._lock:
            self.written += 1
            if self.echo:
                # В контейнере журнал дублируется в stdout: там его собирает
                # лог-агент, а файл в emptyDir переживает только жизнь pod'а.
                print(line, flush=True)
            if self.path is None:
                return record
            with self.path.open("a", encoding="utf-8") as sink:
                sink.write(line + "\n")
                sink.flush()
                os.fsync(sink.fileno())
        return record

    def verify(
        self,
        *,
        answer: str,
        verdict: str,
        score: float,
        latency_ms: float,
        principal: Principal | None,
        route: str,
        mode: str,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Событие успешной проверки: отпечаток ответа, вердикт, оценка и длительность.

        ``score`` и ``threshold`` пишутся теми же именами, что и в ответе
        контракта: по ним видно, насколько близко событие прошло порог решения,
        то есть устойчиво ли оно или держится на последнем знаке после запятой.
        """
        return self.write(
            "verify",
            answer_sha256=hash_answer(answer),
            answer_chars=len(answer),
            verdict=verdict,
            score=round(float(score), 6),
            threshold=round(float(threshold), 6) if threshold is not None else None,
            latency_ms=round(float(latency_ms), 3),
            route=route,
            mode=mode,
            subject=principal.subject if principal else "",
            role=principal.role if principal else "",
            key_id=principal.key_id if principal else "",
        )

    def denial(self, *, reason: str, status: int, route: str, presented_key: str | None) -> dict[str, Any]:
        """Событие отказа: причины и маршрут, но никогда — сам ключ."""
        return self.write(
            "denied",
            reason=reason,
            status=status,
            route=route,
            key_fingerprint=hash_key(presented_key)[:16] if presented_key else "",
        )

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None

    def tail(self, lines: int = 20) -> list[dict[str, Any]]:
        """Последние записи журнала (для диагностики и тестов)."""
        if self.path is None or not self.path.is_file():
            return []
        rows = [line for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        parsed = []
        for line in rows[-lines:]:
            try:
                parsed.append(json.loads(line))
            except ValueError:
                parsed.append({"event": "corrupt", "raw": line[:200]})
        return parsed


#: Поля, которые обязана содержать запись журнала проверки — их проверяет тест
#: и на них ссылается ``docs/ДЕПЛОЙ_ENTERPRISE.md``.
AUDIT_EVENT_FIELDS = (
    "ts",
    "event",
    "answer_sha256",
    "answer_chars",
    "verdict",
    "score",
    "latency_ms",
    "route",
    "mode",
    "subject",
    "role",
)
