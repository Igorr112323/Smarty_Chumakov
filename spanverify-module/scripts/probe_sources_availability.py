#!/usr/bin/env python3
"""Фактическая доступность источников белого списка (пункт 3.1 задания).

Задание требует: недоступный источник помечается ``available: false`` с
**фактической** причиной (код ответа, таймаут), а не «предположительно
недоступен». Этот скрипт и получает такие причины.

Для каждого хоста из ``config/sources_whitelist.json``:

* читается ``robots.txt`` (до любых других запросов) — записываются код
  ответа, время ответа, число правил ``Disallow`` для ``*``;
* выполняется один запрос к корню — записывается код ответа или точный текст
  сетевой ошибки (``DNS``, ``TLS``, ``timeout``, код HTTP);
* между запросами выдерживается пауза не менее 1 секунды.

Результат: ``reports/SOURCES_AVAILABILITY.json``. Ничего не скачивается —
только проверяется доступность, поэтому скрипт безопасно запускать в CI.

Запуск::

    python scripts/probe_sources_availability.py
    python scripts/probe_sources_availability.py --timeout 20 --out reports/SOURCES_AVAILABILITY.json
"""

from __future__ import annotations

import argparse
import json
import socket
import ssl
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WHITELIST_PATH = ROOT / "config" / "sources_whitelist.json"
DEFAULT_OUT = ROOT / "reports" / "SOURCES_AVAILABILITY.json"
MIN_PAUSE = 1.0


def load_whitelist(path: Path = WHITELIST_PATH) -> dict:
    """Прочитать белый список источников."""
    return json.loads(path.read_text(encoding="utf-8"))


def probe_url(url: str, user_agent: str, timeout: float) -> dict:
    """Один запрос. Возвращает код ответа либо точную причину отказа."""
    started = time.perf_counter()
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - только https из белого списка
            body = response.read(200_000)
            return {
                "status": int(response.status),
                "elapsed_s": round(time.perf_counter() - started, 3),
                "bytes": len(body),
                "error": None,
                "error_kind": None,
                "body_head": body[:400].decode("utf-8", "replace"),
            }
    except urllib.error.HTTPError as exc:
        return {
            "status": int(exc.code),
            "elapsed_s": round(time.perf_counter() - started, 3),
            "bytes": 0,
            "error": f"HTTP {exc.code} {exc.reason}",
            "error_kind": "http",
            "body_head": "",
        }
    except urllib.error.URLError as exc:
        reason = exc.reason
        kind = "network"
        if isinstance(reason, ssl.SSLError):
            kind = "tls"
        elif isinstance(reason, socket.gaierror):
            kind = "dns"
        elif isinstance(reason, TimeoutError):
            kind = "timeout"
        return {
            "status": None,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "bytes": 0,
            "error": f"{type(reason).__name__}: {reason}",
            "error_kind": kind,
            "body_head": "",
        }
    except TimeoutError as exc:  # pragma: no cover - зависит от сети
        return {
            "status": None,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "bytes": 0,
            "error": f"timeout: {exc}",
            "error_kind": "timeout",
            "body_head": "",
        }


def count_disallow(robots_body: str) -> int:
    """Сколько правил Disallow объявлено для ``User-agent: *``."""
    rules = 0
    current = None
    for line in robots_body.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        key = key.lower()
        if key == "user-agent":
            current = value
        elif key == "disallow" and current == "*":
            rules += 1
    return rules


def probe_all(whitelist: dict, timeout: float, pause: float) -> dict:
    """Пройти по всем источникам белого списка."""
    user_agent = whitelist.get("user_agent", "SpanVerify-CorpusBot/1.1")
    pause = max(MIN_PAUSE, float(pause))
    results: list[dict] = []
    for source in whitelist.get("sources", []):
        host = source["host"]
        robots = probe_url(f"https://{host}/robots.txt", user_agent, timeout)
        time.sleep(pause)
        root = probe_url(f"https://{host}/", user_agent, timeout)
        time.sleep(pause)
        available = bool(root["status"] and 200 <= int(root["status"]) < 400)
        results.append(
            {
                **{key: value for key, value in source.items() if key != "notes"},
                "available": available,
                "reason": None if available else (root["error"] or f"HTTP {root['status']}"),
                "reason_kind": None if available else root["error_kind"],
                "robots_status": robots["status"],
                "robots_error": robots["error"],
                "robots_disallow_rules": (
                    count_disallow(robots.get("body_head", "")) if robots["status"] == 200 else None
                ),
                "root_status": root["status"],
                "elapsed_s": root["elapsed_s"],
                "checked_at": datetime.now(UTC).isoformat(),
            }
        )
    available = [item for item in results if item["available"]]
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "user_agent": user_agent,
        "pause_seconds": pause,
        "timeout_s": timeout,
        "sources_total": len(results),
        "sources_available": len(available),
        "sources_unavailable": len(results) - len(available),
        "by_level": _by_level(results),
        "sources": results,
    }


def _by_level(results: list[dict]) -> dict[str, dict[str, int]]:
    """Сводка «уровень акта → доступно / недоступно»."""
    out: dict[str, dict[str, int]] = {}
    for item in results:
        level = str(item.get("level", "unknown"))
        row = out.setdefault(level, {"total": 0, "available": 0})
        row["total"] += 1
        if item["available"]:
            row["available"] += 1
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="файл результата")
    parser.add_argument("--timeout", type=float, default=15.0, help="таймаут запроса, с")
    parser.add_argument("--pause", type=float, default=MIN_PAUSE, help="пауза между запросами, с (минимум 1)")
    args = parser.parse_args()

    whitelist = load_whitelist()
    report = probe_all(whitelist, args.timeout, args.pause)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"источников: {report['sources_total']}, доступно: {report['sources_available']}")
    for item in report["sources"]:
        mark = "OK " if item["available"] else "НЕТ"
        print(f"  [{mark}] {item['host']:32s} {item.get('level', ''):10s} {item['reason'] or item['root_status']}")
    print(f"записано: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
