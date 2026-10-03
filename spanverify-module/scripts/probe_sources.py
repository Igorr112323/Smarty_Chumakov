#!/usr/bin/env python3
"""Проверка доступности официальных источников НПА (разведка перед сбором A3).

Зачем: песочница разработки пускает только GitHub, а сборка корпуса A3 требует
официальных публикаций. Поэтому доступность проверяется там, где сборка и будет
идти — в CI (у раннера свой выход в сеть) и на машине человека. Скрипт только
сообщает факты: код ответа, размер, признак «похоже на PDF/HTML».

Правила вежливости соблюдаются и здесь: пауза между запросами, User-Agent с
контактом, никаких поисковых запросов — только прямые адреса публикаций.

Запуск::

    python scripts/probe_sources.py            # таблица в консоль
    python scripts/probe_sources.py --json reports/probe_sources.json
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Контакт в User-Agent: официальные сайты просят представляться, а не скрываться.
USER_AGENT = (
    "SpanVerify-CorpusBot/1.0 (+https://github.com/Igorr112323/Smarty_Chumakov; "
    "research corpus; contact: repository issues)"
)
PAUSE_SECONDS = 1.2

SOURCES = (
    ("publication.pravo.gov.ru", "http://publication.pravo.gov.ru/"),
    ("pravo.gov.ru", "http://pravo.gov.ru/"),
    ("pravo.gov.ru API", "http://publication.pravo.gov.ru/api/Documents"),
    ("fstec.ru", "https://fstec.ru/dokumenty"),
    ("mintrud.gov.ru", "https://mintrud.gov.ru/docs"),
    ("rospotrebnadzor.ru", "https://rospotrebnadzor.ru/documents"),
    ("eec.eaeunion.org", "https://eec.eaeunion.org/"),
    ("ФСТЭК: приказы", "https://fstec.ru/dokumenty/vse-dokumenty/prikazy"),
    ("Минтруд: приказы", "https://mintrud.gov.ru/docs/mintrud/orders"),
    (
        "статья RusHallu-RAG (PDF)",
        "https://dialogue-conf.org/wp-content/uploads/2026/06/SadkovskiiFNasyrovaRSorokinA.087.pdf",
    ),
)


def probe(url: str, timeout: int = 40) -> dict:
    """Сделать один GET и вернуть факт: код, размер, тип содержимого, ошибку."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - адреса из белого списка
            payload = response.read(4096)
            return {
                "url": url,
                "status": int(getattr(response, "status", 0) or response.getcode()),
                "size_hint": len(payload),
                "content_type": response.headers.get("Content-Type", ""),
                "looks_like_pdf": payload[:4] == b"%PDF",
                "seconds": round(time.perf_counter() - started, 2),
                "error": None,
            }
    except urllib.error.HTTPError as error:
        return {
            "url": url,
            "status": error.code,
            "size_hint": 0,
            "content_type": "",
            "looks_like_pdf": False,
            "seconds": None,
            "error": f"HTTP {error.code}",
        }
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return {
            "url": url,
            "status": None,
            "size_hint": 0,
            "content_type": "",
            "looks_like_pdf": False,
            "seconds": None,
            "error": str(error)[:120],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Разведка доступности официальных источников")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--pause", type=float, default=PAUSE_SECONDS)
    args = parser.parse_args()

    results: list[dict] = []
    for name, url in SOURCES:
        result = probe(url)
        result["name"] = name
        results.append(result)
        mark = "ок" if result["status"] == 200 else ("проверка" if result["status"] else "нет доступа")
        detail = (
            f"код={result['status']} размер={result['size_hint']}Б тип={result['content_type'][:40]}"
            if result["status"]
            else f"ошибка={result['error']}"
        )
        print(f"{name:28s} {mark:12s} {detail}")
        # Аннотация CI: числа доступны без скачивания логов.
        print(f"::notice title=источник {name}::{detail}")
        time.sleep(max(0.0, args.pause))

    available = [item["name"] for item in results if item["status"] == 200]
    blocked = [item["name"] for item in results if item["status"] != 200]
    print(f"\nДоступны: {', '.join(available) or 'нет'}")
    print(f"Недоступны: {', '.join(blocked) or 'нет'}")
    print(f"::notice title=разведка источников::доступно={len(available)} из {len(results)}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"results": results, "available": available, "blocked": blocked}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        print(f"Отчёт: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
