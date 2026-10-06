#!/usr/bin/env python3
"""Разведка API официальной публикации НПА (publication.pravo.gov.ru).

Зачем: чтобы собрать корпус A3 на реальных документах, нужно знать фактические
эндпоинты и поля API, а не догадываться. Из песочницы разработки доступен только
GitHub, поэтому разведка запускается в CI, а её результат (усечённые ответы)
записывается в файл и коммитится отдельным шагом — так факты доступны для чтения
без скачивания артефактов.

Скрипт ничего не предполагает: он делает несколько запросов с паузами, аккуратно
обрезает ответы (первые N символов) и сохраняет их вместе с кодами и заголовками.

Запуск::

    python scripts/probe_npa_api.py --out reports/npa_probe.json
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

USER_AGENT = (
    "SpanVerify-CorpusBot/1.0 (+https://github.com/Igorr112323/Smarty_Chumakov; "
    "research corpus; contact: repository issues)"
)
PAUSE_SECONDS = 1.5
SNIPPET = 1200

BASE = "http://publication.pravo.gov.ru"

# Кандидаты эндпоинтов: сам корень API, список документов, типы документов и файл.
CANDIDATES = (
    ("api-root", f"{BASE}/api"),
    ("documents-page", f"{BASE}/api/Documents?pageSize=2&index=1"),
    ("document-types", f"{BASE}/api/DocumentTypes"),
    ("blocks", f"{BASE}/api/Blocks"),
)


def fetch(url: str, timeout: int = 60) -> dict:
    """Один запрос с усечением ответа; возвращает факт без интерпретации."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json, text/html"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - адрес официального API
            raw = response.read(400_000)
            text = raw.decode("utf-8", "replace")
            parsed: object | None = None
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            return {
                "url": url,
                "status": int(getattr(response, "status", 0) or 0),
                "content_type": response.headers.get("Content-Type", ""),
                "bytes": len(raw),
                "json": parsed is not None,
                "keys": sorted(parsed)[:20] if isinstance(parsed, dict) else None,
                "items_count": (
                    len(parsed.get("items", []))
                    if isinstance(parsed, dict) and isinstance(parsed.get("items"), list)
                    else None
                ),
                "first_item_keys": (
                    sorted(parsed["items"][0])[:30]
                    if isinstance(parsed, dict)
                    and isinstance(parsed.get("items"), list)
                    and parsed["items"]
                    and isinstance(parsed["items"][0], dict)
                    else None
                ),
                "snippet": text[:SNIPPET],
                "seconds": round(time.perf_counter() - started, 2),
                "error": None,
            }
    except urllib.error.HTTPError as error:
        return {
            "url": url,
            "status": error.code,
            "content_type": "",
            "bytes": 0,
            "json": False,
            "keys": None,
            "items_count": None,
            "first_item_keys": None,
            "snippet": "",
            "seconds": None,
            "error": f"HTTP {error.code}",
        }
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return {
            "url": url,
            "status": None,
            "content_type": "",
            "bytes": 0,
            "json": False,
            "keys": None,
            "items_count": None,
            "first_item_keys": None,
            "snippet": "",
            "seconds": None,
            "error": str(error)[:140],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Разведка API publication.pravo.gov.ru")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "npa_probe.json")
    parser.add_argument("--pause", type=float, default=PAUSE_SECONDS)
    args = parser.parse_args()

    results: list[dict] = []
    for name, url in CANDIDATES:
        result = fetch(url)
        result["name"] = name
        results.append(result)
        summary = (
            f"код={result['status']} json={result['json']} байт={result['bytes']} "
            f"элементов={result['items_count']} ключи={result['keys']}"
            if result["status"]
            else f"ошибка={result['error']}"
        )
        print(f"{name:16s} {summary}")
        print(f"::notice title=API {name}::{summary}")
        time.sleep(max(0.0, args.pause))

    # Первый полученный идентификатор документа: по нему проверяем выдачу карточки.
    document_id = None
    for result in results:
        if result.get("items_count") and result.get("snippet") and '"id"' in result["snippet"]:
            try:
                payload = json.loads(result["snippet"] if result["snippet"].endswith("}") else "")
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict) and payload.get("items"):
                document_id = str(payload["items"][0].get("id"))
                break
    if document_id is None:
        # Ответ мог быть усечён: пробуем вытащить идентификатор регулярным поиском.
        import re

        for result in results:
            match = re.search(r'"id"\s*:\s*"([0-9a-fA-F-]{8,})"', result.get("snippet") or "")
            if match:
                document_id = match.group(1)
                break

    if document_id:
        for name, url in (
            ("document-card", f"{BASE}/api/Documents/{document_id}"),
            ("document-file", f"{BASE}/api/File/GetFile/{document_id}"),
        ):
            result = fetch(url)
            result["name"] = name
            results.append(result)
            summary = (
                f"код={result['status']} тип={result['content_type'][:35]} байт={result['bytes']}"
                if result["status"]
                else f"ошибка={result['error']}"
            )
            print(f"{name:16s} {summary}")
            print(f"::notice title=API {name}::{summary}")
            time.sleep(max(0.0, args.pause))
    else:
        print("::warning title=API::идентификатор документа не найден в усечённых ответах")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"base": BASE, "results": results, "document_id_probe": document_id}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"Отчёт: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
