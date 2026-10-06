#!/usr/bin/env python3
"""Третья разведка API официальной публикации (шаг 19-2б).

Что уже известно из фактов (``reports/npa_probe.json``, ``reports/npa_probe2.json``):

* ``/api/Documents`` отвечает 200 JSON с полями ``items/currentPage/itemsPerPage/itemsTotalCount/pagesTotalCount``;
* ``pageSize=10`` уменьшает выдачу до 10 элементов, а ``pageSize=5`` даёт 400
  («The value '5' is invalid») — значит, допустимые значения ограничены списком;
* ``robots.txt`` публикации запрещает ``/File``, ``/Search`` и служебные каталоги,
  а ``/api`` и ``/document`` не запрещены.

Этот скрипт добирает недостающие факты: допустимые размеры страницы, смысл
``index``, фильтр по типу документа (нужен полный GUID типа, а не усечённый),
страницу просмотра документа по ``eoNumber`` и карточку по GUID документа.
Ничего не предполагается: всё, что не подтвердилось, остаётся ``None``.

Запуск::

    python scripts/probe_npa_api3.py --out reports/npa_probe3.json
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
MAX_READ = 400_000
HEAD = 1500

BASE = "http://publication.pravo.gov.ru"
FEDERAL_LAW = "Федеральный закон"


def fetch(url: str, timeout: int = 60) -> dict:
    """Запрос с сохранением тела ответа (включая тело ошибки)."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json, text/html, text/plain, */*"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - официальный публичный API
            raw = response.read(MAX_READ)
            status = int(getattr(response, "status", 0) or 0)
            content_type = response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as error:
        raw = error.read(MAX_READ)
        status = int(error.code)
        content_type = error.headers.get("Content-Type", "") if error.headers else ""
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return {
            "url": url,
            "status": None,
            "content_type": "",
            "bytes": 0,
            "json": None,
            "items_count": None,
            "items_per_page": None,
            "items_total_count": None,
            "current_page": None,
            "first_eo_number": None,
            "body_head": "",
            "seconds": None,
            "error": str(error)[:140],
        }

    text = raw.decode("utf-8", "replace")
    parsed: object | None = None
    if "json" in content_type.lower() or text.lstrip()[:1] in "[{":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None

    items = parsed.get("items") if isinstance(parsed, dict) else None
    return {
        "url": url,
        "status": status,
        "content_type": content_type,
        "bytes": len(raw),
        "json": parsed if (parsed is not None and len(raw) <= 200_000) else None,
        "items_count": len(items) if isinstance(items, list) else (len(parsed) if isinstance(parsed, list) else None),
        "items_per_page": parsed.get("itemsPerPage") if isinstance(parsed, dict) else None,
        "items_total_count": parsed.get("itemsTotalCount") if isinstance(parsed, dict) else None,
        "current_page": parsed.get("currentPage") if isinstance(parsed, dict) else None,
        "first_eo_number": (
            items[0].get("eoNumber") if isinstance(items, list) and items and isinstance(items[0], dict) else None
        ),
        "body_head": text[:HEAD],
        "seconds": round(time.perf_counter() - started, 2),
        "error": None if status < 400 else f"HTTP {status}",
    }


def summarize(name: str, result: dict) -> None:
    """Печатает и публикует в аннотации одну краткую строку факта."""
    summary = (
        f"код={result['status']} тип={(result['content_type'] or '-').split(';')[0]} байт={result['bytes']} "
        f"элементов={result['items_count']} всего={result['items_total_count']} страница={result['current_page']} "
        f"первый={result['first_eo_number']}"
    )
    print(f"{name:30s} {summary}")
    print(f"::notice title=API3 {name}::{summary}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Третья разведка API publication.pravo.gov.ru")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "npa_probe3.json")
    parser.add_argument("--pause", type=float, default=PAUSE_SECONDS)
    args = parser.parse_args()

    results: list[dict] = []

    types = fetch(f"{BASE}/api/DocumentTypes")
    types["name"] = "document-types"
    results.append(types)
    summarize(types["name"], types)
    time.sleep(max(0.0, args.pause))

    fz_id = None
    if isinstance(types.get("json"), list):
        for item in types["json"]:
            if isinstance(item, dict) and item.get("name") == FEDERAL_LAW:
                fz_id = str(item.get("id"))
                break
    print(f"::notice title=API3 types::полный идентификатор типа «{FEDERAL_LAW}» = {fz_id}")

    first_page = fetch(f"{BASE}/api/Documents?pageSize=30&index=1")
    first_page["name"] = "documents-page1-size30"
    results.append(first_page)
    summarize(first_page["name"], first_page)
    time.sleep(max(0.0, args.pause))

    first_eo = first_page.get("first_eo_number")
    first_doc_id = None
    if isinstance(first_page.get("json"), dict):
        items = first_page["json"].get("items") or []
        if items and isinstance(items[0], dict):
            first_doc_id = items[0].get("id")

    candidates: list[tuple[str, str]] = [
        ("documents-size-5", f"{BASE}/api/Documents?pageSize=5&index=1"),
        ("documents-size-10", f"{BASE}/api/Documents?pageSize=10&index=1"),
        ("documents-size-20", f"{BASE}/api/Documents?pageSize=20&index=1"),
        ("documents-size-50", f"{BASE}/api/Documents?pageSize=50&index=1"),
        ("documents-size-100", f"{BASE}/api/Documents?pageSize=100&index=1"),
        ("documents-page2-size10", f"{BASE}/api/Documents?pageSize=10&index=2"),
        ("documents-pascal", f"{BASE}/api/Documents?PageSize=10&Index=1"),
        ("documents-type-full-guid", f"{BASE}/api/Documents?pageSize=10&index=1&documentTypes={fz_id}"),
        ("sitemap", f"{BASE}/sitemap.xml"),
    ]
    if first_eo:
        candidates.append(("document-view-by-eo", f"{BASE}/document/{first_eo}"))
    if first_doc_id:
        candidates.append(("document-card-by-guid", f"{BASE}/api/Documents/{first_doc_id}"))

    for name, url in candidates:
        result = fetch(url)
        result["name"] = name
        results.append(result)
        summarize(name, result)
        time.sleep(max(0.0, args.pause))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "base": BASE,
                "federal_law_type_id": fz_id,
                "first_eo_number": first_eo,
                "first_document_id": first_doc_id,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Отчёт: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
