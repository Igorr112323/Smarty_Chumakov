#!/usr/bin/env python3
"""Уточняющая разведка API официальной публикации НПА (шаг 19-2а).

Первая разведка (``probe_npa_api.py``) дала факты: ``/api/DocumentTypes`` отвечает
200 JSON, а ``/api/Documents?pageSize=2&index=1`` — 400. Значит, имена параметров
постраничной выдачи другие. Этот скрипт перебирает варианты имён параметров и
способы получить текст документа, ничего не предполагая заранее: каждый вариант —
отдельный запрос, ответ (включая тело ошибки) усекается и сохраняется.

Запуск::

    python scripts/probe_npa_api2.py --out reports/npa_probe2.json
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

USER_AGENT = (
    "SpanVerify-CorpusBot/1.0 (+https://github.com/Igorr112323/Smarty_Chumakov; "
    "research corpus; contact: repository issues)"
)
PAUSE_SECONDS = 1.5
SNIPPET = 1500
MAX_READ = 400_000

BASE = "http://publication.pravo.gov.ru"
PRAVO = "http://pravo.gov.ru"

QUERY = "персональные данные"


def _quote(value: str) -> str:
    """Кодирует значение параметра запроса, включая кириллицу."""
    return urllib.parse.quote(value, safe="")


def base_candidates(fz_type_id: str | None) -> list[tuple[str, str]]:
    """Список проверяемых адресов; зависит от идентификатора типа «Федеральный закон»."""
    fz = fz_type_id or "82a8bf1c-3bc7-47ed-827f-7affd43a"
    return [
        ("documents-bare", f"{BASE}/api/Documents"),
        ("documents-pagesize", f"{BASE}/api/Documents?pageSize=10"),
        ("documents-pagesize-index", f"{BASE}/api/Documents?pageSize=10&index=1"),
        ("documents-PageSize-Index", f"{BASE}/api/Documents?PageSize=10&Index=1"),
        ("documents-page-size", f"{BASE}/api/Documents?page=1&size=10"),
        ("documents-limit-offset", f"{BASE}/api/Documents?limit=10&offset=0"),
        ("documents-count", f"{BASE}/api/Documents?count=10"),
        ("documents-year-param", f"{BASE}/api/Documents?year=2024&pageSize=10&index=1"),
        ("documents-search-endpoint", f"{BASE}/api/Documents/Search?pageSize=5&index=1&searchTerm={_quote(QUERY)}"),
        ("documents-searchterm", f"{BASE}/api/Documents?searchTerm={_quote(QUERY)}&pageSize=5"),
        ("documents-type-filter", f"{BASE}/api/Documents?documentTypes={fz}&pageSize=5"),
        ("document-types", f"{BASE}/api/DocumentTypes"),
        ("api-description", f"{BASE}/api/Description"),
        ("api-help", f"{BASE}/api/Help"),
        ("robots-publication", f"{BASE}/robots.txt"),
        ("robots-pravo", f"{PRAVO}/robots.txt"),
    ]


def fetch(url: str, timeout: int = 60) -> dict:
    """Один запрос; тело ответа (в том числе у ошибок) усекается, а не отбрасывается."""
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
            "json": False,
            "json_kind": None,
            "keys": None,
            "items_count": None,
            "first_item_keys": None,
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

    json_kind = None
    keys = None
    items_count = None
    first_item_keys = None
    if isinstance(parsed, list):
        json_kind = "list"
        items_count = len(parsed)
        if parsed and isinstance(parsed[0], dict):
            first_item_keys = sorted(parsed[0])[:40]
    elif isinstance(parsed, dict):
        json_kind = "object"
        keys = sorted(parsed)[:40]
        for candidate in ("items", "documents", "result", "data"):
            value = parsed.get(candidate)
            if isinstance(value, list):
                items_count = len(value)
                if value and isinstance(value[0], dict):
                    first_item_keys = sorted(value[0])[:40]
                break
        if items_count is None and isinstance(parsed.get("result"), dict):
            inner = parsed["result"]
            keys = sorted(inner)[:40]
            for candidate in ("items", "documents", "data"):
                value = inner.get(candidate)
                if isinstance(value, list):
                    items_count = len(value)
                    if value and isinstance(value[0], dict):
                        first_item_keys = sorted(value[0])[:40]
                    break

    return {
        "url": url,
        "status": status,
        "content_type": content_type,
        "bytes": len(raw),
        "json": parsed is not None,
        "json_kind": json_kind,
        "keys": keys,
        "items_count": items_count,
        "first_item_keys": first_item_keys,
        "body_head": text[:SNIPPET],
        "seconds": round(time.perf_counter() - started, 2),
        "error": None if status < 400 else f"HTTP {status}",
    }


def find_document_id(result: dict) -> str | None:
    """Ищет GUID документа в теле ответа: и в разобранном JSON, и в тексте."""
    import re

    head = result.get("body_head") or ""
    for pattern in (
        r'"documentId"\s*:\s*"([0-9a-fA-F-]{20,})"',
        r'"id"\s*:\s*"([0-9a-fA-F-]{20,})"',
        r'"eoNumber"\s*:\s*"(\d{13,})"',
    ):
        match = re.search(pattern, head)
        if match:
            return match.group(1)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Уточняющая разведка API publication.pravo.gov.ru")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "npa_probe2.json")
    parser.add_argument("--pause", type=float, default=PAUSE_SECONDS)
    args = parser.parse_args()

    # Сначала типы документов: из них берём фактический идентификатор «Федеральный закон».
    types_result = fetch(f"{BASE}/api/DocumentTypes")
    types_result["name"] = "document-types-first"
    print(
        f"document-types-first код={types_result['status']} json={types_result['json']} элементов={types_result['items_count']}"
    )
    print(f"::notice title=API types::{types_result['status']} элементов={types_result['items_count']}")

    fz_id = None
    try:
        payload = json.loads(types_result["body_head"] if types_result["body_head"].rstrip().endswith("]") else "")
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict) and item.get("name") == "Федеральный закон" and item.get("id"):
                    fz_id = str(item["id"])
                    break
    except json.JSONDecodeError:
        fz_id = None
    print(f"::notice title=API types::Федеральный закон id={fz_id}")
    time.sleep(max(0.0, args.pause))

    results: list[dict] = [types_result]
    for name, url in base_candidates(fz_id):
        if name == "document-types":
            continue
        result = fetch(url)
        result["name"] = name
        results.append(result)
        summary = (
            f"код={result['status']} тип={(result['content_type'] or '-').split(';')[0]} байт={result['bytes']} "
            f"json={result['json_kind']} элементов={result['items_count']} ключи={result['keys']}"
        )
        print(f"{name:28s} {summary}")
        print(f"::notice title=API2 {name}::{summary}")
        time.sleep(max(0.0, args.pause))

    # Текст документа: карточка, выдача файла и HTML-страница просмотра.
    document_id = None
    for result in results:
        candidate = find_document_id(result)
        if candidate:
            document_id = candidate
            break

    extra: list[dict] = []
    if document_id:
        for name, url in (
            ("document-card", f"{BASE}/api/Documents/{document_id}"),
            ("document-file", f"{BASE}/api/File/GetFile/{document_id}"),
            ("document-file-alt", f"{BASE}/api/File/GetDocumentFile/{document_id}"),
            ("document-view-html", f"{BASE}/Document/View/{document_id}"),
        ):
            result = fetch(url)
            result["name"] = name
            extra.append(result)
            summary = (
                f"код={result['status']} тип={(result['content_type'] or '-').split(';')[0]} байт={result['bytes']}"
            )
            print(f"{name:28s} {summary}")
            print(f"::notice title=API2 {name}::{summary}")
            time.sleep(max(0.0, args.pause))
    else:
        print("::warning title=API2::идентификатор документа не найден в проверенных ответах")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "base": BASE,
                "query": QUERY,
                "federal_law_type_id": fz_id,
                "document_id_probe": document_id,
                "results": results,
                "extra": extra,
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
