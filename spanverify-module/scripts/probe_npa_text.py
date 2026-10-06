#!/usr/bin/env python3
"""Четвёртая разведка: где у официальной публикации лежит ТЕКСТ документа (шаг 19-2в).

Известные факты (``reports/npa_probe*.json``):

* ``GET /api/Documents?pageSize=100&index=N`` — список документов (100 элементов,
  всего 1 707 724), поля: ``eoNumber``, ``id``, ``name``, ``complexName``,
  ``number``, ``documentDate``, ``documentTypeId``, ``title``, ``pdfFileLength``;
* ``documentTypes=<полный GUID>`` фильтрует (для «Федеральный закон» — 7878);
* ``GET /document/<eoNumber>`` — 200 HTML (22 КБ) — надо проверить, есть ли в нём текст;
* ``GET /api/Documents/<GUID>`` — 404.

Скрипт берёт страницу просмотра, вырезает теги, считает символы текста и ищет в
нём слова из названия акта, а также собирает все ссылки со страницы — по ним
видно, откуда берётся полный текст (PDF/SVG/HTML). Затем пробует кандидаты
эндпоинтов текста по ``eoNumber`` и GUID. Ничего не предполагается: если признаков
текста нет, в отчёте будет ``text_chars`` мало и ``has_title_words=false``.

Запуск::

    python scripts/probe_npa_text.py --out reports/npa_text_probe.json
"""

from __future__ import annotations

import argparse
import html
import json
import re
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
MAX_READ = 600_000

BASE = "http://publication.pravo.gov.ru"
FEDERAL_LAW = "Федеральный закон"


def fetch(url: str, timeout: int = 60, read_limit: int = MAX_READ) -> dict:
    """Запрос с сохранением тела (в том числе у ошибок)."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json, text/html, text/plain, */*"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - официальный публичный API
            raw = response.read(read_limit)
            status = int(getattr(response, "status", 0) or 0)
            content_type = response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as error:
        raw = error.read(read_limit)
        status = int(error.code)
        content_type = error.headers.get("Content-Type", "") if error.headers else ""
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return {
            "url": url,
            "status": None,
            "content_type": "",
            "bytes": 0,
            "error": str(error)[:140],
            "body": "",
        }
    return {
        "url": url,
        "status": status,
        "content_type": content_type,
        "bytes": len(raw),
        "error": None if status < 400 else f"HTTP {status}",
        "body": raw.decode("utf-8", "replace"),
    }


def strip_tags(markup: str) -> str:
    """Убирает теги и скрипты, декодирует сущности — чтобы измерить объём текста."""
    body = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", markup)
    body = re.sub(r"(?s)<[^>]+>", " ", body)
    body = html.unescape(body)
    return re.sub(r"\s+", " ", body).strip()


def title_words(name: str) -> list[str]:
    """Значимые слова названия акта (для проверки, что текст документа на странице есть)."""
    words = re.findall(r"[А-Яа-яЁё]{6,}", name or "")
    stop = {"которым", "которые", "которого", "разрешается", "утверждении", "изменений", "внесении", "отдельные"}
    return [w for w in words if w.lower() not in stop][:6]


def links(markup: str) -> list[str]:
    """Все адреса со страницы — по ним видно, где лежит полный текст."""
    hrefs = re.findall(r'href="([^"]+)"', markup)
    seen: list[str] = []
    for href in hrefs:
        if href not in seen:
            seen.append(href)
    return seen


def main() -> int:
    parser = argparse.ArgumentParser(description="Разведка текста документов publication.pravo.gov.ru")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "npa_text_probe.json")
    parser.add_argument("--pause", type=float, default=PAUSE_SECONDS)
    args = parser.parse_args()

    sample: list[dict] = []

    # Берём два документа: свежий общего списка и свежий «Федеральный закон».
    for label, url in (
        ("all", f"{BASE}/api/Documents?pageSize=10&index=1"),
        (
            "federal-law",
            f"{BASE}/api/Documents?pageSize=10&index=1&documentTypes=" + "82a8bf1c-3bc7-47ed-827f-7affd43a7f27",
        ),
    ):
        page = fetch(url)
        items = []
        try:
            items = json.loads(page["body"]).get("items", [])
        except (json.JSONDecodeError, AttributeError):
            items = []
        for item in items[:2]:
            if isinstance(item, dict) and item.get("eoNumber"):
                sample.append(
                    {
                        "label": label,
                        **{
                            k: item.get(k)
                            for k in (
                                "eoNumber",
                                "id",
                                "name",
                                "number",
                                "documentDate",
                                "documentTypeId",
                                "pdfFileLength",
                                "pagesCount",
                            )
                        },
                    }
                )
        time.sleep(max(0.0, args.pause))

    view_pages: list[dict] = []
    for entry in sample:
        result = fetch(f"{BASE}/document/{entry['eoNumber']}")
        text = strip_tags(result["body"]) if result["status"] == 200 else ""
        words = title_words(entry.get("name") or "")
        found = [w for w in words if w.lower() in text.lower()]
        hrefs = links(result["body"]) if result["status"] == 200 else []
        marker_links = [h for h in hrefs if re.search(r"(?i)file|pdf|svg|download|text", h)]
        view_pages.append(
            {
                "eoNumber": entry["eoNumber"],
                "document_id": entry.get("id"),
                "name": entry.get("name"),
                "url": result["url"],
                "status": result["status"],
                "content_type": result["content_type"],
                "bytes": result["bytes"],
                "text_chars": len(text),
                "text_head": text[:800],
                "title_words": words,
                "title_words_found": found,
                "has_title_words": bool(words) and len(found) == len(words),
                "marker_links": marker_links[:20],
                "all_links_count": len(hrefs),
                "error": result["error"],
            }
        )
        summary = (
            f"код={result['status']} байт={result['bytes']} текста={len(text)} "
            f"слова_названия={len(found)}/{len(words)} ссылок={len(marker_links)}"
        )
        print(f"view {entry['eoNumber']:16s} {summary}")
        print(f"::notice title=TEXT view {entry['eoNumber']}::{summary}")
        time.sleep(max(0.0, args.pause))

    # Кандидаты эндпоинтов текста — проверяются по первому документу выборки.
    candidates: list[dict] = []
    if sample:
        first = sample[0]
        eo = first["eoNumber"]
        guid = first.get("id") or ""
        for name, url in (
            ("api-document-by-eo", f"{BASE}/api/Documents/{eo}"),
            ("api-document-by-guid", f"{BASE}/api/Documents/{guid}"),
            ("api-document-text-guid", f"{BASE}/api/Documents/{guid}/Text"),
            ("api-text-guid", f"{BASE}/api/Text/{guid}"),
            ("api-file-by-guid", f"{BASE}/api/File/GetFile/{guid}"),
            ("api-file-by-eo", f"{BASE}/api/File/GetFile/{eo}"),
            ("file-pdf-by-eo", f"{BASE}/file/pdf/{eo}"),
            ("document-pdf-by-eo", f"{BASE}/document/{eo}/pdf"),
            (
                "api-document-types-filter-search",
                f"{BASE}/api/Documents?pageSize=100&index=1&SearchTerm=%D0%BF%D0%B5%D1%80%D1%81%D0%BE%D0%BD%D0%B0%D0%BB%D1%8C%D0%BD%D1%8B%D0%B5",
            ),
        ):
            result = fetch(url, read_limit=200_000)
            text = strip_tags(result["body"])
            candidates.append(
                {
                    "name": name,
                    "url": result["url"],
                    "status": result["status"],
                    "content_type": result["content_type"],
                    "bytes": result["bytes"],
                    "text_chars": len(text),
                    "text_head": text[:400],
                    "error": result["error"],
                }
            )
            summary = f"код={result['status']} тип={(result['content_type'] or '-').split(';')[0]} байт={result['bytes']} текста={len(text)}"
            print(f"{name:34s} {summary}")
            print(f"::notice title=TEXT {name}::{summary}")
            time.sleep(max(0.0, args.pause))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"base": BASE, "sample": sample, "view_pages": view_pages, "candidates": candidates},
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
