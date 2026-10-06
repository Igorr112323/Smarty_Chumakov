#!/usr/bin/env python3
"""Пятая разведка: извлекается ли текст из официальных PDF и какие фильтры есть у API (шаг 19-2г).

Два вопроса, без которых нельзя писать загрузчик корпуса A3:

1. **Текст.** Полный текст акта лежит только в PDF: ``/file/pdf?eoNumber=<eoNumber>``
   (проверено: 200, ``application/octet-stream``, ``%PDF``). Нужно знать, извлекается
   ли из этих PDF **текстовый слой** (pypdf) или это сканы без текста.
2. **Фильтры.** ``documentTypes=<GUID>`` и ``pageSize`` (10/30/100) подтверждены,
   ``pageSize=20/50`` отклонён. Чтобы разведка не гадала, скрипт посылает **заведомо
   неверные значения** кандидатов имён параметров: если параметр существует, ASP.NET
   отвечает 400 с ошибкой валидации по этому имени; если параметр неизвестен — 200.

Скрипт ничего не предполагает: по каждому PDF сохраняются фактические числа
(страницы, символы, доля символов на страницу, найден ли номер акта в тексте).

Запуск::

    python scripts/probe_npa_extract.py --out reports/npa_extract_probe.json
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
MAX_PDF_BYTES = 8_000_000

BASE = "http://publication.pravo.gov.ru"

TYPE_NAMES = (
    "Федеральный закон",
    "Постановление",
    "Приказ",
    "Указ",
    "Распоряжение",
    "Положение",
)

# Кандидаты имён параметров: значение заведомо неверное, чтобы увидеть реакцию валидатора.
PARAM_CANDIDATES = (
    "PublishDateFrom",
    "PublishDateTo",
    "DocumentDateFrom",
    "DocumentDateTo",
    "JdRegDateFrom",
    "JdRegDateTo",
    "SignatoryAuthorityId",
    "DocumentTypes",
    "PageSize",
    "Index",
    "Block",
    "Blocks",
    "SearchTerm",
    "SortBy",
    "OrderBy",
)


def fetch(url: str, timeout: int = 90, read_limit: int = MAX_PDF_BYTES) -> dict:
    """Запрос с сохранением тела и факта ошибки."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json, application/pdf, text/html, */*"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - официальный публичный API
            raw = response.read(read_limit)
            return {
                "url": url,
                "status": int(getattr(response, "status", 0) or 0),
                "content_type": response.headers.get("Content-Type", ""),
                "bytes": len(raw),
                "error": None,
                "raw": raw,
            }
    except urllib.error.HTTPError as error:
        raw = error.read(64_000)
        return {
            "url": url,
            "status": int(error.code),
            "content_type": error.headers.get("Content-Type", "") if error.headers else "",
            "bytes": len(raw),
            "error": f"HTTP {error.code}",
            "raw": raw,
        }
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return {"url": url, "status": None, "content_type": "", "bytes": 0, "error": str(error)[:140], "raw": b""}


def pdf_text(raw: bytes) -> dict:
    """Извлечь текст из PDF байтов через pypdf; вернуть факты без интерпретации."""
    from io import BytesIO

    try:
        from pypdf import PdfReader
    except ImportError as error:  # pragma: no cover - в CI pypdf установлен
        return {"pages": None, "chars": None, "chars_per_page": None, "text_head": "", "error": f"нет pypdf: {error}"}

    try:
        reader = PdfReader(BytesIO(raw))
        pages = []
        for page in reader.pages[:60]:
            pages.append(page.extract_text() or "")
        text = "\n".join(pages)
    except Exception as error:  # noqa: BLE001 - разведка должна пережить любой сбой разбора
        return {
            "pages": None,
            "chars": None,
            "chars_per_page": None,
            "text_head": "",
            "error": f"{type(error).__name__}: {error}"[:200],
        }

    chars = len(text)
    return {
        "pages": len(reader.pages),
        "chars": chars,
        "chars_per_page": round(chars / len(reader.pages), 1) if reader.pages else None,
        "text_head": " ".join(text.split())[:400],
        "error": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Разведка извлечения текста из PDF публикации")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "npa_extract_probe.json")
    parser.add_argument("--pause", type=float, default=PAUSE_SECONDS)
    parser.add_argument("--per-type", type=int, default=1, help="сколько документов каждого типа разобрать")
    args = parser.parse_args()

    types_result = fetch(f"{BASE}/api/DocumentTypes")
    types_payload = []
    try:
        types_payload = json.loads(types_result["raw"].decode("utf-8"))
    except json.JSONDecodeError:
        types_payload = []
    type_ids = {
        item.get("name"): item.get("id")
        for item in types_payload
        if isinstance(item, dict) and item.get("name") in TYPE_NAMES
    }
    print(f"типы найдены: {sorted(type_ids)}")
    print(f"::notice title=EXTRACT types::найдено типов {len(type_ids)}: {sorted(type_ids)}")
    time.sleep(max(0.0, args.pause))

    # 1. Какие фильтры реально есть у /api/Documents (по реакции валидатора на неверное значение).
    discovered: list[dict] = []
    for name in PARAM_CANDIDATES:
        url = f"{BASE}/api/Documents?pageSize=10&index=1&{name}={urllib.parse.quote('неверное_значение', safe='')}"
        result = fetch(url, read_limit=4000)
        body = result["raw"].decode("utf-8", "replace")
        known = result["status"] == 400 and name.lower() in body.lower()
        discovered.append(
            {
                "param": name,
                "status": result["status"],
                "recognized": bool(known),
                "body_head": body[:300],
            }
        )
        print(f"param {name:22s} код={result['status']} распознан={known}")
        print(f"::notice title=EXTRACT param {name}::код={result['status']} распознан={known}")
        time.sleep(max(0.0, args.pause))

    # 2. Извлекается ли текст из PDF: по одному документу каждого типа.
    documents: list[dict] = []
    for type_name, type_id in sorted(type_ids.items()):
        listing = fetch(f"{BASE}/api/Documents?pageSize=10&index=1&documentTypes={type_id}", read_limit=200_000)
        try:
            items = json.loads(listing["raw"].decode("utf-8")).get("items", [])
        except json.JSONDecodeError:
            items = []
        picked = [item for item in items if isinstance(item, dict) and item.get("eoNumber")][: args.per_type]
        for item in picked:
            documents.append(
                {
                    "type_name": type_name,
                    "type_id": type_id,
                    "eoNumber": item.get("eoNumber"),
                    "document_id": item.get("id"),
                    "name": item.get("name"),
                    "number": item.get("number"),
                    "documentDate": item.get("documentDate"),
                    "pagesCount": item.get("pagesCount"),
                    "pdfFileLength": item.get("pdfFileLength"),
                }
            )
        time.sleep(max(0.0, args.pause))

    extraction: list[dict] = []
    for entry in documents:
        url = f"{BASE}/file/pdf?eoNumber={entry['eoNumber']}"
        result = fetch(url)
        record = dict(entry)
        record.update(
            {
                "pdf_url": url,
                "pdf_status": result["status"],
                "pdf_content_type": result["content_type"],
                "pdf_bytes": result["bytes"],
                "pdf_error": result["error"],
            }
        )
        if result["status"] == 200 and result["raw"][:4] == b"%PDF":
            extracted = pdf_text(result["raw"])
            record.update({f"text_{key}": value for key, value in extracted.items()})
            # Контроль: номер акта из карточки обязан встречаться в извлечённом тексте.
            number = (entry.get("number") or "").strip()
            record["act_number_found"] = bool(number) and number in (extracted.get("text_head") or "") + ""
            record["act_number_checked"] = number
        else:
            record.update({"text_pages": None, "text_chars": None, "text_error": result["error"]})
        extraction.append(record)
        summary = (
            f"код={result['status']} байт={result['bytes']} страниц={record.get('text_pages')} "
            f"символов={record.get('text_chars')} ошибка={record.get('text_error') or record.get('error') or '-'}"
        )
        print(f"pdf {entry['eoNumber']:16s} {str(entry['type_name'])[:22]:22s} {summary}")
        print(f"::notice title=EXTRACT pdf {entry['eoNumber']}::{str(entry['type_name'])[:20]} {summary}")
        time.sleep(max(0.0, args.pause))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "base": BASE,
                "type_ids": type_ids,
                "params": discovered,
                "documents": documents,
                "extraction": extraction,
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
