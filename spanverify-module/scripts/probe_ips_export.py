#!/usr/bin/env python3
"""Девятая разведка: поисковая выдача ИПС и RTF-выгрузка полного текста (шаг 19-2з).

Факты из ``reports/ips_text_probe.json``:

* ``?searchres=1&a0=<запрос>`` отдаёт страницу с фреймом
  ``?searchlist=1&a0=<запрос>`` — значит, список найденных документов берётся именно
  из ``searchlist``;
* в карточке документа есть адрес ``?savertf=&nd=<N>&page=all`` (кнопка «сохранить в
  RTF») — это кандидат на полный текст акта без сканирования;
* сама карточка текста не содержит (862 знака видимого текста — название и метаданные).

Скрипт проверяет: (1) сколько документов возвращает поиск и в каком виде; (2) какая
выдача у RTF-адреса и совпадает ли извлечённый текст со структурой акта; (3) какие
метаданные (вид, номер, дата) есть в карточке.

Запуск::

    python scripts/probe_ips_export.py --out reports/ips_export_probe.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys as _sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Скрипты запускаются и как модуль (``python -m``), и напрямую (``python scripts/...``),
# поэтому корень модуля добавляется в путь импорта.
_MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(_MODULE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_MODULE_ROOT))

from scripts.rtf_text import rtf_to_text  # noqa: E402 - импорт после правки sys.path

ROOT = Path(__file__).resolve().parents[1]

USER_AGENT = (
    "SpanVerify-CorpusBot/1.0 (+https://github.com/Igorr112323/Smarty_Chumakov; "
    "research corpus; contact: repository issues)"
)
PAUSE_SECONDS = 1.5
PRAVO = "http://pravo.gov.ru"
QUERIES = ("персональные данные", "срок хранения документов")


def fetch(url: str, timeout: int = 120, read_limit: int = 3_000_000) -> dict:
    """Запрос к ИПС (cp1251) с сохранением тела."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html, application/rtf, text/rtf, */*"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - официальный публичный сайт
            raw = response.read(read_limit)
            status = int(getattr(response, "status", 0) or 0)
            content_type = response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as error:
        raw = error.read(read_limit)
        status = int(error.code)
        content_type = error.headers.get("Content-Type", "") if error.headers else ""
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return {"url": url, "status": None, "content_type": "", "bytes": 0, "raw": b"", "error": str(error)[:140]}
    return {
        "url": url,
        "status": status,
        "content_type": content_type,
        "bytes": len(raw),
        "raw": raw,
        "error": None if status < 400 else f"HTTP {status}",
    }


def visible_text(raw: bytes) -> str:
    """Видимый текст HTML-страницы (декодирование cp1251)."""
    html_text = raw.decode("cp1251", "replace")
    body = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html_text)
    body = re.sub(r"(?s)<[^>]+>", " ", body)
    return re.sub(r"\s+", " ", body).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Разведка выдачи и RTF-экспорта ИПС")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "ips_export_probe.json")
    parser.add_argument("--pause", type=float, default=PAUSE_SECONDS)
    args = parser.parse_args()

    searches: list[dict] = []
    found_ids: list[str] = []
    for query in QUERIES:
        encoded = urllib.parse.quote(query.encode("cp1251"))
        result = fetch(f"{PRAVO}/proxy/ips/?searchlist=1&a0={encoded}")
        ids = re.findall(r"nd=(\d+)", result["raw"].decode("cp1251", "replace"))
        unique = []
        for item in ids:
            if item not in unique:
                unique.append(item)
        if not found_ids:
            found_ids = unique[:3]
        text = visible_text(result["raw"])
        searches.append(
            {
                "query": query,
                "url": result["url"],
                "status": result["status"],
                "bytes": result["bytes"],
                "nd_total": len(ids),
                "nd_unique": len(unique),
                "nd_examples": unique[:10],
                "text_chars": len(text),
                "text_head": text[:400],
                "error": result["error"],
            }
        )
        summary = f"код={result['status']} байт={result['bytes']} ссылок_nd={len(ids)} уникальных={len(unique)}"
        print(f"поиск «{query}»: {summary}")
        print(f"::notice title=IPS3 поиск||{query}: {summary}")
        time.sleep(max(0.0, args.pause))

    exports: list[dict] = []
    for nd in found_ids:
        for name, url in (
            ("savertf-page-all", f"{PRAVO}/proxy/ips/?savertf=&nd={nd}&page=all"),
            ("savertf", f"{PRAVO}/proxy/ips/?savertf=&nd={nd}"),
        ):
            result = fetch(url)
            text = rtf_to_text(result["raw"]) if b"{\\rtf" in result["raw"][:200] else ""
            exports.append(
                {
                    "nd": nd,
                    "name": name,
                    "url": url,
                    "status": result["status"],
                    "content_type": result["content_type"],
                    "bytes": result["bytes"],
                    "is_rtf": b"{\\rtf" in result["raw"][:200],
                    "text_chars": len(text),
                    "text_head": text[:400],
                    "has_article": "Статья" in text or "СТАТЬЯ" in text,
                    "error": result["error"],
                }
            )
            summary = (
                f"код={result['status']} тип={(result['content_type'] or '-').split(';')[0]} байт={result['bytes']} "
                f"rtf={exports[-1]['is_rtf']} знаков={len(text)} статья={exports[-1]['has_article']}"
            )
            print(f"экспорт nd={nd} {name}: {summary}")
            print(f"::notice title=IPS3 экспорт nd={nd} {name}::{summary}")
            time.sleep(max(0.0, args.pause))

    cards: list[dict] = []
    for nd in found_ids[:2]:
        result = fetch(f"{PRAVO}/proxy/ips/?docbody=&nd={nd}")
        text = visible_text(result["raw"])
        cards.append(
            {
                "nd": nd,
                "url": result["url"],
                "status": result["status"],
                "bytes": result["bytes"],
                "text_chars": len(text),
                "text_full": text[:700],
                "error": result["error"],
            }
        )
        print(f"карточка nd={nd}: знаков={len(text)}")
        print(f"::notice title=IPS3 карточка nd={nd}::знаков={len(text)}")
        time.sleep(max(0.0, args.pause))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"base": PRAVO, "queries": list(QUERIES), "searches": searches, "exports": exports, "cards": cards},
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
