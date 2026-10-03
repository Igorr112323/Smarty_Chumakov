#!/usr/bin/env python3
"""Десятая разведка: как ИПС отдаёт список найденных документов (шаг 19-2и).

Факты из ``reports/ips_export_probe.json``:

* ``?searchlist=1&a0=<запрос>`` отвечает 200, но в HTML нет ссылок ``nd=`` — это
  «рабочее пространство» поиска, а список результатов подгружается отдельно;
* адреса документов в карточке имеют вид ``?docbody=&nd=<N>``, выгрузка текста —
  ``?savertf=&nd=<N>&page=all`` (проверена здесь же, по известному ``nd``).

Скрипт: (1) сохраняет структуру страницы поиска (адреса, фрагменты вокруг слов
``list``/``json``/``nd=``); (2) проверяет RTF-выгрузку по известному ``nd``;
(3) перебирает кандидаты адресов списка результатов, показывая, где появляются ``nd=``.

Запуск::

    python scripts/probe_ips_list.py --out reports/ips_list_probe.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

_MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

from scripts.rtf_text import rtf_to_text  # noqa: E402 - импорт после правки sys.path

ROOT = Path(__file__).resolve().parents[1]

USER_AGENT = (
    "SpanVerify-CorpusBot/1.0 (+https://github.com/Igorr112323/Smarty_Chumakov; "
    "research corpus; contact: repository issues)"
)
PAUSE_SECONDS = 1.5
PRAVO = "http://pravo.gov.ru"
QUERY = "персональные данные"
KNOWN_ND = "102078782"


def fetch(url: str, timeout: int = 120, read_limit: int = 2_000_000) -> dict:
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


def addresses(html_text: str, limit: int = 60) -> list[str]:
    """Адреса страницы из href/src/action (без повторов)."""
    found = re.findall(r'(?:href|src|action)\s*=\s*["\']([^"\']+)["\']', html_text, flags=re.I)
    seen: list[str] = []
    for item in found:
        if item not in seen:
            seen.append(item)
    return seen[:limit]


def snippets(html_text: str, pattern: str, limit: int = 6) -> list[str]:
    """Фрагменты текста вокруг совпадений (для чтения структуры страницы)."""
    result: list[str] = []
    for match in list(re.finditer(pattern, html_text, flags=re.I))[:limit]:
        result.append(html_text[max(0, match.start() - 100) : match.end() + 100].replace("\n", " "))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Разведка списка результатов ИПС")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "ips_list_probe.json")
    parser.add_argument("--pause", type=float, default=PAUSE_SECONDS)
    args = parser.parse_args()

    encoded = urllib.parse.quote(QUERY.encode("cp1251"))
    search_page = fetch(f"{PRAVO}/proxy/ips/?searchlist=1&a0={encoded}")
    html_text = search_page["raw"].decode("cp1251", "replace")
    search_report = {
        "url": search_page["url"],
        "status": search_page["status"],
        "bytes": search_page["bytes"],
        "addresses": addresses(html_text),
        "nd_occurrences": len(re.findall(r"nd=\d+", html_text)),
        "snippets_list": snippets(html_text, r"(?:searchlist|searchres|list\b)", 8),
        "snippets_json": snippets(html_text, r"json\.js|\.json|ajax", 4),
        "html_head": html_text[:3000],
    }
    print(
        f"поиск: код={search_page['status']} байт={search_page['bytes']} ссылок_nd={search_report['nd_occurrences']} адресов={len(search_report['addresses'])}"
    )
    print(f"::notice title=IPS4 поиск::адресов={len(search_report['addresses'])} nd={search_report['nd_occurrences']}")
    for item in search_report["addresses"][:20]:
        print(f"   адрес: {item}")
    time.sleep(max(0.0, args.pause))

    # RTF-выгрузка по известному документу: полный текст акта без сканирования.
    exports: list[dict] = []
    for name, url in (
        ("savertf-page-all", f"{PRAVO}/proxy/ips/?savertf=&nd={KNOWN_ND}&page=all"),
        ("savertf", f"{PRAVO}/proxy/ips/?savertf=&nd={KNOWN_ND}"),
    ):
        result = fetch(url)
        is_rtf = b"{\\rtf" in result["raw"][:200]
        text = rtf_to_text(result["raw"]) if is_rtf else ""
        exports.append(
            {
                "name": name,
                "url": url,
                "status": result["status"],
                "content_type": result["content_type"],
                "bytes": result["bytes"],
                "is_rtf": is_rtf,
                "text_chars": len(text),
                "text_head": text[:500],
                "error": result["error"],
            }
        )
        summary = f"код={result['status']} тип={(result['content_type'] or '-').split(';')[0]} байт={result['bytes']} rtf={is_rtf} знаков={len(text)}"
        print(f"выгрузка {name}: {summary}")
        print(f"::notice title=IPS4 выгрузка {name}::{summary}")
        time.sleep(max(0.0, args.pause))

    # Кандидаты адреса списка результатов: где появляются ссылки nd=.
    candidates: list[dict] = []
    for name, suffix in (
        ("list-lst", f"searchlist=1&a0={encoded}&lst=1"),
        ("list-sort", f"searchlist=1&a0={encoded}&sort=0&page=1"),
        ("list-first", f"searchlist=1&a0={encoded}&first=1"),
        ("list-view", f"searchlist=1&a0={encoded}&view=1"),
        ("list-page", f"searchlist=1&a0={encoded}&page=1"),
        ("jsonlist", f"jsonlist=1&a0={encoded}"),
        ("json", f"json=1&a0={encoded}"),
        ("list", f"list=1&a0={encoded}"),
        ("searchres-lst", f"searchres=1&a0={encoded}&lst=1"),
        ("searchlist-a1", f"searchlist=1&a0={encoded}&a1=1"),
    ):
        url = f"{PRAVO}/proxy/ips/?{suffix}"
        result = fetch(url, read_limit=400_000)
        text = result["raw"].decode("cp1251", "replace")
        nds = re.findall(r"nd=(\d+)", text)
        unique = sorted(set(nds))[:10]
        candidates.append(
            {
                "name": name,
                "url": url,
                "status": result["status"],
                "bytes": result["bytes"],
                "nd_count": len(nds),
                "nd_examples": unique,
                "content_type": result["content_type"],
                "visible_head": re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text))[:200],
                "error": result["error"],
            }
        )
        summary = f"код={result['status']} байт={result['bytes']} ссылок_nd={len(nds)} примеры={unique[:3]}"
        print(f"список {name:14s} {summary}")
        print(f"::notice title=IPS4 список {name}::{summary}")
        time.sleep(max(0.0, args.pause))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "base": PRAVO,
                "query": QUERY,
                "known_nd": KNOWN_ND,
                "search_page": search_report,
                "exports": exports,
                "candidates": candidates,
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
