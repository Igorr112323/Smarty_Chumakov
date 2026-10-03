#!/usr/bin/env python3
"""Одиннадцатая разведка: последняя проверка ИПС перед выбором источника текста (шаг 19-2н).

Что уже известно (``reports/ips_*.json``):

* ``?searchlist=1&a0=<запрос>`` содержит постраничную навигацию
  (``…&start=20``), но ссылок ``nd=`` в HTML нет — список результатов строится скриптами
  сайта (``?json.js``, ``?script.js``, ``?listsel.js``);
* ``?savertf=&nd=<N>&page=all`` отдаёт 2239 байт как ``application/x-download``, и это
  не RTF (нет заголовка ``{\\rtf``) — надо посмотреть фактические байты;
* ``?docbody=&nd=<N>`` — карточка без текста.

Скрипт делает четыре вещи:

1. сохраняет **все** адреса карточки (ищет ссылку на текст документа среди ``nd=``);
2. читает ``?json.js`` и ``?script.js`` и выписывает строки, где упоминаются
   ``searchlist``, ``stub``, ``nd=``, ``XMLHttp`` — так видно, какой адрес тянет список;
3. вырезает область результатов страницы поиска (заголовок таблицы и первые строки);
4. печатает первые байты выгрузки ``savertf`` (заголовок файла и первые символы).

Если текста в ИПС нет и здесь — источником корпуса A3 будет OCR официальных PDF
(этот путь уже проверен: 2 страницы → 2468 знаков, номер акта найден).

Запуск::

    python scripts/probe_ips_final.py --out reports/ips_final_probe.json
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

ROOT = Path(__file__).resolve().parents[1]

USER_AGENT = (
    "SpanVerify-CorpusBot/1.0 (+https://github.com/Igorr112323/Smarty_Chumakov; "
    "research corpus; contact: repository issues)"
)
PAUSE_SECONDS = 1.5
PRAVO = "http://pravo.gov.ru"
QUERY = "персональные данные"
KNOWN_ND = "102078782"


def fetch(url: str, timeout: int = 120, read_limit: int = 3_000_000) -> dict:
    """Запрос к ИПС (cp1251) с сохранением тела."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html, application/javascript, */*"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - официальный публичный сайт
            raw = response.read(read_limit)
            status = int(getattr(response, "status", 0) or 0)
            content_type = response.headers.get("Content-Type", "")
            disposition = response.headers.get("Content-Disposition", "")
    except urllib.error.HTTPError as error:
        raw = error.read(read_limit)
        status = int(error.code)
        content_type = error.headers.get("Content-Type", "") if error.headers else ""
        disposition = ""
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return {
            "url": url,
            "status": None,
            "content_type": "",
            "disposition": "",
            "bytes": 0,
            "raw": b"",
            "error": str(error)[:140],
        }
    return {
        "url": url,
        "status": status,
        "content_type": content_type,
        "disposition": disposition,
        "bytes": len(raw),
        "raw": raw,
        "error": None if status < 400 else f"HTTP {status}",
    }


def all_addresses(html_text: str) -> list[str]:
    """Все адреса страницы (href/src/action), без ограничения количества."""
    return re.findall(r'(?:href|src|action)\s*=\s*["\']([^"\']+)["\']', html_text, flags=re.I)


def js_lines(text: str, limit: int = 12) -> list[str]:
    """Строки скрипта, где упоминаются адреса выдачи (searchlist/stub/nd=/XMLHttp)."""
    result: list[str] = []
    for line in text.splitlines():
        if re.search(r"searchlist|stub|nd=|XMLHttp|\.asp|searchres", line, flags=re.I):
            result.append(line.strip()[:300])
        if len(result) >= limit:
            break
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Последняя разведка ИПС")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "ips_final_probe.json")
    parser.add_argument("--pause", type=float, default=PAUSE_SECONDS)
    args = parser.parse_args()

    card = fetch(f"{PRAVO}/proxy/ips/?docbody=&nd={KNOWN_ND}")
    card_html = card["raw"].decode("cp1251", "replace")
    card_addresses = all_addresses(card_html)
    nd_links = [item for item in card_addresses if "nd=" in item]
    card_report = {
        "url": card["url"],
        "status": card["status"],
        "bytes": card["bytes"],
        "addresses_total": len(card_addresses),
        "nd_addresses": nd_links[:20],
        "addresses_tail": card_addresses[-15:],
    }
    print(f"карточка: адресов={len(card_addresses)} со ссылкой nd={len(nd_links)}")
    print(f"::notice title=IPS5 карточка::адресов={len(card_addresses)} nd={len(nd_links)}")
    for item in nd_links[:10]:
        print(f"   ссылка: {item}")
    time.sleep(max(0.0, args.pause))

    scripts: list[dict] = []
    for name in ("json.js", "script.js", "listsel.js"):
        result = fetch(f"{PRAVO}/proxy/ips/?{name}")
        text = result["raw"].decode("cp1251", "replace")
        scripts.append(
            {
                "name": name,
                "status": result["status"],
                "bytes": result["bytes"],
                "lines": js_lines(text),
                "head": text[:300],
            }
        )
        print(
            f"скрипт {name}: код={result['status']} байт={result['bytes']} строк-подсказок={len(scripts[-1]['lines'])}"
        )
        print(f"::notice title=IPS5 скрипт {name}::байт={result['bytes']} подсказок={len(scripts[-1]['lines'])}")
        time.sleep(max(0.0, args.pause))

    encoded = urllib.parse.quote(QUERY.encode("cp1251"))
    listing = fetch(f"{PRAVO}/proxy/ips/?searchlist=1&a0={encoded}")
    listing_html = listing["raw"].decode("cp1251", "replace")
    marker = listing_html.find("start=20")
    area = listing_html[max(0, marker - 3000) : marker + 1500] if marker > 0 else listing_html[:4500]
    listing_report = {
        "url": listing["url"],
        "status": listing["status"],
        "bytes": listing["bytes"],
        "docbody_occurrences": len(re.findall(r"docbody", listing_html, flags=re.I)),
        "nd_occurrences": len(re.findall(r"nd=\d+", listing_html)),
        "results_area": re.sub(r"\s+", " ", area)[:4000],
    }
    print(f"список: docbody={listing_report['docbody_occurrences']} nd={listing_report['nd_occurrences']}")
    print(
        f"::notice title=IPS5 список::docbody={listing_report['docbody_occurrences']} nd={listing_report['nd_occurrences']}"
    )
    time.sleep(max(0.0, args.pause))

    export = fetch(f"{PRAVO}/proxy/ips/?savertf=&nd={KNOWN_ND}&page=all")
    raw = export["raw"]
    export_report = {
        "url": export["url"],
        "status": export["status"],
        "content_type": export["content_type"],
        "disposition": export["disposition"],
        "bytes": export["bytes"],
        "is_zip": raw[:2] == b"PK",
        "is_rtf": b"{\\rtf" in raw[:200],
        "header_latin1": raw[:200].decode("latin-1", "replace"),
        "header_cp1251": raw[:200].decode("cp1251", "replace"),
        "error": export["error"],
    }
    print(f"выгрузка: байт={export['bytes']} zip={export_report['is_zip']} rtf={export_report['is_rtf']}")
    print(
        f"::notice title=IPS5 выгрузка::байт={export['bytes']} zip={export_report['is_zip']} rtf={export_report['is_rtf']}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "base": PRAVO,
                "query": QUERY,
                "known_nd": KNOWN_ND,
                "card": card_report,
                "scripts": scripts,
                "listing": listing_report,
                "export": export_report,
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
