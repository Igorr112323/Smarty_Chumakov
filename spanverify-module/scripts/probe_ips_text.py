#!/usr/bin/env python3
"""Восьмая разведка: фрейм с текстом акта и поиск в ИПС (шаг 19-2ж).

Факты из ``reports/ips_probe.json``:

* ``?docbody=&nd=<N>`` отдаёт **карточку** документа (название и метаданные),
  признаков текста там нет (862 знака видимого текста, «Текст документа: Исходная
  редакция»);
* на корневой странице ИПС есть форма поиска (``method=get``, ``action="?"``) с
  полями ``searchres``, ``a0``, ``a1``, ``a3``, ``a6``, ``a7…``, ``a15``, ``a16``,
  ``a17`` — то есть запрос задаётся полем ``a0`` и признаком ``searchres``;
* в карточке нет ссылок ``href=`` с ``nd=``, значит текст открывается через frame
  (``src=``) или через другой адрес.

Скрипт: (1) сохраняет все адреса (``href``/``src``/``action``) карточки, чтобы увидеть
фактическую структуру кадра; (2) пробует адреса кадра с текстом; (3) пробует поиск
через ``a0``/``searchres`` и сообщает фактический объём выдачи.

Запуск::

    python scripts/probe_ips_text.py --out reports/ips_text_probe.json
"""

from __future__ import annotations

import argparse
import json
import re
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
PRAVO = "http://pravo.gov.ru"
ND = "102078782"
QUERY = "персональные данные"


def fetch(url: str, timeout: int = 90, read_limit: int = 800_000) -> dict:
    """Запрос к cp1251-сайту ИПС с декодированием."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html, application/json, */*"},
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
        return {"url": url, "status": None, "content_type": "", "bytes": 0, "html": "", "error": str(error)[:140]}
    html_text = raw.decode("cp1251", "replace")
    return {
        "url": url,
        "status": status,
        "content_type": content_type,
        "bytes": len(raw),
        "html": html_text,
        "error": None if status < 400 else f"HTTP {status}",
    }


def visible_text(html_text: str) -> str:
    """Видимый текст страницы."""
    body = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html_text)
    body = re.sub(r"(?s)<[^>]+>", " ", body)
    return re.sub(r"\s+", " ", body).strip()


def addresses(html_text: str, limit: int = 40) -> list[str]:
    """Все адреса страницы из href/src/action (в порядке появления, без повторов)."""
    found = re.findall(r'(?:href|src|action)\s*=\s*["\']([^"\']+)["\']', html_text, flags=re.I)
    seen: list[str] = []
    for item in found:
        if item not in seen:
            seen.append(item)
    return seen[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description="Разведка текста и поиска в ИПС")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "ips_text_probe.json")
    parser.add_argument("--pause", type=float, default=PAUSE_SECONDS)
    args = parser.parse_args()

    card = fetch(f"{PRAVO}/proxy/ips/?docbody=&nd={ND}")
    card_addresses = addresses(card["html"])
    card_report = {
        "url": card["url"],
        "status": card["status"],
        "bytes": card["bytes"],
        "addresses": card_addresses,
        "framesets": re.findall(r"(?is)<frameset[^>]*>", card["html"])[:5],
        "frames": re.findall(r'(?is)<frame[^>]*src\s*=\s*["\']([^"\']+)["\']', card["html"])[:10],
        "rdk_context": [
            card["html"][max(0, match.start() - 80) : match.end() + 80].replace("\n", " ")
            for match in list(re.finditer(r"rdk", card["html"], flags=re.I))[:3]
        ],
        "text_chars": len(visible_text(card["html"])),
    }
    print(f"карточка: код={card['status']} байт={card['bytes']} адресов={len(card_addresses)}")
    print(f"::notice title=IPS2 карточка::адресов={len(card_addresses)} кадров={len(card_report['frames'])}")
    for item in card_addresses[:20]:
        print(f"   адрес: {item}")
    time.sleep(max(0.0, args.pause))

    frame_candidates: list[dict] = []
    for name, url in (
        ("rdk-0", f"{PRAVO}/proxy/ips/?docbody=&nd={ND}&rdk=0"),
        ("rdk-1", f"{PRAVO}/proxy/ips/?docbody=&nd={ND}&rdk=1"),
        ("rdk-0-first", f"{PRAVO}/proxy/ips/?docbody=&nd={ND}&rdk=0&first=1"),
        ("text", f"{PRAVO}/proxy/ips/?docbody=&nd={ND}&text=1"),
        ("intelsearch", f"{PRAVO}/proxy/ips/?docbody=&nd={ND}&intelsearch=1"),
        ("nd-only-start", f"{PRAVO}/proxy/ips/?start=0&nd={ND}&rdk=0&intelsearch=1"),
    ):
        result = fetch(url)
        text = visible_text(result["html"])
        frame_candidates.append(
            {
                "name": name,
                "url": url,
                "status": result["status"],
                "bytes": result["bytes"],
                "text_chars": len(text),
                "text_head": text[:300],
                "addresses": addresses(result["html"], limit=10),
            }
        )
        summary = f"код={result['status']} байт={result['bytes']} знаков={len(text)}"
        print(f"кадр {name:14s} {summary}")
        print(f"::notice title=IPS2 кадр {name}::{summary}")
        time.sleep(max(0.0, args.pause))

    query = urllib.parse.quote(QUERY.encode("cp1251"))
    searches: list[dict] = []
    for name, url in (
        ("searchres-a0", f"{PRAVO}/proxy/ips/?searchres=1&a0={query}"),
        ("a0-only", f"{PRAVO}/proxy/ips/?a0={query}"),
        ("searchres-a0-a1", f"{PRAVO}/proxy/ips/?searchres=1&a0={query}&a1=1"),
        ("searchres-a0-a16", f"{PRAVO}/proxy/ips/?searchres=1&a0={query}&a16=1&a16type=1&a16value={query}"),
    ):
        result = fetch(url)
        text = visible_text(result["html"])
        nd_found = re.findall(r"nd=\d+", result["html"])
        searches.append(
            {
                "name": name,
                "url": url,
                "status": result["status"],
                "bytes": result["bytes"],
                "text_chars": len(text),
                "nd_occurrences": len(nd_found),
                "nd_examples": sorted(set(nd_found))[:10],
                "query_words_in_text": "персональн" in text.lower(),
                "text_head": text[:400],
                "addresses": addresses(result["html"], limit=15),
            }
        )
        summary = (
            f"код={result['status']} байт={result['bytes']} знаков={len(text)} "
            f"nd={len(nd_found)} слова={searches[-1]['query_words_in_text']}"
        )
        print(f"поиск {name:20s} {summary}")
        print(f"::notice title=IPS2 поиск {name}::{summary}")
        time.sleep(max(0.0, args.pause))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "base": PRAVO,
                "nd": ND,
                "query": QUERY,
                "card": card_report,
                "frames": frame_candidates,
                "searches": searches,
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
