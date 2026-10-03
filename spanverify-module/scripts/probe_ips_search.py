#!/usr/bin/env python3
"""Седьмая разведка: поиск и выдача текста в ИПС «Законодательство России» (шаг 19-2е).

Факты из ``reports/npa_svg_probe.json``:

* у документов официальной публикации ``hasSvg=false`` на всех просмотренных 600
  карточках, а PDF — сканы без текстового слоя; SVG как источник текста не годится;
* ``http://pravo.gov.ru/proxy/ips/?docbody=&nd=102078782`` отвечает 200 HTML и
  содержит **текст акта** (≈9 тыс. знаков); ``pravo.gov.ru`` — в белом списке,
  его ``robots.txt`` разрешает всё;
* OCR сканов работает (tesseract -l rus: 2 страницы → 2468 знаков за 3.15 с,
  номер акта найден), но это запасной путь: распознанный текст может содержать
  ошибки, а в ИПС текст машинный.

Этот скрипт выясняет, как в ИПС **искать по теме**: разбирает формы страницы
(``action``, ``method``, имена полей), считает ссылки вида ``nd=`` и пробует
кандидаты поисковых адресов, показывая фактический объём выдачи. Кодировка ИПС —
cp1251, поэтому запросы кодируются в cp1251, а ответы декодируются из cp1251.

Запуск::

    python scripts/probe_ips_search.py --out reports/ips_probe.json
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
QUERY = "персональные данные"


def fetch(url: str, timeout: int = 90, read_limit: int = 600_000) -> dict:
    """Запрос с декодированием cp1251 (кодировка сайта ИПС)."""
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

    charset = "cp1251" if "1251" in content_type.lower() else "cp1251"
    html_text = raw.decode(charset, "replace")
    if (
        "<meta" in html_text[:2000]
        and "utf-8" in html_text[:2000].lower()
        and "windows-1251" not in html_text[:2000].lower()
    ):
        html_text = raw.decode("utf-8", "replace")
    return {
        "url": url,
        "status": status,
        "content_type": content_type,
        "bytes": len(raw),
        "html": html_text,
        "error": None if status < 400 else f"HTTP {status}",
    }


def visible_text(html_text: str) -> str:
    """Видимый текст страницы (теги, скрипты и стили выброшены)."""
    body = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html_text)
    body = re.sub(r"(?s)<[^>]+>", " ", body)
    return re.sub(r"\s+", " ", body).strip()


def nd_links(html_text: str) -> list[str]:
    """Ссылки на документы ИПС (в них есть ``nd=``)."""
    hrefs = re.findall(r'href="([^"]*nd=[^"]*)"', html_text, flags=re.I)
    seen: list[str] = []
    for href in hrefs:
        if href not in seen:
            seen.append(href)
    return seen


def forms(html_text: str) -> list[dict]:
    """Разобранные формы страницы: action, method и имена полей."""
    result: list[dict] = []
    for match in re.finditer(r"(?is)<form\b([^>]*)>(.*?)</form>", html_text):
        attrs, inner = match.group(1), match.group(2)
        action = re.search(r'action="([^"]*)"', attrs, flags=re.I)
        method = re.search(r'method="([^"]*)"', attrs, flags=re.I)
        names = re.findall(r'<(?:input|select|textarea)\b[^>]*name="([^"]+)"', inner, flags=re.I)
        result.append(
            {
                "action": action.group(1) if action else "",
                "method": (method.group(1) if method else "").lower(),
                "fields": names[:25],
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Разведка поиска в ИПС «Законодательство России»")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "ips_probe.json")
    parser.add_argument("--pause", type=float, default=PAUSE_SECONDS)
    args = parser.parse_args()

    # 1. Известный документ: подтверждаем, что текст акта читается, и сохраняем начало.
    known = fetch(f"{PRAVO}/proxy/ips/?docbody=&nd=102078782")
    known_text = visible_text(known["html"])
    known_report = {
        "url": known["url"],
        "status": known["status"],
        "bytes": known["bytes"],
        "text_chars": len(known_text),
        "text_head": known_text[:600],
        "looks_like_act": any(word in known_text for word in ("ПОСТАНОВЛЕНИЕ", "ФЕДЕРАЛЬНЫЙ ЗАКОН", "Статья")),
        "error": known["error"],
    }
    print(
        f"ИПС документ: код={known['status']} байт={known['bytes']} знаков={len(known_text)} похож_на_акт={known_report['looks_like_act']}"
    )
    print(
        f"::notice title=IPS doc::код={known['status']} знаков={len(known_text)} похож_на_акт={known_report['looks_like_act']}"
    )
    time.sleep(max(0.0, args.pause))

    # 2. Корневая страница: формы поиска и примеры ссылок на документы.
    root = fetch(f"{PRAVO}/proxy/ips/")
    root_links = nd_links(root["html"])
    root_report = {
        "url": root["url"],
        "status": root["status"],
        "bytes": root["bytes"],
        "forms": forms(root["html"]),
        "nd_links_count": len(root_links),
        "nd_links_examples": root_links[:8],
        "text_head": visible_text(root["html"])[:300],
    }
    print(f"ИПС корень: код={root['status']} форм={len(root_report['forms'])} ссылок_nd={len(root_links)}")
    print(f"::notice title=IPS root::форм={len(root_report['forms'])} ссылок_nd={len(root_links)}")
    for form in root_report["forms"][:4]:
        print(f"   форма action={form['action']!r} method={form['method']!r} поля={form['fields'][:12]}")
        print(f"::notice title=IPS форма::{form['action']} поля={form['fields'][:10]}")
    time.sleep(max(0.0, args.pause))

    # 3. Кандидаты поисковых адресов: где ИПС отдаёт выдачу по теме.
    query_cp1251 = urllib.parse.quote(QUERY.encode("cp1251"))
    candidates = [
        ("searchP", f"{PRAVO}/proxy/ips/?searchP={query_cp1251}"),
        ("searchP-searchF", f"{PRAVO}/proxy/ips/?searchP={query_cp1251}&searchF=0"),
        ("text", f"{PRAVO}/proxy/ips/?text={query_cp1251}"),
        ("intelsearch", f"{PRAVO}/proxy/ips/?intelsearch={query_cp1251}"),
        ("query", f"{PRAVO}/proxy/ips/?query={query_cp1251}"),
        ("swd", f"{PRAVO}/proxy/ips/?swd={query_cp1251}"),
        ("start-nd", f"{PRAVO}/proxy/ips/?start=0&nd=102078782"),
    ]
    searches: list[dict] = []
    for name, url in candidates:
        result = fetch(url)
        text = visible_text(result["html"])
        links = nd_links(result["html"])
        has_query_words = "персональн" in text.lower()
        searches.append(
            {
                "name": name,
                "url": url,
                "status": result["status"],
                "bytes": result["bytes"],
                "text_chars": len(text),
                "nd_links_count": len(links),
                "nd_links_examples": links[:5],
                "query_words_in_text": has_query_words,
                "text_head": text[:300],
                "error": result["error"],
            }
        )
        summary = (
            f"код={result['status']} байт={result['bytes']} знаков={len(text)} ссылок_nd={len(links)} "
            f"слова_запроса={has_query_words}"
        )
        print(f"поиск {name:16s} {summary}")
        print(f"::notice title=IPS поиск {name}::{summary}")
        time.sleep(max(0.0, args.pause))

    # 4. Насколько часто произвольный nd даёт текст акта (оценка «плотности» ИПС).
    density: list[dict] = []
    for nd in (1, 1000, 25000, 50000, 75000, 120000):
        result = fetch(f"{PRAVO}/proxy/ips/?docbody=&nd={nd}")
        text = visible_text(result["html"])
        density.append(
            {
                "nd": nd,
                "status": result["status"],
                "text_chars": len(text),
                "looks_like_document": len(text) > 2000,
                "text_head": text[:150],
            }
        )
        print(f"nd={nd}: код={result['status']} знаков={len(text)} похож_на_документ={len(text) > 2000}")
        print(f"::notice title=IPS nd={nd}::код={result['status']} знаков={len(text)}")
        time.sleep(max(0.0, args.pause))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "base": PRAVO,
                "query": QUERY,
                "known_document": known_report,
                "root": root_report,
                "searches": searches,
                "density": density,
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
