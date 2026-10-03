#!/usr/bin/env python3
"""Шестая разведка: откуда взять ТЕКСТ, если официальные PDF — сканы (шаг 19-2д).

Факт из ``reports/npa_extract_probe.json``: ``/file/pdf?eoNumber=...`` отдаёт PDF,
но текстового слоя в них нет — pypdf извлекает ≈1 символ на страницу (это сканы).

Проверяются три пути к тексту, каждый — фактическими запросами:

1. **SVG.** В карточке документа есть поле ``hasSvg``: если портал делает векторную
   (текстовую) версию, её текст можно прочитать напрямую. Скрипт считает долю
   документов с ``hasSvg=true`` на нескольких страницах выдачи и пробует достать SVG.
2. **ИПС «Законодательство России»** (``pravo.gov.ru``, в белом списке): у неё есть
   HTML-выдача текста акта; проверяются кандидаты адресов.
3. **OCR.** Если текста нет ни там, ни там — остаётся распознавание сканов
   (poppler + tesseract с русским языком). Скрипт распознаёт 2 страницы одного PDF
   и печатает фактическое время и объём текста (и найден ли номер акта).

Запуск::

    python scripts/probe_npa_svg.py --out reports/npa_svg_probe.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
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

BASE = "http://publication.pravo.gov.ru"
PRAVO = "http://pravo.gov.ru"


def fetch(url: str, timeout: int = 90, read_limit: int = 400_000) -> dict:
    """Запрос с сохранением тела и статуса."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json, text/html, image/svg+xml, */*"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - официальный публичный сайт
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
        raw = error.read(read_limit)
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


def svg_stats(raw: bytes) -> dict:
    """Факты о теле SVG: сколько <text>/<tspan> и сколько текста внутри них."""
    text = raw.decode("utf-8", "replace")
    texts = re.findall(r"<text\b[^>]*>(.*?)</text>", text, flags=re.S | re.I)
    stripped = [re.sub(r"<[^>]+>", "", item) for item in texts]
    body = " ".join(part.strip() for part in stripped if part.strip())
    return {
        "is_svg": text.lstrip()[:5].lower() in ("<?xml", "<svg"),
        "text_nodes": len(texts),
        "text_chars": len(body),
        "body_head": body[:300],
    }


def ocr_pdf(raw: bytes, pages: int = 2, dpi: int = 200) -> dict:
    """Распознать первые страницы PDF через pdftoppm + tesseract (русский)."""
    tools = {name: shutil.which(name) for name in ("pdftoppm", "tesseract")}
    if not all(tools.values()):
        return {"available": False, "tools": tools, "chars": None, "seconds": None, "text_head": ""}
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "doc.pdf"
        pdf_path.write_bytes(raw)
        started = time.perf_counter()
        subprocess.run(  # noqa: S603 - фиксированные аргументы, входной файл наш
            ["pdftoppm", "-r", str(dpi), "-png", "-f", "1", "-l", str(pages), str(pdf_path), str(Path(tmp) / "page")],
            check=False,
            capture_output=True,
        )
        pieces: list[str] = []
        for image in sorted(Path(tmp).glob("page*.png")):
            result = subprocess.run(  # noqa: S603 - фиксированные аргументы
                ["tesseract", str(image), "stdout", "-l", "rus"],
                check=False,
                capture_output=True,
            )
            pieces.append(result.stdout.decode("utf-8", "replace"))
        text = "\n".join(pieces)
    seconds = round(time.perf_counter() - started, 2)
    return {
        "available": True,
        "tools": tools,
        "chars": len(text),
        "seconds": seconds,
        "text_head": " ".join(text.split())[:400],
        "pages_recognized": len(pieces),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Разведка текстовых версий (SVG, ИПС, OCR)")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "npa_svg_probe.json")
    parser.add_argument("--pause", type=float, default=PAUSE_SECONDS)
    parser.add_argument("--pages", type=int, default=6, help="сколько страниц выдачи просмотреть на hasSvg")
    args = parser.parse_args()

    # 1. Доля документов с hasSvg и первый такой документ.
    svg_true: list[dict] = []
    counters = {"pages": 0, "documents": 0, "has_svg_true": 0}
    for page in range(1, args.pages + 1):
        listing = fetch(f"{BASE}/api/Documents?pageSize=100&index={page}", read_limit=400_000)
        try:
            items = json.loads(listing["raw"].decode("utf-8")).get("items", [])
        except json.JSONDecodeError:
            items = []
        counters["pages"] += 1
        for item in items:
            counters["documents"] += 1
            if isinstance(item, dict) and item.get("hasSvg"):
                if len(svg_true) < 5:
                    svg_true.append(
                        {
                            k: item.get(k)
                            for k in ("eoNumber", "id", "name", "number", "documentDate", "pagesCount", "hasSvg")
                        }
                    )
                counters["has_svg_true"] += 1
        print(f"страница {page}: документов {len(items)}, с hasSvg {counters['has_svg_true']} всего")
        print(f"::notice title=SVG page {page}::hasSvg всего {counters['has_svg_true']} из {counters['documents']}")
        time.sleep(max(0.0, args.pause))

    svg_probe: dict = {"candidates": []}
    if svg_true:
        eo = svg_true[0]["eoNumber"]
        view = fetch(f"{BASE}/document/{eo}")
        links = re.findall(r'href="([^"]+)"', view["raw"].decode("utf-8", "replace"))
        marker = [href for href in links if re.search(r"(?i)svg|file|pdf", href)]
        svg_probe["view_page"] = {"eoNumber": eo, "status": view["status"], "marker_links": marker[:15]}
        print(f"view {eo}: ссылки {marker[:5]}")
        time.sleep(max(0.0, args.pause))
        for name, url in (
            ("svg-by-eo", f"{BASE}/file/svg?eoNumber={eo}"),
            ("svg-lower-by-eo", f"{BASE}/file/Svg?eoNumber={eo}"),
            ("document-svg-by-eo", f"{BASE}/document/{eo}/svg"),
        ):
            result = fetch(url, read_limit=400_000)
            stats = svg_stats(result["raw"])
            svg_probe["candidates"].append(
                {
                    "name": name,
                    "url": url,
                    "status": result["status"],
                    "content_type": result["content_type"],
                    "bytes": result["bytes"],
                    "error": result["error"],
                    **stats,
                }
            )
            print(
                f"{name}: код={result['status']} тип={result['content_type'][:30]} байт={result['bytes']} текст={stats.get('text_chars')}"
            )
            print(
                f"::notice title=SVG {name}::код={result['status']} байт={result['bytes']} узлов_текста={stats.get('text_nodes')} знаков={stats.get('text_chars')}"
            )
            time.sleep(max(0.0, args.pause))
    else:
        svg_probe["view_page"] = None

    # 2. ИПС «Законодательство России» — кандидаты адресов текстовой выдачи.
    ips: list[dict] = []
    for name, url in (
        ("ips-root", f"{PRAVO}/proxy/ips/"),
        ("ips-docbody", f"{PRAVO}/proxy/ips/?docbody=&nd=102078782"),
        (
            "ips-search",
            f"{PRAVO}/proxy/ips/?searchP=%D0%BF%D0%B5%D1%80%D1%81%D0%BE%D0%BD%D0%B0%D0%BB%D1%8C%D0%BD%D1%8B%D0%B5%20%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D0%B5",
        ),
        ("ips-start", f"{PRAVO}/proxy/ips/?start=0&nd=102078782&rdk=0"),
    ):
        result = fetch(url, read_limit=200_000)
        text = result["raw"].decode("utf-8", "replace")
        if "windows-1251" in result["content_type"].lower():
            text = result["raw"].decode("cp1251", "replace")
        visible = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()
        ips.append(
            {
                "name": name,
                "url": url,
                "status": result["status"],
                "content_type": result["content_type"],
                "bytes": result["bytes"],
                "visible_chars": len(visible),
                "visible_head": visible[:300],
                "error": result["error"],
            }
        )
        print(
            f"ips {name}: код={result['status']} тип={result['content_type'][:30]} байт={result['bytes']} текста={len(visible)}"
        )
        print(f"::notice title=IPS {name}::код={result['status']} байт={result['bytes']} знаков={len(visible)}")
        time.sleep(max(0.0, args.pause))

    # 3. OCR: берём PDF первого документа из выдачи (или из выборки) и распознаём 2 страницы.
    ocr: dict = {"available": False}
    listing = fetch(f"{BASE}/api/Documents?pageSize=10&index=1", read_limit=200_000)
    try:
        items = json.loads(listing["raw"].decode("utf-8")).get("items", [])
    except json.JSONDecodeError:
        items = []
    if items:
        eo = items[0].get("eoNumber")
        pdf = fetch(f"{BASE}/file/pdf?eoNumber={eo}", read_limit=8_000_000)
        if pdf["status"] == 200 and pdf["raw"][:4] == b"%PDF":
            ocr = ocr_pdf(pdf["raw"], pages=2)
            number = str(items[0].get("number") or "")
            ocr["eoNumber"] = eo
            ocr["act_number"] = number
            ocr["act_number_found"] = bool(number) and number in (ocr.get("text_head") or "")
            print(
                f"ocr {eo}: доступен={ocr.get('available')} знаков={ocr.get('chars')} секунд={ocr.get('seconds')} "
                f"номер_найден={ocr.get('act_number_found')}"
            )
            print(
                f"::notice title=OCR::доступен={ocr.get('available')} знаков={ocr.get('chars')} секунд={ocr.get('seconds')} номер_найден={ocr.get('act_number_found')}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "base": BASE,
                "counters": counters,
                "svg_true_examples": svg_true,
                "svg_probe": svg_probe,
                "ips": ips,
                "ocr": ocr,
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
