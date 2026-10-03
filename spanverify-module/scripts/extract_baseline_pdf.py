#!/usr/bin/env python3
"""Извлечение чисел baseline из PDF статьи RusHallu-RAG (шаг 1.5 задания).

Задание: закрыть ``baseline_comparison: null`` — скачать PDF статьи
(``dialogue-conf.org/…/SadkovskiiFNasyrovaRSorokinA.087.pdf``), извлечь таблицы 2–4 и
вписать значения с пометкой «данные из статьи, таблица N». Если извлечь не удаётся —
оставить ``null`` и «не извлечено», **не выдумывать числа**.

Скрипт: скачивает PDF, извлекает текст (pypdf), ищет таблицы по подписям
(«Таблица 2», «Table 2») и разбирает строки с метриками (precision/recall/F1/…) в числа.
Что получилось — записывается в ``reports/baseline_from_article.json`` вместе с
фрагментами текста, по которым числа найдены (чтобы проверяющий мог сверить).

Запуск::

    python scripts/extract_baseline_pdf.py --out reports/baseline_from_article.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

ROOT = MODULE_ROOT

PDF_URL = "https://dialogue-conf.org/wp-content/uploads/2026/06/SadkovskiiFNasyrovaRSorokinA.087.pdf"
USER_AGENT = (
    "SpanVerify-CorpusBot/1.0 (+https://github.com/Igorr112323/Smarty_Chumakov; "
    "research corpus; contact: repository issues)"
)

# Подписи таблиц, которые нужны заданию.
TABLE_TITLES = {
    "2": ("Таблица 2", "Table 2"),
    "3": ("Таблица 3", "Table 3"),
    "4": ("Таблица 4", "Table 4"),
}
# Метрики, которые ищутся в строках таблиц (в нижнем регистре).
METRIC_PATTERNS = (
    ("precision", r"precision|точность"),
    ("recall", r"recall|полнота"),
    ("f1", r"f1|f-measure|f-мера"),
    ("accuracy", r"accuracy|точность ответа"),
    ("auc", r"auc|roc"),
    ("fpr", r"fpr|ложн"),
)
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def fetch_pdf(url: str, timeout: int = 120) -> dict:
    """Скачать PDF: вернуть байты и факт статуса."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/pdf, */*"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - адрес из задания
            raw = response.read(12_000_000)
            return {
                "url": url,
                "status": int(getattr(response, "status", 0) or 0),
                "content_type": response.headers.get("Content-Type", ""),
                "bytes": len(raw),
                "raw": raw,
                "error": None,
            }
    except urllib.error.HTTPError as error:
        return {
            "url": url,
            "status": int(error.code),
            "content_type": "",
            "bytes": 0,
            "raw": b"",
            "error": f"HTTP {error.code}",
        }
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return {"url": url, "status": None, "content_type": "", "bytes": 0, "raw": b"", "error": str(error)[:140]}


def pdf_pages(raw: bytes) -> list[str]:
    """Тексты страниц PDF через pypdf (пустой список, если pypdf нет или файл не читается)."""
    from io import BytesIO

    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover - в CI pypdf установлен
        return []
    try:
        reader = PdfReader(BytesIO(raw))
        return [page.extract_text() or "" for page in reader.pages]
    except Exception:  # noqa: BLE001 - некорректный PDF не должен ронять прогон
        return []


def find_tables(pages: list[str]) -> dict:
    """Найти таблицы 2–4 по подписям и вернуть фрагменты текста вокруг них."""
    tables: dict[str, dict] = {}
    for number, titles in TABLE_TITLES.items():
        for page_index, text in enumerate(pages, start=1):
            for title in titles:
                position = text.find(title)
                if position < 0:
                    continue
                fragment = text[position : position + 2000]
                tables[number] = {
                    "page": page_index,
                    "title": title,
                    "fragment": fragment,
                    "title_found": True,
                }
                break
            if number in tables:
                break
    return tables


def parse_rows(fragment: str) -> list[dict]:
    """Разобрать строки таблицы: метрика → числа в строке.

    Числа берутся как есть из текста; ничего не пересчитывается и не округляется.
    """
    rows: list[dict] = []
    for line in fragment.splitlines():
        stripped = line.strip()
        if len(stripped) < 4:
            continue
        lowered = stripped.lower()
        for metric, pattern in METRIC_PATTERNS:
            if re.search(pattern, lowered):
                numbers = [item.replace(",", ".") for item in NUMBER.findall(stripped)]
                rows.append({"metric": metric, "line": stripped[:200], "numbers": numbers})
                break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Извлечение чисел baseline из PDF статьи")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "baseline_from_article.json")
    parser.add_argument("--url", default=PDF_URL)
    args = parser.parse_args()

    fetched = fetch_pdf(args.url)
    payload: dict = {
        "url": args.url,
        "status": fetched["status"],
        "content_type": fetched["content_type"],
        "bytes": fetched["bytes"],
        "error": fetched["error"],
        "extracted": False,
        "tables": {},
        "rows": {},
        "note": "",
    }
    if fetched["status"] == 200 and fetched["raw"][:4] == b"%PDF":
        pages = pdf_pages(fetched["raw"])
        payload["pages"] = len(pages)
        tables = find_tables(pages)
        payload["tables"] = tables
        for number, table in tables.items():
            rows = parse_rows(table["fragment"])
            payload["rows"][number] = rows
        payload["extracted"] = any(rows for rows in payload["rows"].values())
        payload["note"] = (
            "числа взяты из текста статьи (таблицы 2-4); ничего не пересчитано"
            if payload["extracted"]
            else "текст таблиц 2-4 не извлечён из PDF (числа не выдумывались)"
        )
    else:
        payload["note"] = f"PDF не получен: {fetched['error'] or fetched['status']}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"baseline из статьи: извлечено={payload['extracted']} таблиц={sorted(payload['tables'])}")
    print(f"::notice title=baseline::{payload['note']} (таблиц найдено {len(payload['tables'])})")
    print(f"Отчёт: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
