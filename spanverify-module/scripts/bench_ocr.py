"""Замер скорости распознавания: сколько секунд стоит страница скана.

Зачем: стоимость корпуса A3 упирается в OCR (580–2680 с на документ по первому
замеру проекта). Любое ускорение (150 dpi вместо 200, ``--psm 6``, «только
страницы с фактами») имеет смысл ровно настолько, насколько оно измерено.
Поэтому замер пишется в ``reports/OCR_BENCH.json``: конфигурация → секунд на
страницу и знаков на страницу, на одном и том же документе.

Запуск (нужны pdftoppm и tesseract с русским языком):

    python scripts/bench_ocr.py --out reports/OCR_BENCH.json

Документ берётся из каталога PDF (``--pdf``), а если его нет — скачивается один
первый подходящий с publication.pravo.gov.ru, чтобы замер был на реальном акте.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Импорт соседнего скрипта: он не пакет, поэтому грузим по пути.
_spec = importlib.util.spec_from_file_location("fetch_npa_corpus", Path(__file__).with_name("fetch_npa_corpus.py"))
if _spec is None or _spec.loader is None:  # pragma: no cover - защита от поломки пути
    print("не удалось подключить fetch_npa_corpus.py", file=sys.stderr)
    raise SystemExit(2)
fetch_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch_mod)


def measure(pdf: Path, dpi: int, psm: int, oem: int, max_pages: int) -> dict:
    """Распознать документ одной конфигурацией и вернуть фактические секунды."""
    started = time.perf_counter()
    result = fetch_mod.ocr_pdf(pdf, dpi=dpi, max_pages=max_pages, psm=psm, oem=oem)
    seconds = round(time.perf_counter() - started, 2)
    pages = int(result.get("pages") or 0)
    chars = len(result.get("text") or "")
    return {
        "dpi": dpi,
        "psm": psm,
        "oem": oem,
        "pages": pages,
        "chars": chars,
        "seconds_total": seconds,
        "seconds_per_page": round(seconds / pages, 2) if pages else None,
        "chars_per_page": round(chars / pages) if pages else None,
        "error": result.get("error"),
    }


def pick_sample_pdf(pause: float) -> Path | None:
    """Скачать один реальный акт для замера (первый подходящий по темам)."""
    robots = fetch_mod.fetch_robots()
    time.sleep(max(0.0, pause))
    types_result = fetch_mod.fetch("http://publication.pravo.gov.ru/api/DocumentTypes")
    type_ids: dict[str, str] = {}
    if types_result["status"] == 200:
        try:
            for item in json.loads(types_result["raw"].decode("utf-8")):
                if isinstance(item, dict) and item.get("name") in fetch_mod.TYPES and item.get("id"):
                    type_ids[str(item["name"])] = str(item["id"])
        except json.JSONDecodeError:
            return None
    candidates, _ = fetch_mod.collect_candidates(type_ids, 2, pause, robots)
    if not candidates:
        return None
    tmp_dir = Path(tempfile.mkdtemp(prefix="bench-ocr-"))
    for candidate in candidates[:5]:
        url = f"http://publication.pravo.gov.ru/file/pdf?eoNumber={candidate['eo_number']}"
        fetched = fetch_mod.fetch(url)
        time.sleep(max(0.0, pause))
        if fetched["status"] == 200 and fetched["raw"][:4] == b"%PDF":
            path = tmp_dir / f"{candidate['doc_id']}.pdf"
            path.write_bytes(fetched["raw"])
            return path
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Замер скорости OCR на реальном акте")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "OCR_BENCH.json")
    parser.add_argument("--pdf", type=Path, default=None, help="готовый PDF; если не задан — скачать один акт")
    parser.add_argument("--pages", type=int, default=3, help="сколько страниц распознавать в замере")
    parser.add_argument("--pause", type=float, default=1.5, help="пауза между запросами, секунд")
    args = parser.parse_args()

    tools = fetch_mod.ocr_available()
    if not all(tools.values()):
        print(f"нет инструментов OCR: {tools}", file=sys.stderr)
        return 2

    pdf = args.pdf
    if pdf is None:
        pdf = pick_sample_pdf(max(1.0, args.pause))
    if pdf is None or not pdf.exists():
        print("не удалось получить PDF для замера", file=sys.stderr)
        return 2

    runs = [
        measure(pdf, dpi=200, psm=6, oem=1, max_pages=args.pages),
        measure(pdf, dpi=150, psm=6, oem=1, max_pages=args.pages),
    ]
    baseline = next((run for run in runs if run["dpi"] == 200 and run["seconds_per_page"]), None)
    for run in runs:
        if baseline and run["seconds_per_page"] and baseline["seconds_per_page"]:
            run["speedup_vs_200dpi"] = round(baseline["seconds_per_page"] / run["seconds_per_page"], 2)
        else:
            run["speedup_vs_200dpi"] = None

    report = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "document": pdf.name,
        "pages_per_run": args.pages,
        "tools": tools,
        "runs": runs,
        "seconds_per_page_median": (
            statistics.median([run["seconds_per_page"] for run in runs if run["seconds_per_page"]])
            if any(run["seconds_per_page"] for run in runs)
            else None
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
