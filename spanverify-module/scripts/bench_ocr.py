#!/usr/bin/env python3
"""Замер скорости распознавания официальных PDF: факты вместо предположений.

Зачем
-----

Первый прогон сборки корпуса A3 показал, что распознавание одного акта занимает
580–2680 секунд (поле ``ocr_seconds`` в ``data/corpus_a3/sources/sources.json``). Это
на два порядка больше ожидаемого, и без замера непонятно, что именно дорого: рендер
страниц (``pdftoppm``), движок распознавания или конкуренция потоков.

Скрипт берёт настоящий PDF официальной публикации, прогоняет один и тот же файл
несколькими наборами настроек и печатает время каждого этапа. Ничего не
интерпретирует: в отчёт попадают секунды, число знаков и команда.

Настройки, которые сравниваются:

* ``oem`` — движок: 3 (по умолчанию: старый + LSTM) против 1 (только LSTM);
* ``psm`` — разбор страницы: 3 (по умолчанию) против 6 («единый блок текста»);
* ``dpi`` рендера: 200 против 150;
* ``OMP_THREAD_LIMIT`` — число потоков OpenMP внутри tesseract.

Запуск::

    python scripts/bench_ocr.py --out reports/OCR_BENCH.json --pages 2
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

_MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

from scripts.fetch_npa_corpus import USER_AGENT, fetch, ocr_available  # noqa: E402

ROOT = _MODULE_ROOT

# Реальный акт официальной публикации: Приказ Рособрнадзора от 31.08.2023 № 1587.
# Выбран потому, что уже встречался в разведке и заведомо является сканом.
DEFAULT_EO = "0001202310130023"

VARIANTS: tuple[dict, ...] = (
    {
        "name": "по умолчанию (oem 3, psm 3, 200 dpi, потоки не ограничены)",
        "oem": "3",
        "psm": "3",
        "dpi": 200,
        "omp": None,
    },
    {"name": "oem 1 (только LSTM), psm 3, 200 dpi", "oem": "1", "psm": "3", "dpi": 200, "omp": None},
    {"name": "oem 1, psm 6 (единый блок), 200 dpi", "oem": "1", "psm": "6", "dpi": 200, "omp": None},
    {"name": "oem 1, psm 6, 150 dpi", "oem": "1", "psm": "6", "dpi": 150, "omp": None},
    {"name": "oem 1, psm 6, 150 dpi, OMP_THREAD_LIMIT=1", "oem": "1", "psm": "6", "dpi": 150, "omp": "1"},
)


def render(pdf: Path, out_prefix: Path, dpi: int, pages: int) -> dict:
    """Отрисовать первые ``pages`` страниц в PNG; вернуть время и список файлов."""
    started = time.perf_counter()
    result = subprocess.run(  # noqa: S603 - фиксированные аргументы
        ["pdftoppm", "-r", str(dpi), "-png", "-f", "1", "-l", str(pages), str(pdf), str(out_prefix)],
        check=False,
        capture_output=True,
    )
    images = sorted(out_prefix.parent.glob(f"{out_prefix.name}*.png"))
    return {
        "seconds": round(time.perf_counter() - started, 2),
        "images": images,
        "error": result.stderr.decode("utf-8", "replace")[:200] if result.returncode != 0 else None,
    }


def recognise(images: list[Path], oem: str, psm: str, omp: str | None) -> dict:
    """Распознать изображения; вернуть время и число знаков (без оценок качества)."""
    environment = dict(os.environ)
    if omp is not None:
        environment["OMP_THREAD_LIMIT"] = omp
    started = time.perf_counter()
    chars = 0
    error = None
    for image in images:
        result = subprocess.run(  # noqa: S603 - фиксированные аргументы
            ["tesseract", str(image), "stdout", "-l", "rus", "--oem", oem, "--psm", psm],
            check=False,
            capture_output=True,
            env=environment,
        )
        if result.returncode != 0 and error is None:
            error = result.stderr.decode("utf-8", "replace")[:200]
        chars += len(result.stdout.decode("utf-8", "replace"))
    return {"seconds": round(time.perf_counter() - started, 2), "chars": chars, "error": error}


def download_pdf(eo_number: str, destination: Path) -> dict:
    """Скачать официальный PDF по номеру опубликования."""
    url = f"http://publication.pravo.gov.ru/file/pdf?eoNumber={eo_number}"
    started = time.perf_counter()
    fetched = fetch(url)
    seconds = round(time.perf_counter() - started, 2)
    if fetched["status"] != 200 or fetched["raw"][:4] != b"%PDF":
        return {"ok": False, "url": url, "seconds": seconds, "error": fetched["error"] or fetched["status"]}
    destination.write_bytes(fetched["raw"])
    return {"ok": True, "url": url, "seconds": seconds, "bytes": len(fetched["raw"]), "error": None}


def tool_versions() -> dict:
    """Фактические версии инструментов (для протокола измерений)."""
    versions: dict[str, str | None] = {}
    for name, command in (("tesseract", ["tesseract", "--version"]), ("pdftoppm", ["pdftoppm", "-v"])):
        try:
            result = subprocess.run(command, check=False, capture_output=True)  # noqa: S603
            output = (result.stdout or result.stderr).decode("utf-8", "replace")
            versions[name] = output.splitlines()[0].strip() if output.strip() else None
        except FileNotFoundError:
            versions[name] = None
    return versions


def main() -> int:
    parser = argparse.ArgumentParser(description="Замер скорости OCR официальных PDF")
    parser.add_argument("--eo", default=DEFAULT_EO, help="номер опубликования акта для замера")
    parser.add_argument("--pages", type=int, default=2, help="сколько первых страниц распознавать в каждом варианте")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "OCR_BENCH.json")
    args = parser.parse_args()

    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "eo_number": args.eo,
        "pages_per_variant": args.pages,
        "user_agent": USER_AGENT,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor_count": os.cpu_count(),
            "tools": tool_versions(),
        },
        "ocr_tools_found": ocr_available(),
        "variants": [],
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        pdf = tmp_dir / "act.pdf"
        download = download_pdf(args.eo, pdf)
        report["download"] = {key: value for key, value in download.items() if key != "raw"}
        if not download["ok"]:
            report["conclusion"] = "PDF не скачан — замер не выполнен"
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 1

        for index, variant in enumerate(VARIANTS):
            prefix = tmp_dir / f"v{index}"
            rendered = render(pdf, prefix, int(variant["dpi"]), args.pages)
            if rendered["error"] or not rendered["images"]:
                report["variants"].append(
                    {**variant, "render_seconds": rendered["seconds"], "error": rendered["error"]}
                )
                continue
            recognised = recognise(rendered["images"], str(variant["oem"]), str(variant["psm"]), variant["omp"])
            pages_done = len(rendered["images"])
            report["variants"].append(
                {
                    **variant,
                    "pages": pages_done,
                    "render_seconds": rendered["seconds"],
                    "ocr_seconds": recognised["seconds"],
                    "total_seconds": round(rendered["seconds"] + recognised["seconds"], 2),
                    "seconds_per_page": round((rendered["seconds"] + recognised["seconds"]) / max(1, pages_done), 2),
                    "chars": recognised["chars"],
                    "error": recognised["error"],
                }
            )
            print(
                f"{variant['name']}: рендер {rendered['seconds']} с, распознавание {recognised['seconds']} с, "
                f"знаков {recognised['chars']}"
            )

    measured = [item for item in report["variants"] if item.get("seconds_per_page")]
    if measured:
        best = min(measured, key=lambda item: item["seconds_per_page"])
        worst = max(measured, key=lambda item: item["seconds_per_page"])
        report["fastest"] = best["name"]
        report["slowest"] = worst["name"]
        report["speedup"] = round(worst["seconds_per_page"] / max(0.01, best["seconds_per_page"]), 2)
        report["conclusion"] = (
            f"быстрее всех «{best['name']}»: {best['seconds_per_page']} с на страницу против "
            f"{worst['seconds_per_page']} с у «{worst['name']}» (ускорение ×{report['speedup']})"
        )
    else:
        report["conclusion"] = "ни один вариант не отработал — см. поля error"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(report["conclusion"])
    print(f"::notice title=замер OCR::{report['conclusion']}")
    print(f"Отчёт: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
