#!/usr/bin/env python3
"""Урожайность текстового слоя источника: доля документов с годным текстом БЕЗ распознавания.

Зачем
-----

Режим загрузки (только встроенный текстовый слой или распознавание каждого скана) меняет
урожайность волны на порядок, поэтому число должно быть измерено, а не взято из памяти.
Скрипт считает долю по манифесту сбора (`sources.json`) и печатает объём выборки и 95 %
доверительный интервал — этого достаточно, чтобы решение о режиме принимал заказчик, а не
догадка исполнителя.

Определения (печатаются оба, потому что вопросы разные)
-------------------------------------------------------

* **доля по методу** — у документа текст взят из слоя PDF (поле `extraction` не содержит `ocr`):
  ровно столько даст режим «только текстовый слой».
* **доля по объёму слоя** — `text_layer_chars / pages_count >= --min-chars-per-page`: слой есть,
  и в нём достаточное число знаков на страницу. Бывает, что слой формально присутствует, но это
  десятки знаков мусора — такой документ в выборку не попадёт и в текстовом режиме.

Разбиение. Считается по `source_host` отдельно: локальные донорные тексты (не из НПА) в
знаменатель источника не входят, иначе доля завышается. Таблица по видам актов показывает
смещение: если годовые слои встречаются только в одном виде актов, текстовый режим даст
выборку, смещённую по типу документа.

Воспроизведение (корпус v1, 409 записей, из них 272 из НПА)
-----------------------------------------------------------

    python3 scripts/text_layer_yield.py --sources data/corpus_a3/sources/sources.json --all
    python3 scripts/text_layer_yield.py --sources data/corpus_a3/sources/sources.json \\
        --sample 100 --seed 20261011 --json reports/v2/text_layer_yield.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

Z_95 = 1.959963984540054  # двусторонний 95 % квантиль нормального распределения


def wilson_interval(successes: int, total: int, z: float = Z_95) -> tuple[float, float]:
    """Точный биномиальный интервал Уилсона для доли; (0.0, 0.0) при пустой выборке."""
    if total <= 0:
        return (0.0, 0.0)
    phat = successes / total
    denom = 1.0 + z * z / total
    centre = (phat + z * z / (2 * total)) / denom
    half = z * math.sqrt(phat * (1.0 - phat) / total + z * z / (4.0 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def method_of(rec: dict[str, Any]) -> str:
    """Метод извлечения: в манифестах поле `extraction` бывает строкой или словарём."""
    ext = rec.get("extraction")
    if isinstance(ext, dict):
        return str(ext.get("method") or "")
    return str(ext or "")


def layer_chars_of(rec: dict[str, Any]) -> int | None:
    """Знаки встроенного текстового слоя (None, если поле не записано)."""
    ext = rec.get("extraction")
    if isinstance(ext, dict) and isinstance(ext.get("text_layer_chars"), int):
        return int(ext["text_layer_chars"])
    value = rec.get("text_layer_chars")
    return int(value) if isinstance(value, int) else None


def usable(rec: dict[str, Any], min_chars_per_page: int) -> tuple[bool, bool]:
    """(годный метод, годный слой по знакам); записи без метода не оцениваются вовсе."""
    method = method_of(rec)
    if not method:
        return (False, False)
    ok_method = "ocr" not in method.lower()
    chars = layer_chars_of(rec)
    pages = rec.get("pages_count")
    ok_layer = bool(
        isinstance(chars, int) and isinstance(pages, int) and pages > 0 and chars / pages >= min_chars_per_page
    )
    return (ok_method, ok_layer)


def measure(docs: list[dict[str, Any]], min_chars_per_page: int) -> dict[str, Any]:
    """Доли по обоим определениям и разбивка по видам актов (знаменатель = записи с методом)."""
    scored = [rec for rec in docs if method_of(rec)]
    method_hits = 0
    layer_hits = 0
    by_type: dict[str, list[int]] = {}
    for rec in scored:
        ok_method, ok_layer = usable(rec, min_chars_per_page)
        method_hits += int(ok_method)
        layer_hits += int(ok_layer)
        bucket = by_type.setdefault(str(rec.get("act_type") or "без вида акта"), [0, 0, 0])
        bucket[0] += 1
        bucket[1] += int(ok_method)
        bucket[2] += int(ok_layer)
    total = len(scored)
    lo_m, hi_m = wilson_interval(method_hits, total)
    lo_l, hi_l = wilson_interval(layer_hits, total)
    types: dict[str, Any] = {}
    for act_type in sorted(by_type):
        n, hits_m, hits_l = by_type[act_type]
        lo, hi = wilson_interval(hits_m, n)
        types[act_type] = {
            "n": n,
            "docs_with_text_layer": hits_m,
            "share_by_method": round(hits_m / n, 6) if n else None,
            "ci95_by_method": [round(lo, 6), round(hi, 6)],
            "docs_with_usable_layer_chars": hits_l,
        }
    return {
        "n_records_in_manifest": len(docs),
        "n_scored": total,
        "n_without_method": len(docs) - total,
        "docs_with_text_layer": method_hits,
        "share_by_method": round(method_hits / total, 6) if total else None,
        "ci95_by_method": [round(lo_m, 6), round(hi_m, 6)],
        "docs_with_usable_layer_chars": layer_hits,
        "share_by_layer_chars": round(layer_hits / total, 6) if total else None,
        "ci95_by_layer_chars": [round(lo_l, 6), round(hi_l, 6)],
        "by_act_type": types,
    }


def load_documents(path: Path) -> list[dict[str, Any]]:
    """Записи документов из манифеста сбора: `documents` — словарь (ключ = doc_id) или список."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    docs = payload.get("documents")
    if isinstance(docs, dict):
        return [docs[key] for key in sorted(docs)]
    if isinstance(docs, list):
        return list(docs)
    raise SystemExit(f"в {path} нет раздела documents (словаря или списка) — считать нечего")


def format_share(label: str, part: dict[str, Any] | None) -> str:
    """Одна строка отчёта: n, числители, доли и ДИ по обоим определениям."""
    if not part or part["n_scored"] == 0:
        return f"{label}: документов с записанным методом извлечения нет — доля не измерено"
    lo_m, hi_m = part["ci95_by_method"]
    lo_l, hi_l = part["ci95_by_layer_chars"]
    return (
        f"{label}: n={part['n_scored']}, слой из PDF без распознавания {part['docs_with_text_layer']} "
        f"(доля {part['share_by_method']:.4f}, 95 % ДИ [{lo_m:.4f}; {hi_m:.4f}]), "
        f"слой не тоньше порога {part['docs_with_usable_layer_chars']} "
        f"(доля {part['share_by_layer_chars']:.4f}, ДИ [{lo_l:.4f}; {hi_l:.4f}]), "
        f"без метода извлечения {part['n_without_method']}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sources", required=True, help="путь к манифесту сбора sources.json")
    ap.add_argument("--host", default="", help="учесть только эти source_host (через запятую)")
    ap.add_argument("--sample", type=int, default=0, help="размер случайной выборки (0 = вся совокупность)")
    ap.add_argument("--seed", type=int, default=20261011, help="зерно случайной выборки")
    ap.add_argument(
        "--min-chars-per-page",
        type=int,
        default=500,
        help="порог годности слоя: минимальное число знаков на страницу",
    )
    ap.add_argument("--all", action="store_true", help="вся совокупность (то же, что --sample 0)")
    ap.add_argument("--json", dest="json_out", default="", help="куда записать результат")
    args = ap.parse_args(argv)

    docs = load_documents(Path(args.sources))
    hosts = [host.strip() for host in args.host.split(",") if host.strip()]
    if hosts:
        docs = [rec for rec in docs if str(rec.get("source_host") or "") in hosts]

    result: dict[str, Any] = {
        "sources": str(args.sources),
        "hosts_filter": hosts,
        "min_chars_per_page": args.min_chars_per_page,
        "definition_by_method": "поле extraction (метод извлечения) не содержит подстроки ocr",
        "definition_by_layer_chars": f"text_layer_chars / pages_count >= {args.min_chars_per_page}",
        "interval": "95 % доверительный интервал Уилсона для биномиальной доли",
        "by_host": {},
    }
    groups: dict[str, list[dict[str, Any]]] = {}
    for rec in docs:
        groups.setdefault(str(rec.get("source_host") or "без host"), []).append(rec)
    for host in sorted(groups):
        result["by_host"][host] = measure(groups[host], args.min_chars_per_page)
    result["all_hosts"] = measure(docs, args.min_chars_per_page)
    if args.sample and not args.all and args.sample < len(docs):
        rng = random.Random(args.seed)
        picked = rng.sample(docs, args.sample)
        result["sample"] = {"size": args.sample, "seed": args.seed}
        result["sampled"] = measure(picked, args.min_chars_per_page)

    suffix = f" | фильтр по host: {','.join(hosts)}" if hosts else ""
    print(f"источник: {args.sources}{suffix}")
    for host in sorted(result["by_host"]):
        print(format_share(f"  {host}", result["by_host"][host]))
    print(format_share("  вся совокупность", result["all_hosts"]))
    if "sampled" in result:
        print(format_share(f"  выборка n={args.sample} (seed={args.seed})", result["sampled"]))
    for host in sorted(result["by_host"]):
        part = result["by_host"][host]
        if not part["by_act_type"]:
            continue
        print(f"  смещение по видам актов ({host}):")
        for act_type, stats in part["by_act_type"].items():
            share = stats["share_by_method"]
            if share is None:
                print(f"    {act_type}: n={stats['n']}, доля не измерено")
                continue
            lo, hi = stats["ci95_by_method"]
            print(
                f"    {act_type}: n={stats['n']}, с годным слоем {stats['docs_with_text_layer']}, "
                f"доля {share:.4f} ДИ [{lo:.4f}; {hi:.4f}]"
            )

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"json: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
