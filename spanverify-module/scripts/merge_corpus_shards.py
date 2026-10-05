#!/usr/bin/env python3
"""Слияние частей корпуса A3, собранных параллельно на разных раннерах.

Зачем это нужно
---------------

Распознавание сканов официальных публикаций — самая дорогая часть сборки: замер
первого прогона дал 580–2680 секунд на документ (``ocr_seconds`` в манифесте).
Последовательная загрузка 120 документов в такой скорости не укладывается ни в один
разумный лимит времени, поэтому загрузка разрезается на независимые срезы (по видам
актов и по региональным разделам), каждый срез собирается своим заданием, а результаты
сливаются этим скриптом.

Что делает слияние
------------------

1. Читает ``sources.json`` каждой части и сами тексты.
2. Складывает документы в один каталог, **проверяя SHA256 текста**: если файл части
   не совпадает со своим хешем, документ не берётся, а причина попадает в отчёт.
3. При совпадении ``doc_id`` в двух частях берётся первая встреченная запись, факт
   дубликата записывается (``duplicates``) — молча терять документы нельзя.
4. Пересчитывает сводные числа (уровни, регионы, темы, виды, способы извлечения) по
   фактическому составу, а не копирует их из частей.
5. Пишет общий ``sources.json`` и печатает итог.

Запуск::

    python scripts/merge_corpus_shards.py --shards artifacts/corpus-shard-* --out data/corpus_a3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

from scripts.fetch_npa_corpus import region_of  # noqa: E402 - импорт после правки sys.path

ROOT = _MODULE_ROOT


def sha256_bytes(raw: bytes) -> str:
    """SHA256 байтов (тот же способ, что и в загрузчике)."""
    return hashlib.sha256(raw).hexdigest()


def find_manifest(shard: Path) -> Path | None:
    """Найти ``sources.json`` внутри части (каталог части или его подкаталог)."""
    for candidate in (
        shard / "sources.json",
        shard / "sources" / "sources.json",
        shard / "data" / "corpus_a3" / "sources" / "sources.json",
    ):
        if candidate.is_file():
            return candidate
    matches = sorted(shard.rglob("sources.json"))
    return matches[0] if matches else None


def read_shard(shard: Path) -> dict:
    """Прочитать часть: документы с текстами плюс факты о проблемах."""
    manifest_path = find_manifest(shard)
    if manifest_path is None:
        return {"shard": str(shard), "documents": {}, "problems": ["нет sources.json"], "manifest": {}}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return {"shard": str(shard), "documents": {}, "problems": [f"битый sources.json: {error}"], "manifest": {}}

    documents: dict[str, dict] = {}
    problems: list[str] = []
    for doc_id, meta in (manifest.get("documents") or {}).items():
        path = manifest_path.parent / f"{doc_id}.txt"
        if not path.is_file():
            problems.append(f"{doc_id}: файл текста отсутствует")
            continue
        raw = path.read_bytes()
        expected = meta.get("text_sha256")
        if expected and sha256_bytes(raw) != expected:
            problems.append(f"{doc_id}: SHA256 текста не совпал — документ не взят")
            continue
        entry = dict(meta)
        entry["_text_bytes"] = raw
        documents[doc_id] = entry
    return {"shard": str(shard), "documents": documents, "problems": problems, "manifest": manifest}


def summarise(documents: dict[str, dict]) -> dict:
    """Пересчитать сводные числа по фактическому составу документов."""
    by_theme: dict[str, int] = {}
    by_type: dict[str, int] = {}
    by_level: dict[str, int] = {}
    by_region: dict[str, int] = {}
    by_method: dict[str, int] = {}
    chars = 0
    facts = 0
    for doc_id, meta in documents.items():
        placement = region_of(str(meta.get("eo_number") or doc_id.removeprefix("eo-")))
        level = str(meta.get("level") or placement["level"])
        region = str(meta.get("region_name") or placement["region_name"] or placement["region_code"])
        by_theme[str(meta.get("theme"))] = by_theme.get(str(meta.get("theme")), 0) + 1
        by_type[str(meta.get("act_type"))] = by_type.get(str(meta.get("act_type")), 0) + 1
        by_level[level] = by_level.get(level, 0) + 1
        if level == "региональный":
            by_region[region] = by_region.get(region, 0) + 1
        by_method[str(meta.get("extraction"))] = by_method.get(str(meta.get("extraction")), 0) + 1
        chars += int(meta.get("text_chars") or 0)
        facts += int(meta.get("facts_found") or 0)
    return {
        "documents_total": len(documents),
        "by_theme": by_theme,
        "by_type": by_type,
        "by_level": by_level,
        "by_region": by_region,
        "extraction_methods": by_method,
        "text_chars_total": chars,
        "facts_found_total": facts,
    }


def merge(shards: list[Path], out_dir: Path) -> dict:
    """Слить части в ``out_dir/sources`` и вернуть отчёт о слиянии."""
    sources_dir = out_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)

    merged: dict[str, dict] = {}
    duplicates: list[str] = []
    problems: list[str] = []
    shard_reports: list[dict] = []
    base_manifest: dict = {}

    # Уже лежащие в каталоге документы — такая же часть, как и остальные: прогон
    # должен быть накопительным, иначе повторный запуск стирает прошлый результат.
    existing_manifest = sources_dir / "sources.json"
    ordered: list[Path] = []
    if existing_manifest.is_file():
        ordered.append(sources_dir)
    ordered.extend(shards)

    for shard in ordered:
        report = read_shard(shard)
        shard_reports.append(
            {
                "shard": report["shard"],
                "documents": len(report["documents"]),
                "problems": report["problems"],
            }
        )
        problems.extend(f"{shard.name}: {item}" for item in report["problems"])
        if report["manifest"] and not base_manifest:
            base_manifest = report["manifest"]
        for doc_id, meta in report["documents"].items():
            if doc_id in merged:
                duplicates.append(doc_id)
                continue
            merged[doc_id] = meta

    for doc_id, meta in sorted(merged.items()):
        (sources_dir / f"{doc_id}.txt").write_bytes(meta["_text_bytes"])

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": base_manifest.get("method"),
        "user_agent": base_manifest.get("user_agent"),
        "pause_seconds": base_manifest.get("pause_seconds"),
        "whitelist": base_manifest.get("whitelist"),
        "whitelist_config": base_manifest.get("whitelist_config"),
        "robots": base_manifest.get("robots"),
        "sources": base_manifest.get("sources"),
        "merged_from": shard_reports,
        "merge_duplicates": sorted(set(duplicates)),
        "merge_problems": problems,
        **summarise(merged),
        "documents": {
            doc_id: {key: value for key, value in meta.items() if key != "_text_bytes"}
            for doc_id, meta in sorted(merged.items())
        },
    }
    (sources_dir / "sources.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Слияние частей корпуса A3")
    parser.add_argument("--shards", nargs="+", type=Path, help="каталоги частей (артефакты заданий)")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "corpus_a3")
    args = parser.parse_args()

    shards = [path for path in (args.shards or []) if path.is_dir()]
    if not shards:
        print("частей не найдено: сливать нечего", file=sys.stderr)
    manifest = merge(shards, args.out)
    print(f"частей слито: {len(manifest['merged_from'])}")
    print(f"документов: {manifest['documents_total']}")
    print(f"уровни: {manifest['by_level']}")
    print(f"регионы: {manifest['by_region']}")
    print(f"способы извлечения: {manifest['extraction_methods']}")
    print(f"знаков текста: {manifest['text_chars_total']}; фактов найдено: {manifest['facts_found_total']}")
    if manifest["merge_duplicates"]:
        print(f"дубликаты между частями: {len(manifest['merge_duplicates'])}")
    if manifest["merge_problems"]:
        print(f"проблемы слияния: {manifest['merge_problems'][:5]}", file=sys.stderr)
    print(
        "::notice title=A3 слияние::"
        f"документов={manifest['documents_total']} уровни={manifest['by_level']} "
        f"регионы={manifest['by_region']} знаков={manifest['text_chars_total']} "
        f"фактов={manifest['facts_found_total']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
