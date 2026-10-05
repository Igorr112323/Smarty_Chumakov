"""Тесты слияния частей корпуса A3.

Слияние — место, где корпус может тихо испортиться: подменённый текст части,
дубликат документа в двух частях, потеря уже собранных документов при повторном
запуске. Каждый из этих случаев проверяется на настоящих файлах во временном
каталоге; сеть не используется.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.merge_corpus_shards import find_manifest, merge, read_shard, summarise


def _write_shard(root: Path, documents: dict[str, str], *, extra: dict | None = None) -> Path:
    """Создать каталог части с текстами и корректным sources.json."""
    sources = root / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"method": "тестовая часть", "user_agent": "test", "documents": {}}
    for doc_id, text in documents.items():
        path = sources / f"{doc_id}.txt"
        path.write_text(text, encoding="utf-8")
        eo = doc_id.removeprefix("eo-")
        manifest["documents"][doc_id] = {
            "eo_number": eo,
            "theme": "прочее",
            "act_type": "Постановление",
            "extraction": "ocr-tesseract-rus",
            "text_chars": len(text),
            "facts_found": 7,
            "text_file": f"sources/{doc_id}.txt",
            "text_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            **(extra or {}),
        }
    (sources / "sources.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return root


def test_merge_combines_shards_and_recounts_summary(tmp_path: Path) -> None:
    """Две части сливаются в один каталог, сводные числа считаются заново."""
    first = _write_shard(tmp_path / "shard-federal", {"eo-0001202610020001": "Срок хранения 5 лет. " * 10})
    second = _write_shard(tmp_path / "shard-region23", {"eo-2300202610020003": "Ставка 15 процентов. " * 10})

    manifest = merge([first, second], tmp_path / "out")

    assert manifest["documents_total"] == 2
    assert manifest["by_level"] == {"федеральный": 1, "региональный": 1}
    assert manifest["by_region"] == {"Краснодарский край": 1}
    assert manifest["facts_found_total"] == 14
    assert manifest["merge_duplicates"] == []
    assert manifest["merge_problems"] == []

    out_sources = tmp_path / "out" / "sources"
    assert (out_sources / "eo-0001202610020001.txt").is_file()
    assert (out_sources / "eo-2300202610020003.txt").is_file()
    written = json.loads((out_sources / "sources.json").read_text(encoding="utf-8"))
    assert set(written["documents"]) == {"eo-0001202610020001", "eo-2300202610020003"}


def test_merge_rejects_document_with_broken_hash(tmp_path: Path) -> None:
    """Подменённый текст части не попадает в корпус, причина записывается."""
    shard = _write_shard(tmp_path / "shard", {"eo-0001202610020001": "исходный текст " * 10})
    (shard / "sources" / "eo-0001202610020001.txt").write_text("подменено", encoding="utf-8")

    manifest = merge([shard], tmp_path / "out")

    assert manifest["documents_total"] == 0
    assert any("SHA256" in problem for problem in manifest["merge_problems"])


def test_merge_reports_duplicates_and_keeps_first(tmp_path: Path) -> None:
    """Один документ в двух частях берётся один раз, дубликат фиксируется."""
    first = _write_shard(tmp_path / "a", {"eo-0001202610020001": "первая часть " * 10})
    second = _write_shard(tmp_path / "b", {"eo-0001202610020001": "вторая часть " * 10})

    manifest = merge([first, second], tmp_path / "out")

    assert manifest["documents_total"] == 1
    assert manifest["merge_duplicates"] == ["eo-0001202610020001"]
    text = (tmp_path / "out" / "sources" / "eo-0001202610020001.txt").read_text(encoding="utf-8")
    assert text.startswith("первая часть")


def test_merge_is_cumulative_over_existing_corpus(tmp_path: Path) -> None:
    """Повторное слияние не стирает уже собранные документы."""
    out = tmp_path / "out"
    first = _write_shard(tmp_path / "a", {"eo-0001202610020001": "первый акт " * 10})
    merge([first], out)

    second = _write_shard(tmp_path / "b", {"eo-2300202610020003": "второй акт " * 10})
    manifest = merge([second], out)

    assert manifest["documents_total"] == 2
    assert (out / "sources" / "eo-0001202610020001.txt").is_file()
    assert (out / "sources" / "eo-2300202610020003.txt").is_file()


def test_find_manifest_handles_artifact_layouts(tmp_path: Path) -> None:
    """Манифест находится и рядом с текстами, и в вложенной раскладке артефакта."""
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "sources.json").write_text("{}", encoding="utf-8")
    assert find_manifest(flat) == flat / "sources.json"

    nested = tmp_path / "nested" / "data" / "corpus_a3" / "sources"
    nested.mkdir(parents=True)
    (nested / "sources.json").write_text("{}", encoding="utf-8")
    assert find_manifest(tmp_path / "nested") == nested / "sources.json"

    assert find_manifest(tmp_path / "пусто") is None


def test_read_shard_reports_missing_manifest(tmp_path: Path) -> None:
    """Часть без манифеста не роняет слияние, а объясняет причину."""
    empty = tmp_path / "empty"
    empty.mkdir()
    report = read_shard(empty)
    assert report["documents"] == {}
    assert report["problems"] == ["нет sources.json"]


def test_summarise_counts_levels_from_publication_number() -> None:
    """Уровень и регион восстанавливаются из номера опубликования, если их нет в записи."""
    documents = {
        "eo-0001202610020001": {"eo_number": "0001202610020001", "text_chars": 10, "facts_found": 1},
        "eo-7700202610020002": {"eo_number": "7700202610020002", "text_chars": 20, "facts_found": 2},
    }
    summary = summarise(documents)
    assert summary["by_level"] == {"федеральный": 1, "региональный": 1}
    assert summary["by_region"] == {"Москва": 1}
    assert summary["text_chars_total"] == 30
    assert summary["facts_found_total"] == 3
