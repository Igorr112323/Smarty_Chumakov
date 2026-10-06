"""Тесты загрузчика внешнего корпуса RusHallu-RAG.

Сеть в тестах не используется: скачивание подменяется записью заранее известных
байтов, а проверяется то, что важнее — перевод разметки в смещения символов,
склейка документов, контроль SHA256 и содержимое манифеста. Именно эти части
определяют, попадёт ли в отчёт корректный корпус.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.fetch_rushallu import (
    FetchError,
    build_context,
    build_pairs,
    fetch,
    sha256_file,
    spans_to_labels,
)


def test_spans_to_labels_gives_character_offsets() -> None:
    """Спан становится тройкой [начало, конец, 1], и срез ответа равен спану."""
    answer = "Срок хранения составляет десять лет."
    annotation = json.dumps([{"type": "Contradiction", "span": "десять лет"}])
    labels, types, ambiguous = spans_to_labels(answer, annotation)
    assert labels == [[answer.index("десять лет"), answer.index("десять лет") + len("десять лет"), 1]]
    assert types == ["Contradiction"]
    assert ambiguous == 0
    start, end, _ = labels[0]
    assert answer[start:end] == "десять лет"


def test_spans_to_labels_counts_ambiguous_offsets() -> None:
    """Повторяющийся спан берёт первое вхождение, но отмечается как неоднозначный."""
    answer = "пять лет и снова пять лет"
    labels, _, ambiguous = spans_to_labels(answer, json.dumps([{"type": "Excess", "span": "пять лет"}]))
    assert labels == [[0, 8, 1]]
    assert ambiguous == 1


def test_spans_to_labels_rejects_span_absent_in_answer() -> None:
    """Спан, которого нет в ответе дословно, — ошибка загрузки, а не молчаливый пропуск."""
    with pytest.raises(FetchError, match="не найден"):
        spans_to_labels("короткий ответ", json.dumps([{"type": "Missing", "span": "длинная цитата"}]))


def test_empty_annotation_means_clean_pair() -> None:
    """Пустая разметка — это чистая пара: labels пустые, types пустые."""
    labels, types, _ = spans_to_labels("любой ответ", "[]")
    assert labels == []
    assert types == []


def test_build_context_keeps_document_order_and_separators() -> None:
    """Документы склеиваются в исходном порядке и получают заголовки-разделители."""
    context = build_context(
        [{"doc_text": "первый"}, {"doc_text": "второй"}],
    )
    assert context.index("первый") < context.index("второй")
    assert "[ДОКУМЕНТ 1]" in context
    assert "[ДОКУМЕНТ 2]" in context


def test_build_pairs_collects_statistics_by_type() -> None:
    """Статистика загрузки считает чистые пары, спаны и разбивку по типам."""
    rows = [
        {
            "query_id": "1",
            "query_text": "вопрос",
            "docs": "[{'doc_id': 1, 'doc_text': 'Срок пять лет.'}]",
            "model_output": "Срок десять лет.",
            "answer": json.dumps([{"type": "Contradiction", "span": "десять"}]),
            "joined_reference": "Срок пять лет.",
        },
        {
            "query_id": "2",
            "query_text": "вопрос",
            "docs": "[{'doc_id': 2, 'doc_text': 'Срок пять лет.'}]",
            "model_output": "Срок пять лет.",
            "answer": "[]",
            "joined_reference": "Срок пять лет.",
        },
    ]
    pairs, stats = build_pairs(rows, "sberquad")
    assert [pair["id"] for pair in pairs] == ["rushallu-sberquad-1", "rushallu-sberquad-2"]
    assert stats["clean"] == 1
    assert stats["with_hallucination"] == 1
    assert stats["spans"] == 1
    assert stats["types"] == {"Contradiction": 1}
    assert pairs[1]["labels"] == []
    assert pairs[0]["context"].startswith("[ДОКУМЕНТ 1]")


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    """Хеш считается по фактическим байтам файла (в том числе для больших файлов)."""
    import hashlib

    target = tmp_path / "data.bin"
    target.write_bytes(b"spanverify" * 1000)
    assert sha256_file(target) == hashlib.sha256(b"spanverify" * 1000).hexdigest()


def test_fetch_verifies_hashes_and_writes_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Успешная загрузка пишет пары и манифест с подтверждёнными SHA256."""
    import csv as csv_module
    import io

    buffer = io.StringIO()
    writer = csv_module.writer(buffer)
    writer.writerow(
        [
            "",
            "query_id",
            "query_text",
            "docs",
            "reference_docs",
            "joined_reference",
            "rel_position_mod",
            "score_mod",
            "model_output",
            "answer",
        ]
    )
    writer.writerow(
        [
            "0",
            "7",
            "вопрос",
            "[{'doc_id': 1, 'doc_text': 'Срок пять лет.'}]",
            "[]",
            "Срок пять лет.",
            "0",
            "1",
            "Срок десять лет.",
            json.dumps([{"type": "Contradiction", "span": "десять"}]),
        ]
    )
    csv_body = buffer.getvalue()
    real_bytes = csv_body.encode("utf-8")
    import hashlib

    digest = hashlib.sha256(real_bytes).hexdigest()

    def fake_download(name: str, dest: Path, transport: str = "auto") -> str:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(real_bytes)
        return "stub"

    monkeypatch.setattr("scripts.fetch_rushallu.download_file", fake_download)
    monkeypatch.setattr(
        "scripts.fetch_rushallu.FILES",
        {"sberquad-rag.csv": digest, "ruscibench-rag.csv": digest},
    )

    report = fetch(tmp_path / "out", verify=True)
    assert report["summary"]["clean"] == 0
    assert report["summary"]["spans"] == 2  # две пары, по одному спану в каждой
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert all(item["verified"] for item in manifest["files"].values())
    pairs_path = tmp_path / "out" / "pairs.jsonl"
    assert manifest["pairs_sha256"] == sha256_file(pairs_path)
    assert len(pairs_path.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_fetch_fails_loudly_on_hash_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Несовпадение SHA256 останавливает загрузку: данные изменились на стороне источника."""

    def fake_download(name: str, dest: Path, transport: str = "auto") -> str:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(
            b",query_id,query_text,docs,reference_docs,joined_reference,rel_position_mod,score_mod,model_output,answer\n"
        )
        return "stub"

    monkeypatch.setattr("scripts.fetch_rushallu.download_file", fake_download)
    monkeypatch.setattr("scripts.fetch_rushallu.FILES", {"sberquad-rag.csv": "0" * 64})
    with pytest.raises(FetchError, match="SHA256"):
        fetch(tmp_path / "out", verify=True)
