"""Тесты загрузчика корпуса A3: robots.txt, отбор по темам, запись и проверка хешей.

Сеть в тестах не используется: проверяются те части загрузчика, которые решают, **что**
скачивать и **что** записывать в источники. Это те места, где ошибка портит корпус
незаметно: запрещённый путь, чужой документ не по теме, потерянный хеш.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.fetch_npa_corpus import (
    is_allowed,
    ocr_pdf,
    parse_robots,
    theme_of,
    verify_sources,
    write_sources,
)

ROBOTS_PUBLICATION = (
    "User-agent: *\r\n"
    "Disallow: /Error\r\n"
    "Disallow: /Rss\r\n"
    "Disallow: /File\r\n"
    "Disallow: /Search\r\n"
    "Sitemap: http://publication.pravo.gov.ru/sitemap.xml\r\n"
)


def test_robots_parser_reads_disallow_rules() -> None:
    """Правила robots.txt читаются: запрещённые префиксы попадают в отчёт."""
    rules = parse_robots(ROBOTS_PUBLICATION)
    assert set(rules["*"]) == {"/Error", "/Rss", "/File", "/Search"}


def test_robots_allows_api_and_document_paths() -> None:
    """Разрешённые пути разрешены, запрещённые — нет (проверка не «всё разрешено»)."""
    rules = parse_robots(ROBOTS_PUBLICATION)
    assert is_allowed("http://publication.pravo.gov.ru/api/Documents?pageSize=100", rules)
    assert is_allowed("http://publication.pravo.gov.ru/document/0001202610030023", rules)
    assert not is_allowed("http://publication.pravo.gov.ru/File/pdf?eoNumber=1", rules)
    assert not is_allowed("http://publication.pravo.gov.ru/Search?q=1", rules)


def test_robots_empty_disallow_means_everything_allowed() -> None:
    """Пустой ``Disallow`` (как у pravo.gov.ru) означает «разрешено всё»."""
    rules = parse_robots("User-Agent: * Disallow:")
    assert rules == {"*": []}
    assert is_allowed("http://pravo.gov.ru/proxy/ips/?docbody=&nd=1", rules)


def test_theme_detection_matches_priority_topics() -> None:
    """Темы приоритета распознаются по названию акта, посторонние — нет."""
    assert theme_of("Об утверждении Правил хранения архивных документов") == "сроки хранения документов"
    assert theme_of("О внесении изменений в закон о персональных данных") == "персональные данные"
    assert theme_of("Об утверждении инструкции по охране труда") == "охрана труда"
    assert theme_of("О защите информации в государственных информационных системах") == "защита информации"
    assert theme_of("Об утверждении положения о кадровом учёте") == "кадровый учёт"
    assert theme_of("О правилах дорожного движения") is None


def test_write_sources_creates_files_and_manifest(tmp_path: Path) -> None:
    """Источники записываются вместе с манифестом: URL, вид, номер, дата, хеши."""
    documents = [
        {
            "doc_id": "eo-0001",
            "text": "Срок хранения документов составляет пять лет.",
            "source_url": "http://publication.pravo.gov.ru/document/0001",
            "act_type": "Приказ",
            "act_number": "214",
            "act_date": "2025-03-12",
            "pdf_sha256": "a" * 64,
            "extraction": "ocr-tesseract-rus",
        }
    ]
    manifest: dict = {"method": "тест"}
    write_sources(tmp_path, documents, manifest)

    text_path = tmp_path / "sources" / "eo-0001.txt"
    assert text_path.read_text(encoding="utf-8").startswith("Срок хранения")
    payload = json.loads((tmp_path / "sources" / "sources.json").read_text(encoding="utf-8"))
    meta = payload["documents"]["eo-0001"]
    assert meta["source_url"].endswith("/document/0001")
    assert meta["act_type"] == "Приказ" and meta["act_number"] == "214" and meta["act_date"] == "2025-03-12"
    assert meta["text_file"] == "sources/eo-0001.txt"
    assert len(meta["text_sha256"]) == 64
    assert "text" not in meta


def test_verify_sources_reports_mismatch(tmp_path: Path) -> None:
    """Проверка хешей находит расхождение и не молчит о нём."""
    documents = [{"doc_id": "eo-0001", "text": "Текст источника.", "source_url": "u", "act_type": "Приказ"}]
    manifest: dict = {"method": "тест"}
    write_sources(tmp_path, documents, manifest)

    assert verify_sources(tmp_path)["mismatches"] == []
    assert verify_sources(tmp_path / "sources")["mismatches"] == []

    (tmp_path / "sources" / "eo-0001.txt").write_text("Подменённый текст.\n", encoding="utf-8")
    report = verify_sources(tmp_path)
    assert report["checked"] == 1
    assert len(report["mismatches"]) == 1


def test_verify_sources_without_manifest(tmp_path: Path) -> None:
    """Без ``sources.json`` проверка сообщает об этом, а не считает всё верным."""
    report = verify_sources(tmp_path)
    assert report["documents"] == 0
    assert report["mismatches"] == ["нет sources.json"]


def test_ocr_pdf_reports_missing_tools(tmp_path: Path, monkeypatch) -> None:
    """Если инструментов OCR нет, загрузчик возвращает причину, а не пустой текст молча."""
    monkeypatch.setattr("scripts.fetch_npa_corpus.ocr_available", lambda: {"pdftoppm": None, "tesseract": None})
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.4")
    result = ocr_pdf(broken)
    assert result["error"] and "нет инструментов OCR" in result["error"]
    assert result["text"] == ""
