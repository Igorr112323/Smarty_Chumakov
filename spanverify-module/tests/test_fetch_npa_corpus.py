"""Тесты загрузчика A3: robots.txt, отбор по темам, извлечение текста, хеши.

Сеть в тестах не используется: проверяются решающие функции — те, где ошибка тихо
портит корпус (запрещённый robots.txt путь, документ не по теме, пустой текст вместо
скана, потерянный при повторном запуске хеш).
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.fetch_npa_corpus import (
    extract_facts,
    is_allowed,
    load_existing,
    parse_robots,
    pick_extraction,
    plan_downloads,
    theme_of,
    verify_sources,
    write_sources,
)

ROBOTS_PUBLICATION = (
    "User-Agent: *\r\n"
    "Disallow: /Error\r\n"
    "Disallow: /Rss\r\n"
    "Disallow: /app\r\n"
    "Disallow: /js\r\n"
    "Disallow: /File\r\n"
    "Disallow: /Search\r\n"
    "Sitemap: http://publication.pravo.gov.ru/sitemap.xml\r\n"
)


def _document(doc_id: str = "eo-0001", theme: str = "охрана труда") -> dict:
    """Документ для записи в источники (поля те же, что пишет загрузчик)."""
    return {
        "doc_id": doc_id,
        "text": "Повторный инструктаж проводится не реже одного раза в 6 месяцев и оформляется журналом.",
        "source_url": f"http://publication.pravo.gov.ru/document/{doc_id}",
        "pdf_url": f"http://publication.pravo.gov.ru/file/pdf?eoNumber={doc_id}",
        "pdf_sha256": "b" * 64,
        "pdf_bytes": 123456,
        "act_type": "Приказ",
        "act_number": "214",
        "act_date": "2025-03-12",
        "theme": theme,
        "extraction": "ocr-tesseract-rus",
        "facts_found": 7,
    }


def test_robots_parser_and_allowed_paths() -> None:
    """robots.txt разбирается, запрещённые пути запрещены, рабочие — разрешены."""
    rules = parse_robots(ROBOTS_PUBLICATION)
    assert set(rules["*"]) == {"/Error", "/Rss", "/app", "/js", "/File", "/Search"}
    assert is_allowed("http://publication.pravo.gov.ru/api/Documents?pageSize=100", rules)
    assert is_allowed("http://publication.pravo.gov.ru/file/pdf?eoNumber=1", rules)
    assert not is_allowed("http://publication.pravo.gov.ru/Search?q=закон", rules)
    assert not is_allowed("http://publication.pravo.gov.ru/File/pdf?eoNumber=1", rules)


def test_robots_empty_disallow_allows_everything() -> None:
    """Пустой ``Disallow`` (как у pravo.gov.ru) означает «разрешено всё»."""
    rules = parse_robots("User-Agent: * Disallow:")
    assert rules == {"*": []}
    assert is_allowed("http://pravo.gov.ru/proxy/ips/?docbody=&nd=1", rules)


def test_theme_of_matches_priority_topics() -> None:
    """Темы приоритета распознаются по названию акта, посторонние — нет."""
    assert theme_of("Об утверждении Перечня документов с указанием сроков их хранения") == ("сроки хранения документов")
    assert theme_of("Об обработке персональных данных в информационных системах") == "персональные данные"
    assert theme_of("Об утверждении правил защиты информации") == "защита информации"
    assert theme_of("Об утверждении инструкции по охране труда") == "охрана труда"
    assert theme_of("О ведении кадрового учёта") == "кадровый учёт"
    assert theme_of("О подписании Соглашения о финансовых обязательствах") is None


def test_pick_extraction_prefers_text_layer_then_ocr() -> None:
    """Способ извлечения выбирается по фактам: текстовый слой → OCR → отказ."""
    text_layer = {"text": "Слово " * 400, "pages": 2}
    assert pick_extraction(text_layer, {"text": "OCR"}) == "pypdf"
    assert pick_extraction({"text": " ", "pages": 8}, {"text": "распознанный текст"}) == "ocr-tesseract-rus"
    assert pick_extraction({"text": "", "pages": 8}, {"text": ""}) == "none"


def test_write_sources_and_verify_hashes(tmp_path: Path) -> None:
    """Источники пишутся с URL/видом/номером/датой и проверяются по SHA256."""
    write_sources(tmp_path, [_document()], {"generated_at": "тест", "documents": {}})
    text_path = tmp_path / "sources" / "eo-0001.txt"
    assert text_path.exists()
    payload = json.loads((tmp_path / "sources" / "sources.json").read_text(encoding="utf-8"))
    meta = payload["documents"]["eo-0001"]
    assert meta["source_url"].endswith("/document/eo-0001")
    assert meta["act_type"] == "Приказ" and meta["act_number"] == "214" and meta["act_date"] == "2025-03-12"
    assert len(meta["text_sha256"]) == 64
    assert verify_sources(tmp_path / "sources")["mismatches"] == []

    text_path.write_text("подменённый текст\n", encoding="utf-8")
    report = verify_sources(tmp_path / "sources")
    assert report["checked"] == 1 and len(report["mismatches"]) == 1


def test_verify_sources_reports_missing_manifest(tmp_path: Path) -> None:
    """Без ``sources.json`` проверка сообщает об этом, а не считает всё верным."""
    report = verify_sources(tmp_path)
    assert report["documents"] == 0 and report["mismatches"] == ["нет sources.json"]


def test_load_existing_skips_tampered_files(tmp_path: Path) -> None:
    """Повторный запуск берёт только файлы с совпадающим хешем, испорченный — заново."""
    write_sources(tmp_path, [_document(), _document("eo-0002")], {"documents": {}})
    documents, manifest = load_existing(tmp_path)
    assert {item["doc_id"] for item in documents} == {"eo-0001", "eo-0002"}
    assert manifest["documents"]["eo-0002"]["text_sha256"]

    (tmp_path / "sources" / "eo-0002.txt").write_text("испорчено\n", encoding="utf-8")
    documents, _ = load_existing(tmp_path)
    assert {item["doc_id"] for item in documents} == {"eo-0001"}


def test_plan_downloads_covers_themes_and_skips_existing() -> None:
    """План загрузки: уже скачанное пропускается, редкие темы получают свою долю."""
    candidates = []
    for index in range(30):
        theme = ("персональные данные", "охрана труда", "кадровый учёт")[index % 3]
        candidates.append({"doc_id": f"eo-{index:04d}", "theme": theme, "skipped": None})
    candidates.append({"doc_id": "eo-9999", "theme": "охрана труда", "skipped": "robots"})

    planned = plan_downloads(candidates, existing_ids={"eo-0000"}, needed=6, buffer=2)
    assert len(planned) == 8
    assert "eo-0000" not in {item["doc_id"] for item in planned}
    assert "eo-9999" not in {item["doc_id"] for item in planned}
    assert {item["theme"] for item in planned} == {"персональные данные", "охрана труда", "кадровый учёт"}

    assert plan_downloads(candidates, existing_ids=set(), needed=0, buffer=0) == []


def test_extract_facts_needs_values_and_words() -> None:
    """Фактом считается только содержательное предложение со значением."""
    text = (
        "Настоящие правила вводятся в действие с 01.09.2025. "
        "Срок хранения первичных учётных документов составляет пять лет. "
        "Общие положения. Контроль возлагается на руководителя подразделения."
    )
    facts = extract_facts("eo-1", text)
    assert facts, "ожидался хотя бы один факт"
    assert all(len(fact.sentence.split()) >= 8 for fact in facts)
    assert any("пять лет" in fact.value for fact in facts)
