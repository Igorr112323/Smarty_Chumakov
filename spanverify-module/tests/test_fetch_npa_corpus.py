"""Тесты загрузчика A3: robots.txt, отбор по темам, извлечение текста, хеши.

Сеть в тестах не используется: проверяются решающие функции — те, где ошибка тихо
портит корпус (запрещённый robots.txt путь, документ не по теме, пустой текст вместо
скана, потерянный при повторном запуске хеш).
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts import fetch_npa_corpus
from scripts.fetch_npa_corpus import (
    WHITELIST,
    candidate_from_item,
    extract_facts,
    is_allowed,
    load_existing,
    load_whitelist,
    parse_robots,
    pick_extraction,
    plan_downloads,
    region_of,
    scan_list,
    text_layer_is_enough,
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


def test_region_of_reads_level_from_publication_number() -> None:
    """Уровень и регион читаются из номера опубликования, а не из справочника органов."""
    federal = region_of("0001202310130023")
    assert federal["level"] == "федеральный"
    assert federal["region_name"] is None

    krasnodar = region_of("2300202610020003")
    assert krasnodar["level"] == "региональный"
    assert krasnodar["region_code"] == "23"
    assert krasnodar["region_name"] == "Краснодарский край"

    # Второй орган того же региона (приказы краевых министерств) — тот же регион.
    assert region_of("2301202610020001")["region_name"] == "Краснодарский край"
    # Регион вне списка приоритета опознаётся как региональный, но без названия.
    altai = region_of("0400202609090013")
    assert altai["level"] == "региональный"
    assert altai["region_code"] == "04"
    assert altai["region_name"] is None
    # Мусор не должен приводить к выдуманному уровню.
    assert region_of("")["level"] is None
    assert region_of("12")["level"] is None


def test_theme_of_marks_other_topics_only_when_allowed() -> None:
    """Акт вне приоритетных тем отбрасывается, а с --all-themes помечается «прочее»."""
    name = "О внесении изменений в порядок передачи движимого имущества"
    assert theme_of(name) is None
    assert theme_of(name, allow_other=True) == "прочее"
    # Приоритетная тема остаётся приоритетной и в режиме «прочее».
    assert theme_of("Об утверждении сроков хранения документов", allow_other=True) == "сроки хранения документов"


def test_candidate_from_item_fills_region_and_type() -> None:
    """Карточка кандидата: вид акта из заголовка, регион — из номера опубликования."""
    item = {
        "eoNumber": "2300202610020006",
        "name": "Об утверждении Порядка передачи движимого имущества",
        "complexName": 'Постановление Губернатора Краснодарского края от 28.09.2026 № 650\n "Об утверждении"',
        "number": "650",
        "documentDate": "2026-09-28T00:00:00",
        "publishDateShort": "2026-10-02T00:00:00",
        "pagesCount": 7,
        "pdfFileLength": 314492,
        "id": "28e26983-65cf-4556-9678-bf0b49ce5dbe",
    }
    candidate = candidate_from_item(item, theme="прочее", act_type=None)
    assert candidate["doc_id"] == "eo-2300202610020006"
    assert candidate["act_type"] == "Постановление"
    assert candidate["region_name"] == "Краснодарский край"
    assert candidate["level"] == "региональный"
    assert candidate["pages_count"] == 7
    assert candidate["source_url"].endswith("/document/2300202610020006")

    # Если вид акта известен из ветки отбора по типу, заголовок его не перебивает.
    assert candidate_from_item(item, theme="прочее", act_type="Приказ")["act_type"] == "Приказ"


def test_scan_list_refuses_disallowed_path_without_request() -> None:
    """Запрещённый robots.txt путь не запрашивается: причина попадает в статистику."""
    rules = parse_robots(ROBOTS_PUBLICATION)
    stats: dict = {"pages_scanned": 0}
    url = "http://publication.pravo.gov.ru/Search?q=1"
    assert scan_list(url, rules, stats, pause=0.0) is None
    assert stats["pages_scanned"] == 0
    assert stats["skipped_by_robots"] == [url]


def test_text_layer_is_enough_uses_chars_per_page() -> None:
    """Порог пригодности текстового слоя считается на страницу, а не на документ."""
    assert text_layer_is_enough({"text": "я" * 300, "pages": 2}) is True
    assert text_layer_is_enough({"text": "я" * 100, "pages": 2}) is False
    assert text_layer_is_enough({"text": "", "pages": 0}) is False


def test_extract_text_skips_ocr_when_text_layer_is_good(monkeypatch, tmp_path: Path) -> None:
    """OCR не запускается, если текстового слоя хватает: именно это съедало время CI."""
    calls: list[str] = []

    def fake_pypdf(path: Path) -> dict:
        return {"text": "Срок хранения составляет 5 лет. " * 40, "pages": 2, "error": None}

    def fake_ocr(path: Path, dpi: int = 200, max_pages: int = 20) -> dict:
        calls.append("ocr")
        return {"text": "распознано", "pages": 2, "seconds": 5.0, "error": None}

    monkeypatch.setattr(fetch_npa_corpus, "pypdf_text", fake_pypdf)
    monkeypatch.setattr(fetch_npa_corpus, "ocr_pdf", fake_ocr)

    result = fetch_npa_corpus.extract_text(tmp_path / "a.pdf")
    assert result["method"] == "pypdf"
    assert calls == []
    assert result["ocr_skipped"] == "текстовый слой пригоден"


def test_extract_text_falls_back_to_ocr_for_scans(monkeypatch, tmp_path: Path) -> None:
    """Скан без текстового слоя распознаётся; с --no-ocr честно остаётся без текста."""

    def fake_pypdf(path: Path) -> dict:
        return {"text": " ", "pages": 3, "error": None}

    def fake_ocr(path: Path, dpi: int = 200, max_pages: int = 20) -> dict:
        return {"text": "распознанный текст акта", "pages": 3, "seconds": 7.5, "error": None}

    monkeypatch.setattr(fetch_npa_corpus, "pypdf_text", fake_pypdf)
    monkeypatch.setattr(fetch_npa_corpus, "ocr_pdf", fake_ocr)

    recognised = fetch_npa_corpus.extract_text(tmp_path / "a.pdf")
    assert recognised["method"] == "ocr-tesseract-rus"
    assert recognised["ocr_seconds"] == 7.5

    without_ocr = fetch_npa_corpus.extract_text(tmp_path / "a.pdf", allow_ocr=False)
    assert without_ocr["method"] == "none"
    assert without_ocr["ocr_error"] == "OCR отключён ключом --no-ocr"


def test_plan_downloads_balances_levels_and_region_priority() -> None:
    """План держит долю регионов и ставит Краснодарский край первым среди субъектов."""
    candidates: list[dict] = []
    for index in range(40):
        eo = f"0001202610{index:06d}"
        candidates.append(dict(candidate_from_item({"eoNumber": eo, "pagesCount": 5}, "прочее", "Постановление")))
    # Регионы намеренно перечислены в обратном порядке приоритета.
    for code in ("66", "16", "26", "61", "78", "50", "77", "23"):
        for index in range(5):
            eo = f"{code}00202610{index:06d}"
            candidates.append(dict(candidate_from_item({"eoNumber": eo, "pagesCount": 4}, "прочее", None)))

    planned = plan_downloads(candidates, existing_ids=set(), needed=20, buffer=0, regional_share=0.45)
    assert len(planned) == 20
    levels = [item["level"] for item in planned]
    assert levels.count("региональный") == 9
    assert levels.count("федеральный") == 11
    regions = [item["region_name"] for item in planned if item["level"] == "региональный"]
    assert regions[:5] == ["Краснодарский край"] * 5
    assert "Москва" in regions


def test_plan_downloads_prefers_documents_with_usable_length() -> None:
    """Одностраничные и сверхдлинные акты уходят в конец очереди, но не исключаются."""
    candidates = [
        dict(candidate_from_item({"eoNumber": "0001202610000001", "pagesCount": 1}, "прочее", "Приказ")),
        dict(candidate_from_item({"eoNumber": "0001202610000002", "pagesCount": 300}, "прочее", "Приказ")),
        dict(candidate_from_item({"eoNumber": "0001202610000003", "pagesCount": 8}, "прочее", "Приказ")),
    ]
    planned = plan_downloads(candidates, existing_ids=set(), needed=1, buffer=0)
    assert planned[0]["doc_id"] == "eo-0001202610000003"

    # Если выбирать не из чего, короткие акты всё равно берутся — это очередь, не фильтр.
    all_planned = plan_downloads(candidates, existing_ids=set(), needed=3, buffer=0)
    assert len(all_planned) == 3


def test_load_whitelist_reads_config_and_falls_back(tmp_path: Path) -> None:
    """Белый список читается из конфигурации; при её отсутствии — встроенный список."""
    config = tmp_path / "sources_whitelist.json"
    config.write_text(
        json.dumps({"sources": [{"host": "publication.pravo.gov.ru"}, {"host": "pravo.gov.ru"}]}),
        encoding="utf-8",
    )
    assert load_whitelist(config) == ("publication.pravo.gov.ru", "pravo.gov.ru")
    assert load_whitelist(tmp_path / "нет.json") == WHITELIST

    broken = tmp_path / "broken.json"
    broken.write_text("{не json", encoding="utf-8")
    assert load_whitelist(broken) == WHITELIST


def test_project_whitelist_config_is_valid() -> None:
    """Конфигурация источников проекта читается и содержит портал опубликования."""
    hosts = load_whitelist()
    assert "publication.pravo.gov.ru" in hosts
    assert len(hosts) >= 6
