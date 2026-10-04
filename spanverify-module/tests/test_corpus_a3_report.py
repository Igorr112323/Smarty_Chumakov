"""Тесты отчёта по корпусу A3: числа берутся из манифеста, разделы на месте.

Отчёт — то, что читает проверяющий, поэтому проверяется не «текст красиво выглядит», а
конкретные обязательства задания: числа совпадают с манифестом, числа A1 и A3 не
складываются, есть список источников с числом документов, есть раздел «не сделано» и
ответы на два вопроса (документы по источникам; расхождения контекстов).
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.build_corpus_a import build
from scripts.fetch_npa_corpus import write_sources
from scripts.make_corpus_a3_report import END_MARKER, START_MARKER, build_report, update_summary

TEXTS = {
    "eo-1": (
        "Правила хранения документов утверждены приказом от 12.03.2025 № 214. "
        "Срок хранения первичных учётных документов составляет пять лет, если иное не установлено договором. "
        "Документы, содержащие персональные данные работников, хранятся 30 календарных дней. "
        "Журнал учёта событий ведётся не менее 12 месяцев, за исключением электронных журналов."
    ),
    "eo-2": (
        "Инструкция по охране труда вводится в действие с 01.02.2026. "
        "Повторный инструктаж проводится не реже одного раза в 6 месяцев. "
        "Работники обеспечиваются средствами защиты в количестве 2 комплектов. "
        "Проверка знаний проводится комиссией из 3 человек, за исключением дистанционных работников."
    ),
    "eo-3": (
        "Регламент защиты информации определяет порядок доступа к сведениям. "
        "Пароль должен содержать не менее 12 символов и меняется каждые 90 дней. "
        "Резервная копия создаётся ежедневно в 2 часа ночи. "
        "Хранение резервных копий осуществляется 14 календарных дней, если не установлено иное."
    ),
}


def _corpus(tmp_path: Path, target: int = 30) -> Path:
    """Собрать маленький корпус A3 с источниками (для проверки отчёта)."""
    docs = tmp_path / "sources"
    documents = []
    for doc_id, text in TEXTS.items():
        documents.append(
            {
                "doc_id": doc_id,
                "text": text,
                "source_url": f"http://publication.pravo.gov.ru/document/{doc_id}",
                "pdf_url": f"http://publication.pravo.gov.ru/file/pdf?eoNumber={doc_id}",
                "pdf_sha256": "c" * 64,
                "pdf_bytes": 1000,
                "act_type": "Приказ",
                "act_number": "214",
                "act_date": "2025-03-12",
                "theme": "сроки хранения документов",
                "source_host": "publication.pravo.gov.ru",
                "extraction": "ocr-tesseract-rus",
                "text_truncated": False,
            }
        )
    manifest: dict = {
        "method": "тест",
        "sources": [
            {
                "host": "publication.pravo.gov.ru",
                "documents_downloaded": 3,
                "documents_with_text": 3,
                "note": "тест",
            },
            {"host": "pravo.gov.ru", "documents_downloaded": 0, "documents_with_text": 0, "note": "нет машиночитаемой выдачи"},
        ],
        "robots": {"publication.pravo.gov.ru": {"status": 200, "rules": {"*": ["/Search"]}}},
        "by_type": {"Приказ": 3},
        "by_theme": {"сроки хранения документов": 3},
        "ocr": {"dpi": 200},
    }
    write_sources(tmp_path, documents, manifest)
    build(docs, tmp_path, target=target, seed=11, real=True)
    return tmp_path


def test_report_numbers_match_manifest(tmp_path: Path) -> None:
    """Числа отчёта — из манифеста и источников, а не написаны руками."""
    corpus = _corpus(tmp_path)
    report, facts = build_report(corpus)
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    assert facts["pairs"] == manifest["pairs"]
    assert facts["documents"] == manifest["documents"]["count"]
    assert facts["contexts_checked"] == manifest["contexts_checked"]
    assert facts["contexts_problems"] == len(manifest["contexts_problems"]) == 0
    assert str(manifest["pairs"]) in report
    assert "| `publication.pravo.gov.ru` | 3 | 3 |" in report


def test_report_separates_a1_and_a3(tmp_path: Path) -> None:
    """Разделы A1 и A3 раздельны, суммарных чисел нет."""
    report, _ = build_report(_corpus(tmp_path))
    assert "A1" in report and "A3" in report
    assert "не складывать" in report
    assert "Разделение A1 и A3" in report


def test_report_has_sources_checks_and_not_done(tmp_path: Path) -> None:
    """В отчёте есть источники, проверки, ответы на два вопроса и «не сделано»."""
    report, facts = build_report(_corpus(tmp_path))
    assert "## 1. Откуда взяты документы" in report
    assert "## 3. Проверки (задание, пункты 1–5)" in report
    assert "**1) Сколько документов реально скачано" in report
    assert "**2) Есть ли расхождения `context` пар с исходными документами.**" in report
    assert "Расхождений: **0**" in report
    assert "## 6. Не сделано" in report
    assert "из `pravo.gov.ru` документов нет" in report
    assert facts["pairs"] > 0


def test_report_without_corpus_says_so(tmp_path: Path) -> None:
    """Если корпус не собран, отчёт пишет это прямо, без выдуманных чисел."""
    report, facts = build_report(tmp_path)
    assert facts == {"available": False}
    assert "**Корпус не собран:**" in report


def test_update_summary_is_idempotent(tmp_path: Path) -> None:
    """Раздел A3 вставляется между маркерами и не размножается при повторе."""
    path = tmp_path / "CORPUS_REPORT.md"
    path.write_text("# Отчёты по корпусам\n\nЧто-то про A1.\n", encoding="utf-8")
    update_summary(path, "## Раздел A3\n\nЧисла.\n")
    update_summary(path, "## Раздел A3\n\nЧисла.\n")
    text = path.read_text(encoding="utf-8")
    assert text.count(START_MARKER) == 1 and text.count(END_MARKER) == 1
    assert text.count("## Раздел A3") == 1
    assert "Что-то про A1." in text
