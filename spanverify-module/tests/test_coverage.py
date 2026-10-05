"""Покрытие фактов документа ответом: типы ``missing`` и ``partial``.

Тесты воспроизводят именно те формы ответов, на которых старый конвейер давал
recall 0.00 (пропуск значения) и 0.27 (потеря условия), см.
``reports/CORPUS_REPORT.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spanverify.coverage import (  # noqa: E402
    MAX_OMITTED_REPORTS,
    coverage_report,
    extract_salient_facts,
)

DOCUMENT = (
    "4. Срок хранения первичных учётных документов составляет 5 лет. "
    "5. Срок хранения вторичных учётных документов составляет 10 лет. "
    "6. Запрещается выносить учётные документы за пределы архивохранилища. "
    "7. Срок хранения кадровых документов составляет 75 лет, если документы созданы до 2003 года."
)


def test_extract_salient_facts_finds_measurements_and_prohibitions() -> None:
    """Из документа извлекаются измерения и запреты, реквизиты — нет."""
    facts = extract_salient_facts(DOCUMENT)
    kinds = [fact.kind for fact in facts]
    assert kinds.count("измерение") >= 3
    assert "запрет" in kinds
    values = {fact.value for fact in facts}
    assert "5|год" in values and "10|год" in values
    conditional = next(fact for fact in facts if fact.condition == "если")
    assert "кадров" in " ".join(conditional.distinctive)


def test_covered_answer_gives_no_findings() -> None:
    """Ответ, приводящий верное значение, замечаний не получает."""
    report = coverage_report(DOCUMENT, "Срок хранения первичных учётных документов составляет 5 лет.")
    assert report.counts["distorted"] == 0
    assert report.counts["omitted"] == 0
    assert report.risk == 0.0


def test_missing_value_is_detected_and_localized() -> None:
    """Субъект назван, значение опущено → статус omitted и фрагмент в ответе."""
    answer = "Срок хранения первичных учётных документов установлен."
    report = coverage_report(DOCUMENT, answer)
    assert report.counts["omitted"] >= 1
    assert report.risk > 0.0
    span = report.spans[0]
    assert span["status"] == "omitted"
    assert answer[span["start"] : span["end"]]
    assert "документ" in answer[span["start"] : span["end"]].lower()


def test_wrong_value_is_distorted() -> None:
    """Назван субъект, но значение другое → distorted с высокой значимостью."""
    report = coverage_report(DOCUMENT, "Срок хранения первичных учётных документов составляет 30 лет.")
    assert report.counts["distorted"] >= 1
    assert report.risk >= 0.75


def test_lost_condition_is_partial() -> None:
    """Значение верное, но условие «если …» потеряно → distorted (partial)."""
    report = coverage_report(DOCUMENT, "Срок хранения кадровых документов составляет 75 лет.")
    reasons = [match.reason for match in report.flagged]
    assert any("условие" in reason for reason in reasons)
    assert report.counts["distorted"] >= 1


def test_condition_kept_is_covered() -> None:
    """Условие на месте → расхождения нет."""
    answer = "Срок хранения кадровых документов составляет 75 лет, если документы созданы до 2003 года."
    report = coverage_report(DOCUMENT, answer)
    assert report.counts["distorted"] == 0


def test_absent_subject_is_not_reported() -> None:
    """Ответ на часть документа не наказывается: отсутствие субъекта — не замечание."""
    report = coverage_report(DOCUMENT, "Порядок уничтожения определяется локальной комиссией организации.")
    assert report.counts["omitted"] == 0
    assert report.counts["absent"] >= 1
    assert report.spans == []


def test_subject_without_value_is_omitted() -> None:
    """Субъект назван в утверждении без значения — это и есть тип ``missing``."""
    report = coverage_report(DOCUMENT, "Срок хранения вторичных документов установлен.")
    assert report.counts["omitted"] >= 1
    assert any(match.fact.value == "10|год" for match in report.flagged)


def test_plain_mention_without_claim_is_not_reported() -> None:
    """Упоминание темы без утверждения факта замечанием не считается."""
    report = coverage_report(DOCUMENT, "Для вторичных документов действует отдельный порядок уничтожения.")
    assert report.counts["omitted"] == 0


def test_prohibition_lost_modality_is_reported() -> None:
    """Потеря запрета («запрещается» → описательное предложение) — опущение."""
    document = "Запрещается выносить учётные документы за пределы архивохранилища."
    answer = "Учётные документы хранятся в архивохранилище организации."
    report = coverage_report(document, answer)
    assert report.counts["omitted"] >= 1
    assert any(match.fact.kind == "запрет" for match in report.flagged)


def test_omitted_findings_are_limited() -> None:
    """На документе из сотен фактов ответ получает не больше лимита замечаний."""
    sentences = [
        f"Статья {index}. Срок хранения документов вида {index} составляет {index + 1} лет." for index in range(1, 25)
    ]
    document = " ".join(sentences)
    report = coverage_report(document, "Порядок хранения документов определяется локальным актом организации.")
    assert report.counts["omitted"] <= MAX_OMITTED_REPORTS
    assert report.counts["facts"] >= 20


def test_report_as_dict_is_json_serializable() -> None:
    """Отчёт покрытия сериализуется без потерь (используется в API и отчётах)."""
    import json

    payload = coverage_report(DOCUMENT, "Срок хранения первичных учётных документов установлен.").as_dict(
        include_facts=True
    )
    text = json.dumps(payload, ensure_ascii=False)
    assert "omitted" in text and "counts" in text
