"""Тесты вспомогательных скриптов: искажения, испытания, документы, РИД, аналоги.

Проверяется то, что легко сломать незаметно: детерминированность искажений,
отсутствие выдуманных чисел в отчётах, наличие пометки «не подано» в РИД-заявках
и устойчивость обзора аналогов к недоступности источника.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import make_gost_docs  # noqa: E402
import make_rid_package  # noqa: E402
import prior_art_search  # noqa: E402
import robustness  # noqa: E402


def test_paraphrase_and_number_words_change_answer() -> None:
    """Искажения меняют текст детерминированно и не ломают Unicode."""
    answer = "Срок хранения документов составляет 5 лет, в течение года запрещается вынос."
    rng = random.Random(1)
    paraphrased = robustness.paraphrase(answer, rng)
    assert paraphrased != answer
    assert "год" in paraphrased or "период" in paraphrased

    words = robustness.number_words(answer)
    assert "пять лет" in words, "число должно стать словом"

    omitted = robustness.omission(answer)
    assert "5 лет" not in omitted and len(omitted) < len(answer)


def test_numeric_substitution_takes_value_from_context() -> None:
    """Числовая подмена берёт значение из другого факта документа."""
    context = "Срок хранения первичных документов 5 лет. Срок хранения вторичных документов 10 лет."
    answer = "Срок хранения первичных документов 5 лет."
    substituted = robustness.numeric_substitution(answer, context, random.Random(2))
    assert substituted != answer
    assert "10" in substituted


def test_contradiction_flips_modality_or_value() -> None:
    """Противоречие либо переворачивает модальность, либо меняет число."""
    assert "не более" in robustness.contradiction("Хранить не менее 5 лет.")
    assert robustness.contradiction("Срок составляет 5 лет.") != "Срок составляет 5 лет."


def test_render_robustness_report_has_no_invented_metrics() -> None:
    """Отчёт по искажениям не подставляет числа: нет измерения — null и причина."""
    report = {
        "dataset": "d.jsonl",
        "mode": "hf",
        "model": "m",
        "seed": 42,
        "pairs": 10,
        "clean_pairs": 10,
        "document_groups": 3,
        "baseline": {"token_f1": 0.5, "verdict_f1": 0.7, "verdict_fpr": 0.1},
        "distortions": [
            {
                "name": "числовая подмена",
                "pairs": 10,
                "changed_answers": 8,
                "flagged_share_before": 0.1,
                "flagged_share_after": 0.6,
                "delta": 0.5,
                "token_f1_before": 0.5,
                "token_f1_after": None,
                "token_f1_after_reason": "разметка исходного ответа к искажённому тексту не переносится",
            }
        ],
        "duration_s": 1.0,
    }
    text = robustness.render(report)
    assert "числовая подмена" in text
    assert "не переносится" in text
    assert "0.6" in text


def test_gost_documents_use_null_instead_of_guessing(tmp_path: Path) -> None:
    """ГОСТ-документы: отсутствие метрик даёт «нет данных», а не придуманное число."""
    code = make_gost_docs.main(["--out", str(tmp_path), "--metrics", str(tmp_path / "нет.json")])
    assert code == 0
    protocol = (tmp_path / "PROTOKOL_ispytaniy.md").read_text(encoding="utf-8")
    assert "нет данных" in protocol
    assert "не является официальной формой" in protocol
    for name in ("AKT_gotovnosti.md", "VEDOMOST_dokumentov.md"):
        assert (tmp_path / name).is_file()


def test_gost_documents_use_real_metrics_when_present(tmp_path: Path) -> None:
    """Если METRICS.json есть, его значения попадают в протокол как есть."""
    metrics = {"corpora": {"A1": {"modes": {"demo": {"verdicts": {"f1": 0.9, "fpr": 0.1}}}}}}
    metrics_path = tmp_path / "METRICS.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    functional = {
        "environment": {"os": "Linux", "python": "3.11", "cpu": "x86", "cpu_count": 2, "ram_total_mb": 3900},
        "smoke": {"cold_start_seconds": 0.5, "fields_present": {"score": True}},
        "max_length": {"max_ok_chars": 10000},
        "leak": {"growth_mb_per_minute": 0.1},
    }
    functional_path = tmp_path / "functional.json"
    functional_path.write_text(json.dumps(functional), encoding="utf-8")
    assert (
        make_gost_docs.main(
            ["--out", str(tmp_path / "out"), "--metrics", str(metrics_path), "--functional", str(functional_path)]
        )
        == 0
    )
    protocol = (tmp_path / "out" / "PROTOKOL_ispytaniy.md").read_text(encoding="utf-8")
    assert "A1 / demo" in protocol and "0.9" in protocol and "Linux" in protocol


def test_rid_package_is_draft_and_keeps_human_fields_empty(tmp_path: Path) -> None:
    """РИД-комплект: пометка «не подано», поля правообладателя пустые."""
    assert make_rid_package.main(["--out", str(tmp_path)]) == 0
    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert index["status"] == "draft_not_submitted"
    assert "правообладатель" in index["human_fields_required"]
    program = (tmp_path / "ZAYAVKA_programma_EVM.md").read_text(encoding="utf-8")
    assert "НЕ ПОДАНО" in program
    assert "| Правообладатель | ______________________ |" in program


def test_prior_art_markdown_reports_source_failures() -> None:
    """Недоступный источник попадает в отчёт с кодом ошибки, без выдумок."""
    report = {
        "generated_at": "2026-01-01T00:00:00Z",
        "note": "как есть",
        "topics": {
            "тема": {
                "query": "q",
                "openalex": {"status": None, "error": "URLError: EOF", "url": "u", "found": []},
                "crossref": {
                    "status": 200,
                    "error": None,
                    "url": "u",
                    "found": [{"title": "T", "year": 2024, "url": "x"}],
                },
                "github": {"status": 200, "error": None, "url": "u", "found": []},
                "patents": {"status": 403, "error": "HTTP 403", "url": "u", "found": []},
            }
        },
    }
    text = prior_art_search.render_markdown(report)
    assert "URLError: EOF" in text and "HTTP 403" in text
    assert "T 2024" in text


def test_prior_art_get_json_handles_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ошибка сети не поднимается наружу: возвращается словарь с текстом ошибки."""

    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("сеть недоступна")

    monkeypatch.setattr(prior_art_search.urllib.request, "urlopen", boom)
    result = prior_art_search.get_json("https://example.invalid/api")
    assert result["status"] is None
    assert "сеть недоступна" in str(result["error"])
    assert result["data"] is None
