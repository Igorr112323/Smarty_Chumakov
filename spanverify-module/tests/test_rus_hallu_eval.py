"""Тесты оценки на внешнем корпусе: структура отчёта и запрет выдуманных чисел.

Оценка на реальных 1000 парах требует скачанных данных (в CI их нет — лицензия не
подтверждена). Здесь проверяется сама логика на маленьком локальном наборе: что
обе группы метрик считаются, что предсказания сравниваются с разметкой и что в
отчёте нет сравнения с baseline, пока числа статьи не извлечены.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.rus_hallu_eval import evaluate, true_span_texts


def _write_pairs(path: Path) -> Path:
    """Написать маленький набор в формате SpanVerify (как после загрузчика)."""
    answer = "Срок хранения первичных документов составляет десять лет."
    start = answer.index("десять лет")
    pairs = [
        {
            "id": "rushallu-sberquad-1",
            "context": "Согласно регламенту, срок хранения первичных документов составляет пять лет.",
            "answer": answer,
            "labels": [[start, start + len("десять лет"), 1]],
            "meta": {"dataset_version": "test", "citation": "тест", "types": ["Contradiction"]},
        },
        {
            "id": "rushallu-sberquad-2",
            "context": "Срок хранения составляет пять лет.",
            "answer": "Срок хранения составляет пять лет.",
            "labels": [],
            "meta": {"dataset_version": "test", "citation": "тест", "types": []},
        },
    ]
    path.write_text("".join(json.dumps(pair, ensure_ascii=False) + "\n" for pair in pairs), encoding="utf-8")
    return path


def test_true_span_texts_extract_exact_substrings(tmp_path: Path) -> None:
    """Истинные спаны восстанавливаются из смещений: срез ответа равен метке."""
    path = _write_pairs(tmp_path / "pairs.jsonl")
    pair = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    texts = true_span_texts(pair)
    assert texts == ["десять лет"]
    start, end, _ = pair["labels"][0]
    assert pair["answer"][start:end] == texts[0]


def test_evaluate_reports_both_metric_groups(tmp_path: Path) -> None:
    """Отчёт содержит наши метрики, их метрики и честную оговорку про demo-режим."""
    _write_pairs(tmp_path / "pairs.jsonl")
    report = evaluate(tmp_path, mode="demo")
    assert report["pairs"] == 2
    assert set(report["our_metrics"]) == {"tokens", "spans", "answers"}
    assert set(report["their_metrics"]) >= {"rouge1", "rougeL", "accuracy", "jaccard_score", "hamming_loss"}
    assert report["their_metrics"]["pairs"] == 2
    assert "научным результатом не являются" in report["disclaimer"]


def test_baseline_comparison_is_not_invented(tmp_path: Path) -> None:
    """Сравнение с опубликованной работой либо снабжено источником, либо равно null.

    Третьего не дано: число без ссылки на то, откуда оно взято, в отчёт попасть не
    должно. Раньше тест фиксировал только случай «не извлечено»; после того как
    извлечение из PDF заработало, проверяется и второй случай.
    """
    _write_pairs(tmp_path / "pairs.jsonl")
    report = evaluate(tmp_path, mode="demo")
    comparison = report["baseline_comparison"]
    if comparison is None:
        assert "не извлечены" in report["baseline_note"]
        return
    assert comparison["article_url"].startswith("http"), "у числа обязан быть источник"
    assert comparison["note"], "обязано быть сказано, откуда взяты числа"
    assert comparison["tables"], "сравнение без таблиц статьи бессмысленно"


def test_evaluate_handles_clean_pairs_without_labels(tmp_path: Path) -> None:
    """Чистые пары без разметки не ломают расчёт их метрик (нет деления на ноль)."""
    _write_pairs(tmp_path / "pairs.jsonl")
    report = evaluate(tmp_path, mode="demo")
    assert report["their_metrics"]["empty_reference"] == 1
    assert report["their_metrics"]["hamming_loss"] >= 0.0


def test_missing_dataset_raises_with_hint(tmp_path: Path) -> None:
    """Без скачанных данных оценка подсказывает, какой командой их получить."""
    import pytest

    with pytest.raises(FileNotFoundError, match="fetch_rushallu"):
        evaluate(tmp_path / "нет-такого", mode="demo")
