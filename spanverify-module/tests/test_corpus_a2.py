"""Тесты корпуса A2 (естественные ответы модели).

Запуск генерации требует GPU и весов, поэтому в CI проверяется подготовительная
часть: разбор фактов из документа, вопросы и промты, режим ``--dry-run`` и то, что
черновые метки никогда не выдаются за истину (``needs_expert_review``).
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.build_corpus_a2 import (
    build_prompt,
    build_question,
    collect_tasks,
    draft_labels_with_verifier,
    read_facts,
    write_outputs,
)

DOCUMENT = (
    "Регламент учёта № 1. Настоящий документ определяет правила учёта и хранения.\n\n"
    "Для первичных учётных документов срок хранения составляет пять лет.\n\n"
    "Для счёта-фактуры предельный объём одного вложения составляет десять мегабайт если иное не установлено договором."
)


def test_read_facts_extracts_subject_feature_value() -> None:
    """Из документа вытаскиваются субъект, признак и значение каждого факта."""
    facts = read_facts(DOCUMENT)
    assert len(facts) == 2
    assert facts[0]["subject"] == "первичных учётных документов"
    assert facts[0]["feature"] == "срок хранения"
    assert facts[0]["value"] == "пять лет"
    assert facts[1]["feature"] == "предельный объём одного вложения"


def test_read_facts_splits_condition() -> None:
    """Условие отделяется от значения: иначе модель получила бы склейку двух смыслов."""
    facts = read_facts(DOCUMENT)
    assert facts[1]["condition"] == "если иное не установлено договором"
    assert facts[1]["value"] == "десять мегабайт"


def test_question_wording_matches_feature() -> None:
    """Вопрос собирается по признаку факта, а не одним шаблоном на всё."""
    facts = read_facts(DOCUMENT)
    assert build_question(facts[0]) == "Какой срок хранения установлен для первичных учётных документов?"
    assert "объём" in build_question(facts[1]).lower()


def test_prompt_requires_answer_from_document_only() -> None:
    """Промт запрещает отвечать по памяти: иначе ошибки будут не про документ."""
    prompt = build_prompt(DOCUMENT, "Какой срок хранения установлен?")
    assert "строго по документу" in prompt
    assert "в документе нет ответа" in prompt
    assert prompt.endswith("Ответ:")


def test_collect_tasks_is_deterministic_and_limited(tmp_path: Path) -> None:
    """Задания собираются детерминированно и ограничиваются ключом --limit."""
    (tmp_path / "doc-0001.md").write_text(DOCUMENT, encoding="utf-8")
    (tmp_path / "doc-0002.md").write_text(DOCUMENT.replace("№ 1", "№ 2"), encoding="utf-8")
    first = collect_tasks(tmp_path, limit=3, seed=42)
    second = collect_tasks(tmp_path, limit=3, seed=42)
    assert len(first) == 3
    assert [task["prompt"] for task in first] == [task["prompt"] for task in second]
    assert all(task["question"] for task in first)


def test_dry_run_writes_tasks_without_claiming_metrics(tmp_path: Path) -> None:
    """Черновой прогон пишет задания и пустые пары: числа «hf» не выдумываются."""
    (tmp_path / "docs" / "generated").mkdir(parents=True)
    (tmp_path / "docs" / "generated" / "doc-0001.md").write_text(DOCUMENT, encoding="utf-8")
    tasks = collect_tasks(tmp_path / "docs", limit=10, seed=1)
    out = tmp_path / "a2"
    write_outputs(out, tasks, [], {"dry_run": True, "model": "test"})
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dry_run"] is True
    assert manifest["pairs"] == 0
    assert manifest["status"] == "draft"
    assert manifest["needs_expert_review"] is True
    assert (out / "pairs_draft.jsonl").read_text(encoding="utf-8") == ""


def test_draft_labels_are_marked_for_expert_review() -> None:
    """Черновая разметка помечена как предварительная: она не становится истиной."""
    pairs = [
        {
            "id": "corpus-a2-00001",
            "context": "Согласно регламенту, срок хранения первичных документов составляет пять лет. "
            "Срок хранения вторичных документов составляет десять лет.",
            "answer": "Срок хранения первичных документов составляет десять лет.",
            "labels": [],
            "meta": {"kind": "corpus_a2"},
        }
    ]
    drafted = draft_labels_with_verifier(pairs, mode="demo")
    assert drafted[0]["meta"]["draft_verdict"] in {"doubtful", "likely_hallucination"}
    assert drafted[0]["labels"], "черновая разметка обязана найти подмену числа"
    text = drafted[0]["answer"]
    for start, end, label in drafted[0]["labels"]:
        assert label == 1
        assert 0 <= start < end <= len(text)
    assert drafted[0]["meta"].get("needs_expert_review") is None  # ставится генератором, не разметкой


def test_missing_documents_directory_reports_error(tmp_path: Path) -> None:
    """Отсутствие каталога документов — понятная ошибка, а не пустой корпус."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "build_corpus_a2.py"),
            "--docs",
            str(tmp_path / "нет"),
            "--out",
            str(tmp_path / "out"),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 2
    assert "нет каталога документов" in result.stderr
