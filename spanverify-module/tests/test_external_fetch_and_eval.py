"""Тесты загрузчика внешних наборов и оценки на них.

Сеть в тестах не используется: проверяется то, что определяет корректность
результата — реестр (что берём и что исключено), реакция на расхождение хеша,
сборка адаптированных файлов из фикстур и сами прогоны оценки (метрики считаются,
происхождение разметки попадает в отчёт, срез помечается, baseline не выдумывается).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import fetch_external_tests as fetcher
from scripts.external_eval import compact_metrics, merge_into_combined, run
from scripts.make_external_report import render

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _torch_available() -> bool:
    """Есть ли torch: без него режим hf запустить нельзя, и это не ошибка теста."""
    import importlib.util

    return importlib.util.find_spec("torch") is not None


@pytest.fixture(scope="module")
def adapted_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Собрать маленький адаптированный набор из фикстур (как это делает загрузчик)."""
    from spanverify.external_datasets import ragtruth_pairs

    out = tmp_path_factory.mktemp("external")
    adapted = out / "adapted"
    adapted.mkdir(parents=True)
    responses = [
        json.loads(line)
        for line in (FIXTURES / "ragtruth_response_50.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sources = [
        json.loads(line)
        for line in (FIXTURES / "ragtruth_source_50.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pairs, _stats = ragtruth_pairs(responses, sources, task="QA", split="test")
    (adapted / "ragtruth_qa_test.jsonl").write_text(
        "".join(json.dumps(pair, ensure_ascii=False) + "\n" for pair in pairs), encoding="utf-8"
    )
    return out


# ------------------------------------------------------------------- реестр


def test_registry_contains_control_numbers() -> None:
    """Контрольные числа набора зафиксированы в реестре: это эталон самопроверки."""
    controls = fetcher.RAGTRUTH["controls"]
    assert controls["responses"] == 17790
    assert controls["sources"] == 2965
    assert controls["spans"] == 14289
    assert controls["verified_spans"] == 14289
    assert controls["test_qa_good"] == {
        "responses": 875,
        "with_hallucination": 160,
        "clean": 715,
        "spans": 235,
        "sources": 150,
    }
    assert fetcher.RUSHALLU["controls"] == {
        "responses": 1000,
        "clean": 667,
        "with_hallucination": 333,
        "spans": 423,
    }


def test_registry_records_excluded_datasets_with_reasons() -> None:
    """Отказ от наборов зафиксирован в коде с причиной, а не остался в переписке."""
    excluded = fetcher.EXCLUDED
    for name in ("HaDes", "HDM-Bench", "FAVA", "AlsKozlov/legalbench-ru", "Roflmax/Ru-Legal-QA-v1"):
        assert name in excluded and excluded[name]
    assert "reference-free" in excluded["HaDes"]
    assert "некоммерческая" in excluded["HDM-Bench"]


def test_registry_marks_non_human_label_origin() -> None:
    """Вспомогательные наборы помечены как llm/auto: их нельзя выдать за человеческие."""
    assert fetcher.OPTIONAL["halueval_llm_spans"]["label_origin"] == "llm"
    assert fetcher.OPTIONAL["lettucedetect"]["label_origin"] == "auto"
    assert fetcher.RAGTRUTH["label_origin"] == "human"
    assert "MIT" in fetcher.RAGTRUTH["license"]
    assert "не подтверждена" in fetcher.RUSHALLU["license"]


def test_revisions_are_pinned() -> None:
    """Ревизии закреплены, а хеши файлов записаны в реестр (а не вычисляются «на лету»)."""
    assert len(fetcher.RAGTRUTH["revision"]) == 40
    assert len(fetcher.RUSHALLU["revision"]) == 40
    for spec in fetcher.RAGTRUTH["files"].values():
        assert len(spec["sha256"]) == 64
        assert spec["size"] > 0


# ------------------------------------------------------------------ загрузка


def test_hash_mismatch_stops_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Расхождение SHA256 останавливает загрузку: данные изменились на стороне источника."""
    payload = b"{}\n"

    def fake_download(repo: str, revision: str, path: str, dest: Path, transport: str = "auto") -> str:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        return "stub"

    monkeypatch.setattr(fetcher, "download_github_file", fake_download)
    with pytest.raises(fetcher.FetchError, match="SHA256"):
        fetcher.fetch_ragtruth(tmp_path, transport="gh", verify=True)


def test_control_number_mismatch_is_loud(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Несовпадение контрольных чисел тоже останавливает загрузку.

    Хеши фикстур отличаются от закреплённых (они для полного набора), поэтому в
    реестре подменяются и они — иначе тест падал бы на SHA256 и не дошёл до
    проверки контрольных чисел, которую он и проверяет.
    """
    import hashlib

    def digest_of(name: str) -> str:
        """SHA256 файла-фикстуры (фактические байты)."""
        return hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest()

    files = {
        "response.jsonl": {
            **fetcher.RAGTRUTH["files"]["response.jsonl"],
            "sha256": digest_of("ragtruth_response_50.jsonl"),
        },
        "source_info.jsonl": {
            **fetcher.RAGTRUTH["files"]["source_info.jsonl"],
            "sha256": digest_of("ragtruth_source_50.jsonl"),
        },
    }
    monkeypatch.setattr(
        fetcher,
        "RAGTRUTH",
        {
            **fetcher.RAGTRUTH,
            "files": files,
            "controls": {**fetcher.RAGTRUTH["controls"], "responses": 1},
        },
    )
    called = {"n": 0}

    def fake_download(repo: str, revision: str, path: str, dest: Path, transport: str = "auto") -> str:
        called["n"] += 1
        source = FIXTURES / ("ragtruth_response_50.jsonl" if "response" in path else "ragtruth_source_50.jsonl")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes())
        return "stub"

    monkeypatch.setattr(fetcher, "download_github_file", fake_download)
    with pytest.raises(fetcher.FetchError, match="контрольные числа"):
        fetcher.fetch_ragtruth(tmp_path, transport="gh", verify=True)


# -------------------------------------------------------------------- оценка


def test_evaluate_accepts_adapted_file(adapted_dir: Path) -> None:
    """Адаптированный файл принимается ядром без правок: метрики считаются."""
    report = run("ragtruth", "qa", "test", "demo", None, None, adapted_dir)
    assert report["pairs"] == 50
    assert report["label_origin"] == {"human": 50}
    assert report["spans_reference"] == 36
    for key in ("precision", "recall", "f1", "fpr"):
        assert isinstance(report["our_metrics"]["tokens"][key], float)
    assert report["their_metrics"]["pairs"] == 50
    assert report["baseline"]["extracted"] is False
    assert "не извлечён" in report["baseline"]["note"]


def test_evaluate_marks_slice_as_slice(adapted_dir: Path) -> None:
    """Срез помечается: часть набора нельзя выдавать за весь набор."""
    report = run("ragtruth", "qa", "test", "demo", None, 10, adapted_dir)
    assert report["pairs"] == 10
    assert report["pairs_total_in_file"] == 50
    assert "ЭТО СРЕЗ" in report["note"]


def test_evaluate_warns_about_demo_mode(adapted_dir: Path) -> None:
    """Числа demo-режима идут вместе с оговоркой, что это не научный результат."""
    report = run("ragtruth", "qa", "test", "demo", None, None, adapted_dir)
    assert "научным результатом не являются" in report["disclaimer"]
    # Модель задаётся ключом --model, а значение по умолчанию не меняется. Сам
    # прогон требует весов и torch, поэтому проверяем то, что от них не зависит.
    from spanverify.engine import Verifier

    assert Verifier(mode="hf", model_name="BAAI/bge-m3").model_name == "BAAI/bge-m3"
    default = Verifier(mode="demo").model_name
    assert default == "ai-forever/rugpt3small_based_on_gpt2", "значение по умолчанию менять нельзя"
    report_hf = run("ragtruth", "qa", "test", "hf", "BAAI/bge-m3", 2, adapted_dir) if _torch_available() else None
    if report_hf is not None:
        assert report_hf["mode"] == "hf"
        assert report_hf["model"] == "BAAI/bge-m3"


def test_missing_file_gives_hint(tmp_path: Path) -> None:
    """Без скачанных данных понятно, какой командой их получить."""
    with pytest.raises(FileNotFoundError, match="fetch_external_tests"):
        run("ragtruth", "qa", "test", "demo", None, None, tmp_path)


def test_combined_file_keeps_runs_separate(adapted_dir: Path, tmp_path: Path) -> None:
    """Прогоны не перетирают друг друга: каждый лежит под своим ключом."""
    combined = tmp_path / "external_tests.json"
    qa = run("ragtruth", "qa", "test", "demo", None, None, adapted_dir)
    merge_into_combined(qa, combined)
    merged = merge_into_combined(qa, combined)
    assert list(merged["runs"]) == ["ragtruth_qa_test_demo"]
    assert merged["rule"].startswith("внешние наборы — только тест")


def test_compact_metrics_shape(adapted_dir: Path) -> None:
    """Сжатие метрик сохраняет все группы: токены, фрагменты, ответы, вердикты."""
    from spanverify.dataset import read_pairs
    from spanverify.engine import Verifier

    pairs = list(read_pairs(adapted_dir / "adapted" / "ragtruth_qa_test.jsonl"))
    verifier = Verifier(mode="demo")
    metrics = compact_metrics(verifier.evaluate(pairs))
    assert set(metrics) == {"tokens", "spans", "answers", "verdicts"}
    verdicts = metrics["verdicts"]
    assert verdicts["tp"] + verdicts["fp"] + verdicts["fn"] + verdicts["tn"] == 50


# -------------------------------------------------------------------- отчёт


def test_report_table_contains_origin_and_numbers(adapted_dir: Path, tmp_path: Path) -> None:
    """Отчёт содержит колонку происхождения разметки и подписанные числа метрик."""
    combined_path = tmp_path / "external_tests.json"
    report = run("ragtruth", "qa", "test", "demo", None, None, adapted_dir)
    merge_into_combined(report, combined_path)
    combined = json.loads(combined_path.read_text(encoding="utf-8"))
    text = render(combined, {"datasets": {"ragtruth": {}, "rushallu": {}}})
    assert "| Разметка |" in text
    assert "human" in text
    assert "token F1" in text and "FPR" in text and "AUC" in text
    assert "не извлечено" in text
    assert "Режим `hf` **не выполнялся**" in text
    assert "Правило: внешние наборы — только тест" in text


def test_report_without_runs_is_honest(tmp_path: Path) -> None:
    """Пустой файл прогонов не превращается в таблицу с выдуманными числами."""
    text = render({"runs": {}}, None)
    assert "Прогонов нет" in text
    assert "0.0." not in text


# ------------------------------------------------------- аннотации CI


def test_annotation_prints_control_numbers_and_metrics(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Аннотации содержат контрольные числа и метрики: их видно без скачивания артефактов."""
    import subprocess
    import sys

    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "datasets": {
                    "ragtruth": {
                        "revision": "c103204b9ce2",
                        "files": {"response.jsonl": {"verified": True}},
                        "totals": {"responses": 17790, "sources": 2965, "spans": 14289, "verified_spans": 14289},
                    },
                    "rushallu": {
                        "revision": "345907f983da",
                        "files": {"sberquad-rag.csv": {"verified": True}},
                        "totals": {"responses": 1000, "clean": 667, "with_hallucination": 333, "spans": 423},
                    },
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    combined = tmp_path / "external_tests.json"
    combined.write_text(
        json.dumps(
            {
                "runs": {
                    "ragtruth_qa_test_demo": {
                        "pairs": 875,
                        "mode": "demo",
                        "label_origin": {"human": 875},
                        "our_metrics": {"tokens": {"f1": 0.2045, "fpr": 0.1685, "auc": 0.7167}},
                        "their_metrics": {"accuracy": 0.0103, "rougeL": 0.0718},
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "print_external_annotation.py"),
            "--manifest",
            str(manifest),
            "--combined",
            str(combined),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "ответов=17790" in result.stdout
    assert "спанов=14289" in result.stdout
    assert "sha256=совпали" in result.stdout
    assert "token_F1=0.2045" in result.stdout
    assert "разметка=human" in result.stdout


def test_annotation_warns_without_files(tmp_path: Path) -> None:
    """Если манифеста и прогонов нет — предупреждения, а не выдуманные нули."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "print_external_annotation.py"),
            "--manifest",
            str(tmp_path / "нет.json"),
            "--combined",
            str(tmp_path / "нет2.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "::warning" in result.stdout
