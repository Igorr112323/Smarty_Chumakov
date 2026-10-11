"""Независимость выборки v2 и запрет подмены данных после заморозки.

Главное отличие цикла v2 от v1 — не модель и не признаки, а то, что тест обязан
быть *ни разу не виденным*. Если это нарушено, всё измерение превращается в
повторный замер на тренировочных данных, и ни одна метрика уже не значит ничего.
Поэтому проверки здесь жёсткие и работают в обе стороны: они ловят и утечку, и
«зелёну» без данных.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import hf_protocol as v1  # noqa: E402
import hf_protocol_v2 as v2  # noqa: E402

CORPUS = ROOT / "data" / "corpus_v2"
SPLITS = CORPUS / "splits"
CORPUS_MANIFEST = CORPUS / "manifest.json"
SOURCES_MANIFEST = CORPUS / "sources" / "sources.json"
FREEZE = ROOT / "reports" / "v2" / "freeze.json"
MANIFEST = ROOT / "reports" / "v2" / "manifest.json"


def test_absent_corpus_means_absent_measurements() -> None:
    """Если корпуса нет — ни манифеста шага 3, ни замороженной конфигурации быть не может."""
    if SPLITS.is_dir() and (SPLITS / "test.jsonl").is_file():
        return
    assert not MANIFEST.is_file(), "числа на тесте появились раньше тестовой выборки"
    assert not (
        ROOT / "config" / "hf_final_config_v2.json"
    ).is_file(), "конфигурация шага 3 заморожена до сборки выборки: отбирать на val нечего"


def test_v2_is_disjoint_from_every_v1_corpus() -> None:
    if not (SPLITS / "test.jsonl").is_file():
        # Корпуса ещё нет: «зелёным» это не делает цикл (за отсутствием чисел следит
        # test_absent_corpus_means_absent_measurements), но и в красное превращать
        # проверку несуществующих данных нельзя — иначе она перестанет что-то значить.
        return
    report = v2.independence_report(ROOT, SPLITS)
    assert report["v2_documents"] > 0, "в выборке нет документов"
    offenders = {name: block["overlap"] for name, block in report["v1_universes"].items() if block["overlap"]}
    assert not offenders, f"утечка в корпуса v1: {offenders}"


def test_v2_documents_are_never_shared_between_parts() -> None:
    if not (SPLITS / "test.jsonl").is_file():
        return
    _info, problems = v2.audit_sizes(v2.parts_of(SPLITS))
    assert not [item for item in problems if "пересекаются" in item], problems


def test_split_hashes_match_the_freeze_record() -> None:
    """Подмена любого файла разбиения после шага 0 обязана падать, а не измеряться."""
    if not (SPLITS / "test.jsonl").is_file():
        return
    assert FREEZE.is_file(), "шаг 0 не зафиксирован: сравнивать хеши не с чем"
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    recorded = freeze.get("splits_sha256") or {}
    for part, path in v2.parts_of(SPLITS).items():
        assert path.is_file(), f"нет файла {path}"
        assert recorded.get(part) == v1.sha256_file(path), f"{part}: файл разбиения изменён после заморозки"


def test_final_manifest_pins_the_same_test_split() -> None:
    if not MANIFEST.is_file():
        return
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8")) if FREEZE.is_file() else {}
    for part, digest in (manifest.get("splits_sha256") or {}).items():
        assert digest, f"{part}: пустой sha256 в манифесте"
        if freeze.get("splits_sha256", {}).get(part):
            assert digest == freeze["splits_sha256"][part], f"{part}: манифест и заморозка разошлись"
        path = v2.parts_of(SPLITS)[part]
        if path.is_file():
            assert digest == v1.sha256_file(path), f"{part}: файл не тот, что измеряли"
    for name in ("predictions_test.jsonl.gz", "gold_test.jsonl.gz"):
        assert (MANIFEST.parent / name).is_file(), f"манифест есть, а {name} нет — числа не пересчитать"


def test_corpus_manifest_records_the_procedure_as_run() -> None:
    """Режим сбора и проверки генератора — в манифесте, а не в пересказе отчёта."""
    if not CORPUS_MANIFEST.is_file():
        return
    corpus = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
    assert int(corpus.get("shared_groups", -1)) == 0, "пары одного документа попали в разные части"
    for key in ("contexts_problems", "spans_problems", "number_attribution_problems", "plan_problems"):
        assert corpus.get(key) == [], f"{key}: генератор сообщил о проблемах — корпус непригоден"
    assert corpus.get("real_documents") is True
    if SOURCES_MANIFEST.is_file():
        sources = json.loads(SOURCES_MANIFEST.read_text(encoding="utf-8"))
        assert (
            sources.get("text_only") is False
        ), "корпус собран в режиме без распознавания: это другой корпус (см. docs/CORPUS_v2.md, раздел 3)"


def test_v2_hashed_files_keep_lf_in_checkout() -> None:
    """Ровно та ловушка, из-за которой windows-job в v1 падал лишними тестами.

    GitHub-раннер на windows приводит LF к CRLF, и `sha256_file` перестаёт
    совпадать с зафиксированным на Linux. Лечится атрибутом `eol=lf`, и тест
    следит, чтобы лечением не воспользовались «на потом».
    """
    attributes = (ROOT.parent / ".gitattributes").read_text(encoding="utf-8")
    assert "eol=lf" in attributes and "*.jsonl" in attributes
    for path in [
        FREEZE,
        ROOT / "docs" / "METRIC_SPEC_v2.md",
        ROOT / "docs" / "EXPERIMENTS_v2.md",
        ROOT / "config" / "hf_experiments_v2.json",
    ]:
        if path.is_file():
            assert b"\r" not in path.read_bytes(), f"{path.name}: в checkout у файла появились CR"
    for name in ("train.jsonl", "dev.jsonl", "val.jsonl", "test.jsonl"):
        path = SPLITS / name
        if path.is_file():
            assert b"\r" not in path.read_bytes(), f"{path.name}: CR меняет sha256 выборки"
