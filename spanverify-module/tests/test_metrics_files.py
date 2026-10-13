"""Единый файл чисел и его сверка с документами (исправление P0-3 аудита).

Смысл набора: числа в документах больше не берутся из памяти. Есть один файл
``reports/METRICS.json``, собранный прогоном кода, и есть проверка, которая
падает, если документ разошёлся с ним. Тесты ниже защищают это правило и
честные оговорки, которые нельзя потерять при правках.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parent
METRICS_PATH = MODULE_ROOT / "reports" / "METRICS.json"


def _load_script(name: str):
    """Загрузить скрипт из ``scripts/`` по пути (там нет пакета)."""
    path = MODULE_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


def _metrics() -> dict:
    """Прочитать единый файл чисел; без него проверки бессмысленны — пропуск."""
    if not METRICS_PATH.is_file():
        pytest.skip("reports/METRICS.json отсутствует: сначала scripts/collect_metrics.py")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def test_metrics_json_labels_demo_corpus_as_synthetic() -> None:
    """Оговорка о синтетике и требование реальных данных обязаны быть в файле чисел."""
    metrics = _metrics()
    disclaimer = metrics["disclaimer"]
    assert "синтетическ" in disclaimer.lower()
    assert "1200" in disclaimer, "нужно число пар для научного подтверждения"
    assert metrics["demo"]["demo_answer_example"]["verdict"] == "likely_hallucination"


def test_metrics_json_records_group_split_without_shared_subjects() -> None:
    """Сплит в отчётных числах обязан быть групповым и без общих субъектов (P1-3)."""
    metrics = _metrics()
    split = metrics["demo"]["split"]
    assert split["grouped"] is True
    assert split["shared_groups"] == 0
    assert split["train_pairs"] + split["test_pairs"] == 240


def test_metrics_json_keeps_cross_corpus_drop() -> None:
    """Падение качества на чужом генераторе нельзя спрятать (P0-2).

    Если кто-то удалит раздел или подменит числа, тест упадёт: перенос обязан
    показывать и падение F1, и выход FPR за порог 0.10.
    """
    metrics = _metrics()
    cross = metrics.get("cross_corpus")
    assert cross is not None, "кросс-корпусный тест обязан быть в файле чисел"
    assert cross["cross_corpus_f1"] < cross["in_corpus_f1"] - 0.05
    assert cross["cross_fpr"] > 0.10, "FPR на чужом корпусе обязан быть показан честно"


def test_metrics_json_exposes_strict_span_quality() -> None:
    """Строгая метрика фрагментов (узкая разметка) обязана публиковаться рядом с мягкой (P1-4)."""
    metrics = _metrics()
    for name in ("in_corpus", "whole_corpus"):
        spans = metrics["demo"][name]["spans"]
        assert 0.0 <= spans["strict_f1"] < spans["soft_f1"] <= 1.0, name
        assert spans["coverage"] >= 0.9, name


def test_check_numbers_reports_stale_value() -> None:
    """Проверка чисел действительно ловит расхождение, а не «всегда зелёная»."""
    checker = _load_script("check_numbers")
    metrics = _make_fake_metrics()
    stale = Path("FAKE_DOC.md")
    stale.write_text("token F1 0.959 при FPR 0.002\n", encoding="utf-8")
    try:
        problems = checker.check_docs([stale], metrics)
    finally:
        stale.unlink(missing_ok=True)
    assert any("token_f1" in problem for problem in problems)


def test_check_numbers_requires_itog_sections(tmp_path: Path) -> None:
    """ИТОГ.md обязан содержать разделы «заявлено/факт», «не проверено» и «3 шага» (приёмка E)."""
    checker = _load_script("check_numbers")
    (tmp_path / "ИТОГ.md").write_text("# Итог\n\nБез обязательных разделов.\n", encoding="utf-8")
    problems = " ".join(checker.check_required_sections(tmp_path))
    assert "заявлено / факт" in problems
    assert "не проверено" in problems
    assert "трёх шагов" in problems


def test_documents_match_metrics_file() -> None:
    """Сквозная проверка: README, ИТОГ и docs/ сходятся с reports/METRICS.json."""
    checker = _load_script("check_numbers")
    metrics = _metrics()
    docs = [REPO_ROOT / name for name in checker.DEFAULT_DOCS if (REPO_ROOT / name).is_file()]
    assert len(docs) >= 6, "проверять нужно все основные документы"
    assert checker.check_docs(docs, metrics) == []


def _full_suite_requested(config: pytest.Config) -> bool:
    """Запрошен ли полный прогон: аргументы равны ``testpaths``, нет ``-k`` и ``-m``.

    pytest без аргументов подставляет в ``config.args`` значения ``testpaths`` из
    ``pyproject.toml`` (``["tests"]``), поэтому «аргументов нет» — не признак
    полного прогона: сравнивать нужно с самими ``testpaths``. Прогон одного файла
    или отбор по ключевому слову/метке даёт заведомо другое число тестов — такие
    прогоны пропускаются, а не падают.
    """
    testpaths = [str(path) for path in (config.getini("testpaths") or [])]
    args = [str(arg) for arg in (config.args or [])]
    if args and args != testpaths:
        return False
    option = config.option
    return not (getattr(option, "keyword", "") or getattr(option, "markexpr", ""))


class _FakeOption:
    """Замена ``config.option``: только два поля, которые читает проверка прогона."""

    def __init__(self, keyword: str = "", markexpr: str = "") -> None:
        self.keyword = keyword
        self.markexpr = markexpr


class _FakeConfig:
    """Замена ``pytest.Config`` для проверки распознавания полного прогона."""

    def __init__(self, args: list[str], testpaths: tuple[str, ...] = ("tests",), **option: str) -> None:
        self.args = list(args)
        self.option = _FakeOption(**option)
        self._testpaths = list(testpaths)

    def getini(self, name: str) -> list[str]:
        """pytest спрашивает ``testpaths``; другие ключи здесь не используются."""
        assert name == "testpaths", name
        return list(self._testpaths)


def test_full_suite_detection_compares_args_with_testpaths() -> None:
    """Полный прогон распознаётся по совпадению аргументов с ``testpaths``.

    Проверка нужна из-за реальной ошибки: первая версия распознавания считала
    полным прогоном только пустые ``config.args``, но pytest подставляет туда
    ``testpaths`` (``["tests"]``), поэтому в настоящем полном прогоне обе
    сверки числа тестов молча пропускались. Прогон с явным путём, ``-k`` или
    ``-m`` обязан остаться подмножеством.
    """
    assert _full_suite_requested(_FakeConfig(["tests"])), "pytest без аргументов: args == testpaths"
    assert _full_suite_requested(_FakeConfig([])), "пустые аргументы при непустых testpaths"
    assert _full_suite_requested(_FakeConfig(["tests", "conftest.py"], testpaths=("tests", "conftest.py")))
    assert not _full_suite_requested(_FakeConfig(["tests/test_api.py"])), "явный путь — подмножество"
    assert not _full_suite_requested(_FakeConfig(["tests"], keyword="metrics")), "отбор -k — подмножество"
    assert not _full_suite_requested(_FakeConfig(["tests"], markexpr="slow")), "отбор -m — подмножество"


def test_metrics_file_is_not_behind_the_suite(request: pytest.FixtureRequest) -> None:
    """Файл чисел обязан совпадать с фактическим числом тестов этого прогона.

    Причина проверки: два ночных прогона ``ci`` на ``main`` подряд
    (37755784670, 37911250513) падали на шаге «Сверить числа в документах с
    METRICS.json». Job «Единый файл чисел» пересчитывает ``tests.collected``
    заново (``pytest --collect-only``), а закоммиченный файл и документы
    остались с прежним числом после слияния ветки, добавившей тесты. Локально
    расхождение не ловилось: сверка смотрела в устаревший файл.

    Здесь число берётся из самой сессии pytest (коллекция завершена до запуска
    тестов), поэтому проверка ничего не пересчитывает и не запускает второй
    pytest. Прогон подмножества тестов — не полный состав, он пропускается.
    """
    if not _full_suite_requested(request.config):
        pytest.skip("запрошено подмножество тестов: полный состав не собирался")
    collected = request.session.testscollected
    metrics = _metrics()
    tests_block = metrics.get("tests") or {}
    assert tests_block.get("collected") == collected, (
        f"reports/METRICS.json отстал от состава тестов: в файле {tests_block.get('collected')}, "
        f"собрано {collected}; пересоберите файл (scripts/collect_metrics.py) и обновите документы"
    )
    coverage = tests_block.get("coverage_percent")
    assert isinstance(coverage, (int, float)) and not isinstance(coverage, bool), (
        "в reports/METRICS.json нет измеренного покрытия (null): локальная сверка документов "
        "пропускает заявления о покрытии, а ночной CI, где покрытие измеряется, — нет"
    )


def test_document_test_claims_match_this_session(request: pytest.FixtureRequest) -> None:
    """Заявленное в документах число тестов совпадает с собранным в этом прогоне.

    Документы (README, ИТОГ, отчёт о НИР, руководство) называют число тестов
    текстом. Сверка с закоммиченным ``METRICS.json`` этого не ловит, если файл
    устарел вместе с документами, — поэтому число подставляется из текущей
    сессии pytest, а правила чтения строк берутся из ``scripts/check_numbers.py``
    (оговорки «устарело», «аннотации CI», пороги и критерии остаются допустимыми).
    """
    if not _full_suite_requested(request.config):
        pytest.skip("запрошено подмножество тестов: полный состав не собирался")
    collected = request.session.testscollected
    checker = _load_script("check_numbers")
    metrics = json.loads(json.dumps(_metrics()))
    metrics.setdefault("tests", {})["collected"] = collected
    docs = [REPO_ROOT / name for name in checker.DEFAULT_DOCS if (REPO_ROOT / name).is_file()]
    problems = [problem for problem in checker.check_docs(docs, metrics, repo_root=REPO_ROOT) if "tests=" in problem]
    assert problems == [], "документы заявляют число тестов, которого нет в этом прогоне:\n" + "\n".join(problems)


def test_pilot_publishes_control_thresholds_and_draws_png(tmp_path: Path) -> None:
    """Пилот обязан публиковать контроль корпуса (0.8/0.3) и рисовать PNG без зависимостей."""
    pilot = _load_script("pilot_rugpt3small")
    source = (MODULE_ROOT / "scripts" / "pilot_rugpt3small.py").read_text(encoding="utf-8")
    assert "grounded_copy_min" in source and "unsupported_copy_max" in source
    assert '"wording"' in source, "нужна формулировка «модель реальная, корпус синтетический»"

    payload = {
        "auc": {
            "last": {
                "attention_entropy": {"auc_oriented": 0.52},
                "ctx_attention_mass": {"auc_oriented": 0.48},
                "embedding_density": {"auc_oriented": 0.5},
            }
        }
    }
    png = pilot.write_auc_png(payload, tmp_path / "pilot.png")
    data = png.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(data) > 500, "картинка не должна быть пустышкой"


def _make_fake_metrics() -> dict:
    """Минимальный файл чисел для проверки самого чекера (значения нарочно «правильные»)."""
    return {
        "meta": {"version": "1.2.0"},
        "tests": {"collected": 226, "coverage_percent": 86.0},
        "demo": {
            "split": {"grouped": True, "shared_groups": 0, "train_pairs": 168, "test_pairs": 72},
            "in_corpus": {
                "tokens": {"f1": 0.953, "fpr": 0.002, "auc": 0.996, "n": 594},
                "spans": {"strict_f1": 0.28, "soft_f1": 1.0, "coverage": 1.0},
                "answers": {"f1": 1.0, "fpr": 0.0, "auc": 1.0},
            },
            "whole_corpus": {
                "tokens": {"f1": 0.951, "fpr": 0.005, "auc": 0.997, "n": 2001},
                "spans": {"strict_f1": 0.315, "soft_f1": 1.0, "coverage": 1.0},
                "answers": {"f1": 0.996, "fpr": 0.0, "auc": 0.996},
            },
            "validation": {"f1": 0.953, "fpr": 0.002, "auc": 0.996},
            "participation": {"auc_out_of_fold": 1.0, "rows": 8179},
        },
        "cross_corpus": {"in_corpus_f1": 0.953, "cross_corpus_f1": 0.731, "cross_fpr": 0.144},
        "pilot": None,
    }


def test_evaluate_reports_verdict_metrics() -> None:
    """В метриках есть уровень вердикта: правило привязки числа видно только там.

    Токен-уровень считается по сглаженной маске и редко реагирует на одиночную
    подмену числа, а вердикт ответа — реагирует (``doubtful``). Без этого блока
    работа правила D не отражалась бы в отчётах вообще.
    """
    from spanverify.engine import Verifier

    pairs = [
        {
            "id": "attribution-1",
            "context": (
                "Согласно регламенту, срок хранения первичных документов составляет пять лет. "
                "Срок хранения вторичных документов составляет десять лет."
            ),
            "answer": "Срок хранения первичных документов составляет десять лет.",
            "labels": [[45, 59, 1]],
        },
        {
            "id": "clean-1",
            "context": "Согласно регламенту, срок хранения первичных документов составляет пять лет.",
            "answer": "Срок хранения первичных документов составляет пять лет.",
            "labels": [],
        },
    ]
    report = Verifier(mode="demo").evaluate(pairs)
    verdicts = report["verdicts"]
    assert set(verdicts) >= {"tp", "fp", "fn", "tn", "precision", "recall", "f1", "fpr"}
    assert verdicts["tp"] == 1, "подмена числа обязана дать замечание на уровне ответа"
    assert verdicts["tn"] == 1, "чистый ответ не должен получать замечание"
    assert verdicts["recall"] == 1.0
    assert verdicts["fpr"] == 0.0
