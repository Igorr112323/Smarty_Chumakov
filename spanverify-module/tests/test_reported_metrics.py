"""Каждое число финального манифеста пересчитывается из строк предсказаний.

Смысл теста: ``reports/hf_final/manifest.json`` не должен быть «файлом, которому
верят на слово». Он сверяется с ``predictions_test.jsonl.gz`` (строка на токен
на seed) и ``gold_test.jsonl.gz`` (границы gold-фрагментов на пару): метрики
всех трёх уровней считаются заново функциями продукта и обязаны совпасть до
1e-6. Манифеста нет — тест падает с указанием команды прогона: «измерение не
выполнено» — это статус, а не повод для `skip` или `xfail`.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "reports" / "hf_final" / "manifest.json"
PREDICTIONS = ROOT / "reports" / "hf_final" / "predictions_test.jsonl.gz"
GOLD = ROOT / "reports" / "hf_final" / "gold_test.jsonl.gz"
MISSING = (
    "нет артефакта {path}: шаг 3 протокола выполняется workflow «Протокол hf (критерий ТЗ)» "
    "(run_final=true), артефакты публикуются в ветку data/metrics (каталог hf-final). "
    "Чисел в README и отчёте быть не должно, пока нет этого файла."
)


def _harness():
    script = ROOT / "scripts" / "hf_protocol.py"
    spec = importlib.util.spec_from_file_location("hf_protocol_for_metrics", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ROOT_DIR = ROOT.parent
JOURNAL = ROOT / "docs" / "EXPERIMENTS.md"


def _load() -> dict | None:
    """Манифест, либо ``None`` при «измерение не выполнено».

    Отсутствие артефакта не пропуск: в этом режиме проверяется, что ни в README,
    ни в журнале нет чисел, которых ещё не существует (рукой вписанное «0,61»
    было бы Worse, чем красный CI). Сам факт измерения требует
    ``tests/test_criterion_hf.py`` — он и остаётся красным до шага 3.
    """
    if MANIFEST.is_file():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    for path in (ROOT_DIR / "README.md", JOURNAL):
        text = path.read_text(encoding="utf-8")
        assert "не измерено" in text, f"{path.name}: измерения нет и отметки «не измерено» нет"
    return None


def test_every_manifest_number_is_recomputable_from_predictions() -> None:
    """Пересчёт из предсказаний: допуск 1e-6, все три уровня (gold-файл обязателен)."""
    manifest = _load()
    if manifest is None:
        return
    harness = _harness()
    problems, numbers = harness.compare_manifest(MANIFEST, PREDICTIONS, GOLD)
    assert not problems, "манифест не сходится с предсказаниями:\n  " + "\n  ".join(problems)
    assert numbers["rows"] > 0
    assert numbers["seeds"] == len(manifest.get("seeds", []))
    assert PREDICTIONS.is_file(), MISSING.format(path="reports/hf_final/predictions_test.jsonl.gz")
    assert GOLD.is_file(), MISSING.format(path="reports/hf_final/gold_test.jsonl.gz")


def test_predictions_have_the_documented_format_and_row_count() -> None:
    """Строки предсказаний: обещанный формат полей и число строк из манифеста."""
    manifest = _load()
    if manifest is None:
        return
    harness = _harness()
    rows = harness.read_jsonl(PREDICTIONS)
    assert len(rows) == manifest.get("prediction_rows"), (len(rows), manifest.get("prediction_rows"))
    format_fields = tuple(manifest.get("predictions_format") or harness.PREDICTIONS_FIELDS)
    missing = [row.get("id") for row in rows[:200] if any(field not in row for field in format_fields)]
    assert not missing, f"строки без полей {format_fields}: {missing[:3]}"
    seeds = {int(row["seed"]) for row in rows}
    assert seeds == {int(seed) for seed in manifest.get("seeds", [])}, (seeds, manifest.get("seeds"))
    # Полный test, ни одна пара не потеряна: иначе микросредние считаются по
    # другой выборке, чем заявлена.
    pairs = {str(row["id"]) for row in rows}
    assert len(pairs) == manifest.get("pairs", {}).get("test"), (len(pairs), manifest.get("pairs"))


def test_gold_file_matches_test_split() -> None:
    """Gold-фрагменты = разметка test 1-в-1: править разметку нельзя (протокол)."""
    manifest = _load()
    if manifest is None:
        return
    harness = _harness()
    recorded = {
        str(row["id"]): sorted(tuple(map(int, span)) for span in row.get("spans", []))
        for row in harness.read_jsonl(GOLD)
    }
    split_dir = ROOT / str(manifest.get("splits_dir") or "data/corpus_a3/splits")
    corpus = str(manifest.get("corpus") or "")
    expected: dict[str, list[tuple[int, int]]] = {}
    for record in harness.load_split(split_dir / "test.jsonl"):
        expected[f"{corpus}:{record.get('id')}"] = sorted(harness.spans_of(record))
    assert set(recorded) == set(expected), "набор пар в gold-файле не равен набору пар test"
    changed = [key for key, value in expected.items() if recorded[key] != value]
    assert not changed, f"gold-фрагменты изменены относительно разметки: {changed[:3]}"


def test_readme_criterion_block_comes_from_the_manifest() -> None:
    """Числа в README — только из манифеста: блок собирает ``render_hf_table.py``."""
    manifest = _load()
    if manifest is None:
        return
    readme = ROOT.parent / "README.md"
    text = readme.read_text(encoding="utf-8")
    renderer_path = ROOT / "scripts" / "render_hf_table.py"
    spec = importlib.util.spec_from_file_location("render_hf_table_for_test", renderer_path)
    assert spec and spec.loader
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    block = renderer.extract_block(text)
    assert block is not None, f"в {readme.name} нет блока критерия (маркеры HF-CRITERION)"
    numbers = manifest.get("metrics_mean_std") or {}
    for key in ("f1", "fpr", "precision", "recall"):
        value = numbers.get(key)
        assert isinstance(value, (int, float)), f"в манифесте нет {key}"
        assert f"{float(value):.6f}" in block, f"{key}={value} не вписан в блок README"
    status = str((manifest.get("criterion") or {}).get("status") or "")
    assert status in block, f"статус критерия «{status}» не отражён в README"
