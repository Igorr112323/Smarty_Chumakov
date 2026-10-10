"""Разбиение по документам без утечек, и test не участвовал в подборе.

Три независимые проверки, и все три обязаны падать, а не пропускаться, если
протокол нарушен:

1. документы частей не пересекаются (само разбиение — по документам, а не по
   парам: два ответа к одному акту не могут быть одновременно в train и в test);
2. ``sha256`` файла test совпадает с записанным в манифесте шага 3 (подмена
   выборки после заморозки была бы незаметна);
3. порог финальной оценки — ровно тот, что зафиксирован до шага 3 в
   ``docs/EXPERIMENTS.md`` и в ``config/hf_final_config.json`` (подбор порога по
   test — самое дешёвое «улучшение», и оно обнуляет результат).
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CORPORA = {
    "a1": ROOT / "data" / "corpus_a" / "splits",
    "a3": ROOT / "data" / "corpus_a3" / "splits",
}
MANIFEST = ROOT / "reports" / "hf_final" / "manifest.json"
FINAL_CONFIG = ROOT / "config" / "hf_final_config.json"
EXPERIMENTS = ROOT / "docs" / "EXPERIMENTS.md"


def _harness():
    script = ROOT / "scripts" / "hf_protocol.py"
    spec = importlib.util.spec_from_file_location("hf_protocol_for_leakage", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _records(harness, path: Path) -> list[dict]:
    return harness.load_split(path)


@pytest.mark.parametrize("corpus", sorted(CORPORA))
def test_documents_are_disjoint_between_split_parts(corpus: str) -> None:
    """Ни один документ не попадает в две части; каждая пара — ровно в одной."""
    harness = _harness()
    directory = CORPORA[corpus]
    files = {
        "train": directory / "train.jsonl",
        "val": harness.val_path_of(directory),
        "test": directory / "test.jsonl",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    assert not missing, f"нет файлов разбиения: {missing}"
    documents: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    for part, path in files.items():
        records = _records(harness, path)
        documents[part] = {harness.document_of(record) for record in records}
        counts[part] = len(records)
        assert records, f"{corpus}/{part}: пустая выборка"
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = documents[first] & documents[second]
        assert not overlap, f"{corpus}: {first} и {second} делят документы: {sorted(overlap)[:5]}"
    # Пары тоже не должны дублироваться между частями (иначе «удержание» мнимое).
    ids: dict[str, set[str]] = {}
    for part, path in files.items():
        ids[part] = {str(record.get("id")) for record in _records(harness, path)}
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (ids[first] & ids[second]), f"{corpus}: пары пересекаются {first}↔{second}"


@pytest.mark.parametrize("corpus", sorted(CORPORA))
def test_split_files_have_expected_size_and_are_not_trimmed(corpus: str) -> None:
    """Объём выборок не урезан: 840/180/180 для a1 и 842/181/177 для a3.

    Урезание test — самый простой способ «поднять» метрику, поэтому размер
    частей зафиксирован здесь явно, а не выводится из удобства прогона.
    """
    harness = _harness()
    directory = CORPORA[corpus]
    expected = {"a1": {"train": 840, "val": 180, "test": 180}, "a3": {"train": 842, "val": 181, "test": 177}}[corpus]
    for part, pairs in expected.items():
        path = harness.val_path_of(directory) if part == "val" else directory / f"{part}.jsonl"
        records = _records(harness, path)
        assert len(records) == pairs, f"{corpus}/{part}: пар {len(records)}, ожидалось {pairs}"
        for record in records:
            assert str(record.get("answer") or "").strip(), f"{corpus}/{part}: пустой ответ у {record.get('id')}"
            context = record.get("context")
            text = context if isinstance(context, str) else json.dumps(context, ensure_ascii=False)
            assert len(text) > 40, f"{corpus}/{part}: контекст {record.get('id')} обрезан до {len(text)} символов"


def test_feature_extractor_does_not_see_gold() -> None:
    """``hf_grid`` не знает про разметку: признаки считаются по документу и ответу.

    Проверка по AST, а не по подстроке: упоминание «gold» в комментарии — не
    утечка, а параметр с именем ``labels`` или обращение к ``record['labels']``
    в коде извлечения признаков — утечка и есть. Плюс сигнатура: у ``compute_grid``
    нет аргумента, через который разметка могла бы прийти.
    """
    import ast
    import inspect

    source = (ROOT / "spanverify" / "hf_grid.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    identifiers: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
        elif isinstance(node, ast.keyword) and node.arg:
            identifiers.add(node.arg)
    leaked = sorted(identifiers & {"labels", "gold_spans", "bad_spans", "truth", "gold"})
    assert not leaked, f"hf_grid.py обращается к разметке: {leaked}"
    assert "spans_of" not in identifiers and "read_pairs" not in identifiers, "hf_grid читает пары с разметкой"

    sys.path.insert(0, str(ROOT))
    from spanverify import hf_grid

    parameters = list(inspect.signature(hf_grid.compute_grid).parameters)
    assert parameters == ["answer", "context", "loaded", "max_length", "k_values", "window"], parameters


def _require_manifest_or_pending() -> dict | None:
    """Манифеста нет — значит шаг 3 не выполнен: проверяем, что чисел нет в docs.

    Тихо пропускать проверку нельзя, но и врать про «не выполнено» нельзя:
    поэтому в отсутствие артефакта тестируется главное следствие протокола —
    правдоподобных цифр до измерения в README и журнале стоять не должно.
    """
    if MANIFEST.is_file():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    for path in (REPO_ROOT / "README.md", EXPERIMENTS):
        text = path.read_text(encoding="utf-8")
        assert "не измерено" in text, f"{path.name}: измерения нет, а отметки «не измерено» тоже нет — откуда числа?"
    return None


def test_manifest_test_hash_matches_the_file() -> None:
    """sha256 файла test == записанный в манифесте и в заморозке шага 0.

    Плюс сверка объёма: прогон обязан измерить все 177 пар test и 64 документа,
    иначе «улучшение» можно получить просто выкинув трудные пары (срезка выборки
    запрещена протоколом). ``check_splits`` возвращает ``(числа, проблемы)`` —
    перепутать их местами значит вечно проверять не то, поэтому здесь числа
    используются явно.
    """
    manifest = _require_manifest_or_pending()
    if manifest is None:
        return
    harness = _harness()
    numbers, problems = harness.check_splits(MANIFEST)
    assert not problems, "разбиение разошлось с манифестом:\n  " + "\n  ".join(problems)
    actual = harness.sha256_file(Path(str(manifest["splits_dir"])) / "test.jsonl")
    assert numbers["test_sha256"] == actual, "в манифесте sha256 теста не тот, что у файла"
    freeze = ROOT / "reports" / "hf_protocol" / "freeze.json"
    if not freeze.is_file():
        return
    recorded = json.loads(freeze.read_text(encoding="utf-8"))
    part = ((recorded.get("corpora") or {}).get(str(manifest.get("corpus"))) or {}).get("parts") or {}
    test_part = part.get("test") or {}
    assert test_part.get("sha256") == actual, "sha256 test разошёлся с заморозкой шага 0"
    assert int(manifest.get("pairs", {}).get("test") or 0) == int(
        test_part.get("pairs") or 0
    ), "число пар test в прогоне меньше зафиксированного: выборку сокращать нельзя"
    assert int(numbers["documents"]["test"]) == int(
        test_part.get("documents") or 0
    ), "число документов test разошлось с заморозкой шага 0"
    assert int(manifest.get("gold_rows") or 0) == int(
        test_part.get("pairs") or 0
    ), "gold-записей меньше, чем пар test: уровни ответов и фрагментов посчитаны не по всей выборке"


def test_threshold_is_the_one_frozen_before_step_three() -> None:
    """Порог финала = порог из зафиксированной конфигурации и записи в EXPERIMENTS.md.

    Если порог был подобран по test, он не совпадёт с числом, записанным до
    запуска: это и есть проверка на запрет подгонки порога по тесту.
    """
    manifest = _require_manifest_or_pending()
    if manifest is None:
        return
    threshold = float(manifest.get("threshold") or 0.0)
    assert FINAL_CONFIG.is_file(), f"нет {FINAL_CONFIG.name}: конфигурация не была зафиксирована до шага 3"
    config = json.loads(FINAL_CONFIG.read_text(encoding="utf-8"))
    assert str(config.get("step")) == "final-config", "config не размечен как final-config"
    recorded = config.get("threshold")
    assert recorded is not None, "в зафиксированной конфигурации нет порога"
    assert abs(float(recorded) - threshold) <= 1e-6, f"порог финала {threshold} != зафиксированный {recorded}"
    if manifest.get("config_sha256") and FINAL_CONFIG.is_file():
        harness = _harness()
        assert harness.sha256_file(FINAL_CONFIG) == manifest["config_sha256"], "config изменён после фиксации"
    text = EXPERIMENTS.read_text(encoding="utf-8")
    frozen = re.findall(r"Порог, зафиксированный до шага 3:\s*([0-9.]+)", text)
    assert frozen, "в docs/EXPERIMENTS.md нет строки «Порог, зафиксированный до шага 3: …» — порог не был зафиксирован"
    assert abs(float(frozen[-1]) - threshold) <= 1e-6, f"в EXPERIMENTS.md порог {frozen[-1]}, в манифесте {threshold}"
