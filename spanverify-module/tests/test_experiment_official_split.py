"""Эксперимент с официальными разбиениями: тест не участвует в выборе порога.

Смысл набора: когда у корпуса есть официальные сплиты
(``splits/{train,dev,test}.jsonl``), веса, маска и порог выбираются только на
обучающей части, а официальный тест измеряется отдельно — вместе с метриками по
типам, долей ложных пометок на чистых парах, кривой маски и сравнением правила
вердикта. Этого требует пункт «отчёт по официальному test, отдельно от всех
пар» — без него числа считались по парам, на которых выбиралась маска.
"""

from __future__ import annotations

import json
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    import importlib.util
    import sys

    path = MODULE_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


def _build_corpus(tmp_path: Path) -> Path:
    """Небольшой корпус с официальными разбиениями."""
    from spanverify.dataset import generate_pairs, write_pairs

    pairs = generate_pairs(60, seed=42)
    dataset = tmp_path / "pairs.jsonl"
    write_pairs(pairs, dataset)
    records = [pair.to_dict() for pair in pairs]
    splits = tmp_path / "splits"
    splits.mkdir()
    # Разбиение по документам (группам), как у реальных корпусов.
    groups: dict[str, list[dict]] = {}
    for pair in records:
        groups.setdefault(str(pair.get("meta", {}).get("subject", "")), []).append(pair)
    keys = sorted(groups)
    train_keys = keys[: len(keys) * 7 // 10] or keys[:1]
    dev_keys = keys[len(keys) * 7 // 10 : len(keys) * 9 // 10] or keys[-1:]
    test_keys = keys[len(keys) * 9 // 10 :] or keys[-1:]

    def dump(name: str, selected: list[str]) -> int:
        part = [pair for key in selected for pair in groups[key]]
        with (splits / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for pair in part:
                handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
        return len(part)

    counts = {"train": dump("train", train_keys), "dev": dump("dev", dev_keys), "test": dump("test", test_keys)}
    assert counts["train"] and counts["test"], "сплиты обязаны быть непустыми"
    return dataset


def test_experiment_reports_official_test_separately(tmp_path: Path) -> None:
    """Официальный тест попадает в отчёт отдельно и не используется при обучении."""
    runner = _load_script("run_experiments")
    dataset = _build_corpus(tmp_path)
    out = tmp_path / "out"
    assert runner.main(["--dataset", str(dataset), "--mode", "demo", "--out", str(out)]) == 0

    payload = json.loads((out / "experiment.json").read_text(encoding="utf-8"))
    assert payload["splits"] is not None, "официальные разбиения обязаны быть отражены"
    assert "только на официальной обучающей части" in payload["splits"]["rule"]

    sizes = payload["splits"]["sizes"]
    test_block = payload["test_official"]
    assert test_block is not None, "официальный тест обязан быть отдельным блоком"
    assert test_block["pairs"] == sizes["test"]
    assert payload["metrics"]["pairs"] == sizes["train"] + sizes["dev"] + sizes["test"]
    assert payload["by_type_test"], "полнота и F1 по типам на официальном тесте обязательны"
    assert payload["clean_pairs"]["pairs"] >= 1
    assert payload["mask_scan"]["split"] == "official_train"
    assert payload["mask_scan"]["statement"], "вывод о точке маски обязан быть в файле"

    comparison = payload["verdict_rule_comparison"]
    assert comparison["test_pairs"] == sizes["test"]
    assert comparison["tokens_identical"] is True, "токенная маска не зависит от правила вердикта"

    markdown = (out / "experiment.md").read_text(encoding="utf-8")
    assert "Официальный test" in markdown
    assert "не участвовал в выборе параметров" in markdown


def test_experiment_without_splits_keeps_old_path(tmp_path: Path) -> None:
    """Корпус без официальных разбиений идёт по старому пути (весь корпус)."""
    runner = _load_script("run_experiments")
    from spanverify.dataset import generate_pairs, write_pairs

    dataset = tmp_path / "plain.jsonl"
    write_pairs(generate_pairs(60, seed=7), dataset)
    out = tmp_path / "out"
    assert runner.main(["--dataset", str(dataset), "--mode", "demo", "--out", str(out)]) == 0
    payload = json.loads((out / "experiment.json").read_text(encoding="utf-8"))
    assert payload["splits"] is None
    assert payload["test_official"] is None
    assert payload["metrics"]["pairs"] == 60


def test_no_splits_flag_ignores_official_splits(tmp_path: Path) -> None:
    """--no-splits запрещает трогать официальные разбиения (явный сценарий)."""
    runner = _load_script("run_experiments")
    dataset = _build_corpus(tmp_path)
    out = tmp_path / "out"
    assert runner.main(["--dataset", str(dataset), "--mode", "demo", "--out", str(out), "--no-splits"]) == 0
    payload = json.loads((out / "experiment.json").read_text(encoding="utf-8"))
    assert payload["splits"] is None
    assert payload["test_official"] is None
