"""Тесты фиксации шага 3 и замороженного порога.

Два сюжета, оба про единственный запуск на test:

* ``scripts/freeze_final_config.py`` — конфигурация и запись в журнал
  генерируются из свода шага 2, а не переписываются от руки; повторная фиксация
  отказывается, отбор по test и ошибки в выбранной конфигурации — тоже;
* ``run_experiment(fixed_threshold=…)`` — шаг 3 обязан применить ровно то число,
  которое зафиксировано до запуска, а не подбирать порог заново на каждом seed'е
  (иначе «зафиксированный порог» нельзя сверить с манифестом).
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _load(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


freeze = _load("freeze_final_config_for_test", "freeze_final_config.py")
harness = _load("hf_protocol_for_freeze_test", "hf_protocol.py")


def _sweep_payload(**overrides: Any) -> dict[str, Any]:
    """Сводка шага 2 в миниатюре: два эксперимента, выбранный — первый."""
    entry = {
        "number": 1,
        "name": "E1-entropy",
        "features": ["entropy_last", "sim_max_last"],
        "window": 1,
        "merge_gap": 2,
        "classifier": "logreg",
        "params": {"epochs": 20},
        "selection": {"corpus": "a3", "token_f1_val": 0.64321, "token_fpr_val": 0.12, "auc_val": 0.9},
        "per_corpus": {
            "a3": {
                "threshold": 0.412345,
                "tokens": {"f1": 0.64321, "precision": 0.7, "recall": 0.6, "fpr": 0.12, "auc": 0.9},
                "answers": {"f1": 0.55, "fpr": 0.2},
                "spans": {"f1": 0.48},
                "criterion": {"status": "не достигнуто на val"},
                "error": None,
            }
        },
    }
    rejected = {
        "number": 2,
        "name": "E2-baseline",
        "features": ["entropy_last"],
        "window": 0,
        "merge_gap": 2,
        "classifier": "logreg",
        "params": {},
        "selection": {"corpus": "a3", "token_f1_val": 0.49, "token_fpr_val": 0.3, "auc_val": 0.8},
        "per_corpus": {"a3": {"threshold": 0.5, "tokens": {"f1": 0.49}, "error": None}},
    }
    payload: dict[str, Any] = {
        "step": "sweep",
        "protocol": "hf-criterion-v1",
        "limit": 10,
        "rule": "выбирается конфигурация с максимальным token_f1 на val при token_fpr ≤ 0,40",
        "best": "E1-entropy",
        "experiments": [entry, rejected],
        "grid_sha256": "a" * 64,
        "code_commit": "0123456789abcdef",
        "duration_s": 1234.5,
        "model": {"id": "ai-forever/rugpt3small_based_on_gpt2", "revision": "r" * 40},
    }
    payload.update(overrides)
    return payload


@pytest.fixture()
def workspace(tmp_path: Path) -> dict[str, Path]:
    journal = tmp_path / "docs" / "EXPERIMENTS.md"
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text("# Журнал экспериментов\n\nШаги 1–2: не измерено.\n", encoding="utf-8")
    sweeps = tmp_path / "reports" / "hf_protocol"
    sweeps.mkdir(parents=True, exist_ok=True)
    return {
        "root": tmp_path,
        "sweep": sweeps / "sweep.json",
        "config": tmp_path / "config" / "hf_final_config.json",
        "journal": journal,
    }


def _run(workspace: dict[str, Path], payload: dict[str, Any], *extra: str) -> int:
    workspace["sweep"].write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return freeze.main(
        [
            "--sweep",
            str(workspace["sweep"]),
            "--config-out",
            str(workspace["config"]),
            "--journal",
            str(workspace["journal"]),
            "--run-id",
            "42",
            *extra,
        ]
    )


def test_config_is_generated_from_sweep(workspace: dict[str, Path]) -> None:
    """Конфигурация = выбранный эксперимент + порог из train-CV, без домысливания."""
    assert _run(workspace, _sweep_payload()) == 0
    config = json.loads(workspace["config"].read_text(encoding="utf-8"))
    assert config["step"] == "final-config"
    assert config["name"] == "E1-entropy"
    assert config["features"] == ["entropy_last", "sim_max_last"]
    assert config["window"] == 1 and config["merge_gap"] == 2 and config["classifier"] == "logreg"
    assert config["params"] == {"epochs": 20}
    assert abs(config["threshold"] - 0.412345) <= 1e-9, "порог обязан прийти из записи шага 2"
    assert config["seeds"] == [42, 43, 44, 45, 46]
    assert config["source"]["sweep_sha256"] and len(config["source"]["sweep_sha256"]) == 64
    assert config["source"]["ci_run"] == "42"
    assert config["model"]["revision"], "происхождение модели фиксируется вместе с конфигурацией"
    assert "--splits data/corpus_a3/splits" in config["final_command"]
    assert config["val_numbers"]["tokens"]["f1"] == 0.64321


def test_journal_gets_exactly_one_threshold_line(workspace: dict[str, Path]) -> None:
    """Строка порога — одна, и она равна числу в конфигурации (её сверяет тест утечки)."""
    _run(workspace, _sweep_payload())
    text = workspace["journal"].read_text(encoding="utf-8")
    found = re.findall(r"Порог, зафиксированный до шага 3:\s*([0-9.]+)", text)
    assert len(found) == 1, f"строка порога должна быть одна, найдено {found}"
    config = json.loads(workspace["config"].read_text(encoding="utf-8"))
    assert abs(float(found[0]) - config["threshold"]) <= 1e-6
    assert "E1-entropy" in text and "прогон CI `42`" in text


def test_second_freeze_is_refused(workspace: dict[str, Path]) -> None:
    """Повторная фиксация невозможна: шаг 3 выполняется один раз по одному заморозке."""
    _run(workspace, _sweep_payload())
    with pytest.raises(SystemExit, match="уже существует"):
        _run(workspace, _sweep_payload())


def test_dry_run_writes_nothing(workspace: dict[str, Path]) -> None:
    """Черновик не трогает ни конфигурацию, ни журнал — иначе его можно выдать за фиксацию."""
    assert _run(workspace, _sweep_payload(), "--dry-run") == 0
    assert not workspace["config"].exists()
    assert "Шаг 3" not in workspace["journal"].read_text(encoding="utf-8")


def test_selected_experiment_with_error_blocks_freeze(workspace: dict[str, Path]) -> None:
    """Ошибка в выбранной конфигурации — фиксировать нечего, падать нельзя."""
    payload = _sweep_payload()
    payload["experiments"][0]["per_corpus"]["a3"]["error"] = "модель не сошлась"
    with pytest.raises(SystemExit, match="ошибки"):
        _run(workspace, payload)


def test_selection_on_test_blocks_freeze(workspace: dict[str, Path]) -> None:
    """Отбор по test обесценивает измерение — фиксация запрещена, а не «замечена»."""
    payload = _sweep_payload()
    payload["experiments"][0]["selection"]["corpus"] = "a3-test"
    with pytest.raises(SystemExit, match="test"):
        _run(workspace, payload)


def test_more_than_ten_experiments_blocks_freeze(workspace: dict[str, Path]) -> None:
    """Лимит десяти — часть протокола: сводка с 11 прогонами подозрительна."""
    payload = _sweep_payload()
    payload["experiments"] = payload["experiments"] * 6
    payload["limit"] = 10
    with pytest.raises(SystemExit, match="лимита"):
        _run(workspace, payload)


def test_wrong_seed_count_blocks_freeze(workspace: dict[str, Path]) -> None:
    """Протокол — пять seed'ов; четыре или шесть значит, что прогон переподбирали."""
    with pytest.raises(SystemExit, match="5 seed"):
        _run(workspace, _sweep_payload(), "--seeds", "42,43")


def test_stop_rule_picks_best_f1_and_records_deviation(workspace: dict[str, Path]) -> None:
    """Ни одна конфигурация не прошла FPR ≤ 0,40: шаг 3 идёт с максимальной F1 и с записью."""
    payload = _sweep_payload(best=None)
    for item in payload["experiments"]:
        item["per_corpus"]["a3"]["tokens"]["fpr"] = 0.55
        item["selection"]["token_fpr_val"] = 0.55
    payload["experiments"][1]["selection"]["token_f1_val"] = 0.71
    assert _run(workspace, payload) == 0
    config = json.loads(workspace["config"].read_text(encoding="utf-8"))
    assert config["name"] == "E2-baseline", "стоп-правило: берём максимальную F1(val)"
    assert any("стоп-правило" in item for item in config["deviations"]), "отступление обязано быть записано"
    assert "0,71" in json.dumps(config, ensure_ascii=False) or "0.71" in json.dumps(config, ensure_ascii=False)


# ------------------------------------------------------------ замороженный порог


def _records(count: int, doc: str) -> list[dict[str, Any]]:
    """Пары «ответ — документ» с разметкой на токены: ответ из пяти токенов, два золотых."""
    records: list[dict[str, Any]] = []
    for index in range(count):
        answer = "срок десять лет хранения договора"
        records.append(
            {
                "id": f"{doc}-{index}",
                "answer": answer,
                "context": "Документ: срок хранения десять лет, вторичные условия — три года.",
                "labels": [[5, 15, 1]],
                "meta": {"kind": "number", "doc_id": doc, "value": "десять"},
            }
        )
    return records


def _cache(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    from spanverify.hf_grid import GRID_FEATURE_NAMES

    cache: dict[str, dict[str, Any]] = {}
    for position, record in enumerate(records):
        tokens = 5
        arrays = {}
        for name in GRID_FEATURE_NAMES:
            if name == "entropy_last":
                values = [0.9, 0.85, 0.2, 0.15, 0.1]
            elif name == "sim_max_last":
                values = [0.1, 0.2, 0.9, 0.8, 0.15]
            else:
                values = [float((position + index) % 3) / 3.0 for index in range(tokens)]
            arrays[name] = values
        cache[str(record["id"])] = {
            "pair_id": str(record["id"]),
            "model": "model-x",
            "grid_format": 3,
            "arrays": arrays,
            "tokens": [
                {"text": word, "start": start, "end": start + len(word), "scored": 1}
                for word, start in zip(
                    ["срок", "десять", "лет", "хранения", "договора"],
                    [0, 5, 12, 16, 26],
                    strict=True,
                )
            ],
            "meta": {"unmatched_tokens": 0, "seq_len": tokens, "max_length": 1024},
        }
    return cache


_ENTRY = {
    "name": "fixed-threshold",
    "features": ["entropy_last", "sim_max_last"],
    "window": 0,
    "merge_gap": 2,
    "classifier": "logreg",
    "params": {"epochs": 15},
}


def test_fixed_threshold_is_applied_verbatim() -> None:
    """Шаг 3 применяет зафиксированный порог без переселекции — на любом seed'е."""
    train = _records(6, "doc-train")
    val = _records(3, "doc-val")
    cache = _cache(train + val)
    first = harness.run_experiment(cache, train, val, _ENTRY, 42, fixed_threshold=0.375)
    second = harness.run_experiment(cache, train, val, _ENTRY, 43, fixed_threshold=0.375)
    assert "error" not in first, first.get("error")
    assert first["threshold"] == pytest.approx(0.375, abs=1e-9)
    assert second["threshold"] == pytest.approx(0.375, abs=1e-9)
    assert first["threshold_frozen"] is True
    assert first["threshold_selection"]["source"] == "frozen-before-step-3"


def test_threshold_without_freeze_is_still_selected_on_fit() -> None:
    """Без заморозки поведение не меняется: порог подбирается CV по обучающей части."""
    train = _records(6, "doc-train")
    val = _records(3, "doc-val")
    cache = _cache(train + val)
    block = harness.run_experiment(cache, train, val, _ENTRY, 42)
    assert "error" not in block, block.get("error")
    assert block["threshold_frozen"] is False
    assert 0.0 < block["threshold"] < 1.0
    assert block["threshold_selection"].get("source") != "frozen-before-step-3"


def test_mixed_models_in_cache_are_rejected() -> None:
    """Признаки из разных моделей в одном измерении — ошибка, а не «среднее»."""
    with pytest.raises(SystemExit, match="разными моделями"):
        harness.cache_model_provenance({"a": {"model": "m1"}, "b": {"model": "m2"}})


def test_provenance_prefers_revision_from_cache() -> None:
    """Revision весов читается из строк кеша: шаг 3 модель не загружает."""
    info = harness.cache_model_provenance(
        {"a": {"model": "m", "revision": "deadbeef"}, "b": {"model": "m", "revision": "deadbeef"}}
    )
    assert info["id"] == "m" and info["revision"] == "deadbeef" and info["rows"] == 2
