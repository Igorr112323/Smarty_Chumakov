"""Гейт «seed обязан влиять на обучение» (обещан в docs/METRIC_SPEC_v2.md, раздел 4).

Почему этот файл появился только сейчас: раздел 4 спецификации v2 ссылается на
`tests/test_v2_seed_variance.py` как на то, чем требование закрыто, — а файла в дереве не
было. Обещание без проверки хуже его отсутствия: читатель документа认为, что гейт есть.

Что закреплено:

* `bootstrap_docs` — единственный заявленный способ внести случайность, не подменяя
  классификатор продукта: seed меняет **состав** обучающей выборки (бутстрап по документам);
  отбор конфигурации и порог остаются общими;
* линейная модель продукта на одних и тех же данных детерминирована: два вызова с разными
  seed'ами дают **тот же** модельный файл — это и есть причина `std = 0,0` по пяти seed'ам
  в v1 (см. `reports/hf_final/manifest.json`), и тест фиксирует это как ожидаемое поведение,
  а не как «нужно добавить перемешивание мини-батчей»;
* `scripts/hf_protocol_v2.py seedcheck` не пропускает шаг 3 при нулевом разбросе: код возврата
  обязан соответствовать измеренному `f1_std`, а не наоборот.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = ROOT
SCRIPTS = ROOT / "scripts"
for path in (str(SCRIPTS), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import hf_protocol_v2 as v2  # noqa: E402

from spanverify.core import tokenize_with_offsets  # noqa: E402
from spanverify.hf_grid import GRID_FEATURE_NAMES  # noqa: E402

SIGNAL = "entropy_last"
FALLBACK = "mass_last"
SPEC = ROOT / "docs" / "METRIC_SPEC_v2.md"
V1_MANIFEST = ROOT / "reports" / "hf_final" / "manifest.json"


def _pair(index: int, marked: bool) -> dict:
    answer = "три года" if marked else "десять лет"
    tokens = [token.text for token in tokenize_with_offsets(answer)]
    return {
        "id": f"pair-{index}",
        "context": f"Документ {index}: срок хранения первичных документов составляет десять лет.",
        "answer": answer,
        "labels": [[0, len(tokens[0]), 1]] if marked else [],
        "meta": {"kind": "faithful" if not marked else "number_changed", "group": f"doc-{index}"},
    }


def _arrays(marked: bool, count: int, index: int) -> dict[str, list[float]]:
    """Сигнальная колонка совпадает с меткой и усиливается по документу, остальные — константа.

    Сила сигнала разная у разных документов намеренно: иначе бутстрап по документам менял
    бы только порядок одинаковых строк, и «разный состав выборки» не превращался бы в заметно
    другую модель (на ровных данных отличия уходят в 1e-17 и исчезают после округления).
    """
    strength = round(0.95 - 0.08 * index, 4) if marked else round(0.05 + 0.08 * index, 4)
    values = [strength] * count
    return {name: (values if name == SIGNAL else [0.4] * count) for name in GRID_FEATURE_NAMES}


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    """Мини-корпус: шесть документов по одной паре, кеш сетки v3, две гипотезы (с бутстрапом и без)."""
    splits = tmp_path / "splits"
    splits.mkdir(parents=True)
    cache = tmp_path / "cache"
    cache.mkdir()

    records = []
    rows = []
    for index in range(6):
        marked = index % 2 == 0
        pair = _pair(index, marked)
        count = len(list(tokenize_with_offsets(pair["answer"])))
        records.append(pair)
        rows.append(
            {
                "pair_id": pair["id"],
                "grid_format": 3,
                "model": "test/model",
                "revision": "0" * 40,
                "arrays": _arrays(marked, count, index),
            }
        )
    for name in ("train.jsonl", "dev.jsonl", "test.jsonl"):
        with (splits / name).open("w", encoding="utf-8") as handle:
            for pair in records:
                handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    with (cache / "grid_all_00.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    config = {
        "protocol": "seed-variance-fixture",
        "seed": 42,
        "selected": "no-bootstrap",
        "experiments": [
            {
                "name": "no-bootstrap",
                "hypothesis": "обучение на всей обучающей части: seed ни на что не влияет",
                "change": "нет",
                "features": [SIGNAL, FALLBACK],
                "classifier": "logreg",
                "window": 0,
                "merge_gap": 2,
                "epochs": 40,
                "l2": 0.01,
            },
            {
                "name": "with-bootstrap",
                "hypothesis": "seed меняет состав обучающей выборки (бутстрап по документам)",
                "change": "bootstrap_docs = true",
                "features": [SIGNAL, FALLBACK],
                "classifier": "logreg",
                "window": 0,
                "merge_gap": 2,
                "epochs": 40,
                "l2": 0.01,
                "bootstrap_docs": True,
            },
        ],
    }
    (tmp_path / "experiments.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    return tmp_path


def _entry(tmp_path: Path, name: str) -> dict:
    config = json.loads((tmp_path / "experiments.json").read_text(encoding="utf-8"))
    raw = next(item for item in config["experiments"] if item["name"] == name)
    return v2.v1._experiment_entry(1, raw)


def _cache(tmp_path: Path) -> dict:
    return v2.load_cache(tmp_path / "cache")


def _records(tmp_path: Path, name: str = "train.jsonl") -> list[dict]:
    return v2.v1.load_split(tmp_path / "splits" / name)


def test_bootstrap_changes_the_training_multiset_and_is_reproducible(workspace: Path) -> None:
    records = _records(workspace)
    first = v2.bootstrap_records(records, 42)
    second = v2.bootstrap_records(records, 43)
    again = v2.bootstrap_records(records, 42)
    assert len(first) == len(records) == len(second), "бутстрап по документам обязан сохранять объём выборки"
    assert [item["id"] for item in first] == [item["id"] for item in again], "seed 42 обязан воспроизводиться"
    assert [item["id"] for item in first] != [
        item["id"] for item in second
    ], "разные seed'ы обязаны давать разный состав — иначе разброс неоткуда взяться"


def test_seed_alone_does_not_change_the_linear_model(workspace: Path) -> None:
    """Линейная модель продукта детерминирована: разный seed на тех же данных — та же модель.

    Это и есть объяснение `std = 0,0` в v1, поэтому тест проверяет именно отсутствие
    эффекта, а не «мы что-то исправили».
    """
    cache = _cache(workspace)
    entry = _entry(workspace, "no-bootstrap")
    val = _records(workspace, "dev.jsonl")
    train = _records(workspace)
    left, _, left_rows = v2.fit_and_score(cache, entry, train, val, 42)
    right, _, right_rows = v2.fit_and_score(cache, entry, train, val, 7)
    assert left.to_dict() == right.to_dict(), "seed без bootstrap не должен менять веса (свойство train_logreg)"
    assert [round(float(row["score"]), 12) for row in left_rows] == [
        round(float(row["score"]), 12) for row in right_rows
    ]


def test_bootstrap_by_documents_actually_moves_the_model(workspace: Path) -> None:
    """Бутстрап по документам меняет обученную модель: пять seed'ов — пять разных моделей.

    Проверяется эффект на модель, а не «СКО обязательно положительно»: на микроскопической
    фикстуре метрика вырождена (все пороги дают один и тот же F1), и требовать от неё
    ненулевого разброса значило бы подгонять тест под желаемый ответ. Ненулевой разброс
    метрик измеряется командой `seedcheck` на реальном кеше — в прогоне цикла, до шага 3.
    """
    import json

    cache = _cache(workspace)
    train = _records(workspace)
    val = _records(workspace, "dev.jsonl")

    def fingerprints(name: str) -> set[str]:
        entry = _entry(workspace, name)
        found = set()
        for seed in (42, 43, 44, 45, 46):
            model, _, _ = v2.fit_and_score(
                cache, entry, train, val, seed, bootstrap=bool(entry["params"].get("bootstrap_docs"))
            )
            found.add(json.dumps(model.to_dict(), sort_keys=True))
        return found

    assert len(fingerprints("with-bootstrap")) == 5, "разные seed'ы при bootstrap обязаны дать разные модели"
    assert len(fingerprints("no-bootstrap")) == 1, "без bootstrap модель одна — это и есть дефект v1"


def test_seedcheck_gate_refuses_zero_spread(workspace: Path) -> None:
    """Код возврата `seedcheck` обязан следовать за измеренным разбросом, а не за ожиданиями."""
    out = workspace / "out"
    out.mkdir()
    for name in ("no-bootstrap", "with-bootstrap"):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "hf_protocol_v2.py"),
                "seedcheck",
                "--config",
                str(workspace / "experiments.json"),
                "--grid-cache",
                str(workspace / "cache"),
                "--splits",
                str(workspace / "splits"),
                "--experiment",
                name,
                "--seeds",
                "42,43,44,45,46",
                "--out",
                str(out),
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=False,
        )
        payload = json.loads((out / "seedcheck.json").read_text(encoding="utf-8"))
        assert result.returncode == (
            0 if payload["f1_std"] else 1
        ), f"{name}: решение гейта не совпало с измеренным СКО; вывод: {(result.stdout + result.stderr)[-300:]}"
        if name == "no-bootstrap":
            assert payload["f1_std"] in (0.0, None), "без bootstrap разброс обязан быть нулевым"
            assert payload["distinct_f1_values"] == 1
            assert "ЗАПРЕТ ШАГА 3" in result.stdout
        assert payload["bootstrap_docs"] is (name == "with-bootstrap")


def test_specification_gate_chain_is_not_a_dead_reference() -> None:
    """Раздел 4 спецификации обещает команду, тест и реализацию — всё трое обязаны существовать."""
    text = SPEC.read_text(encoding="utf-8")
    section = text.split("## 4.", 1)[1].split("\n## ", 1)[0]
    assert "seedcheck" in section and "bootstrap_docs" in section and "test_v2_seed_variance" in section
    source = (SCRIPTS / "hf_protocol_v2.py").read_text(encoding="utf-8")
    assert "def command_seedcheck" in source, "команда seedcheck из документа отсутствует в скрипте"
    assert (
        "def bootstrap_records" in source and "bootstrap=bool(entry" in source
    ), "бутстрап по документам заявлен в спецификации, но не подключён к обучению"
    assert Path(__file__).is_file(), "файл теста, на который ссылается спецификация, обязан существовать"


def test_v1_zero_std_claim_is_backed_by_the_manifest() -> None:
    """Утверждение «в v1 пять seed'ов дали идентичные числа» сверено с манифестом шага 3."""
    manifest = json.loads(V1_MANIFEST.read_text(encoding="utf-8"))
    numbers = manifest["metrics_mean_std"]
    assert (
        numbers["f1_std"] == 0.0 and numbers["auc_std"] == 0.0
    ), "манифест больше не подтверждает нулевой разброс — поправьте формулировку в docs/METRIC_SPEC_v2.md"
    assert len(set(manifest["metrics_mean_std"].get("auc_per_seed", []))) == 1
