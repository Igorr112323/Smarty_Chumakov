"""Шаг 3 обязан дописать манифест и gold-записи: падение после измерения — это потерянное измерение.

Регрессия на двух реальных отказах финального прогона (run 38061440215):

1. ``aggregate_seeds`` падал на ``statistics.pstdev`` с
   ``AttributeError: 'float' object has no attribute 'numerator'`` (CPython 3.11),
   если среди значений встречался NaN — а NaN появляется законно: AUC не
   вычисляется, когда в выборке один класс. Падение случилось ПОСЛЕ того, как
   числа по всем пяти seed'ам были напечатаны, поэтому ``manifest.json`` не
   появился и подтверждать было нечего.
2. ``gold_test.jsonl.gz`` не записывался вовсе (его sha256 читался из несозданного
   файла) — а без gold-границ нельзя пересчитать уровни ответов и фрагментов,
   то есть половина подтверждений шага 3 была бы недостижима.

Тесты проверяют и то, и другое; последний прогон повторяет ``final`` целиком на
синтетическом кеше признаков и вызывает ``compare_manifest`` — тот же пересчёт,
что запускается в CI шагом «Пересчёт чисел из предсказаний».
"""

from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _harness():
    script = ROOT / "scripts" / "hf_protocol.py"
    spec = importlib.util.spec_from_file_location("hf_final_manifest_under_test", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


hp = _harness()


def _strict_json(path: Path) -> object:
    """Строгий JSON: ``NaN``/``Infinity`` парсер обязан отвергнуть."""

    def reject(name: str) -> str:  # pragma: no cover - сообщение важнее тела
        raise AssertionError(f"{path.name}: недопустимое числовое значение {name}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


# --------------------------------------------------------- агрегация по seed'ам


def test_aggregate_seeds_survives_nan_auc() -> None:
    """NaN в AUC уровня ответов не роняет агрегат и не превращается в выдуманное число."""
    per_seed = [
        {
            "seed": 42,
            "threshold": 0.5,
            "model_sha256": "a" * 64,
            "rows": 10,
            "positive_rows": 4,
            "tokens": {"f1": 0.6, "precision": 0.7, "recall": 0.55, "fpr": 0.02, "auc": 0.9},
            "answers": {"f1": 0.62, "precision": 0.7, "recall": 0.55, "fpr": 0.01, "auc": math.nan},
            "spans": {"f1": 0.3, "precision": 0.4, "recall": 0.25},
        },
        {
            "seed": 43,
            "threshold": 0.5,
            "model_sha256": "b" * 64,
            "rows": 10,
            "positive_rows": 4,
            "tokens": {"f1": 0.4, "precision": 0.5, "recall": 0.35, "fpr": 0.04, "auc": 0.8},
            "answers": {"f1": 0.42, "precision": 0.5, "recall": 0.35, "fpr": 0.03, "auc": math.nan},
            "spans": {"f1": 0.2, "precision": 0.3, "recall": 0.15},
        },
    ]
    aggregate = hp.aggregate_seeds(per_seed)
    assert aggregate["f1"] == pytest.approx(0.5)
    assert aggregate["f1_std"] == pytest.approx(0.1)
    # Не-число не «скругляется» до нуля: это null, то есть «не вычислено».
    assert aggregate["answer_auc"] is None
    assert aggregate["answer_auc_std"] is None
    assert aggregate["auc_per_seed"] == [0.9, 0.8]
    # «Не вычислено» не прячется: видно, сколько seed'ов дали не-число.
    assert aggregate["answer_auc_not_computed"] == 2


def test_aggregate_seeds_identical_values() -> None:
    """Один и тот же замороженный порог даёт одинаковые числа — СКО обязано быть 0."""
    item = {
        "seed": 42,
        "threshold": 0.1106,
        "model_sha256": "c" * 64,
        "rows": 100,
        "positive_rows": 40,
        "tokens": {"f1": 0.595041, "precision": 0.6, "recall": 0.59, "fpr": 0.006565, "auc": 0.92},
        "answers": {"f1": 0.6, "precision": 0.6, "recall": 0.6, "fpr": 0.01, "auc": math.nan},
        "spans": {"f1": 0.23, "precision": 0.25, "recall": 0.22},
    }
    per_seed = [{**item, "seed": seed} for seed in (42, 43, 44, 45, 46)]
    aggregate = hp.aggregate_seeds(per_seed)
    assert aggregate["f1"] == pytest.approx(0.595041)
    assert aggregate["f1_std"] == 0.0
    assert aggregate["fpr_std"] == 0.0
    assert aggregate["answer_f1_std"] == 0.0
    assert aggregate["answer_auc_std"] is None
    assert aggregate["answer_auc_not_computed"] == 5
    assert len(aggregate["f1_per_seed"]) == 5


def test_mean_std_is_none_without_finite_values() -> None:
    """Ни одного вычисленного значения — «не вычислено», а не 0.0."""
    assert hp._mean_std([math.nan, math.nan]) == (None, None)
    assert hp._mean_std([math.inf, 1.0]) == (1.0, 0.0)  # одно вычисленное значение
    mean, std = hp._mean_std([1.0, 2.0, 3.0])
    assert mean == pytest.approx(2.0)
    assert std == pytest.approx(math.sqrt(2.0 / 3.0))


def test_json_safe_replaces_non_finite_numbers(tmp_path: Path) -> None:
    """Манифест обязан читаться строгим парсером: ни NaN, ни Infinity."""
    payload = {
        "tokens": {"f1": 0.5, "auc": math.nan},
        "nested": [{"auc": math.inf, "note": "ок"}],
    }
    hp.write_json(tmp_path / "manifest.json", payload)
    parsed = _strict_json(tmp_path / "manifest.json")
    assert parsed["tokens"] == {"auc": None, "f1": 0.5}
    assert parsed["nested"] == [{"auc": None, "note": "ок"}]
    assert "NaN" not in (tmp_path / "manifest.json").read_text(encoding="utf-8")


# --------------------------------------------------------------- финальный прогон


def _pairs(count: int, prefix: str, rng: random.Random) -> list[dict]:
    words = "сумма договора оплата срок поставки сторона реквизиты".split()
    records = []
    for index in range(count):
        chosen = [rng.choice(words) for _ in range(7)]
        text = " ".join(chosen)
        end = len(" ".join(chosen[:3]))
        records.append(
            {
                "id": f"{prefix}doc{index // 3}_p{index}",
                "context": f"Документ {prefix}{index // 3}: условия поставки.",
                "answer": text,
                "labels": [[0, end, 1]],
            }
        )
    return records


def test_final_writes_gold_predictions_and_matches_manifest(tmp_path: Path) -> None:
    """``final`` на синтетическом кеше: gold-файл создан, пересчёт чисел чистый."""
    from spanverify.core import tokenize_with_offsets

    names = list(hp.GRID_FEATURE_NAMES)
    rng = random.Random(11)
    splits = {part: _pairs(count, part[0], rng) for part, count in (("train", 18), ("val", 6), ("test", 6))}
    split_dir = tmp_path / "splits"
    split_dir.mkdir(parents=True)
    for part, records in splits.items():
        (split_dir / f"{part}.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records), encoding="utf-8"
        )

    cache_rows = []
    for records in splits.values():
        for record in records:
            tokens = list(tokenize_with_offsets(record["answer"]))
            gold = [0] * len(tokens)
            for start, end in [tuple(item[:2]) for item in record["labels"]]:
                for position, token in enumerate(tokens):
                    if token.start < end and token.end > start:
                        gold[position] = 1
            arrays = {
                name: [
                    (
                        (0.9 if value else 0.05) + 0.1 * rng.random()
                        if name == "entmax"
                        else (0.3 if value else 0.2) + 0.4 * rng.random()
                    )
                    for value in gold
                ]
                for name in names
            }
            cache_rows.append(
                {
                    "grid_format": hp.GRID_CACHE_FORMAT,
                    "pair_id": record["id"],
                    "tokens": [token.text for token in tokens],
                    "model": "synthetic/model",
                    "revision": "0" * 40,
                    "arrays": arrays,
                }
            )
    grid_dir = tmp_path / "grid"
    grid_dir.mkdir()
    (grid_dir / "grid_shard0of1.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cache_rows), encoding="utf-8"
    )

    config = json.loads((ROOT / "config" / "hf_final_config.json").read_text(encoding="utf-8"))
    config["threshold"] = 0.5
    config["params"] = dict(config.get("params") or {}, epochs=60)
    config_path = tmp_path / "final_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

    out = tmp_path / "out"
    code = hp.main(
        [
            "final",
            "--config",
            str(config_path),
            "--grid-cache",
            str(grid_dir),
            "--splits",
            str(split_dir),
            "--corpus-name",
            "synthetic",
            "--seeds",
            "42,43",
            "--include-val",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    manifest_path = out / "manifest.json"
    predictions_path = out / "predictions_test.jsonl.gz"
    gold_path = out / "gold_test.jsonl.gz"
    assert manifest_path.is_file() and predictions_path.is_file()
    # Тот баг, из-за которого шага 3 не было видно: файл gold либо не создан,
    # либо его sha256 посчитан по несуществующему файлу.
    assert gold_path.is_file(), "final обязан писать gold_test.jsonl.gz до sha256 и до пересчёта"
    manifest = _strict_json(manifest_path)
    assert manifest["gold_rows"] > 0 and manifest["prediction_rows"] > 0
    assert manifest["gold_sha256"] == hp.sha256_file(gold_path)
    assert manifest["threshold_frozen"] is True
    assert manifest["threshold"] == pytest.approx(0.5)

    gold_rows = list(hp.read_jsonl(gold_path))
    assert all({"id", "spans"} <= set(row) for row in gold_rows)

    problems, numbers = hp.compare_manifest(manifest_path, predictions_path, gold_path)
    assert problems == [], problems
    assert numbers["seeds"] == 2
    assert numbers["f1"] == manifest["metrics_mean_std"]["f1"]
