"""Тесты кривой порога на val (`scripts/val_curve.py`).

Почему это тестируется, а не «проверено один раз»: вывод «правило порога уже даёт максимум F1 при
FPR ≤ 0,40» — единственное основание сказать, что дефицит не в операционной точке. Если скрипт
молча сравнит не то (другая выборка, другое определение FPR), вывод станет несуществующим, поэтому
закреплено три вещи: совпадение с манифестом шага 1, совпадение правила с полным перебором и
отказ при явном расхождении чисел.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for _path in (str(ROOT), str(SCRIPTS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import val_curve  # noqa: E402

PREDICTIONS = ROOT / "reports" / "hf_protocol" / "predictions_val.jsonl.gz"
MANIFEST = ROOT / "reports" / "hf_protocol" / "manifest_val.json"


def _read_rows() -> list[dict]:
    rows = []
    with gzip.open(PREDICTIONS, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            rows.append(
                {
                    "id": record["id"],
                    "start": record["start"],
                    "end": record["end"],
                    "gold": int(record["gold"]),
                    "score": float(record["score"]),
                }
            )
    return rows


@pytest.fixture(scope="module")
def payload(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("curve") / "val_curve.json"
    code = val_curve.main(
        [
            "--predictions",
            str(PREDICTIONS),
            "--manifest",
            str(MANIFEST),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    return json.loads(out.read_text(encoding="utf-8"))


def test_no_point_beats_the_rule_under_the_cap_on_val(payload: dict) -> None:
    """Главное утверждение: под потолком FPR на val нет точки с F1 выше, чем даёт правило."""
    answer = payload["answer"]
    assert answer["есть_точка_выше_под_потолком"] is False, answer
    assert answer["число_таких_точек"] == 0
    assert answer["правило_совпало_с_полным_перебором"] is True
    rule = payload["rule_on_val"]
    capped = payload["best_capped_by_full_scan"]
    assert abs(float(rule["f1"]) - float(capped["f1"])) <= 2e-5
    assert float(capped["fpr"]) <= payload["fpr_cap"]
    # Oracle без ограничения на val совпадает с точкой под потолком: значит потолок FPR не связывает
    # руки — «разрешить больше ложных» нечего, и это исключает объяснение «мешает ограничение FPR».
    oracle = payload["best_unconstrained_oracle"]
    assert abs(float(oracle["f1"]) - float(capped["f1"])) <= 2e-5


def test_recorded_point_reproduces_the_step1_manifest(payload: dict) -> None:
    """Числа на записанном пороге обязаны совпасть с `manifest_val.json` — иначе сравнено не то."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    numbers = manifest["metrics_mean_std"]
    at = payload["at_recorded_threshold"]
    for key in ("precision", "recall", "f1", "fpr"):
        assert abs(float(at[key]) - float(numbers[key])) <= 2e-5, key
    assert payload["manifest_cross_check"]["available"] is True
    assert payload["split"]["tokens"] == int(numbers["rows"])
    assert payload["split"]["positive_tokens"] == int(numbers["positive_rows"])


def test_val_only_guards(payload: dict, tmp_path: Path) -> None:
    assert payload["test_not_read"] is True and payload["not_for_criterion"] is True
    fake = tmp_path / "predictions_test.jsonl.gz"
    fake.write_bytes(b"")
    with pytest.raises(SystemExit) as caught:
        val_curve.main(["--predictions", str(fake), "--manifest", str(MANIFEST), "--out", str(tmp_path / "o.json")])
    assert "val" in str(caught.value)


def _synthetic_rows() -> list[dict]:
    """20 токенов: 10 положительных со скорами выше 0,5, 10 отрицательных — с одним «выбросом»."""
    rows = []
    for index in range(10):
        rows.append({"id": "p", "start": index, "end": index + 1, "gold": 1, "score": 0.9 - 0.01 * index})
    for index in range(10):
        rows.append({"id": "n", "start": index, "end": index + 1, "gold": 0, "score": 0.07 * index})
    return rows


def test_full_scan_and_rule_agree_on_synthetic(tmp_path: Path) -> None:
    rows = _synthetic_rows()
    scan = val_curve.scan_curve(rows, merge_gap=2, cap=0.40)
    import hf_protocol

    rule = hf_protocol.choose_threshold([float(row["score"]) for row in rows], [int(row["gold"]) for row in rows], 0.40)
    assert abs(float(rule["f1"]) - float(scan["best_capped"]["f1"])) <= 2e-5
    # монотонность FPR вдоль кривой (порог убывает) — на ней держится обрыв цикла в choose_threshold
    fpr_values = [float(point["fpr"]) for point in scan["curve"]]
    assert fpr_values == sorted(fpr_values)
    # при полном совпадении знаков метрика равна 1, и потолок FPR этого не портит
    assert float(scan["best_capped"]["f1"]) == pytest.approx(1.0)


def test_manifest_disagreement_is_fatal(tmp_path: Path) -> None:
    """Если манифест не согласуется с кривой — скрипт обязан упасть, а не напечатать удобные числа."""
    rows = _synthetic_rows()
    predictions = tmp_path / "predictions_val.jsonl.gz"
    with gzip.open(predictions, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({**row, "level": "tokens", "pred": 1, "seed": 42, "token": "x"}) + "\n")
    manifest = tmp_path / "manifest_val.json"
    manifest.write_text(
        json.dumps({"merge_gap": 2, "metrics_mean_std": {"threshold": 0.5, "precision": 0.111, "recall": 0.999}}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as caught:
        val_curve.main(
            [
                "--predictions",
                str(predictions),
                "--manifest",
                str(manifest),
                "--out",
                str(tmp_path / "o.json"),
            ]
        )
    assert "не сходится с манифестом" in str(caught.value)
