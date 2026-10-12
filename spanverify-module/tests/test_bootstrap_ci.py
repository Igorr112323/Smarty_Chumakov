"""Тесты ДИ по документам (`scripts/bootstrap_ci.py`).

Что закреплено и почему:

* интервал обязан содержать точечную оценку и быть воспроизводимым (фиксированный seed, 1000 повторов);
* кластеризация по документам — не украшение: если все положительные токены лежат в нескольких
  документах, токен-бутстрап даёт узкий и уверенный интервал, а честный (по документам) — широкий
  вплоть до нуля. Тест построен ровно на таком случае;
* пересчёт сверяется с манифестом, а при расхождении скрипт падает: иначе ДИ можно было бы считать
  по чужим предсказаниям и ничего не заметить;
* порог берётся из манифеста и не переселэктируется — иначе интервал описывал бы другую процедуру.
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

import bootstrap_ci  # noqa: E402

VAL_PREDICTIONS = ROOT / "reports" / "hf_protocol" / "predictions_val.jsonl.gz"
VAL_MANIFEST = ROOT / "reports" / "hf_protocol" / "manifest_val.json"
TEST_PREDICTIONS = ROOT / "reports" / "hf_final" / "predictions_test.jsonl.gz"
TEST_MANIFEST = ROOT / "reports" / "hf_final" / "manifest.json"


def _write_predictions(path: Path, rows: list[dict], gz: bool = True) -> Path:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if gz:
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def _synthetic(docs: int = 24, per_doc: int = 4, threshold: float = 0.5) -> list[dict]:
    rows = []
    for doc in range(docs):
        for index in range(per_doc):
            marked = (doc + index) % 2 == 0
            score = 0.9 if marked else 0.1
            rows.append(
                {
                    "id": f"synth:doc-{doc}",
                    "level": "tokens",
                    "seed": 42,
                    "token": f"t{index}",
                    "start": index,
                    "end": index + 1,
                    "gold": 1 if marked else 0,
                    "score": score,
                    "pred": int(score >= threshold),
                }
            )
    return rows


def test_ci_contains_point_and_is_reproducible(tmp_path: Path) -> None:
    predictions = _write_predictions(tmp_path / "predictions_synth.jsonl.gz", _synthetic())
    outputs = []
    for run in (0, 1):
        out = tmp_path / f"ci{run}.json"
        code = bootstrap_ci.main(
            [
                "--predictions",
                str(predictions),
                "--threshold",
                "0.5",
                "--reps",
                "200",
                "--seed",
                "42",
                "--sample",
                "синтетика",
                "--out",
                str(out),
            ]
        )
        assert code == 0
        outputs.append(json.loads(out.read_text(encoding="utf-8")))
    first, second = outputs
    assert first == second, "две тысячи повторов с одним seed'ом обязаны дать один и тот же интервал"
    lower, upper = first["ci95_f1"]
    assert lower <= first["point_f1"] <= upper
    assert first["reps"] == 200 and first["seed"] == 42
    assert first["documents"] == 24 and first["tokens"] == 96
    assert first["threshold_fixed"] is True
    # синтетика разделелима идеально: точка 1,0 и интервал обязан быть узким сверху
    assert first["point_f1"] == pytest.approx(1.0)


def test_document_clusters_change_the_interval_wide_not_narrow(tmp_path: Path) -> None:
    """Все положительные токены — в двух документах: честный интервал обязан быть широким."""
    rows = []
    for doc in range(20):
        for index in range(5):
            marked = doc < 2 and index < 2
            rows.append(
                {
                    "id": f"synth:doc-{doc}",
                    "level": "tokens",
                    "seed": 42,
                    "token": f"t{index}",
                    "start": index,
                    "end": index + 1,
                    "gold": 1 if marked else 0,
                    "score": 0.9 if marked else 0.05,
                    "pred": int(marked),
                }
            )
    predictions = _write_predictions(tmp_path / "predictions_clustered.jsonl.gz", rows)
    out = tmp_path / "ci.json"
    bootstrap_ci.main(
        ["--predictions", str(predictions), "--threshold", "0.5", "--reps", "300", "--seed", "7", "--out", str(out)]
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    lower, upper = payload["ci95_f1"]
    assert lower == pytest.approx(0.0), "при вытягивании других документов F1 обязан падать в ноль"
    assert upper > lower, "интервал не обязан быть вырожденным"
    assert payload["documents"] == 20


def test_manifest_mismatch_is_fatal(tmp_path: Path) -> None:
    rows = _synthetic(docs=6)
    predictions = _write_predictions(tmp_path / "predictions_val_synth.jsonl.gz", rows)
    manifest = tmp_path / "manifest_val.json"
    manifest.write_text(
        json.dumps({"threshold": 0.5, "metrics_mean_std": {"f1": 0.999, "f1_per_seed": [0.5] * 5}}), encoding="utf-8"
    )
    with pytest.raises(SystemExit) as caught:
        bootstrap_ci.main(
            [
                "--predictions",
                str(predictions),
                "--manifest",
                str(manifest),
                "--reps",
                "50",
                "--out",
                str(tmp_path / "ci.json"),
            ]
        )
    assert "не совпал с манифестом" in str(caught.value)


@pytest.mark.parametrize(
    "predictions,manifest,sample",
    [
        (VAL_PREDICTIONS, VAL_MANIFEST, "val (A3 dev)"),
        (TEST_PREDICTIONS, TEST_MANIFEST, "test (v1, A3)"),
    ],
)
def test_committed_artifacts_reproduce_the_manifest_and_cover_the_point(
    predictions: Path, manifest: Path, sample: str, tmp_path: Path
) -> None:
    if not predictions.is_file():
        pytest.skip(f"нет артефакта {predictions.name} — сверка невозможна")
    out = tmp_path / "ci.json"
    code = bootstrap_ci.main(
        [
            "--predictions",
            str(predictions),
            "--manifest",
            str(manifest),
            "--sample",
            sample,
            "--reps",
            "300",
            "--seed",
            "42",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    numbers = json.loads(manifest.read_text(encoding="utf-8"))["metrics_mean_std"]
    assert payload["manifest_cross_check"]["abs_расхождение"] == 0.0
    assert payload["point_f1"] == pytest.approx(float(numbers["f1"]), abs=2e-6)
    lower, upper = payload["ci95_f1"]
    assert lower <= payload["point_f1"] <= upper
    # порог — замороженное число из манифеста, а не выбранное на этой же выборке
    assert payload["threshold"] == pytest.approx(float(numbers["threshold"]), abs=1e-6)
    assert payload["threshold_fixed"] is True
    assert payload["documents"] > 1
