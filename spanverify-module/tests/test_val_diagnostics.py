"""Тесты диагностики на валидации (`scripts/val_diagnostics.py`).

Что здесь проверяется и почему именно это:

* фикстура строится так, чтобы один набор колонок сетки *полностью* предсказывал метку, а
  другой был шумом — тогда удаление семейства обязано уронить F1, и тест ловит молчаливую
  подмену «все одинаковые»;
* тестовая часть отвергается и как сплит, и как файл кеша (по имени), до любых вычислений;
* нулевое СКО по seed'ам при замороженном пороге — ожидаемый результат, а не недосмотр:
  он и есть зафиксированное свойство харнесса, из-за которого «5 seed'ов» в v1 ничего не
  измеряли;
* oracle-точка (максимум F1 без ограничения) обязана быть помечена как не-результат, иначе
  diagnostic-число очень легко перенести в отчёт о критерии.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for _path in (str(ROOT), str(SCRIPTS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
SCRIPT = SCRIPTS / "val_diagnostics.py"

from spanverify.core import tokenize_with_offsets  # noqa: E402
from spanverify.hf_grid import GRID_FEATURE_NAMES  # noqa: E402

# Колонки, которые в фикстуре кодируют метку (их удаление обязано уронить F1), и шумовые.
SIGNAL_GROUP = "внимание"
NOISE_GROUP = "плотность опоры"


def _names(group: str) -> list[str]:
    """Имена сетки по группе — из того же словаря, что и docs/ПРИЗНАКИ.md."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("rfd", ROOT / "scripts" / "render_features_doc.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [name for name in GRID_FEATURE_NAMES if module.FEATURE_DOC[name][0] == group]


def _write_pair(pair_id: str, marked: bool, answer: str) -> dict:
    tokens = [token.text for token in tokenize_with_offsets(answer)]
    labels = [[0, len(tokens[0]), 1]] if marked else []
    return {
        "id": pair_id,
        "context": "Регламент 343: срок хранения первичных документов составляет десять лет.",
        "answer": answer,
        "labels": labels,
        "meta": {"kind": "diagnostic", "doc": f"doc-{pair_id[-1]}"},
    }


def _values(pair: dict, signal: bool) -> dict[str, list[float]]:
    count = len(list(tokenize_with_offsets(pair["answer"])))
    first = 1 if (pair["labels"] and signal) else 0
    per_token = [first if index == 0 else (1 - first) * 0 for index in range(count)]
    signal_values = [1.0 if flag else 0.0 for flag in per_token]
    inverted = [1.0 - value for value in signal_values]
    names = _names(SIGNAL_GROUP)
    noise = _names(NOISE_GROUP)
    arrays = {name: signal_values for name in names}
    arrays.update({name: inverted[:count] for name in names if name.endswith(("_first", "_middle"))})
    arrays.update({name: [0.5] * count for name in noise})
    for name in GRID_FEATURE_NAMES:
        arrays.setdefault(name, [0.25] * count)
    return arrays


def _write_dataset(tmp_path: Path, *, cache_name: str = "grid_all_00.jsonl") -> tuple[Path, Path]:
    splits = tmp_path / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    cache = tmp_path / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    pairs = [
        _write_pair("pair-0", True, "три года"),
        _write_pair("pair-1", False, "десять лет"),
        _write_pair("pair-2", True, "пять лет"),
        _write_pair("pair-3", False, "семь лет"),
    ]
    rows = [(pair, _values(pair, signal=True)) for pair in pairs]
    for name, subset in (("train.jsonl", rows), ("dev.jsonl", rows)):
        with (splits / name).open("w", encoding="utf-8") as handle:
            for pair, _ in subset:
                handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    with (cache / cache_name).open("w", encoding="utf-8") as handle:
        for pair, arrays in rows:
            handle.write(
                json.dumps(
                    {
                        "pair_id": pair["id"],
                        "grid_format": 3,
                        "model": "test/model",
                        "revision": "0" * 40,
                        "arrays": arrays,
                        "tokens": [
                            text
                            for text, _ in (
                                (t, 0) for t in [token.text for token in tokenize_with_offsets(pair["answer"])]
                            )
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    config = {
        "protocol": "test-diagnostics",
        "seed": 42,
        "experiments": [
            {
                "name": "fixture",
                "hypothesis": "фикстура: сигнал в группах «внимание», шум в «плотность опоры»",
                "change": "нет",
                "features": list(GRID_FEATURE_NAMES),
                "classifier": "logreg",
                "window": 0,
                "merge_gap": 2,
                "epochs": 60,
                "l2": 0.01,
                "folds": 5,
            }
        ],
    }
    config_path = tmp_path / "hf_experiments.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    freeze = tmp_path / "freeze_manifest.json"
    freeze.write_text(json.dumps({"threshold": 0.5}), encoding="utf-8")
    return splits, config_path


def run(tmp_path: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    splits, config = _write_dataset(tmp_path)
    argv = [
        sys.executable,
        str(SCRIPT),
        "--grid-cache",
        str(tmp_path / "cache"),
        "--splits",
        str(splits),
        "--config",
        str(config),
        "--baseline",
        "fixture",
        "--threshold-manifest",
        str(tmp_path / "freeze_manifest.json"),
        "--seeds",
        "42,43",
        "--out-json",
        str(tmp_path / "diag.json"),
        "--out-md",
        str(tmp_path / "diag.md"),
        *extra,
    ]
    return subprocess.run(argv, capture_output=True, text=True, cwd=ROOT, check=False)


def test_diagnostics_reports_point_frontier_and_deltas(tmp_path: Path) -> None:
    proc = run(tmp_path)
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined[-1500:]
    payload = json.loads((tmp_path / "diag.json").read_text(encoding="utf-8"))
    assert payload["test_not_read"] is True and payload["not_for_criterion"] is True
    results = payload["results"]
    baseline = results["baseline"]
    assert (
        baseline["token_f1"] is not None and baseline["token_f1"] > 0.5
    ), "фикстура обязана различаться: F1 базовой линии должен быть выше 0,5"
    assert baseline["token_auc"] > 0.9, "в фикстуре сигнал детерминирован, AUC обязана быть высокой"
    front = results["frontier"]
    assert front["unconstrained"]["f1"] + 1e-9 >= front["capped"]["f1"], "снять ограничение нельзя на ухудшение"
    assert "oracle" in front["unconstrained"]["warning"]
    assert results["smoothing"], "блок сглаживания пуст"
    assert {item["window"] for item in results["smoothing"]} == {0, 1, 2, 3}
    text = (tmp_path / "diag.md").read_text(encoding="utf-8")
    assert "не читалась" in text
    assert "Вклад семейств" in text


def test_removing_signal_family_is_measurable(tmp_path: Path) -> None:
    proc = run(tmp_path, "--ablate", "baseline")
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-1500:]
    payload = json.loads((tmp_path / "diag.json").read_text(encoding="utf-8"))
    ablations = {item["family"]: item for item in payload["results"]["ablations"]}
    assert SIGNAL_GROUP in ablations and NOISE_GROUP in ablations, sorted(ablations)
    assert ablations[SIGNAL_GROUP]["delta_f1"] < 0, "убрать единственный сигнальный family — F1 обязан упасть"
    assert abs(ablations[NOISE_GROUP]["delta_f1"]) < 0.35, "удаление шума не должно валить метрику в ноль"


def test_zero_seed_spread_is_expected_and_explained(tmp_path: Path) -> None:
    proc = run(tmp_path)
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-1500:]
    payload = json.loads((tmp_path / "diag.json").read_text(encoding="utf-8"))
    seeds = payload["results"]["seeds"]
    assert seeds["std_token_f1"] == 0.0, "при замороженном пороге и детерминированном logreg СКО обязано быть 0"
    assert len(seeds["model_sha256_unique"]) == 1
    assert "детерминирован" in seeds["note"], "нулевой разброс обязан быть объяснён в артефакте, а не молчать"


def test_cache_from_test_split_is_refused(tmp_path: Path) -> None:
    splits, config = _write_dataset(tmp_path, cache_name="grid_test_a3_00.jsonl")
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--grid-cache",
            str(tmp_path / "cache"),
            "--splits",
            str(splits),
            "--config",
            str(config),
            "--baseline",
            "fixture",
            "--out-json",
            str(tmp_path / "diag.json"),
            "--out-md",
            str(tmp_path / "diag.md"),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert proc.returncode != 0, "кеш тестовой выборки принят"
    assert "только на val" in (proc.stdout + proc.stderr)
    assert not (tmp_path / "diag.json").exists(), "файлы обязаны не создаваться до отказа"


def test_unknown_baseline_name_is_refused(tmp_path: Path) -> None:
    splits, config = _write_dataset(tmp_path)
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--grid-cache",
            str(tmp_path / "cache"),
            "--splits",
            str(splits),
            "--config",
            str(config),
            "--baseline",
            "нет-такой-строки",
            "--out-json",
            str(tmp_path / "diag.json"),
            "--out-md",
            str(tmp_path / "diag.md"),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert proc.returncode != 0
    assert "базовой линии" in (proc.stdout + proc.stderr)


def _payload() -> dict:
    """Синтетический вывод `val_diagnostics.py` — чтобы рендер раздела 3.4 проверялся без CI."""
    summary = {
        "threshold": 0.1106,
        "token_f1": 0.595041,
        "token_precision": 0.70936,
        "token_recall": 0.512456,
        "token_fpr": 0.006565,
        "token_auc": 0.952672,
        "span_f1": 0.6,
        "model_sha256": "a" * 64,
        "threshold_selection": {"mode": "frozen", "threshold": 0.1106},
        "capped_f1": 0.6041,
        "capped_fpr": 0.39,
    }
    return {
        "protocol": "val-diagnostics-v1",
        "not_for_criterion": True,
        "test_not_read": True,
        "split": {"name": "val (A3 dev)", "file": "x/dev.jsonl", "pairs": 181, "tokens": 7969},
        "labels": {"positive_tokens": 287, "negative_tokens": 7682},
        "baseline": "wide-linear",
        "threshold_source": "заморожен по манифесту шага 3",
        "fpr_cap": 0.4,
        "inputs": {"cache_files": ["grid_all_00.jsonl"], "splits_sha256": "cba705f8aaaa"},
        "code_commit": "5d53c7d",
        "results": {
            "baseline": dict(summary, name="wide-linear", delta_f1=0.0),
            "frontier": {
                "distinct_candidates": 512,
                "capped": {"f1": 0.6041, "fpr": 0.39, "threshold": 0.04},
                "unconstrained": {"f1": 0.61, "fpr": 0.44, "threshold": 0.03},
                "protocol_point": {"f1": 0.595041, "fpr": 0.006565, "threshold": 0.1106},
                "gap_to_criterion_on_val": 0.004959,
            },
            "smoothing": [dict(summary, window=window, delta_f1=0.0, delta_capped_f1=0.0) for window in (0, 1, 2, 3)],
            "ablations": [
                {
                    **summary,
                    "family": "entropy",
                    "removed": ["entropy_last", "entropy_first"],
                    "kept": 34,
                    "delta_f1": -0.02,
                    "delta_capped_f1": -0.02,
                }
            ],
            "seeds": {
                "requested": [42, 43],
                "std_token_f1": 0.0,
                "std_threshold": 0.0,
                "model_sha256_unique": ["a" * 64],
            },
        },
    }


def test_render_section_34_uses_artifact_numbers(tmp_path, monkeypatch) -> None:
    import json

    import render_features_doc as rfd

    artifact = tmp_path / "val_diagnostics.json"
    artifact.write_text(json.dumps(_payload(), ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(rfd, "DIAGNOSTICS", artifact)
    text = "\n".join(rfd.diagnostics_section())
    assert text.startswith("### 3.4.")
    assert "0,595041" in text, "F1 базовой линии не попала в раздел"
    assert "0,95267" in text, "AUC базовой линии не попала в раздел"
    assert "entropy" in text and "-0,020000" in text, "вклад семейства не попал в раздел"
    assert "Не измерено" not in text


def test_render_section_34_stays_silent_without_artifact(tmp_path, monkeypatch) -> None:
    import render_features_doc as rfd

    monkeypatch.setattr(rfd, "DIAGNOSTICS", tmp_path / "нет-такого-файла.json")
    text = "\n".join(rfd.diagnostics_section())
    assert "**Не измерено.**" in text, "без артефакта раздел обязан молчать, а не оценивать"
    assert "val_diagnostics.py" in text and "hf-val-diagnostics.yml" in text
