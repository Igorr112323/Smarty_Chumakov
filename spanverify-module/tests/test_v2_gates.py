"""Гейты шага 3 цикла v2: замереть можно только поверх выполненного шага 2.

Смысл этих тестов — не «код работает», а «код отказывается работать в неправильном
порядке». Каждый отказ проверяется отдельно, иначе любой из них мог бы молча
превратиться в пропуск, и шаг 3 запустился бы без отбора, без порога или без
разброса seed'ов.
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import hf_protocol_v2 as v2  # noqa: E402
from test_v2_harness import write_pairs  # noqa: E402


def make_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Каталог с разбиением из трёх пар и пустым полом объёма (иначе пол заслоняет гейты)."""
    monkeypatch.setattr(v2, "FLOORS", {"train": (0, 0), "val": (0, 0), "test": (0, 0)})
    split_dir = tmp_path / "splits"
    write_pairs(split_dir, {"train": 3, "val": 3, "test": 3})
    return split_dir


def final_args(tmp_path: Path, split_dir: Path, config_path: Path, **overrides: object) -> Namespace:
    payload = {
        "config": str(config_path),
        "splits": str(split_dir),
        "grid_cache": str(tmp_path / "grid"),
        "out": str(tmp_path / "reports"),
        "corpus_name": "v2",
        "seeds": "42,43,44,45,46",
        "include_val": True,
        "freeze": str(tmp_path / "reports" / "freeze.json"),
        "sweep": str(tmp_path / "reports" / "sweep.json"),
        "seedcheck": str(tmp_path / "reports" / "seedcheck.json"),
        "allow_zero_std": "",
    }
    payload.update(overrides)
    return Namespace(**payload)


def write_final_config(path: Path, name: str = "wide-linear-v2", step: str = "final-config-v2") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "step": step,
                "protocol": v2.PROTOCOL,
                "name": name,
                "features": ["entropy_last", "mass_last"],
                "classifier": "logreg",
                "window": 0,
                "merge_gap": 2,
                "params": {"epochs": 10, "l2": 0.1, "folds": 5},
                "threshold": 0.55,
                "threshold_rule": "E1",
                "seeds": [42, 43, 44, 45, 46],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def freeze_ok(tmp_path: Path, split_dir: Path) -> Path:
    from hf_protocol import sha256_file

    return write_json(
        tmp_path / "reports" / "freeze.json",
        {
            "protocol": v2.PROTOCOL,
            "splits_sha256": {part: sha256_file(split_dir / name) for part, name in parts_files(split_dir)},
            "independence": {"overlap_total": 0},
        },
    )


def parts_files(split_dir: Path):
    yield "train", "train.jsonl"
    yield "val", (Path(v1_val_name(split_dir)))
    yield "test", "test.jsonl"


def v1_val_name(split_dir: Path) -> str:
    return "dev.jsonl" if (split_dir / "dev.jsonl").is_file() else "val.jsonl"


def sweep_ok(
    tmp_path: Path, name: str = "wide-linear-v2", triggered: bool = False, selected: str | None = None
) -> Path:
    return write_json(
        tmp_path / "reports" / "sweep.json",
        {
            "protocol": v2.PROTOCOL,
            "selection_rule": "максимум F1 на val при FPR ≤ 0,40",
            "selected": name if selected is None else selected,
            "stop_rule": {"triggered": triggered},
            "results": [
                {
                    "number": 2,
                    "name": name,
                    "features_count": 2,
                    "e1_val_chosen": {"threshold": 0.55, "f1": 0.71, "fpr": 0.2},
                    "e0_train_cv": {"threshold": 0.44, "f1": 0.66, "fpr": 0.18},
                }
            ],
        },
    )


def seedcheck_ok(tmp_path: Path, std: float = 0.011) -> Path:
    return write_json(
        tmp_path / "reports" / "seedcheck.json",
        {"protocol": v2.PROTOCOL, "f1_std": std, "distinct_f1_values": 5, "bootstrap_docs": True},
    )


def run_final(args: Namespace) -> tuple[int, str]:
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = v2.command_final(args)
    return code, buffer.getvalue()


def test_final_refuses_without_step_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    split_dir = make_workspace(tmp_path, monkeypatch)
    config = write_final_config(tmp_path / "config" / "hf_final_config_v2.json")
    code, text = run_final(final_args(tmp_path, split_dir, config))
    assert code == 1
    assert "нет файла заморозки" in text, "шаг 3 без шага 0 запрещён"


def test_final_refuses_when_independence_broken(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    split_dir = make_workspace(tmp_path, monkeypatch)
    config = write_final_config(tmp_path / "config" / "hf_final_config_v2.json")
    freeze_ok(tmp_path, split_dir)
    freeze = json.loads((tmp_path / "reports" / "freeze.json").read_text(encoding="utf-8"))
    freeze["independence"]["overlap_total"] = 7
    write_json(tmp_path / "reports" / "freeze.json", freeze)
    sweep_ok(tmp_path)
    seedcheck_ok(tmp_path)
    code, text = run_final(final_args(tmp_path, split_dir, config))
    assert code == 1 and "пересечение с v1 = 7" in text


def test_final_refuses_on_changed_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Подмена тестовой выборки после заморозки обязана падать, а не измеряться."""
    split_dir = make_workspace(tmp_path, monkeypatch)
    config = write_final_config(tmp_path / "config" / "hf_final_config_v2.json")
    freeze_ok(tmp_path, split_dir)
    # Подмена теста после заморозки: файл остаётся валидным (пустой файл упал бы на
    # формате, и проверка хеша не была бы проверкой), но содержимое уже другое.
    extra = {
        "id": "test-99",
        "context": "чужой документ",
        "answer": "значение 7 дней",
        "labels": [[8, 15, 1]],
        "meta": {"group": "doc-подмена"},
    }
    with (split_dir / "test.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(extra, ensure_ascii=False) + "\n")
    sweep_ok(tmp_path)
    seedcheck_ok(tmp_path)
    code, text = run_final(final_args(tmp_path, split_dir, config))
    assert code == 1 and "!= зафиксированный на шаге 0" in text


def test_final_refuses_when_stop_rule_fired(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    split_dir = make_workspace(tmp_path, monkeypatch)
    config = write_final_config(tmp_path / "config" / "hf_final_config_v2.json")
    freeze_ok(tmp_path, split_dir)
    sweep_ok(tmp_path, triggered=True)
    seedcheck_ok(tmp_path)
    code, text = run_final(final_args(tmp_path, split_dir, config))
    assert code == 1
    assert "СТОП-ПРАВИЛО ШАГА 2" in text and "измерение" in text.lower()


def test_final_refuses_unselected_config_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    split_dir = make_workspace(tmp_path, monkeypatch)
    config = write_final_config(tmp_path / "config" / "hf_final_config_v2.json", name="мечта-заказчика")
    freeze_ok(tmp_path, split_dir)
    sweep_ok(tmp_path, name="wide-linear-v2")
    seedcheck_ok(tmp_path)
    code, text = run_final(final_args(tmp_path, split_dir, config))
    assert code == 1 and "!= отобранная на val" in text


def test_final_refuses_different_feature_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    split_dir = make_workspace(tmp_path, monkeypatch)
    config = write_final_config(tmp_path / "config" / "hf_final_config_v2.json")
    config.write_text(
        json.dumps({**json.loads(config.read_text(encoding="utf-8")), "features": ["entropy_last"]}), encoding="utf-8"
    )
    freeze_ok(tmp_path, split_dir)
    sweep_ok(tmp_path)
    seedcheck_ok(tmp_path)
    code, text = run_final(final_args(tmp_path, split_dir, config))
    assert code == 1 and "состав признаков" in text


def test_final_refuses_without_seed_variance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    split_dir = make_workspace(tmp_path, monkeypatch)
    config = write_final_config(tmp_path / "config" / "hf_final_config_v2.json")
    freeze_ok(tmp_path, split_dir)
    sweep_ok(tmp_path)
    code, text = run_final(final_args(tmp_path, split_dir, config))
    assert code == 1 and "нет проверки seed'ов" in text


def test_final_refuses_zero_std_without_an_explicit_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    split_dir = make_workspace(tmp_path, monkeypatch)
    config = write_final_config(tmp_path / "config" / "hf_final_config_v2.json")
    freeze_ok(tmp_path, split_dir)
    sweep_ok(tmp_path)
    seedcheck_ok(tmp_path, std=0.0)
    code, text = run_final(final_args(tmp_path, split_dir, config))
    assert code == 1 and "std(F1) на val равен 0" in text


def test_zero_std_accepted_only_with_a_recorded_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Явная причина не отменяет гейт, а записывает его обход в манифест — поэтому
    сама проверка «гейты пройдены» выглядит как выход на кеш (его нет)."""
    split_dir = make_workspace(tmp_path, monkeypatch)
    config = write_final_config(tmp_path / "config" / "hf_final_config_v2.json")
    freeze_ok(tmp_path, split_dir)
    sweep_ok(tmp_path)
    seedcheck_ok(tmp_path, std=0.0)
    with pytest.raises(SystemExit, match="кеш"):
        run_final(final_args(tmp_path, split_dir, config, allow_zero_std="модель детерминирована полностью"))


def test_all_gates_pass_reaches_the_missing_grid_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Если все гейты пройдены, дальше обязан быть кеш сетки — порядок проверок зафиксирован."""
    split_dir = make_workspace(tmp_path, monkeypatch)
    config = write_final_config(tmp_path / "config" / "hf_final_config_v2.json")
    freeze_ok(tmp_path, split_dir)
    sweep_ok(tmp_path)
    seedcheck_ok(tmp_path)
    with pytest.raises(SystemExit, match="кеш"):
        run_final(final_args(tmp_path, split_dir, config))


def test_final_refuses_a_config_that_was_not_marked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    split_dir = make_workspace(tmp_path, monkeypatch)
    config = write_final_config(tmp_path / "config" / "hf_final_config_v2.json", step="just-a-config")
    with pytest.raises(SystemExit, match="final-config-v2"):
        run_final(final_args(tmp_path, split_dir, config))


def test_freeze_on_a_measured_corpus_reports_broken_independence(tmp_path: Path, capsys) -> None:
    """Шаг 0 на A3 обязан сказать «независимость нарушена» — corpus v1 уже измерен."""
    args = Namespace(
        splits=str(ROOT / "data" / "corpus_a3" / "splits"),
        config=str(ROOT / "config" / "hf_experiments_v2.json"),
        seeds="42,43,44,45,46",
        out=str(tmp_path / "отчёт"),
    )
    code = v2.command_freeze(args)
    text = capsys.readouterr().out
    assert code == 1
    assert "независимость нарушена" in text
    assert json.loads((tmp_path / "отчёт" / "freeze.json").read_text(encoding="utf-8"))["problems"]
