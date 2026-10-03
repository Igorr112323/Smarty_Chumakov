"""Тесты конфигурации."""

from __future__ import annotations

import json

import pytest

from spanverify.config import DEFAULTS, Config


def test_defaults_are_loaded():
    config = Config()
    assert config.backend == "surrogate"
    assert config.k_neighbors == 5
    assert 0.0 < config.threshold <= 1.0


def test_from_dict_ignores_unknown_keys_and_casts_types():
    config = Config.from_dict({"backend": "hf", "k_neighbors": "7", "threshold": "0.4", "нет": 1})
    assert config.backend == "hf"
    assert config.k_neighbors == 7
    assert config.threshold == pytest.approx(0.4)


def test_with_overrides_ignores_none_and_keeps_instance():
    base = Config()
    updated = base.with_overrides(threshold=0.25, host=None)
    assert updated.threshold == pytest.approx(0.25)
    assert updated.host == base.host
    assert base.threshold != updated.threshold  # исходный конфиг не меняется


def test_with_overrides_keeps_untouched_fields():
    """Переопределение одного поля не должно сбрасывать остальные к значениям по умолчанию."""
    base = Config(threshold=0.42, k_neighbors=9, backend="hf")
    updated = base.with_overrides(port=9100)
    assert updated.port == 9100
    assert updated.threshold == pytest.approx(0.42)
    assert updated.k_neighbors == 9
    assert updated.backend == "hf"


def test_with_overrides_ignores_unknown_keys():
    base = Config(threshold=0.31)
    assert base.with_overrides(неизвестное_поле=5) == base


def test_with_overrides_without_arguments_returns_same_values():
    base = Config(backend="hf")
    assert base.with_overrides() == base


def test_load_reads_json_file(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"backend": "hf", "port": 9100}), encoding="utf-8")
    config = Config.load(path)
    assert config.backend == "hf"
    assert config.port == 9100


def test_load_ignores_missing_file(tmp_path):
    config = Config.load(tmp_path / "нет-файла.json")
    assert config == Config()


def test_environment_overrides_file(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"k_neighbors": 3}), encoding="utf-8")
    monkeypatch.setenv("SPANVERIFY_K_NEIGHBORS", "11")
    config = Config.load(path)
    assert config.k_neighbors == 11


def test_save_creates_parent_directories(tmp_path):
    path = tmp_path / "a" / "b" / "config.json"
    Config().save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(DEFAULTS).issubset(data.keys())


def test_describe_mentions_key_parameters():
    text = Config(backend="hf", k_neighbors=7).describe()
    assert "hf" in text and "7" in text


def test_runtime_roots_include_frozen_bundle(monkeypatch, tmp_path):
    """В собранном exe конфиг ищется рядом с бинарником и в бандле."""
    import sys as _sys

    from spanverify.config import resolve_runtime_path, runtime_roots

    bundle = tmp_path / "bundle"
    (bundle / "config").mkdir(parents=True)
    (bundle / "config" / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(_sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setattr(_sys, "frozen", True, raising=False)
    monkeypatch.chdir(tmp_path)
    assert str(bundle) in [str(root) for root in runtime_roots()]
    assert resolve_runtime_path("config/config.json") == bundle / "config" / "config.json"


def test_resolve_runtime_path_falls_back(monkeypatch, tmp_path):
    import sys as _sys

    from spanverify.config import resolve_runtime_path

    monkeypatch.setattr(_sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.delattr(_sys, "frozen", raising=False)
    monkeypatch.chdir(tmp_path)
    resolved = resolve_runtime_path("config/absent.json")
    assert resolved.name == "absent.json" and not resolved.exists()


def test_detector_loads_calibrator_from_bundle(monkeypatch, tmp_path):
    """Калибратор подхватывается из бандла приложения (onefile exe)."""
    import json
    import sys as _sys

    from spanverify import Config, Detector
    from spanverify.calibration import IsotonicCalibrator

    bundle = tmp_path / "bundle"
    (bundle / "config").mkdir(parents=True)
    calibrator = IsotonicCalibrator.fit([0.1, 0.5, 0.9], [0, 0, 1])
    (bundle / "config" / "calibration.json").write_text(
        json.dumps(calibrator.to_dict()), encoding="utf-8"
    )
    monkeypatch.setattr(_sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setattr(_sys, "frozen", True, raising=False)
    monkeypatch.chdir(tmp_path)

    detector = Detector(Config(backend="surrogate"))
    assert detector.calibrator is not None


def test_all_defaults_map_to_config_fields():
    known = set(Config().to_dict())
    assert set(DEFAULTS) == known
