"""Тесты загрузки обученных параметров: приоритет источников и смысл source.

Проверяется именно то, от чего зависит обещание «.exe работает автономно»:
встроенные в бандл веса не должны подменяться случайным файлом из текущей
папки, но осознанно положенные рядом с приложением — должны иметь приоритет.
"""

from __future__ import annotations

import json
import sys

from spanverify.engine import WeightsBundle


def _write_weights(path, threshold, mass=0.8):
    """Записать минимальный корректный файл параметров и вернуть путь."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "weights": {
                    "attention_entropy": round(1 - mass, 4),
                    "ctx_attention_mass": mass,
                    "embedding_density": 0.0,
                },
                "threshold": threshold,
                "span_z": 0.0,
                "span_floor": 0.3,
                "span_cap": 0.5,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_weights_from_bundle_win_over_current_directory(tmp_path, monkeypatch):
    """Собранное приложение берёт веса из бандла, даже если в текущей папке есть config/."""
    bundle_dir = tmp_path / "meipass"
    _write_weights(bundle_dir / "config" / "weights.json", threshold=0.7)
    work_dir = tmp_path / "cwd"
    _write_weights(work_dir / "config" / "weights.json", threshold=0.1, mass=0.0)
    monkeypatch.chdir(work_dir)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle_dir), raising=False)

    bundle = WeightsBundle.load()

    assert bundle.source == "embedded", "веса должны быть признаны встроенными"
    assert bundle.threshold == 0.7, "порог взят не из бандла"
    assert bundle.weights["ctx_attention_mass"] == 0.8, "веса признаков взяты не из бандла"


def test_weights_near_executable_win_over_bundle(tmp_path, monkeypatch):
    """Параметры, обученные пользователем рядом с приложением, важнее встроенных."""
    bundle_dir = tmp_path / "meipass"
    _write_weights(bundle_dir / "config" / "weights.json", threshold=0.7)
    exe_dir = tmp_path / "release"
    _write_weights(exe_dir / "config" / "weights.json", threshold=0.42, mass=0.5)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle_dir), raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe_dir / "spanverify.exe"), raising=False)

    bundle = WeightsBundle.load()

    assert bundle.source == "disk", "параметры рядом с приложением должны иметь приоритет"
    assert bundle.threshold == 0.42


def test_weights_fall_back_to_defaults_when_nothing_found(tmp_path):
    """Если файла параметров нет, приложение честно помечает их как значения по умолчанию."""
    bundle = WeightsBundle.load(tmp_path / "нет-такого-файла.json")

    assert bundle.source == "defaults"
    assert bundle.loaded is False, "значения по умолчанию нельзя считать обученными"
