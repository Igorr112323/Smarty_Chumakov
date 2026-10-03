"""Тесты честного разделения корпуса и единого источника версии.

Дефект, который здесь закрывается: построчный шаффл оставлял один и тот же
субъект и шаблон вопроса и в обучении, и в «отложенной» части, поэтому метрика
переставала быть отложенной. Официальное разделение — групповое.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

from spanverify import __version__
from spanverify.dataset import generate_pairs
from spanverify.train import _split_pairs, shared_groups, train

ROOT = Path(__file__).resolve().parent.parent


def test_group_split_has_no_shared_groups() -> None:
    """Групповое разделение не пересекается по субъекту и шаблону."""
    pairs = [pair.to_dict() for pair in generate_pairs(n_pairs=80, seed=1312)]
    train_pairs, test_pairs = _split_pairs(pairs, test_size=0.3, seed=42, group=True)
    assert shared_groups(train_pairs, test_pairs) == 0
    assert train_pairs and test_pairs


def test_row_split_demonstrates_old_defect() -> None:
    """Построчный шаффл (старое поведение) действительно даёт пересечение групп."""
    pairs = [pair.to_dict() for pair in generate_pairs(n_pairs=80, seed=1312)]
    train_pairs, test_pairs = _split_pairs(pairs, test_size=0.3, seed=42, group=False)
    assert shared_groups(train_pairs, test_pairs) > 0


def test_train_uses_group_split_by_default() -> None:
    """Обучение по умолчанию групповое, и в отчёте это зафиксировано."""
    default = inspect.signature(train).parameters["group_split"].default
    assert default is True
    report = train([pair.to_dict() for pair in generate_pairs(n_pairs=60, seed=99)], seed=42)
    split = report.stats["split"]
    assert split["grouped"] is True
    assert split["shared_groups"] == 0


def test_version_matches_pyproject_and_weights() -> None:
    """Версия пакета, pyproject и обученных весов — одна и та же."""
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, flags=re.MULTILINE)
    assert match, "в pyproject.toml нет поля version"
    assert match.group(1) == __version__
    weights = json.loads((ROOT / "config" / "weights.json").read_text(encoding="utf-8"))
    assert (
        weights.get("version") == __version__
    ), f"веса обучены версией {weights.get('version')}, пакет — {__version__}"


def test_participation_artifact_version_matches() -> None:
    """Артефакт оценки участия ИИ тоже помечен текущей версией."""
    payload = json.loads((ROOT / "config" / "participation.json").read_text(encoding="utf-8"))
    assert payload.get("version") == __version__
    assert payload.get("calibrated_on", "").startswith("synthetic")
