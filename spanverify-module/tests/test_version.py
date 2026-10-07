"""Версия релиза одна: пакет и метаданные сборки не имеют права разъехаться.

Причина проверки: релиз публикуется с одним номером; если ``_version.py`` и
``pyproject.toml`` разойдутся, собранный артефакт будет называться не так, как
его описывают документы и тег релиза.
"""

from __future__ import annotations

import re
from pathlib import Path

from spanverify import __version__

MODULE_ROOT = Path(__file__).resolve().parents[1]


def test_package_version_matches_pyproject() -> None:
    """Один номер версии: источник в пакете совпадает с метаданными сборки."""
    pyproject = (MODULE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    assert match, "в pyproject.toml не найдена строка version"
    assert match.group(1) == __version__, (
        f"версия пакета {__version__} не совпадает с pyproject.toml ({match.group(1)}): "
        "релиз обязан быть одним номером"
    )
