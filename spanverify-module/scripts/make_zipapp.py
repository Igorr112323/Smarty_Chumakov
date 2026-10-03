#!/usr/bin/env python3
"""Сборка одного переносимого файла spanverify.pyz (zipapp).

Конфигурация и калибратор упаковываются внутрь архива, приложение читает их
оттуда, поэтому один файл работает сам по себе (нужен Python 3.10+).

    python scripts/make_zipapp.py            # release/spanverify.pyz
    python scripts/make_zipapp.py --out dist
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipapp
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ("spanverify", "config", "data")


def build(out_dir: Path, name: str = "spanverify.pyz") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / name
    if target.exists():
        target.unlink()

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "app"
        stage.mkdir()
        for item in INCLUDE:
            source = ROOT / item
            if not source.exists():
                continue
            if source.is_dir():
                shutil.copytree(
                    source, stage / item,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
            else:
                shutil.copy2(source, stage / item)
        zipapp.create_archive(stage, target, interpreter="/usr/bin/env python3",
                              main="spanverify.cli:main")

    target.chmod(0o755)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Сборка spanverify.pyz")
    parser.add_argument("--out", type=Path, default=ROOT / "release")
    parser.add_argument("--name", default="spanverify.pyz")
    args = parser.parse_args()

    target = build(args.out, args.name)
    size = target.stat().st_size / 1024
    print(f"Готово: {target} ({size:.0f} КБ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
