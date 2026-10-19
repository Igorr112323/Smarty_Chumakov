#!/usr/bin/env python3
"""Обновить ``reports/CURRENT_METRICS.json``, не потеряв чужих прогонов.

Зачем отдельный шаг. ``scripts/collect_metrics.py`` собирает единый файл чисел из
артефактов прогонов. После изменения кода (признаки, знаки, нормировка) числа
демонстрационного корпуса, корпусов A и A3 меняются — их надо пересобрать. Но
часть блоков собирается из артефактов, которых на локальной машине нет: пилот на
реальной модели, прогоны режима ``hf``, внешние срезы. Простой перезапуск
``collect_metrics.py`` на такой машине **затёр бы** их значения (блок превратился
бы в ``null``) — это потеря честных измерений, а не обновление.

Правило этого скрипта:

* блок пересобирается, если свежий сбор дал для него значения;
* блок сохраняется из прежнего файла, если свежий сбор его не дал (нет
  артефакта) — и это явно записывается в ``meta.refresh.preserved`` вместе с
  коммитом, на котором блок был получен;
* ничего не считается руками: значения обоих наборов приходят из
  ``collect_metrics.py`` и из прежнего файла.

Запуск::

    python scripts/refresh_metrics.py --dry-run     # показать, что изменится
    python scripts/refresh_metrics.py               # обновить reports/CURRENT_METRICS.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
METRICS = ROOT / "reports" / "CURRENT_METRICS.json"

#: Блоки, которые на локальной машине без весов модели и без артефактов CI
#: пересобрать нельзя. Для них сохраняется прежнее значение.
PRESERVABLE = ("pilot", "hf_experiments", "hf_external_slices", "external_tests", "release", "defects")


def _empty(value: Any) -> bool:
    """Пустой ли блок: None, пустой словарь или словарь из ``available: false``."""
    if value in (None, {}, []):
        return True
    if isinstance(value, dict):
        if value.get("available") is False:
            return True
        return not any(not _empty(item) for item in value.values())
    return False


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        return ""


def collect(target: Path) -> dict:
    """Собрать свежий файл чисел во временный путь тем же скриптом, что и CI."""
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "collect_metrics.py"),
            "--out",
            str(target),
            "--coverage-json",
            str(ROOT / "reports" / "coverage.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=3600,
        env={**__import__("os").environ, "PYTHONUTF8": "1"},
    )
    if proc.returncode != 0:
        raise SystemExit(f"collect_metrics.py упал:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    return json.loads(target.read_text(encoding="utf-8"))


def merge(old: dict, fresh: dict) -> tuple[dict, list[str], list[str]]:
    """Свежие блоки + сохранённые чужие прогоны; вернуть (файл, обновлено, сохранено)."""
    merged = dict(fresh)
    refreshed: list[str] = []
    preserved: list[str] = []
    old_commit = (old.get("meta") or {}).get("commit", "unknown")
    for name, value in old.items():
        if name in ("meta", "commands"):
            continue
        if name not in merged:
            merged[name] = value
            preserved.append(name)
            continue
        if not _empty(merged.get(name)):
            if merged.get(name) != value:
                refreshed.append(name)
            continue
        if _empty(value):
            continue
        # Свежий сбор пуст, а прежний содержит измерение: сохраняем прежнее.
        merged[name] = value
        preserved.append(name)
    meta = dict(merged.get("meta") or {})
    meta["refresh"] = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "head_commit": _git("rev-parse", "HEAD"),
        "worktree": "dirty" if _git("status", "--porcelain") else "clean",
        "preserved_from_commit": old_commit,
        "preserved_blocks": sorted(preserved),
        "refreshed_blocks": sorted(refreshed),
        "rule": (
            "блоки пересобраны из артефактов прогонов; блоки, артефактов которых нет на этой машине, "
            "сохранены из прежнего файла и перечислены в preserved_blocks"
        ),
    }
    merged["meta"] = meta
    return merged, sorted(refreshed), sorted(preserved)


def main() -> int:
    parser = argparse.ArgumentParser(description="Обновить reports/CURRENT_METRICS.json без потери чужих прогонов")
    parser.add_argument("--dry-run", action="store_true", help="показать различия и выйти")
    parser.add_argument("--out", default=str(METRICS))
    args = parser.parse_args()

    old = json.loads(METRICS.read_text(encoding="utf-8")) if METRICS.is_file() else {}
    tmp = Path("/tmp/spanverify-metrics-fresh.json")
    fresh = collect(tmp)
    merged, refreshed, preserved = merge(old, fresh)

    print(f"обновлено блоков: {refreshed or '—'}")
    print(f"сохранено блоков (нет артефактов на этой машине): {preserved or '—'}")
    if args.dry_run:
        return 0
    out = Path(args.out)
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"записано: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
