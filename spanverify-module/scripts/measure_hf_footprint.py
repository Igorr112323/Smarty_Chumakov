"""Измерение фактических объёмов режима признаков языковой модели (ТЗ п. 4.2).

Договор требует (Приложение № 3, п. 4.2, стр. 12–13): «Для режима признаков
языковой модели потребуются библиотеки Python объёмом около 500 МБ и файл весов
модели объёмом около 530 МБ, сохраняемые локально заранее» и «Объём хранения
составит не более 100 МБ для режима без весов модели и не более 1,5 ГБ для режима
с весами».

Скрипт ничего не оценивает: он измеряет каталоги и записывает байты. Отдельно
считаются

* **библиотеки** — прирост каталога ``site-packages`` относительно базового
  замера, снятого до установки зависимостей режима hf (в этом приросте ровно
  то, что требует режим: torch, transformers, tokenizers, sentencepiece и их
  зависимости);
* **веса** — каталог кеша Hugging Face, куда скачаны только рабочие форматы
  модели (``pytorch_model.bin`` и файлы токенизатора; служебные ``flax_model.msgpack``,
  ``tf_model.h5`` не скачиваются намеренно);
* **поставка** — каталог ``release/`` (или собранный ``.pyz``), то есть то, что
  переносится на компьютер пользователя без весов;
* **хранение** — поставка плюс библиотеки плюс веса.

Запуск (в CI — на раннере, где сеть доступна)::

    python scripts/measure_hf_footprint.py --baseline /tmp/baseline.json \
        --out ../reports/contract --model ai-forever/rugpt3small_based_on_gpt2
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def dir_bytes(path: Path) -> int:
    """Полный размер каталога в байтах (без символических ссылок)."""
    total = 0
    if not path.exists():
        return 0
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except OSError:  # каталог исчез во время обхода
            continue
    return total


def site_packages() -> Path:
    return Path(sysconfig.get_paths()["purelib"])


def git_commit() -> str:
    value = os.environ.get("GITHUB_SHA")
    if value:
        return value
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        return "unknown"


def snapshot(path: Path) -> dict[str, int]:
    """Снимок размеров: общий и по крупнейшим подкаталогам (для протокола)."""
    data: dict[str, int] = {"total": dir_bytes(path)}
    if path.exists():
        children: list[tuple[str, int]] = []
        for child in sorted(path.iterdir()):
            if child.is_dir():
                children.append((child.name, dir_bytes(child)))
        for name, size in sorted(children, key=lambda item: -item[1])[:15]:
            data[name] = size
    return data


def hf_freeze() -> list[str]:
    """Версии пакетов режима hf — чтобы объём был привязан к конкретным версиям."""
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        ).stdout
    except (subprocess.SubprocessError, FileNotFoundError):  # pragma: no cover
        return []
    wanted = (
        "torch",
        "transformers",
        "tokenizers",
        "sentencepiece",
        "safetensors",
        "numpy",
        "huggingface",
        "regex",
        "requests",
    )
    return [
        line.strip()
        for line in out.splitlines()
        if line.strip() and line.split("==")[0].lower().startswith(wanted)
    ]


def mb(value: int) -> float:
    """Мегабайты, как их считает человек в отчёте: 1 МБ = 1 000 000 байт.

    Договор называет величины «около 500 МБ» и «не более 1,5 ГБ» без уточнения
    кратности. Считаем десятичные мегабайты (как в спецификациях размеров
    поставки) и дополнительно печатаем МиБ, чтобы расхождение было видно, а не
    спрятано.
    """
    return round(value / 1_000_000, 1)


def mib(value: int) -> float:
    return round(value / 1024 / 1024, 1)


def build_payload(
    check_id: str,
    metrics: dict[str, object],
    note: str,
    gate: str,
    command: str,
) -> dict[str, object]:
    return {
        "id": check_id,
        "commit": git_commit(),
        "seed": SEED,
        "mode": "hf",
        "dataset": str(metrics.get("model") or ""),
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": command,
        "metrics": metrics,
        "note": note,
        "gate": gate,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Объёмы режима hf: библиотеки и веса")
    parser.add_argument("--baseline", type=Path, default=None, help="замер до установки")
    parser.add_argument("--out", type=Path, required=True, help="каталог артефактов")
    parser.add_argument("--model", default="ai-forever/rugpt3small_based_on_gpt2")
    parser.add_argument("--cache", type=Path, default=None, help="каталог кеша весов")
    parser.add_argument("--delivery", type=Path, default=ROOT / "release")
    args = parser.parse_args()

    cache = args.cache or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    base_total = 0
    if args.baseline and args.baseline.is_file():
        base_total = int(json.loads(args.baseline.read_text(encoding="utf-8"))["total"])

    libs_now = snapshot(site_packages())
    libs_bytes = max(0, libs_now["total"] - base_total)
    weights_bytes = dir_bytes(cache)
    delivery_bytes = dir_bytes(args.delivery)
    storage_with_weights = delivery_bytes + libs_bytes + weights_bytes
    storage_without_weights = delivery_bytes

    files = []
    for item in sorted(cache.rglob("*")):
        if item.is_file():
            files.append(
                {
                    "path": str(item.relative_to(cache)),
                    "bytes": item.stat().st_size,
                }
            )
    files.sort(key=lambda item: -int(item["bytes"]))

    command = (
        "python scripts/measure_hf_footprint.py "
        f"--out ../reports/contract --model {args.model} --cache {cache}"
    )
    common = {
        "model": args.model,
        "cache": str(cache),
        "libs_bytes": libs_bytes,
        "libs_mb": mb(libs_bytes),
        "libs_mib": mib(libs_bytes),
        "weights_bytes": weights_bytes,
        "weights_mb": mb(weights_bytes),
        "weights_mib": mib(weights_bytes),
        "delivery_bytes": delivery_bytes,
        "delivery_mb": mb(delivery_bytes),
        "storage_without_weights_mb": mb(storage_without_weights),
        "storage_with_weights_mb": mb(storage_with_weights),
        "weight_files": files[:20],
        "weight_files_total": len(files),
        "site_packages_top": {
            key: value for key, value in list(libs_now.items())[:10]
        },
        "pip_freeze_hf": hf_freeze(),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    footprint = dict(common)
    footprint.update(
        {
            "contract_libs_mb": 500,
            "contract_weights_mb": 530,
            "libs_vs_contract_percent": round(
                100.0 * libs_bytes / 1_000_000 / 500 - 100.0, 1
            ),
            "weights_vs_contract_percent": round(
                100.0 * weights_bytes / 1_000_000 / 530 - 100.0, 1
            ),
        }
    )
    (args.out / "4.2.5-hf-footprint.json").write_text(
        json.dumps(
            build_payload(
                "4.2.5-hf-footprint",
                footprint,
                "Объёмы измерены командой на раннере: прирост каталога библиотек и "
                "каталог кеша весов. Договор называет ориентиры «около 500 МБ» и "
                "«около 530 МБ»; отклонения приведены отдельными полями.",
                "ТЗ п. 4.2: библиотеки режима признаков около 500 МБ, веса около 530 МБ",
                command,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    storage = dict(common)
    storage.update(
        {
            "limit_without_weights_mb": 100,
            "limit_with_weights_mb": 1500,
            "without_weights_ok": storage_without_weights <= 100_000_000,
            "with_weights_ok": storage_with_weights <= 1_500_000_000,
        }
    )
    (args.out / "4.2.6-storage.json").write_text(
        json.dumps(
            build_payload(
                "4.2.6-storage",
                storage,
                "Хранение без весов — каталог поставки; с весами — поставка плюс "
                "библиотеки режима hf плюс кеш весов.",
                "ТЗ п. 4.2: хранение не более 100 МБ без весов и не более 1,5 ГБ с весами",
                command,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"библиотеки: {mb(libs_bytes)} МБ ({mib(libs_bytes)} МиБ)")
    print(f"веса:       {mb(weights_bytes)} МБ ({mib(weights_bytes)} МиБ), файлов {len(files)}")
    print(f"поставка:   {mb(delivery_bytes)} МБ")
    print(f"хранение с весами: {mb(storage_with_weights)} МБ")
    if os.environ.get("GITHUB_ACTIONS"):
        print(
            f"::notice title=объёмы режима hf::библиотеки {mb(libs_bytes)} МБ, "
            f"веса {mb(weights_bytes)} МБ, поставка {mb(delivery_bytes)} МБ, "
            f"хранение с весами {mb(storage_with_weights)} МБ"
        )
        for item in files[:5]:
            print(
                f"::notice title=файл весов::{item['path']} — {mb(int(item['bytes']))} МБ"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
