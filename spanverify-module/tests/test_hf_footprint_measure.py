"""Проверки измерителя объёмов режима hf (``scripts/measure_hf_footprint.py``).

Скрипт не запускается в CI-тестах целиком (ему нужны torch и кеш весов), но его
арифметика обязана быть проверяемой: от неё зависят числа приёмки по ТЗ п. 4.2.
Здесь проверяются три вещи:

1. урезание считается по расширениям и каталогам, а не «на глаз»: заголовки,
   метаданные, кеши и заглушки типов не попадают в объём, а модули и разделяемые
   библиотеки — попадают;
2. жёсткие ссылки одного файла не удваивают объём (иначе кеш весов раздувается
   вдвое, ведь Hugging Face раскладывает один файл в ``blobs`` и ``snapshots``);
3. десятичные мегабайты и МиБ считаются раздельно: договор называет «МБ», и
   подмена одной величины другой запрещена.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]


def _load_measure():
    """Загрузить скрипт измерения по пути (в ``scripts/`` нет пакета)."""
    path = MODULE_ROOT / "scripts" / "measure_hf_footprint.py"
    spec = importlib.util.spec_from_file_location("measure_hf_footprint", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("measure_hf_footprint", module)
    spec.loader.exec_module(module)
    return module


def test_prune_keeps_loadable_files(tmp_path: Path) -> None:
    """В объём библиотек входят модули и .so, но не заголовки и метаданные."""
    measure = _load_measure()
    pkg = tmp_path / "torch"
    (pkg / "lib").mkdir(parents=True)
    (pkg / "include").mkdir()
    (pkg / "custom.dist-info").mkdir()
    (pkg / "lib" / "libtorch.so").write_bytes(b"a" * 1000)
    (pkg / "mod.py").write_bytes(b"b" * 100)
    (pkg / "include" / "at.h").write_bytes(b"c" * 5000)
    (pkg / "stub.pyi").write_bytes(b"d" * 300)
    (pkg / "custom.dist-info" / "METADATA").write_text("x" * 700, encoding="utf-8")

    full, _ = measure.package_bytes(tmp_path, "torch", prune=False)
    pruned, dropped = measure.package_bytes(tmp_path, "torch", prune=True)

    assert full == 7100
    assert pruned == 1100, "модуль и разделяемая библиотека обязаны остаться в объёме"
    grouped = dict(dropped)
    assert sum(grouped.values()) == 6000
    assert any("заголовки" in name for name in grouped)
    assert any("нерабочие каталоги" in name for name in grouped)
    assert any("метаданные" in name for name in grouped)


def test_prune_suffixes_do_not_touch_python_or_binaries() -> None:
    """Запрещено урезать то, что Python загружает: .py, .so, .pyd, .json."""
    measure = _load_measure()
    for suffix in (".py", ".so", ".pyd", ".json", ".bin", ".txt"):
        assert suffix not in measure.PRUNE_SUFFIXES, suffix
    for name in ("include", "__pycache__"):
        assert name in measure.PRUNE_DIRS


def test_megabytes_are_decimal_and_mib_separate() -> None:
    """1 МБ = 1 000 000 байт; МиБ считается отдельной функцией."""
    measure = _load_measure()
    assert measure.mb(1_000_000) == 1.0
    assert measure.mib(1_048_576) == 1.0
    assert measure.mb(1_500_000_000) == 1500.0


def test_runtime_probe_mentions_hf_mode() -> None:
    """Проба обязана поднимать именно режим hf, иначе объём библиотек неверен."""
    measure = _load_measure()
    assert 'mode="hf"' in measure.RUNTIME_PROBE
    assert "sys.modules" in measure.RUNTIME_PROBE


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))


def _make_cache(root: Path) -> Path:
    """Собрать кеш вида Hugging Face: refs/main, два снимка, блобы."""
    cache = root / "hf"
    repo = cache / "hub" / "models--org--model"
    blobs = repo / "blobs"
    blobs.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("rev-new", encoding="utf-8")
    for (
        name,
        weights_name,
        size,
    ) in (
        ("rev-new", "model.safetensors", 4096),
        ("rev-old", "pytorch_model.bin", 8192),
    ):
        snapshot = repo / "snapshots" / name
        snapshot.mkdir(parents=True)
        blob = blobs / f"{name}-weights"
        blob.write_bytes(b"w" * size)
        (snapshot / weights_name).symlink_to(blob)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        (snapshot / "vocab.json").write_text("{}", encoding="utf-8")
    return cache


def test_model_weights_use_main_revision_and_one_format(tmp_path: Path) -> None:
    """Веса — ревизия refs/main и один формат, а не все форматы всех ревизий."""
    measure = _load_measure()
    cache = _make_cache(tmp_path)

    weights = measure.model_weights(cache)

    # rev-old (pytorch_model.bin, 8192 Б) не выбран: снимок не той ревизии.
    # rev-new выбран в формате safetensors — его библиотека предпочитает.
    assert weights["selected_format"] == "model.safetensors (safetensors)"
    assert weights["weights_bytes"] == 4096
    assert weights["aux_bytes"] == 2 + 2, "файлы токенизатора и конфиг входят в объём"
    assert weights["bytes"] == 4100
    assert weights["by_format"] == {"model.safetensors (safetensors)": 4096}
    assert weights["revisions"] == {"models--org--model": "rev-new"}
    # Объём кеша целиком (в том числе файл другой ревизии) остаётся в артефакте
    # отдельным полем: урезание видно, а не спрятано.
    assert measure.dir_bytes_unique(cache)["unique"] > int(weights["bytes"])


def test_model_weights_fall_back_to_pytorch_without_safetensors(tmp_path: Path) -> None:
    """Если safetensors нет, весами считается pytorch_model.bin."""
    measure = _load_measure()
    cache = _make_cache(tmp_path)
    (cache / "hub" / "models--org--model" / "snapshots" / "rev-new" / "model.safetensors").unlink()
    blob = cache / "hub" / "models--org--model" / "blobs" / "rev-new-weights"
    (cache / "hub" / "models--org--model" / "snapshots" / "rev-new" / "pytorch_model.bin").symlink_to(blob)

    weights = measure.model_weights(cache)

    assert weights["selected_format"] == "pytorch_model.bin (PyTorch)"
    assert weights["weights_bytes"] == 4096


def test_model_weights_report_all_formats_of_selected_revision(tmp_path: Path) -> None:
    """Если в выбранной ревизии лежат два формата, видно оба, а считается один."""
    measure = _load_measure()
    cache = _make_cache(tmp_path)
    snapshot = cache / "hub" / "models--org--model" / "snapshots" / "rev-new"
    blob = cache / "hub" / "models--org--model" / "blobs" / "rev-new-bin"
    blob.write_bytes(b"b" * 3000)
    (snapshot / "pytorch_model.bin").symlink_to(blob)

    weights = measure.model_weights(cache)

    assert weights["selected_format"] == "model.safetensors (safetensors)"
    assert weights["by_format"] == {
        "model.safetensors (safetensors)": 4096,
        "pytorch_model.bin (PyTorch)": 3000,
    }
    assert weights["bytes"] == 4096 + 4, "в объём идёт только выбранный формат и файлы модели"


def test_hardlinks_are_counted_once(tmp_path: Path) -> None:
    """Один файл, разложенный жёсткими ссылками, не удваивает объём кеша."""
    measure = _load_measure()
    root = tmp_path / "cache"
    root.mkdir()
    blob = root / "weights.bin"
    blob.write_bytes(b"w" * 4096)
    os.link(blob, root / "snapshot.bin")
    (root / "tokenizer.json").write_bytes(b"{" + b"t" * 100 + b"}")

    sizes = measure.dir_bytes_unique(root)

    assert sizes["files"] == 3
    assert sizes["unique"] == 4096 + 102
    assert sizes["naive"] == 2 * 4096 + 102, "наивная сумма нужна как контроль дублей"


def test_model_weights_without_refs_use_all_snapshots(tmp_path: Path) -> None:
    """Без ссылки refs/main берутся все снимки: объём не занижается."""
    measure = _load_measure()
    cache = _make_cache(tmp_path)
    (cache / "hub" / "models--org--model" / "refs" / "main").unlink()

    weights = measure.model_weights(cache)

    assert weights["revision_used"].startswith("все снимки")
    # Формат всё равно один (safetensors — его предпочитает библиотека), но в разбор
    # попадают оба снимка: видно, что вторая ревизия тоже лежит на диске.
    assert weights["weights_bytes"] == 4096
    assert weights["by_format"] == {
        "model.safetensors (safetensors)": 4096,
        "pytorch_model.bin (PyTorch)": 8192,
    }
    assert weights["bytes"] == 4096 + 8, "файлы модели из двух снимков считаются один раз"


def test_model_weights_catch_model_shards(tmp_path: Path) -> None:
    """Разбитые на части веса считаются все: shard-файлы — одна модель."""
    measure = _load_measure()
    cache = _make_cache(tmp_path)
    snapshot = cache / "hub" / "models--org--model" / "snapshots" / "rev-new"
    (snapshot / "model.safetensors").unlink()
    blobs = cache / "hub" / "models--org--model" / "blobs"
    for index in (1, 2):
        blob = blobs / f"shard{index}"
        blob.write_bytes(b"s" * 2048)
        (snapshot / f"model-0000{index}-of-00002.safetensors").symlink_to(blob)

    weights = measure.model_weights(cache)

    assert weights["weights_bytes"] == 4096, "две части по 2048 байт — это модель целиком"
    assert weights["by_format"]["model.safetensors (safetensors)"] == 4096
