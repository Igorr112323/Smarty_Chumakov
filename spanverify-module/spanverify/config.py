"""Конфигурация SpanVerify.

Все параметры имеют разумные значения по умолчанию и переопределяются
(в порядке возрастания приоритета):

    файл config/*.json  ->  переменные окружения SPANVERIFY_*  ->  аргументы CLI
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

ENV_PREFIX = "SPANVERIFY_"


def packaged_text(relative_path: str) -> str | None:
    """Прочитать файл из однофайловой сборки (zipapp/.pyz).

    В собранном архиве каталоги ``config/`` и ``data/`` лежат рядом с кодом,
    но не существуют как файлы на диске, поэтому обычное чтение не работает.
    """
    archive = getattr(sys, "argv", [""])[0]
    if not str(archive).endswith(".pyz"):
        return None
    import zipfile

    try:
        with zipfile.ZipFile(archive) as zf:
            return zf.read(relative_path).decode("utf-8")
    except (KeyError, OSError, zipfile.BadZipFile):
        return None


def read_runtime_text(relative_path: str) -> str | None:
    """Прочитать файл поставки с диска (ядра запуска) или из архива zipapp."""
    relative = str(relative_path).replace("\\", "/").lstrip("./")
    for root in runtime_roots():
        candidate = root / relative
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    return packaged_text(relative)


def runtime_roots() -> list[Path]:
    """Каталоги, где искать файлы поставки (config/, data/).

    Порядок: распакованный бандл PyInstaller (``sys._MEIPASS``), папка рядом
    с исполняемым файлом, текущая папка, корень репозитория/пакета.
    Позволяет одному и тому же коду работать из исходников, из ``.exe`` в
    режиме onefile и из zipapp.
    """
    roots: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(Path(meipass))
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent)
        roots.append(Path.cwd())
    else:
        roots.append(Path.cwd())
        roots.append(Path(__file__).resolve().parent.parent)
    unique: list[Path] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return unique


def resolve_runtime_path(relative: str) -> Path:
    """Найти файл поставки по относительному пути (для *.exe и zipapp)."""
    for root in runtime_roots():
        candidate = root / relative
        if candidate.is_file():
            return candidate
    return Path(runtime_roots()[-1] / relative)


def default_config_path() -> Path | None:
    """Путь к ``config/config.json`` в поставке (если он есть)."""
    for root in runtime_roots():
        candidate = root / "config" / "config.json"
        if candidate.is_file():
            return candidate
    return None


DEFAULTS: dict[str, Any] = {
    # --- бэкенд признаков ---
    "backend": "surrogate",  # "surrogate" (демо, stdlib) | "hf" (реальная модель)
    "hf_model": "cointegrated/rubert-tiny2",
    "hf_max_tokens": 512,
    # --- признаки ---
    "k_neighbors": 5,  # k в kNN-плотности контекстных эмбеддингов
    "vector_dim": 4096,  # размерность хешированного пространства признаков
    "density_ref": 0.35,  # опорная шкала косинусной близости (-> 1.0)
    "w_predictability": 0.70,  # вес признака предсказуемости (энтропийный)
    "w_density": 0.30,  # вес признака контекстной плотности
    "smoothing_window": 7,  # сглаживание покадровой оценки, токенов
    # --- сегментация и порог ---
    "threshold": 0.5,  # порог по калиброванной вероятности
    "min_span_tokens": 6,  # минимальная длина «машинного» фрагмента
    "merge_gap_tokens": 6,  # склейка фрагментов с разрывом меньше N токенов
    # --- калибровка ---
    "calibration_path": "config/calibration.json",
    "max_fpr": 0.10,  # ограничение на долю ложных срабатываний
    "folds": 5,
    "seed": 1312,
    # --- сервер ---
    "host": "0.0.0.0",
    "port": 8000,
}

INT_FIELDS = {
    "k_neighbors",
    "vector_dim",
    "smoothing_window",
    "min_span_tokens",
    "merge_gap_tokens",
    "folds",
    "seed",
    "port",
    "hf_max_tokens",
}
FLOAT_FIELDS = {"w_predictability", "w_density", "threshold", "max_fpr", "density_ref"}


@dataclass(frozen=True)
class Config:
    """Неизменяемый набор параметров детектора."""

    backend: str = "surrogate"
    hf_model: str = "cointegrated/rubert-tiny2"
    hf_max_tokens: int = 512
    k_neighbors: int = 5
    vector_dim: int = 4096
    density_ref: float = 0.35
    w_predictability: float = 0.70
    w_density: float = 0.30
    smoothing_window: int = 7
    threshold: float = 0.5
    min_span_tokens: int = 6
    merge_gap_tokens: int = 6
    calibration_path: str = "config/calibration.json"
    max_fpr: float = 0.10
    folds: int = 5
    seed: int = 1312
    host: str = "0.0.0.0"
    port: int = 8000

    # ---------- создание ----------

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None = None) -> Config:
        """Собрать конфиг из словаря, игнорируя неизвестные ключи."""
        if not data:
            return cls()
        known = {f.name for f in fields(cls)}
        clean: dict[str, Any] = {}
        for key, value in data.items():
            if key not in known:
                continue
            if key in INT_FIELDS:
                value = int(value)
            elif key in FLOAT_FIELDS:
                value = float(value)
            clean[key] = value
        return cls(**clean)

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> Config:
        """Прочитать конфиг из JSON-файла (если он есть) + переменные окружения.

        Если путь не задан, используется ``config/config.json`` рядом с
        запуском или с исполняемым файлом.
        """
        data: dict[str, Any] = {}
        resolved = Path(path) if path is not None else default_config_path()
        if resolved is not None and resolved.is_file():
            with resolved.open("r", encoding="utf-8") as fh:
                data.update(json.load(fh))
        elif path is None:
            embedded = read_runtime_text("config/config.json")
            if embedded:
                data.update(json.loads(embedded))
        for key, _value in DEFAULTS.items():
            env = os.environ.get(ENV_PREFIX + key.upper())
            if env is not None:
                data[key] = env
        return cls.from_dict(data)

    # ---------- изменение ----------

    def with_overrides(self, **kwargs: Any) -> Config:
        """Вернуть копию конфига с переопределёнными параметрами.

        Значения ``None`` игнорируются — это удобно для аргументов CLI,
        которые по умолчанию не заданы.
        """
        known = {f.name for f in fields(self)}
        clean: dict[str, Any] = {}
        for key, value in kwargs.items():
            if value is None or key not in known:
                continue
            if key in INT_FIELDS:
                value = int(value)
            elif key in FLOAT_FIELDS:
                value = float(value)
            clean[key] = value
        return replace(self, **clean) if clean else self

    # ---------- вывод ----------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | os.PathLike[str]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
        return p

    def describe(self) -> str:
        return (
            f"backend={self.backend} k={self.k_neighbors} "
            f"w=({self.w_predictability}/{self.w_density}) threshold={self.threshold:.4f}"
        )
