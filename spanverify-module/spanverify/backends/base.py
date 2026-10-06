"""Базовый интерфейс бэкенда признаков."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any


class BackendUnavailable(RuntimeError):
    """Бэкенд запрошен, но его зависимости недоступны."""


@dataclass
class BackendResult:
    """Признаки, посчитанные для последовательности словоформ.

    predictability[i] — насколько i-й токен «предсказуем» (близость к 1)
    или «неожидан» (близость к 0) в своём контексте.
    vectors[i] — контекстный вектор токена для оценки плотности окружения.
    """

    predictability: list[float]
    vectors: list[dict[int, float]] | None = None
    # informative[i] == False для токенов без стилевого сигнала (служебные слова).
    informative: list[bool] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class Backend:
    """Протокол бэкенда."""

    name: str = "base"
    description: str = ""

    def available(self) -> bool:  # pragma: no cover - тривиально
        return True

    def process(self, words: Sequence[str], text: str, dim: int = 4096) -> BackendResult:
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - отладочное
        return f"<{type(self).__name__} name={self.name!r}>"


def get_backend(name: str, **kwargs: Any) -> Backend:
    """Фабрика бэкендов. Импорт ``hf`` ленивый: torch грузится только при выборе."""
    key = (name or "surrogate").strip().lower()
    if key in {"surrogate", "demo", "stub"}:
        from .surrogate import SurrogateBackend

        return SurrogateBackend(**kwargs)
    if key in {"hf", "transformers", "model", "real"}:
        from .hf import HFBackend

        return HFBackend(**kwargs)
    raise ValueError(f"неизвестный бэкенд: {name!r} (ожидается 'surrogate' или 'hf')")


Loader = Callable[[str], Backend]
