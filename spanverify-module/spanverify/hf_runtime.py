"""Режим ``hf``: признаки из реальной языковой модели (пункт 2.1 реестра).

Модуль намеренно разделён на две части.

**Чистая часть** (без ``torch``) — вся арифметика признаков:
:func:`entropy_of_row`, :func:`context_mass_of_row`, :func:`normalize_mass`,
:func:`aggregate_layers`, :func:`support_distance`. Она покрыта обычными
тестами и считается на настоящих числах, а не на заглушках.

**Тонкая часть** (:class:`ModelRunner`) — только загрузка весов и прямой проход
модели. Всё, что раньше было дефектами пилота, вынесено сюда и исправлено:

* кэш весов — модель грузится один раз на процесс (раньше каждый вызов
  ``hf_features`` читал веса заново, из-за чего прогон 48 пар шёл часами);
* длинный текст — скользящее окно с перекрытием вместо обрезки по 512 токенам
  (:func:`spanverify.alignment.plan_windows`);
* нехватка памяти — пакетная обработка окон с автоматическим уменьшением
  пакета при ``MemoryError`` / ``OutOfMemoryError`` и выгрузкой тензоров;
* несколько слоёв и голов внимания вместо одного последнего слоя;
* сопоставление сабтокенов с символами ответа через
  :func:`spanverify.alignment.align_subwords`.

Подмены режима нет: если веса недоступны, поднимается
:class:`~spanverify.backends.base.BackendUnavailable` с фактической причиной —
режим ``demo`` вместо ``hf`` молча не подставляется никогда.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .alignment import align_subwords, batch_sizes, merge_window_values, plan_windows
from .backends.base import BackendUnavailable

__all__ = [
    "ModelRunner",
    "aggregate_layers",
    "context_mass_of_row",
    "entropy_of_row",
    "normalize_mass",
    "resolve_heads",
    "support_distance",
    "clear_model_cache",
    "model_cache_info",
]

EPS = 1e-12


# --------------------------------------------------------------------------
# Чистая арифметика признаков (тестируется без torch)
# --------------------------------------------------------------------------


def entropy_of_row(row: Sequence[float], *, normalize: bool = True) -> float:
    """Энтропия Шеннона одной строки внимания, нормированная на ``log(n)``.

    ``row`` — распределение внимания одного токена-запроса по ключам.
    Нормировка на ``log(n)`` приводит значение к ``[0, 1]`` и делает окна
    разной длины сравнимыми (без неё длинное окно всегда «неувереннее»).
    """
    values = [float(v) for v in row if v > 0.0]
    if not values:
        return 0.0
    total = sum(values)
    if total <= EPS:
        return 0.0
    entropy = 0.0
    for value in values:
        p = value / total
        entropy -= p * math.log(p)
    if not normalize:
        return entropy
    denominator = math.log(max(2, len(row)))
    return max(0.0, min(1.0, entropy / denominator))


def context_mass_of_row(row: Sequence[float], context_positions: Sequence[int]) -> float:
    """Доля внимания токена, ушедшая на токены документа-контекста."""
    if not context_positions:
        return 0.0
    total = sum(float(v) for v in row)
    if total <= EPS:
        return 0.0
    mass = 0.0
    for position in context_positions:
        if 0 <= position < len(row):
            mass += float(row[position])
    return max(0.0, min(1.0, mass / total))


def normalize_mass(mass: float, n_context: int, n_total: int, n_answer: int = 0) -> float:
    """Нормировка массы внимания (итерация 2.2 «а»).

    Сырая масса растёт просто оттого, что контекст длинный: если документ
    занимает 90 % последовательности, то и внимания на него придётся около
    90 % даже у выдуманного токена. Поэтому масса делится на ожидаемую при
    равномерном внимании долю ``n_context / n_total`` — получается «во сколько
    раз токен смотрит на документ чаще, чем случайно», и приводится к ``[0, 1]``
    логистическим сжатием. Длина ответа входит слабой поправкой: у длинных
    ответов больше внутренних связей, доля на контекст механически падает.
    """
    if n_total <= 0 or n_context <= 0:
        return 0.0
    expected = n_context / float(n_total)
    if expected <= EPS:
        return 0.0
    ratio = mass / expected
    if n_answer > 0:
        ratio *= 1.0 + math.log1p(n_answer) / 20.0
    # Логистика с центром в 1.0: «как случайно» → 0.5, вдвое чаще → ~0.73.
    return 1.0 / (1.0 + math.exp(-(ratio - 1.0)))


def resolve_heads(spec: str | Sequence[int] | None, total: int) -> list[int]:
    """Список индексов голов внимания по описанию.

    Поддерживаются ``"all"`` (все головы), ``"first"``, ``"last"``,
    ``"half"`` (первая половина) и явный список индексов. Нужен для итерации
    2.2 «в»: сравнение нескольких голов, а не усреднение по всем вслепую.
    """
    if total <= 0:
        return []
    if spec is None or (isinstance(spec, str) and spec.strip().lower() in {"", "all"}):
        return list(range(total))
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key == "first":
            return [0]
        if key == "last":
            return [total - 1]
        if key == "half":
            return list(range(max(1, total // 2)))
        parts = [p for p in key.replace(";", ",").split(",") if p.strip()]
        out: list[int] = []
        for part in parts:
            try:
                index = int(part)
            except ValueError:
                continue
            index = index if index >= 0 else total + index
            if 0 <= index < total:
                out.append(index)
        return out or list(range(total))
    out = []
    for value in spec:
        index = int(value)
        index = index if index >= 0 else total + index
        if 0 <= index < total:
            out.append(index)
    return out or list(range(total))


def aggregate_layers(values: Sequence[Sequence[float]], how: str = "mean") -> list[float]:
    """Свести значения нескольких слоёв к одному ряду.

    ``how``: ``"mean"`` (среднее по слоям), ``"max"``, ``"last"``.
    Нужен для итерации 2.2 «в»: сравнение слоёв и их комбинации.
    """
    rows = [list(row) for row in values if row is not None]
    if not rows:
        return []
    length = min(len(row) for row in rows)
    if how == "last":
        return rows[-1][:length]
    out: list[float] = []
    for index in range(length):
        column = [row[index] for row in rows]
        out.append(max(column) if how == "max" else sum(column) / len(column))
    return out


def support_distance(
    answer_positions: Sequence[int],
    supported: Sequence[bool],
) -> list[float]:
    """Расстояние до ближайшего подтверждённого токена (итерация 2.2 «б»).

    Для каждого токена ответа считается, сколько токенов отделяет его от
    ближайшего токена, нашедшего опору в документе, и результат приводится к
    ``[0, 1]``: 0 — сам подтверждён, 1 — опоры нет вовсе. Признак ловит
    типичную картину выдумки: выдуманный фрагмент стоит «в стороне» от всего
    подтверждённого.
    """
    n = len(answer_positions)
    if n == 0:
        return []
    anchors = [index for index in range(n) if index < len(supported) and supported[index]]
    if not anchors:
        return [1.0] * n
    out: list[float] = []
    for index in range(n):
        distance = min(abs(index - anchor) for anchor in anchors)
        out.append(min(1.0, distance / max(1.0, n / 2.0)))
    return out


# --------------------------------------------------------------------------
# Загрузка модели и прямой проход
# --------------------------------------------------------------------------

_CACHE: dict[tuple[str, str], Any] = {}


def clear_model_cache() -> None:
    """Выгрузить закэшированные веса (используется тестами и CLI)."""
    _CACHE.clear()


def model_cache_info() -> dict[str, Any]:
    """Что лежит в кэше весов: ключи и их количество."""
    return {"size": len(_CACHE), "keys": [f"{name}@{device}" for name, device in _CACHE]}


def _dependency_error(exc: BaseException, model_name: str) -> BackendUnavailable:
    """Понятное сообщение вместо трассировки (требование к продукту)."""
    hint = (
        "Режим 'hf' требует пакетов torch и transformers и локально доступных весов.\n"
        "  1) pip install -r requirements-hf.txt\n"
        f"  2) заранее скачайте веса: huggingface-cli download {model_name}\n"
        "  3) офлайн-запуск: HF_HUB_OFFLINE=1 и переменная HF_HOME с путём к весам.\n"
        "Подмена режимом 'demo' не выполняется: demo — лексический суррогат, "
        "его числа нельзя выдавать за результат реальной модели."
    )
    return BackendUnavailable(f"{type(exc).__name__}: {exc}\n{hint}")


@dataclass
class LoadedModel:
    """Загруженные веса и всё, что нужно для прямого прохода."""

    tokenizer: Any
    model: Any
    torch: Any
    device: str
    n_layers: int
    n_heads: int
    max_positions: int
    model_name: str


class ModelRunner:
    """Прямой проход модели с окнами, пакетами и кэшем весов."""

    def __init__(
        self,
        model_name: str = "ai-forever/rugpt3small_based_on_gpt2",
        device: str | None = None,
        window: int | None = None,
        overlap: int = 64,
        batch: int = 4,
        k: int = 5,
    ) -> None:
        self.model_name = model_name
        self.device_request = device
        self.window = window
        self.overlap = overlap
        self.batch = max(1, int(batch))
        self.k = max(1, int(k))

    # ---------------------------------------------------------------- веса

    def load(self) -> LoadedModel:
        """Загрузить (или достать из кэша) токенизатор и модель."""
        try:
            import torch  # noqa: PLC0415
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover - зависит от окружения
            raise _dependency_error(exc, self.model_name) from exc

        device = self.device_request or ("cuda" if torch.cuda.is_available() else "cpu")
        key = (self.model_name, device)
        cached = _CACHE.get(key)
        if cached is not None:
            return cached

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                attn_implementation="eager",  # иначе attentions не возвращаются
            )
        except Exception as exc:  # pragma: no cover - зависит от окружения
            raise _dependency_error(exc, self.model_name) from exc

        model.eval()
        model.config.output_attentions = True
        model.config.output_hidden_states = True
        model.to(device)
        config = model.config
        loaded = LoadedModel(
            tokenizer=tokenizer,
            model=model,
            torch=torch,
            device=device,
            n_layers=int(getattr(config, "num_hidden_layers", 0) or getattr(config, "n_layer", 0) or 0),
            n_heads=int(getattr(config, "num_attention_heads", 0) or getattr(config, "n_head", 0) or 0),
            max_positions=int(
                getattr(config, "max_position_embeddings", 0) or getattr(config, "n_positions", 0) or 1024
            ),
            model_name=self.model_name,
        )
        _CACHE[key] = loaded
        return loaded

    # ---------------------------------------------------------------- проход

    def encode(self, loaded: LoadedModel, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """Текст → идентификаторы сабтокенов и их символьные смещения."""
        encoded = loaded.tokenizer(text, return_offsets_mapping=True, truncation=False)
        offsets = encoded["offset_mapping"]
        if hasattr(offsets, "tolist"):
            offsets = offsets.tolist()
        if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], list):
            offsets = offsets[0]
        return list(encoded["input_ids"]), [(int(s), int(e)) for s, e in offsets]

    def run(
        self,
        loaded: LoadedModel,
        input_ids: Sequence[int],
        layers: Sequence[int],
        heads: Sequence[int],
        answer_start_token: int,
    ) -> dict[str, list[float]]:
        """Признаки по сабтокенам для всей последовательности.

        Возвращает ряды длиной ``len(input_ids)``: ``entropy`` (нормированная
        энтропия внимания), ``mass`` (доля внимания на контекст) и
        ``mass_norm`` (та же масса после нормировки на длину контекста).
        Длинные последовательности обрабатываются окнами с перекрытием,
        окна — пакетами; при нехватке памяти пакет уменьшается.
        """
        torch = loaded.torch
        total = len(input_ids)
        window_size = self.window or min(loaded.max_positions, 512)
        window_size = max(16, min(window_size, loaded.max_positions))
        windows = plan_windows(total, window_size, min(self.overlap, window_size // 2))

        entropy_rows: list[list[float]] = [[] for _ in windows]
        mass_rows: list[list[float]] = [[] for _ in windows]
        mass_norm_rows: list[list[float]] = [[] for _ in windows]
        density_rows: list[list[float]] = [[] for _ in windows]

        batch = self.batch
        index = 0
        while index < len(windows):
            chunk = windows[index : index + batch]
            try:
                results = self._forward(loaded, input_ids, chunk, layers, heads, answer_start_token, k=self.k)
            except (MemoryError, RuntimeError) as exc:  # pragma: no cover - зависит от железа
                message = str(exc).lower()
                if batch > 1 and ("memory" in message or "alloc" in message or isinstance(exc, MemoryError)):
                    batch = max(1, batch // 2)
                    if hasattr(torch, "cuda") and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise
            for offset, (ent, mass, mass_norm, density) in enumerate(results):
                entropy_rows[index + offset] = ent
                mass_rows[index + offset] = mass
                mass_norm_rows[index + offset] = mass_norm
                density_rows[index + offset] = density
            index += len(chunk)

        return {
            "entropy": merge_window_values(total, windows, entropy_rows, default=0.5),
            "mass": merge_window_values(total, windows, mass_rows, default=0.0),
            "mass_norm": merge_window_values(total, windows, mass_norm_rows, default=0.0),
            "density": merge_window_values(total, windows, density_rows, default=0.0),
            "windows": [float(len(windows))],
        }

    def _forward(
        self,
        loaded: LoadedModel,
        input_ids: Sequence[int],
        windows: Sequence[Any],
        layers: Sequence[int],
        heads: Sequence[int],
        answer_start_token: int,
        k: int = 5,
    ) -> list[tuple[list[float], list[float], list[float], list[float]]]:
        """Прямой проход для пакета окон. Тензоры освобождаются сразу."""
        torch = loaded.torch
        pad_id = getattr(loaded.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(loaded.tokenizer, "eos_token_id", None) or 0
        width = max(w.length for w in windows)
        batch_ids = []
        batch_mask = []
        for window in windows:
            piece = list(input_ids[window.start : window.end])
            pad = width - len(piece)
            batch_ids.append(piece + [pad_id] * pad)
            batch_mask.append([1] * len(piece) + [0] * pad)

        tensor_ids = torch.tensor(batch_ids, dtype=torch.long, device=loaded.device)
        tensor_mask = torch.tensor(batch_mask, dtype=torch.long, device=loaded.device)
        out: list[tuple[list[float], list[float], list[float], list[float]]] = []
        with torch.no_grad():
            outputs = loaded.model(
                input_ids=tensor_ids,
                attention_mask=tensor_mask,
                output_attentions=True,
                output_hidden_states=True,
            )
            attentions = outputs.attentions  # кортеж слоёв: (batch, heads, seq, seq)
            hidden_states = getattr(outputs, "hidden_states", None)
            n_layers = len(attentions)
            chosen_layers = [index for index in layers if 0 <= index < n_layers] or [n_layers - 1]
            for position, window in enumerate(windows):
                length = window.length
                context_positions = [i for i in range(length) if window.start + i < answer_start_token]
                answer_count = length - len(context_positions)
                per_layer_entropy: list[list[float]] = []
                per_layer_mass: list[list[float]] = []
                for layer_index in chosen_layers:
                    layer = attentions[layer_index][position]  # (heads, seq, seq)
                    chosen_heads = [h for h in heads if 0 <= h < layer.shape[0]] or list(range(layer.shape[0]))
                    selected = layer[chosen_heads].mean(dim=0)  # (seq, seq)
                    rows = selected[:length, :length].detach().to("cpu").tolist()
                    per_layer_entropy.append([entropy_of_row(row) for row in rows])
                    per_layer_mass.append([context_mass_of_row(row, context_positions) for row in rows])
                    del selected
                entropy = aggregate_layers(per_layer_entropy, "mean")
                mass = aggregate_layers(per_layer_mass, "mean")
                mass_norm = [normalize_mass(value, len(context_positions), length, answer_count) for value in mass]
                density = self._density(torch, hidden_states, position, length, context_positions, chosen_layers[-1], k)
                out.append((entropy, mass, mass_norm, density))
            del outputs, attentions
        del tensor_ids, tensor_mask
        if hasattr(torch, "cuda") and torch.cuda.is_available():  # pragma: no cover - зависит от железа
            torch.cuda.empty_cache()
        return out

    @staticmethod
    def _density(
        torch: Any,
        hidden_states: Any,
        position: int,
        length: int,
        context_positions: Sequence[int],
        layer_index: int,
        k: int,
    ) -> list[float]:
        """kNN-плотность токена среди контекстных представлений документа.

        Считается в тензорах (косинус + top-k), а не в питоне: на окне 512
        токенов питоновский перебор занимал бы секунды на каждую пару.
        """
        if hidden_states is None or not context_positions:
            return [0.0] * length
        layer = hidden_states[min(layer_index + 1, len(hidden_states) - 1)][position][:length]
        normalized = layer / layer.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        context = normalized[list(context_positions)]
        similarity = normalized @ context.transpose(0, 1)
        top = min(max(1, k), similarity.shape[-1])
        values = similarity.topk(top, dim=-1).values.mean(dim=-1)
        return [max(0.0, min(1.0, float(v))) for v in values.detach().to("cpu").tolist()]


# --------------------------------------------------------------------------
# Сборка признаков по словам ответа
# --------------------------------------------------------------------------


def words_from_subwords(
    text: str,
    word_spans: Sequence[tuple[int, int]],
    offsets: Sequence[tuple[int, int]],
    values: Sequence[float],
    default: float = 0.0,
) -> list[float]:
    """Свести значения сабтокенов к значениям слов ответа (итерация 2.2 «в»).

    Агрегация — среднее по сабтокенам слова. Слово без сабтокенов (обрезано
    окном или это пробел) получает ``default``.
    """
    aligned = align_subwords(text, word_spans, offsets)
    out: list[float] = []
    for indices in aligned:
        picked = [float(values[i]) for i in indices if 0 <= i < len(values)]
        out.append(sum(picked) / len(picked) if picked else default)
    return out


def hf_offline_hint() -> str:
    """Подсказка о текущем офлайн-режиме HuggingFace (для отчётов)."""
    flags = {
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", ""),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", ""),
        "HF_HOME": os.environ.get("HF_HOME", ""),
    }
    return "; ".join(f"{key}={value or '-'}" for key, value in flags.items())


def batch_plan(n_windows: int, batch: int) -> list[tuple[int, int]]:
    """Публичная обёртка над :func:`spanverify.alignment.batch_sizes`."""
    return batch_sizes(n_windows, batch)
