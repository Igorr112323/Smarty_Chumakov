"""Работа с реальной моделью transformers: загрузка, окно, внимание (пункт 2.1).

Модуль закрывает четыре обязательных требования промта к режиму ``hf``:

1. **Сабтокены ↔ символы.** Быстрый токенизатор отдаёт ``offset_mapping``; по нему
   каждое слово ответа получает диапазон сабтокенов, а признаки считаются по
   среднему внутри диапазона.
2. **Несколько слоёв и голов.** Слой задаётся именем (``first``/``middle``/``last``),
   номером или отрицательным индексом; разброс энтропии по головам сохраняется в
   ``meta`` (пилоту нужно видеть, «размазано» ли внимание).
3. **Скользящее окно.** Текст длиннее ``max_tokens`` режется на перекрывающиеся
   окна; признак токена усредняется по окнам, где токен встретился. Длинный
   документ не теряет конец из-за truncation.
4. **Кэш весов и устойчивость к OOM.** Модель загружается один раз на процесс
   (ключ — имя, устройство, ``cache_dir``); при нехватке памяти окно уменьшается
   вдвое и запрос повторяется, факт уменьшения пишется в ``meta``. Ошибка
   загрузки весов не подменяется демо-режимом: поднимается ``BackendUnavailable``.

Точка входа — :func:`model_features`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .core import Token, split_chunks, tokenize_with_offsets

EPS = 1e-9

#: Кэш загруженных моделей на процесс: ``(model, device, cache_dir)`` → модель.
_MODEL_CACHE: dict[tuple[str, str, str], LoadedModel] = {}


def clear_model_cache() -> None:
    """Забыть загруженные модели (нужно тестам и долгим сервисам)."""
    _MODEL_CACHE.clear()


@dataclass
class LoadedModel:
    """Загруженная модель: токенизатор, сеть, torch и вид архитектуры."""

    name: str
    tokenizer: Any
    model: Any
    torch: Any
    device: str
    causal: bool
    layers: int
    hidden_size: int


def load_model(
    model_name: str,
    device: str | None = None,
    cache_dir: str | None = None,
    *,
    local_files_only: bool | None = None,
) -> LoadedModel:
    """Загрузить модель в кэш процесса (повторный вызов бесплатен).

    ``model_name`` — идентификатор на HuggingFace или путь к локальной папке с
    весами. Для существующей папки сеть не используется (``local_files_only``).
    Ошибка загрузки превращается в понятное сообщение; тихого перехода в
    демо-режим не существует.
    """
    from .backends.base import BackendUnavailable

    try:
        import torch  # noqa: PLC0415
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - зависит от окружения
        raise BackendUnavailable(
            "режим 'hf' требует torch и transformers: pip install -r requirements-hf.txt "
            f"(причина: {type(exc).__name__}: {exc})"
        ) from exc

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if cache_dir:
        os.environ.setdefault("HF_HOME", str(cache_dir))
    key = (model_name, resolved_device, str(cache_dir or ""))
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    is_local = Path(model_name).is_dir()
    local_only = is_local if local_files_only is None else local_files_only
    common: dict[str, Any] = {"local_files_only": local_only}
    if cache_dir:
        common["cache_dir"] = str(cache_dir)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, **common)
    except Exception as exc:
        raise BackendUnavailable(
            f"не удалось загрузить токенизатор {model_name}: {type(exc).__name__}: {exc}. "
            "Проверьте доступ к весам, путь к локальной папке или имя модели."
        ) from exc

    model = None
    causal_flag = True
    last_error: Exception | None = None
    # Порядок попыток зависит от архитектуры: у энкодера (BERT/RoBERTa/…) нет
    # языковой головы, и попытка загрузить её даёт поток предупреждений о
    # недостающих весах. Поэтому сначала читаем конфиг и выбираем класс.
    model_type = ""
    try:
        model_type = str(getattr(AutoConfig.from_pretrained(model_name, **common), "model_type", "") or "").lower()
    except Exception:  # noqa: BLE001 - конфиг может быть недоступен, тогда пробуем оба класса
        model_type = ""
    encoder_types = {
        "bert",
        "distilbert",
        "roberta",
        "camembert",
        "xlm-roberta",
        "electra",
        "albert",
        "deberta",
        "deberta-v2",
        "mpnet",
        "sentence-transformer",
        "modernbert",
    }
    attempts = (
        ((AutoModel, False), (AutoModelForCausalLM, True))
        if model_type in encoder_types
        else (
            (AutoModelForCausalLM, True),
            (AutoModel, False),
        )
    )
    for factory, causal in attempts:
        try:
            candidate = factory.from_pretrained(
                model_name,
                output_attentions=True,
                output_hidden_states=True,
                attn_implementation="eager",
                **common,
            )
        except Exception as exc:  # noqa: BLE001 - следующий вариант может сработать
            last_error = exc
            continue
        model, causal_flag = candidate, causal
        break
    if model is None:
        raise BackendUnavailable(
            f"не удалось загрузить модель {model_name}: {type(last_error).__name__}: {last_error}. "
            "Проверьте доступ к весам или укажите другую модель."
        ) from last_error

    # Служебные предупреждения transformers (например, о длине последовательности
    # при токенизации без усечения) не должны попадать в вывод CLI.
    try:
        import logging  # noqa: PLC0415

        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    except Exception:  # pragma: no cover - логирование не критично
        pass
    model.eval()
    model.config.output_attentions = True
    model.config.output_hidden_states = True
    try:
        model.to(resolved_device)
    except Exception as exc:  # pragma: no cover - зависит от железа
        raise BackendUnavailable(f"не удалось разместить модель на {resolved_device}: {exc}") from exc

    config = getattr(model, "config", None)
    layers = int(
        getattr(config, "num_hidden_layers", None)
        or getattr(config, "n_layer", None)
        or getattr(config, "num_layers", 0)
        or 0
    )
    hidden_size = int(getattr(config, "hidden_size", None) or getattr(config, "n_embd", 0) or 0)
    loaded = LoadedModel(
        name=model_name,
        tokenizer=tokenizer,
        model=model,
        torch=torch,
        device=resolved_device,
        causal=causal_flag,
        layers=layers,
        hidden_size=hidden_size,
    )
    _MODEL_CACHE[key] = loaded
    return loaded


def resolve_layer(spec: int | str, total: int) -> int:
    """Индекс слоя по имени/номеру; результат всегда в ``[0, total - 1]``."""
    if total <= 0:
        return 0
    if isinstance(spec, bool):  # bool — подкласс int, отсекаем отдельно
        return total - 1
    if isinstance(spec, int):
        index = spec if spec >= 0 else total + spec
        return max(0, min(total - 1, index))
    key = str(spec).strip().lower()
    if key in {"last", "-1", ""}:
        return total - 1
    if key in {"first", "0"}:
        return 0
    if key == "middle":
        return (total - 1) // 2
    try:
        index = int(key)
    except ValueError:
        return total - 1
    return max(0, min(total - 1, index if index >= 0 else total + index))


def window_ranges(total_tokens: int, max_tokens: int, stride: int | None = None) -> list[tuple[int, int]]:
    """Диапазоны скользящего окна по токенам: ``[(start, end), ...]``.

    Последнее окно всегда доходит до конца текста, перекрытие не меньше
    половины окна. Для текста короче окна — один диапазон.
    """
    if total_tokens <= max_tokens:
        return [(0, total_tokens)]
    step = stride or max(1, max_tokens // 2)
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < total_tokens:
        end = min(total_tokens, start + max_tokens)
        ranges.append((start, end))
        if end >= total_tokens:
            break
        start += step
    return ranges


@dataclass
class WindowResult:
    """Результат обработки одного окна."""

    offsets: list[tuple[int, int]]  # символьные смещения сабтокенов (абсолютные)
    entropy: list[float]  # нормированная энтропия внимания (среднее по головам)
    head_spread: list[float]  # разброс энтропии между головами
    mass: list[float]  # масса внимания на контекст
    hidden: list[list[float]]
    seq_len: int
    window_start: int
    window_chars: int


def run_window(
    loaded: LoadedModel,
    piece: str,
    char_start: int,
    split_char: int,
    layer: int,
    *,
    max_tokens: int,
) -> WindowResult:
    """Прогнать один кусок текста и вернуть внимание, массу и скрытые состояния.

    ``split_char`` — граница контекста и ответа в абсолютных символах: позиции
    с началом смещения меньше неё считаются контекстом (по ним считается масса).
    """
    torch = loaded.torch
    encoded = loaded.tokenizer(piece, return_tensors="pt", truncation=True, max_length=max_tokens)
    encoded = {key: value.to(loaded.device) for key, value in encoded.items()}
    with torch.no_grad():
        outputs = loaded.model(**encoded)
    attentions = outputs.attentions[layer][0]  # (heads, seq, seq)
    hidden = outputs.hidden_states[min(layer + 1, len(outputs.hidden_states) - 1)][0]
    seq_len = int(attentions.shape[-1])
    probs = attentions.clamp_min(EPS)
    entropy_per_head = -(probs * probs.log()).sum(dim=-1)  # (heads, seq)
    norm = max(EPS, math.log(max(2, seq_len)))
    entropy_norm = entropy_per_head / norm
    entropy = entropy_norm.mean(dim=0).tolist()
    head_spread = entropy_norm.std(dim=0).tolist() if entropy_norm.shape[0] > 1 else [0.0] * seq_len
    raw_offsets = loaded.tokenizer(piece, return_offsets_mapping=True, truncation=True, max_length=max_tokens)[
        "offset_mapping"
    ]
    if hasattr(raw_offsets, "tolist"):
        raw_offsets = raw_offsets.tolist()
    flat = list(raw_offsets[0]) if raw_offsets and isinstance(raw_offsets[0][0], (list, tuple)) else list(raw_offsets)
    offsets = [(int(a) + char_start, int(b) + char_start) for a, b in flat]
    context_positions = [i for i, (a, b) in enumerate(offsets) if b > a and a < split_char]
    if context_positions:
        context_index = torch.tensor(context_positions, device=attentions.device)
        mass = attentions.mean(dim=0).index_select(-1, context_index).sum(-1).tolist()
    else:
        mass = [0.0] * seq_len
    return WindowResult(
        offsets=offsets,
        entropy=[float(x) for x in entropy],
        head_spread=[float(x) for x in head_spread],
        mass=[float(x) for x in mass],
        hidden=hidden.detach().to("cpu").tolist(),
        seq_len=seq_len,
        window_start=char_start,
        window_chars=len(piece),
    )


def _oom_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "out of memory" in text or "cannot allocate" in text or "std::bad_alloc" in text


@dataclass
class ModelFeatures:
    """Признаки по словам ответа, посчитанные реальной моделью."""

    entropy: list[float] = field(default_factory=list)
    mass: list[float] = field(default_factory=list)
    density: list[float] = field(default_factory=list)
    tokens: list[Token] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


def model_features(
    answer: str,
    context: str | Sequence[str] | None,
    model_name: str,
    *,
    k: int = 5,
    device: str | None = None,
    max_tokens: int = 512,
    layer: int | str = "last",
    cache_dir: str | None = None,
    feature_cache: str | None = None,
    answer_tokens: Sequence[Token] | None = None,
) -> ModelFeatures:
    """Признаки реальной модели по словам ответа.

    Возвращает энтропию внимания (``entropy``), массу внимания на контекст
    (``mass``) и плотность представлений (``density``) — по значению на
    содержательный токен ответа. Длинные тексты идут скользящим окном.
    """
    loaded = load_model(model_name, device=device, cache_dir=cache_dir)
    tokens = list(answer_tokens) if answer_tokens is not None else tokenize_with_offsets(answer)
    if not tokens:
        return ModelFeatures(meta={"backend": "hf", "model": model_name, "empty": True})

    chunks = split_chunks(context)
    context_text = chunks.text
    prompt = f"{context_text}\n{answer}" if context_text else answer
    split_char = len(prompt) - len(answer)
    layer_index = resolve_layer(layer, max(1, loaded.layers))

    cache_path = _feature_cache_path(feature_cache, model_name, prompt, layer_index, max_tokens, k)
    if cache_path is not None and cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return ModelFeatures(
            entropy=payload["entropy"],
            mass=payload["mass"],
            density=payload["density"],
            tokens=tokens,
            meta={**payload["meta"], "cached": True},
        )

    tokenized = loaded.tokenizer(prompt, return_offsets_mapping=True, truncation=False)
    total_tokens = len(tokenized["input_ids"])
    all_offsets = [(int(a), int(b)) for a, b in tokenized["offset_mapping"]]
    current_max = max(64, int(max_tokens))
    ranges = window_ranges(total_tokens, current_max)
    per_token_entropy: dict[int, list[float]] = {index: [] for index in range(len(tokens))}
    per_token_mass: dict[int, list[float]] = {index: [] for index in range(len(tokens))}
    per_token_density: dict[int, list[float]] = {index: [] for index in range(len(tokens))}
    shrunk: list[int] = []
    window_meta: list[dict[str, Any]] = []
    context_token_total = 0

    index = 0
    while index < len(ranges):
        start, end = ranges[index]
        piece_offsets = [offset for offset in all_offsets[start:end] if offset[1] > offset[0]]
        if not piece_offsets:
            index += 1
            continue
        char_start = piece_offsets[0][0]
        char_end = piece_offsets[-1][1]
        piece = prompt[char_start:char_end]
        try:
            window = run_window(loaded, piece, char_start, split_char, layer_index, max_tokens=current_max)
        except Exception as exc:  # noqa: BLE001 - важно отличить OOM от прочих ошибок
            if _oom_error(exc) and current_max > 64:
                current_max //= 2
                shrunk.append(current_max)
                ranges = window_ranges(total_tokens, current_max)
                index = 0
                continue
            raise
        if window.seq_len <= 0:
            index += 1
            continue
        answer_positions = [i for i, (a, b) in enumerate(window.offsets) if b > a and a >= split_char]
        context_positions = [i for i, (a, b) in enumerate(window.offsets) if b > a and a < split_char]
        context_token_total += len(context_positions)
        window_meta.append(
            {
                "start_char": window.window_start,
                "chars": window.window_chars,
                "seq_len": window.seq_len,
                "answer_tokens": len(answer_positions),
                "context_tokens": len(context_positions),
                "mean_head_spread": round(statistics.fmean(window.head_spread), 4) if window.head_spread else 0.0,
            }
        )
        span_of_token: list[tuple[int, int] | None] = []
        for token in tokens:
            first = last = None
            for i, (a, b) in enumerate(window.offsets):
                if b <= a:
                    continue
                if a < token.end and b > token.start:
                    if first is None:
                        first = i
                    last = i
            span_of_token.append((first, last + 1) if first is not None else None)
        context_vectors = [window.hidden[i] for i in context_positions]
        for token_index, span in enumerate(span_of_token):
            if span is None:
                continue
            first, last = span
            last = min(last, window.seq_len)
            values = window.entropy[first:last]
            if not values:
                continue
            per_token_entropy[token_index].append(statistics.fmean(values))
            per_token_mass[token_index].append(statistics.fmean(window.mass[first:last]))
            if context_vectors:
                vector = _mean_vector(window.hidden[first:last])
                best = sorted((_cosine(vector, other) for other in context_vectors), reverse=True)[: max(1, k)]
                per_token_density[token_index].append(statistics.fmean(best))
        index += 1

    entropy_values: list[float] = []
    mass_values: list[float] = []
    density_values: list[float] = []
    for token_index in range(len(tokens)):
        entropy_values.append(
            statistics.fmean(per_token_entropy[token_index]) if per_token_entropy[token_index] else 0.5
        )
        mass_values.append(
            min(1.0, max(0.0, statistics.fmean(per_token_mass[token_index]))) if per_token_mass[token_index] else 0.0
        )
        density_values.append(
            min(1.0, max(0.0, statistics.fmean(per_token_density[token_index])))
            if per_token_density[token_index]
            else 0.0
        )

    meta = {
        "backend": "hf",
        "model": model_name,
        "causal": loaded.causal,
        "layer": layer,
        "layer_index": layer_index,
        "layers_total": loaded.layers,
        "seq_len": total_tokens,
        "context_tokens": context_token_total,
        "windows": len(window_meta),
        "window_details": window_meta[:8],
        "shrunk_max_tokens": shrunk,
        "k": k,
        "cached": False,
    }
    features = ModelFeatures(
        entropy=entropy_values,
        mass=mass_values,
        density=density_values,
        tokens=tokens,
        meta=meta,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "entropy": entropy_values,
                    "mass": mass_values,
                    "density": density_values,
                    "meta": meta,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return features


def _mean_vector(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        return []
    size = len(vectors[0])
    totals = [0.0] * size
    for vector in vectors:
        for index, value in enumerate(vector):
            totals[index] += value
    return [value / len(vectors) for value in totals]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    numerator = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return max(0.0, numerator / (norm_a * norm_b))


def _feature_cache_path(
    cache_dir: str | None,
    model_name: str,
    prompt: str,
    layer_index: int,
    max_tokens: int,
    k: int,
) -> Path | None:
    """Путь к файлу кэша признаков (``None`` — кэш выключен)."""
    if not cache_dir:
        return None
    digest = hashlib.sha256(
        json.dumps(
            {"model": model_name, "layer": layer_index, "max_tokens": max_tokens, "k": k, "text": prompt},
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()[:32]
    return Path(cache_dir) / f"features-{digest}.json"
