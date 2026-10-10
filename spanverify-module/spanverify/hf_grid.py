"""Расширенная сетка признаков режима hf: один проход модели, много вариаций.

Зачем отдельный модуль, а не разрастание ``hf_features``: протокол измерения
(``docs/METRIC_SPEC.md``) требует перебирать признаки — слои, агрегации по
головам, ``k`` плотности, лексическое и числовое совпадение с документом,
привязку числа, признаки соседства, — и при этом не трогать product-путь, по
которому живут публичный контракт и 550 тестов. Здесь только то, что нужно
измерению: один прямой проход по модели и словарь «имя признака → значения по
токенам ответа».

Две вещи сделаны иначе, чем в ``hf_features``, и это не косметика:

* ``GridModel`` грузит модель и токенизатор один раз на прогон, а не на пару
  (на 1023 парах это ~25 минут праздного ожидания);
* редукции по слоям и похожести считаются тензорными операциями и слоями по
  очереди, а не вложенными циклами по ``.tolist()``; карты внимания целиком не
  дублируются (стек из 12 слоёв на длине 1024 — это ~600 МП, что на 16 ГБ
  раннера уже риск).

Признаки считаются только по документу и ответу: gold-разметка в них не
попадает, модуль про разметку ничего не знает.

Формат строки кеша — v3: наряду с массивами в неё кладутся токены ответа с
смещениями и флаг ``scored``, чтобы потребитель мог сопоставить признаки с
разметкой, не пересчитывая токенизацию, и проверить выравнивание отдельно.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .core import Token, numbers_in, split_chunks, tokenize_with_offsets
from .features import (
    context_mass_lift,
    distance_decay,
    expected_context_mass,
    is_scored_token,
    map_token_positions,
    normalised_context_mass,
    normalised_support_distance,
    number_attribution,
    resolve_layer,
    similarity_contrast,
    similarity_margin,
)

__all__ = [
    "GRID_CACHE_FORMAT",
    "GRID_FEATURE_NAMES",
    "GridModel",
    "compute_grid",
    "grid_model_info",
    "grid_model_revision",
    "load_grid_model",
]

#: Формат кеша сетки (v3 — «широкие» признаки, посчитанные одним проходом).
GRID_CACHE_FORMAT = 3

#: Имена признаков. Порядок фиксирован: он попадает в манифест, и по нему
#: сверяется содержимое кеша — расхождение «голова просит признак, которого в
#: кеше нет» должно быть видно до обучения, а не после испорченных метрик.
GRID_FEATURE_NAMES: tuple[str, ...] = (
    # Энтропия внимания: слои и агрегации по головам.
    "entropy_last",
    "entropy_first",
    "entropy_middle",
    "entropy_m4",
    "entropy_mean_layers",
    "entropy_max_layers",
    "entropy_head_max",
    "entropy_head_disagreement",
    # Масса внимания на документ: слои, агрегации, нормировка и «подъём».
    "mass_last",
    "mass_first",
    "mass_middle",
    "mass_mean_layers",
    "mass_max_layers",
    "mass_max_head",
    "mass_norm_last",
    "mass_lift_last",
    # Плотность представлений при разных k и слоях + контекстные похожести.
    "density_k1_last",
    "density_k3_last",
    "density_k5_last",
    "density_k3_middle",
    "sim_max_last",
    "sim_contrast_last",
    "sim_margin_last",
    "sim_distance_last",
    "sim_decay_last",
    # Лексика и числа: совпадение токена с документом (признак, не правило).
    "lex_match",
    "num_match",
    "trigram_match",
    # Правило привязки числа к объекту — как признак.
    "attrib_borrowed",
    "attrib_supported",
    # Соседство и позиция.
    "mass_win3",
    "entropy_win3",
    "position",
    "length",
    "sentence_position",
    "sentence_support",
)

#: Редукции по окну ( lever «агрегация» ): сглаживание соседами ±1 токен.
WINDOW_RADIUS = 1

DEFAULT_ENTROPY = 0.5  # «неизвестно» для токена без позиции в модели


@dataclass
class GridModel:
    """Модель и токенизатор, загруженные один раз на весь прогон."""

    model_name: str
    model: Any
    tokenizer: Any
    device: str = "cpu"
    revision: str = ""

    @property
    def layers(self) -> int:
        return int(getattr(self.model.config, "n_layer", 0) or 0)


def grid_model_revision(model_name: str) -> str:  # pragma: no cover - требует доступа к huggingface_hub
    """Revision весов модели: числа протокола обязаны быть привязаны к ним."""
    try:  # pragma: no cover - зависит от сети
        from huggingface_hub import model_info  # noqa: PLC0415

        return str(model_info(model_name).sha or "")
    except Exception:  # noqa: BLE001 - отсутствие сети не валит прогон, только запись
        return ""


def load_grid_model(  # pragma: no cover - требует torch и весов
    model_name: str, device: str | None = None
) -> GridModel:
    """Загрузить модель для сетки (ленивый импорт torch — как в ``hf_features``)."""
    from .backends.base import BackendUnavailable  # noqa: PLC0415
    from .backends.hf import HFBackend  # noqa: PLC0415

    ok, reason = HFBackend.dependencies()
    if not ok:
        raise BackendUnavailable(reason)
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, attn_implementation="eager", output_attentions=True, output_hidden_states=True
    )
    model.eval()
    model.config.output_attentions = True
    model.config.output_hidden_states = True
    chosen = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(chosen)
    return GridModel(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        device=chosen,
        revision=grid_model_revision(model_name),
    )


def grid_model_info(loaded: GridModel) -> dict[str, Any]:
    """Сведения о модели для манифеста: id, revision, размерность, устройство."""
    config = loaded.model.config
    return {
        "model": loaded.model_name,
        "revision": loaded.revision,
        "layers": int(getattr(config, "n_layer", 0) or 0),
        "heads": int(getattr(config, "n_head", 0) or 0),
        "hidden": int(getattr(config, "n_embd", 0) or 0),
        "tokenizer_size": len(loaded.tokenizer),
        "device": loaded.device,
    }


def _offsets_of_model_tokens(  # pragma: no cover - требует hf-токенизатора
    tokenizer: Any, prompt: str, max_length: int
) -> tuple[list[tuple[int, int]], str]:
    """Смещения подслов модели в тексте промпта и способ их получения.

    Быстрый токенмайкер GPT-2 не отдаёт ``offset_mapping`` (или отдаёт нули) —
    тогда смещения восстанавливаются жадным совпадением подслова с текстом.
    Способ пишется в метаданные строки кеша: выравнивание признаков — то, что
    обязано быть проверяемым, иначе сдвиг на слово утонет в метриках.
    """
    try:
        raw = tokenizer(prompt, return_offsets_mapping=True, truncation=True, max_length=max_length)
        offsets = raw["offset_mapping"]
    except Exception:  # noqa: BLE001 - не все токенизаторы это умеют
        offsets = None
    if offsets is not None:
        if hasattr(offsets, "tolist"):
            offsets = offsets.tolist()
        if offsets and isinstance(offsets[0][0], (list, tuple)):
            offsets = offsets[0]
        pairs = [(int(start), int(end)) for start, end in offsets]
        if any(end > start for start, end in pairs):
            return pairs, "tokenizer"
    return _offsets_by_alignment(tokenizer, prompt, max_length), "alignment"


def _offsets_by_alignment(
    tokenizer: Any, prompt: str, max_length: int
) -> list[tuple[int, int]]:  # pragma: no cover - требует hf-токенизатора
    """Восстановить смещения подслов жадным совпадением (для GPT-2 без карты)."""
    encoded = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=max_length)
    ids = list(encoded["input_ids"])
    if ids and isinstance(ids[0], list):
        ids = list(ids[0])
    pieces = [str(tokenizer.decode([int(token)])).replace("Ġ", " ").replace("▁", " ") for token in ids]
    pairs: list[tuple[int, int]] = []
    cursor = 0
    for piece in pieces:
        word = piece.strip()
        if not word:
            pairs.append((cursor, cursor))
            continue
        found = prompt.find(word, cursor)
        if found < 0:
            found = prompt.find(word)
        if found < 0:
            pairs.append((cursor, cursor))
            continue
        pairs.append((found, found + len(word)))
        cursor = found + len(word)
    return pairs


def _window(values: Sequence[float], radius: int) -> list[float]:
    """Сглаживание скользящим окном радиуса ``radius`` (края — усечённое окно)."""
    total = len(values)
    if total == 0 or radius <= 0:
        return [float(value) for value in values]
    out: list[float] = []
    for index in range(total):
        left = max(0, index - radius)
        right = min(total, index + radius + 1)
        out.append(statistics.fmean(values[left:right]))
    return out


def _covers(spans: Sequence[tuple[int, int]], token: Token) -> bool:
    return any(token.start < end and token.end > start for start, end in spans)


def _mean(values: Sequence[float], fallback: float = 0.0) -> float:
    return statistics.fmean(values) if values else fallback


def _document_lexicon(text: str) -> dict[str, Any]:
    """Словарь документа: стеммы слов, числа, стеммы по предложениям."""
    from .normalize import stem_text  # noqa: PLC0415

    stems: set[str] = set()
    for stem in stem_text(text):
        if len(stem) >= 4:
            stems.add(stem[:5])
    sentence_stems = [
        {stem[:5] for stem in stem_text(sentence) if len(stem) >= 4}
        for sentence in (chunk for chunk in text.split(".") if chunk.strip())
    ]
    return {
        "stems": stems,
        "numbers": set(numbers_in(text)),
        "sentence_stems": [item for item in sentence_stems if item],
    }


def _trigram_match(word: str, stems: set[str]) -> float:
    """Максимум перекрытия триграмм токена с триграммами слов документа.

    Непрерывная версия лексического совпадения: «документооборот» разделяет с
    «документов» часть триграмм, и двоичный признак этого не видит.
    """
    text = word.lower().strip("«».,;:!?()")
    if len(text) < 3 or not stems:
        return 0.0
    left = {text[index : index + 3] for index in range(len(text) - 2)}
    best = 0.0
    for candidate in stems:
        right = {candidate[index : index + 3] for index in range(len(candidate) - 2)}
        if not right:
            continue
        value = len(left & right) / len(right)
        if value > best:
            best = value
    return best


def compute_grid(  # pragma: no cover - требует torch и весов: unit-тесты покрывают pure-python часть модуля
    answer: str,
    context: str | Sequence[str] | None,
    loaded: GridModel,
    *,
    max_length: int = 1024,
    k_values: tuple[int, ...] = (1, 3, 5),
    window: int = WINDOW_RADIUS,
) -> dict[str, Any]:
    """Сетка признаков пары «ответ — документ» за один проход модели.

    Длина каждого массива равна числу токенов нашего разбиения ответа — ровно
    как в ``FeatureMatrix``: фильтровать содержательные токены обязан
    потребитель (``is_scored_token``), иначе выравнивание с разметкой разойдётся.
    Токен, для которого модель не дала позиции (обрезка ``max_length``),
    получает нейтральные значения и учитывается в ``meta['unmatched_tokens']``:
    обрезка обязана быть видна в манифесте, а не спрятана в признаках.
    """
    import torch  # noqa: PLC0415

    tokens: list[Token] = list(tokenize_with_offsets(answer))
    chunks = split_chunks(context)
    context_text = chunks.text
    prompt = f"{context_text}\n{answer}" if context_text else answer
    answer_start = len(prompt) - len(answer)

    encoded = loaded.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    encoded = {key: value.to(loaded.device) for key, value in encoded.items() if hasattr(value, "to")}
    seq_len = int(encoded["attention_mask"].sum().item())
    with torch.no_grad():
        outputs = loaded.model(**encoded)

    offsets, offsets_mode = _offsets_of_model_tokens(loaded.tokenizer, prompt, max_length)
    model_spans = [
        (position, start, end)
        for position, (start, end) in enumerate(offsets)
        if position < seq_len and end > start and start >= answer_start
    ]
    answer_positions = [position for position, _start, _end in model_spans]
    first_answer = min(answer_positions) if answer_positions else seq_len
    context_positions = list(range(first_answer))[:seq_len]
    token_positions = map_token_positions([(token.start, token.end) for token in tokens], model_spans, answer_start)

    attentions = tuple(outputs.attentions)  # (layers,) × (1, heads, seq, seq)
    layers = len(attentions)
    layer_roles = {role: resolve_layer(role, layers) for role in ("first", "middle", "last", "m4")}
    log_scale = max(1e-9, math.log(max(2, seq_len)))

    # Поверхности по слоям: (S,) — среднее и максимум по головам, максимум
    # «растерянности» отдельной головы и несогласие голов между собой.
    entropy_mean: list[list[float]] = []
    entropy_head_max: list[list[float]] = []
    mass_mean: list[list[float]] = []
    mass_head_max: list[list[float]] = []
    disagreement: list[float] | None = None
    for layer_attentions in attentions:
        probabilities = layer_attentions.clamp_min(1e-9)
        entropy = -(probabilities * probabilities.log()).sum(dim=-1) / log_scale  # (heads, seq)
        if context_positions:
            mass = layer_attentions[..., : len(context_positions)].sum(dim=-1)  # (heads, seq)
        else:  # pragma: no cover - пустой контекст
            mass = torch.zeros(layer_attentions.shape[:2], device=layer_attentions.device)
        entropy_mean.append(entropy.mean(dim=0).tolist())
        entropy_head_max.append(entropy.max(dim=0).values.tolist())
        mass_mean.append(mass.mean(dim=0).tolist())
        mass_head_max.append(mass.max(dim=0).values.tolist())
        if disagreement is None:
            spread = entropy.std(dim=0)  # разброс энтропии по головам: несогласие голов
            disagreement = (spread / (entropy.mean(dim=0) + 1e-9)).clamp(0.0, 1.0).tolist()
    if disagreement is None:  # pragma: no cover - модель без attentions
        disagreement = [0.0] * seq_len

    hidden_states = tuple(outputs.hidden_states or ())
    similarity = _similarity_rows(
        hidden_states=hidden_states,
        layer_roles=layer_roles,
        answer_positions=answer_positions,
        context_positions=context_positions,
        seq_len=seq_len,
        k_values=k_values,
    )

    lexicon = _document_lexicon(context_text)
    document_numbers = lexicon["numbers"]
    attributions = number_attribution(answer, context)
    borrowed = [(int(item["start"]), int(item["end"])) for item in attributions if item.get("borrowed")]
    supported = [(int(item["start"]), int(item["end"])) for item in attributions if not item.get("borrowed")]
    from .core import split_sentences  # noqa: PLC0415

    answer_sentences = list(split_sentences(answer))
    sentence_support = _sentence_support(answer, answer_sentences, lexicon)

    arrays: dict[str, list[float]] = {name: [] for name in GRID_FEATURE_NAMES}
    unmatched = 0
    total_tokens = max(1, len(tokens))
    for index, (token, positions) in enumerate(zip(tokens, token_positions, strict=False)):
        if not positions:
            unmatched += 1
            for name in GRID_FEATURE_NAMES:
                arrays[name].append(DEFAULT_ENTROPY if name.startswith("entropy") else 0.0)
            continue
        last = layer_roles["last"]
        for role, name in (
            ("last", "entropy_last"),
            ("first", "entropy_first"),
            ("middle", "entropy_middle"),
            ("m4", "entropy_m4"),
        ):
            table = entropy_mean[layer_roles[role]]
            arrays[name].append(_mean([table[position] for position in positions], DEFAULT_ENTROPY))
        arrays["entropy_mean_layers"].append(
            _mean(
                [statistics.fmean(entropy_mean[layer][position] for layer in range(layers)) for position in positions],
                DEFAULT_ENTROPY,
            )
        )
        arrays["entropy_max_layers"].append(
            _mean(
                [max(entropy_mean[layer][position] for layer in range(layers)) for position in positions],
                DEFAULT_ENTROPY,
            )
        )
        arrays["entropy_head_max"].append(
            _mean(
                [
                    statistics.fmean(entropy_head_max[layer][position] for layer in range(layers))
                    for position in positions
                ],
                DEFAULT_ENTROPY,
            )
        )
        arrays["entropy_head_disagreement"].append(_mean([disagreement[position] for position in positions]))

        for role, name in (("last", "mass_last"), ("first", "mass_first"), ("middle", "mass_middle")):
            table = mass_mean[layer_roles[role]]
            arrays[name].append(_mean([table[position] for position in positions]))
        arrays["mass_mean_layers"].append(
            _mean([statistics.fmean(mass_mean[layer][position] for layer in range(layers)) for position in positions])
        )
        arrays["mass_max_layers"].append(
            _mean([max(mass_mean[layer][position] for layer in range(layers)) for position in positions])
        )
        arrays["mass_max_head"].append(
            _mean(
                [statistics.fmean(mass_head_max[layer][position] for layer in range(layers)) for position in positions]
            )
        )
        observed = float(mass_mean[last][max(positions)]) if context_positions else 0.0
        expected = expected_context_mass(len(context_positions), min(positions))
        arrays["mass_norm_last"].append(normalised_context_mass(observed, expected))
        arrays["mass_lift_last"].append(context_mass_lift(observed, expected))

        row = similarity.get(max(positions), {})
        for name in (
            "density_k1_last",
            "density_k3_last",
            "density_k5_last",
            "density_k3_middle",
            "sim_max_last",
            "sim_contrast_last",
            "sim_margin_last",
            "sim_distance_last",
            "sim_decay_last",
        ):
            arrays[name].append(float(row.get(name, 0.0)))

        word = token.text.lower().strip("«».,;:!?()")
        stems = {stem[:5] for stem in (word,)} if len(word) >= 4 else set()
        arrays["lex_match"].append(1.0 if stems and (stems & lexicon["stems"]) else 0.0)
        token_numbers = set(numbers_in(token.text))
        arrays["num_match"].append(1.0 if token_numbers and token_numbers & document_numbers else 0.0)
        arrays["trigram_match"].append(_trigram_match(token.text, lexicon["stems"]))
        arrays["attrib_borrowed"].append(1.0 if _covers(borrowed, token) else 0.0)
        arrays["attrib_supported"].append(1.0 if _covers(supported, token) else 0.0)
        arrays["position"].append(index / total_tokens)
        arrays["length"].append(min(1.0, len(token.text.strip()) / 20))
        sentence_number = next(
            (number for number, (start, end) in enumerate(answer_sentences) if start <= token.start < end), -1
        )
        if sentence_number >= 0:
            start, end = answer_sentences[sentence_number]
            arrays["sentence_position"].append(min(1.0, max(0.0, (token.start - start) / max(1, end - start))))
        else:
            arrays["sentence_position"].append(0.5)
        arrays["sentence_support"].append(
            float(sentence_support[sentence_number]) if 0 <= sentence_number < len(sentence_support) else 0.0
        )

    arrays["mass_win3"] = _window(arrays["mass_last"], window)
    arrays["entropy_win3"] = _window(arrays["entropy_last"], window)

    meta = {
        "backend": "hf",
        "model": loaded.model_name,
        "revision": loaded.revision,
        "layers": layers,
        "layer_roles": layer_roles,
        "seq_len": seq_len,
        "prompt_chars": len(prompt),
        "max_length": max_length,
        "context_tokens": len(context_positions),
        "answer_model_tokens": len(answer_positions),
        "unmatched_tokens": unmatched,
        "offsets_mode": offsets_mode,
        "k_values": list(k_values),
        "window": window,
        "grid_format": GRID_CACHE_FORMAT,
        "answer_tokens": len(tokens),
    }
    return {
        "arrays": arrays,
        "tokens": [
            {"text": token.text, "start": token.start, "end": token.end, "scored": int(is_scored_token(token.text))}
            for token in tokens
        ],
        "meta": meta,
    }


def _sentence_support(answer: str, sentences: Sequence[tuple[int, int]], lexicon: dict[str, Any]) -> list[float]:
    """Похожесть каждого предложения ответа на лучшее предложение документа.

    Токен «внутри неподтверждённого предложения» и токен внутри подтверждённого —
    разные случаи, и маска без этого признака их не различает. Доля стеммов
    предложения ответа, найденных в одном предложении документа, — та же мера,
    что использует проверка покрытия, но как непрерывный признак.
    """
    from .normalize import stem_text  # noqa: PLC0415

    document: list[set[str]] = lexicon.get("sentence_stems") or []
    out: list[float] = []
    for start, end in sentences:
        stems = {stem[:5] for stem in stem_text(answer[start:end]) if len(stem) >= 4}
        if not stems or not document:
            out.append(0.0)
            continue
        out.append(max((len(stems & candidate) / len(stems)) for candidate in document))
    return out


def _similarity_rows(  # pragma: no cover - требует torch: матрица косинусов по hidden_states
    *,
    hidden_states: tuple[Any, ...],
    layer_roles: dict[str, int],
    answer_positions: Sequence[int],
    context_positions: Sequence[int],
    seq_len: int,
    k_values: Sequence[int],
) -> dict[int, dict[str, float]]:
    """Похожесть подслов ответа на подсловы документа — матрицей, один раз.

    Ключ результата — позиция подслова модели; строка — значения ``density_k*``
    и ``sim_*``. Все редукции (топ-k, максимум, второй максимум, средний фон,
    расстояние до опоры, спад по расстоянию) считаются по строкам матрицы:
    вложенный цикл «токен × токен контекста» в Python давал ~4 с на пару и не
    позволял перебрать несколько ``k``.
    """
    result: dict[int, dict[str, float]] = {}
    if not hidden_states or not context_positions or not answer_positions:
        return result
    import torch  # noqa: PLC0415

    def table(role: str) -> Any:
        index = min(layer_roles[role] + 1, len(hidden_states) - 1)
        return hidden_states[index][0]

    def cosine(rows_source: Any, hidden: Any) -> Any:
        rows = hidden[list(rows_source)].float()
        context = hidden[list(context_positions)].float()
        rows = rows / rows.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        context = context / context.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return rows @ context.T

    top_k = max(1, max(k_values))
    cosine_last = cosine(answer_positions, table("last"))
    values, indices = torch.topk(cosine_last, k=min(top_k, cosine_last.shape[-1]), dim=-1)
    ordered = torch.sort(cosine_last, dim=-1, descending=True).values
    mean_row = cosine_last.mean(dim=-1)
    second_row = ordered[:, 1] if ordered.shape[1] > 1 else ordered[:, 0]
    middle_top = None
    if len(hidden_states) > 2 and "middle" in layer_roles:
        middle_top = torch.topk(
            cosine(answer_positions, table("middle")), k=min(3, len(context_positions)), dim=-1
        ).values.mean(dim=-1)
    decay_scale = max(1.0, 0.1 * seq_len)

    for order, position in enumerate(answer_positions):
        best = float(values[order, 0].item())
        distance = int(abs(position - context_positions[int(indices[order, 0].item())]))
        entry = {
            "sim_max_last": best,
            "sim_contrast_last": similarity_contrast(best, float(mean_row[order].item())),
            "sim_margin_last": similarity_margin(best, float(second_row[order].item())),
            "sim_distance_last": normalised_support_distance(distance, seq_len),
            "sim_decay_last": distance_decay(best, distance, decay_scale),
            "density_k3_middle": float(middle_top[order].item()) if middle_top is not None else 0.0,
        }
        for k in k_values:
            take = min(int(k), values.shape[-1])
            entry[f"density_k{k}_last"] = float(values[order, :take].mean().item())
        result[int(position)] = entry
    return result
