"""Три признака недостоверности ответа относительно контекста.

Признаки (по одному значению на токен ответа, все в [0, 1]):

1. ``attention_entropy`` — нормированная энтропия внимания последнего слоя
   (режим ``hf``) либо стилометрическая предсказуемость (режим ``demo``).
   Чем выше — тем менее уверенно токен «поддерживается» моделью.
2. ``ctx_attention_mass`` — доля массы внимания, пришедшая на токены контекста
   (режим ``hf``) либо масса лексической поддержки токена контекстом
   (режим ``demo``). В итоговом риске входит со знаком минус: больше опоры на
   контекст — ниже риск.
3. ``embedding_density`` — 1 − среднее косинусное расстояние до ``k = 5``
   ближайших чанков контекста (режим ``hf``: скрытые состояния; режим
   ``demo``: лексические векторы с символьными n-граммами).

Итоговый риск: ``r = w1·Ĥ + w2·(1 − m̂) + w3·d̂`` после нормировки признаков
в [0, 1]. Веса подбираются на валидации (см. ``train.py``).

Режим ``demo`` не использует torch: это лексические суррогаты признаков. Он
нужен, чтобы конвейер, интерфейс и сборка проверялись без весов модели; его
числа не являются научным результатом.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .core import Token, number_value, numbers_in, split_chunks
from .lexicon import BOILERPLATE_WORDS, PARAPHRASE_WORDS, STOPWORDS, find_phrase_hits
from .vectors import hash_index, normalize

FEATURE_NAMES = ("attention_entropy", "ctx_attention_mass", "embedding_density")
SCORED_MIN_LEN = 3
# Значения по умолчанию — не «на глаз»: это режим, к которому сходится перебор
# весов на демонстрационном корпусе (масса опоры на контекст несёт основную
# нагрузку). Обучение на своих данных заменяет их (``spanverify train``).
DEFAULT_WEIGHTS = {"attention_entropy": 0.1, "ctx_attention_mass": 0.8, "embedding_density": 0.1}
DEMO_WARNING = (
    "ДЕМО-РЕЖИМ: признаки считаются лексическими суррогатами без весов языковой модели. "
    "Он проверяет работоспособность конвейера и интерфейса, а не достоверность ответа. "
    "Научные выводы возможны только в режиме 'hf'."
)

_CONTENT_MIN_LEN = 3


def is_scored_token(text: str, min_len: int = SCORED_MIN_LEN) -> bool:
    """Содержательный токен: по нему осмысленно судить о достоверности.

    Пунктуация и короткие служебные слова не могут быть выдумкой, поэтому они
    исключаются из оценки риска (но остаются в посимвольных координатах ответа).
    Единый предикат используется и обучением, и движком — иначе метрики
    обучения и поведение API разошлись бы.
    """
    word = text.strip().strip(".,;:!?()«»\"'—–-").lower().replace("ё", "е")
    if not word:
        return False
    if len(word) < min_len and not word.isdigit():
        return False
    return not (word in STOPWORDS and len(word) <= min_len)


@dataclass
class FeatureMatrix:
    """Матрица признаков по токенам ответа."""

    attention_entropy: list[float] = field(default_factory=list)
    ctx_attention_mass: list[float] = field(default_factory=list)
    embedding_density: list[float] = field(default_factory=list)
    token_risk: list[float] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.attention_entropy)

    def as_rows(self) -> list[dict[str, float]]:
        return [
            {
                "attention_entropy": self.attention_entropy[i],
                "ctx_attention_mass": self.ctx_attention_mass[i],
                "embedding_density": self.embedding_density[i],
                "risk": self.token_risk[i],
            }
            for i in range(len(self))
        ]


# ---------------------------------------------------------------- нормировка


def scale(values: Sequence[float]) -> list[float]:
    """Нормировка по максимуму: сохраняет абсолютный смысл нуля.

    Признаки имеют собственные шкалы: масса опоры на контекст равна 0, когда
    поддержки нет, и 1, когда токен полностью подтверждён. Сдвиг min–max эту
    границу разрушает (любой ответ получал бы токен с максимальным риском),
    поэтому нормируем делением на максимум.
    """
    if not values:
        return []
    high = max(values)
    if high <= 1e-12:
        return [0.0] * len(values)
    return [min(1.0, max(0.0, value / high)) for value in values]


def _clamp01(value: float) -> float:
    """Ограничить значение отрезком [0, 1] (NaN → 0)."""
    if value != value:  # NaN
        return 0.0
    return min(1.0, max(0.0, float(value)))


# Историческое имя (использовалось в отчётах и тестах).
minmax = scale


def combine(
    entropy: Sequence[float],
    mass: Sequence[float],
    density: Sequence[float],
    weights: dict[str, float] | None = None,
) -> list[float]:
    """Взвешенная комбинация признаков: r = w1·Ĥ + w2·(1 − m̂) + w3·d̂.

    Нормируется только ``attention_entropy``: у неё нет абсолютного нуля, важен
    лишь порядок токенов внутри ответа. ``ctx_attention_mass`` и
    ``embedding_density`` имеют абсолютный смысл (0 = опоры нет, 1 = токен
    подтверждён), поэтому они НЕ делятся на собственный максимум: иначе ответ,
    где слабо поддержаны все токены, выглядел бы полностью подтверждённым.
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    total = sum(abs(weights.get(name, 0.0)) for name in FEATURE_NAMES) or 1.0
    e_hat = scale(entropy)
    m_hat = [_clamp01(value) for value in mass]
    d_hat = [_clamp01(value) for value in density]
    out: list[float] = []
    for i in range(len(e_hat)):
        risk = (
            weights["attention_entropy"] * e_hat[i]
            + weights["ctx_attention_mass"] * (1.0 - m_hat[i])
            + weights["embedding_density"] * d_hat[i]
        )
        out.append(min(1.0, max(0.0, risk / total)))
    return out


# ---------------------------------------------------------------- demo-режим


def _is_content(token: Token) -> bool:
    word = token.word.lower()
    return bool(word) and (len(word) >= _CONTENT_MIN_LEN or word.isdigit())


def demo_features(
    answer: str,
    context: str | Sequence[str] | None,
    answer_tokens: Sequence[Token] | None = None,
    k: int = 5,
    dim: int = 2048,
) -> FeatureMatrix:
    """Лексические суррогаты трёх признаков (без весов модели)."""
    from .core import tokenize_with_offsets

    tokens = list(answer_tokens) if answer_tokens is not None else tokenize_with_offsets(answer)
    if not tokens:
        return FeatureMatrix(meta={"backend": "demo", "empty": True})

    chunks = split_chunks(context)
    context_tokens = tokenize_with_offsets(chunks.text) if chunks.text else []

    # --- контекстные слова (для лексической поддержки и словаря опоры) ---
    context_numbers = numbers_in(chunks.text) if chunks.text else set()
    context_words: list[tuple[str, frozenset[str]]] = [
        (token.word, _trigrams(token.word)) for token in context_tokens if _is_content(token)
    ]

    # --- признаки ---
    phrase_hits = find_phrase_hits([token.text for token in tokens])
    seen: dict[str, int] = {}
    entropy: list[float] = []
    mass: list[float] = []
    density: list[float] = []

    for index, token in enumerate(tokens):
        word = token.word.lower()
        content = _is_content(token)
        seen[word] = seen.get(word, 0) + 1

        # 1. Предсказуемость «самого по себе»: шаблонность, повтор, ровность.
        smooth = 0.0
        if word in BOILERPLATE_WORDS:
            smooth += 0.6
        if index in phrase_hits:
            smooth += 0.4
        if seen[word] > 1:
            smooth += 0.2
        entropy.append(min(1.0, smooth))

        # 2. Лексическая поддержка контекстом (суррогат массы внимания).
        support = 0.0
        if content and context_words:
            support = _support_overlap(word, context_words)
        value = number_value(token)
        if value is not None and context_numbers:
            # Число подтверждено, только если такое же значение есть в контексте.
            support = 1.0 if value in context_numbers else 0.0
        elif word in PARAPHRASE_WORDS:
            # Служебный оборот-парафраз: лексический суррогат не отличает
            # «составляет» от «установлен в размере», поэтому считаем, что опора
            # есть. Явное упрощение демо-режима (см. PARAPHRASE_WORDS).
            support = 1.0
        mass.append(min(1.0, max(0.0, support)))

        # 3. Близость к ближайшим словам контекста (доля общих триграмм).
        #    Считается по словам, а не по целым чанкам: пересечение триграмм
        #    короткого слова с длинным предложением почти всегда около нуля и
        #    не различает токены.
        grams = _trigrams(token.word) if content else frozenset()
        if grams and context_words:
            similarities = sorted((_jaccard(grams, context_gram) for _, context_gram in context_words), reverse=True)
            top = similarities[: max(1, k)]
            density.append(min(1.0, max(0.0, sum(top) / len(top))))
        else:
            density.append(0.0)

    _apply_number_consistency(tokens, context_numbers, mass)

    return FeatureMatrix(
        attention_entropy=entropy,
        ctx_attention_mass=mass,
        embedding_density=density,
        meta={
            "backend": "demo",
            "token_count": len(tokens),
            "context_tokens": len(context_tokens),
            "context_numbers": sorted(context_numbers),
            "k": k,
        },
    )


def _trigrams(text: str, n: int = 3) -> frozenset[str]:
    """Символьные триграммы слова с границами — устойчивы к формам слова."""
    value = text.lower().replace("ё", "е").strip(".,;:!?()«»\"'—–-")
    if not value:
        return frozenset()
    padded = "^" + value + "$"
    return frozenset(padded[i : i + n] for i in range(max(1, len(padded) - n + 1)))


def _stem(word: str, n: int = 4) -> str:
    """Грубая основа слова: первые ``n`` символов (для русского этого достаточно)."""
    clean = word.lower().replace("ё", "е").strip(".,;:!?()«»\"'—–-")
    return clean[:n] if len(clean) > n else clean


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _support_overlap(word: str, context_words: Sequence[tuple[str, frozenset[str]]]) -> float:
    """Опора токена на контекст: общая основа = 1.0, иначе доля общих триграмм.

    Это осознанно «жёсткий» суррогат: он даёт почти бинарную шкалу (слово либо
    поддержано документом, либо нет), поэтому порог маски на нём осмыслен.
    """
    stem = _stem(word)
    grams: frozenset[str] | None = None
    best = 0.0
    for context_word, context_grams in context_words:
        if _stem(context_word) == stem:
            return 1.0
        if grams is None:
            grams = _trigrams(word)
        best = max(best, 0.7 * _jaccard(grams, context_grams))
    return best


def _apply_number_consistency(tokens: Sequence[Token], context_numbers: set[str], mass: list[float]) -> None:
    """Число, противоречащее контексту, не может считаться поддержанным.

    Если в контексте есть числовые значения, но конкретное число ответа там
    отсутствует — это признак подмены факта (сильнейший сигнал в демо-режиме).
    """
    if not context_numbers:
        return
    for index, token in enumerate(tokens):
        value = number_value(token)
        if value is not None and value not in context_numbers:
            mass[index] = 0.0


def _lexical_vector(token: Token, dim: int) -> dict[int, float]:
    return _lexical_vector_from_text(token.lower, dim)


def _lexical_vector_from_text(text: str, dim: int, n: int = 3) -> dict[int, float]:
    """Разреженный вектор: слово + символьные n-граммы (устойчив к формам)."""
    value = text.lower().replace("ё", "е")
    vector: dict[int, float] = {}
    for word in value.split():
        clean = word.strip(".,;:!?()«»\"'")
        if not clean:
            continue
        vector[hash_index("w:" + clean, dim)] = vector.get(hash_index("w:" + clean, dim), 0.0) + 1.0
        padded = "^" + clean + "$"
        for i in range(max(1, len(padded) - n + 1)):
            gram = padded[i : i + n]
            idx = hash_index("g:" + gram, dim)
            vector[idx] = vector.get(idx, 0.0) + 0.5
    return normalize(vector)


# ---------------------------------------------------------------- hf-режим


def hf_features(
    answer: str,
    context: str | Sequence[str] | None,
    model_name: str = "ai-forever/rugpt3small_based_on_gpt2",
    k: int = 5,
    device: str | None = None,
    max_length: int = 1024,
    answer_tokens: Sequence[Token] | None = None,
    layer: int | str = "last",
) -> FeatureMatrix:
    """Реальные признаки модели: энтропия внимания, масса на контекст, плотность.

    Требует ``torch`` и ``transformers`` (ленивый импорт внутри функции).

    ``layer`` выбирает слой внимания: ``"first"``, ``"middle"``, ``"last"``,
    отрицательный индекс (``-4`` — четвёртый с конца) или целое число. Это нужно
    пилоту: он сравнивает информативность признаков по слоям и не должен
    зависеть от того, что «полезный» слой оказался не последним.
    """
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    from .backends.base import BackendUnavailable  # noqa: PLC0415
    from .core import tokenize_with_offsets  # noqa: PLC0415

    tokens = list(answer_tokens) if answer_tokens is not None else tokenize_with_offsets(answer)
    if not tokens:
        return FeatureMatrix(meta={"backend": "hf", "empty": True})

    chunks = split_chunks(context)
    context_text = chunks.text
    prompt = f"{context_text}\n{answer}" if context_text else answer
    answer_start = len(prompt) - len(answer)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", output_attentions=True)
    except Exception as exc:  # pragma: no cover - зависит от окружения
        raise BackendUnavailable(
            f"не удалось загрузить модель {model_name}: {type(exc).__name__}: {exc}. "
            "Проверьте доступ к весам или укажите другую модель."
        ) from exc

    model.eval()
    torch_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(torch_device)

    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    attention_mask = encoded["attention_mask"]
    with torch.no_grad():
        outputs = model(**encoded)

    layer_index = resolve_layer(layer, len(outputs.attentions))
    attentions = outputs.attentions[layer_index][0]  # (heads, seq, seq)
    hidden = None
    if outputs.hidden_states:
        hidden = outputs.hidden_states[min(layer_index + 1, len(outputs.hidden_states) - 1)][0]

    # Позиции токенов ответа (по смещениям быстрого токенизатора).
    offsets = tokenizer(prompt, return_offsets_mapping=True, truncation=True, max_length=max_length)["offset_mapping"]
    seq_len = int(attention_mask.sum().item())
    answer_positions = [
        i for i, (start, end) in enumerate(offsets.tolist()) if end > start and start >= answer_start and i < seq_len
    ]
    answer_start_token = min(answer_positions) if answer_positions else seq_len
    context_positions = [i for i in range(seq_len) if i < answer_start_token]

    eps = 1e-9
    probs = attentions.clamp_min(eps)
    entropy = -(probs * probs.log()).sum(dim=-1).mean(dim=0)  # (seq,)
    import math  # noqa: PLC0415

    entropy = entropy / max(eps, math.log(max(2, seq_len)))

    entropy_values: list[float] = []
    mass_values: list[float] = []
    density_values: list[float] = []

    # Векторы контекстных чанков и токенов — для плотности.
    context_token_embeddings: list[list[float]] = []
    hidden_cpu = None
    if hidden is not None:
        hidden_cpu = hidden.detach().to("cpu")
        for position in context_positions:
            context_token_embeddings.append(hidden_cpu[position].tolist())

    for local_index, _token in enumerate(tokens):
        if local_index >= len(answer_positions):
            entropy_values.append(0.5)
            mass_values.append(0.0)
            density_values.append(0.0)
            continue
        position = answer_positions[local_index]
        entropy_values.append(float(entropy[position].item()))
        row = attentions[:, position, :].mean(dim=0)  # (seq,)
        mass = float(row[context_positions].sum().item()) if context_positions else 0.0
        mass_values.append(min(1.0, max(0.0, mass)))

        if hidden_cpu is not None and context_token_embeddings:
            vector = hidden_cpu[position].tolist()
            similarities = sorted(
                (_cosine_dense(vector, other) for other in context_token_embeddings),
                reverse=True,
            )
            top = similarities[: max(1, k)]
            density_values.append(min(1.0, max(0.0, sum(top) / len(top))))
        else:  # pragma: no cover - модель без hidden_states
            density_values.append(0.0)

    return FeatureMatrix(
        attention_entropy=entropy_values,
        ctx_attention_mass=mass_values,
        embedding_density=density_values,
        meta={
            "backend": "hf",
            "model": model_name,
            "layer": layer if isinstance(layer, (int, str)) else str(layer),
            "layer_index": layer_index,
            "layers_total": len(outputs.attentions),
            "seq_len": seq_len,
            "context_tokens": len(context_positions),
            "k": k,
        },
    )


def resolve_layer(layer: int | str, total: int) -> int:
    """Индекс слоя внимания по имени или числу.

    Поддерживаются ``"first"``, ``"middle"``, ``"last"``, отрицательные индексы
    (``-4`` — четвёртый с конца) и обычные целые числа. Индекс всегда попадает
    в диапазон ``[0, total - 1]``.
    """
    if total <= 0:
        return 0
    if isinstance(layer, bool):  # bool — подкласс int, отсекаем отдельно
        return total - 1
    if isinstance(layer, int):
        index = layer if layer >= 0 else total + layer
        return max(0, min(total - 1, index))
    key = str(layer).strip().lower()
    if key in {"last", "-1", ""}:
        return total - 1
    if key == "first":
        return 0
    if key == "middle":
        return (total - 1) // 2
    try:
        index = int(key)
    except ValueError:
        return total - 1
    return max(0, min(total - 1, index if index >= 0 else total + index))


def _cosine_dense(a: Sequence[float], b: Sequence[float]) -> float:
    numerator = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return max(0.0, numerator / (norm_a * norm_b))


def extract_features(
    answer: str,
    context: str | Sequence[str] | None,
    mode: str = "demo",
    **kwargs: Any,
) -> FeatureMatrix:
    """Единая точка входа: признаки заказанного режима."""
    if mode == "hf":
        return hf_features(answer, context, **kwargs)
    return demo_features(answer, context, **kwargs)
