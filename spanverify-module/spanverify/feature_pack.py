"""Доработка признаков: нормировка массы, расстояние до опоры, лексико-семантика.

Пункт 2.2 промта требует не одной попытки, а нескольких итераций доработки
признаков, каждая с числом (AUC + 95 % ДИ, бутстрэп). Здесь собраны признаки,
которые добавляются поверх базовых ``attention_entropy`` / ``ctx_attention_mass`` /
``embedding_density``:

Итерация 1 — база (замеряет ``scripts/feature_iterations.py``, ничего не меняя).
Итерация 2 — нормировка массы внимания: масса делится на число токенов контекста
    (``mass_per_context_token``) и на длину ответа (``mass_per_answer_token``);
    энтропия нормируется на ``log(seq_len)``.
Итерация 3 — позиция токена в ответе (``position``) и расстояние по токенам до
    ближайшего подтверждающего фрагмента документа (``evidence_distance``).
Итерация 4 — лексико-семантические признаки поверх внимания: совпадение чисел и
    единиц измерения (``number_support``, ``unit_support``), конфликт отрицания
    (``negation_conflict``), конфликт модальности (``modality_conflict``),
    подтверждение дат (``date_support``).

Все признаки детерминированы, масштабированы в [0, 1] и считаются без сети.
Гипотеза «масса внимания на контекст информативнее сырой энтропии» проверяется
сравнением AUC пары ``ctx_attention_mass`` и ``attention_entropy``: результат
пишется в отчёт, а не подгоняется.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from .core import Token
from .normalize import canonical_unit, lemmatize, measure_mentions, normalize_text

__all__ = [
    "FEATURE_ITERATIONS",
    "EXTRA_FEATURE_NAMES",
    "extra_features",
    "iteration_features",
]

# Названия добавляемых признаков (порядок соответствует итерациям).
EXTRA_FEATURE_NAMES: tuple[str, ...] = (
    "mass_per_context_token",
    "mass_per_answer_token",
    "position",
    "evidence_distance",
    "number_support",
    "unit_support",
    "negation_conflict",
    "modality_conflict",
    "date_support",
)

# Итерации доработки: какие признаки добавляются на каждой.
FEATURE_ITERATIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("1-baseline", ()),
    ("2-mass-normalized", ("mass_per_context_token", "mass_per_answer_token")),
    ("3-position-evidence", ("position", "evidence_distance")),
    (
        "4-lexical-semantic",
        ("number_support", "unit_support", "negation_conflict", "modality_conflict", "date_support"),
    ),
)

NEGATION_WORDS = ("не", "нет", "нельзя", "запрещ", "кроме", "без")
MODALITY_WORDS = ("запрещ", "не допуска", "обязан", "должен", "вправе", "подлежит")
DATE_RE = re.compile(r"\b(19|20)\d{2}\b")


@dataclass(frozen=True)
class EvidenceIndex:
    """Индекс опоры: какие числа/единицы/годы документа в каком порядке встречаются."""

    numbers: tuple[float, ...]
    units: tuple[str, ...]
    years: tuple[str, ...]
    lemmas: frozenset[str]


def build_evidence_index(context: str) -> EvidenceIndex:
    """Собрать индекс опоры из документа: числа, единицы, годы, леммы."""
    if not context:
        return EvidenceIndex((), (), (), frozenset())
    mentions = measure_mentions(context)
    numbers = tuple(mention.value for mention in mentions)
    units = tuple(unit for unit in (mention.unit for mention in mentions) if unit)
    normalized = normalize_text(context)
    years = tuple(match.group(0) for match in DATE_RE.finditer(normalized))
    lemmas = frozenset(
        lemma
        for lemma in (lemmatize(word) for word in re.findall(r"[A-Za-zА-Яа-яЁё]{3,}", normalize_text(context)))
        if lemma
    )
    return EvidenceIndex(numbers, units, years, lemmas)


def _unit_mentions_by_value(context: str) -> dict[float, set[str]]:
    mapping: dict[float, set[str]] = {}
    for mention in measure_mentions(context):
        mapping.setdefault(mention.value, set())
        if mention.unit:
            mapping[mention.value].add(mention.unit)
    return mapping


def _token_has_evidence(token: Token, index_: EvidenceIndex) -> bool:
    """Есть ли у токена лексическая опора в документе (число, год или лемма)."""
    word = normalize_text(token.word)
    if not word:
        return False
    if word.isdigit():
        return float(word) in index_.numbers
    if DATE_RE.fullmatch(word):
        return word in index_.years
    lemma = lemmatize(word)
    return bool(lemma) and lemma in index_.lemmas


def extra_features(
    answer: str,
    context: str | None,
    tokens: Sequence[Token],
    *,
    mass: Sequence[float] | None = None,
    context_tokens: int | None = None,
) -> dict[str, list[float]]:
    """Дополнительные признаки по токенам ответа.

    ``mass`` — масса внимания на контекст (режим hf) либо её лексический суррогат
    (режим demo). Если не передана, признаки нормировки массы заполняются нулями:
    тогда итерация 2 честно показывает 0.5 (случайный уровень), а не «улучшение».
    """
    size = len(tokens)
    empty = {name: [0.0] * size for name in EXTRA_FEATURE_NAMES}
    if size == 0:
        return empty
    index_ = build_evidence_index(context or "")
    unit_map = _unit_mentions_by_value(context or "")
    answer_mentions = measure_mentions(answer)
    answer_numbers_by_token: dict[int, float] = {}
    for mention in answer_mentions:
        for position, token in enumerate(tokens):
            if token.start >= mention.start and token.end <= mention.end:
                answer_numbers_by_token[position] = mention.value
    context_size = max(1, int(context_tokens or max(1, len(index_.lemmas))))
    evidence_flags = [_token_has_evidence(token, index_) for token in tokens]
    context_modal = any(marker in normalize_text(context or "") for marker in MODALITY_WORDS)
    context_negated = _has_negation(context or "")
    answer_negated = _has_negation(answer)

    mass_values = [float(value) for value in (mass or [])]
    while len(mass_values) < size:
        mass_values.append(0.0)

    features: dict[str, list[float]] = {name: [] for name in EXTRA_FEATURE_NAMES}
    nearest = _nearest_evidence_distance(evidence_flags)
    for position, token in enumerate(tokens):
        value = mass_values[position]
        # Итерация 2: абсолютная масса растёт с длиной контекста просто потому, что
        # токенов больше; делим на число токенов контекста и на длину ответа.
        features["mass_per_context_token"].append(_clamp(value / context_size))
        features["mass_per_answer_token"].append(_clamp(value * size / context_size))
        # Итерация 3: позиция и расстояние до ближайшего подтверждённого токена.
        features["position"].append(_clamp(position / max(1, size - 1)))
        features["evidence_distance"].append(_clamp(nearest[position] / max(1, size)))
        # Итерация 4: лексико-семантические признаки.
        number_value = answer_numbers_by_token.get(position)
        supported = float(number_value is not None and number_value in index_.numbers)
        features["number_support"].append(supported)
        unit_supported = 0.0
        if number_value is not None:
            unit = canonical_unit(token.word)
            if unit and unit in unit_map.get(number_value, set()):
                unit_supported = 1.0
        features["unit_support"].append(unit_supported)
        features["negation_conflict"].append(float(context_negated != answer_negated))
        features["modality_conflict"].append(
            float(context_modal and not any(marker in normalize_text(token.word) for marker in MODALITY_WORDS))
        )
        year = DATE_RE.fullmatch(normalize_text(token.word))
        features["date_support"].append(float(bool(year) and normalize_text(token.word) in index_.years))
    return features


def _has_negation(text: str) -> bool:
    lowered = normalize_text(text)
    return any(re.search(rf"(^|[^a-zа-я]){marker}", lowered) is not None for marker in NEGATION_WORDS)


def _nearest_evidence_distance(flags: Sequence[bool]) -> list[int]:
    """Расстояние (в токенах) до ближайшего токена с опорой в документе."""
    size = len(flags)
    distance = [size] * size
    last = None
    for position in range(size):
        if flags[position]:
            last = position
        if last is not None:
            distance[position] = position - last
    last = None
    for position in range(size - 1, -1, -1):
        if flags[position]:
            last = position
        if last is not None:
            distance[position] = min(distance[position], last - position)
    return distance


def _clamp(value: float) -> float:
    if value != value:  # NaN
        return 0.0
    return min(1.0, max(0.0, float(value)))


def iteration_features(
    iteration: str,
    base: dict[str, Sequence[float]],
    extras: dict[str, Sequence[float]],
) -> dict[str, Sequence[float]]:
    """Набор признаков, доступный на данной итерации (база + добавленные)."""
    available: dict[str, Sequence[float]] = dict(base)
    added: tuple[str, ...] = ()
    for name, names in FEATURE_ITERATIONS:
        added = (*added, *names)
        if name == iteration:
            break
    for name in added:
        if name in extras:
            available[name] = extras[name]
    return available
