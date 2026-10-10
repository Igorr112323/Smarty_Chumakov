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

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .core import Token, number_value, numbers_in, split_chunks
from .lexicon import BOILERPLATE_WORDS, PARAPHRASE_WORDS, STOPWORDS, find_phrase_hits
from .normalize import numbers_to_digits, words_to_number
from .vectors import hash_index, normalize

FEATURE_NAMES = ("attention_entropy", "ctx_attention_mass", "embedding_density")
# Признаки-кандидаты: считаются рядом с рабочими, но НЕ входят в итоговый риск
# (в DEFAULT_WEIGHTS их нет). Они нужны, чтобы измерить на реальной модели,
# даёт ли нормировка массы на длину контекста прибавку к AUC, и только после
# измерения решать, переводить ли их в рабочие.
DIAGNOSTIC_FEATURES = (
    "ctx_attention_mass_norm",
    "ctx_attention_mass_lift",
    # Итерация 2: расстояние до подтверждающего фрагмента.
    "ctx_max_similarity",
    "ctx_support_distance",
    "ctx_similarity_decay",
    # Итерация 3: выделенность максимума над фоном и над вторым источником.
    "ctx_sim_contrast",
    "ctx_sim_margin",
)
# ---------------------------------------------------------------- признаки головы
#
# Голова (логистическая регрессия ``train._train_and_compare_head``) получает на
# вход не только три рабочих признака, но и производные строки ответа, и — с этой
# итерации — диагностику итераций 2–3. Список имён сохраняется в ``head.json`` в
# поле ``features``, поэтому применение головы обязано собирать строку по именам,
# а не по позициям: иначе голова, обученная на восьми признаках, молча
# применилась бы к шести (именно так терялись числа в прогоне 37472951524).
HEAD_FEATURE_DERIVED = ("risk", "position", "length")
HEAD_FEATURE_BASE: tuple[str, ...] = (*FEATURE_NAMES, *HEAD_FEATURE_DERIVED)
# Имя признака головы -> поле FeatureMatrix. Производные (risk/position/length)
# здесь отсутствуют сознательно: они вычисляются, а не хранятся.
HEAD_FEATURE_FIELDS: dict[str, str] = {
    "attention_entropy": "attention_entropy",
    "ctx_attention_mass": "ctx_attention_mass",
    "embedding_density": "embedding_density",
    "ctx_attention_mass_norm": "ctx_mass_norm",
    "ctx_attention_mass_lift": "ctx_mass_lift",
    "ctx_mass_norm": "ctx_mass_norm",
    "ctx_mass_lift": "ctx_mass_lift",
    "ctx_mass_expected": "ctx_mass_expected",
    "ctx_max_similarity": "ctx_max_similarity",
    "ctx_support_distance": "ctx_support_distance",
    "ctx_similarity_decay": "ctx_similarity_decay",
    "ctx_sim_contrast": "ctx_sim_contrast",
    "ctx_sim_margin": "ctx_sim_margin",
}
# Кандидаты в голову: диагностику включаем только после измерения AUC на
# валидации (см. scripts/train_hf_a3.py). Порядок фиксирован — он определяет
# порядок перебора при отборе признаков и должен быть воспроизводим.
HEAD_FEATURE_CANDIDATES: tuple[str, ...] = (
    "ctx_attention_mass_norm",
    "ctx_attention_mass_lift",
    "ctx_max_similarity",
    "ctx_support_distance",
    "ctx_similarity_decay",
    "ctx_sim_contrast",
    "ctx_sim_margin",
)


def head_feature_series(features: FeatureMatrix, name: str) -> list[float]:
    """Значения признака головы по имени (пустой список, если поля нет)."""
    field = HEAD_FEATURE_FIELDS.get(name)
    if field is None:
        return []
    return [float(value) for value in (getattr(features, field, None) or [])]


def head_feature_vector(
    features: FeatureMatrix,
    risk: Sequence[float],
    index: int,
    total: int,
    token: Token | None,
    names: Sequence[str] = HEAD_FEATURE_BASE,
) -> list[float]:
    """Строка признаков головы для токена ``index`` в порядке имён ``names``.

    Единая функция для обучения и применения: если они построят строку каждая по
    своему, голова начнёт получать перепутанные признаки, и это не будет видно
    ни в каких метриках обучения.
    """
    row: list[float] = []
    for name in names:
        if name == "risk":
            row.append(float(risk[index]) if index < len(risk) else 0.0)
        elif name == "position":
            row.append(index / max(1, total))
        elif name == "length":
            text = token.text.strip() if token is not None else ""
            row.append(min(1.0, len(text) / 20))
        else:
            series = head_feature_series(features, name)
            row.append(float(series[index]) if index < len(series) else 0.0)
    return row


def has_head_features(matrix: FeatureMatrix, names: Sequence[str]) -> bool:
    """Есть ли в матрице все признаки головы (кроме производных).

    Нужна для проверки кеша: кеш старого формата содержит только три рабочих
    признака, и голова с восемью признаками получила бы нули вместо диагностики
    — метрики упали бы без единого сообщения об ошибке.
    """
    for name in names:
        if name in HEAD_FEATURE_DERIVED:
            continue
        if name not in HEAD_FEATURE_FIELDS:
            return False
        if not head_feature_series(matrix, name):
            return False
    return True


MEASUREMENT_SUBJECT_WINDOW = 6  # сколько слов перед числом считаем его субъектом
MEASUREMENT_MATCH_MIN = 0.5  # порог совпадения субъекта ответа с измерением контекста
SCORED_MIN_LEN = 3
# Значения по умолчанию — не «на глаз»: это режим, к которому сходится перебор
# весов на демонстрационном корпусе (масса опоры на контекст несёт основную
# нагрузку). Обучение на своих данных заменяет их (``spanverify train``).
DEFAULT_WEIGHTS = {"attention_entropy": 0.1, "ctx_attention_mass": 0.8, "embedding_density": 0.1}
DEMO_WARNING = (
    "ДЕМО-РЕЖИМ: признаки считаются лексическими суррогатами без весов языковой модели. "
    "Он проверяет работоспособность конвейера и интерфейса, а не достоверность ответа. "
    "Подмена числа ловится, если число есть у другого объекта документа (например, "
    "«первичные — пять лет» против «вторичные — десять лет»); перефразирования и "
    "таблицы требуют режима 'hf'. Научные выводы возможны только в режиме 'hf'."
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
    # Диагностические (не участвуют в риске): нормированная масса опоры.
    ctx_mass_expected: list[float] = field(default_factory=list)
    ctx_mass_norm: list[float] = field(default_factory=list)
    ctx_mass_lift: list[float] = field(default_factory=list)
    # Диагностические (итерация 2): максимум похожести и где он найден.
    ctx_max_similarity: list[float] = field(default_factory=list)
    ctx_support_distance: list[float] = field(default_factory=list)
    ctx_similarity_decay: list[float] = field(default_factory=list)
    ctx_sim_contrast: list[float] = field(default_factory=list)
    ctx_sim_margin: list[float] = field(default_factory=list)
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


def expected_context_mass(context_tokens: int, position: int) -> float:
    """Доля контекстных позиций среди позиций, доступных causal-вниманию.

    Сырая масса внимания на контекст несёт в себе артефакт длины: если контекст
    занимает 90 % последовательности, то даже при полностью равномерном
    внимании токен ответа «отдаст» контексту около 0,9. Сравнивать такие
    величины между документами разной длины и между токенами на разном
    удалении от начала ответа нельзя. Ожидаемая масса — это значение при
    равномерном внимании: сколько контекстных позиций лежит в causal-префиксе
    ``[0, position]``.
    """
    visible = position + 1
    if visible <= 0 or context_tokens <= 0:
        return 0.0
    return min(1.0, context_tokens / float(visible))


def normalised_context_mass(observed: float, expected: float, eps: float = 1e-9) -> float:
    """Observed/Expected: во сколько раз масса выше равномерной.

    1,0 — «как при равномерном внимании», > 1 — токен смотрит в контекст
    больше случайного, < 1 — смотрит в собственный префикс ответа. Значение
    не ограничено сверху: обрезка до [0, 1] съедала бы именно те сильные
    случаи, которые мы ищем.
    """
    if expected <= eps:
        return 0.0
    return float(observed) / float(expected)


def similarity_contrast(maximum: float, mean_value: float) -> float:
    """Выделенность максимума над средним уровнем похожести по контексту.

    Признак итерации 3. Сам по себе максимум похожести высок у «гладких»
    документов, где любой токен похож на любой: это свойство текста, а не
    признак опоры. Вычитание среднего уровня оставляет только превышение над
    фоном, то есть именно наличие источника, а не общую похожесть документа.
    """
    return float(maximum) - float(mean_value)


def similarity_margin(maximum: float, second: float) -> float:
    """Отрыв лучшего источника от второго по силе.

    Если максимум лишь немного выше второго, опора неопределённа: токен похож
    на много мест сразу, и «подтверждение» ничем не выделяется. Большой отрыв
    означает один явный источник.
    """
    return float(maximum) - float(second)


def normalised_support_distance(distance: int, seq_len: int) -> float:
    """Расстояние до опорного токена контекста, нормированное на длину текста.

    Признак итерации 2: важен не только сам максимум похожести, но и то, как
    далеко в документе лежит подтверждающий фрагмент. Нормировка на длину
    последовательности нужна, чтобы «200 токенов» в коротком и длинном акте
    не означали разного.
    """
    if seq_len <= 1:
        return 0.0
    return min(1.0, max(0.0, float(distance) / float(seq_len - 1)))


def distance_decay(similarity: float, distance: int, scale: float) -> float:
    """Похожесть со штрафом за удалённость опоры: ``sim · exp(−d/τ)``.

    Чем дальше подтверждающий фрагмент, тем меньше он годится как опора для
    конкретного утверждения. Множитель сохраняет знак и шкалу похожести, поэтому
    признак остаётся сопоставимым с плотностью.
    """
    if scale <= 0:
        return 0.0
    import math  # noqa: PLC0415

    return float(similarity) * math.exp(-abs(int(distance)) / float(scale))


def context_mass_lift(observed: float, expected: float) -> float:
    """Аддитивная разность «наблюдаемая − ожидаемая» масса в [-1, 1]."""
    return max(-1.0, min(1.0, float(observed) - float(expected)))


def scale_robust(values: Sequence[float], quantile: float = 1.0) -> list[float]:
    """Нормировка по квантили вместо максимума: устойчивость к одиночному пику.

    :func:`scale` делит на максимум, и этого достаточно, пока распределение без
    тяжёлых хвостов. Энтропия внимания реальной модели часто имеет один
    аномальный токен: деление на него сжимает риск остальных к нулю. Это
    гипотеза, а не измеренная причина прогона 37472951524 (там оценка
    подхватила чужую голову). Функция нужна, чтобы гипотезу можно было
    включить и измерить отдельно.

    ``quantile = 1.0`` — прежнее поведение (максимум), поэтому демонстрационный
    режим и все прежние числа не меняются.
    """
    if not values:
        return []
    if quantile >= 1.0:
        return scale(values)
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, int(round(quantile * (len(ordered) - 1)))))
    high = ordered[index]
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
    entropy_quantile: float = 1.0,
) -> list[float]:
    """Взвешенная комбинация признаков: r = w1·Ĥ + w2·(1 − m̂) + w3·d̂.

    Нормируется только ``attention_entropy``: у неё нет абсолютного нуля, важен
    лишь порядок токенов внутри ответа. ``ctx_attention_mass`` и
    ``embedding_density`` имеют абсолютный смысл (0 = опоры нет, 1 = токен
    подтверждён), поэтому они НЕ делятся на собственный максимум: иначе ответ,
    где слабо поддержаны все токены, выглядел бы полностью подтверждённым.

    ``entropy_quantile`` < 1 включает нормировку по квантили вместо максимума —
    это нужно режиму ``hf`` (см. :func:`scale_robust`); значение 1.0 сохраняет
    прежнее поведение.
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    total = sum(abs(weights.get(name, 0.0)) for name in FEATURE_NAMES) or 1.0
    e_hat = scale_robust(entropy, entropy_quantile)
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


def _chunk_support_profile(
    word: str,
    grams: frozenset[str],
    context_words: Sequence[tuple[str, frozenset[str]]],
    word_chunk: Sequence[int],
    chunk_total: int,
) -> tuple[float, float, float, int]:
    """Профиль похожести токена на чанки контекста: (макс, второй, средний, чанк-источник).

    Лексический аналог признаков итераций 2–3: в режиме ``hf`` те же величины
    считаются по скрытым состояниям (косинус эмбеддинга токена с эмбеддингами
    контекстных позиций, см. :func:`hf_features`). Похожесть на чанк — максимум
    по словам чанка: подтверждение может дать и одно слово. «Фон» — среднее по
    непустым чанкам, из него вычитается максимум (признак ``ctx_sim_contrast``),
    чтобы отсечь «гладкие» документы, где похожи все на всех.
    """
    stem = _stem(word)
    best_per_chunk: dict[int, float] = {}
    for (context_word, context_grams), chunk_index in zip(context_words, word_chunk, strict=False):
        if _stem(context_word) == stem:
            similarity = 1.0
        else:
            similarity = 0.7 * _jaccard(grams, context_grams)
        current = best_per_chunk.get(chunk_index, 0.0)
        if similarity > current:
            best_per_chunk[chunk_index] = similarity
    if not best_per_chunk:
        return 0.0, 0.0, 0.0, -1
    ordered = sorted(best_per_chunk.values(), reverse=True)
    maximum = ordered[0]
    second = ordered[1] if len(ordered) > 1 else 0.0
    # Средний уровень считаем по всем чанкам документа, а не только по тем, где
    # что-то нашлось: иначе «фон» зависел бы от числа совпадений и признак
    # превращался бы в повтор максимума.
    denominator = max(1, chunk_total)
    mean_value = sum(ordered) / denominator
    best_chunk = max(best_per_chunk.items(), key=lambda pair: (pair[1], -pair[0]))[0]
    return maximum, second, mean_value, best_chunk


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
    # Контекст нормализуется: иначе «25» в документе и «двадцать пять» в ответе
    # не совпадали, и корректный ответ получал ложное замечание.
    context_numbers = numbers_in(numbers_to_digits(chunks.text)) if chunks.text else set()
    context_words: list[tuple[str, frozenset[str]]] = [
        (token.word, _trigrams(token.word)) for token in context_tokens if _is_content(token)
    ]
    # Индекс чанка для каждого содержательного слова и число токенов в чанках —
    # нужно признакам итераций 2–3 (где именно в документе стоит опора).
    word_chunk: list[int] = []
    chunk_word_count: list[int] = []
    chunk_total = len(chunks.chunks)
    if chunk_total:
        chunk_word_count = [0] * chunk_total
        boundaries = [
            (start, start + len(chunk)) for start, chunk in zip(chunks.positions, chunks.chunks, strict=False)
        ]
        chunk_index = 0
        for token in context_tokens:
            if not _is_content(token):
                continue
            while chunk_index + 1 < len(boundaries) and token.start >= boundaries[chunk_index + 1][0]:
                chunk_index += 1
            while chunk_index > 0 and token.start < boundaries[chunk_index][0]:
                chunk_index -= 1
            word_chunk.append(chunk_index)
            chunk_word_count[chunk_index] += 1
        chunk_total = len(boundaries)
    # Начало каждого чанка в координатах «число содержательных слов контекста» —
    # по нему считается расстояние до подтверждающего фрагмента.
    chunk_start_tokens: list[int] = []
    running = 0
    for count in chunk_word_count:
        chunk_start_tokens.append(running)
        running += count

    # --- признаки ---
    phrase_hits = find_phrase_hits([token.text for token in tokens])
    seen: dict[str, int] = {}
    entropy: list[float] = []
    mass: list[float] = []
    density: list[float] = []
    # Диагностика итераций 2–3: заполняется рядом с рабочими признаками, но в
    # итоговый риск не входит — в голову её добавляет только отбор по AUC.
    max_similarity_values: list[float] = []
    support_distance_values: list[float] = []
    decay_values: list[float] = []
    contrast_values: list[float] = []
    margin_values: list[float] = []

    # Числа прописью занимают несколько токенов («двадцать» + «пять»), поэтому
    # значение считается для группы целиком, а не по отдельному токену.
    token_values = _group_number_values(tokens)
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
        value = token_values[index] if index < len(token_values) else None
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

        # Диагностика итераций 2–3: где в документе лежит опора и насколько она
        # выделена на фоне остальных чанков. Позиция токена ответа отсчитывается
        # после контекста — так же, как в режиме hf (там одна последовательность
        # «контекст + ответ», и расстояние считается по позициям модели).
        if grams and context_words:
            max_similarity, second_similarity, mean_similarity, best_chunk = _chunk_support_profile(
                token.word, grams, context_words, word_chunk, chunk_total
            )
        else:
            max_similarity, second_similarity, mean_similarity, best_chunk = 0.0, 0.0, 0.0, -1
        sequence_len = max(1, len(context_tokens) + len(tokens))
        if 0 <= best_chunk < len(chunk_start_tokens):
            center = chunk_start_tokens[best_chunk] + chunk_word_count[best_chunk] // 2
            support_gap = abs(len(context_tokens) + index - center)
        else:
            support_gap = sequence_len
        max_similarity_values.append(max_similarity)
        support_distance_values.append(normalised_support_distance(support_gap, sequence_len))
        decay_values.append(distance_decay(max_similarity, support_gap, max(1.0, 0.1 * sequence_len)))
        contrast_values.append(similarity_contrast(max_similarity, mean_similarity))
        margin_values.append(similarity_margin(max_similarity, second_similarity))

    _apply_number_consistency(tokens, context_numbers, mass)
    # Дефект D: число проверяется внутри своего объекта, а не «где-то в контексте».
    _apply_number_attribution(tokens, answer, chunks.text, mass)
    # Нормировка и «подъём» массы — по уже исправленной массе, иначе голова
    # получала бы признак, вычисленный до коррекции чисел.
    mean_mass = statistics.fmean(mass) if mass else 0.0
    mass_expected_values = [1.0] * len(mass)
    mass_norm_values = list(mass)
    mass_lift_values = [value - mean_mass for value in mass]

    return FeatureMatrix(
        attention_entropy=entropy,
        ctx_attention_mass=mass,
        embedding_density=density,
        ctx_mass_expected=mass_expected_values,
        ctx_mass_norm=mass_norm_values,
        ctx_mass_lift=mass_lift_values,
        ctx_max_similarity=max_similarity_values,
        ctx_support_distance=support_distance_values,
        ctx_similarity_decay=decay_values,
        ctx_sim_contrast=contrast_values,
        ctx_sim_margin=margin_values,
        meta={
            "backend": "demo",
            "token_count": len(tokens),
            "context_tokens": len(context_tokens),
            "context_numbers": sorted(context_numbers),
            "context_chunks": chunk_total,
            "k": k,
        },
    )


def _group_number_values(tokens: Sequence[Token]) -> list[str | None]:
    """Каноническое значение числа для каждого токена с учётом группы токенов.

    «двадцать пять» — два токена, но одно число. Раньше каждый токен
    сравнивался с контекстом отдельно, и корректный ответ с числом прописью
    получал нулевую опору: в документе «25», а в ответе токены «20» и «5».
    Отсюда ложные замечания на чистых парах и провал на кросс-корпусе, где
    генератор пишет числа словами.
    """
    values: list[str | None] = [None] * len(tokens)
    index = 0
    while index < len(tokens):
        if number_value(tokens[index]) is None:
            index += 1
            continue
        end = index
        group: list[str] = []
        while end < len(tokens) and number_value(tokens[end]) is not None:
            group.append(tokens[end].word.lower().replace("ё", "е"))
            end += 1
        if all(item.isdigit() for item in group):
            joined = "".join(group)
            value: str | None = str(int(joined)) if joined.isdigit() else None
        else:
            total = words_to_number(group)
            value = str(total) if total is not None else None
        for position in range(index, end):
            values[position] = value
        index = end
    return values


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


# Слова-«реквизиты документа»: число сразу после них — это номер документа
# («регламенту 669», «приказ 45»), а не измерение факта. Без этого правила номер
# документа становился вторым «измерением» и давал ложные срабатывания.
_DOCUMENT_NUMBER_STEMS = frozenset(
    {
        "регл",
        "прик",
        "пост",
        "расп",
        "зако",
        "указ",
        "инст",
        "пись",
        "коде",
        "номе",
        "форм",
        "блан",
        "пасп",
        "прот",
        "доку",
        "№",
    }
)

# Слова, которые не описывают объект измерения: связки, предлоги, глаголы-связки.
_SUBJECT_SKIP_WORDS = frozenset(
    {
        "и",
        "а",
        "но",
        "или",
        "не",
        "в",
        "во",
        "на",
        "с",
        "со",
        "по",
        "для",
        "от",
        "до",
        "из",
        "о",
        "об",
        "при",
        "за",
        "к",
        "ко",
        "у",
        "это",
        "был",
        "была",
        "было",
        "быть",
        "составляет",
        "составляют",
        "составляла",
        "составлял",
        "равен",
        "равна",
        "равно",
        "равняться",
        "установлен",
        "установлена",
        "установлено",
        "установлены",
        "определен",
        "определена",
        "определено",
        "должен",
        "должна",
        "должно",
        "может",
        "могут",
        "согласно",
        "то",
        "же",
        "также",
        "более",
        "менее",
        "только",
        "уже",
        "всего",
    }
)


@dataclass(frozen=True)
class Measurement:
    """Измерение в контексте: числовое значение и слова его субъекта."""

    value: str
    subject: tuple[str, ...]
    start: int
    end: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "subject": list(self.subject),
            "start": self.start,
            "end": self.end,
        }


@dataclass(frozen=True)
class NumberAttribution:
    """Решение по числу ответа: чьё это число и подтверждено ли оно своим объектом."""

    start: int
    end: int
    value: str
    subject: tuple[str, ...]
    matched_value: str | None
    matched_subject: tuple[str, ...]
    borrowed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "value": self.value,
            "subject": list(self.subject),
            "matched_value": self.matched_value,
            "matched_subject": list(self.matched_subject),
            "ok": not self.borrowed,
        }


def _subject_words(tokens: Sequence[Token], index: int) -> tuple[str, ...]:
    """Слова-субъект перед числом: берём окно назад до границы предложения."""
    window: list[str] = []
    position = index - 1
    while position >= 0 and len(window) < MEASUREMENT_SUBJECT_WINDOW:
        token = tokens[position]
        if token.text.strip() and token.text.strip()[-1] in ".!?…;:":
            break
        word = token.word.lower()
        if word and word not in _SUBJECT_SKIP_WORDS and not word.isdigit():
            window.append(word)
        position -= 1
    window.reverse()
    return tuple(window)


def _is_document_number(tokens: Sequence[Token], index: int) -> bool:
    """Число, стоящее **вплотную** к слову-реквизиту («регламенту 669») — номер документа.

    Важно именно соседство: в «срок хранения документов составляет пять лет» между
    словом «документов» и числом стоит глагол, и это настоящее измерение факта.
    """
    if index <= 0:
        return False
    word = tokens[index - 1].word.lower()
    return bool(word) and _stem(word) in _DOCUMENT_NUMBER_STEMS


def _context_text(context: str | Sequence[str] | None) -> str:
    """Привести контекст к строке: API и CLI принимают и список фрагментов."""
    if not context:
        return ""
    if isinstance(context, str):
        return context
    return split_chunks(context).text


def context_measurements(context: str | Sequence[str] | None) -> list[Measurement]:
    """Найти измерения контекста: числа вместе со словами их объектов.

    Пример: «срок хранения первичных документов составляет пять лет» → измерение
    со значением ``5`` и субъектом ``("срок", "хранения", "первичных", "документов")``.
    Нужны, чтобы отличать «это число есть в документе» от «это число стоит рядом
    со своим объектом» (дефект D: подмена числа на число из другого факта).
    """
    from .core import tokenize_with_offsets

    found: list[Measurement] = []
    tokens = tokenize_with_offsets(_context_text(context))
    for index, token in enumerate(tokens):
        value = number_value(token)
        if value is None:
            continue
        if _is_document_number(tokens, index):
            continue
        found.append(
            Measurement(
                value=value,
                subject=_subject_words(tokens, index),
                start=token.start,
                end=token.end,
            )
        )
    return found


def _stems(words: Sequence[str]) -> set[str]:
    """Основы слов для сравнения субъектов (переиспользует _stem)."""
    return {_stem(word) for word in words if word}


def _distinctive_measurements(measurements: Sequence[Measurement]) -> list[tuple[str, ...]]:
    """Субъекты без слов, общих для всех измерений (общие слова не различают объекты).

    Если после удаления общих слов не остаётся ничего, возвращаются полные субъекты.
    """
    if not measurements:
        return []
    stem_sets = [_stems(item.subject) for item in measurements]
    common = set.intersection(*stem_sets) if all(stem_sets) else set()
    distinctive: list[tuple[str, ...]] = []
    for item in measurements:
        filtered = tuple(word for word in item.subject if _stem(word) not in common)
        distinctive.append(filtered or item.subject)
    return distinctive


def _distinctive_match(answer_subject: Sequence[str], distinctive: Sequence[str]) -> float:
    """Доля различающих слов субъекта измерения, найденных в субъекте ответа.

    Считается именно вхождение (containment), а не Jaccard: различающих слов мало
    («первичных» против «вторичных»), и объединение множеств размывало бы сигнал.
    """
    distinctive_stems = _stems(distinctive)
    if not distinctive_stems:
        return 0.0
    answer_stems = _stems(answer_subject)
    return len(distinctive_stems & answer_stems) / len(distinctive_stems)


def number_attribution(answer: str, context: str | Sequence[str] | None) -> list[dict[str, Any]]:
    """Отчёт по каждому числу ответа: к какому объекту контекста оно относится.

    Возвращает список словарей (см. :class:`NumberAttribution`). Пустой список
    означает, что привязывать нечего: в ответе нет чисел или в контексте меньше
    двух измерений.
    """
    from .core import tokenize_with_offsets

    measurements = context_measurements(context)
    tokens = tokenize_with_offsets(answer)
    return [item.as_dict() for item in _attribute_tokens(tokens, measurements)]


def _attribute_tokens(tokens: Sequence[Token], measurements: Sequence[Measurement]) -> list[NumberAttribution]:
    """Для каждого числа ответа найти ближайший по смыслу субъект контекста."""
    if len(measurements) < 2:
        return []
    distinctive = _distinctive_measurements(measurements)
    results: list[NumberAttribution] = []
    # Значение берётся для группы токенов: «двадцать пять» — одно число, а не
    # «20» и «5». Иначе число прописью не совпадало бы с «25» в документе и
    # обвинялось бы в заимствовании у чужого объекта — отсюда ложные
    # замечания на совершенно корректных ответах (исправление B4).
    group_values = _group_number_values(tokens)
    for index, token in enumerate(tokens):
        value = group_values[index] if index < len(group_values) else None
        if value is None:
            continue
        subject = _subject_words(tokens, index)
        if not subject:
            # Слов-субъекта нет — привязывать не к чему, вслепую не обвиняем.
            continue
        best_index, best_score = -1, 0.0
        for position, _item in enumerate(measurements):
            score = _distinctive_match(subject, distinctive[position])
            if score > best_score:
                best_index, best_score = position, score
        if best_index < 0 or best_score < MEASUREMENT_MATCH_MIN:
            continue
        matched = measurements[best_index]
        if matched.value == value:
            continue
        borrowed = any(item.value == value for position, item in enumerate(measurements) if position != best_index)
        if not borrowed:
            continue
        results.append(
            NumberAttribution(
                start=token.start,
                end=token.end,
                value=value,
                subject=subject,
                matched_value=matched.value,
                matched_subject=matched.subject,
                borrowed=True,
            )
        )
    return results


def _apply_number_attribution(
    tokens: Sequence[Token], answer: str, context: str | Sequence[str] | None, mass: list[float]
) -> None:
    """Число, взятое из другого объекта документа, не может считаться подтверждённым.

    Это закрытие дефекта D: раньше число получало опору уже за то, что встречается
    где-то в контексте («пять лет» у первичных и «десять лет» у вторичных — ответ
    про первичные со словом «десять» считался подтверждённым).
    """
    del answer  # токены ответа уже содержат смещения; текст нужен только вызывающему
    if not context:
        return
    measurements = context_measurements(context)
    if len(measurements) < 2:
        return
    for item in _attribute_tokens(tokens, measurements):
        for index, token in enumerate(tokens):
            if token.start == item.start and token.end == item.end:
                mass[index] = 0.0


def _apply_number_consistency(tokens: Sequence[Token], context_numbers: set[str], mass: list[float]) -> None:
    """Число, противоречащее контексту, не может считаться поддержанным.

    Если в контексте есть числовые значения, но конкретное число ответа там
    отсутствует — это признак подмены факта (сильнейший сигнал в демо-режиме).
    """
    if not context_numbers:
        return
    # Значение берётся для группы токенов, а не по одному токену: иначе
    # «двадцать пять» распадается на «20» и «5», которых в документе нет,
    # и корректный ответ с числом прописью получает нулевую опору.
    # Тот же дефект, что закрывался в _attribute_tokens (B4), но в этой
    # функции он оставался незамеченным.
    group_values = _group_number_values(tokens)
    for index, token in enumerate(tokens):
        value = group_values[index] if index < len(group_values) else None
        if value is None:
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


def _span_overlap(left: tuple[int, int], right: tuple[int, int]) -> int:
    """Число общих символов у двух отрезков (0, если отрезки не пересекаются)."""
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def map_token_positions(
    token_spans: Sequence[tuple[int, int]],
    model_spans: Sequence[tuple[int, int, int]],
    answer_start: int,
) -> list[list[int]]:
    """Сопоставить токены нашего разбиения с позициями модели по символам.

    ``token_spans`` — отрезки в координатах ответа, ``model_spans`` — тройки
    ``(позиция, начало, конец)`` в координатах общего промпта. Возвращается
    список позиций модели для каждого нашего токена (пустой список, если
    пересечения нет). Разбиение модели на подслова не совпадает с нашим, и
    сопоставление по индексу сдвигало признаки на соседние слова; здесь каждый
    наш токен получает все позиции модели, которые с ним существенно
    перекрываются, — признаки такого токена усредняются по его подсловам.
    """
    mapping: list[list[int]] = []
    for start, end in token_spans:
        absolute = (start + answer_start, end + answer_start)
        hits = [
            (position, _span_overlap(absolute, (span_start, span_end)))
            for position, span_start, span_end in model_spans
            if _span_overlap(absolute, (span_start, span_end)) > 0
        ]
        if not hits:
            mapping.append([])
            continue
        best = max(overlap for _position, overlap in hits)
        threshold = max(1, best // 2)
        mapping.append(sorted(position for position, overlap in hits if overlap >= threshold))
    return mapping


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
    from .backends.base import BackendUnavailable  # noqa: PLC0415

    # Зависимости проверяются до импорта torch: иначе при их отсутствии вылетал
    # бы голый ModuleNotFoundError, и пользователь не понимал бы, что делать.
    # Тихая подмена hf на demo здесь запрещена: если запрошен реальный режим,
    # ответ должен быть либо реальным, либо явной ошибкой (дефект A3 реестра).
    from .backends.hf import HFBackend  # noqa: PLC0415
    from .core import tokenize_with_offsets  # noqa: PLC0415

    ok, reason = HFBackend.dependencies()
    if not ok:
        raise BackendUnavailable(reason)

    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

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
    # Скрытые состояния нужны третьему признаку (плотность представлений), а
    # карты внимания — первым двум. Флаги ставим в конфиге: так одинаково
    # работают и старые, и новые версии transformers.
    model.config.output_attentions = True
    model.config.output_hidden_states = True
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
    # Разные версии transformers отдают смещения то как тензор/`BatchEncoding`
    # (с методом ``tolist``), то как обычный список списков — приводим к списку
    # кортежей, иначе на свежих версиях падало бы AttributeError.
    raw_offsets = tokenizer(prompt, return_offsets_mapping=True, truncation=True, max_length=max_length)[
        "offset_mapping"
    ]
    if hasattr(raw_offsets, "tolist"):
        raw_offsets = raw_offsets.tolist()
    offsets = list(raw_offsets)
    if offsets and isinstance(offsets[0], (list, tuple)) and offsets[0] and isinstance(offsets[0][0], (list, tuple)):
        offsets = list(offsets[0])
    offsets = [(int(start), int(end)) for start, end in offsets]
    seq_len = int(attention_mask.sum().item())
    model_spans = [
        (i, start, end)
        for i, (start, end) in enumerate(offsets)
        if end > start and start >= answer_start and i < seq_len
    ]
    answer_positions = [position for position, _start, _end in model_spans]
    answer_start_token = min(answer_positions) if answer_positions else seq_len
    context_positions = [i for i in range(seq_len) if i < answer_start_token]
    token_positions = map_token_positions(
        [(token.start, token.end) for token in tokens],
        model_spans,
        answer_start,
    )

    eps = 1e-9
    probs = attentions.clamp_min(eps)
    entropy = -(probs * probs.log()).sum(dim=-1).mean(dim=0)  # (seq,)
    import math  # noqa: PLC0415

    entropy = entropy / max(eps, math.log(max(2, seq_len)))

    entropy_values: list[float] = []
    mass_values: list[float] = []
    density_values: list[float] = []
    mass_expected_values: list[float] = []
    mass_norm_values: list[float] = []
    mass_lift_values: list[float] = []
    max_sim_values: list[float] = []
    support_distance_values: list[float] = []
    decay_values: list[float] = []
    contrast_values: list[float] = []
    margin_values: list[float] = []

    # Векторы контекстных чанков и токенов — для плотности.
    context_token_embeddings: list[list[float]] = []
    hidden_cpu = None
    if hidden is not None:
        hidden_cpu = hidden.detach().to("cpu")
        for position in context_positions:
            context_token_embeddings.append(hidden_cpu[position].tolist())

    for positions in token_positions:
        if not positions:
            entropy_values.append(0.5)
            mass_values.append(0.0)
            density_values.append(0.0)
            mass_expected_values.append(0.0)
            mass_norm_values.append(0.0)
            mass_lift_values.append(0.0)
            max_sim_values.append(0.0)
            support_distance_values.append(0.0)
            decay_values.append(0.0)
            contrast_values.append(0.0)
            margin_values.append(0.0)
            continue
        entropy_values.append(statistics.fmean(float(entropy[position].item()) for position in positions))
        mass_parts: list[float] = []
        density_parts: list[float] = []
        norm_parts: list[float] = []
        lift_parts: list[float] = []
        max_sim_parts: list[float] = []
        distance_parts: list[float] = []
        decay_parts: list[float] = []
        contrast_parts: list[float] = []
        margin_parts: list[float] = []
        for position in positions:
            row = attentions[:, position, :].mean(dim=0)  # (seq,)
            observed = float(row[context_positions].sum().item()) if context_positions else 0.0
            mass_parts.append(observed)
            # Ожидаемая масса — сколько контекстных позиций доступно на этой
            # позиции при равномерном внимании. Без этого сравнения длинные
            # контексты кажутся «более поддержанными» просто за счёт размера.
            expected = expected_context_mass(len(context_positions), position)
            norm_parts.append(normalised_context_mass(observed, expected))
            lift_parts.append(context_mass_lift(observed, expected))
            if hidden_cpu is not None and context_token_embeddings:
                vector = hidden_cpu[position].tolist()
                # Считаем похожесть вместе с позицией источника: для признаков
                # итерации 2 важно не только значение максимума, но и где именно
                # в документе лежит подтверждающий фрагмент.
                scored = [
                    (_cosine_dense(vector, other), source_position)
                    for other, source_position in zip(context_token_embeddings, context_positions, strict=False)
                ]
                best_similarity, best_position = max(scored, key=lambda pair: pair[0])
                distance = int(abs(position - best_position))
                # Масштаб затухания — 10 % длины: устойчив к размеру документа.
                decay_scale = max(1.0, 0.1 * seq_len)
                max_sim_parts.append(float(best_similarity))
                distance_parts.append(normalised_support_distance(distance, int(seq_len)))
                decay_parts.append(distance_decay(best_similarity, distance, decay_scale))
                similarities = sorted((value for value, _position in scored), reverse=True)
                top = similarities[: max(1, k)]
                density_parts.append(sum(top) / len(top))
                # Фон и второй источник — для признаков итерации 3: один и тот же
                # максимум означает разное в «гладком» и в «рваном» документе.
                mean_similarity = statistics.fmean(similarities) if similarities else 0.0
                second_similarity = (
                    similarities[1] if len(similarities) > 1 else (similarities[0] if similarities else 0.0)
                )
                contrast_parts.append(similarity_contrast(best_similarity, mean_similarity))
                margin_parts.append(similarity_margin(best_similarity, second_similarity))
            else:  # pragma: no cover - модель без hidden_states
                density_parts.append(0.0)
                max_sim_parts.append(0.0)
                distance_parts.append(0.0)
                decay_parts.append(0.0)
                contrast_parts.append(0.0)
                margin_parts.append(0.0)
        mass_values.append(min(1.0, max(0.0, statistics.fmean(mass_parts))))
        density_values.append(min(1.0, max(0.0, statistics.fmean(density_parts))))
        # Для токена, разбитого на несколько сабтокенов, ожидаемая масса
        # считается по первой позиции: остальные имеют тот же префикс плюс
        # свои собственные, уже принадлежащие ответу.
        first_position = min(positions)
        expected_token = expected_context_mass(len(context_positions), first_position)
        mass_expected_values.append(expected_token)
        mass_norm_values.append(statistics.fmean(norm_parts) if norm_parts else 0.0)
        mass_lift_values.append(statistics.fmean(lift_parts) if lift_parts else 0.0)
        max_sim_values.append(statistics.fmean(max_sim_parts) if max_sim_parts else 0.0)
        support_distance_values.append(statistics.fmean(distance_parts) if distance_parts else 0.0)
        decay_values.append(statistics.fmean(decay_parts) if decay_parts else 0.0)
        contrast_values.append(statistics.fmean(contrast_parts) if contrast_parts else 0.0)
        margin_values.append(statistics.fmean(margin_parts) if margin_parts else 0.0)

    return FeatureMatrix(
        attention_entropy=entropy_values,
        ctx_attention_mass=mass_values,
        embedding_density=density_values,
        ctx_mass_expected=mass_expected_values,
        ctx_mass_norm=mass_norm_values,
        ctx_mass_lift=mass_lift_values,
        ctx_max_similarity=max_sim_values,
        ctx_support_distance=support_distance_values,
        ctx_similarity_decay=decay_values,
        ctx_sim_contrast=contrast_values,
        ctx_sim_margin=margin_values,
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


# Реестр активного кеша. Нужен потому, что признаки считаются не только из
# Verifier, но и из обучения (train()), которое создаёт верификатор внутри себя:
# явный параметр до туда не дошёл бы. Реестр — единственный способ покрыть все
# пути без правки каждого вызывающего места. Тесты обязаны его очищать.
_ACTIVE_FEATURE_CACHE: dict[str, FeatureMatrix] | None = None


def set_feature_cache(cache: Mapping[str, FeatureMatrix] | None) -> None:
    """Включить (или выключить при ``None``) глобальный кеш признаков."""
    global _ACTIVE_FEATURE_CACHE  # noqa: PLW0603
    if cache is None:
        _ACTIVE_FEATURE_CACHE = None
    elif isinstance(cache, (CataloguedFeatureCache, CountingFeatureCache)):
        # Копировать нельзя: вместе с dict() потерялся бы реестр модели, и
        # проверка «кешь принадлежит этой модели» перестала бы что-либо видеть.
        _ACTIVE_FEATURE_CACHE = cache
    else:
        _ACTIVE_FEATURE_CACHE = dict(cache)


def feature_cache_models(cache: Mapping[str, FeatureMatrix] | None) -> set[str]:
    """Модели, которым принадлежат строки кеша (пустое множество — реестра нет)."""
    info = getattr(cache, "info", None) if cache is not None else None
    if not info:
        return set()
    return {str(model) for model in info.get("models", ())}


def feature_cache_fields(cache: Mapping[str, FeatureMatrix] | None) -> set[str]:
    """Признаки, реально сохранённые в строках кеша."""
    info = getattr(cache, "info", None) if cache is not None else None
    if not info:
        return set()
    return {str(name) for name in info.get("fields", ())}


def validate_feature_cache(
    cache: Mapping[str, FeatureMatrix] | None,
    mode: str,
    model_name: str = "",
    head_features: Sequence[str] = (),
    sample_size: int = 32,
) -> dict[str, Any]:
    """Явная проверка: годится ли этот кеш для запрошенного режима и модели.

    Промах по ключу опасен не ошибкой, а тишиной: признаки молча считаются
    заново другой моделью (часы вместо секунд) либо, что хуже, голова получает
    нули вместо признаков, которых в кеш не писали. Поэтому несоответствие
    превращается здесь в :class:`BackendUnavailable` (текст содержит
    ``model mismatch``) или :class:`ValueError` — с командами на исправление.

    Проверка стоит один раз на кеш (результат отмечается на объекте кеша), а не
    на каждую пару: в кросс-валидации ``extract_features`` вызывают тысячи раз.
    """
    from .backends.base import BackendUnavailable  # noqa: PLC0415 - ленивый импорт, как в hf_features

    if cache is None or not len(cache):
        return {"checked": False, "reason": "кеш пуст или выключен"}

    marker = (mode, model_name, tuple(head_features))
    validated = getattr(cache, "validated", None)
    if validated is not None and marker in validated:
        return {"checked": True, "reason": "уже проверено", "models": feature_cache_models(cache)}

    models = feature_cache_models(cache)
    fields = feature_cache_fields(cache)
    if mode == "hf" and models and model_name and model_name not in models:
        raise BackendUnavailable(
            f"HF cache model mismatch: запрошена модель {model_name!r}, "
            f"а кеш посчитан для {', '.join(sorted(models))!r}. "
            "Очистите кеш или передайте то же --model, что использовалось при "
            "предпосчёте (scripts/precompute_features.py). Ключей в кеше: "
            f"{len(cache)}. "
            f"Проверка выборки: {_cache_key_sample(cache, sample_size)}"
        )
    if mode == "hf" and not models and model_name:
        # Кеш формата v1: имя модели в строках не сохранялось. Сам по себе он
        # не противоречив, но проверить принадлежность модели нельзя — значит,
        # признаки может считать и другая модель. Говорим об этом прямо.
        raise BackendUnavailable(
            f"HF cache model mismatch: кеш не содержит имени модели (формат v1), "
            f"а запрошена {model_name!r}. Пересоберите кеш: "
            "python scripts/precompute_features.py --dataset <корпус> --mode hf "
            f"--model {model_name} --out <каталог>. Ключей в кеше: {len(cache)}. "
            "Признаки режима hf из кеша v1 доступны только в объёме трёх рабочих "
            "признаков — этого недостаточно для головы с диагностикой."
        )
    missing = [name for name in head_features if name not in HEAD_FEATURE_DERIVED and name not in fields]
    if missing:
        raise ValueError(
            "HF cache mismatch: в кеше нет строк признаков "
            + ", ".join(sorted(missing))
            + ". Предпосчёт писал кеш старого формата (только три рабочих "
            "признака). Пересоберите кеш этой же версией скрипта: "
            "python scripts/precompute_features.py --dataset <корпус> --mode hf "
            f"--model {model_name or '<модель>'} --out <каталог>. "
            f"Полей в кеше: {len(fields) or 3}."
        )
    if validated is not None:
        validated.add(marker)
    return {
        "checked": True,
        "models": sorted(models),
        "fields": sorted(fields),
        "rows": len(cache),
    }


def _cache_key_sample(cache: Mapping[str, FeatureMatrix], limit: int = 32) -> str:
    """Несколько ключей кеша для диагностики (префиксы, не содержимое)."""
    try:
        keys = [str(key) for key in list(iter(cache))[:limit]]
    except TypeError:  # pragma: no cover - некартиноподобный кеш
        return "ключи недоступны"
    return "префиксы ключей: " + ", ".join(key[:12] for key in keys[:4]) + "…"


def get_feature_cache() -> Mapping[str, FeatureMatrix] | None:
    """Текущий активный кеш признаков (``None`` — кеш выключен)."""
    return _ACTIVE_FEATURE_CACHE


class CountingFeatureCache(dict):
    """Кеш признаков со счётчиком попаданий и промахов.

    Промах опасен не ошибкой, а тишиной: если ключ не совпал, признаки молча
    считаются моделью заново, эксперимент снова идёт часами, а в отчёте этого
    не видно. Счётчик делает расхождение ключей видимым в логе job'а.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.hits = 0
        self.misses = 0
        # Реестр происхождения строк — см. CataloguedFeatureCache: считающий кеш
        # тоже обязан проходить проверку модели, а не только считать попадания.
        self.info: dict[str, Any] = {
            "models": set(),
            "modes": set(),
            "formats": set(),
            "fields": set(),
            "rows": 0,
        }
        self.validated: set[tuple[str, str]] = set()

    def get(self, key: str, default: Any = None) -> Any:
        if dict.__contains__(self, key):
            self.hits += 1
            return dict.__getitem__(self, key)
        self.misses += 1
        return default

    def __repr__(self) -> str:  # pragma: no cover - отладочное
        return f"<CountingFeatureCache {len(self)} ключей, попаданий {self.hits}, промахов {self.misses}>"


def load_feature_cache(paths: Any, counting: bool = False) -> dict[str, FeatureMatrix]:
    """Прочитать кеш из файла(.ов) JSONL или из каталога с шардами.

    Шарды пишутся независимыми job'ами в ``features_*.jsonl``; здесь они
    склеиваются в один словарь. Порядок не важен: обращение по ключу.

    Формат строки кеша — v2: кроме трёх рабочих признаков сохраняются все
    диагностические массивы и служебные поля (``model``, ``mode``,
    ``cache_format``, ``layer``). Поля v1 (только три признака) читаются как
    раньше: устаревший кеш не роняет прогон, но не даёт диагностики — и это
    проверяется явно (см. :func:`validate_feature_cache`), а не тихо.
    """
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    if isinstance(paths, (str, Path)):
        paths = [paths]
    cache: dict[str, FeatureMatrix] = CountingFeatureCache() if counting else CataloguedFeatureCache()
    for raw in paths or ():
        path = Path(raw)
        if path.is_dir():
            # rglob: gh run download кладёт каждый артефакт в свой подкаталог.
            files = sorted(path.rglob("features_*.jsonl"))
        else:
            files = [path]
        for file in files:
            if not file.is_file():
                continue
            for line in file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = row.get("key")
                if not key:
                    continue
                arrays: dict[str, list[float]] = {}
                for name, attribute in CACHE_ARRAY_FIELDS.items():
                    values = row.get(name)
                    if values:
                        arrays[attribute] = [float(value) for value in values]
                cache[key] = FeatureMatrix(
                    **arrays,
                    meta={
                        "from_cache": True,
                        "pair_id": row.get("pair_id"),
                        "model": row.get("model"),
                        "mode": row.get("mode"),
                        "cache_format": row.get("cache_format"),
                        "layer": row.get("layer"),
                    },
                )
                info = getattr(cache, "info", None)
                if info is not None:
                    if row.get("model"):
                        info["models"].add(str(row["model"]))
                    if row.get("mode"):
                        info["modes"].add(str(row["mode"]))
                    if row.get("cache_format") is not None:
                        info["formats"].add(str(row["cache_format"]))
                    info["rows"] += 1
                    for name in row:
                        if name in CACHE_ARRAY_FIELDS:
                            info["fields"].add(name)
    return cache


# Поле строки кеша -> атрибут FeatureMatrix. Единый источник для записи
# (scripts/precompute_features.py) и чтения: расхождение здесь проявилось бы как
# «признак внезапно равен нулю», то есть как испорченный результат, а не ошибка.
CACHE_ARRAY_FIELDS: dict[str, str] = {
    "attention_entropy": "attention_entropy",
    "ctx_attention_mass": "ctx_attention_mass",
    "embedding_density": "embedding_density",
    "ctx_mass_expected": "ctx_mass_expected",
    "ctx_mass_norm": "ctx_mass_norm",
    "ctx_mass_lift": "ctx_mass_lift",
    "ctx_max_similarity": "ctx_max_similarity",
    "ctx_support_distance": "ctx_support_distance",
    "ctx_similarity_decay": "ctx_similarity_decay",
    "ctx_sim_contrast": "ctx_sim_contrast",
    "ctx_sim_margin": "ctx_sim_margin",
}


class CataloguedFeatureCache(dict):
    """Кеш с реестром происхождения строк (модель, режим, формат, поля).

    Нужен, чтобы несовпадение кеша и запроса превращалось в ошибку при старте,
    а не в тихий пересчёт модели посреди кросс-валидации или в нулевые признаки
    у обученной головы.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.info: dict[str, Any] = {
            "models": set(),
            "modes": set(),
            "formats": set(),
            "fields": set(),
            "rows": 0,
        }
        self.validated: set[tuple[str, str]] = set()


# Модель, на которой считаются признаки режима hf в экспериментах и в кеше.
# Должна совпадать у предпосчёта и у оценки: имя входит в ключ кеша. Расхождение
# (в конфиге по умолчанию стоит rubert-tiny2, а кеш писался для rugpt3small)
# давало тихий промах: оценка либо пересчитывала признаки другой моделью, либо
# падала, если torch не установлен.
HF_MODEL_DEFAULT = "ai-forever/rugpt3small_based_on_gpt2"


def feature_cache_key(
    answer: str,
    context: str | Sequence[str] | None,
    mode: str,
    model_name: str = "",
) -> str:
    """Ключ кеша признаков: ответ + контекст + режим + модель.

    Счёт признаков на реальной модели занимает минуты на пару, и в
    кросс-валидации каждая пара встречается в нескольких фолдах. Кеш делает
    дорогой проход однократным. Ключ включает модель и режим: признаки разных
    моделей несопоставимы, и подмена одного другим исказила бы результат.
    """
    import hashlib  # noqa: PLC0415

    if isinstance(context, str):
        context_text = context
    elif context:
        context_text = "\n".join(str(item) for item in context)
    else:
        context_text = ""
    payload = f"{mode}|{model_name}|{answer}|{context_text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def extract_features(
    answer: str,
    context: str | Sequence[str] | None,
    mode: str = "demo",
    cache: Mapping[str, FeatureMatrix] | None = None,
    cache_key: str | None = None,
    **kwargs: Any,
) -> FeatureMatrix:
    """Единая точка входа: признаки заказанного режима, при `cache` — из кеша.

    Кеш — единственный способ уложить эксперимент на реальной модели в разумное
    время: `hf_features` делает прямой проход по модели, а в кросс-валидации
    признаки каждой пары требуются в каждом фолде.
    """
    if cache is None:
        cache = _ACTIVE_FEATURE_CACHE
    if cache is not None:
        if getattr(cache, "info", None):
            # Кеш с реестром: сначала явная проверка модели, потом чтение. Так
            # расхождение «кеш посчитан другой моделью» видно в момент первого
            # обращения, а не по испорченным метрикам в конце прогона.
            validate_feature_cache(cache, mode, str(kwargs.get("model_name", "")))
        key = cache_key or feature_cache_key(answer, context, mode, str(kwargs.get("model_name", "")))
        cached = cache.get(key)
        if cached is not None:
            return cached
    if mode == "hf":
        return hf_features(answer, context, **kwargs)
    return demo_features(answer, context, **kwargs)
