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
from collections.abc import Sequence
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
)
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
    # Контекст нормализуется: иначе «25» в документе и «двадцать пять» в ответе
    # не совпадали, и корректный ответ получал ложное замечание.
    context_numbers = numbers_in(numbers_to_digits(chunks.text)) if chunks.text else set()
    context_words: list[tuple[str, frozenset[str]]] = [
        (token.word, _trigrams(token.word)) for token in context_tokens if _is_content(token)
    ]

    # --- признаки ---
    phrase_hits = find_phrase_hits([token.text for token in tokens])
    seen: dict[str, int] = {}
    entropy: list[float] = []
    mass: list[float] = []
    density: list[float] = []

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

    _apply_number_consistency(tokens, context_numbers, mass)
    # Дефект D: число проверяется внутри своего объекта, а не «где-то в контексте».
    _apply_number_attribution(tokens, answer, chunks.text, mass)

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
            continue
        entropy_values.append(statistics.fmean(float(entropy[position].item()) for position in positions))
        mass_parts: list[float] = []
        density_parts: list[float] = []
        norm_parts: list[float] = []
        lift_parts: list[float] = []
        max_sim_parts: list[float] = []
        distance_parts: list[float] = []
        decay_parts: list[float] = []
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
            else:  # pragma: no cover - модель без hidden_states
                density_parts.append(0.0)
                max_sim_parts.append(0.0)
                distance_parts.append(0.0)
                decay_parts.append(0.0)
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
