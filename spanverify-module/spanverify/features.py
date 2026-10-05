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

from .core import Token, number_value, numbers_in, split_chunks, tokenize_with_offsets
from .lexicon import BOILERPLATE_WORDS, PARAPHRASE_WORDS, STOPWORDS, find_phrase_hits
from .vectors import hash_index, normalize

FEATURE_NAMES = ("attention_entropy", "ctx_attention_mass", "embedding_density")
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
    token_risk: list[float] = field(default_factory=list)
    # Признаки доработки (пункт 2.2): нормировка массы, позиция/расстояние до
    # опоры, лексико-семантические сигналы. Не входят в свёртку risk по умолчанию —
    # их вклад измеряется отдельно (scripts/feature_iterations.py), чтобы не
    # менять порог и веса без числа.
    extra: dict[str, list[float]] = field(default_factory=dict)
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
    # Дефект D: число проверяется внутри своего объекта, а не «где-то в контексте».
    _apply_number_attribution(tokens, answer, chunks.text, mass)
    from .feature_pack import extra_features  # noqa: PLC0415 - локальный импорт без цикла

    extra = extra_features(
        answer,
        chunks.text,
        tokens,
        mass=mass,
        context_tokens=len(context_tokens),
    )

    return FeatureMatrix(
        attention_entropy=entropy,
        ctx_attention_mass=mass,
        embedding_density=density,
        extra=extra,
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
    for index, token in enumerate(tokens):
        value = number_value(token)
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
    max_length: int = 512,
    answer_tokens: Sequence[Token] | None = None,
    layer: int | str = "last",
    cache_dir: str | None = None,
    feature_cache: str | None = None,
) -> FeatureMatrix:
    """Реальные признаки модели: энтропия внимания, масса на контекст, плотность.

    Работа выполняется модулем :mod:`spanverify.hfmodel`: он кэширует веса,
    поддерживает причинные модели и энкодеры, режет длинный текст скользящим
    окном и уменьшает окно при нехватке памяти. Здесь результат лишь
    приводится к общей матрице признаков.

    ``layer`` выбирает слой внимания: ``"first"``, ``"middle"``, ``"last"``,
    отрицательный индекс (``-4`` — четвёртый с конца) или целое число. Это нужно
    пилоту: он сравнивает информативность признаков по слоям.

    ``feature_cache`` — каталог кэша посчитанных признаков (повторный прогон по
    тем же текстам не пересчитывает модель). ``cache_dir`` — каталог кэша весов.
    """
    from .feature_pack import extra_features  # noqa: PLC0415 - локальный импорт без цикла
    from .hfmodel import model_features  # noqa: PLC0415 - torch импортируется лениво

    tokens = list(answer_tokens) if answer_tokens is not None else tokenize_with_offsets(answer)
    if not tokens:
        return FeatureMatrix(meta={"backend": "hf", "empty": True})
    result = model_features(
        answer,
        context,
        model_name,
        k=k,
        device=device,
        max_tokens=max_length,
        layer=layer,
        cache_dir=cache_dir,
        feature_cache=feature_cache,
        answer_tokens=tokens,
    )
    context_text = split_chunks(context).text
    extra = extra_features(
        answer,
        context_text,
        tokens,
        mass=result.mass,
        context_tokens=int((result.meta or {}).get("context_tokens", 0) or 0),
    )
    return FeatureMatrix(
        attention_entropy=list(result.entropy),
        ctx_attention_mass=list(result.mass),
        embedding_density=list(result.density),
        extra=extra,
        meta=dict(result.meta or {}),
    )


def resolve_layer(layer: int | str, total: int) -> int:
    """Индекс слоя внимания по имени или числу (общий с модулем работы с моделью).

    Поддерживаются ``"first"``, ``"middle"``, ``"last"``, отрицательные индексы
    (``-4`` — четвёртый с конца) и обычные целые числа.
    """
    from .hfmodel import resolve_layer as _resolve  # noqa: PLC0415

    return _resolve(layer, total)


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
