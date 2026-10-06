"""Нормализация текста, чисел и единиц измерения (исправление C3).

Зачем модуль: признаки продукта требовали почти дословного совпадения ответа с
документом, поэтому метод рассыпался на перефразировании и на числах прописью
(падение 0,953 → 0,717 на другом генераторе и 0,10–0,20 на внешних наборах).
Здесь собраны преобразования, которые приводят текст к виду, где сравнивать можно
по смыслу, а не по символам:

* числа прописью → цифры («пять лет» → «5 лет», «двадцать пять» → «25»);
* единицы измерения → каноническая форма («лет», «года», «год» → «год»);
* даты → канонический вид («24 июня 2025 г.» → «24.06.2025»);
* лемматизация грубым стеммером: усечение русских окончаний, чтобы
  «хранения» и «хранение» совпадали.

Стеммер намеренно грубый и без словарей: продукт обязан работать без сторонних
зависимостей и без загрузки моделей. Он нужен только для сопоставления, а не для
формирования ответа человеку.
"""

from __future__ import annotations

import re

__all__ = [
    "words_to_number",
    "numbers_to_digits",
    "normalize_numbers",
    "canonical_units",
    "normalize_dates",
    "stem_word",
    "stem_text",
    "normalize_for_match",
    "VALUES_EQUAL",
]


# Единицы: каждая форма → канон. Порядок важен: длинные формы раньше коротких.
_UNITS: dict[str, str] = {
    "года": "год",
    "годов": "год",
    "году": "год",
    "годом": "год",
    "годе": "год",
    "лет": "год",
    "гг": "год",
    "г.": "год",
    "месяца": "месяц",
    "месяцев": "месяц",
    "месяцу": "месяц",
    "месяцем": "месяц",
    "месяце": "месяц",
    "мес": "месяц",
    "недели": "неделя",
    "неделю": "неделя",
    "недель": "неделя",
    "неделя": "неделя",
    "дней": "день",
    "дня": "день",
    "дню": "день",
    "днём": "день",
    "дне": "день",
    "суток": "сутки",
    "часов": "час",
    "часа": "час",
    "часу": "час",
    "часом": "час",
    "часе": "час",
    "минут": "минута",
    "минуты": "минута",
    "минуту": "минута",
    "процента": "процент",
    "процентов": "процент",
    "проценту": "процент",
    "процентом": "процент",
    "%": "процент",
    "рублей": "рубль",
    "рубля": "рубль",
    "рубле": "рубль",
    "рублю": "рубль",
    "руб.": "рубль",
    "тысяч": "тысяча",
    "тысячи": "тысяча",
    "тыс.": "тысяча",
    "экземпляра": "экземпляр",
    "экземпляров": "экземпляр",
    "экземпляре": "экземпляр",
    "экземпляр": "экземпляр",
}

_UNIT_WORDS = sorted(_UNITS, key=len, reverse=True)

_NUMBER_WORDS: dict[str, int] = {
    "ноль": 0,
    "один": 1,
    "одна": 1,
    "одно": 1,
    "одного": 1,
    "одной": 1,
    "одному": 1,
    "два": 2,
    "две": 2,
    "двух": 2,
    "двумя": 2,
    "три": 3,
    "трёх": 3,
    "тремя": 3,
    "четыре": 4,
    "четырёх": 4,
    "пять": 5,
    "пяти": 5,
    "шесть": 6,
    "шести": 6,
    "семь": 7,
    "семи": 7,
    "восемь": 8,
    "восьми": 8,
    "девять": 9,
    "девяти": 9,
    "десять": 10,
    "десяти": 10,
    "одиннадцать": 11,
    "двенадцать": 12,
    "тринадцать": 13,
    "четырнадцать": 14,
    "пятнадцать": 15,
    "шестнадцать": 16,
    "семнадцать": 17,
    "восемнадцать": 18,
    "девятнадцать": 19,
    "двадцать": 20,
    "тридцать": 30,
    "сорок": 40,
    "пятьдесят": 50,
    "шестьдесят": 60,
    "семьдесят": 70,
    "восемьдесят": 80,
    "девяносто": 90,
    "сто": 100,
    "двести": 200,
    "триста": 300,
    "четыреста": 400,
    "пятьсот": 500,
    "тысяча": 1000,
    "тысячи": 1000,
    "тысяч": 1000,
}

_MONTHS: dict[str, int] = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
}

# Окончания для грубого стеммера: длинные раньше коротких.
_ENDINGS = (
    "иями",
    "ями",
    "ами",
    "иях",
    "ях",
    "ах",
    "ов",
    "ев",
    "ой",
    "ый",
    "ий",
    "ая",
    "яя",
    "ое",
    "ее",
    "ые",
    "ие",
    "ых",
    "их",
    "ем",
    "ом",
    "ым",
    "им",
    "ам",
    "ям",
    "ам",
    "ах",
    "ую",
    "юю",
    "ью",
    "ия",
    "ии",
    "ей",
    "ей",
    "а",
    "я",
    "о",
    "е",
    "ы",
    "и",
    "у",
    "ю",
    "ь",
    "й",
)

_NUM_WORD_RE = re.compile(
    r"\b(" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
_DIGITS_RE = re.compile(r"\d+(?:[.,]\d+)?")
# Знаки («%») не образуют границы слова, поэтому для них отдельная ветка без \b.
_UNIT_WORDS_LETTERS = [word for word in _UNIT_WORDS if word[:1].isalpha()]
_UNIT_WORDS_SYMBOLS = [word for word in _UNIT_WORDS if not word[:1].isalpha()]
_UNIT_RE = re.compile(
    "|".join(
        filter(
            None,
            [
                r"\b(" + "|".join(re.escape(word) for word in _UNIT_WORDS_LETTERS) + r")\b",
                r"(" + "|".join(re.escape(word) for word in _UNIT_WORDS_SYMBOLS) + r")" if _UNIT_WORDS_SYMBOLS else "",
            ],
        )
    ),
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"\b(\d{1,2})\s+(" + "|".join(_MONTHS) + r")\s+(\d{4})\s*(?:г(?:ода)?\.?)?(?![А-Яа-яЁё])",
    re.IGNORECASE,
)
_DATE_NUM_RE = re.compile(r"\b(\d{1,2})[.](\d{1,2})[.](\d{2,4})\b")
_WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z]+(?:[-'][А-Яа-яЁёA-Za-z]+)*", re.UNICODE)
# Токены для сопоставления: слова и числа (числа терять нельзя — иначе
# «10 лет» и «3 года» станут одинаковыми и подмена числа не будет видна).
_TOKEN_RE = re.compile(r"[А-Яа-яЁёA-Za-z]+|\d+(?:[.,]\d+)?", re.UNICODE)

VALUES_EQUAL = "equal"


def words_to_number(words: list[str]) -> int | None:
    """Публичная обёртка: число из последовательности слов («двадцать пять» → 25)."""
    return _words_to_number_impl(words)


def _words_to_number_impl(words: list[str]) -> int | None:
    """Собрать число из последовательности слов («двадцать пять» → 25)."""
    total = 0
    current = 0
    found = False
    for word in words:
        value = _NUMBER_WORDS.get(word.lower())
        if value is None:
            return None
        found = True
        if value >= 1000:
            current = max(current, 1) * value
            total += current
            current = 0
        elif value >= 100:
            current += value
        else:
            current += value
    if not found:
        return None
    return total + current


def numbers_to_digits(text: str) -> str:
    """Заменить числа прописью на цифры.

    >>> numbers_to_digits("срок хранения пять лет")
    'срок хранения 5 лет'
    >>> numbers_to_digits("двадцать пять лет")
    '25 лет'
    """
    if not text:
        return ""

    words = list(_WORD_RE.finditer(text))
    index = 0
    result: list[str] = []
    position = 0
    while index < len(words):
        match = words[index]
        if match.group(0).lower() not in _NUMBER_WORDS:
            index += 1
            continue
        # Собираем подряд идущие числительные.
        group: list[str] = []
        end = match.end()
        cursor = index
        while cursor < len(words):
            candidate = words[cursor]
            if candidate.group(0).lower() not in _NUMBER_WORDS:
                break
            group.append(candidate.group(0))
            end = candidate.end()
            cursor += 1
        value = _words_to_number_impl(group)
        if value is None:
            index += 1
            continue
        result.append(text[position : match.start()])
        result.append(str(value))
        position = end
        index = cursor
    result.append(text[position:])
    return "".join(result)


def canonical_units(text: str) -> str:
    """Привести единицы измерения к канонической форме.

    >>> canonical_units("срок хранения 5 лет")
    'срок хранения 5 год'
    """
    if not text:
        return ""

    def replace(match: re.Match[str]) -> str:
        return _UNITS.get(match.group(0).lower(), match.group(0))

    return _UNIT_RE.sub(replace, text)


def normalize_dates(text: str) -> str:
    """Привести даты к виду ``ДД.ММ.ГГГГ``.

    >>> normalize_dates("24 июня 2025 г.")
    '24.06.2025'
    """
    if not text:
        return ""

    def replace_words(match: re.Match[str]) -> str:
        day, month, year = match.group(1), match.group(2).lower(), match.group(3)
        return f"{int(day):02d}.{_MONTHS[month]:02d}.{year}"

    def replace_digits(match: re.Match[str]) -> str:
        day, month, year = match.group(1), match.group(2), match.group(3)
        if len(year) == 2:
            year = f"20{year}"
        return f"{int(day):02d}.{int(month):02d}.{year}"

    text = _DATE_RE.sub(replace_words, text)
    return _DATE_NUM_RE.sub(replace_digits, text)


def normalize_numbers(text: str) -> str:
    """Полная нормализация чисел, единиц и дат (порядок преобразований важен)."""
    if not text:
        return ""
    return canonical_units(normalize_dates(numbers_to_digits(text)))


def stem_word(word: str) -> str:
    """Грубый стеммер: усечение русского окончания.

    >>> stem_word("хранения")
    'хранен'
    >>> stem_word("хранение")
    'хранен'
    """
    if not word:
        return ""
    lowered = word.lower().replace("ё", "е")
    if len(lowered) <= 4:
        return lowered
    for ending in _ENDINGS:
        if lowered.endswith(ending) and len(lowered) - len(ending) >= 3:
            return lowered[: -len(ending)]
    return lowered


def stem_text(text: str) -> list[str]:
    """Стеммы всех слов текста в нижнем регистре."""
    return [stem_word(word) for word in _WORD_RE.findall(text or "")]


def normalize_for_match(text: str) -> str:
    """Нормализовать текст для сопоставления: цифры, единицы, даты, стеммы.

    Используется там, где нужно понять, «про то же ли» фрагмент ответа, что и
    фрагмент документа — без требований к дословному совпадению.

    Числа сохраняются: иначе «10 лет» и «3 года» превратились бы в одинаковое
    «год», и подмена числа перестала бы отличаться от правильного значения.
    """
    if not text:
        return ""
    prepared = normalize_numbers(text)
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(prepared):
        token = match.group(0)
        tokens.append(stem_word(token) if token[:1].isalpha() else token)
    return " ".join(tokens)
