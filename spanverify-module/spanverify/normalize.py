"""Нормализация русских текстов для устойчивых сравнений.

Зачем модуль отдельный: устойчивость метода к перефразированию, числам прописью и
иным шаблонам формулировки (пункт 2.5 промта) упирается не в модель, а в то, **чем
сравнивать** слова и числа. Здесь собраны детерминированные (без внешних зависимостей
и без сети) преобразования:

* ``normalize_text`` — регистр, «ё», типографические дефисы и кавычки, пробелы;
* ``lemmatize`` — упрощённая русская морфология по окончаниям плюс словарь частых
  юридических форм; нужна, чтобы «срок хранения» и «сроки хранения» совпадали;
* ``numbers_in_text`` — канонические числа: цифры, записи словами («двадцать пять»),
  смешанные («5 (пять) лет»), с единицей измерения;
* ``UNIT_KINDS`` — приведение единиц к каноническому виду («рабочих дней» → рабочий
  день, «года» → год, «%» → процент).

Все функции чистые и воспроизводимые: один вход → один выход, без состояния.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

__all__ = [
    "UNIT_KINDS",
    "NumberMention",
    "canonical_unit",
    "frequency_mentions",
    "measure_mentions",
    "lemmatize",
    "lemma_sequence",
    "normalize_text",
    "numbers_in_text",
    "token_lemmas",
]

# ---------------------------------------------------------------- регистр и ё

_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"
_QUOTES = "«»„“”"
_SPACES = "\u00a0\u2007\u202f\u2009\u200a"


def normalize_text(text: str) -> str:
    """Привести текст к сравнимому виду: регистр, «ё», дефисы, кавычки, пробелы."""
    if not text:
        return ""
    out = text.strip().lower().replace("ё", "е")
    for dash in _DASHES:
        out = out.replace(dash, "-")
    for quote in _QUOTES:
        out = out.replace(quote, '"')
    for space in _SPACES:
        out = out.replace(space, " ")
    return re.sub(r"[ \t]+", " ", out)


# ---------------------------------------------------------------- лемматизация

# Словарь частых юридических форм: правило по окончанию даёт ошибку, а эти слова
# встречаются в каждом акте (пункт 2.5), поэтому их формы перечислены явно.
IRREGULAR: dict[str, str] = {
    "год": "год",
    "года": "год",
    "году": "год",
    "годом": "год",
    "годе": "год",
    "годы": "год",
    "годов": "год",
    "годам": "год",
    "годах": "год",
    "лет": "год",
    "сутки": "сутки",
    "суток": "сутки",
    "суткам": "сутки",
    "сутках": "сутки",
    "день": "день",
    "дня": "день",
    "дней": "день",
    "дням": "день",
    "днях": "день",
    "дне": "день",
    "месяц": "месяц",
    "месяца": "месяц",
    "месяцев": "месяц",
    "месяцы": "месяц",
    "месяцам": "месяц",
    "месяцах": "месяц",
    "рубль": "рубль",
    "рубля": "рубль",
    "рублей": "рубль",
    "рублях": "рубль",
    "процент": "процент",
    "процента": "процент",
    "процентов": "процент",
    "процентах": "процент",
    "работа": "работа",
    "работы": "работа",
    "работе": "работа",
    "работу": "работа",
    "работ": "работа",
    "работой": "работа",
    "рабочий": "рабочий",
    "рабочих": "рабочий",
    "рабочим": "рабочий",
    "рабочего": "рабочий",
    "рабочие": "рабочий",
    "человек": "человек",
    "человека": "человек",
    "человеку": "человек",
    "люди": "человек",
    "людей": "человек",
    "лицо": "лицо",
    "лица": "лицо",
    "лиц": "лицо",
    "лицами": "лицо",
    "лицу": "лицо",
    "лице": "лицо",
    "гражданин": "гражданин",
    "граждане": "гражданин",
    "гражданам": "гражданин",
    "гражданина": "гражданин",
    "документ": "документ",
    "документы": "документ",
    "документов": "документ",
    "документа": "документ",
    "документам": "документ",
    "документах": "документ",
    "документе": "документ",
    "срок": "срок",
    "сроки": "срок",
    "сроков": "срок",
    "срока": "срок",
    "сроком": "срок",
    "сроках": "срок",
    "сроку": "срок",
    "порядок": "порядок",
    "порядка": "порядок",
    "порядке": "порядок",
    "порядком": "порядок",
    "порядки": "порядок",
    "порядках": "порядок",
    "обязанность": "обязанность",
    "обязанности": "обязанность",
    "обязанностей": "обязанность",
    "обязанностью": "обязанность",
    "право": "право",
    "права": "право",
    "прав": "право",
    "правом": "право",
    "требование": "требование",
    "требования": "требование",
    "требований": "требование",
    "требованиям": "требование",
    "требованию": "требование",
    "заявление": "заявление",
    "заявления": "заявление",
    "заявлений": "заявление",
    "заявлению": "заявление",
    "обращение": "обращение",
    "обращения": "обращение",
    "обращений": "обращение",
    "обращению": "обращение",
    "ответ": "ответ",
    "ответа": "ответ",
    "ответы": "ответ",
    "ответов": "ответ",
    "услуга": "услуга",
    "услуги": "услуга",
    "услуг": "услуга",
    "услугам": "услуга",
    "услугу": "услуга",
    "нужно": "нужно",
    "необходимо": "необходимо",
    "запрещается": "запрещать",
    "запрещено": "запрещать",
    "запрещен": "запрещать",
    "запрещены": "запрещать",
    "допускается": "допускать",
    "обязан": "обязан",
    "обязана": "обязан",
    "обязаны": "обязан",
    "должен": "должен",
    "должна": "должен",
    "должны": "должен",
    "вправе": "вправе",
    "осуществляется": "осуществлять",
    "устанавливается": "устанавливать",
    "утверждается": "утверждать",
    "определяется": "определять",
    "составляет": "составлять",
    "является": "являться",
    "возлагается": "возлагать",
    "обеспечивается": "обеспечивать",
    "применяется": "применять",
    "хранится": "хранить",
    "хранятся": "хранить",
}

# Окончания по частям речи. Порядок важен: длинные проверяются раньше коротких.
_ENDINGS: tuple[tuple[str, ...], ...] = (
    # причастия и прилагательные
    (
        "ейшими",
        "ейшего",
        "ейших",
        "ейший",
        "ейшая",
        "ейшее",
        "ейшем",
        "ыми",
        "ими",
        "ого",
        "ему",
        "ому",
        "ей",
        "ые",
        "ых",
        "ым",
        "ой",
        "ая",
        "ое",
        "ый",
        "ий",
        "ые",
        "ую",
        "юю",
        "его",
        "ее",
    ),
    # глаголы
    (
        "ироваться",
        "ываться",
        "ировать",
        "оваться",
        "ывается",
        "ируется",
        "ается",
        "яется",
        "ится",
        "ется",
        "овать",
        "ывать",
        "ить",
        "ать",
        "ять",
        "еть",
        "уть",
        "ить",
        "ал",
        "ала",
        "али",
        "ила",
        "или",
        "ила",
        "ло",
        "ли",
        "ет",
        "ут",
        "ют",
        "ит",
        "ат",
        "ят",
    ),
    # существительные
    (
        "ениями",
        "аниями",
        "ениям",
        "аниям",
        "ениями",
        "ениями",
        "ениях",
        "ениями",
        "остями",
        "остях",
        "остей",
        "ость",
        "остью",
        "ости",
        "ями",
        "ами",
        "ах",
        "ях",
        "ов",
        "ев",
        "ий",
        "ия",
        "ию",
        "ие",
        "ии",
        "ей",
        "ям",
        "ам",
        "ом",
        "ем",
        "ой",
        "ей",
        "ую",
        "юю",
        "а",
        "я",
        "о",
        "е",
        "у",
        "ю",
        "ы",
        "и",
        "ь",
    ),
)

# Минимальная основа: короче двух символов не считаем словоразличимым.
_MIN_STEM = 3


def lemmatize(word: str) -> str:
    """Упрощённая лемма слова: словарь форм, затем отсечение окончаний.

    Полноценная морфология (pymorphy3) не входит в обязательные зависимости
    продукта, поэтому используется детерминированное правило. Для сравнения
    «срок хранения» / «сроки хранения» этого достаточно, и это проверено
    тестами: ``tests/test_normalize.py``.
    """
    token = normalize_text(word).strip().strip('".,;:!?()[]{}')
    if not token:
        return ""
    if token in IRREGULAR:
        return IRREGULAR[token]
    if token.isdigit():
        return token
    for group in _ENDINGS:
        for ending in group:
            if token.endswith(ending) and len(token) - len(ending) >= _MIN_STEM:
                return token[: len(token) - len(ending)]
    return token


def lemma_sequence(words: Iterable[str]) -> tuple[str, ...]:
    """Леммы последовательности слов без пустых значений."""
    return tuple(lemma for lemma in (lemmatize(word) for word in words) if lemma)


def token_lemmas(text: str) -> tuple[str, ...]:
    """Леммы слов текста (без пунктуации)."""
    return lemma_sequence(re.findall(r"[A-Za-zА-Яа-яЁё0-9]+(?:-[A-Za-zА-Яа-яЁё0-9]+)*", text))


# ---------------------------------------------------------------- единицы

# Canonical-вид единицы: «рабочих дней» и «рабочий день» — одна единица.
UNIT_KINDS: dict[str, tuple[str, ...]] = {
    "год": ("год", "года", "году", "годы", "годов", "лет", "г", "гг"),
    "месяц": ("месяц", "месяца", "месяцев", "мес"),
    "сутки": ("сутки", "суток", "суткам", "сутках", "дн", "сутками"),
    "календарный день": ("календарный", "календарных", "календарного", "календарные", "календарным"),
    "рабочий день": ("рабочий", "рабочих", "рабочего", "рабочие", "рабочими"),
    "час": ("час", "часа", "часов", "ч"),
    "минута": ("минута", "минуты", "минут", "мин"),
    "процент": ("процент", "процента", "процентов", "%"),
    "рубль": ("рубль", "рубля", "рублей", "руб", "₽"),
    "копейка": ("копейка", "копейки", "копеек", "коп"),
    "штука": ("штука", "штуки", "штук", "шт"),
    "страница": ("страница", "страницы", "страниц", "стр"),
    "килограмм": ("килограмм", "килограмма", "килограммов", "кг"),
    "метр": ("метр", "метра", "метров", "м"),
    "квадратный метр": ("кв", "м2", "квадратный", "квадратных"),
    "мегабайт": ("мегабайт", "мегабайта", "мегабайтов", "мб", "mб"),
    "гигабайт": ("гигабайт", "гигабайта", "гигабайтов", "гб"),
    "день": ("день", "дня", "дней"),
}

_UNIT_LOOKUP: dict[str, str] = {}
for _canonical, _forms in UNIT_KINDS.items():
    for _form in _forms:
        _UNIT_LOOKUP.setdefault(normalize_text(_form), _canonical)


def canonical_unit(word: str) -> str | None:
    """Канонический вид единицы измерения или ``None``, если это не единица."""
    token = normalize_text(word).strip('".,;:!?()[]{}')
    if not token:
        return None
    if token in _UNIT_LOOKUP:
        return _UNIT_LOOKUP[token]
    lemma = lemmatize(token)
    return _UNIT_LOOKUP.get(lemma)


# ---------------------------------------------------------------- числа

# Числительные. Составные собираются последовательным сложением/умножением:
# «двадцать пять» → 25, «сто двадцать» → 120, «тысяча» → 1000.
_NUM_WORDS: dict[str, int] = {
    "ноль": 0,
    "нуль": 0,
    "один": 1,
    "одна": 1,
    "одну": 1,
    "два": 2,
    "две": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
    "десять": 10,
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
    "шестьсот": 600,
    "семьсот": 700,
    "восемьсот": 800,
    "девятьсот": 900,
    # Падежные формы (в актах: «не менее пяти лет», «в течение трёх дней»).
    "двух": 2,
    "трех": 3,
    "четырех": 4,
    "пяти": 5,
    "шести": 6,
    "семи": 7,
    "восьми": 8,
    "девяти": 9,
    "десяти": 10,
    "одиннадцати": 11,
    "двенадцати": 12,
    "четырнадцати": 14,
    "пятнадцати": 15,
    "двадцати": 20,
    "тридцати": 30,
    "сорока": 40,
    "пятидесяти": 50,
    "шестидесяти": 60,
    "семидесяти": 70,
    "восьмидесяти": 80,
    "девяноста": 90,
    "ста": 100,
    "двухсот": 200,
    "трехсот": 300,
    "четырехсот": 400,
    "пятисот": 500,
    "шестисот": 600,
    "семисот": 700,
    "восьмисот": 800,
    # Собирательные числительные: «трое суток», «двое рабочих дней».
    "двое": 2,
    "трое": 3,
    "четверо": 4,
    "пятеро": 5,
    "шестеро": 6,
    "семеро": 7,
}
_MULTIPLIERS = {"тысяча": 1000, "тысячи": 1000, "тысяч": 1000, "миллион": 1_000_000, "миллиона": 1_000_000}
_ORDINALS_ONE = {
    "первый": 1,
    "первого": 1,
    "первом": 1,
    "второй": 2,
    "второго": 2,
    "третий": 3,
    "третьего": 3,
    "четвертый": 4,
    "пятый": 5,
    "шестой": 6,
    "седьмой": 7,
    "восьмой": 8,
    "девятый": 9,
    "десятый": 10,
}


@dataclass(frozen=True)
class NumberMention:
    """Число в тексте: каноническое значение, единица измерения и смещения."""

    value: float
    text: str
    start: int
    end: int
    unit: str | None = None
    source: str = "digits"  # digits | words | mixed

    def as_dict(self) -> dict[str, object]:
        return {
            "value": self.value,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "unit": self.unit,
            "source": self.source,
        }


_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+")


def _collect_number_words(text: str, start: int) -> tuple[float | None, int, str]:
    """Собрать идущие подряд числительные, начиная со смещения ``start``.

    Возвращает ``(значение, конец, способ)``; ``None`` — если с этого места
    числительного нет.
    """
    matches = list(_WORD_RE.finditer(text, start))
    if not matches or matches[0].start() != start:
        return None, start, "words"
    total = 0
    current = 0
    cursor = start
    used_ordinal = False
    for match in matches:
        word = normalize_text(match.group(0))
        if word in _ORDINALS_ONE:
            # Порядковое числительное: «третий» = 3. Используется в перечислениях.
            if current or total:
                break
            current = _ORDINALS_ONE[word]
            used_ordinal = True
            cursor = match.end()
            continue
        if word in _NUM_WORDS:
            current += _NUM_WORDS[word]
        elif word in _MULTIPLIERS:
            total += (current or 1) * _MULTIPLIERS[word]
            current = 0
        else:
            break
        cursor = match.end()
    if current == 0 and not used_ordinal:
        return None, start, "words"
    value = float(total + current)
    return value, cursor, "words"


def numbers_in_text(text: str) -> list[NumberMention]:
    """Все числа текста: цифрами, словами, «5 (пять)» и с единицей измерения.

    Примеры:

    * ``«пять лет»`` → ``value=5, unit='год'``;
    * ``«срок хранения 25 лет»`` → ``value=25``;
    * ``«не менее 5 (пяти) рабочих дней»`` → ``value=5, unit='рабочий день'``.
    """
    mentions: list[NumberMention] = []
    if not text:
        return mentions
    lowered = normalize_text(text)
    index = 0
    length = len(lowered)
    while index < length:
        char = lowered[index]
        if char.isdigit():
            start = index
            while index < length and (lowered[index].isdigit() or lowered[index] in ".,"):
                index += 1
            raw = lowered[start:index].rstrip(".,")
            try:
                value = float(raw.replace(",", "."))
            except ValueError:  # pragma: no cover - защита от нестандартной записи
                index = start + 1
                continue
            end = start + len(raw)
            unit = _unit_after(lowered, end)
            source = "digits"
            # «5 (пять) лет» — уточнение словом в скобках: запись считается
            # смешанной, значение берётся из цифр.
            tail = lowered[end:].lstrip()
            skip_until = end
            if tail.startswith("("):
                closing = tail.find(")")
                inside = tail[1:closing] if closing > 0 else tail[1:60]
                if inside and any(word in _NUM_WORDS or word in _ORDINALS_ONE for word in inside.split()):
                    source = "mixed"
                    # «5 (пяти) процентов» — уточнение того же числа словом: слово в
                    # скобках не должно попасть в список вторым числом.
                    skip_until = end + len(lowered[end:]) - len(tail) + (closing + 1 if closing > 0 else 0)
            mentions.append(NumberMention(value=value, text=lowered[start:end], start=start, end=end, unit=unit, source=source))
            index = max(index, skip_until)
            continue
        match = _WORD_RE.match(lowered, index)
        if match and normalize_text(match.group(0)) in _NUM_WORDS | _ORDINALS_ONE | _MULTIPLIERS:
            value, words_end, _way = _collect_number_words(lowered, index)
            if value is not None and words_end > index:
                unit = _unit_after(lowered, words_end)
                mentions.append(
                    NumberMention(
                        value=value,
                        text=lowered[index:words_end],
                        start=index,
                        end=words_end,
                        unit=unit,
                        source="words",
                    )
                )
                index = words_end
                continue
        index += 1
    return _dedupe_overlapping(mentions)


def _unit_after(text: str, position: int, window: int = 3) -> str | None:
    """Единица измерения сразу после числа (в пределах ``window`` слов)."""
    words = [match.group(0) for match in _WORD_RE.finditer(text, position, position + 40)]
    for word in words[:window]:
        unit = canonical_unit(word)
        if unit:
            return unit
    # Символьные единицы: %, ₽, м².
    tail = text[position : position + 8]
    for symbol, canonical in (("%", "процент"), ("₽", "рубль")):
        if symbol in tail:
            return canonical
    return None


def _dedupe_overlapping(mentions: Sequence[NumberMention]) -> list[NumberMention]:
    """Убрать пересекающиеся упоминания (оставить более полное, цифровое)."""
    ordered = sorted(mentions, key=lambda item: (item.start, -(item.end - item.start)))
    kept: list[NumberMention] = []
    for mention in ordered:
        if kept and mention.start < kept[-1].end:
            if mention.source == "digits" and kept[-1].source != "digits":
                kept[-1] = mention
            continue
        kept.append(mention)
    return kept


# Частотные обороты — тоже измерение («раз в сутки», «каждый час», «каждые 2 дня»).
_FREQUENCY_RE = re.compile(r"\b(раз в|каждые|каждый|каждая|каждое)\s+([A-Za-zА-Яа-яЁё0-9]+)")


def frequency_mentions(text: str) -> list[NumberMention]:
    """Измерения-частоты: «раз в сутки» → ``value=1, unit='сутки'``.

    В нормативных актах периодичность задаётся не числом, а оборотом; без этого
    правила факт «периодичность резервного копирования — раз в сутки» вообще не
    извлекается, и проверить его покрытие нельзя.
    """
    found: list[NumberMention] = []
    lowered = normalize_text(text)
    for match in _FREQUENCY_RE.finditer(lowered):
        word = match.group(2)
        value = 1.0
        if word.isdigit():
            # «каждые 2 дня»: число уже разобрано numbers_in_text, здесь пропускаем.
            continue
        unit = canonical_unit(word)
        if unit is None:
            continue
        found.append(
            NumberMention(
                value=value,
                text=match.group(0),
                start=match.start(),
                end=match.end(),
                unit=unit,
                source="frequency",
            )
        )
    return found


def measure_mentions(text: str) -> list[NumberMention]:
    """Все измерения текста: числа (цифрами и словами) плюс частотные обороты."""
    mentions = [*numbers_in_text(text), *frequency_mentions(text)]
    mentions.sort(key=lambda item: (item.start, -(item.end - item.start)))
    kept: list[NumberMention] = []
    for mention in mentions:
        if kept and mention.start < kept[-1].end:
            continue
        kept.append(mention)
    return kept


def numbers_values(text: str) -> list[float]:
    """Только значения чисел (удобно для тестов и быстрых проверок)."""
    return [mention.value for mention in numbers_in_text(text)]


def canonical_number_string(mention: NumberMention) -> str:
    """Каноническая запись числа с единицей: ``5|год`` — ключ сравнения."""
    value = mention.value
    if value == int(value):
        number = str(int(value))
    else:
        number = f"{value:.4f}".rstrip("0").rstrip(".")
    unit = mention.unit or ""
    return f"{number}|{unit}"


def numbers_compatible(left: NumberMention, right: NumberMention) -> bool:
    """Совпадают ли числа по значению с учётом единицы измерения.

    Единицы обязаны совпадать, если обе известны; неизвестная единица у одной из
    сторон не мешает сравнению (иначе «5 лет» никогда не сравнялось бы с «5»).
    Значения с плавающей точкой сравниваются с допуском 1e-6.
    """
    if abs(left.value - right.value) > 1e-6:
        return False
    if left.unit and right.unit and left.unit != right.unit:
        return False
    return True


def has_number_words(text: str) -> bool:
    """Есть ли в тексте числа, записанные словами (для отчёта об устойчивости)."""
    return any(mention.source == "words" for mention in numbers_in_text(text))


def iter_windows(length: int, window: int, step: int) -> Iterator[tuple[int, int]]:
    """Окна фиксированного размера с перекрытием (для длинных текстов)."""
    if window <= 0:
        raise ValueError("окно должно быть положительным")
    if step <= 0:
        raise ValueError("шаг должен быть положительным")
    start = 0
    while start < length:
        yield start, min(length, start + window)
        if start + window >= length:
            break
        start += step
