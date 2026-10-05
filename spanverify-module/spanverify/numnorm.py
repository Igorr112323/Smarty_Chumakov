"""Нормализация чисел, единиц измерения и словоформ (пункт 2.5 реестра).

Зачем модуль нужен. Признаки продукта опирались на дословное совпадение слов
ответа со словами документа. Из-за этого метод рассыпался на перефразировании и
на числах прописью: «5 лет» и «пять лет» считались разными сведениями, «10 МБ» и
«десять мегабайт» — тоже. Здесь собрано всё, что приводит такие записи к одному
каноническому виду **без сторонних зависимостей** (продукт работает на стандартной
библиотеке).

Что умеет:

* :func:`words_to_number` — русские числительные прописью → целое число
  («двадцать пять» → 25, «полтора» → 1.5, «трое суток» → 3);
* :func:`normalize_numbers` — заменяет в тексте числительные прописью на цифры,
  сохраняя соответствие символьных позиций (возвращает и карту смещений);
* :func:`canonical_unit` — единица измерения → канонический код («мегабайт»,
  «Мб», «МБ» → ``MB``; «рабочих дней» → ``day_work``);
* :func:`measurements` — пары «число + единица» в каноническом виде;
* :func:`stem` — лёгкая морфологическая нормализация русского слова
  (усечение частотных окончаний) без словаря;
* :func:`normalize_text` — полный канонический вид строки для сравнения.

Все функции чистые и детерминированные: один вход — один выход, что требуется
шагом воспроизводимости в CI.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

__all__ = [
    "canonical_unit",
    "measurements",
    "normalize_numbers",
    "normalize_text",
    "stem",
    "words_to_number",
    "Measure",
]

# --------------------------------------------------------------------------
# Числительные прописью
# --------------------------------------------------------------------------

# Основы числительных: ключ — нормализованная основа (без окончания), значение —
# числовое значение. Падежные формы покрываются усечением окончаний в _num_stem.
_UNITS: dict[str, int] = {
    "ноль": 0,
    "нул": 0,
    "один": 1,
    "одн": 1,
    "перв": 1,
    "два": 2,
    "две": 2,
    "двух": 2,
    "втор": 2,
    "дво": 2,
    "три": 3,
    "трех": 3,
    "трёх": 3,
    "трет": 3,
    "тро": 3,
    "четыре": 4,
    "четырех": 4,
    "четырёх": 4,
    "четверт": 4,
    "четвер": 4,
    "пят": 5,
    "шест": 6,
    "сем": 7,
    "семер": 7,
    "восем": 8,
    "восьм": 8,
    "девят": 9,
    "десят": 10,
    "одиннадцат": 11,
    "двенадцат": 12,
    "тринадцат": 13,
    "четырнадцат": 14,
    "пятнадцат": 15,
    "шестнадцат": 16,
    "семнадцат": 17,
    "восемнадцат": 18,
    "девятнадцат": 19,
}

_TENS: dict[str, int] = {
    "двадцат": 20,
    "тридцат": 30,
    "сорок": 40,
    "сорока": 40,
    "пятьдесят": 50,
    "пятидесят": 50,
    "шестьдесят": 60,
    "шестидесят": 60,
    "семьдесят": 70,
    "семидесят": 70,
    "восемьдесят": 80,
    "восьмидесят": 80,
    "девяност": 90,
}

_HUNDREDS: dict[str, int] = {
    "ст": 100,
    "сто": 100,
    "сот": 100,
    "двест": 200,
    "двухсот": 200,
    "трист": 300,
    "трехсот": 300,
    "трёхсот": 300,
    "четырест": 400,
    "четырехсот": 400,
    "пятьсот": 500,
    "пятисот": 500,
    "шестьсот": 600,
    "шестисот": 600,
    "семьсот": 700,
    "семисот": 700,
    "восемьсот": 800,
    "восьмисот": 800,
    "девятьсот": 900,
    "девятисот": 900,
}

_MULTIPLIERS: dict[str, int] = {
    "тысяч": 1000,
    "тыс": 1000,
    "миллион": 1_000_000,
    "млн": 1_000_000,
    "миллиард": 1_000_000_000,
    "млрд": 1_000_000_000,
}

# Дробные и особые формы, которые нельзя собрать из разрядов.
_SPECIAL: dict[str, float] = {
    "полтора": 1.5,
    "полторы": 1.5,
    "полутора": 1.5,
    "половина": 0.5,
    "половины": 0.5,
    "полугода": 0.5,
    "сутки": 1.0,
}

# Окончания, которые отсекаются при приведении числительного к основе.
# Порядок важен: сначала длинные.
_NUM_ENDINGS = (
    "ыми",
    "ими",
    "ого",
    "его",
    "ому",
    "ему",
    "ыми",
    "ых",
    "их",
    "ый",
    "ий",
    "ая",
    "яя",
    "ое",
    "ее",
    "ую",
    "юю",
    "ом",
    "ем",
    "ью",
    "ью",
    "ей",
    "ой",
    "ам",
    "ям",
    "ах",
    "ях",
    "ми",
    "ью",
    "и",
    "ы",
    "а",
    "я",
    "о",
    "е",
    "у",
    "ю",
    "ь",
)


def _num_stem(word: str) -> str:
    """Основа числительного: нижний регистр, 'ё'→'е' не делаем (см. UTF-8/ё)."""
    word = word.lower().strip()
    if word in _TENS or word in _HUNDREDS or word in _MULTIPLIERS or word in _UNITS:
        return word
    for ending in _NUM_ENDINGS:
        if len(word) > len(ending) + 1 and word.endswith(ending):
            candidate = word[: -len(ending)]
            if candidate in _UNITS or candidate in _TENS or candidate in _HUNDREDS or candidate in _MULTIPLIERS:
                return candidate
    return word


def _word_value(word: str) -> tuple[str, float] | None:
    """Разряд и значение одного слова-числительного или ``None``."""
    low = word.lower()
    if low in _SPECIAL:
        return "special", _SPECIAL[low]
    stem_ = _num_stem(low)
    if stem_ in _MULTIPLIERS:
        return "mult", float(_MULTIPLIERS[stem_])
    if stem_ in _HUNDREDS:
        return "hundred", float(_HUNDREDS[stem_])
    if stem_ in _TENS:
        return "ten", float(_TENS[stem_])
    if stem_ in _UNITS:
        return "unit", float(_UNITS[stem_])
    return None


def words_to_number(text: str) -> float | None:
    """Числительное прописью → число. ``None``, если это не числительное.

    >>> words_to_number("двадцать пять")
    25.0
    >>> words_to_number("полтора")
    1.5
    >>> words_to_number("тысяча двести тридцать четыре")
    1234.0
    >>> words_to_number("договор") is None
    True
    """
    words = [w for w in re.split(r"[\s\-]+", text.strip()) if w]
    if not words:
        return None
    total = 0.0
    current = 0.0
    seen = False
    for word in words:
        parsed = _word_value(word)
        if parsed is None:
            return None
        kind, value = parsed
        seen = True
        if kind == "mult":
            current = (current or 1.0) * value
            total += current
            current = 0.0
        elif kind == "special":
            current += value
        else:
            current += value
    if not seen:
        return None
    return total + current


_WORD_RE = re.compile(r"[А-Яа-яЁё]+|\d+(?:[.,]\d+)?", re.UNICODE)


def normalize_numbers(text: str) -> tuple[str, list[tuple[int, int, str]]]:
    """Заменить числительные прописью на цифры.

    Возвращает ``(новый_текст, замены)``, где ``замены`` — список
    ``(начало, конец, цифровая_запись)`` в координатах **исходного** текста.
    Исходные координаты нужны, чтобы разметка фрагментов оставалась верной:
    продукт показывает человеку фрагмент исходного ответа, а сравнивает
    нормализованный.
    """
    matches = list(_WORD_RE.finditer(text))
    replacements: list[tuple[int, int, str]] = []
    index = 0
    while index < len(matches):
        # Жадно набираем максимальную цепочку слов-числительных.
        if _word_value(matches[index].group()) is None:
            index += 1
            continue
        end_index = index
        while end_index + 1 < len(matches) and _word_value(matches[end_index + 1].group()) is not None:
            # Не склеиваем «пять пять» (перечисление) — только разные разряды.
            left = _word_value(matches[end_index].group())
            right = _word_value(matches[end_index + 1].group())
            assert left is not None and right is not None
            order = {"hundred": 3, "ten": 2, "unit": 1, "mult": 4, "special": 1}
            if order[right[0]] >= order[left[0]] and right[0] != "mult":
                break
            end_index += 1
        chunk = text[matches[index].start() : matches[end_index].end()]
        value = words_to_number(chunk)
        if value is not None:
            replacements.append((matches[index].start(), matches[end_index].end(), _fmt(value)))
        index = end_index + 1

    out: list[str] = []
    cursor = 0
    for start, end, value in replacements:
        out.append(text[cursor:start])
        out.append(value)
        cursor = end
    out.append(text[cursor:])
    return "".join(out), replacements


def _fmt(value: float) -> str:
    """Числовое значение → каноническая строка (без хвоста ``.0``)."""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.4f}".rstrip("0").rstrip(".")


# --------------------------------------------------------------------------
# Единицы измерения
# --------------------------------------------------------------------------

# Канонический код -> варианты написания (основы; сравнение по началу слова).
_UNIT_FORMS: dict[str, tuple[str, ...]] = {
    "year": ("год", "года", "году", "лет", "года", "годов", "г", "гг"),
    "month": ("месяц", "мес"),
    "week": ("недел",),
    "day_work": ("рабочих дней", "рабочий день", "рабочих дня", "рабочие дни"),
    "day": ("дн", "день", "дня", "дней", "сутк", "суток"),
    "hour": ("час", "часа", "часов", "ч"),
    "minute": ("минут", "мин"),
    "second": ("секунд", "сек", "с"),
    "MB": ("мегабайт", "мб", "mb", "мбайт"),
    "GB": ("гигабайт", "гб", "gb", "гбайт"),
    "KB": ("килобайт", "кб", "kb", "кбайт"),
    "byte": ("байт",),
    "percent": ("процент", "%"),
    "rub": ("рубл", "руб", "₽"),
    "piece": ("штук", "шт", "экземпляр", "экз"),
    "person": ("человек", "чел", "лиц"),
    "copy": ("копи",),
}

# Единицы с многословным написанием проверяются раньше однословных.
_MULTIWORD_UNITS = {code: forms for code, forms in _UNIT_FORMS.items() if any(" " in f for f in forms)}


def canonical_unit(text: str) -> str | None:
    """Единица измерения → канонический код или ``None``.

    >>> canonical_unit("мегабайт")
    'MB'
    >>> canonical_unit("рабочих дней")
    'day_work'
    >>> canonical_unit("лет")
    'year'
    """
    low = " ".join(text.lower().replace("ё", "е").split())
    if not low:
        return None
    for code, forms in _MULTIWORD_UNITS.items():
        for form in forms:
            if " " in form and low.startswith(form.replace("ё", "е")):
                return code
    first = low.split()[0].strip(".,;:()")
    for code, forms in _UNIT_FORMS.items():
        for form in forms:
            form = form.replace("ё", "е")
            if " " in form:
                continue
            if first == form:
                return code
    # Частичное совпадение по основе (падежные формы: «годами», «часах»).
    for code, forms in _UNIT_FORMS.items():
        for form in forms:
            form = form.replace("ё", "е")
            if " " in form or len(form) < 3:
                continue
            if first.startswith(form):
                return code
    return None


@dataclass(frozen=True)
class Measure:
    """Измеримая величина: значение + каноническая единица + место в тексте."""

    value: float
    unit: str | None
    start: int
    end: int
    raw: str

    @property
    def key(self) -> str:
        """Каноническая запись для сравнения («25|year»)."""
        return f"{_fmt(self.value)}|{self.unit or '-'}"


_DIGIT_RE = re.compile(r"\d+(?:[.,]\d+)?")


def measurements(text: str) -> list[Measure]:
    """Все величины текста в каноническом виде (цифры и прописью).

    Числа прописью распознаются наравне с цифрами, поэтому «пять лет» и
    «5 лет» дают одинаковый :attr:`Measure.key`.
    """
    found: list[Measure] = []
    occupied: list[tuple[int, int]] = []

    # 1) числительные прописью
    _, replacements = normalize_numbers(text)
    for start, end, digits in replacements:
        tail = text[end : end + 24]
        unit = canonical_unit(tail.strip()) if tail.strip() else None
        found.append(Measure(float(digits), unit, start, end, text[start:end]))
        occupied.append((start, end))

    # 2) цифровые записи
    for match in _DIGIT_RE.finditer(text):
        if any(s <= match.start() < e for s, e in occupied):
            continue
        raw = match.group().replace(",", ".")
        tail = text[match.end() : match.end() + 24]
        unit = canonical_unit(tail.strip()) if tail.strip() else None
        found.append(Measure(float(raw), unit, match.start(), match.end(), match.group()))

    found.sort(key=lambda m: m.start)
    return found


# --------------------------------------------------------------------------
# Лёгкая морфология
# --------------------------------------------------------------------------

# Окончания русских слов по убыванию длины. Словаря нет: продукт не тянет
# зависимости, а для сравнения «документ/документов/документами» хватает
# усечения. Это заявлено в документации как приближение, а не как лемматизация.
_ENDINGS = (
    "иями",
    "ениями",
    "ования",
    "ование",
    "овании",
    "ованию",
    "ями",
    "ами",
    "ием",
    "иям",
    "ией",
    "иях",
    "его",
    "ого",
    "ему",
    "ому",
    "ыми",
    "ими",
    "ей",
    "ой",
    "ай",
    "ий",
    "ый",
    "ом",
    "ем",
    "ам",
    "ям",
    "ах",
    "ях",
    "ов",
    "ев",
    "ие",
    "ые",
    "ин",
    "ых",
    "их",
    "ую",
    "юю",
    "ть",
    "ти",
    "ая",
    "яя",
    "ее",
    "ue",
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

_MIN_STEM = 4


def stem(word: str) -> str:
    """Нормализованная основа слова (усечение частотных окончаний).

    «документами», «документов», «документ» → «документ».
    Регистр приводится к нижнему, «ё» → «е» (чтобы «учет» и «учёт» совпадали);
    при этом исходный текст пользователю показывается без изменений.
    """
    word = unicodedata.normalize("NFC", word).lower().replace("ё", "е")
    word = word.strip(".,;:!?()«»\"'—-")
    if len(word) <= _MIN_STEM:
        return word
    for ending in _ENDINGS:
        if word.endswith(ending) and len(word) - len(ending) >= _MIN_STEM:
            return word[: -len(ending)]
    return word


_TOKEN_RE = re.compile(r"[А-Яа-яЁёA-Za-z]+|\d+(?:[.,]\d+)?", re.UNICODE)


def normalize_text(text: str) -> str:
    """Канонический вид строки: числа цифрами, слова — основами.

    Используется там, где нужно сравнение «по смыслу написанного», а не по
    буквам: покрытие фактов, устойчивость к перефразированию, поиск опоры.
    """
    digits, _ = normalize_numbers(text)
    parts = []
    for match in _TOKEN_RE.finditer(digits):
        token = match.group()
        if token[0].isdigit():
            parts.append(_fmt(float(token.replace(",", "."))))
        else:
            parts.append(stem(token))
    return " ".join(parts)
