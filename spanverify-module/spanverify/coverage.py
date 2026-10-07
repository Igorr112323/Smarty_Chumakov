"""Покрытие фактов документа ответом (исправление B2/B3: типы missing и partial).

Постановка. Раньше конвейер отвечал на вопрос «есть ли в ответе слова, которых нет
в документе». Это ловило подмену числа и выдуманное утверждение, но **пропуск
сведения** не ловился вообще: `by_mode.missing.verdict_recall = 0.0`, а `partial`
— 0,2667. Причина не в пороге, а в том, что проверки покрытия не существовало:
никто не сверял факты документа с тем, что ответ про них сказал.

Здесь реализована эта сверка. Для каждого факта документа определяется статус:

* ``mentioned``  — значение факта есть в ответе;
* ``distorted``  — ответ говорит об этом предмете, но значение другое;
* ``omitted``    — ответ пересказывает именно это предложение, а значение
  выброшено — это и есть **missing**;
* ``partial``    — значение на месте, но условие (оговорка, исключение, порядок)
  отброшено — это **partial**;
* ``irrelevant`` — ответ про другой факт; замечаний нет.

Модуль текстовый и детерминированный: он работает поверх обученной головы, как и
правило привязки числа к объекту, и не требует модели. Нормализация берётся из
:mod:`spanverify.normalize`, поэтому «пять лет» и «5 лет» для него одно и то же.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .core import split_sentences
from .normalize import normalize_for_match, stem_text

__all__ = [
    "Fact",
    "FactCoverage",
    "extract_facts",
    "cover_facts",
    "missing_coverages",
    "partial_coverages",
    "STATUS_MENTIONED",
    "STATUS_DISTORTED",
    "STATUS_OMITTED",
    "STATUS_PARTIAL",
    "STATUS_IRRELEVANT",
]

STATUS_MENTIONED = "mentioned"
STATUS_DISTORTED = "distorted"
STATUS_OMITTED = "omitted"
STATUS_PARTIAL = "partial"
STATUS_IRRELEVANT = "irrelevant"

# Паттерны значений: от специфичных к общим (как в scripts/corpus_real.py, но
# продукт не зависит от скриптов, поэтому список продублирован здесь).
_MONTHS = "января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря"
_UNITS = "лет|года|годов|году|год|месяцев|месяца|месяц|мес|недель|недели|неделю|дней|дня|день|суток|часов|часа|час|минут|минуты|минуту"
_WORD_NUMBERS = (
    "одного|одной|одному|один|одна|двух|двумя|два|две|трёх|тремя|три|четырёх|четыре|пяти|пять|"
    "шести|шесть|семи|семь|восьми|восемь|девяти|девять|десяти|десять|одиннадцать|двенадцать|"
    "тринадцать|четырнадцать|пятнадцать|шестнадцать|семнадцать|восемнадцать|девятнадцать|"
    "двадцать|тридцать|сорок|пятьдесят|шестьдесят|семьдесят|восемьдесят|девяносто|сто"
)
_MEASURES = (
    "экземпляров|экземпляра|экземпляр|процентов|процента|процент|рублей|рубля|рубль|"
    "документов|документа|человек|сотрудников|сотрудника|дней|дня|день|раз|раза|штук|штуки"
)

VALUE_PATTERNS: tuple[tuple[str, str], tuple[str, str], ...] = (
    ("date_day_month_year", (rf"\d{{1,2}}\s+(?:{_MONTHS})\s+\d{{4}}(?:\s*г(?:ода)?\.?)?",)),
    ("date_numeric", (r"\d{1,2}\.\d{1,2}\.\d{2,4}",)),
    ("year", (r"\d{4}\s*(?:год|года|году|г\.)",)),
    ("money", (r"\d[\d\s]*(?:,\d+)?\s*(?:руб(?:лей|ля|\.)?|тыс\.?\s*руб(?:\.|лей)?)",)),
    ("percent", (r"\d+(?:[,.]\d+)?\s*(?:процент(?:ов|а)?|%)",)),
    ("word_term", (rf"(?:{_WORD_NUMBERS})\s+(?:{_UNITS})",)),
    ("digit_term", (rf"\d[\d\s]*\s*(?:{_UNITS})",)),
    ("digit_measure", (rf"\d[\d\s]*(?:,\d+)?\s+(?:{_MEASURES})",)),
    ("count", (r"\d[\d\s]{0,12}\d",)),
    ("single_number", (r"\d",)),
    # Длина ограничена тремя словами: более длинные хвосты захватывали всё
    # предложение целиком и ломали извлечение фактов (проверено тестами).
    (
        "text_after_verb",
        (
            r"(?:составляет|составляют|равен|равно|равна|равны|установлен|"
            r"устанавливается|определяется|принимает)\s+"
            r"([А-Яа-яЁё]+(?:\s+[А-Яа-яЁё]+){0,2})",
        ),
    ),
)

# Маркеры условия: их отсутствие в ответе при наличии значения — признак partial.
CONDITION_MARKERS = (
    "если",
    "в случае",
    "при условии",
    "за исключением",
    "кроме случаев",
    "кроме",
    "по согласованию",
    "в порядке",
    "установленном",
    "при этом",
    "в течение",
    "не позднее",
    "не ранее",
    "за исключением случаев",
)

# Служебные слова: они не могут быть предметом факта.
_STOP_SUBJECT = {
    "и",
    "а",
    "но",
    "или",
    "в",
    "во",
    "на",
    "за",
    "по",
    "от",
    "до",
    "из",
    "с",
    "со",
    "к",
    "у",
    "о",
    "об",
    "при",
    "для",
    "не",
    "ни",
    "что",
    "как",
    "этом",
    "этой",
    "этот",
    "эти",
    "этих",
    "том",
    "его",
    "ее",
    "их",
    "быть",
    "был",
    "была",
    "были",
    "составляет",
    "составляют",
    "является",
    "являются",
    "должен",
    "должна",
    "должно",
    "должны",
    "может",
    "могут",
    "имеет",
    "имеют",
    "устанавливается",
    "устанавливают",
    "определяется",
    "также",
    "том числе",
    "числе",
}

_WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z]+", re.UNICODE)


@dataclass(frozen=True)
class Fact:
    """Факт документа: предложение, значение в нём и предмет значения."""

    sentence: str
    sentence_start: int
    value: str
    value_start: int  # смещение в документе
    value_end: int
    kind: str
    subject: str  # нормализованный предмет («срок хранения»)
    condition: str  # нормализованное условие, если есть

    @property
    def norm_value(self) -> str:
        return normalize_for_match(self.value)


@dataclass(frozen=True)
class FactCoverage:
    """Чем ответ закрыл факт документа."""

    fact: Fact
    status: str
    answer_start: int  # куда указывать в ответе; -1, если указывать некуда
    answer_end: int
    overlap: float  # доля совпавших стеммов ответа и факта (для диагностики)
    subject_start: int = -1  # где в ответе стоит предмет факта (для сужения границ)
    subject_end: int = -1


def _find_values(sentence: str) -> list[tuple[int, int, str, str]]:
    """Значения предложения: ``(начало, конец, текст, вид)`` без пересечений."""
    matches: list[tuple[int, int, str, str, int]] = []
    for order, (kind, (pattern,)) in enumerate(VALUE_PATTERNS):
        for found in re.finditer(pattern, sentence, flags=re.IGNORECASE):
            text = found.group(0).strip()
            if not text:
                continue
            matches.append((found.start(), found.end(), text, kind, order))
    matches.sort(key=lambda item: (item[0], item[4], -(item[1] - item[0])))
    chosen: list[tuple[int, int, str, str, int]] = []
    for candidate in matches:
        if any(candidate[0] < taken[1] and taken[0] < candidate[1] for taken in chosen):
            continue
        chosen.append(candidate)
    chosen.sort(key=lambda item: item[0])
    return [(start, end, text, kind) for start, end, text, kind, _ in chosen]


def _subject_of(sentence: str, value_start: int, max_words: int = 4) -> str:
    """Предмет значения: значимые слова слева от него.

    Слова берутся в порядке «от значения назад», чтобы «первичных документов»
    попало в предмет, а начало длинного предложения — нет.
    """
    left = sentence[:value_start]
    words = _WORD_RE.findall(left)
    picked: list[str] = []
    for word in reversed(words):
        lowered = word.lower()
        if lowered in _STOP_SUBJECT or len(lowered) < 3:
            continue
        picked.append(lowered)
        if len(picked) >= max_words:
            break
    return " ".join(reversed(picked))


def _condition_of(sentence: str) -> str:
    """Условие факта: часть предложения после маркера условия."""
    lowered = sentence.lower()
    best: tuple[int, int] | None = None
    for marker in CONDITION_MARKERS:
        position = lowered.find(marker)
        if position < 0:
            continue
        if best is None or position < best[0]:
            best = (position, position + len(sentence[position:].split(",")[0]))
    if best is None:
        return ""
    return sentence[best[0] : best[1]].strip(" ,.;:")


def extract_facts(context: str, min_words: int = 5, max_facts: int = 60) -> list[Fact]:
    """Собрать факты документа: предложения со значениями.

    Ограничения нужны, чтобы один длинный акт не перевесил остальные: не более
    ``max_facts`` фактов и не короче ``min_words`` слов в предложении.
    """
    facts: list[Fact] = []
    if not context:
        return facts
    for start, end in split_sentences(context):
        sentence = context[start:end]
        if len(sentence.split()) < min_words:
            continue
        values = _find_values(sentence)
        if not values:
            continue
        # Предпочитаем содержательные значения: даты, сроки, суммы полезнее голых цифр.
        detailed = [item for item in values if item[3] not in {"count", "single_number"}]
        value_start, value_end, value_text, value_kind = (detailed or values)[0]
        facts.append(
            Fact(
                sentence=sentence,
                sentence_start=start,
                value=value_text,
                value_start=start + value_start,
                value_end=start + value_end,
                kind=value_kind,
                subject=_subject_of(sentence, value_start),
                condition=_condition_of(sentence),
            )
        )
        if len(facts) >= max_facts:
            break
    return facts


def _overlap(stems_answer: list[str], stems_fact: list[str]) -> float:
    """Доля стеммов факта, встретившихся в ответе."""
    if not stems_fact:
        return 0.0
    pool = set(stems_answer)
    hits = sum(1 for stem in stems_fact if stem in pool)
    return hits / len(stems_fact)


def _subject_in(subject: str, stems_answer: list[str], threshold: float = 0.6) -> bool:
    """Есть ли предмет факта в ответе (не менее ``threshold`` его слов)."""
    if not subject:
        return False
    words = stem_text(subject)
    if not words:
        return False
    pool = set(stems_answer)
    hits = sum(1 for word in words if word in pool)
    return hits / len(words) >= threshold


def _subject_position(answer: str, subject: str) -> tuple[int, int]:
    """Где в ответе стоит предмет факта: границы первого и последнего его слова.

    Нужно, чтобы указывать на конкретное место, а не на всё предложение: при
    пропуске значения именно здесь ответ «недоговорил».
    """
    if not subject or not answer:
        return -1, -1
    words = [word for word in re.findall(r"[А-Яа-яЁёA-Za-z]+", subject.lower()) if len(word) > 2]
    if not words:
        return -1, -1
    lowered = answer.lower()
    first = -1
    last = -1
    for word in words:
        position = lowered.find(word)
        if position < 0:
            continue
        first = position if first < 0 else min(first, position)
        last = max(last, position + len(word))
    return first, last


def cover_facts(
    answer: str,
    context: str,
    *,
    omit_overlap: float = 0.55,
    subject_threshold: float = 0.6,
    omit_subject_threshold: float = 1.0,
) -> list[FactCoverage]:
    """Сверить факты документа с ответом.

    ``omit_overlap`` — насколько близко ответ должен пересказывать предложение
    факта, чтобы отсутствие значения считалось пропуском. Порог нужен, чтобы
    короткий ответ по одному факту не получал замечания за все остальные факты
    документа (иначе растёт доля ложных замечаний на чистых парах).

    ``omit_subject_threshold`` — отдельный, более строгий порог для пропуска.
    Он равен единице: все слова предмета должны присутствовать в ответе. Это
    остановка от типичной ошибки: документ хранит «первичные — 10 лет» и
    «вторичные — 5 лет», ответ про первичные верен, но механизм видел бы
    совпадение по общим словам («срок», «хранения», «документов») и требовал бы
    упоминания вторичных. Проверено тестом
    ``test_number_from_own_object_is_grounded``.
    """
    coverages: list[FactCoverage] = []
    if not answer or not context:
        return coverages

    norm_answer = normalize_for_match(answer)
    norm_answer_words = set(norm_answer.split())
    answer_sentences = list(split_sentences(answer))

    for fact in extract_facts(context):
        norm_value = fact.norm_value
        value_present = bool(norm_value) and norm_value in norm_answer
        stems_fact = stem_text(fact.sentence)

        # Ищем предложение ответа, наиболее похожее на предложение факта.
        best_span = (-1, -1)
        best_overlap = 0.0
        for start, end in answer_sentences:
            overlap = _overlap(stem_text(answer[start:end]), stems_fact)
            if overlap > best_overlap:
                best_overlap, best_span = overlap, (start, end)

        subject_stems = stem_text(fact.subject)
        subject_here = _subject_in(fact.subject, list(norm_answer_words), subject_threshold)
        # Для пропуска предмет должен присутствовать целиком (см. доку строку).
        subject_full = _subject_in(fact.subject, list(norm_answer_words), omit_subject_threshold)

        if value_present:
            status = STATUS_MENTIONED
            # Значение на месте, но условие отброшено — partial.
            if fact.condition:
                condition_stems = stem_text(fact.condition)
                condition_hits = sum(1 for stem in condition_stems if stem in norm_answer_words)
                condition_ratio = condition_hits / max(1, len(condition_stems))
                if condition_ratio < 0.5:
                    status = STATUS_PARTIAL
        elif subject_full and best_overlap >= omit_overlap:
            # Ответ пересказывает предложение факта, но значения в нём нет.
            # Если при этом в ответе стоит другое значение того же вида — это
            # подмена (distorted), а не пропуск: так contradiction не попадает
            # в метрику missing и не портит её.
            status = STATUS_OMITTED
            if best_span != (-1, -1):
                other_values = _find_values(answer[best_span[0] : best_span[1]])
                same_kind = [item for item in other_values if item[3] == fact.kind]
                if not same_kind:
                    same_kind = other_values
                if any(normalize_for_match(item[2]) != norm_value for item in same_kind):
                    status = STATUS_DISTORTED
        elif (
            subject_here >= 0
            and best_overlap >= omit_overlap
            and _subject_in(fact.subject, list(norm_answer_words), 0.8)
        ):
            # Ответ говорит об этом же предмете, но значение другое. Порог к
            # предмету строже, чем для подмены числом: иначе ответ про один
            # объект документа получал бы замечания за все остальные объекты.
            status = STATUS_DISTORTED
        elif subject_full and best_overlap >= omit_overlap:
            status = STATUS_OMITTED
        else:
            status = STATUS_IRRELEVANT

        if status == STATUS_IRRELEVANT:
            coverages.append(FactCoverage(fact, status, -1, -1, round(best_overlap, 4)))
            continue

        if best_span == (-1, -1):
            best_span = (0, len(answer))
        if status == STATUS_MENTIONED and not subject_stems:
            best_span = (-1, -1)
        subj_start, subj_end = _subject_position(answer, fact.subject)
        coverages.append(
            FactCoverage(
                fact,
                status,
                best_span[0],
                best_span[1],
                round(best_overlap, 4),
                subj_start,
                subj_end,
            )
        )
    return coverages


def missing_coverages(coverages: list[FactCoverage]) -> list[FactCoverage]:
    """Факты, пропущенные ответом (тип missing)."""
    return [item for item in coverages if item.status == STATUS_OMITTED]


def partial_coverages(coverages: list[FactCoverage]) -> list[FactCoverage]:
    """Факты, переданные без условия (тип partial)."""
    return [item for item in coverages if item.status == STATUS_PARTIAL]
