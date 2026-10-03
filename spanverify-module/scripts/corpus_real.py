"""Корпус A3: пары на фрагментах **реальных** опубликованных нормативных документов.

Зачем отдельный модуль
----------------------

Корпус A1 (``scripts/build_corpus_a.py``) строит документы сам: факты, предложения и
ответы создаёт генератор, поэтому у каждой пары ``meta.synthetic_document = true``.
Корпус A3 собирается тем же генератором (та же таксономия, те же режимы, тот же
баланс, те же сплиты), но **документы настоящие**: тексты официально опубликованных
актов, скачанные загрузчиком (``scripts/fetch_npa_corpus.py``) и лежащие в
``data/corpus_a3/sources/``.

Что здесь делается:

* ``normalize_text`` — единственная нормализация пробелов; проверка «context дословно
  найден в файле источника» сравнивает именно нормализованные строки, поэтому одна и
  та же функция применяется и при сборке, и в проверке;
* ``split_sentences`` / ``find_values`` — разбор реального текста на предложения и
  поиск в них значений (даты, сроки, суммы, проценты, количества);
* ``extract_facts`` — «факт» реального документа: предложение (дословно), значение в
  нём и его смещения (``answer[start:end] == span`` выполняется по построению);
* ``build_real_variant`` — ответ по режиму и разметка (метка — смещения в **ответе**);
* ``check_contexts_in_sources`` / ``check_number_attribution`` — проверки, которые
  отличают A3 от A1: контекст обязан дословно лежать в файлах источников, а новое
  число при «атрибуции числа» — в другом месте того же документа.

Ничего не выдумывается: если подходящего места в документе нет, факт или режим
пропускается (``ValueError``/``None``), а не подменяется шаблоном.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Нормализация и разбор текста
# ---------------------------------------------------------------------------

_WHITESPACE_IN_LINE = re.compile(r"[ \t]+")
_SPACES_AROUND_NEWLINE = re.compile(r" *\n *")
_MULTIPLE_NEWLINES = re.compile(r"\n{3,}")
_SINGLE_NEWLINE = re.compile(r"(?<!\n)\n(?!\n)")
_SENTENCE_BOUNDARY = re.compile(r"[.!?…]+(?=\s+(?:[А-ЯЁA-Z0-9«\"(]|$))")

_MONTHS = "января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря"
_WORD_NUMBERS = (
    "пять|шесть|семь|восемь|девять|десять|одиннадцать|двенадцать|тринадцать|четырнадцать|"
    "пятнадцать|шестнадцать|семнадцать|восемнадцать|девятнадцать|двадцать|тридцать|сорок|"
    "пятьдесят|шестьдесят|семьдесят|восемьдесят|девяносто|сто|двести|триста|четыреста|пятьсот"
)
_UNITS = (
    "календарных дней|рабочих дней|рабочих дня|календарных дня|суток|сутки|"
    "дней|дня|день|недель|недели|неделю|месяцев|месяца|месяц|лет|года|год|часов|часа|час|минут|минуты|минуту"
)

# Порядок важен: более специфичные образцы проверяются раньше общих.
VALUE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("date_day_month_year", rf"\d{{1,2}}\s+(?:{_MONTHS})\s+\d{{4}}(?:\s*г(?:ода)?\.?)?"),
    ("date_numeric", r"\d{1,2}\.\d{1,2}\.\d{2,4}"),
    ("date_month_year", rf"(?:{_MONTHS})\s+\d{{4}}(?:\s*г(?:ода)?\.?)?"),
    ("year", r"\d{4}\s*(?:год|года|году|г\.)"),
    ("money", r"\d[\d\s]*(?:,\d+)?\s*(?:руб(?:лей|ля|\.)?|тыс\.?\s*руб(?:\.|лей)?|млн\s*руб(?:\.|лей)?)"),
    ("percent", r"\d+(?:[,.]\d+)?\s*(?:процент(?:ов|а)?|%)"),
    ("word_term", rf"(?:{_WORD_NUMBERS})\s+(?:{_UNITS})"),
    ("digit_term", rf"\d[\d\s]*\s*(?:{_UNITS})"),
    ("word_count", rf"(?:{_WORD_NUMBERS})\s+(?:документов|документа|человек|сотрудников|дней|раза|раз)"),
    ("count", r"\d[\d\s]{0,12}\d"),
    ("single_number", r"\d"),
)

# Условия, которые режим ``partial`` отбрасывает (ищутся в начале придаточной части).
CONDITION_MARKERS = (
    "если ",
    "в случае ",
    "при условии",
    "за исключением",
    "кроме случаев",
    "по согласованию",
    "в порядке, установленном",
)


def normalize_text(text: str) -> str:
    """Единая нормализация: переводы строк, неразрывные пробелы, мягкие переносы.

    Функция **идемпотентна**: повторное применение ничего не меняет. Это важно, потому
    что проверка «контекст дословно найден в источнике» нормализует и контекст, и файл.
    Одиночные переводы строки становятся пробелом (в источниках это перенос по ширине
    страницы), пустая строка остаётся границей абзаца.
    """
    if not text:
        return ""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\u00a0", " ").replace("\u202f", " ").replace("\u2009", " ")
    cleaned = cleaned.replace("\u00ad", "").replace("\ufeff", "")
    cleaned = _SINGLE_NEWLINE.sub(" ", cleaned)
    cleaned = _WHITESPACE_IN_LINE.sub(" ", cleaned)
    cleaned = _SPACES_AROUND_NEWLINE.sub("\n", cleaned)
    cleaned = _MULTIPLE_NEWLINES.sub("\n\n", cleaned)
    return cleaned.strip()


def split_sentences(text: str) -> list[tuple[int, int, str]]:
    """Разбить нормализованный текст на предложения: ``(начало, конец, текст)``.

    Границей считается ``.``/``!``/``?``/``…``, после которого идёт пробел и заглавная
    буква, цифра или кавычка. Текст предложения — точный срез исходной строки, поэтому
    контексты, собранные из соседних предложений, дословно присутствуют в источнике.
    """
    if not text:
        return []
    sentences: list[tuple[int, int, str]] = []
    start = 0
    for match in _SENTENCE_BOUNDARY.finditer(text):
        end = match.end()
        fragment = text[start:end].strip()
        if fragment:
            offset = start + (len(text[start:end]) - len(text[start:end].lstrip()))
            sentences.append((offset, offset + len(fragment), fragment))
        start = end
    tail = text[start:].strip()
    if tail:
        offset = start + (len(text[start:]) - len(text[start:].lstrip()))
        sentences.append((offset, offset + len(tail), tail))
    return sentences


def find_values(sentence: str) -> list[tuple[int, int, str, str]]:
    """Найти в предложении значения: ``(начало, конец, текст, вид)``.

    Пересечения отбрасываются: если образцы совпали на одном участке, остаётся более
    специфичный (он идёт раньше в ``VALUE_PATTERNS``) и более длинный.
    """
    matches: list[tuple[int, int, str, str, int]] = []
    for order, (kind, pattern) in enumerate(VALUE_PATTERNS):
        for found in re.finditer(pattern, sentence, flags=re.IGNORECASE):
            text = found.group(0).strip()
            if not text or not text[0].isalnum() and not text[0].isdigit():
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


@dataclass(frozen=True)
class RealFact:
    """Факт реального документа: предложение (дословно) и значение в нём."""

    doc_id: str
    sentence_index: int
    sentence: str
    value: str
    value_kind: str
    value_start: int
    value_end: int
    fact_start: int
    fact_end: int

    def response(self) -> str:
        """Эталонный ответ — выдержка из документа (экстрактивный эталон)."""
        return self.sentence

    def sentence_without(self, replacement: str) -> str:
        """Предложение с заменённым значением (остальной текст не меняется)."""
        return self.sentence[: self.value_start] + replacement + self.sentence[self.value_end :]


def extract_facts(doc_id: str, text: str, min_words: int = 8, max_facts: int = 200) -> list[RealFact]:
    """Собрать факты документа: предложения со значениями, не короче ``min_words`` слов.

    Возвращает не более ``max_facts`` фактов на документ — чтобы один длинный акт не
    перевесил остальные документы при разбиении на сплиты.
    """
    facts: list[RealFact] = []
    for index, (start, _end, sentence) in enumerate(split_sentences(text)):
        if len(sentence.split()) < min_words:
            continue
        values = find_values(sentence)
        if not values:
            continue
        # Берём значение «покрупнее»: описательные виды (даты, сроки, суммы) полезнее голых чисел.
        detailed = [item for item in values if item[3] not in {"count", "single_number"}]
        value_start, value_end, value_text, value_kind = (detailed or values)[0]
        # Абсолютные смещения предложения в документе нужны для проверки «число есть в другом месте».
        absolute_start, absolute_end = start, start + len(sentence)
        facts.append(
            RealFact(
                doc_id=doc_id,
                sentence_index=index,
                sentence=sentence,
                value=value_text,
                value_kind=value_kind,
                value_start=value_start,
                value_end=value_end,
                fact_start=absolute_start,
                fact_end=absolute_end,
            )
        )
        if len(facts) >= max_facts:
            break
    return facts


def build_context(text: str, fact: RealFact, neighbours: int = 1, max_chars: int = 1600) -> str:
    """Контекст пары: предложение факта и соседние предложения (дословный срез текста).

    Срез берётся из нормализованного текста, поэтому он гарантированно присутствует в
    файле источника; ограничение ``max_chars`` не даёт контекстам разрастаться.
    """
    sentences = split_sentences(text)
    position = next((index for index, item in enumerate(sentences) if item[0] == fact.fact_start), None)
    if position is None:
        return fact.sentence
    first = max(0, position - neighbours)
    last = min(len(sentences) - 1, position + neighbours)
    start = sentences[first][0]
    end = sentences[last][1]
    context = text[start:end]
    while len(context) > max_chars and first < position:
        first += 1
        context = text[sentences[first][0] : end]
    return context


# ---------------------------------------------------------------------------
# Режимы ответа и разметка
# ---------------------------------------------------------------------------

# Общие с корпусом A1 дополнительные детали: их в документе нет.
EXTRA_DETAILS = (
    "архивная копия на внешнем носителе",
    "опись вложений в двух экземплярах",
    "отметка службы безопасности",
    "реестр передачи в архив",
)

MISSING_CLAIM = "Значение не приведено"
OVERSIGHT_CLAIM = "Точное значение приведено в документе"


def _condition_clause(sentence: str) -> tuple[int, int] | None:
    """Найти придаточную часть с условием (её можно отбросить в режиме ``partial``).

    Ищется ближайшая запятая перед маркером условия; границей считается конец
    предложения. Если маркер стоит в начале предложения, часть не отбрасывается —
    иначе от предложения ничего не останется.
    """
    lowered = sentence.lower()
    for marker in CONDITION_MARKERS:
        position = lowered.find(marker)
        if position <= 0:
            continue
        comma = sentence.rfind(",", 0, position)
        if comma <= 0:
            continue
        end = len(sentence)
        return comma, end
    return None


def _clauses(sentence: str) -> list[tuple[int, int]]:
    """Разбить предложение на части по запятым и точкам с запятой: ``(начало, конец)``."""
    parts: list[tuple[int, int]] = []
    start = 0
    for found in re.finditer(r"[,;]", sentence):
        parts.append((start, found.start()))
        start = found.end()
    parts.append((start, len(sentence)))
    return [(item[0], item[1]) for item in parts if sentence[item[0] : item[1]].strip()]


def _without_value_clause(sentence: str, value_start: int, value_end: int) -> str | None:
    """Убрать из предложения часть, несущую значение; вернуть остаток (дословный).

    Возвращает ``None``, если предложение состоит из одной части: тогда убрать значение
    без искажения смысла нельзя, и факт пропускается (никаких шаблонов вместо текста).
    """
    parts = _clauses(sentence)
    target = next((item for item in parts if item[0] <= value_start < item[1] or item[0] < value_end <= item[1]), None)
    if target is None or len(parts) < 2:
        return None
    kept = [sentence[item[0] : item[1]].strip(" ,;") for item in parts if item != target]
    kept = [item for item in kept if item]
    if not kept:
        return None
    remainder = ", ".join(kept)
    if not remainder.endswith((".", "!", "?")):
        remainder += "."
    return remainder


def _mark(answer: str, fragment: str) -> list[list[int]]:
    """Смещения фрагмента в ответе; фрагмент обязан присутствовать дословно."""
    start = answer.find(fragment)
    if start < 0:
        raise ValueError(f"фрагмент {fragment!r} отсутствует в ответе")
    return [start, start + len(fragment), 1]


def build_real_variant(
    fact: RealFact,
    mode: str,
    rng: random.Random,
    same_kind_donors: list[str],
    other_place_donors: list[str],
) -> dict | None:
    """Собрать ответ и разметку для одного режима на реальном факте.

    ``same_kind_donors`` — значения того же вида из других документов корпуса (для
    ``contradiction``: подставляется значение, которого в этом документе нет),
    ``other_place_donors`` — значения из других предложений **того же** документа
    (для ``number_attribution``).

    Возвращает ``None``, если для режима нет подходящего материала: лучше пропустить
    факт, чем подставить шаблон и выдать его за реальный документ.
    """
    if mode == "faithful":
        return {"answer": fact.response(), "labels": [], "spans_text": []}

    if mode == "contradiction":
        candidates = [value for value in same_kind_donors if value and value not in fact.sentence]
        if not candidates:
            return None
        replacement = rng.choice(candidates)
        answer = fact.sentence_without(replacement)
        return {"answer": answer, "labels": [_mark(answer, replacement)], "spans_text": [replacement]}

    if mode == "number_attribution":
        candidates = [value for value in other_place_donors if value and value not in fact.sentence]
        if not candidates:
            return None
        replacement = rng.choice(candidates)
        answer = fact.sentence_without(replacement)
        return {"answer": answer, "labels": [_mark(answer, replacement)], "spans_text": [replacement]}

    if mode == "partial":
        clause = _condition_clause(fact.sentence)
        if clause is None or fact.value not in fact.sentence:
            return None
        answer = (fact.sentence[: clause[0]] + fact.sentence[clause[1] :]).strip()
        answer = re.sub(r"\s+", " ", answer)
        if fact.value not in answer:
            # Условие оказалось раньше значения — тогда отбрасывать нечего.
            return None
        return {"answer": answer, "labels": [_mark(answer, fact.value)], "spans_text": [fact.value]}

    if mode == "unconfirmed":
        detail = f"Также требуется {rng.choice(EXTRA_DETAILS)}."
        answer = f"{fact.sentence} {detail}"
        return {"answer": answer, "labels": [_mark(answer, detail)], "spans_text": [detail]}

    if mode == "excess":
        detail = f"а также требуется {rng.choice(EXTRA_DETAILS)}"
        answer = fact.sentence.rstrip(".") + f", {detail}."
        return {"answer": answer, "labels": [_mark(answer, detail)], "spans_text": [detail]}

    if mode == "missing":
        remainder = _without_value_clause(fact.sentence, fact.value_start, fact.value_end)
        if remainder is None:
            return None
        answer = f"{remainder} {MISSING_CLAIM}."
        return {"answer": answer, "labels": [_mark(answer, MISSING_CLAIM)], "spans_text": [MISSING_CLAIM]}

    if mode == "oversight":
        remainder = _without_value_clause(fact.sentence, fact.value_start, fact.value_end)
        if remainder is None:
            return None
        answer = f"{remainder} {OVERSIGHT_CLAIM}."
        return {"answer": answer, "labels": [_mark(answer, OVERSIGHT_CLAIM)], "spans_text": [OVERSIGHT_CLAIM]}

    raise ValueError(f"неизвестный режим: {mode}")


# ---------------------------------------------------------------------------
# Проверки, отличающие A3 от A1
# ---------------------------------------------------------------------------


def load_sources(sources_dir: Path) -> dict[str, str]:
    """Прочитать файлы источников: ``{doc_id: нормализованный текст}``.

    Читаются ``*.txt`` корня каталога (файлы, которые положил загрузчик) — подкаталоги
    игнорируются, чтобы случайный мусор не попал в проверку.
    """
    sources: dict[str, str] = {}
    for path in sorted(sources_dir.glob("*.txt")):
        sources[path.stem] = normalize_text(path.read_text(encoding="utf-8"))
    return sources


def check_contexts_in_sources(pairs: list[dict], sources_dir: Path) -> dict:
    """Проверка №1: каждый ``context`` дословно (после нормализации) есть в источниках.

    Это главная защита от синтетики: если контекст собрать генератором, а не вырезать из
    документа, проверка падает.
    """
    sources = load_sources(sources_dir)
    checked = 0
    problems: list[str] = []
    for pair in pairs:
        checked += 1
        doc_id = pair["meta"].get("doc_id") or pair["meta"].get("group")
        source = sources.get(str(doc_id))
        if source is None:
            problems.append(f"{pair['id']}: источник {doc_id!r} не найден в {sources_dir}")
            continue
        context = normalize_text(pair["context"])
        if context not in source:
            problems.append(f"{pair['id']}: контекст не найден дословно в источнике {doc_id}")
    return {"checked": checked, "problems": problems}


def check_number_attribution(pairs: list[dict], sources_dir: Path) -> dict:
    """Проверка №3: при «атрибуции числа» новое число есть в другом месте того же документа.

    Для каждой пары режима ``number_attribution`` берётся размеченный фрагмент ответа и
    проверяется, что он встречается в тексте документа **за пределами** предложения-факта.
    """
    sources = load_sources(sources_dir)
    checked = 0
    problems: list[str] = []
    for pair in pairs:
        if pair["meta"].get("mode") != "number_attribution":
            continue
        checked += 1
        doc_id = str(pair["meta"].get("doc_id") or pair["meta"].get("group"))
        source = sources.get(doc_id)
        if source is None:
            problems.append(f"{pair['id']}: источник {doc_id!r} не найден")
            continue
        spans = pair["meta"].get("span_texts") or []
        if not spans:
            problems.append(f"{pair['id']}: нет размеченного фрагмента")
            continue
        fragment = spans[0]
        fact_sentence = pair["meta"].get("fact_sentence", "")
        outside = source.replace(fact_sentence, " ", 1) if fact_sentence else source
        if fragment not in outside:
            problems.append(f"{pair['id']}: значение {fragment!r} не найдено в другом месте документа {doc_id}")
    return {"checked": checked, "problems": problems}


def pair_sha256(pair: dict) -> str:
    """SHA256 пары по канонической JSON-записи (для манифеста «хеш каждой пары»)."""
    canonical = json.dumps(pair, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
