"""Атомарные факты документа и покрытие их ответом (пункты 2.4 и 3.2 реестра).

Зачем модуль нужен. До него продукт умел искать в ответе *лишнее* (выдуманное,
подменённое), но не умел искать *недостающее*: тип расхождения ``missing``
не обнаруживался вообще (recall 0,0), ``partial`` — 0,2667. Причина
архитектурная: конвейер оценивал риск каждого токена ответа, а пропуск сведения
токенов не создаёт — его нет.

Решение: отдельный проход «от документа к ответу». Из документа извлекаются
атомарные факты (субъект, признак, значение, условие), затем проверяется, какие
из них ответ **упомянул**, какие **исказил**, какие **опустил**. Опущенное
значение и опущенное условие дают фрагменты в ответе с типами ``missing`` и
``partial``.

Модуль работает на стандартной библиотеке и на :mod:`spanverify.numnorm`,
поэтому устойчив к числам прописью и к падежным формам.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .numnorm import Measure, measurements, stem

__all__ = [
    "Fact",
    "FactCoverage",
    "coverage",
    "coverage_spans",
    "extract_facts",
    "fact_kinds",
    "split_document_sentences",
]

# Виды атомарных фактов, которые ищем в нормативных актах (п. 3.2 задания:
# срок, число, перечень, запрет, порядок, ответственное лицо, форма документа).
KIND_TERM = "term"  # срок
KIND_NUMBER = "number"  # числовое значение, объём, размер
KIND_LIST = "list"  # перечень
KIND_PROHIBITION = "prohibition"  # запрет
KIND_ORDER = "order"  # порядок совершения действия
KIND_RESPONSIBLE = "responsible"  # ответственное лицо / орган
KIND_FORM = "form"  # форма документа

ALL_KINDS = (
    KIND_TERM,
    KIND_NUMBER,
    KIND_LIST,
    KIND_PROHIBITION,
    KIND_ORDER,
    KIND_RESPONSIBLE,
    KIND_FORM,
)

# Маркеры видов фактов. Списки намеренно короткие и проверяемые: каждый элемент
# встречается в реальных актах и закрыт тестом.
_TERM_MARKERS = (
    "срок",
    "не позднее",
    "не ранее",
    "в течение",
    "ежегодно",
    "ежеквартально",
    "ежемесячно",
    "периодичность",
    "хранится",
    "хранению",
    "хранения",
)
_PROHIBITION_MARKERS = (
    "не допускается",
    "запрещается",
    "запрещено",
    "не вправе",
    "не подлежит",
    "недопустимо",
    "не может быть",
    "исключается",
)
_ORDER_MARKERS = (
    "в порядке",
    "порядок",
    "осуществляется в",
    "проводится в",
    "устанавливается",
    "определяется",
    "оформляется",
)
_RESPONSIBLE_MARKERS = (
    "осуществляет",
    "обеспечивает",
    "утверждает",
    "организует",
    "несёт ответственность",
    "несет ответственность",
    "возлагается на",
    "уполномочен",
    "руководитель",
    "должностное лицо",
    "комиссия",
)
_FORM_MARKERS = (
    "по форме",
    "форма",
    "формы",
    "приложению",
    "приложение",
    "бланк",
    "образец",
    "реквизит",
)
_LIST_MARKERS = ("следующ", "включает", "содержит", "перечень", "состав", ":")

# Слова-заглушки: ответ упомянул признак, но вместо значения поставил общую
# формулировку. Нужны для поиска пропусков (тип ``missing``).
VAGUE_MARKERS = (
    "установлен",
    "установлена",
    "установлено",
    "установлены",
    "определен",
    "определён",
    "определена",
    "определено",
    "определены",
    "предусмотрен",
    "предусмотрена",
    "предусмотрено",
    "предусмотрены",
    "приведен",
    "приведён",
    "приведена",
    "приведены",
    "указан",
    "указана",
    "указано",
    "указаны",
    "регламенте",
    "документе",
    "соответствующ",
    "надлежащ",
    "в установленном",
)

_STOP = {
    "и",
    "или",
    "а",
    "но",
    "для",
    "по",
    "на",
    "в",
    "во",
    "с",
    "со",
    "к",
    "от",
    "до",
    "из",
    "за",
    "при",
    "о",
    "об",
    "не",
    "же",
    "ли",
    "бы",
    "то",
    "как",
    "что",
    "это",
    "его",
    "их",
    "ее",
    "который",
    "которая",
    "которые",
    "быть",
    "есть",
    "составляет",
    "является",
}

_WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z]{2,}", re.UNICODE)


def _content_stems(text: str) -> set[str]:
    """Основы содержательных слов (без предлогов и связок)."""
    out: set[str] = set()
    for match in _WORD_RE.finditer(text):
        base = stem(match.group())
        if base and base not in _STOP and len(base) >= 3:
            out.add(base)
    return out


@dataclass(frozen=True)
class Fact:
    """Атомарный факт документа.

    ``subject``  — о чём факт (подлежащее / предмет регулирования);
    ``feature``  — признак (срок хранения, предельный объём, ответственный…);
    ``value``    — значение в том виде, как оно записано в документе;
    ``condition``— условие («если …»), если оно есть;
    ``kind``     — вид факта из :data:`ALL_KINDS`;
    ``start``/``end`` — границы предложения-источника в тексте документа.
    """

    subject: str
    feature: str
    value: str
    kind: str
    start: int
    end: int
    condition: str | None = None
    sentence: str = ""
    measures: tuple[str, ...] = field(default=())

    @property
    def subject_stems(self) -> set[str]:
        return _content_stems(self.subject)

    @property
    def feature_stems(self) -> set[str]:
        return _content_stems(self.feature)

    @property
    def value_stems(self) -> set[str]:
        return _content_stems(self.value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "feature": self.feature,
            "value": self.value,
            "condition": self.condition,
            "kind": self.kind,
            "start": self.start,
            "end": self.end,
            "measures": list(self.measures),
        }


# --------------------------------------------------------------------------
# Извлечение фактов
# --------------------------------------------------------------------------

# Границы предложений в **документе**: в отличие от ответа, документ содержит
# переводы строк и пункты-перечисления, поэтому разделителем служит и конец
# абзаца. Сокращения («ст.», «п.», «№») намеренно не считаются концом
# предложения — иначе каждый номер статьи рвал бы факт пополам.
_ABBREV = (
    "ст",
    "п",
    "пп",
    "подп",
    "ч",
    "абз",
    "гл",
    "разд",
    "рис",
    "табл",
    "см",
    "др",
    "т",
    "г",
    "гг",
    "руб",
    "коп",
    "тыс",
    "млн",
    "млрд",
    "им",
    "проч",
)
_DOC_SENTENCE_RE = re.compile(r"(?<=[.!?…;])[ \t\r\n]+(?=[«\"(]?[A-ZА-ЯЁ0-9])|\n{2,}")


def split_document_sentences(text: str) -> list[tuple[int, int]]:
    """Границы предложений документа с учётом абзацев и сокращений."""
    if not text or not text.strip():
        return []
    bounds: list[int] = []
    for match in _DOC_SENTENCE_RE.finditer(text):
        head = text[max(0, match.start() - 12) : match.start()].rstrip(".;")
        last_word = re.split(r"[^А-Яа-яЁёA-Za-z]", head)[-1].lower()
        if last_word in _ABBREV:
            continue
        bounds.append(match.end())
    result: list[tuple[int, int]] = []
    start = 0
    for boundary in bounds:
        result.append((start, boundary))
        start = boundary
    if start < len(text):
        result.append((start, len(text)))
    return [(s, e) for s, e in result if text[s:e].strip()]


_CONDITION_RE = re.compile(r"\b(если|при условии|в случае|за исключением|кроме случаев)\b", re.IGNORECASE)
_FOR_RE = re.compile(r"^для\s+(.{3,80}?)\s+(срок|предельн|периодичн|порядок|форма|ответствен)", re.IGNORECASE)


def _kind_of(sentence: str) -> str | None:
    """Вид факта по маркерам предложения (первый подошедший)."""
    low = sentence.lower()
    if any(marker in low for marker in _PROHIBITION_MARKERS):
        return KIND_PROHIBITION
    if any(marker in low for marker in _TERM_MARKERS):
        return KIND_TERM
    if any(marker in low for marker in _FORM_MARKERS):
        return KIND_FORM
    if any(marker in low for marker in _RESPONSIBLE_MARKERS):
        return KIND_RESPONSIBLE
    if any(marker in low for marker in _LIST_MARKERS):
        return KIND_LIST
    if any(marker in low for marker in _ORDER_MARKERS):
        return KIND_ORDER
    return None


def _subject_of(sentence: str) -> str:
    """Субъект факта: конструкция «Для X …» или первая именная группа."""
    match = _FOR_RE.match(sentence.strip())
    if match:
        return match.group(1).strip()
    words = [w for w in re.split(r"\s+", sentence.strip()) if w]
    head: list[str] = []
    for word in words[:6]:
        head.append(word)
        if word.endswith((".", ":", ";")):
            break
        low = word.lower()
        if low in {"составляет", "является", "устанавливается", "определяется", "осуществляется", "—", "-"}:
            head.pop()
            break
    return " ".join(head).strip(" .,:;")


def _feature_of(sentence: str, subject: str) -> str:
    """Признак факта: часть предложения между субъектом и значением."""
    rest = sentence
    if subject and subject in sentence:
        rest = sentence[sentence.index(subject) + len(subject) :]
    rest = rest.strip(" .,:;—-")
    cut = re.split(r"\b(составляет|равен|равна|равно|устанавливается|—|-|:)\b", rest, maxsplit=1)
    return cut[0].strip(" .,:;")[:120]


def _value_of(sentence: str, measures: list[Measure]) -> str:
    """Значение факта: величина, либо часть после сказуемого."""
    if measures:
        first = measures[0]
        tail = sentence[first.start : min(len(sentence), first.end + 24)]
        return tail.strip(" .,:;")
    cut = re.split(r"\b(составляет|равен|равна|равно|устанавливается|осуществляется|—)\b", sentence, maxsplit=1)
    if len(cut) >= 3:
        return cut[2].strip(" .,:;")[:120]
    return sentence.strip(" .,:;")[:120]


def extract_facts(document: str, *, min_chars: int = 25, max_facts: int = 400) -> list[Fact]:
    """Извлечь атомарные факты из текста документа.

    Один факт = одно предложение, в котором нашёлся распознаваемый вид факта.
    Предложения короче ``min_chars`` и предложения без вида факта пропускаются:
    заголовки и реквизиты фактами не считаются.
    """
    facts: list[Fact] = []
    for start, end in split_document_sentences(document):
        sentence = document[start:end].strip()
        if len(sentence) < min_chars:
            continue
        kind = _kind_of(sentence)
        if kind is None:
            continue
        condition = None
        cond_match = _CONDITION_RE.search(sentence)
        if cond_match:
            condition = sentence[cond_match.start() :].strip(" .,;")
        core = sentence[: cond_match.start()] if cond_match else sentence
        measures = measurements(core)
        subject = _subject_of(core)
        feature = _feature_of(core, subject)
        value = _value_of(core, measures)
        if not value:
            continue
        facts.append(
            Fact(
                subject=subject,
                feature=feature,
                value=value,
                kind=kind,
                start=start,
                end=end,
                condition=condition,
                sentence=sentence,
                measures=tuple(m.key for m in measures),
            )
        )
        if len(facts) >= max_facts:
            break
    return facts


def fact_kinds(facts: list[Fact]) -> dict[str, int]:
    """Сводка «вид факта → количество» (для отчёта по корпусу)."""
    out = dict.fromkeys(ALL_KINDS, 0)
    for fact in facts:
        out[fact.kind] = out.get(fact.kind, 0) + 1
    return out


# --------------------------------------------------------------------------
# Покрытие фактов ответом
# --------------------------------------------------------------------------

STATUS_OK = "covered"  # упомянут и значение совпадает
STATUS_DISTORTED = "distorted"  # упомянут, но значение другое
STATUS_OMITTED = "omitted"  # признак упомянут, значения нет (missing)
STATUS_PARTIAL = "partial"  # значение есть, условие потеряно
STATUS_ABSENT = "absent"  # ответ об этом факте вообще не говорит


@dataclass(frozen=True)
class FactCoverage:
    """Результат сверки одного факта документа с ответом."""

    fact: Fact
    status: str
    answer_start: int | None = None
    answer_end: int | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "kind": self.fact.kind,
            "subject": self.fact.subject,
            "feature": self.fact.feature,
            "document_value": self.fact.value,
            "answer_start": self.answer_start,
            "answer_end": self.answer_end,
            "detail": self.detail,
        }


def _overlap(left: set[str], right: set[str]) -> float:
    if not left:
        return 0.0
    return len(left & right) / len(left)


def _last_position(answer: str, stems: set[str]) -> int:
    """Позиция конца последнего слова ответа, чья основа входит в ``stems``."""
    position = 0
    for match in _WORD_RE.finditer(answer):
        if stem(match.group()) in stems:
            position = match.end()
    return position


def _clause_bounds(answer: str, from_position: int) -> tuple[int, int]:
    """Границы клаузы ответа, начиная с ``from_position``.

    Клауза — участок между запятыми/тире/точкой с запятой. Берём **последнюю**
    клаузу хвоста: именно в ней обычно стоит формулировка-заглушка
    («…, значения приведены в регламенте»).
    """
    tail = answer[from_position:]
    if not tail.strip():
        return from_position, len(answer)
    parts = [m for m in re.finditer(r"[,;—]|\s-\s", tail)]
    if parts:
        last = parts[-1]
        start = from_position + last.end()
    else:
        start = from_position
    end = len(answer)
    dot = answer.find(".", start)
    if dot > start:
        end = dot
    while start < end and answer[start] in " \t\n":
        start += 1
    return start, end


def coverage(
    answer: str,
    facts: list[Fact],
    *,
    subject_threshold: float = 0.5,
    feature_threshold: float = 0.6,
) -> list[FactCoverage]:
    """Сверить ответ с фактами документа.

    Для каждого факта определяется, обращается ли к нему ответ (по совпадению
    основ субъекта и признака), и если да — совпало ли значение.

    Пороги ``subject_threshold`` и ``feature_threshold`` подобраны на обучающей
    части корпуса (см. ``reports/experiments/``) и не настраиваются по тесту.
    """
    answer_stems = _content_stems(answer)
    answer_measures = {m.key for m in measurements(answer)}
    results: list[FactCoverage] = []

    for fact in facts:
        subject_hit = _overlap(fact.subject_stems, answer_stems)
        feature_hit = _overlap(fact.feature_stems, answer_stems)
        # Нужны оба совпадения: иначе ответ про «срок хранения счетов-фактур»
        # сверялся бы с фактом про «срок хранения актов» и давал ложный
        # пропуск. Проверено тестом tests/test_facts.py::test_other_subject.
        if subject_hit < subject_threshold or feature_hit < feature_threshold:
            results.append(FactCoverage(fact, STATUS_ABSENT))
            continue

        # Факт адресован ответом. Проверяем значение.
        fact_measures = set(fact.measures)
        value_present: bool
        if fact_measures:
            value_present = bool(fact_measures & answer_measures)
        else:
            value_present = _overlap(fact.value_stems, answer_stems) >= 0.6

        anchor = _last_position(answer, fact.subject_stems | fact.feature_stems)
        clause_start, clause_end = _clause_bounds(answer, anchor)

        if not value_present:
            # Значение не найдено: либо пропуск (missing), либо искажение.
            other_measures = answer_measures - fact_measures
            if fact_measures and other_measures:
                status = STATUS_DISTORTED
            else:
                status = STATUS_OMITTED
            results.append(
                FactCoverage(
                    fact,
                    status,
                    clause_start,
                    clause_end,
                    detail=f"значение документа «{fact.value[:60]}» в ответе не найдено",
                )
            )
            continue

        if fact.condition:
            condition_stems = _content_stems(fact.condition)
            if _overlap(condition_stems, answer_stems) < 0.5:
                start, end = _value_span(answer, fact)
                results.append(
                    FactCoverage(
                        fact,
                        STATUS_PARTIAL,
                        start,
                        end,
                        detail=f"условие «{fact.condition[:60]}» в ответе отсутствует",
                    )
                )
                continue

        results.append(FactCoverage(fact, STATUS_OK, clause_start, clause_end))
    return results


def _value_span(answer: str, fact: Fact) -> tuple[int, int]:
    """Границы значения факта внутри ответа (узкие, без расширения)."""
    fact_measures = set(fact.measures)
    for measure in measurements(answer):
        if measure.key in fact_measures:
            # расширяем до единицы измерения справа, если она есть
            end = measure.end
            tail = re.match(r"\s+[А-Яа-яЁёA-Za-z%]+", answer[end : end + 24])
            if tail and fact.value:
                end = end + tail.end()
            return measure.start, end
    stems_ = fact.value_stems
    starts = [m.start() for m in _WORD_RE.finditer(answer) if stem(m.group()) in stems_]
    ends = [m.end() for m in _WORD_RE.finditer(answer) if stem(m.group()) in stems_]
    if starts:
        return min(starts), max(ends)
    return 0, min(len(answer), 1)


def coverage_spans(
    answer: str,
    facts: list[Fact],
    *,
    max_spans: int = 3,
    subject_threshold: float = 0.5,
    feature_threshold: float = 0.6,
) -> list[dict[str, Any]]:
    """Фрагменты ответа, объясняемые пропуском или усечением факта документа.

    Возвращает список словарей ``{start, end, kind, reason}``:

    * ``kind="missing"`` — факт документа адресован ответом, но значение
      заменено общей формулировкой;
    * ``kind="partial"`` — значение приведено, но потеряно условие документа.

    Фрагменты узкие: границы ставятся по клаузе (для пропуска) или по самому
    значению (для усечения), а не по целому предложению.
    """
    found: list[dict[str, Any]] = []
    for item in coverage(
        answer,
        facts,
        subject_threshold=subject_threshold,
        feature_threshold=feature_threshold,
    ):
        if item.status not in {STATUS_OMITTED, STATUS_PARTIAL}:
            continue
        if item.answer_start is None or item.answer_end is None:
            continue
        start, end = item.answer_start, item.answer_end
        if end <= start or end > len(answer):
            continue
        fragment = answer[start:end]
        if not fragment.strip():
            continue
        if item.status == STATUS_OMITTED:
            # Пропуск подтверждаем только если на месте значения стоит
            # формулировка-заглушка: иначе ответ просто не об этом факте.
            low = fragment.lower()
            if not any(marker in low for marker in VAGUE_MARKERS):
                continue
            kind = "missing"
        else:
            kind = "partial"
        found.append(
            {
                "start": start,
                "end": end,
                "kind": kind,
                "reason": item.detail,
                "fact_kind": item.fact.kind,
            }
        )
    found.extend(_oversight_spans(answer, facts, subject_threshold))
    return _narrowest(found)[:max_spans]


def _oversight_spans(answer: str, facts: list[Fact], subject_threshold: float) -> list[dict[str, Any]]:
    """Фрагменты типа ``oversight``: предмет назван, но значения потеряны.

    Отличается от ``missing`` тем, что ответ не называет и признак: «Для X
    правила учёта определены, значения приведены в регламенте». Такой ответ
    формально не противоречит документу, но ценности не несёт, и приёмка
    считает его расхождением.
    """
    answer_stems = _content_stems(answer)
    if measurements(answer):
        return []
    low_answer = answer.lower()
    if not any(marker in low_answer for marker in VAGUE_MARKERS):
        return []
    addressed = [
        fact for fact in facts if fact.measures and _overlap(fact.subject_stems, answer_stems) >= subject_threshold
    ]
    if not addressed:
        return []
    anchor = _last_position(answer, set().union(*(f.subject_stems for f in addressed)))
    start, end = _clause_bounds(answer, anchor)
    if end <= start or not answer[start:end].strip():
        return []
    return [
        {
            "start": start,
            "end": end,
            "kind": "oversight",
            "reason": f"значения документа ({len(addressed)} факт(ов)) в ответе не приведены",
            "fact_kind": addressed[0].kind,
        }
    ]


def _narrowest(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Из пересекающихся фрагментов оставить самый узкий.

    Несколько фактов документа могут указывать на один и тот же участок ответа
    (например, два срока одного субъекта). Расширять фрагмент в этом случае
    нельзя — это как раз то «расширение до предложения», от которого мы
    уходим (пункт 2.3 реестра), поэтому берём минимальный по длине.
    """
    priority = {"missing": 0, "partial": 0, "oversight": 1}
    ordered = sorted(spans, key=lambda s: (s["end"] - s["start"], priority.get(s["kind"], 2), s["start"]))
    kept: list[dict[str, Any]] = []
    for span in ordered:
        if any(span["start"] < k["end"] and span["end"] > k["start"] for k in kept):
            continue
        kept.append(span)
    return sorted(kept, key=lambda s: s["start"])
