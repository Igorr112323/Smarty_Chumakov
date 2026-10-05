"""Покрытие фактов документа ответом: что упомянуто, что искажено, что опущено.

Зачем модуль: до него конвейер умел только смотреть «от ответа к документу» — найти
в ответе место, которого документ не подтверждает. Из-за этого два типа расхождений
не обнаруживались вовсе:

* ``missing`` — существенное сведение документа в ответе опущено (recall 0.00 в
  ``reports/CORPUS_REPORT.md``);
* ``partial`` — сведение приведено, но потеряно условие/оговорка (recall 0.27).

Здесь реализован обратный проход: из документа извлекаются **атомарные факты** (срок,
число, запрет, обязанность, перечень, ответственное лицо, форма документа), затем
каждый факт сопоставляется с ответом и получает один из статусов:

    covered   — факт упомянут и не искажён;
    distorted — субъект упомянут, но значение/условие расходится (``partial``,
                ``contradiction``, ``number_attribution``);
    omitted   — субъект упомянут, а обязательное значение потеряно (``missing``,
                ``oversight``);
    absent    — субъект в ответе не упоминается вовсе (не замечание: ответ мог
                сознательно отвечать только на часть документа).

Именно ``distorted`` и ``omitted`` дают замечания (со смещениями **в ответе**), а
``absent`` только считается в статистике — иначе любой краткий ответ получал бы
замечания за то, что не пересказывает весь акт.

Модуль детерминирован, не требует сети и внешних моделей, работает на русских
текстах (см. :mod:`spanverify.normalize`). Решение о риске принимает
:func:`coverage_report`: возвращаются факты, смещения и уровень риска.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from .core import split_sentences, tokenize_with_offsets
from .normalize import (
    NumberMention,
    lemmatize,
    measure_mentions,
    normalize_text,
    numbers_in_text,
)

__all__ = [
    "DOCUMENT_FACT_KINDS",
    "CoverageReport",
    "DocumentFact",
    "FactMatch",
    "coverage_report",
    "extract_salient_facts",
]

DOCUMENT_FACT_KINDS = ("измерение", "запрет", "обязанность", "перечень", "лицо", "форма")

# Маркеры условий и оговорок: если они есть в предложении-факте, а в ответе их нет,
# факт считается приведённым частично (``partial``).
CONDITION_MARKERS: tuple[str, ...] = (
    "при условии",
    "в случае",
    "при наличии",
    "в течение",
    "не позднее",
    "не ранее",
    "за исключением",
    "по истечении",
    "с учетом",
    "в зависимости",
    "если",
    "при этом",
    "как правило",
)

# Маркеры запрета и обязанности. Это предикаты нормы, а не «вежливые» обороты:
# «необходимо отметить» — не обязанность, поэтому такие слова в список не входят.
PROHIBITION_MARKERS: tuple[str, ...] = ("запреща", "не допуска", "не вправе", "запрет", "не может быть")
OBLIGATION_MARKERS: tuple[str, ...] = ("обязан", "должен", "подлежит", "вправе")


def _has_marker(text: str, markers: Sequence[str]) -> bool:
    """Есть ли маркер как отдельное слово или устойчивая связка (не подстрока)."""
    lowered = normalize_text(text)
    for marker in markers:
        if " " in marker:
            if marker in lowered:
                return True
            continue
        if re.search(rf"(^|[^a-zа-я]){re.escape(marker)}", lowered):
            return True
    return False

# Маркеры утверждения: по ним видно, что ответ формулирует факт, а не упоминает тему.
# Нужны, чтобы «пропуск значения» ловился там, где значение ожидалось, и не срабатывал
# на простом упоминании предмета (иначе любой пересказ получает замечания).
CLAIM_MARKERS: tuple[str, ...] = (
    "составля",
    "установ",
    "равн",
    "привед",
    "указыва",
    "определя",
    "подлеж",
    "обязан",
    "вправе",
    "запрещ",
    "не может",
    "не допуска",
    "требу",
    "не превыша",
    "не менее",
    "не более",
    "в течение",
    "не позднее",
)

# Слова, которые не различают субъекты (встречаются почти в каждом факте).
SUBJECT_STOPWORDS: frozenset[str] = frozenset(
    {
        "и",
        "в",
        "во",
        "на",
        "с",
        "со",
        "по",
        "к",
        "ко",
        "о",
        "об",
        "от",
        "до",
        "для",
        "при",
        "из",
        "у",
        "за",
        "над",
        "под",
        "этот",
        "тот",
        "который",
        "свой",
        "весь",
        "также",
        "или",
        "либо",
        "а",
        "но",
        "что",
        "чтобы",
        "если",
        "не",
        "ни",
        "быть",
        "был",
        "была",
        "были",
        "являться",
        "составлять",
        "устанавливать",
        "определять",
        "хранить",
        "иметь",
        "мочь",
        "иной",
        "другой",
        "случай",
        "случаях",
        "порядок",
        "срок",
        "размер",
        "числе",
        "том",
        "могут",
        "может",
    }
)

# Глаголы-связки, отделяющие субъект от значения.
VALUE_INTRODUCERS: tuple[str, ...] = (
    "составляет",
    "составляют",
    "равен",
    "равна",
    "равно",
    "устанавливается",
    "установлен",
    "устанавливаются",
    "определяется",
    "определен",
    "не менее",
    "не более",
    "в размере",
    "в течение",
    "не позднее",
    "не позднее чем",
    "продолжительностью",
)

# Сколько замечаний по покрытию максимум даёт один ответ (защита от «шума» на
# длинных документах: замечания должны быть редкими и значимыми).
MAX_OMITTED_REPORTS = 3
MAX_DISTORTED_REPORTS = 5

# Доли и пороги сопоставления субъектов.
SUBJECT_MATCH_THRESHOLD = 0.6
# Расхождение значения утверждается только при сильном совпадении субъекта:
# ошибка здесь дороже пропуска, потому что даёт ложное замечание на верном ответе.
DISTORTION_OVERLAP = 0.75
# Для запретов и обязанностей слова-маркеры («запрещается») не входят в предмет,
# поэтому порог мягче: предмет в ответе может быть назван синонимом.
SUBJECT_MATCH_THRESHOLD_MODAL = 0.45
# Лемма, встречающаяся более чем в половине фактов документа, не различает факты
# («срок», «хранение», «документ» в акте об archivном деле) и исключается.
DOCUMENT_FREQUENCY_MAX = 0.5
MIN_DISTINCTIVE_LEMMAS = 2


@dataclass(frozen=True)
class DocumentFact:
    """Атомарный факт документа."""

    id: str
    kind: str
    sentence: str
    start: int
    end: int
    subject: tuple[str, ...]  # леммы субъекта
    subject_text: str
    value: str | None  # каноническая запись значения («5|год»)
    number: NumberMention | None
    predicate: str | None
    condition: str | None = None  # найденный маркер условия

    @property
    def distinctive(self) -> tuple[str, ...]:
        """Леммы субъекта без служебных слов — по ним идёт сопоставление."""
        return tuple(lemma for lemma in self.subject if lemma and lemma not in SUBJECT_STOPWORDS)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "sentence": self.sentence[:400],
            "start": self.start,
            "end": self.end,
            "subject": list(self.distinctive),
            "subject_text": self.subject_text,
            "value": self.value,
            "condition": self.condition,
        }


@dataclass(frozen=True)
class FactMatch:
    """Результат сопоставления факта документа с ответом."""

    fact: DocumentFact
    status: str  # covered | distorted | omitted | absent
    reason: str
    answer_start: int | None = None
    answer_end: int | None = None
    answer_text: str = ""
    overlap: float = 0.0
    severity: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "fact": self.fact.as_dict(),
            "status": self.status,
            "reason": self.reason,
            "answer_span": [self.answer_start, self.answer_end],
            "answer_text": self.answer_text[:300],
            "overlap": round(self.overlap, 4),
            "severity": round(self.severity, 4),
        }


@dataclass
class CoverageReport:
    """Итог проверки покрытия фактов документа ответом."""

    facts: list[DocumentFact] = field(default_factory=list)
    matches: list[FactMatch] = field(default_factory=list)
    spans: list[dict[str, Any]] = field(default_factory=list)
    risk: float = 0.0
    counts: dict[str, int] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def flagged(self) -> list[FactMatch]:
        return [match for match in self.matches if match.status in {"distorted", "omitted"}]

    def as_dict(self, include_facts: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "risk": round(self.risk, 4),
            "counts": dict(self.counts),
            "flagged": [match.as_dict() for match in self.flagged],
            "spans": [dict(span) for span in self.spans],
            "meta": dict(self.meta),
        }
        if include_facts:
            payload["facts"] = [fact.as_dict() for fact in self.facts]
        return payload

    def summary(self, limit: int = 5) -> str:
        """Короткая текстовая сводка (для CLI и журнала, без содержимого документа)."""
        parts = [
            f"фактов={self.counts.get('facts', 0)}",
            f"искажено={self.counts.get('distorted', 0)}",
            f"опущено={self.counts.get('omitted', 0)}",
            f"нет в ответе={self.counts.get('absent', 0)}",
        ]
        if self.flagged:
            sample = "; ".join(f"{match.fact.kind}: {match.reason}" for match in self.flagged[:limit])
            parts.append(f"примеры: {sample}")
        return ", ".join(parts)


# ---------------------------------------------------------------------- извлечение


def _sentences(text: str) -> list[tuple[int, int]]:
    """Границы предложений с учётом переводов строк.

    ``core.split_sentences`` разбивает по «знак + пробел + заглавная» и не считает
    границей перевод строки, поэтому абзац из нескольких фактов оставался одним
    предложением и из него извлекался только первый факт. Здесь переводы строк
    считаются границами наравне с точкой.
    """
    spans: list[tuple[int, int]] = []
    for start, end in split_sentences(text):
        cursor = start
        for piece in text[start:end].split("\n"):
            length = len(piece)
            if piece.strip():
                shift = len(piece) - len(piece.lstrip())
                spans.append((cursor + shift, cursor + shift + len(piece.strip())))
            cursor += length + 1
    return spans


def _clause_before(sentence: str, position: int) -> str:
    """Часть предложения до значения — кандидат в «субъект»."""
    prefix = sentence[:position]
    for marker in (";", ":", " - ", " — "):
        index = prefix.rfind(marker)
        if index > 0:
            prefix = prefix[index + len(marker) :]
    return prefix.strip()


def _subject_lemmas(prefix: str, max_words: int = 12) -> tuple[str, ...]:
    """Леммы субъекта: последние ``max_words`` слов части перед значением."""
    tokens = [token.text for token in tokenize_with_offsets(prefix) if token.word]
    words = [token for token in tokens if token.strip()]
    lemmas = [lemmatize(word) for word in words[-max_words:]]
    return tuple(lemma for lemma in lemmas if lemma)


def _find_condition(sentence: str) -> str | None:
    lowered = normalize_text(sentence)
    for marker in CONDITION_MARKERS:
        if marker in lowered:
            return marker
    return None


def _predicate(text: str) -> str | None:
    for marker in (*PROHIBITION_MARKERS, *OBLIGATION_MARKERS):
        if _has_marker(text, (marker,)):
            return marker
    return None


def _is_structural_number(prefix: str) -> bool:
    """Число — реквизит (номер статьи/пункта/документа), а не измерение факта."""
    tail = normalize_text(prefix).strip().rstrip(".,;:(")
    if not tail:
        return False
    last = tail.split()[-1] if tail.split() else ""
    return last in {
        "статья",
        "статьи",
        "статье",
        "статьей",
        "пункт",
        "пункта",
        "пункте",
        "часть",
        "части",
        "частью",
        "приложение",
        "приложения",
        "№",
        "no",
        "п",
        "ч",
        "ст",
        "рис",
        "таблица",
        "таблицы",
        "форма",
        "форм",
        "от",
        "года",
        "г",
        "редакция",
        "редакции",
    }


def extract_salient_facts(document: str, *, max_facts: int = 400) -> list[DocumentFact]:
    """Извлечь атомарные факты документа (не более ``max_facts``).

    Извлекаются только «проверяемые» факты: измерения с единицей измерения или
    числом-значением, запреты, обязанности. Предложения без проверяемых сведений
    пропускаются — иначе любой длинный акт даёт тысячи «фактов» и любое замечание
    перестаёт быть значимым.
    """
    facts: list[DocumentFact] = []
    if not document:
        return facts
    counter = 0
    for start, end in _sentences(document):
        sentence = document[start:end]
        if not sentence.strip():
            continue
        lowered = normalize_text(sentence)
        predicate = _predicate(sentence)
        mentions = measure_mentions(sentence)
        condition = _find_condition(sentence)
        sentence_kind: str | None = None
        accepted: list[NumberMention] = []
        for candidate in mentions:
            prefix = _clause_before(sentence, candidate.start)
            if _is_structural_number(prefix):
                continue
            # Значение факта: число с единицей измерения либо «не менее/в размере N».
            has_unit = candidate.unit is not None
            if has_unit or any(marker in normalize_text(prefix) for marker in ("не менее", "не более", "в размере")):
                accepted.append(candidate)
            if len(accepted) >= 4:
                break
        mention: NumberMention | None = accepted[0] if accepted else None
        if accepted:
            sentence_kind = "измерение"
        elif _has_marker(sentence, PROHIBITION_MARKERS):
            sentence_kind = "запрет"
            predicate = predicate or "запреща"
        elif _has_marker(sentence, OBLIGATION_MARKERS):
            sentence_kind = "обязанность"

        if sentence_kind is None:
            continue
        if accepted:
            # Несколько измерений в одном предложении («срок — 5 лет, объём — 10 Мб»).
            for candidate in accepted:
                counter += 1
                subject_prefix = _clause_before(sentence, candidate.start)
                facts.append(
                    DocumentFact(
                        id=f"fact-{counter:04d}",
                        kind="измерение",
                        sentence=sentence.strip(),
                        start=start,
                        end=end,
                        subject=_subject_lemmas(subject_prefix),
                        subject_text=subject_prefix.strip()[:200],
                        value=f"{_canonical_number(candidate)}|{candidate.unit or ''}",
                        number=candidate,
                        predicate=predicate,
                        condition=condition,
                    )
                )
                if len(facts) >= max_facts:
                    return facts
            continue
        counter += 1
        if mention is not None:
            subject_prefix = _clause_before(sentence, mention.start)
            value = f"{_canonical_number(mention)}|{mention.unit or ''}"
        else:
            subject_prefix = sentence[:200]
            value = None
        if sentence_kind != "измерение":
            # Для запрета/обязанности слово-маркер («запрещается», «обязан») — это
            # модальность, а не часть субъекта: сравнение идёт по объекту.
            marker = predicate or ""
            subject_prefix = " ".join(
                word_part
                for word_part in subject_prefix.split()
                if not normalize_text(word_part).startswith(marker[:6])
            )
            if not subject_prefix.strip():
                subject_prefix = sentence[:200]
        subject_text = subject_prefix.strip()[:200]
        facts.append(
            DocumentFact(
                id=f"fact-{counter:04d}",
                kind=sentence_kind,
                sentence=sentence.strip(),
                start=start,
                end=end,
                subject=_subject_lemmas(subject_prefix),
                subject_text=subject_text,
                value=value,
                number=mention,
                predicate=predicate,
                condition=condition,
            )
        )
        if len(facts) >= max_facts:
            break
    return facts


def _canonical_number(mention: NumberMention) -> str:
    value = mention.value
    if value == int(value):
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------------- сопоставление


def _answer_units(
    answer: str,
) -> list[tuple[int, int, str, tuple[str, ...], list[NumberMention], str, list[NumberMention]]]:
    """Единицы ответа: предложение (или его клауза) со своими леммами и числами."""
    units: list[tuple[int, int, str, tuple[str, ...], list[NumberMention], str, list[NumberMention]]] = []
    for start, end in _sentences(answer):
        sentence = answer[start:end]
        text = sentence.strip()
        if not text:
            continue
        lemmas = tuple(
            lemma
            for lemma in (lemmatize(token.word) for token in tokenize_with_offsets(text) if token.word)
            if lemma
        )
        numbers = numbers_in_text(text)
        # Клаузы внутри предложения: покрытие чаще всего касается одной клаузы.
        parts = _split_clauses(sentence, base=start)
        for part_start, part_end, part_text in parts:
            part_lemmas = tuple(
                lemma
                for lemma in (lemmatize(token.word) for token in tokenize_with_offsets(part_text) if token.word)
                if lemma
            )
            units.append(
                (
                    part_start,
                    part_end,
                    part_text,
                    part_lemmas or lemmas,
                    numbers_in_text(part_text),
                    sentence,
                    numbers,
                )
            )
    return units


def _split_clauses(sentence: str, base: int = 0) -> list[tuple[int, int, str]]:
    """Разбить предложение на клаузы по запятым, тире и союзам.

    ``base`` — смещение предложения внутри ответа: возвращаемые границы
    абсолютные, а не относительно предложения.
    """
    parts: list[tuple[int, int, str]] = []
    cursor = 0
    for index, char in enumerate(sentence):
        if char in ",;—" or (char == "-" and index > 0 and sentence[index - 1] == " "):
            fragment = sentence[cursor:index]
            stripped = fragment.strip()
            if stripped:
                offset = base + cursor + fragment.index(stripped)
                parts.append((offset, offset + len(stripped), stripped))
            cursor = index + 1
    tail = sentence[cursor:]
    stripped = tail.strip()
    if stripped:
        offset = base + cursor + tail.index(stripped)
        parts.append((offset, offset + len(stripped), stripped))
    return parts or [(base, base + len(sentence), sentence)]


def _overlap(fact_lemmas: Sequence[str], unit_lemmas: Sequence[str]) -> float:
    """Доля различающих лемм субъекта, найденных в единице ответа (вхождение)."""
    distinctive = {lemma for lemma in fact_lemmas if lemma not in SUBJECT_STOPWORDS}
    if not distinctive:
        return 0.0
    return len(distinctive & set(unit_lemmas)) / len(distinctive)


def _numbers_match(fact: DocumentFact, numbers: Sequence[NumberMention]) -> bool | None:
    """Совпадает ли значение факта с числами единицы ответа.

    ``True`` — значение найдено, ``False`` — есть числа, но другого значения,
    ``None`` — чисел в единице нет.
    """
    if fact.number is None:
        return None
    for mention in numbers:
        if abs(mention.value - fact.number.value) > 1e-6:
            continue
        if fact.number.unit and mention.unit and fact.number.unit != mention.unit:
            continue
        return True
    return False if numbers else None


def _claim_present(text: str) -> bool:
    """Есть ли в единице ответа маркер утверждения факта (а не упоминания темы)."""
    lowered = normalize_text(text)
    return any(marker in lowered for marker in CLAIM_MARKERS)


def _condition_present(fact: DocumentFact, unit_text: str) -> bool:
    if not fact.condition:
        return True
    return fact.condition in normalize_text(unit_text)


def _find_answer_evidence(answer: str, fact: DocumentFact) -> str:
    """Отпечаток значения факта в ответе — для отчёта (без копирования текста)."""
    if fact.number is None:
        return ""
    for mention in numbers_in_text(answer):
        if abs(mention.value - fact.number.value) <= 1e-6:
            return mention.text
    return ""


def coverage_report(document: str, answer: str, *, max_facts: int = 400) -> CoverageReport:
    """Сопоставить факты документа с ответом и вернуть замечания по покрытию."""
    facts = extract_salient_facts(document, max_facts=max_facts)
    report = CoverageReport(facts=facts)
    units = _answer_units(answer)
    if not facts or not units:
        report.counts = {"facts": len(facts), "covered": 0, "distorted": 0, "omitted": 0, "absent": len(facts)}
        report.meta = {
            "threshold": SUBJECT_MATCH_THRESHOLD,
            "min_distinctive_lemmas": MIN_DISTINCTIVE_LEMMAS,
            "note": "нет фактов или пустой ответ",
        }
        return report

    pending_omitted: list[FactMatch] = []
    pending_distorted: list[FactMatch] = []
    covered = 0
    absent = 0

    # Леммы, встречающиеся у всех фактов документа («срок», «хранения», «документ»),
    # не различают субъекты: без их удаления ответ про «первичные документы» совпадал
    # бы и с фактом про «вторичные» (проверено тестом test_neighbour_facts_do_not_match).
    subject_sets = [set(fact.distinctive) for fact in facts if fact.distinctive]
    document_frequency: dict[str, int] = {}
    for subject in subject_sets:
        for lemma in subject:
            document_frequency[lemma] = document_frequency.get(lemma, 0) + 1
    total_documents = max(1, len(subject_sets))
    common = {
        lemma
        for lemma, count in document_frequency.items()
        if count / total_documents > DOCUMENT_FREQUENCY_MAX
    }

    # Сопоставление идёт в два прохода. Первый считает совпадение каждого факта с
    # каждой единицей ответа, второй выбирает для единицы **лучший** факт. Без
    # второго прохода ответ про «вторичные документы» совпадал бы и с фактом про
    # «первичные»: у них общие слова («срок», «хранения», «документ»), и любое
    # расхождение значения давало бы ложное замечание.
    fact_subjects: list[tuple[DocumentFact, tuple[str, ...]]] = []
    for fact in facts:
        base_subject = fact.distinctive
        if fact.kind == "измерение":
            used = tuple(lemma for lemma in base_subject if lemma not in common)
            # Если после отсечения общих слов не осталось ни одного различающего,
            # факт неразличим среди фактов документа — его нельзя ни подтвердить,
            # ни опровергнуть по субъекту.
            if not used:
                report.matches.append(
                    FactMatch(fact=fact, status="covered", reason="факт неразличим среди фактов документа")
                )
                covered += 1
                continue
        else:
            used = base_subject
        fact_subjects.append((fact, used))

    overlaps: dict[tuple[str, int], float] = {}
    for fact, used in fact_subjects:
        for index, unit in enumerate(units):
            overlaps[(fact.id, index)] = _overlap(used, unit[3])

    best_fact_for_unit: dict[int, tuple[str, float]] = {}
    for (fact_id, index), value in overlaps.items():
        current = best_fact_for_unit.get(index)
        if current is None or value > current[1]:
            best_fact_for_unit[index] = (fact_id, value)

    for fact, distinctive in fact_subjects:
        base_subject = fact.distinctive
        if len(base_subject) < MIN_DISTINCTIVE_LEMMAS:
            covered += 1
            report.matches.append(FactMatch(fact=fact, status="covered", reason="субъект неразличим"))
            continue
        best: tuple[float, int] | None = None
        for index in range(len(units)):
            overlap = overlaps.get((fact.id, index), 0.0)
            if best is None or overlap > best[0]:
                best = (overlap, index)
        assert best is not None
        overlap, index = best
        start, end, text, unit_lemmas, numbers, sentence_text, sentence_numbers = units[index]
        # Совпадение по различающим словам может быть высоким при разном предмете:
        # «для протоколы согласования» есть в фактах про разные признаки. Поэтому
        # дополнительно требуем совпадения по полному субъекту — тогда «срок ответа»
        # не подменяется «предельным объёмом» с тем же объектом.
        if fact.kind == "измерение":
            full = _overlap(base_subject, unit_lemmas)
            if full < SUBJECT_MATCH_THRESHOLD:
                absent += 1
                report.matches.append(
                    FactMatch(
                        fact=fact,
                        status="absent",
                        reason="в ответе назван другой предмет/признак",
                        overlap=full,
                    )
                )
                continue
        threshold = SUBJECT_MATCH_THRESHOLD if fact.kind == "измерение" else SUBJECT_MATCH_THRESHOLD_MODAL
        if overlap < threshold:
            absent += 1
            report.matches.append(
                FactMatch(fact=fact, status="absent", reason="субъект в ответе не упоминается", overlap=overlap)
            )
            continue
        # Конкуренция: если с этой же единицей ответа сильнее совпал другой факт, значит
        # отвели речь о нём, а не о текущем. Сравнивать значения в таком случае нельзя.
        winner = best_fact_for_unit.get(index)
        if winner is not None and winner[0] != fact.id and winner[1] > overlap + 0.05:
            absent += 1
            report.matches.append(
                FactMatch(
                    fact=fact,
                    status="absent",
                    reason="в ответе назван другой факт документа",
                    overlap=overlap,
                )
            )
            continue

        if fact.kind == "измерение":
            match = _numbers_match(fact, numbers)
            match_sentence = _numbers_match(fact, sentence_numbers)
            if match is not True and match_sentence is not None:
                match = match_sentence
            if match is True and _condition_present(fact, sentence_text):
                covered += 1
                report.matches.append(FactMatch(fact=fact, status="covered", reason="значение совпало", overlap=overlap))
            elif match is True:
                pending_distorted.append(
                    FactMatch(
                        fact=fact,
                        status="distorted",
                        reason=f"потеряно условие «{fact.condition}»",
                        answer_start=start,
                        answer_end=end,
                        answer_text=text,
                        overlap=overlap,
                        severity=0.75,
                    )
                )
            elif match is False and overlap >= DISTORTION_OVERLAP:
                pending_distorted.append(
                    FactMatch(
                        fact=fact,
                        status="distorted",
                        reason="значение расходится с документом",
                        answer_start=start,
                        answer_end=end,
                        answer_text=text,
                        overlap=overlap,
                        severity=0.9,
                    )
                )
            elif match is False:
                covered += 1
                report.matches.append(
                    FactMatch(
                        fact=fact,
                        status="covered",
                        reason="совпадение по субъекту слабое — расхождение не утверждается",
                        overlap=overlap,
                    )
                )
            elif _claim_present(sentence_text):
                pending_omitted.append(
                    FactMatch(
                        fact=fact,
                        status="omitted",
                        reason="назван субъект, но значение не приведено",
                        answer_start=start,
                        answer_end=end,
                        answer_text=text,
                        overlap=overlap,
                        severity=0.6,
                    )
                )
            else:
                covered += 1
                report.matches.append(
                    FactMatch(
                        fact=fact,
                        status="covered",
                        reason="только упоминание темы, не утверждение",
                        overlap=overlap,
                    )
                )
        else:  # запрет | обязанность | перечень | лицо | форма
            doc_markers = PROHIBITION_MARKERS if fact.kind == "запрет" else OBLIGATION_MARKERS
            if _has_marker(text, doc_markers):
                covered += 1
                report.matches.append(
                    FactMatch(fact=fact, status="covered", reason="модальность подтверждена", overlap=overlap)
                )
            else:
                pending_omitted.append(
                    FactMatch(
                        fact=fact,
                        status="omitted",
                        reason=f"потеряна модальность «{fact.kind}»",
                        answer_start=start,
                        answer_end=end,
                        answer_text=text,
                        overlap=overlap,
                        severity=0.55,
                    )
                )

    # Замечания отдаём порциями: сначала самые значимые, затем самые подтверждённые
    # по субъекту. Так на длинных актах ответ не получает «простыню» замечаний.
    pending_distorted.sort(key=lambda item: (-item.severity, -item.overlap))
    pending_omitted.sort(key=lambda item: (-item.severity, -item.overlap))
    chosen = [*pending_distorted[:MAX_DISTORTED_REPORTS], *pending_omitted[:MAX_OMITTED_REPORTS]]
    for match in pending_distorted[MAX_DISTORTED_REPORTS:]:
        report.matches.append(FactMatch(fact=match.fact, status="covered", reason="замечание не выдано (лимит)", overlap=match.overlap))
        covered += 1
    for match in pending_omitted[MAX_OMITTED_REPORTS:]:
        report.matches.append(FactMatch(fact=match.fact, status="covered", reason="замечание не выдано (лимит)", overlap=match.overlap))
        covered += 1
    report.matches.extend(chosen)

    report.counts = {
        "facts": len(facts),
        "covered": covered,
        "distorted": len(pending_distorted[:MAX_DISTORTED_REPORTS]),
        "omitted": len(pending_omitted[:MAX_OMITTED_REPORTS]),
        "absent": absent,
        "suppressed": max(0, len(pending_distorted) - MAX_DISTORTED_REPORTS)
        + max(0, len(pending_omitted) - MAX_OMITTED_REPORTS),
    }
    report.risk = max([match.severity for match in chosen], default=0.0)
    report.spans = [
        {
            "start": match.answer_start,
            "end": match.answer_end,
            "kind": match.fact.kind,
            "status": match.status,
            "reason": match.reason,
            "severity": match.severity,
            "fact_id": match.fact.id,
            "fact_value": match.fact.value,
        }
        for match in chosen
    ]
    report.meta = {
        "threshold": SUBJECT_MATCH_THRESHOLD,
        "threshold_modal": SUBJECT_MATCH_THRESHOLD_MODAL,
        "distortion_overlap": DISTORTION_OVERLAP,
        "min_distinctive_lemmas": MIN_DISTINCTIVE_LEMMAS,
        "limits": {"omitted": MAX_OMITTED_REPORTS, "distorted": MAX_DISTORTED_REPORTS},
        "answer_units": len(units),
        "common_lemmas_removed": len(common),
        "document_frequency_max": DOCUMENT_FREQUENCY_MAX,
    }
    return report
