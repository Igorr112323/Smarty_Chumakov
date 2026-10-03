"""Демонстрационный корпус пар «контекст — ответ» с разметкой по токенам.

Каждая пара строится вокруг факта (атрибут — истинное значение):

* **достоверный ответ** берёт значение из контекста (метка 0);
* **недостоверный ответ** подменяет значение (метка 1) и/или добавляет
  утверждения, которых в контексте нет.

Метки ставятся по символам в ответе, поэтому их можно пересчитать в токенные
и считать precision/recall/F1 конвейера.

Правила, без которых корпус был бы бесполезен (и которые проверены тестами):

* один и тот же шаблон не повторяется внутри документа-пары;
* подмена — именно по «слоту» факта, а не случайная замена слова;
* в «выдуманных» фрагментах нет ни одного слова из контекста, иначе метка
  перестаёт соответствовать признаку.

Корпус синтетический: числа на нём проверяют конвейер, а не качество на
реальных документах.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "Fact",
    "Pair",
    "make_pair",
    "generate_pairs",
    "write_pairs",
    "DatasetFormatError",
    "read_pairs",
    "validate_pair",
    "corpus_statistics",
    "DATASET_VERSION",
]

DATASET_VERSION = "2.0-synthetic-pairs"

# ---------------------------------------------------------------- факты

# Единицы измерения с формами склонения по числу.
UNIT_FORMS: dict[str, dict[str, str]] = {
    "years": {"one": "год", "few": "года", "many": "лет"},
    "months": {"one": "месяц", "few": "месяца", "many": "месяцев"},
    "days24": {"one": "сутки", "few": "суток", "many": "суток"},
    "workdays": {"one": "рабочий день", "few": "рабочих дня", "many": "рабочих дней"},
    "caldays": {"one": "календарный день", "few": "календарных дня", "many": "календарных дней"},
}


def unit_form(value: int, kind: str) -> str:
    """Правильная форма единицы: 5 лет, 3 года, 1 год."""
    forms = UNIT_FORMS[kind]
    tail = abs(value) % 100
    if 11 <= tail <= 14:
        return forms["many"]
    last = tail % 10
    if last == 1:
        return forms["one"]
    if 2 <= last <= 4:
        return forms["few"]
    return forms["many"]


SUBJECTS = [
    ("срок хранения первичных документов", "years", [3, 4, 5, 6, 10, 25, 45]),
    ("срок хранения актов проверок", "years", [3, 4, 5, 6, 10]),
    ("срок хранения личных карточек работников", "years", [50, 75]),
    ("срок хранения журналов инструктажа", "years", [5, 10, 25]),
    ("период проведения внутреннего аудита", "months", [6, 9, 12]),
    ("срок ответа на запрос надзорного органа", "workdays", [5, 10, 15, 30]),
    ("срок хранения накладных", "years", [4, 5, 10]),
    ("срок рассмотрения обращения", "caldays", [15, 30, 45]),
    ("срок действия пропуска", "months", [3, 6, 12]),
    ("срок хранения протоколов заседаний", "years", [3, 5, 10]),
    ("период инвентаризации склада", "months", [6, 12, 24]),
    ("срок хранения технических паспортов", "years", [10, 20, 25]),
    ("срок мониторинга оборудования", "days24", [7, 14, 30]),
    ("срок хранения договоров подряда", "years", [5, 10, 25]),
    ("срок обжалования предписания", "workdays", [10, 15, 20]),
]

CONTEXT_TEMPLATES = [
    "Регламент {number}: {subject} составляет {value_phrase}.",
    "Внутренний регламент {number}. {Subject} составляет {value_phrase}.",
    "Согласно регламенту {number}, {subject} установлен в размере {value_phrase}.",
]
CONTEXT_TAIL = [  # noqa: RUF100
    "Документ утверждён приказом директора.",
    "Контроль исполнения возложен на службу делопроизводства.",
    "Изменения вносятся распоряжением руководителя.",
    "Ответственный за исполнение — начальник отдела.",
]

FAQ_TEMPLATES = [
    ("Каков {subject}?", "{Subject} составляет {value_phrase}."),
    ("Какой {subject} установлен регламентом?", "Регламент устанавливает {value_phrase}."),
    ("Что говорится о таком показателе, как {subject}?", "Установленный {subject} — {value_phrase}."),
    ("На какой {subject} следует ориентироваться?", "{Subject} установлен в размере {value_phrase}."),
]

FABRICATIONS = [
    "Дополнительно требуется согласование с внешним аудитором и архивным агентством.",
    "Показатель пересматривается ежеквартально профильной комиссией совета директоров.",
    "Значение индексируется с учётом инфляции и уточняется статистическим управлением.",
    "Контроль передаётся независимому оператору электронного документооборота.",
    "Применяется повышающий коэффициент, рассчитанный экспертной организацией.",
]

EXTRA_TRUE = [
    "Регламент {number} размещён на внутреннем портале.",
    "Ответственный за хранение — служба делопроизводства.",
    "Копия документа направляется в юридический отдел.",
]

# Числительные для формулировки значения словами.
NUMERALS = {
    3: "три",
    4: "четыре",
    5: "пять",
    6: "шесть",
    7: "семь",
    9: "девять",
    10: "десять",
    12: "двенадцать",
    14: "четырнадцать",
    15: "пятнадцать",
    20: "двадцать",
    24: "двадцать четыре",
    25: "двадцать пять",
    30: "тридцать",
    45: "сорок пять",
    50: "пятьдесят",
    75: "семьдесят пять",
}


@dataclass
class Fact:
    """Факт-основа пары."""

    number: int
    subject_template: str
    unit_kind: str
    value: int
    template: str
    tail: str

    @property
    def unit(self) -> str:
        return unit_form(self.value, self.unit_kind)

    @property
    def subject(self) -> str:
        return self.subject_template

    @property
    def subject_titled(self) -> str:
        return self.subject_template[0].upper() + self.subject_template[1:]

    @property
    def value_words(self) -> str:
        return NUMERALS.get(self.value, str(self.value))

    def value_phrase(self, value: int | None = None) -> str:
        """«5 лет» / «3 года» — значение с правильно склонённой единицей."""
        chosen = self.value if value is None else value
        return f"{chosen} {unit_form(chosen, self.unit_kind)}"


@dataclass
class Pair:
    """Пара «контекст — ответ» с разметкой по символам ответа."""

    id: str
    context: str
    answer: str
    labels: list[list] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "context": self.context,
            "answer": self.answer,
            "labels": self.labels,
            "meta": self.meta,
        }


def _pick_unused(rng: random.Random, pool: list, used: dict[str, set]) -> object:
    key = f"pool{len(pool)}"
    taken: set = used.setdefault(key, set())
    available = [item for item in pool if repr(item) not in taken]
    if not available:
        taken.clear()
        available = list(pool)
    choice = rng.choice(available)
    taken.add(repr(choice))
    return choice


def make_pair(rng: random.Random | None = None, index: int = 0, hallucination_rate: float = 0.5) -> Pair:
    """Собрать одну пару «контекст — ответ» со разметкой."""
    rng = rng or random.Random()
    used: dict[str, set] = {}

    number = rng.randint(100, 999)
    subject_template, unit_kind, values = rng.choice(SUBJECTS)
    true_value = rng.choice(values)
    template = _pick_unused(rng, CONTEXT_TEMPLATES, used)
    tail = _pick_unused(rng, CONTEXT_TAIL, used)
    fact = Fact(number, subject_template, unit_kind, true_value, template, tail)

    context = fact.template.format(
        number=fact.number,
        subject=fact.subject,
        Subject=fact.subject_titled,
        value_phrase=fact.value_phrase(),
    ).replace("составляет составляет", "составляет")
    if rng.random() < 0.4:
        context += " " + fact.tail

    question, answer_template = _pick_unused(rng, FAQ_TEMPLATES, used)
    question = question.format(subject=fact.subject)

    hallucinate = rng.random() < hallucination_rate
    if not hallucinate:
        answer = answer_template.format(
            subject=fact.subject,
            Subject=fact.subject_titled,
            value_phrase=fact.value_phrase(),
        )
        labels: list[list] = []
        if rng.random() < 0.3:
            extra = _pick_unused(rng, EXTRA_TRUE, used).format(number=fact.number)
            answer = f"{answer} {extra}"
            # Утверждение верное — значит, оно должно быть подтверждено документом,
            # иначе получился бы «недостоверный токен с меткой 0» и шум в обучении.
            context = f"{context} {extra}"
        kind = "faithful"
    elif rng.random() < 0.7:
        # Подмена значения — основной тип недостоверности.
        wrong_value = rng.choice([v for v in values if v != fact.value])
        answer = answer_template.format(
            subject=fact.subject,
            Subject=fact.subject_titled,
            value_phrase=fact.value_phrase(wrong_value),
        )
        labels, kind = _value_span(answer, wrong_value), "value_substitution"
    else:
        # Выдуманное утверждение: в контексте таких сведений нет.
        fabrication = _pick_unused(rng, FABRICATIONS, used)
        base = answer_template.format(
            subject=fact.subject, Subject=fact.subject_titled, value_phrase=fact.value_phrase()
        )
        answer = f"{base} {fabrication}"
        start = answer.find(fabrication)
        labels = [[start, start + len(fabrication), 1]]
        kind = "fabrication"

    return Pair(
        id=f"pair-{index:05d}",
        context=context,
        answer=answer,
        labels=labels,
        meta={
            "kind": kind,
            "subject": fact.subject,
            "true_value": fact.value,
            "unit": fact.unit,
            "question": question,
            "dataset_version": DATASET_VERSION,
        },
    )


def _value_span(answer: str, value: int) -> list[list]:
    """Разметка значения: цифрами или словом (что встретилось в ответе раньше)."""
    digit = str(value)
    word = NUMERALS.get(value, "")
    candidates: list[tuple[int, int]] = []
    if digit in answer:
        candidates.append((answer.find(digit), len(digit)))
    if word and word in answer:
        candidates.append((answer.find(word), len(word)))
    if not candidates:
        return []
    start, length = min(candidates, key=lambda item: item[0])
    return [[start, start + length, 1]]


def generate_pairs(
    n_pairs: int = 240,
    seed: int = 1312,
    hallucination_rate: float = 0.5,
) -> list[Pair]:
    """Сгенерировать корпус пар с заданным seed (воспроизводимо)."""
    rng = random.Random(seed)
    return [make_pair(rng, index=index, hallucination_rate=hallucination_rate) for index in range(n_pairs)]


def write_pairs(pairs: list[Pair], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
    return target


REQUIRED_PAIR_FIELDS = ("context", "answer")
ALLOWED_PAIR_FIELDS = ("id", "context", "answer", "labels", "meta")
PAIR_EXAMPLE = (
    '{"id": "p1", "context": "текст документа-источника", "answer": "текст ответа", '
    '"labels": [[12, 27, 1]], "meta": {"kind": "faithful"}}'
)


class DatasetFormatError(ValueError):
    """Корпус не соответствует схеме пар «контекст — ответ».

    Ошибка схемы — это не «плохие метрики», а невозможность считать метрики
    вообще: раньше чужой формат принимался молча, и ``evaluate`` печатал
    ``F1 0.000, AUC=nan`` с кодом возврата 0 (дефект E / N1).
    """


def _schema_problem(path: Path, line_number: int, reasons: list[str]) -> DatasetFormatError:
    """Собрать понятное сообщение об ошибке схемы с примером корректной строки."""
    where = f"{path}: строка {line_number}"
    lines = [f"{where}: корпус не соответствует формату «контекст — ответ»."]
    lines.extend(f"  - {reason}" for reason in reasons)
    lines.append(f"  ожидалось: {', '.join(REQUIRED_PAIR_FIELDS)}; допустимы также {', '.join(ALLOWED_PAIR_FIELDS)}")
    lines.append(f"  пример корректной строки: {PAIR_EXAMPLE}")
    return DatasetFormatError("\n".join(lines))


def validate_pair(data: object, *, path: str | Path, line_number: int) -> dict:
    """Проверить одну пару корпуса и вернуть её как словарь.

    Проверяются: тип записи (объект), обязательные ключи ``context``/``answer``
    (непустые строки), отсутствие неизвестных ключей, ``labels`` — список троек
    ``[start, end, label]`` в границах ответа, ``meta`` — объект, ``id`` — строка.
    """
    target = Path(path)
    if not isinstance(data, dict):
        raise _schema_problem(target, line_number, [f"ожидался объект JSON, найдено: {type(data).__name__}"])
    reasons: list[str] = []
    missing = [name for name in REQUIRED_PAIR_FIELDS if name not in data]
    if missing:
        reasons.append(f"отсутствуют обязательные поля: {', '.join(missing)}")
    unknown = [name for name in data if name not in ALLOWED_PAIR_FIELDS]
    if unknown:
        reasons.append(f"неизвестные поля: {', '.join(sorted(unknown))}")
    for name in REQUIRED_PAIR_FIELDS:
        if name in data and not isinstance(data[name], str):
            reasons.append(f"поле {name} должно быть строкой, найдено: {type(data[name]).__name__}")
    answer = data.get("answer") if isinstance(data.get("answer"), str) else ""
    if "labels" in data:
        labels = data["labels"]
        if not isinstance(labels, list):
            reasons.append("labels должен быть списком троек [start, end, label]")
        else:
            for position, label in enumerate(labels, start=1):
                if not isinstance(label, list | tuple) or len(label) != 3:
                    reasons.append(f"метка №{position} должна быть тройкой [start, end, label]: {label!r}")
                    continue
                start, end, flag = label
                if not all(isinstance(value, int) for value in (start, end, flag)):
                    reasons.append(f"метка №{position}: значения должны быть целыми: {label!r}")
                    continue
                if flag not in (0, 1):
                    reasons.append(f"метка №{position}: недопустимая метка {flag} (допустимо 0 или 1)")
                if not 0 <= start <= end <= len(answer):
                    reasons.append(
                        f"метка №{position} выходит за границы ответа: [{start}, {end}] "
                        f"при длине ответа {len(answer)}"
                    )
    if "meta" in data and not isinstance(data["meta"], dict):
        reasons.append("meta должен быть объектом")
    if "id" in data and not isinstance(data["id"], str):
        reasons.append("id должен быть строкой")
    if reasons:
        raise _schema_problem(target, line_number, reasons)
    return dict(data)


def read_pairs(path: str | Path) -> Iterator[dict]:
    """Читать корпус JSONL со проверкой схемы (см. :class:`DatasetFormatError`).

    Формат: одна пара на строку, обязательны ``context`` и ``answer``,
    ``labels`` — смещения в символах ответа. Чужой формат (например, исторический
    ``text``/``label``) отвергается с указанием строки и примера.
    """
    target = Path(path)
    total = 0
    with target.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            total += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError as error:
                raise _schema_problem(
                    target,
                    line_number,
                    [f"строка не является корректным JSON: {error.msg} (позиция {error.colno})"],
                ) from error
            yield validate_pair(data, path=target, line_number=line_number)
    if total == 0:
        raise DatasetFormatError(
            f"{target}: файл пуст — метрики считать не на чем.\n"
            f"  ожидалось: по одной паре на строку; пример: {PAIR_EXAMPLE}"
        )


def corpus_statistics(pairs: list[Pair] | list[dict]) -> dict:
    """Статистика корпуса: размеры, доли классов, словарь."""
    kinds: dict[str, int] = {}
    total_answer = total_context = total_flagged = 0
    vocabulary: set[str] = set()
    for pair in pairs:
        data = pair.to_dict() if isinstance(pair, Pair) else pair
        kind = data.get("meta", {}).get("kind", "unknown")
        kinds[kind] = kinds.get(kind, 0) + 1
        answer = data["answer"]
        total_answer += len(answer)
        total_context += len(data["context"])
        for start, end, label in data.get("labels", []):
            if int(label) == 1:
                total_flagged += end - start
        vocabulary.update(word.strip(".,;:!?—–").lower() for word in answer.split())
    count = len(pairs)
    return {
        "pairs": count,
        "kinds": kinds,
        "hallucination_rate": round(kinds.get("faithful", 0) / count, 4) if count else 0.0,
        "mean_answer_chars": round(total_answer / count, 1) if count else 0.0,
        "mean_context_chars": round(total_context / count, 1) if count else 0.0,
        "flagged_share_of_answers": round(total_flagged / total_answer, 4) if total_answer else 0.0,
        "vocabulary": len(vocabulary),
        "dataset_version": DATASET_VERSION,
    }


def main() -> None:  # pragma: no cover - утилита
    import argparse

    parser = argparse.ArgumentParser(description="Генерация корпуса пар «контекст — ответ»")
    parser.add_argument("--n", type=int, default=240)
    parser.add_argument("--seed", type=int, default=1312)
    parser.add_argument("--rate", type=float, default=0.5, help="доля недостоверных ответов")
    parser.add_argument("--out", default="data/demo_pairs.jsonl")
    args = parser.parse_args()

    pairs = generate_pairs(args.n, seed=args.seed, hallucination_rate=args.rate)
    path = write_pairs(pairs, args.out)
    print(f"Записано: {path}")
    for key, value in corpus_statistics(pairs).items():
        print(f"  {key}: {value}")


if __name__ == "__main__":  # pragma: no cover
    main()
