"""Второй генератор корпуса — для кросс-корпусного теста (P0-2).

Зачем нужен второй генератор
----------------------------
Метрики на первом корпусе (``spanverify.dataset``) доказывают работоспособность
конвейера, но не перенос метода: и корпус, и признаки писал один и тот же
проект. Чтобы получить честное число, нужен **другой** корпус — с другими
шаблонами документов, другими числами (записанными словами), другими
формулировками ответов и другим набором предметов. Обучение идёт на корпусе A,
оценка — на корпусе B; ожидаемый честный результат — заметное падение метрики.

Важно: корпус B всё равно синтетический. Его число ограничивает вывод сверху:
это проверка переноса между генераторами, а не измерение на реальных
регламентах (для последнего нужны размеченные пары заказчика).
"""

from __future__ import annotations

import random
from typing import Any

__all__ = ["DATASET_VERSION", "generate_alt_pairs"]

DATASET_VERSION = "alt-generator 1.0 (числа словами, другие шаблоны)"

# Числа записаны словами — этого нет в корпусе A, поэтому проверяется и
# устойчивость к форме записи, а не только к значению.
NUM_WORDS: dict[int, str] = {
    5: "пять",
    7: "семь",
    10: "десять",
    14: "четырнадцать",
    15: "пятнадцать",
    20: "двадцать",
    25: "двадцать пять",
    30: "тридцать",
    45: "сорок пять",
    60: "шестьдесят",
    90: "девяносто",
}

SUBJECTS: tuple[tuple[str, str], ...] = (
    ("срок поверки приборов учёта", "лет"),
    ("периодичность сверки с контрагентами", "раз в год"),
    ("срок ответа на письменное обращение", "дней"),
    ("норма выдачи спецодежды", "лет"),
    ("лимит остатка наличных в кассе", "дней"),
    ("срок хранения актов выполненных работ", "лет"),
    ("частота плановых обходов помещений", "раз в год"),
    ("срок рассмотрения служебной записки", "дней"),
    ("периодичность замены фильтров вентиляции", "раз в год"),
    ("срок годности реагентов в лаборатории", "лет"),
    ("частота калибровки датчиков давления", "раз в год"),
    ("срок хранения путевых листов", "лет"),
)

DOC_PREFIXES = (
    "Внутренний порядок 41-П",
    "Инструкция по делопроизводству ИД-7",
    "Технологическая карта службы эксплуатации",
    "Положение о внутреннем контроле",
    "Распоряжение по административным вопросам",
    "Методические указания по учёту",
)

DOC_CLOSERS = (
    "Ответственный за исполнение — начальник отдела.",
    "Контроль возложен на службу внутреннего аудита.",
    "Сведения вносятся в журнал учёта.",
    "Отклонения оформляются отдельным актом.",
)

ANSWER_TEMPLATES = (
    "{subject_cap} — {value} {unit}.",
    "По внутреннему порядку {subject} — это {value} {unit}.",
    "Документ называет срок: {subject_cap} — {value} {unit}.",
    "Как указано в положении, {subject_cap} составляет {value} {unit}.",
    "{subject_cap} установлен(а) в размере {value} {unit}.",
)

FABRICATIONS = (
    "За нарушение предусмотрен штраф в размере пяти тысяч рублей.",
    "Отчётность направляется в вышестоящую организацию ежеквартально.",
    "Допускается продление срока по устному распоряжению руководителя.",
    "Проверка проводится комиссией из трёх человек с выездом на место.",
    "Ответственный обязан уведомить надзорный орган в течение суток.",
)


def _wrong_value(rng: random.Random, correct: int) -> int:
    """Подобрать другое число из того же словаря (чтобы запись осталась словами)."""
    options = [value for value in NUM_WORDS if value != correct]
    return rng.choice(options)


def generate_alt_pairs(n_pairs: int = 120, seed: int = 4242) -> list[dict[str, Any]]:
    """Сгенерировать корпус B: другой генератор, числа словами, иные шаблоны.

    Возвращает список пар в том же формате, что и корпус A (``id``, ``context``,
    ``answer``, ``labels``, ``meta``), чтобы по ним работал обычный конвейер.
    """
    rng = random.Random(seed)
    pairs: list[dict[str, Any]] = []
    for index in range(n_pairs):
        subject, unit = SUBJECTS[index % len(SUBJECTS)]
        correct = rng.choice(list(NUM_WORDS))
        wrong = _wrong_value(rng, correct)
        context = f"{rng.choice(DOC_PREFIXES)}: {subject} — {NUM_WORDS[correct]} {unit}. " f"{rng.choice(DOC_CLOSERS)}"
        kind = rng.choices(["faithful", "value_substitution", "fabrication"], weights=[0.4, 0.45, 0.15])[0]
        labels: list[list[int]] = []
        if kind == "faithful":
            answer = rng.choice(ANSWER_TEMPLATES).format(
                subject=subject,
                subject_cap=subject[0].upper() + subject[1:],
                value=NUM_WORDS[correct],
                unit=unit,
            )
        elif kind == "value_substitution":
            answer = rng.choice(ANSWER_TEMPLATES).format(
                subject=subject,
                subject_cap=subject[0].upper() + subject[1:],
                value=NUM_WORDS[wrong],
                unit=unit,
            )
            start = answer.find(NUM_WORDS[wrong])
            labels = [[start, start + len(NUM_WORDS[wrong]), 1]]
        else:
            answer = rng.choice(FABRICATIONS)
            labels = [[0, len(answer), 1]]
        pairs.append(
            {
                "id": f"alt-{index:05d}",
                "context": context,
                "answer": answer,
                "labels": labels,
                "meta": {
                    "kind": kind,
                    "subject": subject,
                    "true_value": correct,
                    "unit": unit,
                    "question": f"Что устанавливает документ про «{subject}»?",
                    "dataset_version": DATASET_VERSION,
                },
            }
        )
    return pairs
