"""Синтетический демонстрационный корпус (для проверки конвейера и тестов).

Генератор собирает смешанные документы: чередование «человеческих» и
«машинных» фрагментов с известной разметкой по символам.

Различие классов задано теми же свойствами, которые ищет детектор:

* машинный фрагмент — узкий повторяющийся словарь терминов, шаблонные
  связки, ровная структура предложений;
* человеческий фрагмент — уникальная конкретика (фамилии, числа, даты,
  устройства), неровная структура, единичные упоминания терминов.

ВАЖНО. Корпус синтетический, и обе группы текстов написаны одним
генератором. Метрики на нём проверяют только то, что конвейер считает и
размечает; они НЕ являются оценкой качества на реальных документах.
Боевая калибровка — на размеченном корпусе, см. README.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

__all__ = [
    "make_ai_paragraph",
    "make_human_paragraph",
    "make_mixed_document",
    "generate_dataset",
    "write_dataset",
    "read_dataset",
    "dataset_statistics",
    "DATASET_VERSION",
]

DATASET_VERSION = "1.1-synthetic"

# ---------- словари машинного стиля ----------

AI_OPENERS = [
    "Важно отметить, что",
    "Следует отметить, что",
    "Таким образом,",
    "Кроме того,",
    "В рамках данного подхода",
    "На основе проведённого анализа",
    "В современном мире",
    "Прежде всего необходимо подчеркнуть, что",
    "Необходимо отметить, что",
]

AI_SUBJECTS = [
    "данный метод",
    "предложенный подход",
    "рассматриваемый механизм",
    "данная архитектура",
    "предложенное решение",
    "указанный алгоритм",
    "данный модуль",
]

AI_PREDICATES = [
    "обеспечивает эффективное решение поставленной задачи",
    "позволяет оптимизировать ключевые процессы",
    "обеспечивает повышение общей эффективности",
    "позволяет существенно улучшить качество результатов",
    "обеспечивает устойчивое функционирование системы",
    "позволяет достичь высоких показателей качества",
]

AI_TAILS = [
    "что подтверждает эффективность предложенного решения",
    "что обеспечивает высокое качество получаемых результатов",
    "что является ключевым фактором успешной реализации",
    "что позволяет достичь поставленных целей",
    "что свидетельствует о высокой эффективности подхода",
    "что обеспечивает комплексное решение задачи",
]

AI_EXTRA = [
    "Ключевой аспект заключается в комплексной оптимизации параметров.",
    "Особое внимание следует уделить повышению эффективности процессов.",
    "Данное решение позволяет оптимизировать ключевые характеристики системы.",
    "Предложенный подход представляет собой комплексное решение задачи.",
    "Существенное значение имеет устойчивое функционирование механизма.",
]

# ---------- словари человеческого стиля ----------

HUMAN_OPENERS = [
    "Вчера на семинаре",
    "По моим наблюдениям,",
    "Честно говоря,",
    "Тут вышла заминка:",
    "Мы прогнали три прогона,",
    "Пока неясно, но",
    "Пётр предложил иначе —",
    "В черновике от {date}",
    "На созвоне {name} сказал(а), что",
    "Я перечитал(а) письмо {name} —",
    "Вечером пришёл ответ от {name}:",
]

HUMAN_CLAUSES = [
    "цифры разошлись примерно на {pct} % между прогонами",
    "на ноутбуке это считалось {mins} минут, а на кластере — {secs} секунд",
    "половина датасета ({names}) оказалась с битыми подписями аугментаций",
    "пришлось вручную перепроверять {num} примеров",
    "ошибка вылезала только при batch_size={bs}, что странно",
    "гипотезу пришлось отбросить: корреляция оказалась {corr}",
    "формулу ({ver}) я так и не проверил до конца",
    "спасибо рецензенту за замечание про утечку данных",
    "ноутбук {device} перегрелся на {temp} градусах уже к обеду",
    "договорились созвониться в {time} и сверить таблицы",
    "в отчёте {name} не сходятся суммы в столбце «{col}»",
    "мы переименовали {old} в {new}, чтобы не путаться",
    "замеры с {device} дали разброс {corr} по трём сессиям",
]

HUMAN_CLOSERS = [
    "надо будет переделать до пятницы",
    "запишу это в блокнот, чтобы не забыть",
    "похоже, дело в предобработке",
    "пока оставлю как есть",
    "спрошу у {name}, она(он) это уже делал(а)",
    "в общем, непонятная история",
    "вынесу в отдельный тикет на {ticket}",
    "перепроверю завтра утром",
]

NEUTRAL = [
    "Оборудование установили в корпусе {letter}, аудитория {room}.",
    "Заседание назначено на вторник, {time}.",
    "Отчёт сдали в срок, замечаний не было.",
    "Средняя температура в помещении держалась около {temp} градусов.",
    "Копию отправили в бухгалтерию вместе с актом.",
]

NAMES = [
    "Лариса Петровна",
    "Игорь Савельев",
    "Марина Чух",
    "Артём Долин",
    "Ольга Ким",
    "Дмитрий Абаренов",
    "Настя Величко",
    "Сергей Тюрин",
    "Юля Марченко",
    "Вадим Осокин",
    "Костя Жуков",
    "Аня Стрельцова",
    "Роберт Ким",
    "Илья Бахметьев",
    "Вера Соколова",
]
DEVICES = ["кластер «Пирогов»", "сервер node-07", "рабочая станция WS-3", "стенд А-2", "GPU-нода v100-3"]
COLUMNS = ["Прогноз", "Факт", "Отклонение", "Комментарий", "Итого"]
TICKETS = ["TURB-{}, ".format(i) for i in range(100, 140)]  # noqa: UP032


def _fill(template: str, rng: random.Random) -> str:
    """Подставить в шаблон уникальную конкретику."""
    values = {
        "name": rng.choice(NAMES).split()[0],
        "full_name": rng.choice(NAMES),
        "date": f"{rng.randint(1, 28)}.0{rng.randint(1, 9)}",
        "pct": f"{rng.uniform(1.5, 19.0):.1f}".replace(".", ","),
        "mins": rng.randint(20, 90),
        "secs": rng.randint(45, 400),
        "num": rng.choice([200, 240, 317, 512, 660, 1024]),
        "bs": rng.choice([3, 7, 11, 16]),
        "corr": f"0,{rng.randint(10, 89)}",
        "ver": f"{rng.randint(1, 4)}.{rng.randint(1, 9)}",
        "device": rng.choice(DEVICES),
        "temp": rng.randint(58, 88),
        "time": f"{rng.randint(9, 18)}:{rng.choice(['00', '15', '30', '45'])}",
        "col": rng.choice(COLUMNS),
        "old": rng.choice(["scores_v2", "train_final", "exp3", "raw_labels"]),
        "new": rng.choice(["scores_v3", "train_clean", "exp4", "labels_fixed"]),
        "ticket": rng.choice(TICKETS),
        "letter": rng.choice(["А", "Б", "В", "Г"]),
        "room": rng.randint(101, 480),
        "names": ", ".join(rng.sample(NAMES, 2)),
    }
    try:
        return template.format(**values)
    except (KeyError, IndexError):
        return template


def _ai_sentence(rng: random.Random) -> str:
    opener = rng.choice(AI_OPENERS)
    subject = rng.choice(AI_SUBJECTS)
    predicate = rng.choice(AI_PREDICATES)
    tail = rng.choice(AI_TAILS)
    if rng.random() < 0.2:
        return f"{opener} {rng.choice(AI_EXTRA)}"
    body = f"{subject} {predicate}"
    return f"{opener} {body}, {tail}."


def _pick_unused(rng: random.Random, pool: list[str], used: set[str]) -> str:
    """Выбрать шаблон, не использованный ранее в этом документе.

    Человек в одном документе не повторяет одну и ту же конструкцию
    дважды — иначе синтетический «человеческий» текст начинает выглядеть
    шаблоннее, чем он бывает в жизни.
    """
    available = [item for item in pool if item not in used]
    if not available:
        used.clear()
        available = list(pool)
    choice = rng.choice(available)
    used.add(choice)
    return choice


def _human_sentence(rng: random.Random, used: dict[str, set[str]] | None = None) -> str:
    used = used if used is not None else {}
    openers = used.setdefault("openers", set())
    clauses = used.setdefault("clauses", set())
    closers = used.setdefault("closers", set())
    opener = _fill(_pick_unused(rng, HUMAN_OPENERS, openers), rng)
    clause = _fill(_pick_unused(rng, HUMAN_CLAUSES, clauses), rng)
    closer = _fill(_pick_unused(rng, HUMAN_CLOSERS, closers), rng)
    if rng.random() < 0.35:
        return f"{opener} {clause}."
    if rng.random() < 0.5:
        return f"{opener} {clause} — {closer}."
    return f"{opener} {clause}; плюс ко всему, {closer}."


def _neutral_sentence(rng: random.Random) -> str:
    return _fill(rng.choice(NEUTRAL), rng)


def make_ai_paragraph(rng: random.Random | None = None, sentences: int = 4) -> str:
    rng = rng or random.Random()
    return " ".join(_ai_sentence(rng) for _ in range(max(1, sentences)))


def make_human_paragraph(rng: random.Random | None = None, sentences: int = 4) -> str:
    rng = rng or random.Random()
    used: dict[str, set[str]] = {}
    return " ".join(_human_sentence(rng, used) for _ in range(max(1, sentences)))


def make_mixed_document(
    rng: random.Random | None = None,
    segments: int = 5,
    min_sentences: int = 2,
    max_sentences: int = 4,
    neutral_probability: float = 0.2,
) -> dict:
    """Документ из чередующихся фрагментов с разметкой по символам.

    Возвращает словарь ``{"text": str, "labels": [[start, end, label], ...]}``,
    где ``label`` = 1 для машинного фрагмента и 0 для человеческого.
    """
    rng = rng or random.Random()
    parts: list[str] = []
    labels: list[list[int]] = []
    cursor = 0
    label = rng.choice([0, 1])

    used: dict[str, set[str]] = {}
    for _ in range(max(1, segments)):
        count = rng.randint(min_sentences, max_sentences)
        if label == 1:
            sentences = [_ai_sentence(rng) for _ in range(count)]
        else:
            sentences = [_human_sentence(rng, used) for _ in range(count)]
        if rng.random() < neutral_probability:
            sentences.insert(rng.randrange(len(sentences) + 1), _neutral_sentence(rng))
        chunk = " ".join(sentences)
        if parts:
            parts.append(" ")
            cursor += 1
        start = cursor
        parts.append(chunk)
        cursor += len(chunk)
        labels.append([start, cursor, label])
        label = 1 - label

    return {"text": "".join(parts), "labels": labels}


def generate_dataset(
    n_documents: int = 240,
    seed: int = 1312,
    segments: int = 5,
) -> list[dict]:
    """Список документов с разметкой (id, text, labels)."""
    rng = random.Random(seed)
    documents: list[dict] = []
    for i in range(n_documents):
        doc = make_mixed_document(rng, segments=segments)
        documents.append({"id": f"demo-{i:04d}", "text": doc["text"], "labels": doc["labels"]})
    return documents


def write_dataset(documents: list[dict], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for doc in documents:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
    return p


def read_dataset(path: str | Path) -> Iterator[dict]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def dataset_statistics(documents: list[dict]) -> dict:
    """Краткая статистика корпуса: размер, доля машинного текста, словарь."""
    total_chars = total_ai = 0
    vocab: Counter[str] = Counter()
    for doc in documents:
        text = doc["text"]
        total_chars += len(text)
        for start, end, label in doc.get("labels", []):
            if int(label) == 1:
                total_ai += end - start
        for word in text.lower().split():
            vocab[word.strip(".,;:!?—–()»«\"'")] += 1
    return {
        "documents": len(documents),
        "characters": total_chars,
        "ai_share_chars": round(total_ai / total_chars, 4) if total_chars else 0.0,
        "vocabulary": len(vocab),
        "mean_document_chars": round(total_chars / len(documents), 1) if documents else 0.0,
        "dataset_version": DATASET_VERSION,
    }


def main() -> None:  # pragma: no cover - утилита CLI
    import argparse

    parser = argparse.ArgumentParser(description="Генерация демонстрационного корпуса")
    parser.add_argument("--n", type=int, default=240, help="число документов")
    parser.add_argument("--seed", type=int, default=1312)
    parser.add_argument("--segments", type=int, default=5)
    parser.add_argument("--out", default="data/demo_dataset.jsonl")
    args = parser.parse_args()

    docs = generate_dataset(args.n, seed=args.seed, segments=args.segments)
    path = write_dataset(docs, args.out)
    print(f"Записано: {path}")
    for key, value in dataset_statistics(docs).items():
        print(f"  {key}: {value}")


if __name__ == "__main__":  # pragma: no cover
    main()
