"""Шаблон разметки для реальных документов.

    python scripts/make_markup_template.py --out data/мои_пары.jsonl --count 30

Скрипт создаёт файл с заготовками: контекст и ответ пустые, разметка пустая.
Заполните пары своими документами и разметьте недостоверные фрагменты ответа —
диапазонами символов `[начало, конец, 1]`. После этого можно обучать:

    python -m spanverify train --dataset data/мои_пары.jsonl --out config/weights.json
    python -m spanverify evaluate --dataset data/мои_пары.jsonl

Формат строки (по одной паре на строку, JSON Lines):

    {"id": "pair-00001", "context": "текст документа",
     "answer": "текст ответа", "labels": [[20, 23, 1]], "meta": {"source": "мой документ"}}

Правила разметки:

* `labels` — только недостоверные фрагменты; правильные места не размечаются;
* диапазоны указываются **по символам ответа**, нумерация с нуля;
* если в ответе всё верно — `"labels": []`;
* не размечайте пунктуацию и отдельные служебные слова: конвейер их не оценивает.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXAMPLE_CONTEXT = "Регламент 343: срок хранения первичных документов составляет 10 лет."
EXAMPLE_ANSWER = "Срок хранения первичных документов составляет 3 года."


def main(argv: list[str] | None = None) -> int:
    """Записать шаблон разметки и подсказку по формату."""
    parser = argparse.ArgumentParser(description="Шаблон разметки реальных пар")
    parser.add_argument("--out", default="data/my_pairs.jsonl")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument(
        "--with-examples", action="store_true", default=True, help="оставить одну заполненную пару как образец"
    )
    args = parser.parse_args(argv)

    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        if args.with_examples:
            handle.write(
                json.dumps(
                    {
                        "id": "example-filled",
                        "context": EXAMPLE_CONTEXT,
                        "answer": EXAMPLE_ANSWER,
                        "labels": [[53, 54, 1]],
                        "meta": {"source": "образец: заполните своими данными"},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        for index in range(args.count):
            handle.write(
                json.dumps(
                    {
                        "id": f"pair-{index:05d}",
                        "context": "",
                        "answer": "",
                        "labels": [],
                        "meta": {"source": "заполните", "document": f"документ-{index // 5 + 1}"},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Шаблон записан: {target}")
    print("Заполните context/answer и разметку labels, затем запустите:")
    print(f"  python -m spanverify train --dataset {target} --out config/weights.json")
    return 0


if __name__ == "__main__":  # pragma: no cover - утилита
    raise SystemExit(main())
