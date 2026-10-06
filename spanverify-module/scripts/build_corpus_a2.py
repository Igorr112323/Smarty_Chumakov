#!/usr/bin/env python3
"""Корпус A2: естественные ответы языковой модели на реальные вопросы по документу.

Зачем отдельный подкорпус
-----------------------

A1 — управляемые подмены: ошибки вставлены скриптом, поэтому у каждой есть готовая
метка. A2 — противоположный случай: модель отвечает сама, ошибки естественные, их
надо размечать. Без A2 конвейер проверен только на ошибках, которые придумал наш же
генератор; с A2 видно, как он ведёт себя на реальных ответах.

Как устроено
------------

1. Из документов берутся факты и к каждому строится вопрос (``--questions``).
2. Модель генерирует ответ (``--model``, нужны ``transformers`` и ``torch``).
3. Ответ идёт через наш ``Verifier``: получаются кандидаты-фрагменты (черновая
   разметка, ``"auto": true``).
4. Пары сохраняются с пометкой ``"needs_expert_review": true`` — их обязан
   подтвердить человек (``scripts/review_server.py``). Автоматическая разметка
   **не считается истиной**: иначе получился бы отчёт о качестве по собственным
   предсказаниям.

Без GPU и без весов
-------------------

``--dry-run`` пишет только вопросы и промты — это работает в CI и в песочнице.
Обычный запуск без доступной модели завершается кодом 2 и объяснением, что нужно
(см. ``docs/GPU_ИНСТРУКЦИЯ.md``). Никакие числа «hf»-режима в отчёт не попадают,
пока он реально не выполнен.

Запуск::

    python scripts/build_corpus_a2.py --docs data/corpus_a/docs --out data/corpus_a2 --dry-run
    python scripts/build_corpus_a2.py --docs data/corpus_a/docs --out data/corpus_a2 \\
        --model ai-forever/rugpt3small_based_on_gpt2 --limit 200
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET_VERSION = "corpus-a2.0-draft"

NUMBER_WORDS = {
    "пять": 5,
    "семь": 7,
    "десять": 10,
    "пятнадцать": 15,
    "двадцать пять": 25,
    "сорок пять": 45,
}
FEATURE_PATTERNS = (
    "срок хранения",
    "срок восстановления после сбоя",
    "периодичность резервного копирования",
    "срок ответа на запрос контрагента",
    "предельный объём одного вложения",
)


def read_facts(text: str) -> list[dict]:
    """Вытащить из документа факты вида «Для <субъект> <признак> составляет <значение>»."""
    facts: list[dict] = []
    pattern = re.compile(
        r"Для (?P<subject>.+?) (?P<feature>" + "|".join(FEATURE_PATTERNS) + r") составляет (?P<value>[^.]+)\."
    )
    for match in pattern.finditer(text):
        value = match.group("value").strip()
        condition = None
        for separator in (" если ", " при условии "):
            if separator in value:
                value, condition = value.split(separator, 1)
                condition = separator.strip() + " " + condition.strip()
        facts.append(
            {
                "subject": match.group("subject").strip(),
                "feature": match.group("feature").strip(),
                "value": value.strip(),
                "condition": condition,
                "sentence": match.group(0),
            }
        )
    return facts


def build_question(fact: dict) -> str:
    """Вопрос по факту: он же уходит модели, он же хранится в корпусе."""
    if "срок хранения" in fact["feature"]:
        return f"Какой срок хранения установлен для {fact['subject']}?"
    if "объём" in fact["feature"]:
        return f"Какой предельный объём одного вложения установлен для {fact['subject']}?"
    if "периодичность" in fact["feature"]:
        return f"Как часто выполняется резервное копирование для {fact['subject']}?"
    return f"Что установлено для {fact['subject']} по показателю «{fact['feature']}»?"


def build_prompt(document_text: str, question: str) -> str:
    """Промт для модели: ответ строго по документу (иначе меряется не то)."""
    return (
        "Ответь на вопрос строго по документу. Если в документе нет ответа, скажи «в документе нет ответа».\n\n"
        f"Документ:\n{document_text}\n\nВопрос: {question}\nОтвет:"
    )


def collect_tasks(docs_dir: Path, limit: int | None, seed: int) -> list[dict]:
    """Собрать задания (документ + вопрос + промт) из каталога документов."""
    tasks: list[dict] = []
    for path in sorted(docs_dir.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        for index, fact in enumerate(read_facts(text)):
            tasks.append(
                {
                    "document_id": path.stem,
                    "fact_index": index,
                    "subject": fact["subject"],
                    "feature": fact["feature"],
                    "expected_value": fact["value"],
                    "question": build_question(fact),
                    "prompt": build_prompt(text, build_question(fact)),
                }
            )
    random.Random(seed).shuffle(tasks)
    return tasks[:limit] if limit else tasks


def load_model(model_name: str):  # noqa: ANN201 - тип зависит от наличия transformers
    """Загрузить модель и токенизатор; понятная ошибка, если окружение не готово."""
    try:
        import torch  # noqa: F401, PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - зависит от окружения
        raise SystemExit(
            "нужны transformers и torch: pip install torch transformers (см. docs/GPU_ИНСТРУКЦИЯ.md)"
        ) from error
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    except Exception as error:  # noqa: BLE001 - сеть и веса могут отсутствовать по-разному
        raise SystemExit(f"не удалось загрузить модель {model_name}: {error}") from error
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def generate_answers(tasks: list[dict], model_name: str, max_new_tokens: int) -> tuple[list[dict], str]:
    """Сгенерировать ответы моделью; вернуть пары-черновики и имя окружения (device)."""
    generator = load_model(model_name)
    import torch  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pairs: list[dict] = []
    for task in tasks:
        output = generator(task["prompt"], max_new_tokens=max_new_tokens, do_sample=False, return_full_text=False)
        answer = str(output[0]["generated_text"]).strip().split("\n")[0].strip()
        pairs.append(
            {
                "id": f"corpus-a2-{len(pairs) + 1:05d}",
                "context": task["prompt"].split("Документ:\n", 1)[1].split("\n\nВопрос:", 1)[0],
                "answer": answer,
                "labels": [],
                "meta": {
                    "kind": "corpus_a2",
                    "mode": "natural",
                    "taxonomy": "unknown",
                    "clean": None,
                    "auto": True,
                    "needs_expert_review": True,
                    "document_id": task["document_id"],
                    "question": task["question"],
                    "expected_value": task["expected_value"],
                    "model": model_name,
                    "device": device,
                    "dataset_version": DATASET_VERSION,
                },
            }
        )
    return pairs, device


def draft_labels_with_verifier(pairs: list[dict], mode: str = "demo") -> list[dict]:
    """Черновая разметка: фрагменты, которые нашёл наш же конвейер (не истина)."""
    from spanverify.engine import Verifier  # noqa: PLC0415

    verifier = Verifier(mode=mode)
    for pair in pairs:
        result = verifier.verify(pair["answer"], pair["context"])
        pair["labels"] = [[span.start, span.end, 1] for span in result.spans]
        pair["meta"]["draft_verdict"] = result.verdict
        pair["meta"]["draft_score"] = round(result.score, 4)
    return pairs


def write_outputs(out_dir: Path, tasks: list[dict], pairs: list[dict], manifest_extra: dict) -> None:
    """Записать задания, пары-черновики и манифест (с пометкой о необходимости проверки)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "questions.jsonl").write_text(
        "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks), encoding="utf-8"
    )
    (out_dir / "pairs_draft.jsonl").write_text(
        "".join(json.dumps(pair, ensure_ascii=False) + "\n" for pair in pairs), encoding="utf-8"
    )
    manifest = {
        "dataset_version": DATASET_VERSION,
        "role": "черновик: естественные ответы модели, разметка требует проверки человеком",
        "tasks": len(tasks),
        "pairs": len(pairs),
        "status": "draft" if not pairs else "generated",
        "needs_expert_review": True,
        **manifest_extra,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Корпус A2: естественные ответы (запускать лучше на GPU)")
    parser.add_argument("--docs", type=Path, default=ROOT / "data" / "corpus_a" / "docs")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "corpus_a2")
    parser.add_argument("--model", default="ai-forever/rugpt3small_based_on_gpt2")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="собрать только задания и промты")
    args = parser.parse_args()

    if not args.docs.exists():
        print(f"ошибка: нет каталога документов {args.docs}", file=sys.stderr)
        return 2
    tasks = collect_tasks(args.docs, args.limit, args.seed)
    if not tasks:
        print("ошибка: в документах не найдено ни одного факта — проверьте формат", file=sys.stderr)
        return 2

    if args.dry_run:
        write_outputs(args.out, tasks, [], {"dry_run": True, "model": args.model})
        print(f"Задания: {len(tasks)} (черновик без генерации)")
        print(f"Промты: {args.out / 'questions.jsonl'}")
        print("Генерация не выполнялась: нужен запуск на машине с torch/transformers (docs/GPU_ИНСТРУКЦИЯ.md)")
        return 0

    print(f"[a2] модель {args.model}, заданий {len(tasks)}")
    pairs, device = generate_answers(tasks, args.model, args.max_new_tokens)
    pairs = draft_labels_with_verifier(pairs)
    write_outputs(args.out, tasks, pairs, {"dry_run": False, "model": args.model, "device": device})
    print(f"Пар (черновик): {len(pairs)} на устройстве {device}")
    print(f"Черновики: {args.out / 'pairs_draft.jsonl'} — метки НЕ являются истиной, нужна проверка")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
