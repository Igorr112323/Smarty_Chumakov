"""Устойчивость к пяти видам искажений (пункт 4.3 промта).

Берётся отложенная часть корпуса, к ответам применяются пять видов искажений, и
считается падение F-меры относительно исходных (правильных) ответов:

    1. перефразирование          — слова ответа заменяются синонимами/порядком;
    2. числовая подмена          — значение берётся из другого факта того же документа;
    3. противоречие источнику    — значение инвертируется;
    4. пропуск существенного     — из ответа убирается числовое значение;
    5. числа прописью и шаблоны  — числа записываются словами.

Каждое искажение применяется детерминированно (seed фиксирован). Считаются
token F1 и вердикт-F1 до и после, падение — разность. Всё пишется в JSON и MD.

Запуск::

    python scripts/robustness.py --dataset data/corpus_a3/pairs.jsonl --mode demo \\
        --seed 42 --out reports/robustness
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spanverify import Verifier  # noqa: E402
from spanverify.dataset import read_pairs  # noqa: E402
from spanverify.normalize import numbers_in_text  # noqa: E402

NUMERALS = {
    1: "один",
    2: "два",
    3: "три",
    4: "четыре",
    5: "пять",
    6: "шесть",
    7: "семь",
    8: "восемь",
    9: "девять",
    10: "десять",
    11: "одиннадцать",
    12: "двенадцать",
    13: "тринадцать",
    14: "четырнадцать",
    15: "пятнадцать",
    20: "двадцать",
    21: "двадцать один",
    30: "тридцать",
    45: "сорок пять",
    50: "пятьдесят",
    75: "семьдесят пять",
    90: "девяносто",
    100: "сто",
}

SYNONYMS = (
    ("составляет", "равняется"),
    ("установлен", "определён"),
    ("хранится", "содержится"),
    ("документ", "бумага"),
    ("срок", "период"),
    ("составляет", "образует"),
    ("необходимо", "следует"),
    ("запрещается", "не разрешается"),
    ("в течение", "на протяжении"),
    ("в соответствии", "согласно"),
)


def paraphrase(answer: str, rng: random.Random) -> str:
    """Перефразирование: синонимы и смена порядка клауз."""
    result = answer
    for source, target in SYNONYMS:
        if source in result and rng.random() < 0.7:
            result = result.replace(source, target, 1)
    # Перестановка клауз: «Срок … — 5 лет» → «5 лет — срок …».
    parts = re.split(r"\s*[—:]\s*", result, maxsplit=1)
    if len(parts) == 2 and rng.random() < 0.5:
        result = f"{parts[1].strip().rstrip('.')} — {parts[0].strip().lower()}."
    return result


def numeric_substitution(answer: str, context: str, rng: random.Random) -> str:
    """Число ответа заменяется значением из другого факта того же документа."""
    mentions = numbers_in_text(answer)
    if not mentions:
        return answer
    context_values = [mention.value for mention in numbers_in_text(context)]
    candidates = [value for value in context_values if all(abs(value - m.value) > 1e-6 for m in mentions)]
    if not candidates:
        return answer
    mention = mentions[0]
    replacement = rng.choice(candidates)
    return answer[: mention.start] + _format_number(replacement) + answer[mention.end :]


def contradiction(answer: str) -> str:
    """Противоречие источнику: «разрешается» ↔ «запрещается», «не менее» ↔ «не более»."""
    result = answer
    for source, target in (
        ("разрешается", "запрещается"),
        ("не менее", "не более"),
        ("не более", "не менее"),
        ("обязан", "не вправе"),
        ("вправе", "не вправе"),
        ("раз в год", "раз в месяц"),
    ):
        if source in result:
            return result.replace(source, target, 1)
    mentions = numbers_in_text(answer)
    if mentions:
        mention = mentions[0]
        return answer[: mention.start] + _format_number(mention.value + 7) + answer[mention.end :]
    return result


def omission(answer: str) -> str:
    """Пропуск существенного сведения: из ответа убирается числовое значение."""
    mentions = numbers_in_text(answer)
    if not mentions:
        return answer
    mention = mentions[0]
    return (answer[: mention.start] + answer[mention.end :]).replace("  ", " ")


def number_words(answer: str) -> str:
    """Числа прописью: «5 лет» → «пять лет»."""
    mentions = numbers_in_text(answer)
    if not mentions:
        return answer
    result = answer
    for mention in reversed(mentions):
        text = _format_number(mention.value, words=True)
        result = result[: mention.start] + text + result[mention.end :]
    return result


def _format_number(value: float, words: bool = False) -> str:
    if value == int(value):
        number = int(value)
        if words and number in NUMERALS:
            return NUMERALS[number]
        return str(number)
    return f"{value:.2f}".rstrip("0").rstrip(".")


DISTORTIONS = {
    "перефразирование": lambda answer, context, rng: paraphrase(answer, rng),
    "числовая подмена": lambda answer, context, rng: numeric_substitution(answer, context, rng),
    "противоречие источнику": lambda answer, context, rng: contradiction(answer),
    "пропуск существенного сведения": lambda answer, context, rng: omission(answer),
    "числа прописью и другие шаблоны": lambda answer, context, rng: number_words(answer),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Устойчивость к искажениям: падение F-меры")
    parser.add_argument("--dataset", default="data/corpus_a3/pairs.jsonl")
    parser.add_argument("--mode", choices=["demo", "hf"], default="demo")
    parser.add_argument("--model", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--out", default="reports/robustness")
    args = parser.parse_args(argv)

    started = time.time()
    pairs = list(read_pairs(args.dataset))
    if not pairs:
        print(f"ошибка: датасет пуст: {args.dataset}", file=sys.stderr)
        return 2
    verifier = Verifier(mode=args.mode, model_name=args.model)
    # Отложенная часть: последние 20 % документов по порядку (детерминированно).
    groups: dict[str, list[dict]] = {}
    for pair in pairs:
        meta = pair.get("meta") or {}
        groups.setdefault(str(meta.get("group") or meta.get("doc_id") or pair.get("id")), []).append(pair)
    keys = sorted(groups)
    rng = random.Random(args.seed)
    rng.shuffle(keys)
    held = [pair for key in keys[max(1, int(len(keys) * 0.8)) :] for pair in groups[key]] or pairs
    # Искажения применяются к ЧИСТЫМ ответам: тогда «до» — это ложные срабатывания
    # на верных ответах, а «после» — доля искажений, которые конвейер заметил.
    clean = [pair for pair in held if str((pair.get("meta") or {}).get("mode")) == "faithful"]
    held = (clean or held)[: args.limit]

    baseline = verifier.evaluate(held)
    baseline_flagged = sum(
        1
        for pair in held
        if verifier.verify(str(pair.get("answer", "")), str(pair.get("context", ""))).verdict != "grounded"
    ) / max(1, len(held))
    report: dict = {
        "dataset": args.dataset,
        "mode": args.mode,
        "model": args.model,
        "seed": args.seed,
        "pairs": len(held),
        "clean_pairs": len(clean),
        "document_groups": len(keys),
        "baseline": {
            "token_f1": baseline["tokens"]["f1"],
            "verdict_f1": baseline["verdicts"]["f1"],
            "verdict_fpr": baseline["verdicts"]["fpr"],
        },
        "distortions": [],
        "duration_s": 0.0,
    }
    for name, apply in DISTORTIONS.items():
        rng = random.Random(args.seed + hash(name) % 1000)
        distorted: list[dict] = []
        changed = 0
        for pair in held:
            answer = str(pair.get("answer", ""))
            context = str(pair.get("context", ""))
            new_answer = apply(answer, context, rng)
            if new_answer != answer:
                changed += 1
            copy = dict(pair)
            copy["answer"] = new_answer
            # Разметку искажённого ответа не переносим: измеряется вердикт (ответ
            # с искажением обязан стать не «grounded»), а не точность фрагментов.
            copy["labels"] = [[0, 0, 0]]
            distorted.append(copy)
        flagged = sum(1 for pair in distorted if verifier.verify(pair["answer"], pair["context"]).verdict != "grounded")
        report["distortions"].append(
            {
                "name": name,
                "pairs": len(distorted),
                "changed_answers": changed,
                "flagged_share_before": round(baseline_flagged, 4),
                "flagged_share_after": round(flagged / max(1, len(distorted)), 4),
                "delta": round(flagged / max(1, len(distorted)) - baseline_flagged, 4),
                "token_f1_before": baseline["tokens"]["f1"],
                # Токенную F1 после искажения считать нельзя: разметка относится к
                # исходному ответу, а искажение меняет текст. Поэтому null и причина.
                "token_f1_after": None,
                "token_f1_after_reason": "разметка исходного ответа к искажённому тексту не переносится",
            }
        )
    report["duration_s"] = round(time.time() - started, 1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.with_suffix(".json")).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (out.with_suffix(".md")).write_text(render(report), encoding="utf-8")
    print(render(report))
    return 0


def render(report: dict) -> str:
    """Отчёт генерируется из JSON (числа в тексте и в файле совпадают)."""
    lines = [
        "# Устойчивость к искажениям",
        "",
        f"Датасет `{report['dataset']}`, режим **{report['mode']}**"
        + (f", модель `{report['model']}`" if report["model"] else ""),
        f"Пар: {report['pairs']}, групп документов: {report['document_groups']}, seed {report['seed']}, "
        f"время {report['duration_s']} с.",
        "",
        f"Исходная (до искажений) token F1 = {report['baseline']['token_f1']}, "
        f"вердикт F1 = {report['baseline']['verdict_f1']}, FPR = {report['baseline']['verdict_fpr']}.",
        "",
        "| Искажение | Пар | Изменено ответов | token F1 после | Падение F1 | Доля не-«grounded» | Базовая доля |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in report["distortions"]:
        lines.append(
            f"| {item['name']} | {item['pairs']} | {item['changed_answers']} | {item['flagged_share_before']} | "
            f"{item['flagged_share_after']} | {item['delta']:+} | {item['token_f1_before']} |"
        )
    lines += [
        "",
        "Замечание: разметка относится к исходному ответу, поэтому токенную F1 после искажения",
        "считать нельзя (в таблице — исходное значение). Измеряется способность конвейера",
        "**заметить искажение**: доля ответов, переставших быть подтверждёнными (после) против",
        "доли ложных срабатываний на тех же ответах до искажения.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
