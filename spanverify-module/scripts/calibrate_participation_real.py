"""Перекалибровка доли участия ИИ на реальных текстах (пункт 2.6 промта).

Машинная часть смеси — ответы реальной модели (пары корпуса A2/A3), человеческая —
дословные предложения реальных актов из ``sources/``. Доля смеси известна по
построению: 0 / 0.25 / 0.5 / 0.75 / 1. Считаются AUC вне фолдов и средняя
абсолютная ошибка оценки доли; в ``participation.calibrated_on`` записывается
честное описание, на чём именно калибровалось.

Запуск::

    python scripts/calibrate_participation_real.py --sources data/corpus_a3/sources \\
        --pairs data/corpus_a3/pairs.jsonl --mode hf --seed 42 --out config/participation.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spanverify import Verifier  # noqa: E402
from spanverify.dataset import read_pairs  # noqa: E402
from spanverify.participation import ParticipationModel, token_rows  # noqa: E402

FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


def human_sentences(sources_dir: Path, min_chars: int = 60) -> list[str]:
    """Дословные предложения реальных актов — «человеческая» часть смеси."""
    sentences: list[str] = []
    for path in sorted(sources_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for line in text.replace("\n", " ").split(". "):
            candidate = line.strip()
            if len(candidate) >= min_chars:
                sentences.append(candidate + ".")
    return sentences


def machine_sentences(pairs_path: Path, limit: int = 400) -> list[str]:
    """Ответы реальной модели/корпуса — «машинная» часть смеси."""
    sentences: list[str] = []
    if not pairs_path.exists():
        return sentences
    for pair in read_pairs(pairs_path):
        answer = str(pair.get("answer", ""))
        for part in answer.replace("\n", " ").split(". "):
            candidate = part.strip()
            if len(candidate) >= 40:
                sentences.append(candidate + ".")
        if len(sentences) >= limit:
            break
    return sentences


def build_samples(
    human: Sequence[str],
    machine: Sequence[str],
    count: int,
    seed: int,
) -> list[dict]:
    """Смеси с известной долей ИИ: предложения двух источников перемешиваются."""
    rng = random.Random(seed)
    samples: list[dict] = []
    if not human or not machine:
        return samples
    for _ in range(count):
        fraction = rng.choice(FRACTIONS)
        total = rng.randint(4, 6)
        ai_count = int(round(fraction * total))
        parts = [(1, rng.choice(machine)) for _ in range(ai_count)]
        parts += [(0, rng.choice(human)) for _ in range(total - ai_count)]
        rng.shuffle(parts)
        text_parts: list[str] = []
        spans: list[tuple[int, int, int]] = []
        offset = 0
        for label, sentence in parts:
            spans.append((offset, offset + len(sentence), label))
            text_parts.append(sentence)
            offset += len(sentence) + 1
        samples.append(
            {
                "text": " ".join(text_parts),
                "context": rng.choice(human),
                "spans": spans,
                "ai_fraction": fraction,
            }
        )
    return samples


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Калибровка доли участия ИИ на реальных текстах")
    parser.add_argument("--sources", type=Path, default=ROOT / "data" / "corpus_a3" / "sources")
    parser.add_argument("--pairs", type=Path, default=ROOT / "data" / "corpus_a3" / "pairs.jsonl")
    parser.add_argument("--mode", choices=["demo", "hf"], default="demo")
    parser.add_argument("--model", default=None)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ROOT / "config" / "participation.json")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "participation_real.json")
    args = parser.parse_args(argv)

    started = time.time()
    human = human_sentences(args.sources)
    machine = machine_sentences(args.pairs)
    if not human or not machine:
        print(
            f"ошибка: не хватает текстов (человеческих {len(human)}, машинных {len(machine)}); "
            "калибровка на синтетике не подменяется молча",
            file=sys.stderr,
        )
        return 2

    samples = build_samples(human, machine, args.samples, args.seed)
    verifier = Verifier(mode=args.mode, model_name=args.model)
    rows: list[list[float]] = []
    labels: list[int] = []
    groups: list[int] = []
    fractions: list[float] = []
    for index, sample in enumerate(samples):
        matrix = verifier.features_for(sample["text"], sample["context"])
        token_rows_list, tokens = token_rows(sample["text"], matrix)
        for row, token in zip(token_rows_list, tokens, strict=False):
            label = 0
            for start, end, span_label in sample["spans"]:
                if start <= token.start < end:
                    label = span_label
                    break
            rows.append(list(row))
            labels.append(int(label))
            groups.append(index)
            fractions.append(float(sample["ai_fraction"]))
    if not rows or len(set(labels)) < 2:
        print("ошибка: недостаточно размеченных строк для калибровки", file=sys.stderr)
        return 2

    model = ParticipationModel.fit(rows, labels, seed=args.seed, version="")
    model.calibrated_on = (
        f"реальные акты ({len(human)} предложений из {args.sources}) + машинная часть "
        f"({len(machine)} ответов реальной модели из {args.pairs}); смеси 0/0.25/0.5/0.75/1, "
        f"{len(samples)} текстов, режим {args.mode}"
    )
    # Средняя абсолютная ошибка оценки доли: по каждому тексту сравниваем известную
    # долю и оценку модели (вне обучения порог не подбирается — оценивается модель).
    errors: list[float] = []
    by_group: dict[int, list[int]] = {}
    for index, _row in enumerate(rows):
        by_group.setdefault(groups[index], []).append(index)
    for _index, indices in sorted(by_group.items()):
        group_rows = [rows[i] for i in indices]
        scores = model.score_rows(group_rows)
        if not scores:
            continue
        errors.append(abs(sum(scores) / len(scores) - fractions[indices[0]]))
    mae = round(sum(errors) / len(errors), 4) if errors else None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(model.to_dict(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    report = {
        "mode": args.mode,
        "model": args.model,
        "seed": args.seed,
        "samples": len(samples),
        "token_rows": len(rows),
        "human_sentences": len(human),
        "machine_sentences": len(machine),
        "auc_out_of_fold": model.auc_out_of_fold,
        "mean_absolute_error": mae,
        "calibrated_on": model.calibrated_on,
        "duration_s": round(time.time() - started, 1),
        "written_to": str(args.out),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
