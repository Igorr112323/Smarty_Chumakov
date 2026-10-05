"""Эксперименты: обучение, метрики на отложенной части, отчёт в reports/.

    python scripts/run_experiments.py --dataset data/demo_pairs.jsonl --out reports/experiments

Что делает:

1. обучает конвейер (перебор весов, порог маски, голова, калибровка);
2. считает сквозные метрики через публичный ``verify()`` — то же, что отдаёт API;
3. сохраняет отчёт в Markdown и JSON, а также обученные параметры.

Все числа берутся из запуска. Если корпус демонстрационный (синтетический), это
прямо печатается в отчёте: научные выводы требуют реальной разметки.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spanverify.dataset import corpus_statistics, generate_pairs, read_pairs, write_pairs  # noqa: E402
from spanverify.engine import Verifier  # noqa: E402
from spanverify.train import save_training_artifacts, train  # noqa: E402

REPORT_TEMPLATE = """# Эксперимент: {dataset}

Режим: `{mode}`. Seed: {seed}. Дата запуска: {timestamp}.
Корпус: {pairs_line}.

## Метрики по всему корпусу (сквозной путь `verify()`; `evaluate`)

| Показатель | Значение |
| --- | --- |
| token precision | {precision:.3f} |
| token recall | {recall:.3f} |
| **token F1** | **{f1:.3f}** |
| FPR по токенам | {fpr:.3f} |
| AUC по токенам | {auc:.3f} |
| F1 по ответам (порог {answer_threshold:.3f}) | {answer_f1:.3f} |
| строгий F1 фрагментов (IoU ≥ 0.5, узкая разметка) | {strict_span_f1:.3f} |
| полнота накрытия фрагментов | {containment:.3f} |
| F1 при расширенной разметке | {expanded:.3f} |

Критерий качества мастер-промта (token F1 ≥ 0.90 при FPR ≤ 0.10): **{gate}**.

## Выбранные параметры

* веса: {weights}
* порог маски: z={span_z}, floor={span_floor}, cap={span_cap}
* порог решения: {threshold:.4f} (целевой FPR {target_fpr})
* сигнал: {signal}; голова: {head}
* фолды головы (AUC): {folds}

## Оговорка

{synthetic_note}
"""


def load_records(dataset: Path, pairs: int, seed: int) -> list[dict]:
    """Прочитать корпус или сгенерировать его при отсутствии файла."""
    if dataset.is_file():
        return list(read_pairs(dataset))
    generated = generate_pairs(pairs, seed=seed)
    write_pairs(generated, dataset)
    return [pair.to_dict() for pair in generated]


def main(argv: list[str] | None = None) -> int:
    """Запустить эксперимент и сохранить отчёт."""
    parser = argparse.ArgumentParser(description="Эксперименты SpanVerify")
    parser.add_argument("--dataset", default="data/demo_pairs.jsonl")
    parser.add_argument("--out", default="reports/experiments")
    parser.add_argument("--mode", choices=["demo", "hf"], default="demo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--target-fpr", type=float, default=0.1)
    parser.add_argument("--pairs", type=int, default=240, help="сгенерировать, если файла нет")
    parser.add_argument("--model", default=None, help="модель для режима hf (имя или путь к папке с весами)")
    args = parser.parse_args(argv)

    dataset = Path(args.dataset)
    records = load_records(dataset, args.pairs, 1312)
    started = time.time()
    report = train(
        records,
        mode=args.mode,
        seed=args.seed,
        folds=args.folds,
        target_fpr=args.target_fpr,
        dataset_name=str(dataset),
    )
    verifier = Verifier(mode=args.mode, weights=report.bundle, model_name=args.model)
    metrics = verifier.evaluate(records)
    bundle = report.bundle

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = save_training_artifacts(report, out_dir / "weights.json", root=out_dir)

    tokens = metrics["tokens"]
    spans = metrics["spans"]
    answers = metrics["answers"]
    gate = "ДОСТИГНУТ" if tokens["f1"] >= 0.90 and tokens["fpr"] <= 0.10 else "НЕ достигнут"
    synthetic_note = (
        "Корпус демонстрационный (синтетические пары). Числа описывают "
        "работоспособность и самосогласованность конвейера, а не качество на "
        "реальных документах: для научных выводов нужен режим `hf` и разметка "
        "настоящих пар."
        if args.mode == "demo"
        else "Режим `hf`: признаки считаются реальной моделью. Данные всё равно "
        "синтетические, если корпус не был заменён на размеченные документы."
    )
    stats = corpus_statistics(records)
    text = REPORT_TEMPLATE.format(
        dataset=dataset,
        mode=args.mode,
        seed=args.seed,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        pairs_line=f"{stats['pairs']} пар, типы {stats['kinds']}",
        precision=tokens["precision"],
        recall=tokens["recall"],
        f1=tokens["f1"],
        fpr=tokens["fpr"],
        auc=tokens["auc"],
        answer_f1=answers["f1"],
        answer_threshold=answers["threshold"],
        containment=spans["recall_containment"],
        expanded=spans["f1_expanded_labels"],
        strict_span_f1=spans["f1"],
        gate=gate,
        weights={k: round(v, 3) for k, v in bundle.weights.items()},
        span_z=bundle.span_z,
        span_floor=bundle.span_floor,
        span_cap=bundle.span_cap,
        threshold=bundle.threshold,
        target_fpr=bundle.target_fpr,
        signal=(bundle.meta or {}).get("signal", "—"),
        head=(bundle.head or {}).get("type", "none"),
        folds=[round(fold["auc"], 3) for fold in (report.folds or [])],
        synthetic_note=synthetic_note,
    )
    markdown_path = out_dir / "experiment.md"
    markdown_path.write_text(text, encoding="utf-8")
    payload = {
        "dataset": str(dataset),
        "mode": args.mode,
        "seed": args.seed,
        "duration_s": round(time.time() - started, 2),
        "corpus": stats,
        "metrics": metrics,
        "bundle": bundle.to_dict(),
        "folds": report.folds,
        "gate": gate,
        "artifacts": {name: str(path) for name, path in written.items()},
    }
    json_path = out_dir / "experiment.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Отчёт: {markdown_path}")
    print(f"Данные: {json_path}")
    print(f"token F1={tokens['f1']:.3f} FPR={tokens['fpr']:.3f} — критерий {gate}")
    return 0


if __name__ == "__main__":  # pragma: no cover - утилита
    raise SystemExit(main())
