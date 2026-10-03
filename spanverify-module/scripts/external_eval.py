#!/usr/bin/env python3
"""Оценка SpanVerify на внешних размеченных наборах.

Считаются две группы метрик на одних и тех же предсказаниях:

* **наши** — token precision/recall/F1, FPR, AUC; строгий span-F1 (IoU ≥ 0.5),
  полнота накрытия и мягкий F1; метрики уровня ответа (поймана ли галлюцинация);
  всё — через ``Verifier.evaluate``, то есть тем же кодом, что обслуживает API;
* **их** — accuracy / Jaccard / hamming и ROUGE-1/2/L (перевод
  ``metrics/span_metrics.py`` из репозитория бенчмарка), чтобы числа были сравнимы
  с публикацией, а не только с самими собой.

Правила, которые соблюдает скрипт
---------------------------------

* внешние наборы — **только тест**: скрипт ничего не обучает и не калибрует,
  калибровка берётся из ``config/`` (наш корпус A);
* происхождение разметки (``human`` / ``llm`` / ``auto``) попадает в отчёт всегда;
* числа режима ``demo`` помечаются как ненаучные прямо в отчёте;
* если числа baseline из статьи не извлечены — так и написано (``null``), а не
  заменено похожими цифрами;
* срез ``--limit`` отмечается в отчёте: нельзя выдавать часть за весь набор.

Запуск::

    python scripts/external_eval.py --dataset ragtruth --task qa --split test --mode demo \
        --json reports/ext_ragtruth_qa.json
    python scripts/external_eval.py --dataset rushallu --mode demo --json reports/ext_rushallu.json
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

from spanverify.dataset import read_pairs  # noqa: E402
from spanverify.engine import Verifier  # noqa: E402
from spanverify.external_datasets import summarize_pairs  # noqa: E402
from spanverify.rushallu_metrics import evaluate_spans  # noqa: E402

DEMO_DISCLAIMER = (
    "РЕЖИМ DEMO: признаки лексические, без весов языковой модели. Числа показывают "
    "работоспособность конвейера на внешнем наборе, но научным результатом не являются."
)

BASELINE_NOTES = {
    "ragtruth": (
        "Baseline статьи RAGTruth (ACL 2024) не извлечён: PDF недоступен в этой среде "
        "(arxiv.org закрыт), значения таблиц не парсились. Числа не выдумываются."
    ),
    "rushallu": (
        "Baseline статьи RusHallu-RAG (Диалог-2026) не извлечён: PDF недоступен в этой среде. "
        "Их метрики на наших предсказаниях посчитаны и приведены в their_metrics."
    ),
}


def load_pairs(dataset: str, task: str | None, split: str, out_dir: Path) -> tuple[list[dict], Path]:
    """Прочитать адаптированный файл нужного набора (его готовит загрузчик)."""
    if dataset == "ragtruth":
        name = f"ragtruth_{task}_test.jsonl" if task and task != "all" else f"ragtruth_{split}.jsonl"
        path = out_dir / "adapted" / name
        hint = "python scripts/fetch_external_tests.py --dataset ragtruth --verify --adapt --out " f"{out_dir}"
    elif dataset == "rushallu":
        path = out_dir / "adapted" / "rushallu.jsonl"
        hint = f"python scripts/fetch_external_tests.py --dataset rushallu --verify --adapt --out {out_dir}"
    else:
        raise SystemExit(f"неизвестный набор: {dataset}")
    if not path.is_file():
        raise FileNotFoundError(f"нет адаптированного файла {path}. Сначала: {hint}")
    return list(read_pairs(path)), path


def their_metrics(pairs: list[dict], predicted: list[list[str]]) -> dict:
    """Их метрики на наших предсказаниях (accuracy / Jaccard / hamming / ROUGE)."""
    references: list[list[str]] = []
    for pair in pairs:
        answer = pair["answer"]
        references.append([answer[int(start) : int(end)] for start, end, label in pair["labels"] if int(label) == 1])
    result = evaluate_spans(references, predicted)
    return {key: (round(value, 4) if isinstance(value, float) else value) for key, value in result.items()}


def compact_metrics(metrics: dict) -> dict:
    """Сжать дерево метрик ``evaluate`` до чисел, которые попадают в отчёт."""
    tokens, spans, answers, verdicts = metrics["tokens"], metrics["spans"], metrics["answers"], metrics["verdicts"]
    return {
        "tokens": {
            key: round(tokens[key], 4) if tokens[key] == tokens[key] else None
            for key in ("precision", "recall", "f1", "fpr", "auc")
        },
        "spans": {
            "strict_f1_iou_0_5": round(spans["f1"], 4),
            "coverage": round(spans["recall_containment"], 4),
            "soft_f1_expanded": round(spans["f1_expanded_labels"], 4),
            "mean_width_ratio": round(spans["mean_width_ratio"], 2),
        },
        "answers": {
            key: round(answers[key], 4) if answers[key] == answers[key] else None
            for key in ("precision", "recall", "f1", "fpr", "auc")
        },
        "verdicts": {
            key: (round(verdicts[key], 4) if isinstance(verdicts[key], float) else verdicts[key])
            for key in ("tp", "fp", "fn", "tn", "precision", "recall", "f1", "fpr")
        },
    }


def run(
    dataset: str,
    task: str | None,
    split: str,
    mode: str,
    model: str | None,
    limit: int | None,
    out_dir: Path,
) -> dict:
    """Прогнать набор, посчитать обе группы метрик и собрать отчёт."""
    all_pairs, path = load_pairs(dataset, task, split, out_dir)
    pairs = all_pairs[:limit] if limit else all_pairs
    verifier = Verifier(mode=mode, model_name=model)
    started = time.perf_counter()
    predicted: list[list[str]] = []
    for pair in pairs:
        result = verifier.verify(pair["answer"], pair["context"])
        predicted.append([pair["answer"][span.start : span.end] for span in result.spans])
    metrics = verifier.evaluate(pairs)
    duration = time.perf_counter() - started

    summary = summarize_pairs(pairs)
    report = {
        "dataset": dataset,
        "task": task or "all",
        "split": split,
        "mode": mode,
        "model": model,
        "device": "cpu (demo-режим не использует веса)",
        "pairs": len(pairs),
        "pairs_total_in_file": len(all_pairs),
        "limit": limit,
        "source_file": str(path),
        "duration_s": round(duration, 1),
        "label_origin": summary["label_origin"],
        "labelled_pairs": summary["with_hallucination"],
        "spans_reference": summary["spans"],
        "unverified_spans": summary["unverified_spans"],
        "our_metrics": compact_metrics(metrics),
        "their_metrics": their_metrics(pairs, predicted),
        "baseline": {"extracted": False, "note": BASELINE_NOTES.get(dataset, "числа статьи не извлечены")},
        "disclaimer": DEMO_DISCLAIMER if mode == "demo" else "Режим hf: признаки считаются на весах языковой модели.",
    }
    if limit:
        report["note"] = (
            f"ЭТО СРЕЗ: {len(pairs)} пар из {len(all_pairs)}. Числа относятся только к срезу, " "не ко всему набору."
        )
    return report


def merge_into_combined(report: dict, combined_path: Path) -> dict:
    """Дописать результат в общий файл ``external_tests.json`` (по наборам и задачам)."""
    combined: dict = {"runs": {}, "rule": "внешние наборы — только тест; калибровка на корпусе A"}
    if combined_path.is_file():
        try:
            combined = json.loads(combined_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            combined = {"runs": {}}
    combined.setdefault("runs", {})
    key = f"{report['dataset']}_{report['task']}_{report['split']}_{report['mode']}"
    combined["runs"][key] = report
    combined["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    combined["disclaimer"] = (
        "Числа режима demo получены лексическими признаками и научным результатом не являются. "
        "Колонка label_origin указывает происхождение разметки: human / llm / auto."
    )
    combined_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return combined


def main() -> int:
    parser = argparse.ArgumentParser(description="Оценка на внешних размеченных наборах")
    parser.add_argument("--dataset", required=True, choices=["ragtruth", "rushallu"])
    parser.add_argument("--task", default=None, choices=["qa", "summary", "data2txt", "all"])
    parser.add_argument("--split", default="test", choices=["test", "train", "all"])
    parser.add_argument("--mode", default="demo", choices=["demo", "hf"])
    parser.add_argument(
        "--model",
        default=None,
        help="модель для режима hf (по умолчанию — значение из config, не меняется)",
    )
    parser.add_argument("--limit", type=int, default=None, help="взять первые N пар (срез)")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "external")
    parser.add_argument("--json", type=Path, default=None, help="куда записать отчёт прогона")
    parser.add_argument(
        "--combined",
        type=Path,
        default=ROOT / "reports" / "external_tests.json",
        help="общий файл результатов",
    )
    args = parser.parse_args()

    try:
        report = run(args.dataset, args.task, args.split, args.mode, args.model, args.limit, args.out)
    except (FileNotFoundError, OSError) as error:
        print(f"ошибка: {error}", file=sys.stderr)
        return 2

    tokens = report["our_metrics"]["tokens"]
    verdicts = report["our_metrics"]["verdicts"]
    theirs = report["their_metrics"]
    print(f"{report['dataset']}/{report['task']} ({report['split']}, режим {report['mode']}): {report['pairs']} пар")
    print(
        f"  наши: token F1={tokens['f1']:.3f} FPR={tokens['fpr']:.3f} AUC={tokens['auc']:.3f} | "
        f"ответы: recall={verdicts['recall']:.3f} FPR={verdicts['fpr']:.3f}"
    )
    print(
        f"  их: accuracy={theirs['accuracy']:.3f} Jaccard={theirs['jaccard_score']:.3f} "
        f"ROUGE-L={theirs['rougeL']:.3f}"
    )
    print(f"  разметка: {report['label_origin']} | baseline: не извлечён")
    if report.get("note"):
        print(f"  {report['note']}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Отчёт: {args.json}")
    combined = merge_into_combined(report, args.combined)
    print(f"Общий файл: {args.combined} (записей: {len(combined['runs'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
