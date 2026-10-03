#!/usr/bin/env python3
"""Эксперименты и отчётность по качеству SpanVerify.

Скрипт обучает калибратор на обучающей части корпуса и измеряет качество
на отложенной: метрики по токенам, по фрагментам (span-level), точность
оценки доли участия ИИ, сравнение с тривиальным baseline и абляции по
признакам. Результаты — JSON + Markdown + SVG-график (без внешних
зависимостей, чтобы отчёт собирался в контуре заказчика).

Примеры:

    python scripts/run_experiments.py --dataset data/demo_dataset.jsonl \
        --out reports/demo

    python scripts/run_experiments.py --dataset data/real.jsonl \
        --backend hf --model cointegrated/rubert-tiny2 --out reports/hf

Разметка датасета: JSONL, по одному документу на строку:

    {"id": "...", "text": "...", "labels": [[start, end, 1], [start, end, 0]]}
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spanverify import Config, Detector  # noqa: E402
from spanverify.calibration import metrics_at  # noqa: E402
from spanverify.demo_data import generate_dataset, read_dataset  # noqa: E402
from spanverify.text import tokenize  # noqa: E402
from spanverify.training import token_labels, train_calibrator  # noqa: E402

VERDICTS = ("likely_ai", "mixed", "likely_human")


# ---------- метрики ----------


def span_metrics(predicted: list[tuple[int, int]], truth: list[tuple[int, int]], iou_min: float = 0.5) -> dict:
    """Сопоставление найденных фрагментов с истинными по пересечению (IoU)."""
    matched_truth: set[int] = set()
    matched_pred: set[int] = set()
    for i, (ps, pe) in enumerate(predicted):
        best_j, best_iou = -1, 0.0
        for j, (ts, te) in enumerate(truth):
            intersection = max(0, min(pe, te) - max(ps, ts))
            union = max(pe, te) - min(ps, ts)
            iou = intersection / union if union else 0.0
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_min:
            matched_pred.add(i)
            matched_truth.add(best_j)

    tp = len(matched_pred)
    fp = len(predicted) - tp
    fn = len(truth) - len(matched_truth)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "iou_min": iou_min,
    }


def char_level_metrics(predicted: list[tuple[int, int]], truth: list[tuple[int, int]]) -> dict:
    """Точность локализации на уровне символов (не зависит от дробления фрагментов)."""

    def mask(spans: list[tuple[int, int]]) -> set[int]:
        out: set[int] = set()
        for start, end in spans:
            out.update(range(start, end))
        return out

    pred, true = mask(predicted), mask(truth)
    intersection = len(pred & true)
    precision = intersection / len(pred) if pred else 0.0
    recall = intersection / len(true) if true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def char_iou(predicted: list[tuple[int, int]], truth: list[tuple[int, int]]) -> float:
    def mask(spans: list[tuple[int, int]]) -> set[int]:
        out: set[int] = set()
        for start, end in spans:
            out.update(range(start, end))
        return out

    pred, true = mask(predicted), mask(truth)
    union = pred | true
    return len(pred & true) / len(union) if union else 1.0


# ---------- прогон ----------


@dataclass
class DatasetEntry:
    doc_id: str
    text: str
    truth_spans: list[tuple[int, int]]
    truth_share: float


def load_entries(path: Path | None, n_synthetic: int, seed: int) -> list[DatasetEntry]:
    documents = list(read_dataset(path)) if path else generate_dataset(n_synthetic, seed=seed)
    entries: list[DatasetEntry] = []
    for doc in documents:
        text = doc["text"]
        truth = [(int(s), int(e)) for s, e, label in doc.get("labels", []) if int(label) == 1]
        share = sum(e - s for s, e in truth) / len(text) if text else 0.0
        entries.append(DatasetEntry(str(doc.get("id", "?")), text, truth, share))
    return entries


def evaluate(
    detector: Detector,
    entries: list[DatasetEntry],
    threshold: float,
) -> dict:
    token_true: list[int] = []
    token_pred: list[int] = []
    token_prob: list[float] = []
    span_matches: list[dict] = []
    span_matches_loose: list[dict] = []
    char_metrics: list[dict] = []
    ious: list[float] = []
    share_errors: list[float] = []
    verdict_counts = {name: {"correct_docs": 0, "docs": 0} for name in ("ai", "human")}

    for entry in entries:
        result = detector.analyze(entry.text, threshold=threshold)
        predicted = [(s.start_char, s.end_char) for s in result.spans]

        tokens = [t for t in tokenize(entry.text) if t.is_word]
        if tokens:
            probs = detector.token_probabilities(entry.text)
            labels = token_labels(tokens, [[s, e, 1] for s, e in entry.truth_spans])
            if len(probs) == len(labels):
                token_prob.extend(probs)
                token_true.extend(labels)
                token_pred.extend(1 if p >= threshold else 0 for p in probs)

        span_matches.append(span_metrics(predicted, entry.truth_spans, iou_min=0.5))
        span_matches_loose.append(span_metrics(predicted, entry.truth_spans, iou_min=0.25))
        char_metrics.append(char_level_metrics(predicted, entry.truth_spans))
        ious.append(char_iou(predicted, entry.truth_spans))
        share_errors.append(abs(result.ai_fraction - entry.truth_share))

        bucket = "ai" if entry.truth_share >= 0.5 else ("human" if entry.truth_share <= 0.2 else None)
        if bucket:
            verdict_counts[bucket]["docs"] += 1
            expected = "likely_ai" if bucket == "ai" else "likely_human"
            if result.verdict == expected:
                verdict_counts[bucket]["correct_docs"] += 1

    tp = sum(1 for t, p in zip(token_true, token_pred, strict=False) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(token_true, token_pred, strict=False) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(token_true, token_pred, strict=False) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(token_true, token_pred, strict=False) if t == 0 and p == 0)
    token_metrics = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "hdr": tn / (tn + fp) if tn + fp else 0.0,
    }
    token_metrics["f1"] = (
        2
        * token_metrics["precision"]
        * token_metrics["recall"]
        / (token_metrics["precision"] + token_metrics["recall"])
        if token_metrics["precision"] + token_metrics["recall"]
        else 0.0
    )

    n = len(span_matches)
    return {
        "documents": len(entries),
        "tokens": len(token_true),
        "tokens_ai_share": sum(token_true) / len(token_true) if token_true else 0.0,
        "threshold": threshold,
        "token_metrics": token_metrics,
        "span_metrics": {key: sum(m[key] for m in span_matches) / n for key in ("precision", "recall", "f1")},
        "span_metrics_iou025": {
            key: sum(m[key] for m in span_matches_loose) / n for key in ("precision", "recall", "f1")
        },
        "char_metrics": {key: sum(m[key] for m in char_metrics) / n for key in ("precision", "recall", "f1")},
        "char_iou_mean": sum(ious) / len(ious) if ious else 0.0,
        "share_mae": sum(share_errors) / len(share_errors) if share_errors else 0.0,
        "share_max_error": max(share_errors) if share_errors else 0.0,
        "document_verdicts": verdict_counts,
        "probability_samples": {
            "ai": [round(p, 4) for t, p in zip(token_true, token_prob, strict=False) if t == 1][:5000],
            "human": [round(p, 4) for t, p in zip(token_true, token_prob, strict=False) if t == 0][:5000],
        },
    }


def _complement(entry: DatasetEntry) -> list[tuple[int, int]]:
    """Человеческие участки документа (для повторного прохода по разметке)."""
    spans = sorted(entry.truth_spans)
    out: list[tuple[int, int]] = []
    cursor = 0
    for start, end in spans:
        if start > cursor:
            out.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < len(entry.text):
        out.append((cursor, len(entry.text)))
    return out


def threshold_sweep(detector: Detector, entries: list[DatasetEntry]) -> list[dict]:
    """Метрики по токенам для сетки порогов (для отчёта и выбора порога)."""
    scores: list[float] = []
    labels: list[int] = []
    for entry in entries:
        tokens = [t for t in tokenize(entry.text) if t.is_word]
        probs = detector.token_probabilities(entry.text)
        flags = token_labels(tokens, [[s, e, 1] for s, e in entry.truth_spans])
        if len(probs) == len(flags):
            scores.extend(probs)
            labels.extend(flags)
    grid = [round(0.1 * i, 2) for i in range(1, 10)]
    return [metrics_at(scores, labels, thr) for thr in grid]


# ---------- вывод ----------


def histogram_svg(ai: list[float], human: list[float], path: Path, bins: int = 20) -> None:
    """Гистограмма калиброванных вероятностей по классам (чистый SVG)."""
    width, height, pad = 720, 320, 40
    counts_ai = [0] * bins
    counts_hu = [0] * bins
    for value in ai:
        counts_ai[min(bins - 1, int(value * bins))] += 1
    for value in human:
        counts_hu[min(bins - 1, int(value * bins))] += 1
    top = max(counts_ai + counts_hu + [1])
    bar = (width - 2 * pad) / bins

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{pad}" y="24" font-family="sans-serif" font-size="15">'
        "Распределение калиброванной вероятности «текст от ИИ»</text>",
    ]
    for i in range(bins):
        for counts, color, offset in ((counts_ai, "#e5484d", 0.0), (counts_hu, "#12855f", bar / 2)):
            h_value = (counts[i] / top) * (height - 2 * pad - 20)
            x = pad + i * bar + offset
            y = height - pad - h_value
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar / 2 - 1:.1f}" '
                f'height="{h_value:.1f}" fill="{color}" opacity="0.75"/>'
            )
    for i in range(0, bins + 1, 5):
        x = pad + i * bar
        parts.append(
            f'<text x="{x:.1f}" y="{height - pad + 16}" font-family="sans-serif" '
            f'font-size="11" text-anchor="middle">{i / bins:.1f}</text>'
        )
    parts.append(
        f'<text x="{width - pad}" y="{height - pad + 16}" font-family="sans-serif" '
        'font-size="11" text-anchor="end">p(ИИ)</text>'
    )
    parts.append(
        f'<text x="{pad}" y="{height - pad + 34}" font-family="sans-serif" font-size="12">'
        '<tspan fill="#e5484d">■ машинные токены</tspan>  '
        '<tspan fill="#12855f">■ человеческие токены</tspan></text>'
    )
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def markdown_report(report: dict, sweep: list[dict], backend: str, dataset: str) -> str:
    tm = report["token_metrics"]
    sm = report["span_metrics"]
    sm_loose = report["span_metrics_iou025"]
    cm = report["char_metrics"]
    lines = [
        "# Отчёт по качеству SpanVerify",
        "",
        f"- Корпус: `{dataset}`",
        f"- Бэкенд: `{backend}`",
        f"- Документов: {report['documents']}, токенов: {report['tokens']} "
        f"(доля машинных токенов {report['tokens_ai_share']:.1%})",
        f"- Порог: {report['threshold']:.4f}",
        "",
        "## Метрики на отложенной части",
        "",
        "| Уровень | Метрика | Значение |",
        "|---|---|---|",
        f"| Токены | Precision | {tm['precision']:.3f} |",
        f"| Токены | Recall | {tm['recall']:.3f} |",
        f"| Токены | F1 | {tm['f1']:.3f} |",
        f"| Токены | FPR | {tm['fpr']:.3f} |",
        f"| Токены | HDR (доля верно опознанных человеческих) | {tm['hdr']:.3f} |",
        f"| Фрагменты (IoU ≥ 0.5) | Precision | {sm['precision']:.3f} |",
        f"| Фрагменты (IoU ≥ 0.5) | Recall | {sm['recall']:.3f} |",
        f"| Фрагменты (IoU ≥ 0.5) | F1 | {sm['f1']:.3f} |",
        f"| Фрагменты (IoU ≥ 0.25) | F1 | {sm_loose['f1']:.3f} |",
        f"| Символы (по разметке) | Precision | {cm['precision']:.3f} |",
        f"| Символы (по разметке) | Recall | {cm['recall']:.3f} |",
        f"| Символы (по разметке) | F1 | {cm['f1']:.3f} |",
        f"| Символы | Средний IoU разметки | {report['char_iou_mean']:.3f} |",
        "",
        "## Оценка доли участия ИИ",
        "",
        f"- Средняя абсолютная ошибка: {report['share_mae']:.3f} доли документа",
        f"- Максимальная ошибка: {report['share_max_error']:.3f}",
        "",
        "## Сетка порогов (по токенам)",
        "",
        "| Порог | Precision | Recall | F1 | FPR | HDR |",
        "|---|---|---|---|---|---|",
    ]
    for row in sweep:
        lines.append(
            f"| {row['threshold']:.2f} | {row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['f1']:.3f} | {row['fpr']:.3f} | {row['hdr']:.3f} |"
        )
    lines += [
        "",
        "## Ограничения",
        "",
        "- На синтетическом корпусе метрики завышены по построению: оба класса",
        "  порождены одним генератором. Числа имеют смысл только как проверка",
        "  конвейера; для отчётности по договору нужен размеченный реальный корпус",
        "  (режим `--backend hf`).",
        "- Пороговая оценка доли участия ИИ консервативна: она не завышает долю",
        "  на человеческих текстах, но может занижать её на коротких машинных",
        "  вставках.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Эксперименты SpanVerify")
    parser.add_argument(
        "--dataset", type=Path, default=None, help="JSONL с разметкой; если не указан — синтетический корпус"
    )
    parser.add_argument("--n", type=int, default=240, help="размер синтетического корпуса")
    parser.add_argument("--seed", type=int, default=1312)
    parser.add_argument("--backend", choices=["surrogate", "hf"], default="surrogate")
    parser.add_argument("--model", default=None, help="модель для режима hf")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--max-fpr", type=float, default=0.1)
    parser.add_argument("--out", type=Path, default=Path("reports/experiment"))
    args = parser.parse_args()

    overrides = {"backend": args.backend, "max_fpr": args.max_fpr, "seed": args.seed}
    if args.model:
        overrides["hf_model"] = args.model
    config = Config.load(ROOT / "config" / "config.json").with_overrides(**overrides)
    detector = Detector(config)

    entries = load_entries(args.dataset, args.n, args.seed)
    rng = random.Random(args.seed)
    rng.shuffle(entries)
    split = max(1, int(len(entries) * (1 - args.test_size)))
    train, test = entries[:split], entries[split:]

    print(f"Корпус: {len(entries)} документов (обучение {len(train)}, отложенная {len(test)})")
    report = train_calibrator(
        detector,
        [
            {
                "id": e.doc_id,
                "text": e.text,
                "labels": [[s, end, 1] for s, end in e.truth_spans] + [[s, end, 0] for s, end in _complement(e)],
            }
            for e in train
        ],
        config=config,
        dataset_name=str(args.dataset or "synthetic"),
    )
    evaluation_detector = Detector(config.with_overrides(threshold=report.threshold), calibrator=report.calibrator)

    print(f"Калибратор обучен: {report.summary()}")
    evaluation = evaluate(evaluation_detector, test, report.threshold)
    sweep = threshold_sweep(evaluation_detector, test)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(
        json.dumps(
            {
                "evaluation": evaluation,
                "threshold_sweep": sweep,
                "training": {"stats": report.stats, "cross_validation": report.cross_validation.get("mean", {})},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.out / "report.md").write_text(
        markdown_report(evaluation, sweep, args.backend, str(args.dataset or "synthetic")),
        encoding="utf-8",
    )
    histogram_svg(
        evaluation["probability_samples"]["ai"],
        evaluation["probability_samples"]["human"],
        args.out / "probabilities.svg",
    )

    tm = evaluation["token_metrics"]
    print(
        f"Токены:  P={tm['precision']:.3f} R={tm['recall']:.3f} F1={tm['f1']:.3f} "
        f"FPR={tm['fpr']:.3f} HDR={tm['hdr']:.3f}"
    )
    print(
        f"Фрагменты: P={evaluation['span_metrics']['precision']:.3f} "
        f"R={evaluation['span_metrics']['recall']:.3f} "
        f"F1={evaluation['span_metrics']['f1']:.3f}"
    )
    cm = evaluation["char_metrics"]
    print(
        f"Символы:  P={cm['precision']:.3f} R={cm['recall']:.3f} F1={cm['f1']:.3f} "
        f"IoU={evaluation['char_iou_mean']:.3f}"
    )
    print(f"Доля участия ИИ: MAE={evaluation['share_mae']:.3f}")
    print(f"Отчёт: {args.out / 'report.md'} | {args.out / 'report.json'} | " f"{args.out / 'probabilities.svg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
