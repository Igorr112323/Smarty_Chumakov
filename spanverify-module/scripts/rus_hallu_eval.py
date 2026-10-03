#!/usr/bin/env python3
"""Оценка SpanVerify на внешнем бенчмарке RusHallu-RAG.

Считаются **два набора метрик на одних и тех же предсказаниях**:

* наши (token precision/recall/F1, FPR, AUC; строгий span-F1 при IoU ≥ 0.5;
  полнота накрытия разметки) — через ``Verifier.evaluate``, то есть тем же кодом,
  что обслуживает API;
* их (accuracy / Jaccard / hamming, ROUGE-1/2/L) — перевод
  ``metrics/span_metrics.py`` из репозитория бенчмарка
  (см. ``spanverify/rushallu_metrics.py``).

Важно: корпус B — **только тест**. Ничего на нём не обучается и не калибруется;
калибровка берётся из ``config/`` (корпус A). Это условие внешней проверки.

Что выводится честно
--------------------

* ``mode``: ``demo`` (то, что работает в CI) или ``hf`` (нужен GPU);
* отдельным полем — «строгий span-F1»: конвейер расширяет найденный токен до границ
  предложения, и на разметке бенчмарка (точные цитаты) это штрафуется;
* сравнение с baseline статьи выполняется, ТОЛЬКО если числа извлечены из PDF;
  иначе в отчёте стоит «не извлечено» (по правилу: не выдумывать числа).

Запуск::

    python scripts/rus_hallu_eval.py --data data/external/rushallu --mode demo \
        --json reports/rus_hallu_eval.json
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
from spanverify.rushallu_metrics import evaluate_spans  # noqa: E402

DISCLAIMER = (
    "RusHallu-RAG — внешний тест с человеческой разметкой спанов. Числа режима demo "
    "получены лексическими суррогатами признаков и научным результатом не являются: "
    "научный вывод требует режима hf на GPU, который в этой среде не запускался."
)


def predicted_span_texts(answer: str, spans) -> list[str]:
    """Тексты предсказанных фрагментов (в формате их метрик — строки)."""
    return [answer[span.start : span.end] for span in spans]


def true_span_texts(pair: dict) -> list[str]:
    """Истинные спаны как строки (из смещений ``labels``)."""
    answer = pair["answer"]
    return [answer[int(start) : int(end)] for start, end, label in pair.get("labels", []) if int(label) == 1]


def evaluate(data_dir: Path, mode: str, limit: int | None = None) -> dict:
    """Прогнать наш конвейер по внешнему набору и посчитать обе группы метрик."""
    pairs_path = data_dir / "pairs.jsonl"
    if not pairs_path.is_file():
        raise FileNotFoundError(
            f"нет {pairs_path}: сначала запустите scripts/fetch_rushallu.py --out {data_dir} --verify"
        )
    pairs = list(read_pairs(pairs_path))
    if limit:
        pairs = pairs[:limit]

    verifier = Verifier(mode=mode)
    started = time.perf_counter()
    predictions: list[list[str]] = []
    references: list[list[str]] = []
    for pair in pairs:
        result = verifier.verify(pair["answer"], pair["context"])
        predictions.append(predicted_span_texts(pair["answer"], result.spans))
        references.append(true_span_texts(pair))

    ours = verifier.evaluate(pairs)
    theirs = evaluate_spans(references, predictions)
    duration = time.perf_counter() - started

    return {
        "dataset": "RusHallu-RAG (SberQuAD-RAG + ruSciBench-RAG)",
        "dataset_dir": str(data_dir),
        "dataset_version": pairs[0].get("meta", {}).get("dataset_version") if pairs else None,
        "citation": pairs[0].get("meta", {}).get("citation") if pairs else None,
        "mode": mode,
        "pairs": len(pairs),
        "duration_s": round(duration, 1),
        "disclaimer": DISCLAIMER,
        "our_metrics": {
            "tokens": {
                key: (round(ours["tokens"][key], 4) if key != "threshold" else ours["tokens"][key])
                if isinstance(ours["tokens"].get(key), int | float)
                else None
                for key in ("precision", "recall", "f1", "fpr", "auc")
            },
            "spans": {
                "strict_f1_iou_0_5": round(ours["spans"]["f1"], 4),
                "coverage": round(ours["spans"]["recall_containment"], 4),
                "soft_f1_expanded": round(ours["spans"]["f1_expanded_labels"], 4),
                "mean_width_ratio": round(ours["spans"]["mean_width_ratio"], 2),
            },
            "answers": {
                key: (round(ours["answers"][key], 4) if isinstance(ours["answers"].get(key), int | float) else None)
                for key in ("precision", "recall", "f1", "fpr", "auc")
            },
        },
        "their_metrics": {
            key: (round(value, 4) if isinstance(value, float) else value)
            for key, value in theirs.items()
        },
        "baseline_comparison": None,
        "baseline_note": (
            "Сравнение с baseline статьи не выполнено: значения из Таблиц 2-4 не извлечены "
            "в этом окружении (PDF не парсился). Числа не выдумываются."
        ),
        "prediction_examples": [
            {
                "id": pair["id"],
                "true": references[index][:3],
                "predicted": predictions[index][:3],
            }
            for index, pair in enumerate(pairs[:5])
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Оценка SpanVerify на RusHallu-RAG")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "external" / "rushallu")
    parser.add_argument("--mode", default="demo", choices=["demo", "hf"])
    parser.add_argument("--limit", type=int, default=None, help="взять только первые N пар")
    parser.add_argument("--json", type=Path, default=None, help="куда записать отчёт")
    args = parser.parse_args()

    try:
        report = evaluate(args.data, args.mode, limit=args.limit)
    except (FileNotFoundError, OSError) as error:
        print(f"ошибка: {error}", file=sys.stderr)
        return 2

    tokens = report["our_metrics"]["tokens"]
    spans = report["our_metrics"]["spans"]
    theirs = report["their_metrics"]
    print(f"Пар: {report['pairs']} (режим {report['mode']}, {report['duration_s']} с)")
    print(
        f"Наши метрики: token F1={tokens['f1']:.3f} FPR={tokens['fpr']:.3f} AUC={tokens['auc']:.3f} | "
        f"строгий span-F1={spans['strict_f1_iou_0_5']:.3f} полнота={spans['coverage']:.3f} "
        f"ширина ×{spans['mean_width_ratio']:.1f}"
    )
    print(
        f"Их метрики: accuracy={theirs['accuracy']:.3f} Jaccard={theirs['jaccard_score']:.3f} "
        f"hamming={theirs['hamming_loss']:.3f} ROUGE-1={theirs['rouge1']:.3f} "
        f"ROUGE-2={theirs['rouge2']:.3f} ROUGE-L={theirs['rougeL']:.3f}"
    )
    print(f"Оговорка: {report['disclaimer']}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Отчёт: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
