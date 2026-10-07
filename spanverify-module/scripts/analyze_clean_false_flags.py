"""Разбор ложных пометок на чистых парах официального теста — по готовому кешу.

Скрипт ничего не обучает и не подбирает: он берёт обученный бандл из файла
прогона (``reports/experiments/a3/weights.json``) и кеш признаков, посчитанный
тем же прогоном на раннере, и прогоняет чистые пары официального теста через
публичный ``verify()`` — тот же путь, что отдаёт API.

Цель — таблица, а не новый порог: какие признаки у помеченных токенов, какой
ширины фрагменты, сколько чистых ответов помечено одним токеном против длинной
клаузы. Результаты пишутся в ``data/metrics/`` (JSON + Markdown) и в отчёт
прогона ничего не добавляют.

Запуск (на раннере, с кешем прогона)::

    python scripts/analyze_clean_false_flags.py \
        --dataset data/corpus_a3/pairs.jsonl \
        --splits data/corpus_a3/splits \
        --bundle reports/experiments/a3/weights.json \
        --mode hf --model ai-forever/rugpt3small_based_on_gpt2 \
        --features-cache reports/hf-cache-a3 --run-id 37609202780 \
        --out-md data/metrics/clean_false_flags_a3.md \
        --out-json data/metrics/clean_false_flags_a3.json
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
from spanverify.engine import Verifier, WeightsBundle  # noqa: E402
from spanverify.features import (  # noqa: E402
    feature_cache_key,
    load_feature_cache,
    set_feature_cache,
)


def _clean_test_pairs(splits_dir: Path) -> list[dict]:
    """Чистые пары официального теста: в разметке нет фрагментов недостоверности."""
    test_path = splits_dir / "test.jsonl"
    pairs = list(read_pairs(test_path))
    return [pair for pair in pairs if not any(int(label[2]) == 1 for label in pair.get("labels", []))]


def _pair_rows(verifier: Verifier, pairs: list[dict], mode: str, model: str) -> list[dict]:
    """По одной строке на чистую пару: вердикт, флаги, фрагменты, признаки."""
    rows: list[dict] = []
    for pair in pairs:
        answer, context = pair.get("answer", ""), pair.get("context", "")
        if mode == "hf":
            key = feature_cache_key(answer, context, "hf", model)
            if verifier.features_cache is None or key not in verifier.features_cache:
                raise SystemExit(f"кеш неполон: пара {pair.get('id')} не найдена — модель запускать нельзя")
        result = verifier.verify(answer, context, with_tokens=True)
        flagged_tokens = [token for token in result.tokens if token.get("flagged")]
        mask_flagged = [token for token in flagged_tokens]
        spans = []
        for span in result.spans:
            spans.append(
                {
                    "start": span.start,
                    "end": span.end,
                    "chars": span.end - span.start,
                    "n_tokens": span.n_tokens,
                    "source": span.source,
                    "label": span.label,
                    "risk": round(span.risk, 4),
                    "text": span.text[:80],
                }
            )
        attention = [token["attention_entropy"] for token in flagged_tokens]
        mass = [token["ctx_attention_mass"] for token in flagged_tokens]
        # Правило вердикта до правки («любой фрагмент»): спорно, если фрагмент
        # вообще есть и скор ниже порога. Считается из того же результата,
        # повторный проход не нужен.
        legacy_flag = bool(result.spans) or result.score >= result.threshold
        rows.append(
            {
                "id": str(pair.get("id")),
                "verdict": result.verdict,
                "score": round(result.score, 4),
                "threshold": round(result.threshold, 4),
                "flagged_tokens": len(mask_flagged),
                "spans": spans,
                "span_chars_total": sum(span["chars"] for span in spans),
                "max_span_chars": max((span["chars"] for span in spans), default=0),
                "mean_attention_entropy": round(sum(attention) / len(attention), 4) if attention else None,
                "mean_ctx_attention_mass": round(sum(mass) / len(mass), 4) if mass else None,
                "flagged_by_legacy_rule": bool(legacy_flag),
            }
        )
    return rows


def _summary(rows: list[dict]) -> dict:
    """Агрегаты для отчёта: доли, ширины, одиночные токены против клауз."""
    total = len(rows)
    flagged = [row for row in rows if row["verdict"] not in ("grounded", "empty")]
    single = [row for row in flagged if row["flagged_tokens"] == 1]
    multi = [row for row in flagged if row["flagged_tokens"] >= 2]
    text_only = [row for row in flagged if row["flagged_tokens"] == 0]
    legacy_flagged = [row for row in rows if row["flagged_by_legacy_rule"]]
    widths = sorted(row["max_span_chars"] for row in flagged)
    entropy = [row["mean_attention_entropy"] for row in flagged if row["mean_attention_entropy"] is not None]
    mass = [row["mean_ctx_attention_mass"] for row in flagged if row["mean_ctx_attention_mass"] is not None]
    return {
        "clean_test_pairs": total,
        "flagged_pairs": len(flagged),
        "flagged_share": round(len(flagged) / total, 4) if total else 0.0,
        "flagged_by_single_token": len(single),
        "flagged_by_two_plus_tokens": len(multi),
        "flagged_by_text_rules_only": len(text_only),
        "flagged_by_legacy_rule": len(legacy_flagged),
        "legacy_share": round(len(legacy_flagged) / total, 4) if total else 0.0,
        "max_span_chars_median": widths[len(widths) // 2] if widths else 0,
        "max_span_chars_max": widths[-1] if widths else 0,
        "mean_attention_entropy_of_flagged": round(sum(entropy) / len(entropy), 4) if entropy else None,
        "mean_ctx_attention_mass_of_flagged": round(sum(mass) / len(mass), 4) if mass else None,
    }


def _markdown(payload: dict) -> str:
    summary = payload["summary"]
    bundle = payload["bundle"]
    lines = [
        f"# Ложные пометки на чистых парах официального теста ({payload['dataset']})",
        "",
        f"Прогон: `{payload['run_id']}`. Кеш признаков и бандл — из того же прогона;"
        " модель не запускалась, ничего не подбиралось: бандл применён как есть.",
        "",
        "Команда:",
        "",
        "```",
        payload["command"],
        "```",
        "",
        "## Итог",
        "",
        "| Показатель | Значение |",
        "| --- | --- |",
        f"| Чистых пар в официальном тесте | {summary['clean_test_pairs']} |",
        f"| Помечено вердиктом (новое правило) | {summary['flagged_pairs']} ({summary['flagged_share']:.3f}) |",
        f"| Помечено одним токеном маски | {summary['flagged_by_single_token']} |",
        f"| Помечено двумя и более токенами | {summary['flagged_by_two_plus_tokens']} |",
        f"| Помечено текстовыми правилами без токенов маски | {summary['flagged_by_text_rules_only']} |",
        f"| Помечалось бы старым правилом «любой фрагмент» | {summary['flagged_by_legacy_rule']}"
        f" ({summary['legacy_share']:.3f}) |",
        f"| Медиана ширины самого широкого фрагмента, знаков | {summary['max_span_chars_median']} |",
        f"| Максимальная ширина фрагмента, знаков | {summary['max_span_chars_max']} |",
        f"| Средняя энтропия внимания помеченных токенов | {summary['mean_attention_entropy_of_flagged']} |",
        f"| Средняя масса внимания к контексту помеченных токенов | {summary['mean_ctx_attention_mass_of_flagged']} |",
        "",
        f"Веса бандла: {bundle['weights']}; сигнал: {bundle['signal']}; порог маски"
        f" z={bundle['span_z']}, floor={bundle['span_floor']}, cap={bundle['span_cap']};"
        f" порог ответа {bundle['threshold']}.",
        "",
        "## Помеченные чистые пары",
        "",
        "| Пара | Вердикт | Скор | Токенов помечено | Фрагментов | Ширина макс., знаков | Текст фрагмента |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["rows"]:
        if row["verdict"] in ("grounded", "empty"):
            continue
        text = (row["spans"][0]["text"] if row["spans"] else "—").replace("|", "/")
        lines.append(
            f"| {row['id']} | {row['verdict']} | {row['score']:.3f} | {row['flagged_tokens']} "
            f"| {len(row['spans'])} | {row['max_span_chars']} | {text} |"
        )
    lines += [
        "",
        "## Не помеченные чистые пары",
        "",
        f"{summary['clean_test_pairs'] - summary['flagged_pairs']} пар: вердикт `grounded`.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Разбор ложных пометок на чистых парах теста по кешу прогона")
    parser.add_argument("--dataset", default="data/corpus_a3/pairs.jsonl")
    parser.add_argument("--splits", default="data/corpus_a3/splits")
    parser.add_argument("--bundle", default="reports/experiments/a3/weights.json")
    parser.add_argument("--mode", choices=["demo", "hf"], default="hf")
    parser.add_argument("--model", default="ai-forever/rugpt3small_based_on_gpt2")
    parser.add_argument("--features-cache", action="append", default=[], metavar="PATH")
    parser.add_argument("--run-id", default=None, help="номер прогона, из которого кеш и бандл")
    parser.add_argument("--out-md", default="data/metrics/clean_false_flags_a3.md")
    parser.add_argument("--out-json", default="data/metrics/clean_false_flags_a3.json")
    args = parser.parse_args(argv)

    pairs = _clean_test_pairs(Path(args.splits))
    if not pairs:
        print("чистых пар в официальном тесте нет — считать нечего", file=sys.stderr)
        return 2

    bundle = WeightsBundle.load(args.bundle)
    cache = None
    if args.mode == "hf":
        if not args.features_cache:
            print("режим hf требует --features-cache (кеш прогона)", file=sys.stderr)
            return 2
        cache = load_feature_cache(args.features_cache, counting=True)
        set_feature_cache(cache)
        print(f"кеш признаков: {len(cache)} ключей из {len(args.features_cache)} пути(ей)", flush=True)
    verifier = Verifier(mode=args.mode, model_name=args.model, weights=bundle, features_cache=cache)

    started = time.time()
    rows = _pair_rows(verifier, pairs, args.mode, args.model)
    if cache is not None and getattr(cache, "misses", 0):
        print(f"кеш неполон: промахов {cache.misses} — запуск без модели невозможен", file=sys.stderr)
        return 2
    payload = {
        "dataset": args.dataset,
        "splits": str(args.splits),
        "bundle_file": args.bundle,
        "run_id": args.run_id,
        "mode": args.mode,
        "model": args.model,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_s": round(time.time() - started, 1),
        "rule": (
            "разбор по готовому кешу прогона: бандл применён как есть, " "пороги не подбирались, модель не запускалась"
        ),
        "command": " ".join(sys.argv),
        "bundle": {
            "weights": bundle.weights,
            "threshold": round(bundle.threshold, 4),
            "span_z": bundle.span_z,
            "span_floor": bundle.span_floor,
            "span_cap": bundle.span_cap,
            "signal": (bundle.meta or {}).get("signal"),
        },
        "summary": _summary(rows),
        "rows": rows,
    }

    out_md = Path(args.out_md)
    out_json = Path(args.out_json)
    for path in (out_md, out_json):
        path.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(_markdown(payload), encoding="utf-8")

    summary = payload["summary"]
    print(f"чистых пар: {summary['clean_test_pairs']}, помечено: {summary['flagged_pairs']}")
    print(
        f"одним токеном: {summary['flagged_by_single_token']}, "
        f"двумя и более: {summary['flagged_by_two_plus_tokens']}, "
        f"текстовыми правилами без токенов маски: {summary['flagged_by_text_rules_only']}"
    )
    print(f"Отчёт: {out_md}\nДанные: {out_json}")
    return 0


if __name__ == "__main__":  # pragma: no cover - утилита
    raise SystemExit(main())
