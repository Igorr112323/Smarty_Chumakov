"""Калибровка агрегации вердикта: параметры подбираются на dev, тест — один раз.

Зачем. Токенная маска работает по каждому токену ответа, а вердикт выносится по
всему ответу. На реальных актах (корпус A3) длинный ответ из ~100 токенов при
токенном FPR даже 6 % почти всегда содержит пару ложных токенов маски, и правило
«два токена — ответ спорный» помечает почти все честные ответы (прогон
37609202780: вердикт FPR 0.775 в режиме hf, 0.3875 в режиме demo). Агрегацию надо
калибровать отдельно от маски — по тем же правилам, что и всё остальное:

* параметры правила перебираются **только на dev** официального разбиения;
* выбирается точка, у которой F1 максимален при ограничении ``FPR ≤ target``;
* выбранные параметры замораживаются в ``config/verdict.json`` (артефакт идёт в
  поставку и в исполняемый файл);
* официальный ``test`` измеряется **один раз** уже замороженным правилом и в
  подборе не участвует.

Правило (``corroborated``) требует согласия сигналов: одиночный фрагмент маски
ответ не обвиняет, а признаки агрегации (число помеченных токенов, длина самого
длинного фрагмента, доля помеченных токенов, текстовые правила) задают, сколько
согласия нужно.

Запуск::

    python scripts/calibrate_verdict.py \
        --dataset data/corpus_a3/pairs.jsonl --mode demo \
        --out reports/verdict_calibration.json --freeze config/verdict.json

Все числа в отчёте — из этого прогона: выбранное правило, метрики dev (по нему
шёл выбор) и метрики test (сняты один раз после заморозки).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spanverify.dataset import read_pairs  # noqa: E402
from spanverify.engine import (  # noqa: E402
    VERDICT_CORROBORATED,
    Verifier,
    verdict_config_from_params,
)
from spanverify.features import load_feature_cache, set_feature_cache  # noqa: E402
from spanverify.train import train  # noqa: E402

DEFAULT_FREEZE = "config/verdict.json"


def _git_commit() -> str:
    """Коммит рабочего дерева — числа без него не привязываются к состоянию кода."""
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            ).stdout.strip()
            or "неизвестно"
        )
    except (OSError, subprocess.SubprocessError):
        return "неизвестно"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pair_truth(pair: dict) -> int:
    """1 — в разметке есть фрагмент недостоверности, 0 — пара чистая."""
    return 1 if pair.get("labels") else 0


def evidence(verifier: Verifier, pairs: list[dict]) -> list[dict[str, Any]]:
    """Собрать по каждой паре всё, что нужно правилу агрегации вердикта.

    Ничего не подбирается: сохраняются ровно те величины, которые вернул
    ``verify()`` (число помеченных токенов маски, длины фрагментов, источники,
    скор ответа и порог). Правило потом пересчитывается по этой таблице сколько
    нужно раз — повторных прогонов модели для перебора не требуется.
    """
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        result = verifier.verify(pair["answer"], pair.get("context"))
        mask = [span for span in result.spans if span.source == "mask"]
        text_spans = [span for span in result.spans if span.source != "mask"]
        stats = result.stats or {}
        scored = int(stats.get("scored_tokens") or 0)
        mask_tokens = sum(int(span.n_tokens) for span in mask)
        rows.append(
            {
                "id": pair.get("id"),
                "y": _pair_truth(pair),
                "mode": (pair.get("meta") or {}).get("mode"),
                "score": float(result.score),
                "threshold": float(result.threshold),
                "spans_total": len(result.spans),
                "mask_spans": len(mask),
                "mask_tokens": mask_tokens,
                "max_fragment_chars": max((len(span.text) for span in mask), default=0),
                "max_fragment_tokens": max((int(span.n_tokens) for span in mask), default=0),
                "text_spans": len(text_spans),
                "text_sources": sorted({span.source for span in text_spans}),
                "scored_tokens": scored,
                "flagged_share": round(mask_tokens / scored, 6) if scored else 0.0,
                "mean_risk": float(stats.get("mean_risk") or 0.0),
                "p90_risk": float(stats.get("p90_risk") or 0.0),
                "raw_score": float(stats.get("raw_score") or 0.0),
            }
        )
    return rows


def apply_rule(rows: list[dict[str, Any]], params: dict[str, Any]) -> list[bool]:
    """Пересчитать вердикт «спорный/недостоверный» по таблице свидетельств."""
    flags: list[bool] = []
    min_tokens = int(params["min_mask_tokens"])
    min_chars = int(params["min_fragment_chars"])
    min_share = float(params["min_flagged_share"])
    text_decisive = bool(params["text_rule_decisive"])
    for row in rows:
        if row["score"] >= row["threshold"]:
            flags.append(True)
            continue
        if text_decisive and row["text_spans"] > 0:
            flags.append(True)
            continue
        strong = (
            row["mask_tokens"] >= min_tokens
            and row["max_fragment_chars"] >= min_chars
            and row["flagged_share"] >= min_share
        )
        flags.append(bool(strong))
    return flags


def legacy_flags(rows: list[dict[str, Any]], rule: str) -> list[bool]:
    """Флаги исторических правил — по той же таблице свидетельств.

    Нужны для сравнения «до/после» на одном и том же прогоне и одном тесте:
    правило вердикта не влияет ни на маску, ни на токенные метрики, поэтому
    сравнение честное.
    """
    flags: list[bool] = []
    for row in rows:
        if row["score"] >= row["threshold"]:
            flags.append(True)
        elif rule == "any_span":
            flags.append(row["spans_total"] > 0)
        else:  # min_two_tokens
            flags.append(row["text_spans"] > 0 or row["mask_tokens"] >= 2)
    return flags


def rules_report(rows: list[dict[str, Any]], params: dict[str, Any]) -> dict[str, Any]:
    """Метрики трёх правил на одной и той же таблице свидетельств."""
    return {
        "rule_any_span": metrics(rows, legacy_flags(rows, "any_span")),
        "rule_min_two_tokens": metrics(rows, legacy_flags(rows, "min_two_tokens")),
        "rule_corroborated": metrics(rows, apply_rule(rows, params)),
    }


def metrics(rows: list[dict[str, Any]], flags: list[bool]) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for row, flag in zip(rows, flags, strict=True):
        truth = row["y"] == 1
        if flag and truth:
            tp += 1
        elif flag and not truth:
            fp += 1
        elif not flag and truth:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "flagged_share": round((tp + fp) / len(rows), 4) if rows else 0.0,
        "pairs": len(rows),
    }


def grid() -> list[dict[str, Any]]:
    """Сетка параметров правила: только те, что осмысленны для вердикта по ответу.

    Нижняя граница ``min_mask_tokens`` — 2, а не 1: «одиночный фрагмент маски не
    обвиняет ответ» — это объявленное требование к продукту (прогон 37532050295:
    правило «любой фрагмент» помечало 1047 ответов из 1200), а не результат
    подбора. Параметры ниже двух токенов в сетку не входят.

    ``text_rule_decisive`` зафиксировано в True и в сетку не входит: закрытие
    дефекта D требует, чтобы число, взятое у чужого объекта документа, делало
    ответ не «подтверждённым» (это проверяют тесты
    ``tests/test_number_attribution.py``). Точность правила повышается не
    отключением его решающей роли, а требованием однозначности привязки в
    ``features._attribute_tokens``.
    """
    out: list[dict[str, Any]] = []
    for min_tokens, min_chars, min_share in itertools.product(
        (2, 3, 4, 5, 6, 8, 10, 12, 16),
        (0, 20, 40, 60, 80, 120, 160, 200, 300, 400),
        (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5),
    ):
        text_decisive = True
        out.append(
            {
                "min_mask_tokens": min_tokens,
                "min_fragment_chars": min_chars,
                "min_flagged_share": min_share,
                "text_rule_decisive": text_decisive,
            }
        )
    return out


def choose(
    dev_rows: list[dict[str, Any]], target_fpr: float
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Выбрать параметры на dev: максимум F1 при FPR ≤ target.

    Правило выбора объявлено заранее и не меняется по результату:

    1. точки с ``FPR ≤ target`` — берём максимум F1;
    2. если таких точек в сетке нет (ограничение недостижимо на dev), берём
       максимум F1 в пределах двойного ограничения ``2·target`` — это ближайшая
       к цели достижимая точка, и она не выдаётся за выполнение цели;
    3. если и это недостижимо, берём максимум F1 по всей сетке.

    Тай-брейк во всех случаях — меньше ложных срабатываний, затем более строгие
    параметры (больше токенов, длиннее фрагмент): правило предсказуемо, а не
    подобрано под красивый результат.
    """
    evaluated: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for params in grid():
        evaluated.append((params, metrics(dev_rows, apply_rule(dev_rows, params))))

    def ok(limit: float) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        return [item for item in evaluated if item[1]["fpr"] <= limit]

    within = ok(target_fpr)
    if within:
        stage = f"выбор внутри цели FPR ≤ {target_fpr}"
    else:
        within = ok(min(1.0, 2 * target_fpr))
        stage = (
            f"цель FPR ≤ {target_fpr} на dev недостижима ни одной точкой сетки: "
            f"взята ближайшая достижимая зона FPR ≤ {min(1.0, 2 * target_fpr)}"
        )
    if not within:
        within = evaluated
        stage = f"ни одна точка сетки не уложилась в FPR ≤ {min(1.0, 2 * target_fpr)}: взят максимум F1 по сетке"

    def key(item: tuple[dict[str, Any], dict[str, Any]]) -> tuple[float, float, int, int]:
        params, result = item
        return (result["f1"], -result["fpr"], int(params["min_mask_tokens"]), int(params["min_fragment_chars"]))

    within.sort(key=key, reverse=True)
    best_params = within[0][0]
    frontier = [{"params": params, "dev": result} for params, result in sorted(within, key=key, reverse=True)[:25]]
    selection = {
        "stage": stage,
        "target_fpr": target_fpr,
        "grid_size": len(evaluated),
        "points_within_target": len(ok(target_fpr)),
        "points_within_double_target": len(ok(min(1.0, 2 * target_fpr))),
        "rule": (
            "параметры выбираются на dev по объявленному правилу: максимум F1 при FPR ≤ цели, "
            "иначе ближайшая достижимая зона; тест в выборе не участвует"
        ),
    }
    return best_params, frontier, selection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Калибровка агрегации вердикта на dev, тест — один раз")
    parser.add_argument("--dataset", default="data/corpus_a3/pairs.jsonl")
    parser.add_argument("--splits", default=None, help="каталог официальных разбиений (по умолчанию рядом с корпусом)")
    parser.add_argument("--mode", choices=["demo", "hf"], default="demo")
    parser.add_argument("--model", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--target-fpr", type=float, default=0.1)
    parser.add_argument("--out", default="reports/verdict_calibration.json")
    parser.add_argument("--freeze", default=DEFAULT_FREEZE)
    parser.add_argument(
        "--also",
        default="data/corpus_a/pairs.jsonl",
        help="второй корпус для контроля: замороженное правило применяется как есть ('' — не проверять)",
    )
    parser.add_argument("--no-train", action="store_true", help="взять существующий бандл (для проверки правила)")
    parser.add_argument("--bundle", default="config/weights.json", help="бандл для --no-train")
    parser.add_argument(
        "--evidence-dir",
        default=None,
        help="каталог для таблиц свидетельств (dev/test) — позволяет пересчитывать правила без обучения",
    )
    parser.add_argument("--save-bundle", default=None, help="сохранить обученный бандл по указанному пути")
    parser.add_argument(
        "--features-cache",
        action="append",
        default=[],
        metavar="PATH",
        help="кеш предпосчитанных признаков (файл JSONL или каталог с шардами); " "можно указать несколько раз",
    )
    parser.add_argument(
        "--span-expansion",
        default="clause",
        choices=("clause", "mask", "clause_capped", "mask_padded"),
        help="стратегия расширения границ фрагментов (P0 шаг 1)",
    )
    args = parser.parse_args(argv)

    dataset = Path(args.dataset)
    splits_dir = Path(args.splits) if args.splits else dataset.parent / "splits"
    started = time.time()

    splits: dict[str, list[dict]] = {}
    for name in ("train", "dev", "test"):
        path = splits_dir / f"{name}.jsonl"
        if not path.is_file():
            print(f"нет официального разбиения {path}", file=sys.stderr)
            return 2
        splits[name] = list(read_pairs(path))
    print("разбиения: " + ", ".join(f"{name} {len(part)}" for name, part in splits.items()), flush=True)

    # Кеш признаков: в режиме hf без кеша каждый вызов verifier.verify() считает
    # признаки моделью заново (~6,4 с/пара × 1200 пар ≈ 2 часа). С кешем — минуты.
    cache = load_feature_cache(args.features_cache, counting=True, strict=True) if args.features_cache else None
    set_feature_cache(cache)
    if cache:
        print(f"кеш признаков: {len(cache)} ключей", flush=True)

    if args.no_train:
        verifier = Verifier(
            mode=args.mode, model_name=args.model, weights_path=args.bundle, features_cache=cache,
            span_expansion=args.span_expansion,
        )
        bundle_note = f"бандл из {args.bundle} (обучение не запускалось)"
    else:
        train_verifier = Verifier(
            mode=args.mode, model_name=args.model, features_cache=cache,
            span_expansion=args.span_expansion,
        )
        report = train(
            splits["train"],
            mode=args.mode,
            seed=args.seed,
            folds=args.folds,
            target_fpr=args.target_fpr,
            dataset_name=f"{dataset} (официальная обучающая часть)",
            verifier=train_verifier,
        )
        bundle = report.bundle
        verifier = Verifier(
            mode=args.mode, model_name=args.model, weights=bundle, features_cache=cache,
            span_expansion=args.span_expansion,
        )
        bundle_note = f"обучение на train ({len(splits['train'])} пар), seed {args.seed}, фолдов {args.folds}"

    if args.save_bundle:
        bundle_path = Path(args.save_bundle)
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        if args.no_train:
            print(f"бандл не сохраняется: взят готовый {args.bundle}", flush=True)
        else:
            verifier.bundle.save(bundle_path)
            print(f"бандл сохранён: {bundle_path}", flush=True)

    def dump_evidence(name: str, rows: list[dict[str, Any]]) -> None:
        if not args.evidence_dir:
            return
        target = Path(args.evidence_dir) / f"evidence_{Path(args.dataset).parent.name}_{name}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"таблица свидетельств: {target}", flush=True)

    dev_rows = evidence(verifier, splits["dev"])
    dump_evidence("dev", dev_rows)
    print(f"свидетельства dev: {len(dev_rows)} пар", flush=True)

    params, frontier, selection = choose(dev_rows, args.target_fpr)
    frozen = verdict_config_from_params(params)
    frozen["chosen_on"] = "dev"
    frozen["target_fpr"] = args.target_fpr
    print("выбрано на dev: " + json.dumps(params, ensure_ascii=False), flush=True)
    print("правило выбора: " + selection["stage"], flush=True)

    dev_rules = rules_report(dev_rows, params)
    print("dev, три правила на одной таблице: " + json.dumps(dev_rules, ensure_ascii=False), flush=True)

    # Тест — один раз и уже замороженным правилом.
    test_rows = evidence(verifier, splits["test"])
    dump_evidence("test", test_rows)
    test_rules = rules_report(test_rows, params)
    print("test (один прогон, правило заморожено): " + json.dumps(test_rules, ensure_ascii=False), flush=True)

    # Контроль на втором корпусе: замороженное правило применяется как есть,
    # ничего не подбирается. Нужно, чтобы видеть перенос правила на другой корпус.
    control: dict[str, Any] | None = None
    if args.also:
        also = Path(args.also)
        if also.is_file():
            control_rows = {}
            for name in ("dev", "test"):
                path = also.parent / "splits" / f"{name}.jsonl"
                if path.is_file():
                    control_rows[name] = evidence(verifier, list(read_pairs(path)))
            for name, rows in control_rows.items():
                dump_evidence(f"control_{name}", rows)
            if control_rows:
                control = {
                    "dataset": str(also),
                    "rules_by_split": {name: rules_report(rows, params) for name, rows in control_rows.items()},
                }
                print("контроль на втором корпусе: " + json.dumps(control, ensure_ascii=False), flush=True)
        else:
            print(f"контрольный корпус не найден: {also}", flush=True)

    payload = {
        "dataset": str(dataset),
        "splits": str(splits_dir),
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "dataset_sha256": _sha256(dataset) if dataset.is_file() else None,
        "mode": args.mode,
        "model": verifier.model_name,
        "seed": args.seed,
        "target_fpr": args.target_fpr,
        "bundle": bundle_note,
        "commit": _git_commit(),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duration_s": round(time.time() - started, 1),
        "command": "python scripts/calibrate_verdict.py " + " ".join(sys.argv[1:]),
        "rule": VERDICT_CORROBORATED,
        "params": params,
        "chosen_on": "dev",
        "selection": selection,
        "dev": dev_rules,
        "test": test_rules,
        "control_other_corpus": control,
        "frontier_dev": frontier,
        "rule_statement": (
            "вердикт — из прогона: параметры агрегации перебраны и выбраны на dev, заморожены в "
            f"{args.freeze}, test снят один раз после заморозки; старые правила посчитаны на той же таблице"
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.freeze:
        freeze_path = Path(args.freeze)
        freeze_path.parent.mkdir(parents=True, exist_ok=True)
        freeze_path.write_text(
            json.dumps({k: v for k, v in frozen.items() if k != "dev_metrics"}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"заморожено: {freeze_path}", flush=True)

    if cache is not None:
        hits = getattr(cache, "hits", 0)
        misses = getattr(cache, "misses", 0)
        print(f"кеш признаков: попаданий {hits}, промахов {misses}", flush=True)

    print(f"отчёт: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
