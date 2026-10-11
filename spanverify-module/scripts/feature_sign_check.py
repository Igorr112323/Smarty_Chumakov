#!/usr/bin/env python3
"""Знак и информативность каждого признака сетки на валидации (A3 dev).

Зачем отдельный скрипт. В ``docs/ПРИЗНАКИ.md`` зафиксировано: пилот на 96 парах
дал значимые отклонения AUC от 0,5, и все — **ниже** 0,5. AUC ниже 0,5 означает
либо что признак информативен с обратным знаком, либо что он не информативен.
Прежде чем писать «признак X информативен», знак надо измерить. Делать это надо
по валидации (A3 dev), а не по тесту: выбор знака по тесту = подгонка.

Что считается. Для каждого из 36 признаков сетки по токенам валидационной части:

* ``auc`` — площадь под ROC по определению Mann–Whitney с половинным весом связей
  (ранговый тест), знак исходный: «чем больше значение, тем вероятнее метка»;
* ``auc_inverted`` — ``1 - auc`` (признак с обратным знаком, то есть «вычтенный»);
* доверительные интервалы — кластерный бутстреп по парам (не по токенам: токены
  одной пары зависимы), percentiles,seed фиксирован;
* ``significant_original`` / ``significant_inverted`` — ДИ не накрывает 0,5;
* ``sign_by_val`` — «исходный», «инвертированный» или «не значим».

Линейная модель знака не боится (вес может быть отрицательным), поэтому этот
отчёт нужен не для обучения, а для формулировок: без него нельзя писать ни
«признак информативен», ни «знак перевёрнут».

Запуск (из ``spanverify-module``; в CI — job «Знак признаков на валидации»):

    python3 scripts/feature_sign_check.py --grid-cache reports/hf-cache-a3 \\
        --splits data/corpus_a3/splits --bootstrap 5000 \\
        --out-json reports/hf_protocol/feature_signs_a3dev.json \\
        --out-md reports/hf_protocol/feature_signs_a3dev.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))
if str(MODULE_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT / "scripts"))

import hf_protocol  # noqa: E402

from spanverify.hf_grid import GRID_FEATURE_NAMES  # noqa: E402

PROTOCOL = "feature-signs-v1"
RULE = "знак и значимость измеряются только на валидации (A3 dev); тестовая часть не читается"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auc_scores(scores: list[float], labels: list[int]) -> float:
    """Mann–Whitney AUC: P(оценка положительного выше оценки отрицательного), связи = 0,5."""
    pos = sum(1 for value in labels if value)
    neg = len(labels) - pos
    if not pos or not neg:
        raise ValueError("нужны оба класса")
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    cursor = 0
    while cursor < len(order):
        last = cursor
        while last + 1 < len(order) and scores[order[last + 1]] == scores[order[cursor]]:
            last += 1
        average = (cursor + last) / 2.0 + 1.0
        for step in range(cursor, last + 1):
            ranks[order[step]] = average
        cursor = last + 1
    sum_pos = sum(ranks[index] for index, value in enumerate(labels) if value)
    return (sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def cluster_bootstrap_ci(
    groups: list[tuple[list[float], list[int]]], iterations: int, seed: int
) -> tuple[float, float]:
    """Процентильный ДИ по кластерной пересэмплировке пар (токены внутри пары зависимы)."""
    if not groups:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(iterations):
        scores: list[float] = []
        labels: list[int] = []
        for _ in range(len(groups)):
            group = groups[rng.randrange(len(groups))]
            scores.extend(group[0])
            labels.extend(group[1])
        if sum(1 for value in labels if value) == 0 or all(labels):
            continue
        try:
            values.append(auc_scores(scores, labels))
        except ValueError:  # pragma: no cover - защита на вырожденных выборках
            continue
    if len(values) < max(20, iterations // 20):
        return (float("nan"), float("nan"))
    values.sort()
    low = values[int(0.025 * (len(values) - 1))]
    high = values[int(0.975 * (len(values) - 1))]
    return (low, high)


def load_val_records(splits_dir: Path) -> tuple[list[dict[str, Any]], Path]:
    """Только валидационная часть. Обращение к ``test.jsonl`` запрещено и проверяется явно."""
    val_path = hf_protocol.val_path_of(splits_dir)
    if val_path.name.startswith("test"):
        raise SystemExit(f"{val_path.name}: знак признака нельзя выбирать по тестовой части")
    return hf_protocol.load_split(val_path), val_path


def measure(args: argparse.Namespace) -> dict[str, Any]:
    splits_dir = Path(args.splits)
    records, val_path = load_val_records(splits_dir)
    cache_dir = Path(args.grid_cache)
    paths = hf_protocol._cache_paths(cache_dir)
    # Тестовая выборка v1 (A3 test) использована для финальной оценки и закрыта. Кеш сетки
    # лежит отдельными артефактами (`grid-test-*`), и случайное скачивание одного из них —
    # самый дешёвый способ испортить измерение: проверка по имени файла, а не по содержимому.
    forbidden = [p.name for p in paths if "test" in p.name.lower()]
    if forbidden:
        raise SystemExit(
            "в списке файлов кеша есть тестовая выборка: "
            + ", ".join(sorted(forbidden))
            + " — измерение знака делается только на валидации"
        )
    cache = hf_protocol.load_grid_cache(paths)
    rows = hf_protocol.prepare(records, cache, list(GRID_FEATURE_NAMES))
    labels = [int(row["gold"]) for row in rows]
    positives = sum(labels)
    if not positives or positives == len(labels):
        raise SystemExit("вырожденная разметка валидации: AUC не определена")

    # индексы токенов по парам — для кластерного бутстрапа
    by_pair: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_pair.setdefault(str(row["id"]), []).append(index)
    pair_groups = list(by_pair.values())

    features: list[dict[str, Any]] = []
    for name in GRID_FEATURE_NAMES:
        values = [float(row["features"].get(name, 0.0)) for row in rows]
        distinct = len(set(values))
        entry: dict[str, Any] = {"name": name, "tokens": len(values), "distinct_values": distinct}
        if distinct < 2:
            entry.update(
                {
                    "auc": None,
                    "auc_inverted": None,
                    "ci_low": None,
                    "ci_high": None,
                    "significant_original": False,
                    "significant_inverted": False,
                    "sign_by_val": "признак постоянен (AUC не определена)",
                }
            )
            features.append(entry)
            continue
        auc = auc_scores(values, labels)
        groups = [([values[index] for index in group], [labels[index] for index in group]) for group in pair_groups]
        low, high = (float("nan"), float("nan"))
        if args.bootstrap:
            low, high = cluster_bootstrap_ci(groups, args.bootstrap, args.seed)
        usable = not math.isnan(low)
        significant_original = bool(usable and low > 0.5)
        significant_inverted = bool(usable and high < 0.5)
        if not usable:
            sign = "не значим (бутстреп не дал интервала)"
        elif significant_original:
            sign = "исходный"
        elif significant_inverted:
            sign = "инвертированный"
        elif auc > 0.5:
            sign = "исходный (не значимо)"
        elif auc < 0.5:
            sign = "инвертированный (не значимо)"
        else:
            sign = "нет направления"
        entry.update(
            {
                "auc": round(auc, 6),
                "auc_inverted": round(1.0 - auc, 6),
                "ci_low": None if not usable else round(low, 6),
                "ci_high": None if not usable else round(high, 6),
                "significant_original": significant_original,
                "significant_inverted": significant_inverted,
                "sign_by_val": sign,
                "mean_positive": round(statistics.fmean(v for v, g in zip(values, labels, strict=True) if g), 6),
                "mean_negative": round(statistics.fmean(v for v, g in zip(values, labels, strict=True) if not g), 6),
            }
        )
        features.append(entry)

    payload = {
        "step": "feature-signs",
        "protocol": PROTOCOL,
        "rule": RULE,
        "split": {"name": "val (A3 dev)", "file": str(val_path), "pairs": len(records), "tokens": len(rows)},
        "labels": {"positive_tokens": positives, "negative_tokens": len(labels) - positives},
        "bootstrap_iterations": args.bootstrap,
        "seed": args.seed,
        "clusters": len(pair_groups),
        "model": hf_protocol.cache_model_provenance(cache),
        "inputs": {
            "grid_cache": str(cache_dir),
            "splits": str(splits_dir),
            "splits_sha256": sha256_file(val_path),
        },
        "auc_definition": "Mann–Whitney P(score(pos) > score(neg)), связи = 0,5; 0,5 — признака нет",
        "ci_definition": (
            f"процентильный кластерный бутстреп по парам, {args.bootstrap} итераций, seed {args.seed}; "
            "токены одной пары пересэмплируются вместе"
        ),
        "features": features,
        "counts": {
            "significant_original": sum(1 for item in features if item["significant_original"]),
            "significant_inverted": sum(1 for item in features if item["significant_inverted"]),
            "auc_below_half": sum(1 for item in features if item["auc"] is not None and item["auc"] < 0.5),
            "constant": sum(1 for item in features if item["distinct_values"] < 2),
        },
    }
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    split = payload["split"]
    lines = [
        "# Знак и информативность признаков сетки на валидации (A3 dev)",
        "",
        f"Правило: {payload['rule']}. Выборка: пар `{split['pairs']}`, токенов ответа `{split['tokens']}`"
        f" (положительных `{payload['labels']['positive_tokens']}`, отрицательных"
        f" `{payload['labels']['negative_tokens']}`). Модель: `{payload['model']['id']}`"
        f" (revision `{str(payload['model']['revision'])[:12]}…`). Бутстреп:"
        f" `{payload['bootstrap_iterations']}` повторов по парам, seed `{payload['seed']}`.",
        "",
        f"Определение AUC: {payload['auc_definition']} ДИ: {payload['ci_definition']}.",
        "",
        "| Признак | AUC (исходный знак) | AUC (инвертированный) | ДИ исходного | Значимо | Знак по val |",
        "|---|---|---|---|---|---|",
    ]
    for item in payload["features"]:
        if item["auc"] is None:
            lines.append(f"| `{item['name']}` | — | — | — | нет | {item['sign_by_val']} |")
            continue
        ci = f"[{item['ci_low']:.3f}; {item['ci_high']:.3f}]" if item["ci_low"] is not None else "нет интервала"
        significant = (
            "да, исходный"
            if item["significant_original"]
            else ("да, обратный" if item["significant_inverted"] else "нет")
        )
        lines.append(
            f"| `{item['name']}` | {item['auc']:.6f} | {item['auc_inverted']:.6f} | {ci} | {significant}"
            f" | {item['sign_by_val']} |"
        )
    counts = payload["counts"]
    lines += [
        "",
        f"Итог: AUC ниже 0,5 у `{counts['auc_below_half']}` из `{len(payload['features'])}` признаков;"
        f" значимо с исходным знаком — `{counts['significant_original']}`, значимо с обратным —"
        f" `{counts['significant_inverted']}`, постоянных (AUC не определена) — `{counts['constant']}`.",
        "",
        "Что из этого следует и чего не следует. Знак здесь нужен, чтобы честно прочитать"
        " AUC ниже 0,5: если ДИ лежит ниже 0,5, признак информативен с обратным знаком (и"
        " линейная модель сама даст ему отрицательный вес, а ручное «вычитание энтропии» не"
        " требуется). Если ДИ накрывает 0,5 — направление не установлено, и вывод об"
        " информативности делать нельзя. Никаких выводов о тестовой части этот файл не содержит.",
        "",
    ]
    return "\n".join(lines) + "\n"


def annotate(payload: dict[str, Any]) -> None:
    """Одна сводная аннотация прогона: числа можно прочитать без скачивания артефакта."""
    rows = [
        f"{item['name']}={item['auc']}/{item['auc_inverted']}/{item['sign_by_val']}" for item in payload["features"]
    ]
    message = " | ".join(rows)
    if len(message) > 60_000:  # pragma: no cover - защита от отказа публикации аннотации
        message = message[:60_000] + "…"
    print(f"::notice title=знак признаков (val, AUC/AUC_inv/знак):: {message}")
    counts = payload["counts"]
    print(
        "::notice title=итог по знакам:: "
        f"AUC<0,5: {counts['auc_below_half']}; значимо исходный: {counts['significant_original']}; "
        f"значимо обратный: {counts['significant_inverted']}; постоянных: {counts['constant']}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Знак и информативность признаков на валидации")
    parser.add_argument("--grid-cache", required=True, help="каталог или файл кеша сетки (v3)")
    parser.add_argument("--splits", default="data/corpus_a3/splits", help="каталог разбиения по документам")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", default="reports/hf_protocol/feature_signs_a3dev.json")
    parser.add_argument("--out-md", default="reports/hf_protocol/feature_signs_a3dev.md")
    parser.add_argument("--annotate", action="store_true", help="напечатать аннотации GitHub Actions")
    args = parser.parse_args(argv)

    payload = measure(args)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(payload), encoding="utf-8")
    counts = payload["counts"]
    print(
        f"признаков: {len(payload['features'])} | токенов val: {payload['split']['tokens']} | "
        f"AUC<0,5: {counts['auc_below_half']} | значимо исходный: {counts['significant_original']} | "
        f"значимо обратный: {counts['significant_inverted']}"
    )
    print(f"записано: {out_json} и {out_md}")
    if args.annotate:
        annotate(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
