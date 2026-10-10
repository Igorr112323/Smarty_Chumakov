"""Предпосчёт признаков реальной модели в кеш: один прямой проход на пару.

Зачем отдельный скрипт
----------------------

Счёт признаков режимом ``hf`` — это прямой проход по языковой модели. На CPU
одна пара стоит секунды, а в кросс-валидации с ``--folds 5`` признаки каждой
пары требуются в каждом фолде. Без кеша эксперимент на корпусе A3 (1298 пар)
не укладывался в лимит job'а: прогон 37446204812 выбрал 120 минут целиком,
потому что считал одни и те же признаки пять раз.

Кеш делает дорогой проход однократным. Дальше ``run_experiments.py`` читает
готовые признаки (``--features-cache``), и кросс-валидация стоит минуты, а не
часы.

Шарды
-----

Если и одного прохода мало, корпус делится: ``--shard 0 --shards 4`` берёт
каждую четвёртую пару, начиная с нулевой. Шарды независимы, считаются
параллельными job'ами и склеиваются простым объединением файлов — порядок в
кеше не важен, обращение идёт по ключу.

Ключ кеша — ``feature_cache_key`` (ответ + контекст + режим + модель). Он же
используется верификатором при чтении, поэтому расхождение невозможно: считать
и читать будет один и тот же код.

Режим сетки (``--grid``)
-----------------------

Для протокола измерения (``docs/METRIC_SPEC.md``) нужны широкие признаки —
несколько слоёв, агрегации по головам, разные ``k`` плотности, лексика и числа.
Один проход по модели отдаёт их все сразу (``spanverify.hf_grid``), поэтому
флаг ``--grid`` пишет формат v3: ``grid_*.jsonl`` с полями ``arrays`` (36
массивов по токенам ответа), ``tokens`` (текст и смещения — чтобы сопоставить
признаки с разметкой, не пересобирая токенизатор) и ``meta`` (слой, длина
последовательности, число несопоставленных токенов, revision модели). Строки
раскладываются по файлам частей разбиения (``--split-files``), потому что
обучающая часть и отложенный test считаются разными job'ами и в разном порядке
доступа: test появляется только в шаге 3.

Выход
-----

``<out>/features_shard{K}of{N}.jsonl`` — по строке на пару;
``<out>/skipped_shard{K}of{N}.jsonl`` — пары, которые не посчитались;
``<out>/precompute_shard{K}of{N}.json`` — сводка шарда (пар, секунд, модель).

Ненулевой код возврата, если были пропуски: метрики на неполном корпусе были бы
искажением, а не результатом.
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
from spanverify.features import (  # noqa: E402
    CACHE_ARRAY_FIELDS,
    HF_MODEL_DEFAULT,
    extract_features,
    feature_cache_key,
)
from spanverify.hf_grid import GRID_CACHE_FORMAT, GRID_FEATURE_NAMES  # noqa: E402

MODEL_DEFAULT = HF_MODEL_DEFAULT
PROGRESS_EVERY = 10


def dump_matrix(key: str, matrix, pair_id: str, mode: str = "hf", model: str = "") -> dict:
    """Строка кеша: ключ и признаки.

    Формат v2 — сохранены все посчитанные массивы, а не только три рабочих
    признака. Это то, что делает отбор признаков на реальной модели вообще
    возможным: прямой проход по весам считается часами, и менять состав
    признаков, перезапуская его на каждый вариант, nobody может. Плюс
    служебные поля (режим, модель, формат, слой), по которым проверяется
    соответствие кеша запросу: кеш, посчитанный другой моделью или старой
    версией скрипта, должен быть отвергнут явно (см. validate_feature_cache).
    """
    row: dict = {
        "key": key,
        "pair_id": pair_id,
        "mode": mode,
        "cache_format": 2,
    }
    if model:
        row["model"] = model
    for name, field in CACHE_ARRAY_FIELDS.items():
        values = getattr(matrix, field, None) or []
        if values:
            row[name] = [float(x) for x in values]
    # Слой внимания: признаки разных слоёв несопоставимы, и в отчёте это должно
    # быть видно даже если meta матрицы потерялась (кеш читается без модели).
    layer = (getattr(matrix, "meta", None) or {}).get("layer")
    if layer is not None:
        row["layer"] = layer
    return row


def dump_grid_row(key: str, pair_id: str, result: dict, model: str, revision: str = "") -> dict:
    """Строка кеша сетки (v3): признаки, токены ответа и метаданные прогона.

    ``tokens`` пишутся вместе с массивами намеренно: потребитель, который
    пересобирает токенизатор сам, рискует разъехаться с признаками на один
    токен — и метрики при этом выглядят правдоподобно.
    """
    row: dict = {
        "key": key,
        "pair_id": pair_id,
        "mode": "hf",
        "grid_format": GRID_CACHE_FORMAT,
        "cache_format": GRID_CACHE_FORMAT,
        "model": model,
        # Revision весов рядом с признаками: числа протокола обязаны быть
        # привязаны к конкретной ревизии модели, а не только к её имени.
        "revision": str(revision or ""),
        "arrays": {name: [float(value) for value in result["arrays"].get(name, [])] for name in GRID_FEATURE_NAMES},
        "tokens": list(result["tokens"]),
        "meta": dict(result["meta"]),
    }
    return row


def parse_split_files(spec: str) -> list[tuple[str, Path]]:
    """``train=path,dev=path`` → [(имя части, файл)]. Имя входит в имя файла выхода."""
    out: list[tuple[str, Path]] = []
    for item in str(spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--split-files ждёт формат имя=путь, получено «{item}»")
        out.append((name.strip(), Path(path.strip())))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Предпосчёт признаков режима hf в кеш")
    parser.add_argument("--dataset", default="data/corpus_a3/pairs.jsonl")
    parser.add_argument("--out", default="reports/hf-cache")
    parser.add_argument("--mode", choices=["demo", "hf"], default="hf")
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--shard", type=int, default=0, help="номер шарда, начиная с нуля")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="взять только первые N пар (0 — все)")
    parser.add_argument("--progress-every", type=int, default=PROGRESS_EVERY)
    parser.add_argument(
        "--participation",
        action="store_true",
        help="дополнительно посчитать признаки синтетического корпуса доли участия ИИ "
        "(240 пар) — без них обучение головы участия в режиме hf пропускается",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="режим сетки: широкие признаки за один проход (spanverify.hf_grid), формат v3",
    )
    parser.add_argument(
        "--split-files",
        default="",
        help="части разбиения списком «имя=путь» через запятую: выход — grid_<имя>_<шард>.jsonl",
    )
    parser.add_argument("--max-length", type=int, default=1024, help="обрезка последовательности для режима сетки")
    parser.add_argument("--k-values", default="1,3,5", help="k для плотности представлений (режим сетки)")
    parser.add_argument("--window", type=int, default=1, help="радиус сглаживания соседами (режим сетки)")
    args = parser.parse_args(argv)

    if args.shards < 1 or not (0 <= args.shard < args.shards):
        parser.error(f"--shard должен быть в [0, {args.shards - 1}]")

    dataset = Path(args.dataset)
    if not dataset.is_file():
        print(f"корпус не найден: {dataset}", file=sys.stderr)
        return 2

    if args.grid:
        return _run_grid(args, dataset)

    records = list(read_pairs(dataset))
    if args.limit:
        records = records[: args.limit]
    if args.participation:
        # Корпус участия ИИ — свои тексты, и в кеше их нет: обучение головы
        # участия в режиме hf тогда пропускается (считать их моделью посреди
        # обучения — второй проход, которого кеш как раз и избегает). Предпосчитанные
        # здесь, они дают participation_hf.json без единого лишнего прохода.
        from spanverify.participation import build_participation_corpus

        extra = [
            {"id": f"participation-{index:04d}", "answer": item["text"], "context": item["context"]}
            for index, item in enumerate(build_participation_corpus(count=240, seed=42 + 1985))
        ]
        records = records + extra
    selected = [(index, record) for index, record in enumerate(records) if index % args.shards == args.shard]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"shard{args.shard}of{args.shards}" if args.shards > 1 else "all"
    features_path = out_dir / f"features_{suffix}.jsonl"
    skipped_path = out_dir / f"skipped_{suffix}.jsonl"
    summary_path = out_dir / f"precompute_{suffix}.json"

    print(
        f"шард {args.shard}/{args.shards}: пар в шарде {len(selected)} из {len(records)} "
        f"({dataset}), режим {args.mode}, модель {args.model}",
        flush=True,
    )

    started = time.time()
    done = 0
    skipped = 0
    with features_path.open("w", encoding="utf-8") as out, skipped_path.open("w", encoding="utf-8") as bad:
        for position, (index, record) in enumerate(selected, start=1):
            data = record.to_dict() if hasattr(record, "to_dict") else dict(record)
            answer = str(data.get("answer", ""))
            context = data.get("context", "")
            pair_id = str(data.get("id", index))
            # model_name понимает только режим hf: demo-признаки лексические и
            # модели у них нет (иначе demo_features упадёт на лишнем аргументе).
            model_name = args.model if args.mode == "hf" else ""
            call_kwargs = {"model_name": model_name} if args.mode == "hf" else {}
            try:
                matrix = extract_features(answer, context, mode=args.mode, **call_kwargs)
                key = feature_cache_key(answer, context, args.mode, model_name)
                row = dump_matrix(key, matrix, pair_id, mode=args.mode, model=model_name)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                done += 1
            except Exception as exc:  # noqa: BLE001 - причина записывается в файл пропусков
                skipped += 1
                bad.write(
                    json.dumps(
                        {"pair_id": pair_id, "error": f"{type(exc).__name__}: {exc}"},
                        ensure_ascii=False,
                    )
                    + "\n",
                )
                bad.flush()
                print(f"  пропуск {pair_id}: {type(exc).__name__}: {exc}", flush=True)
            if position % args.progress_every == 0 or position == len(selected):
                elapsed = time.time() - started
                per_pair = elapsed / max(1, done + skipped)
                left = (len(selected) - position) * per_pair
                print(
                    f"  [{position}/{len(selected)}] сделано {done}, пропущено {skipped}, "
                    f"{per_pair:.2f} с/пару, осталось ≈{left / 60:.1f} мин",
                    flush=True,
                )

    elapsed = time.time() - started
    summary = {
        "dataset": str(dataset),
        "mode": args.mode,
        "model": args.model,
        "cache_format": 2,
        "fields": sorted(CACHE_ARRAY_FIELDS),
        "shard": args.shard,
        "shards": args.shards,
        "pairs_total": len(records),
        "pairs_in_shard": len(selected),
        "pairs_done": done,
        "pairs_skipped": skipped,
        "seconds": round(elapsed, 1),
        "seconds_per_pair": round(elapsed / max(1, len(selected)), 3),
        "features_file": features_path.name,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    if skipped:
        print(f"ВНИМАНИЕ: пропущено пар {skipped} — метрики были бы неполными", file=sys.stderr)
        return 1
    return 0


def _run_grid(args, dataset: Path) -> int:
    """Одна загрузка модели на весь прогон, строки v3 по частям разбиения.

    Модель грузится ровно один раз: ``hf_features`` делает это на каждую пару,
    и на тысяче пар праздное ожидание сравнимо по времени с самим счётом.
    """
    from spanverify.features import feature_cache_key
    from spanverify.hf_grid import compute_grid, grid_model_info, load_grid_model

    targets: list[tuple[str, list[dict]]] = []
    for name, path in parse_split_files(args.split_files) or [("", dataset)]:
        if not path.is_file():
            print(f"файл не найден: {path}", file=sys.stderr)
            return 2
        records = [(record.to_dict() if hasattr(record, "to_dict") else dict(record)) for record in read_pairs(path)]
        if args.limit:
            records = records[: args.limit]
        targets.append((name, records))
    total = sum(len(records) for _name, records in targets)
    selected: list[tuple[str, int, dict]] = []
    for name, records in targets:
        for index, record in enumerate(records):
            if index % args.shards == args.shard:
                selected.append((name, index, record))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"shard{args.shard}of{args.shards}" if args.shards > 1 else "all"
    k_values = tuple(int(value) for value in str(args.k_values).split(",") if value.strip()) or (1, 3, 5)

    print(
        f"сетка v{GRID_CACHE_FORMAT}: пар в прогоне {len(selected)} из {total}, модель {args.model}, "
        f"признаков {len(GRID_FEATURE_NAMES)}",
        flush=True,
    )
    loaded = load_grid_model(args.model)
    info = grid_model_info(loaded)
    handles: dict = {}  # файловые объекты по частям разбиения
    started = time.time()
    done = skipped = 0
    per_split: dict[str, int] = {}
    unmatched_total = 0
    truncated_total = 0
    try:
        for name, _index, record in selected:
            answer = str(record.get("answer", ""))
            context = record.get("context", "")
            pair_id = str(record.get("id") or "")
            label = name or "pairs"
            features_path = out_dir / f"grid_{label}_{suffix}.jsonl"
            if label not in handles:
                handles[label] = features_path.open("a", encoding="utf-8")
            handle = handles[label]
            try:
                result = compute_grid(
                    answer,
                    context,
                    loaded,
                    max_length=args.max_length,
                    k_values=k_values,
                    window=args.window,
                )
                key = feature_cache_key(answer, context, "hf", args.model)
                row = dump_grid_row(key, pair_id, result, args.model, str(info.get("revision") or ""))
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                done += 1
                per_split[label] = per_split.get(label, 0) + 1
                unmatched_total += int(result["meta"].get("unmatched_tokens", 0))
                if int(result["meta"].get("seq_len", 0)) >= args.max_length:
                    truncated_total += 1
            except Exception as exc:  # noqa: BLE001 - причина пишется в файл пропусков
                skipped += 1
                print(f"  пропуск {pair_id}: {type(exc).__name__}: {exc}", flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    elapsed = time.time() - started
    summary = {
        "mode": "grid",
        "grid_format": GRID_CACHE_FORMAT,
        "dataset": str(dataset),
        "split_files": args.split_files,
        "model": args.model,
        "model_info": info,
        "feature_names": list(GRID_FEATURE_NAMES),
        "max_length": args.max_length,
        "k_values": list(k_values),
        "window": args.window,
        "shard": args.shard,
        "shards": args.shards,
        "pairs_total": total,
        "pairs_done": done,
        "pairs_skipped": skipped,
        "pairs_per_split": per_split,
        "unmatched_tokens": unmatched_total,
        "truncated_pairs": truncated_total,
        "seconds": round(elapsed, 1),
        "seconds_per_pair": round(elapsed / max(1, done + skipped), 3),
    }
    # Имя сводки включает части прогона: артефакты шардов и корпусов сливаются в
    # один каталог (`merge-multiple: true`), и одинаковые имена затёрли бы друг друга.
    labels = "_".join(str(name) for name, _records in targets if name) or "pairs"
    (out_dir / f"grid_summary_{suffix}_{labels}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if skipped:
        print(f"ВНИМАНИЕ: пропущено пар {skipped} — метрики были бы неполными", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
