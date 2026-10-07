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
from spanverify.features import HF_MODEL_DEFAULT, extract_features, feature_cache_key  # noqa: E402

MODEL_DEFAULT = HF_MODEL_DEFAULT
PROGRESS_EVERY = 10


def dump_matrix(key: str, matrix, pair_id: str) -> dict:
    """Строка кеша: ключ и три рабочих признака."""
    return {
        "key": key,
        "pair_id": pair_id,
        "attention_entropy": [float(x) for x in matrix.attention_entropy],
        "ctx_attention_mass": [float(x) for x in matrix.ctx_attention_mass],
        "embedding_density": [float(x) for x in matrix.embedding_density],
    }


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
    args = parser.parse_args(argv)

    if args.shards < 1 or not (0 <= args.shard < args.shards):
        parser.error(f"--shard должен быть в [0, {args.shards - 1}]")

    dataset = Path(args.dataset)
    if not dataset.is_file():
        print(f"корпус не найден: {dataset}", file=sys.stderr)
        return 2

    records = list(read_pairs(dataset))
    if args.limit:
        records = records[: args.limit]
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
                out.write(json.dumps(dump_matrix(key, matrix, pair_id), ensure_ascii=False) + "\n")
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


if __name__ == "__main__":
    raise SystemExit(main())
