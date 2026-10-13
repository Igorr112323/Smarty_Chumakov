"""Замер уровня плотности признака на подтверждённых токенах (demo-режим).

Зачем: шкала ``embedding_density`` в demo-режиме — средняя схожесть с k
ближайшими словами документа; на подтверждённом токене она равна не 1, а своему
реальному уровню. Чтобы признак читался так же, как остальные признаки опоры
(1 — опора есть), его шкала нормируется на измеренную величину
``features.DEMO_DENSITY_REFERENCE``. Константа измеряется здесь — на
ОБУЧАЮЩЕЙ части корпуса, никогда на тестовой: порог и веса по-прежнему
выбираются только обучением.

Запуск (значение константы получено именно этой командой)::

    python scripts/density_reference.py --dataset data/corpus_a/splits/train.jsonl \\
        --out reports/DENSITY_REFERENCE.json --seed 42

Отчёт содержит: число пар и токенов, медиану и среднее «сырой» плотности для
токенов с лексической опорой и без неё, а также действующее значение
``DEMO_DENSITY_REFERENCE`` с указанием коммита и даты прогона.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spanverify.core import tokenize_with_offsets  # noqa: E402
from spanverify.dataset import read_pairs  # noqa: E402
from spanverify.features import (  # noqa: E402
    DEMO_DENSITY_REFERENCE,
    extract_features,
    is_scored_token,
)

ROOT = Path(__file__).resolve().parents[1]


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        return "unknown"


def measure(dataset: str, limit: int = 0) -> dict:
    pairs = list(read_pairs(dataset))
    if limit > 0:
        pairs = pairs[:limit]
    supported: list[float] = []
    unsupported: list[float] = []
    tokens = 0
    for pair in pairs:
        matrix = extract_features(pair["answer"], pair["context"], mode="demo")
        for index, token in enumerate(tokenize_with_offsets(pair["answer"])):
            if index >= len(matrix.embedding_density) or not is_scored_token(token.text):
                continue
            tokens += 1
            # Признак отдаётся уже нормированным; восстанавливаем «сырое»
            # значение, чтобы константа измерялась в исходных единицах.
            raw = matrix.embedding_density[index] * DEMO_DENSITY_REFERENCE
            if matrix.ctx_attention_mass[index] >= 0.999:
                supported.append(raw)
            else:
                unsupported.append(raw)
    return {
        "pairs": len(pairs),
        "tokens_scored": tokens,
        "supported_tokens": len(supported),
        "unsupported_tokens": len(unsupported),
        "supported_median": round(statistics.median(supported), 4) if supported else None,
        "supported_mean": round(statistics.fmean(supported), 4) if supported else None,
        "unsupported_median": round(statistics.median(unsupported), 4) if unsupported else None,
        "unsupported_mean": round(statistics.fmean(unsupported), 4) if unsupported else None,
        "constant_in_code": DEMO_DENSITY_REFERENCE,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Замер уровня плотности на подтверждённых токенах")
    parser.add_argument("--dataset", default="data/corpus_a/splits/train.jsonl")
    parser.add_argument("--out", default="reports/DENSITY_REFERENCE.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="ограничить число пар (для быстрых прогонов)")
    args = parser.parse_args()

    started = time.time()
    measured = measure(args.dataset, limit=args.limit)
    payload = {
        "kind": "density_reference",
        "commit": git_commit(),
        "seed": args.seed,
        "mode": "demo",
        "dataset": args.dataset,
        "command": (
            "python scripts/density_reference.py "
            f"--dataset {args.dataset} --out {args.out} --seed {args.seed}"
            + (f" --limit {args.limit}" if args.limit else "")
        ),
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_s": round(time.time() - started, 2),
        "note": (
            "замер на обучающей части; тестовая часть в измерении константы не участвует; "
            "сырая плотность восстановлена как значение признака × DEMO_DENSITY_REFERENCE"
        ),
        "metrics": measured,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{out}: пар {measured['pairs']}, токенов {measured['tokens_scored']}")
    print(
        f"подтверждённые: медиана {measured['supported_median']} (n={measured['supported_tokens']}); "
        f"без опоры: медиана {measured['unsupported_median']} (n={measured['unsupported_tokens']})"
    )
    print(f"константа в коде: {DEMO_DENSITY_REFERENCE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
