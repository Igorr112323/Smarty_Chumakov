#!/usr/bin/env python3
"""Измерение времени одного ответа и потребления памяти — только измерение.

Скрипт закрывает п. 4.1 (7) и (10) технического задания: объём оперативной памяти и время
обработки одного ответа должны быть измерены при испытаниях и указаны в руководстве. Отдельного
инструмента для этого в репозитории не было: `scripts/functional_tests.py` измеряет рост по длине
контекста и память через `tracemalloc`, но не даёт медианы и 95-го перцентиля по пачке ответов.
Здесь — ровно то, что требует пункт: 50 ответов разной длины, медиана, p95, длина в токенах, пик
памяти на одну пару и на процесс целиком.

Ничего не подбирается и не меняется: пороги, веса и признаки читаются из поставки (`config/`), как
их видит обычный запуск. Скрипт не является проверкой: у него нет критерия и он не возвращает
ненулевой код — иначе измерение превратилось бы в гейт, привязанный к скорости конкретной машины.

    python3 scripts/measure_latency.py --pairs 50 --out reports/contract_check/latency.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spanverify.core import tokenize_with_offsets  # noqa: E402
from spanverify.engine import Verifier  # noqa: E402


def _rss_kib() -> int:
    """Пик резидентной памяти процесса, КиБ (Linux — КиБ, macOS — байты)."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Linux" else value / 1024)


def _cpu_info() -> tuple[int, str]:
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count() or 1
    model = "неизвестно"
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return count, model


def _mem_mib(key: str) -> float:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            name, _, rest = line.partition(":")
            if name.strip() == key:
                return round(int(rest.strip().split()[0]) / 1024, 1)
    except (OSError, ValueError):
        pass
    return 0.0


def _machine() -> dict[str, Any]:
    count, model = _cpu_info()
    try:
        load = [round(value, 2) for value in os.getloadavg()]
    except (AttributeError, OSError):
        load = None
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": count,
        "load_avg_1_5_15_min": load,
        "cpu_model": model,
        "mem_total_mib": _mem_mib("MemTotal"),
        "mem_available_mib": _mem_mib("MemAvailable"),
        "gpu": "отсутствует (nvidia-smi не найден)" if shutil.which("nvidia-smi") is None else "найден nvidia-smi",
        "torch": "установлен" if importlib.util.find_spec("torch") else "не установлен",
        "transformers": "установлен" if importlib.util.find_spec("transformers") else "не установлен",
    }


def _commit() -> str:
    done = subprocess.run(  # noqa: S603 - фиксированная команда, без пользовательского ввода
        ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
        cwd=ROOT.parent,
        check=False,
    )
    return done.stdout.strip() if done.returncode == 0 else "не git-дерево"


def pick_pairs(source: Path, count: int) -> list[dict[str, Any]]:
    """Ровно `count` пар, равномерно покрывающих диапазон длин ответов (детерминированно).

    Отбор идёт по индексам от первого до последнего элемента включительно: шаг считается как
    `(n − 1) / (count − 1)`, иначе самая длинная пара в выборку не попадает и фраза «ответы разной
    длины» не выполняется — при 50 из 1200 терялся бы весь верхний дециль.
    """
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if str(row.get("answer", "")).strip() and str(row.get("context", "")).strip()]
    rows.sort(key=lambda row: (len(str(row["answer"])), str(row.get("id", ""))))
    if count <= 1:
        return rows[:count]
    if len(rows) <= count:
        return rows
    step = (len(rows) - 1) / (count - 1)
    return [rows[int(round(index * step))] for index in range(count)]


def measure(verifier: Verifier, pairs: list[dict[str, Any]], warmup: int = 3) -> dict[str, Any]:
    for row in pairs[:warmup]:  # прогрев: первый вызов платит за импорт и чтение config/
        verifier.verify(str(row["answer"]), str(row["context"]))

    samples: list[dict[str, Any]] = []
    tracemalloc.start()
    for row in pairs:
        started = time.perf_counter()
        result = verifier.verify(str(row["answer"]), str(row["context"]))
        seconds = time.perf_counter() - started
        peak = tracemalloc.get_traced_memory()[1]
        tokens = len(tokenize_with_offsets(str(row["answer"])))
        samples.append(
            {
                "id": row.get("id", ""),
                "answer_chars": len(str(row["answer"])),
                "context_chars": len(str(row["context"])),
                "tokens": tokens,
                "seconds": round(seconds, 4),
                "ms_per_token": round(1000 * seconds / tokens, 3) if tokens else None,
                "tracemalloc_peak_kib": round(peak / 1024, 1),
                "score": result.score,
                "spans": len(result.spans),
                "verdict": result.verdict,
            }
        )
    batch_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    times = [sample["seconds"] for sample in samples]
    if len(times) >= 2:
        p95 = statistics.quantiles(times, n=100)[94]
    else:
        p95 = times[0]
    return {
        "pairs_measured": len(samples),
        "seconds": {
            "median": round(statistics.median(times), 4),
            "p95": round(p95, 4),
            "min": round(min(times), 4),
            "max": round(max(times), 4),
            "mean": round(statistics.fmean(times), 4),
        },
        "tokens": {
            "min": min(sample["tokens"] for sample in samples),
            "median": statistics.median([sample["tokens"] for sample in samples]),
            "max": max(sample["tokens"] for sample in samples),
        },
        "memory": {
            "tracemalloc_peak_kib_over_batch": round(batch_peak / 1024, 1),
            "tracemalloc_peak_kib_max_single_pair": max(sample["tracemalloc_peak_kib"] for sample in samples),
            "peak_rss_kib_after_imports_and_batch": _rss_kib(),
        },
        "samples": samples,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="измерение времени одного ответа и пика памяти")
    parser.add_argument("--pairs", type=int, default=50, help="сколько пар измерить (равномерно по длине ответа)")
    parser.add_argument("--source", default=None, help="корпус-источник пар (по умолчанию A3, иначе демо)")
    parser.add_argument("--mode", default=None, help="режим верификатора: demo, surrogate или hf")
    parser.add_argument("--weights", default="config/weights.json")
    parser.add_argument("--out", default="reports/contract_check/latency.json")
    args = parser.parse_args(argv)

    if args.source:
        source = Path(args.source)
    else:
        real = ROOT / "data" / "corpus_a3" / "pairs.jsonl"
        source = real if real.is_file() else ROOT / "data" / "demo_pairs.jsonl"
    if not source.is_file():
        print(f"источник пар не найден: {source}", file=sys.stderr)
        return 0

    rss_before = _rss_kib()
    verifier = Verifier(mode=args.mode, weights_path=args.weights)
    rss_after_load = _rss_kib()

    pairs = pick_pairs(source, args.pairs)
    if not pairs:
        print(f"в источнике нет пар с непустым ответом: {source}", file=sys.stderr)
        return 0

    payload = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_commit": _commit(),
        "source": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
        "mode": verifier.mode,
        "model_name": verifier.model_name,
        "machine": _machine(),
        "memory_load": {
            "peak_rss_kib_before_imports": rss_before,
            "peak_rss_kib_after_verifier_load": rss_after_load,
            "delta_kib": rss_after_load - rss_before,
        },
        "note": (
            "измерение, а не проверка: критерия нет; память — пик резидентной памяти процесса "
            "(resource.ru_maxrss) и пик Python-объектов (tracemalloc); время — однопоточный вызов "
            "Verifier.verify после трёх прогревочных вызовов"
        ),
        "result": measure(verifier, pairs),
    }

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    seconds = payload["result"]["seconds"]
    tokens = payload["result"]["tokens"]
    memory = payload["result"]["memory"]
    print(f"источник: {payload['source']}, режим: {payload['mode']}, пар: {payload['result']['pairs_measured']}")
    print(
        f"время одного ответа: медиана {seconds['median']} с, p95 {seconds['p95']} с, "
        f"мин {seconds['min']} с, макс {seconds['max']} с"
    )
    print(f"длины ответов в токенах: мин {tokens['min']}, медиана {tokens['median']}, макс {tokens['max']}")
    print(
        f"память: пик RSS {memory['peak_rss_kib_after_imports_and_batch']} КиБ, загрузка верификатора "
        f"+{payload['memory_load']['delta_kib']} КиБ, одна пара (tracemalloc) до "
        f"{memory['tracemalloc_peak_kib_max_single_pair']} КиБ"
    )
    print(f"записано: {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
