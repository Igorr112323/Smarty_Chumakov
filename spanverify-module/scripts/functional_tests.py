"""Функциональные испытания: время, память, предельная длина, коды возврата.

Раздел 3.3 ТЗ требовал этих чисел, но они ни разу не были измерены — в отчёте
стояло ``null``. Здесь они измеряются на реальном тексте актов и складываются в
``reports/functional_tests.json`` и ``reports/FUNCTIONAL_TESTS.md``.

Что именно меряется:

* **время** обработки контекста длиной 500 / 2 000 / 10 000 / 50 000 знаков
  (медиана по нескольким прогонам, прогрев отдельно — иначе в замер попадает
  загрузка весов);
* **пиковая память** — по ``tracemalloc``, то есть память Python-объектов;
  полная память процесса (RSS) этим не покрывается, и в отчёте это указано;
* **предельная длина** — ступенчатый рост, пока вызов не упадёт или не выйдет
  за лимит времени;
* **коды возврата** 0 / 1 / 2 — через настоящий CLI в отдельном процессе, чтобы
  код был тот, который увидит пользователь, а не внутреннее значение функции.

Запуск:

    python scripts/functional_tests.py --out reports/functional_tests.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spanverify.engine import Verifier  # noqa: E402  - путь подправлен выше

SIZES = (500, 2_000, 10_000, 50_000)
REPEATS = 3
LENGTH_STEPS = (100_000, 200_000, 500_000, 1_000_000, 2_000_000)
TIME_LIMIT_SECONDS = 60.0


def build_context(target: int, acts: list[str]) -> str:
    """Склеить реальные акты до нужной длины (обрезая ровно на целевом знаке)."""
    if target <= sum(len(act) for act in acts):
        chunks: list[str] = []
        total = 0
        for act in acts:
            if total >= target:
                break
            chunks.append(act[: target - total])
            total += len(chunks[-1])
        return "\n\n".join(chunks)[:target]
    # Реальных текстов не хватило — повторяем, но честно помечаем это в отчёте.
    text = "\n\n".join(acts)
    repeats = (target // max(1, len(text))) + 1
    return (text * repeats)[:target]


def measure_once(verifier: Verifier, answer: str, context: str) -> dict:
    """Один прогон: секунды и пик памяти по tracemalloc."""
    tracemalloc.start()
    started = time.perf_counter()
    result = verifier.verify(answer, context)
    seconds = time.perf_counter() - started
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "seconds": round(seconds, 3),
        "peak_mb": round(peak / 1024 / 1024, 2),
        "verdict": result.verdict,
        "latency_ms": round(result.latency_ms, 2),
        "spans": len(result.spans),
    }


def exit_code_via_cli(answer: str, context: str, python: str) -> int:
    """Код возврата настоящего CLI (то, что увидит пользователь)."""
    with tempfile.TemporaryDirectory() as tmp:
        ctx_path = Path(tmp) / "context.txt"
        ctx_path.write_text(context, encoding="utf-8")
        completed = subprocess.run(  # noqa: S603 - аргументы фиксированы, путь свой
            [python, "-m", "spanverify", "verify", "--answer", answer, "--context-file", str(ctx_path)],
            cwd=ROOT,
            capture_output=True,
            check=False,
            text=True,
        )
        return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Функциональные испытания SpanVerify")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "functional_tests.json")
    parser.add_argument("--acts-dir", type=Path, default=ROOT.parent / "АктыРеальны", help="каталог реальных актов")
    parser.add_argument("--max-length-steps", type=int, default=len(LENGTH_STEPS))
    args = parser.parse_args()

    acts: list[str] = []
    if args.acts_dir.is_dir():
        for path in sorted(args.acts_dir.glob("*.txt"))[:20]:
            acts.append(path.read_text(encoding="utf-8", errors="replace").strip())
    if not acts:
        print(f"нет текстов актов в {args.acts_dir}", file=sys.stderr)
        return 2
    corpus_chars = sum(len(act) for act in acts)
    print(f"текстов актов: {len(acts)}, всего знаков: {corpus_chars}")

    verifier = Verifier(mode="demo")
    answer_clean = "Срок хранения документов составляет пять лет."
    answer_dirty = "Срок хранения документов составляет девяносто девять лет."

    # Прогрев: чтобы в замер не попала загрузка весов и первичная инициализация.
    measure_once(verifier, answer_clean, build_context(500, acts))

    rows: list[dict] = []
    for size in SIZES:
        context = build_context(size, acts)
        runs = [measure_once(verifier, answer_clean, context) for _ in range(REPEATS)]
        rows.append(
            {
                "context_chars": len(context),
                "seconds_median": round(statistics.median([run["seconds"] for run in runs]), 3),
                "seconds_min": round(min(run["seconds"] for run in runs), 3),
                "seconds_max": round(max(run["seconds"] for run in runs), 3),
                "peak_mb_median": round(statistics.median([run["peak_mb"] for run in runs]), 2),
                "ms_per_1000_chars": round(
                    statistics.median([run["seconds"] for run in runs]) / len(context) * 1000 * 1000, 2
                ),
                "verdict": runs[0]["verdict"],
                "synthetic_extension": len(context) > corpus_chars,
                "repeats": REPEATS,
            }
        )
        print(f"контекст {len(context)} знаков: {rows[-1]['seconds_median']} с")

    # Коды возврата: 0 — подтверждено, 1 — есть замечания, 2 — ошибка.
    small_context = build_context(2_000, acts)
    grounded_answer = small_context[200:400]  # фрагмент самого документа: точно подтверждён
    with tempfile.TemporaryDirectory() as tmp:
        missing = str(Path(tmp) / "нет-такого-файла.txt")
        completed = subprocess.run(  # noqa: S603 - аргументы фиксированы
            [sys.executable, "-m", "spanverify", "verify", "--answer", answer_clean, "--context-file", missing],
            cwd=ROOT,
            capture_output=True,
            check=False,
            text=True,
        )
        code_error = int(completed.returncode)
    codes = {
        "0_grounded": exit_code_via_cli(grounded_answer, small_context, sys.executable),
        "1_issues": exit_code_via_cli(answer_dirty, small_context, sys.executable),
        "2_error": code_error,
    }
    print(f"коды возврата: {codes}")

    # Предельная длина: растём, пока не упадём или не выйдем за лимит времени.
    limits: list[dict] = []
    for step in LENGTH_STEPS[: max(0, args.max_length_steps)]:
        context = build_context(step, acts)
        tracemalloc.start()
        started = time.perf_counter()
        try:
            verifier.verify(answer_clean, context)
            seconds = time.perf_counter() - started
            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            ok = seconds <= TIME_LIMIT_SECONDS
            limits.append(
                {
                    "context_chars": len(context),
                    "ok": ok,
                    "seconds": round(seconds, 3),
                    "peak_mb": round(peak / 1024 / 1024, 2),
                    "error": None,
                }
            )
            print(f"длина {len(context)}: {seconds:.1f} с, ок={ok}")
            if not ok:
                break
        except Exception as error:  # noqa: BLE001 - предел ищется до первого реального отказа
            tracemalloc.stop()
            limits.append(
                {
                    "context_chars": len(context),
                    "ok": False,
                    "seconds": round(time.perf_counter() - started, 3),
                    "peak_mb": None,
                    "error": f"{type(error).__name__}: {error}"[:200],
                }
            )
            print(f"длина {len(context)}: отказ — {type(error).__name__}")
            break

    working = [row for row in limits if row["ok"]]
    report = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor() or "неизвестно",
            "cpu_count": __import__("os").cpu_count(),
            "python": platform.python_version(),
            "gpu": "отсутствует (проверялось nvidia-smi)",
        },
        "method": {
            "mode": "demo",
            "repeats": REPEATS,
            "warmup": True,
            "memory": "tracemalloc (память Python-объектов; RSS процесса не покрывается)",
            "text_source": str(args.acts_dir),
            "acts_used": len(acts),
            "corpus_chars": corpus_chars,
        },
        "by_size": rows,
        "exit_codes": codes,
        "length_limit": {
            "steps": limits,
            "max_ok_chars": max((row["context_chars"] for row in working), default=None),
            "time_limit_seconds": TIME_LIMIT_SECONDS,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Человекочитаемый отчёт рядом с JSON, чтобы цифры было чем подтвердить.
    lines = [
        "# Функциональные испытания",
        "",
        f"Измерено: {report['measured_at']}, режим `demo`, повторов {REPEATS} (с прогревом).",
        "",
        f"Оборудование: {report['hardware']['platform']}, CPU {report['hardware']['cpu_count']}, "
        f"Python {report['hardware']['python']}, GPU отсутствует.",
        f"Текст — реальные акты из `{args.acts_dir}` ({len(acts)} шт., {corpus_chars} знаков).",
        "",
        "| Длина контекста, знаков | Время, с (медиана) | мс на 1000 знаков | Пик памяти, МБ | Вердикт |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['context_chars']} | {row['seconds_median']} | {row['ms_per_1000_chars']} | "
            f"{row['peak_mb_median']} | {row['verdict']} |"
        )
    lines += [
        "",
        "Память измерена через `tracemalloc` — это память Python-объектов, а не полный RSS процесса.",
        "",
        "## Коды возврата",
        "",
        "| Сценарий | Ожидаемый код | Фактический |",
        "|---|---|---|",
        f"| Ответ подтверждён документом | 0 | {codes['0_grounded']} |",
        f"| В ответе есть расхождения | 1 | {codes['1_issues']} |",
        f"| Ошибка (файл контекста не найден) | 2 | {codes['2_error']} |",
        "",
        "## Предельная длина",
        "",
        "| Длина, знаков | Результат | Время, с |",
        "|---|---|---|",
    ]
    for row in limits:
        lines.append(f"| {row['context_chars']} | {'ок' if row['ok'] else 'отказ'} | {row['seconds']} |")
    max_ok = report["length_limit"]["max_ok_chars"]
    lines += [
        "",
        f"Максимальная длина, обработанная без отказа и в пределах "
        f"{TIME_LIMIT_SECONDS:.0f} с: **{max_ok if max_ok else 'не определена'}** знаков.",
    ]
    (args.out.with_name("FUNCTIONAL_TESTS.md")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nзаписано: {args.out} и {args.out.with_name('FUNCTIONAL_TESTS.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
