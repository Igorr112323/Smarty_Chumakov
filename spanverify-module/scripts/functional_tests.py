"""Функциональные испытания лабораторного образца (пункт 4.4 промта).

Измеряется ровно то, что обещано в отчёте: размеры поставки, время холодного
запуска до ``GET /health``, время обработки пары текстов разной длины в двух
режимах, пиковая память и диск, максимальная длина текста, коды возврата,
смоук-тест API и утечка памяти на сокращённом цикле.

Все числа — фактические, получены на том железе, где запущен скрипт; ОС, CPU, RAM
и версия Python пишутся в отчёт. Если артефактов сборки (.exe/.pyz/.zip) нет, в
отчёте стоит ``null`` и причина, а не выдуманное число.

Запуск::

    python scripts/functional_tests.py --mode demo --lengths 500,2000,10000,50000 \\
        --cycle-seconds 120 --out reports/FUNCTIONAL_TESTS.md
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spanverify._version import __version__  # noqa: E402

SENTENCES = (
    "Срок хранения первичных учётных документов составляет 5 лет.",
    "Срок хранения личных карточек работников составляет 75 лет.",
    "Запрещается выносить документы за пределы архивохранилища.",
    "Обращение гражданина рассматривается в течение 30 календарных дней.",
    "Ответственность за хранение возложена на службу делопроизводства.",
)


def make_pair(target_chars: int) -> tuple[str, str]:
    """Пара «контекст — ответ» заданной длины (в символах)."""
    buffer: list[str] = []
    size = 0
    index = 0
    while size < target_chars:
        sentence = SENTENCES[index % len(SENTENCES)]
        if index % len(SENTENCES) == 0:
            sentence = sentence.replace("5 лет", f"{5 + (index // len(SENTENCES)) % 40} лет")
        buffer.append(sentence)
        size += len(sentence) + 1
        index += 1
    context = " ".join(buffer)
    answer = (
        f"Срок хранения первичных учётных документов составляет {5 + (index // len(SENTENCES)) % 40} лет. "
        "Документы хранятся в архивохранилище организации."
    )
    return context, answer


def file_sizes() -> dict[str, dict[str, object]]:
    """Размеры артефактов сборки в байтах (или null с причиной)."""
    candidates = {
        "exe": ROOT / "release" / "spanverify.exe",
        "pyz": ROOT / "release" / "spanverify.pyz",
        "zip": ROOT / "release" / "spanverify-win.zip",
    }
    sizes: dict[str, dict[str, object]] = {}
    for name, path in candidates.items():
        if path.is_file():
            sizes[name] = {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "reason": None}
        else:
            sizes[name] = {"path": str(path.relative_to(ROOT)), "bytes": None, "reason": "артефакт не собран"}
    return sizes


def peak_memory_mb() -> float:
    """Пиковая память процесса в мегабайтах (RSS по данным ОС)."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux: килобайты, macOS: байты.
    return usage / 1024 if sys.platform != "darwin" else usage / (1024 * 1024)


def directory_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return round(total / (1024 * 1024), 2)


def measure_processing(lengths: Sequence[int], mode: str, model: str | None) -> list[dict]:
    """Время и память обработки пары текстов заданной длины."""
    from spanverify import Verifier

    rows: list[dict] = []
    verifier = Verifier(mode=mode, model_name=model)
    for length in lengths:
        context, answer = make_pair(length)
        started = time.perf_counter()
        result = verifier.verify(answer, context)
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "chars": len(context),
                "answer_chars": len(answer),
                "mode": mode,
                "seconds": round(elapsed, 4),
                "verdict": result.verdict,
                "spans": len(result.spans),
                "peak_memory_mb": round(peak_memory_mb(), 1),
            }
        )
    return rows


def max_length_probe(mode: str, model: str | None, limits: Sequence[int]) -> dict:
    """Максимальная длина текста с успешной обработкой и поведение при превышении."""
    from spanverify import Verifier

    verifier = Verifier(mode=mode, model_name=model)
    table: list[dict] = []
    for limit in limits:
        context, answer = make_pair(limit)
        started = time.perf_counter()
        try:
            result = verifier.verify(answer, context)
            status = "ok"
            verdict = result.verdict
            note = ""
        except MemoryError as exc:  # pragma: no cover - зависит от железа
            status = "MemoryError"
            verdict = None
            note = str(exc)[:120]
        except Exception as exc:  # noqa: BLE001 - отчёт должен фиксировать факт
            status = type(exc).__name__
            verdict = None
            note = str(exc)[:120]
        table.append(
            {
                "chars": len(context),
                "status": status,
                "verdict": verdict,
                "seconds": round(time.perf_counter() - started, 4),
                "note": note,
            }
        )
    ok = [row for row in table if row["status"] == "ok"]
    return {
        "attempts": table,
        "max_ok_chars": max((row["chars"] for row in ok), default=0),
        "behavior_on_excess": next((row for row in table if row["status"] != "ok"), None),
    }


def exit_code_probe() -> list[dict]:
    """Коды возврата verify: 0 — подтверждено, 1 — спорно, 2 — ошибка вызова."""
    cases = [
        (
            "подтверждённый ответ",
            [
                "-m",
                "spanverify",
                "verify",
                "--json",
                "--answer",
                "Срок хранения составляет 5 лет.",
                "--context",
                "Срок хранения составляет 5 лет.",
            ],
        ),
        (
            "спорный ответ",
            [
                "-m",
                "spanverify",
                "verify",
                "--json",
                "--answer",
                "Срок хранения составляет 5 лет.",
                "--context",
                "Срок хранения составляет 30 лет.",
            ],
        ),
        ("ошибка вызова", ["-m", "spanverify", "verify", "--json"]),
    ]
    rows: list[dict] = []
    for name, args in cases:
        process = subprocess.run([sys.executable, *args], capture_output=True, text=True, check=False)
        rows.append(
            {
                "case": name,
                "command": "python " + " ".join(args),
                "exit_code": process.returncode,
                "stderr_head": (process.stderr or "").strip().splitlines()[:1],
                "has_traceback": "Traceback" in (process.stderr or ""),
            }
        )
    return rows


def smoke_test(port: int, mode: str) -> dict:
    """Смоук-тест: GET /health → POST /v1/verify, проверка полей контракта."""
    payload = json.dumps(
        {
            "answer": "Срок хранения первичных учётных документов составляет 5 лет.",
            "context": "Срок хранения первичных учётных документов составляет 5 лет.",
            "mode": mode,
        }
    ).encode("utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "spanverify",
            "server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--no-browser",
            "--quiet",
            "--mode",
            mode,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(ROOT),
    )
    started = time.perf_counter()
    cold_start = None
    health = None
    response: dict | None = None
    error = None
    try:
        for _ in range(120):
            time.sleep(0.5)
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as answer:
                    health = json.loads(answer.read().decode("utf-8"))
                cold_start = time.perf_counter() - started
                break
            except (urllib.error.URLError, TimeoutError, OSError):
                continue
        if health is not None:
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/verify",
                data=payload,
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
            with urllib.request.urlopen(request, timeout=120) as answer:
                response = json.loads(answer.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - факт важнее исключения
        error = f"{type(exc).__name__}: {exc}"
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:  # pragma: no cover
            process.kill()
    required = ("score", "spans", "ai_share", "ai_participation", "threshold")
    return {
        "cold_start_seconds": round(cold_start, 3) if cold_start else None,
        "health": health,
        "fields_present": {name: bool(response and name in response) for name in required} if response else {},
        "verdict": (response or {}).get("verdict"),
        "grounded_on_confirmed_answer": (response or {}).get("verdict") == "grounded",
        "error": error,
    }


def leak_probe(seconds: int, mode: str) -> dict:
    """Сокращённый цикл непрерывной работы с замером роста памяти.

    Полные 8 часов в CI не прогоняются: в отчёте указывается фактическая
    длительность цикла и оценка роста памяти за минуту.
    """
    from spanverify import Verifier

    verifier = Verifier(mode=mode)
    context, answer = make_pair(2000)
    samples: list[tuple[float, float]] = []
    started = time.perf_counter()
    iterations = 0
    while time.perf_counter() - started < seconds:
        verifier.verify(answer, context)
        iterations += 1
        if iterations % 10 == 0:
            samples.append((time.perf_counter() - started, peak_memory_mb()))
    duration = time.perf_counter() - started
    first = samples[0][1] if samples else peak_memory_mb()
    last = samples[-1][1] if samples else peak_memory_mb()
    return {
        "duration_seconds": round(duration, 1),
        "iterations": iterations,
        "memory_first_mb": round(first, 1),
        "memory_last_mb": round(last, 1),
        "growth_mb_per_minute": round((last - first) / max(1e-9, duration / 60), 4),
        "requested_duration_seconds": seconds,
        "note": "сокращённый цикл вместо 8 часов — фактическая длительность указана",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Функциональные испытания лабораторного образца")
    parser.add_argument("--mode", choices=["demo", "hf"], default="demo")
    parser.add_argument("--model", default=None)
    parser.add_argument("--lengths", default="500,2000,10000,50000")
    parser.add_argument("--max-lengths", default="100000,200000")
    parser.add_argument("--cycle-seconds", type=int, default=60)
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--out", default="reports/FUNCTIONAL_TESTS.md")
    parser.add_argument("--json-out", default="reports/functional_tests.json")
    parser.add_argument("--skip-server", action="store_true")
    args = parser.parse_args(argv)

    lengths = [int(item) for item in args.lengths.split(",") if item.strip()]
    max_lengths = [int(item) for item in args.max_lengths.split(",") if item.strip()]
    report: dict = {
        "version": __version__,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "cpu": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(),
            "ram_total_mb": (
                round(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024), 1)
                if hasattr(os, "sysconf")
                else None
            ),
        },
        "sizes": file_sizes(),
        "processing": measure_processing(lengths, args.mode, args.model),
        "max_length": max_length_probe(args.mode, args.model, max_lengths),
        "exit_codes": exit_code_probe(),
        "smoke": None if args.skip_server else smoke_test(args.port, args.mode),
        "leak": leak_probe(args.cycle_seconds, args.mode),
        "disk": {"reports_mb": directory_size_mb(ROOT / "reports"), "data_mb": directory_size_mb(ROOT / "data")},
        "peak_memory_mb": round(peak_memory_mb(), 1),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    out.write_text(render(report), encoding="utf-8")
    print(render(report))
    return 0


def render(report: dict) -> str:
    """Отчёт из JSON: числа не дублируются руками."""
    env = report["environment"]
    lines = [
        "# Функциональные испытания лабораторного образца",
        "",
        f"Версия: {report['version']}, отчёт создан {report['generated_at']}.",
        f"Среда: {env['os']}, Python {env['python']}, CPU: {env['cpu']} ({env['cpu_count']} ядер), "
        f"RAM {env['ram_total_mb']} МБ. Режим измерений: **{report['processing'][0]['mode'] if report['processing'] else '—'}**.",
        "",
        "## Размеры поставки",
        "",
        "| Артефакт | Байт | Причина отсутствия |",
        "|---|---|---|",
    ]
    for name, item in report["sizes"].items():
        lines.append(
            f"| {item['path']} | {item['bytes'] if item['bytes'] is not None else 'null'} | {item['reason'] or '—'} |"
        )
    lines += [
        "",
        "## Время обработки пары текстов",
        "",
        "| Символов в документе | Секунд | Вердикт | Фрагментов | Пик памяти, МБ |",
        "|---|---|---|---|---|",
    ]
    for row in report["processing"]:
        lines.append(
            f"| {row['chars']} | {row['seconds']} | {row['verdict']} | {row['spans']} | {row['peak_memory_mb']} |"
        )
    lines += [
        "",
        "## Максимальная длина текста",
        "",
        f"Максимальная успешно обработанная длина: {report['max_length']['max_ok_chars']} символов.",
        "",
        "| Символов | Статус | Секунд | Примечание |",
        "|---|---|---|---|",
    ]
    for row in report["max_length"]["attempts"]:
        lines.append(f"| {row['chars']} | {row['status']} | {row['seconds']} | {row['note'] or '—'} |")
    lines += ["", "## Коды возврата verify", "", "| Случай | Команда | Код | Traceback |", "|---|---|---|---|"]
    for row in report["exit_codes"]:
        lines.append(
            f"| {row['case']} | `{row['command']}` | {row['exit_code']} | {'да' if row['has_traceback'] else 'нет'} |"
        )
    smoke = report.get("smoke")
    lines += ["", "## Смоук-тест сервиса", ""]
    if smoke is None:
        lines.append("Смоук-тест пропущен (флаг --skip-server).")
    elif smoke.get("error"):
        lines.append(f"Смоук-тест не прошёл: {smoke['error']}.")
    else:
        lines += [
            f"Холодный запуск до `GET /health`: {smoke['cold_start_seconds']} с.",
            f"Поля контракта в ответе `POST /v1/verify`: {smoke['fields_present']}.",
            f"Подтверждённый ответ получает вердикт `grounded`: {'да' if smoke['grounded_on_confirmed_answer'] else 'нет'}.",
        ]
    leak = report["leak"]
    lines += [
        "",
        "## Непрерывный прогон (утечка памяти)",
        "",
        f"Длительность: {leak['duration_seconds']} с (запрошено {leak['requested_duration_seconds']} с), "
        f"итераций {leak['iterations']}; память {leak['memory_first_mb']} → {leak['memory_last_mb']} МБ, "
        f"рост {leak['growth_mb_per_minute']} МБ/мин.",
        f"Пиковая память процесса: {report['peak_memory_mb']} МБ. Диск: reports {report['disk']['reports_mb']} МБ, "
        f"data {report['disk']['data_mb']} МБ.",
        "",
        leak["note"] + ".",
    ]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
