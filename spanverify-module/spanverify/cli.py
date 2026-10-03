"""Командный интерфейс SpanVerify.

    python -m spanverify analyze --text "..." --json
    python -m spanverify analyze --file document.txt
    python -m spanverify demo --n 240 --out data/demo_dataset.jsonl
    python -m spanverify calibrate --dataset data/demo_dataset.jsonl
    python -m spanverify serve --port 8000
    python -m spanverify selftest
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import __version__
from .calibration import IsotonicCalibrator, cross_validate
from .training import train_calibrator
from .config import Config
from .demo_data import (
    dataset_statistics,
    generate_dataset,
    make_ai_paragraph,
    make_human_paragraph,
    make_mixed_document,
    read_dataset,
    write_dataset,
)
from .detector import Detector, _moving_average
from .text import tokenize


def _load_config(args: argparse.Namespace) -> Config:
    config = Config.load(getattr(args, "config", None))
    return config.with_overrides(
        backend=getattr(args, "backend", None),
        threshold=getattr(args, "threshold", None),
        host=getattr(args, "host", None),
        port=getattr(args, "port", None),
        calibration_path=getattr(args, "calibration", None),
    )


def _print_result(result, as_json: bool, quiet: bool = False) -> None:
    if as_json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return
    print(f"Вердикт: {result.verdict} | доля ИИ: {result.ai_fraction:.1%} символов, "
          f"{result.ai_fraction_tokens:.1%} слов")
    print(f"Порог: {result.threshold:.3f} | калибровка: {'да' if result.calibrated else 'нет'} | "
          f"бэкенд: {result.backend}")
    for warning in result.warnings:
        print(f"[!] {warning}", file=sys.stderr)
    if not result.spans and not quiet:
        print("Фрагментов выше порога не найдено.")
    for span in result.spans:
        snippet = span.text if len(span.text) <= 90 else span.text[:90] + "…"
        print(f"  #{span.index + 1} [{span.start_char}:{span.end_char}] "
              f"токенов={span.n_tokens} p={span.mean_prob:.3f} — {snippet}")


# ---------- команды ----------

def cmd_analyze(args: argparse.Namespace) -> int:
    config = _load_config(args)
    if args.text:
        text = args.text
    elif args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        print("Укажите --text, --file или передайте текст через stdin.", file=sys.stderr)
        return 2

    detector = Detector(config)
    started = time.perf_counter()
    if args.explain:
        payload = detector.explain(text, threshold=config.threshold)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            _print_result(detector.analyze(text, threshold=config.threshold), False)
            print("\nТокен  Сырая  Калибр.  Метка")
            for row in payload["tokens"][: args.limit]:
                print(f"{row['token'][:18]:18s} {row['raw']:.3f}  {row['prob']:.3f}  "
                      f"{'ИИ' if row['flag'] else '—'}")
    else:
        result = detector.analyze(text, threshold=config.threshold)
        _print_result(result, args.json)
    if args.time:
        print(f"Время: {time.perf_counter() - started:.3f} с", file=sys.stderr)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    config = _load_config(args)
    detector = Detector(config)
    rng_seed = args.seed
    import random

    rng = random.Random(rng_seed)
    print("=== Машинный фрагмент ===")
    _print_result(detector.analyze(make_ai_paragraph(rng)), False)
    print("\n=== Человеческий фрагмент ===")
    _print_result(detector.analyze(make_human_paragraph(rng)), False)
    print("\n=== Смешанный документ ===")
    document = make_mixed_document(rng)
    _print_result(detector.analyze(document["text"]), False)
    print("\nРазметка (истина):", json.dumps(document["labels"], ensure_ascii=False))
    if args.out:
        docs = generate_dataset(args.n, seed=rng_seed)
        path = write_dataset(docs, args.out)
        print(f"\nКорпус записан: {path}")
        for key, value in dataset_statistics(docs).items():
            print(f"  {key}: {value}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    config = _load_config(args)
    detector = Detector(config)
    dataset_path = Path(args.dataset)
    if not dataset_path.is_file():
        print(f"Датасет не найден: {dataset_path}. Сначала: python -m spanverify demo "
              f"--out {dataset_path}", file=sys.stderr)
        return 2

    documents = list(read_dataset(dataset_path))
    report = train_calibrator(detector, documents, config=config, dataset_name=str(dataset_path))

    mean = report.cross_validation["mean"]
    print(f"Документов: {report.stats['documents']} | токенов: {report.stats['tokens']} | "
          f"доля ИИ-токенов: {report.calibrator.meta['ai_token_share']:.1%}")
    print(f"Кросс-валидация ({config.folds} фолдов, отложенная часть):")
    print(f"  precision={mean['precision']:.3f} recall={mean['recall']:.3f} f1={mean['f1']:.3f}")
    print(f"  FPR={mean['fpr']:.3f} (лимит {config.max_fpr}) HDR={mean['hdr']:.3f} "
          f"accuracy={mean['accuracy']:.3f}")

    out = Path(args.out)
    report.calibrator.save(out)
    print(f"Порог: {report.threshold:.4f} | калибратор сохранён: {out}")
    print(f"Проверка: python -m spanverify analyze --text \"...\" --threshold {report.threshold:.4f}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .api import serve

    config = _load_config(args)
    if args.port:
        config = config.with_overrides(port=args.port)
    if args.host:
        config = config.with_overrides(host=args.host)
    if args.open_browser:
        import threading
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{config.port}")).start()
    serve(config, quiet=args.quiet)
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """Быстрая проверка работоспособности сборки (используется в CI и в exe)."""
    config = _load_config(args)
    checks: list[tuple[str, bool, str]] = []

    detector = Detector(config)
    sample = "Важно отметить, что данный метод обеспечивает эффективное решение поставленной задачи."
    result = detector.analyze(sample)
    checks.append(("детектор отвечает", result.n_word_tokens > 0, f"{result.n_word_tokens} токенов"))
    checks.append(("доля ИИ в диапазоне [0,1]", 0.0 <= result.ai_fraction <= 1.0,
                   f"{result.ai_fraction:.3f}"))
    checks.append(("калибратор", detector.calibrator is not None,
                   "загружен" if detector.calibrator else "отсутствует (не критично)"))

    tokens = tokenize("Проверка токенизации: 42 слова, дефис-пример.")
    checks.append(("токенизация непрерывна",
                   all(t.end <= len("Проверка токенизации: 42 слова, дефис-пример.") for t in tokens),
                   f"{len(tokens)} токенов"))

    from .api import create_server

    try:
        server = create_server(config.with_overrides(host="127.0.0.1", port=0), quiet=True)
        port = server.server_address[1]
        import threading
        import urllib.request

        threading.Thread(target=server.serve_forever, daemon=True).start()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        checks.append(("HTTP /health", payload.get("status") == "ok", f"порт {port}"))
        server.shutdown()
        server.server_close()
    except Exception as exc:  # noqa: BLE001
        checks.append(("HTTP /health", False, f"{type(exc).__name__}: {exc}"))

    failed = 0
    for name, ok, detail in checks:
        print(f"[{'OK ' if ok else 'FAIL'}] {name} — {detail}")
        failed += 0 if ok else 1
    version_line = f"SpanVerify {__version__} | Python {sys.version.split()[0]} | {config.describe()}"
    print(version_line)
    return 1 if failed else 0


def cmd_config(args: argparse.Namespace) -> int:
    config = _load_config(args)
    print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))
    return 0


# ---------- сборка парсера ----------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spanverify",
        description="SpanVerify — локализация фрагментов, написанных ИИ, и оценка доли участия модели.",
    )
    parser.add_argument("--version", action="version", version=f"spanverify {__version__}")

    # Общие параметры доступны и до, и после имени команды
    # (spanverify --backend hf analyze ... и spanverify analyze --backend hf ...).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="путь к JSON-конфигу", default=argparse.SUPPRESS)
    common.add_argument("--backend", choices=["surrogate", "hf"], help="бэкенд признаков",
                        default=argparse.SUPPRESS)
    common.add_argument("--threshold", type=float, help="порог срабатывания (0..1)",
                        default=argparse.SUPPRESS)
    common.add_argument("--calibration", help="путь к файлу калибратора",
                        default=argparse.SUPPRESS)
    for action in common._actions:
        parser._add_action(action)

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("analyze", help="проверить текст", parents=[common])
    p.add_argument("--text")
    p.add_argument("--file")
    p.add_argument("--json", action="store_true", help="вывод в JSON")
    p.add_argument("--explain", action="store_true", help="покадровая таблица")
    p.add_argument("--limit", type=int, default=30, help="строк в таблице")
    p.add_argument("--time", action="store_true", help="замер времени")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("demo", parents=[common], help="демонстрация + генерация корпуса")
    p.add_argument("--n", type=int, default=240)
    p.add_argument("--seed", type=int, default=1312)
    p.add_argument("--out", help="куда записать корпус JSONL")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("calibrate", parents=[common], help="обучить калибратор и подобрать порог")
    p.add_argument("--dataset", default="data/demo_dataset.jsonl")
    p.add_argument("--out", default="config/calibration.json")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("serve", help="запустить API и веб-интерфейс", parents=[common])
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--open-browser", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("selftest", parents=[common], help="проверка работоспособности сборки")
    p.set_defaults(func=cmd_selftest)

    p = sub.add_parser("config", parents=[common], help="показать действующий конфиг")
    p.set_defaults(func=cmd_config)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"Ошибка: {type(exc).__name__}: {exc}", file=sys.stderr)
        if getattr(args, "backend", None) == "hf":
            print("Подсказка: установите зависимости режима 'hf': "
                  "pip install -r requirements-hf.txt", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
