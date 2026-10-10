"""Командная строка SpanVerify.

    spanverify verify    --answer "... --context "..."   # проверить ответ
    spanverify train     --dataset data/demo_pairs.jsonl --out config/weights.json
    spanverify calibrate --dataset data/demo_pairs.jsonl
    spanverify evaluate  --dataset data/demo_pairs.jsonl
    spanverify selftest                                  # само-проверка конвейера
    spanverify server    --port 8765 [--no-browser]      # HTTP API + интерфейс
    spanverify demo      --pairs 240                     # корпус → обучение → метрики
    spanverify config                                    # действующие параметры
    spanverify analyze   "текст"                         # историческая ветка: только текст

Коды возврата:

* ``0`` — проверка пройдена (ответ опирается на контекст) либо команда успешна;
* ``1`` — проверка нашла сомнительные фрагменты (это результат, а не сбой);
* ``2`` — ошибка: неверные аргументы, отсутствующий файл, сбой пайплайна.
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .core import configure_stdio
from .dataset import (
    DATASET_VERSION,
    DatasetFormatError,
    corpus_statistics,
    generate_pairs,
    read_pairs,
    write_pairs,
)
from .engine import WEIGHTS_FILENAME, Verifier
from .server import DEFAULT_PORT, serve

EXIT_OK = 0
EXIT_ISSUES = 1
EXIT_ERROR = 2


def json_safe(value: Any) -> Any:
    """Заменить нечисловые float (nan/inf) на ``null``.

    По RFC 8259 ``NaN`` и ``Infinity`` не являются допустимыми значениями JSON:
    строгие парсеры на них падают. Метрики без положительных примеров дают
    ``AUC = nan`` — в выводе это должно быть ``null``, а не ``NaN``.
    """
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def _print_json(payload: Any) -> None:
    """Напечатать JSON без NaN (см. :func:`json_safe`)."""
    print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False))


DEFAULT_DATASET = "data/demo_pairs.jsonl"
BACKENDS = ("demo", "surrogate", "stub", "hf")


def _mode(args: argparse.Namespace) -> str:
    """Режим работы: новый флаг --mode или исторический --backend."""
    chosen = getattr(args, "mode", None) or getattr(args, "backend", None) or "demo"
    return "demo" if chosen == "stub" else str(chosen)


# ------------------------------------------------------------------ вывод


def _print_result(result: Any, as_json: bool, tokens: bool) -> int:
    """Напечатать результат проверки; вернуть код возврата."""
    payload = result.to_dict()
    if as_json:
        _print_json(payload)
    else:
        verdicts = {
            "grounded": "ОПОРА НА КОНТЕКСТ ЕСТЬ",
            "doubtful": "ЕСТЬ СОМНИТЕЛЬНЫЕ ФРАГМЕНТЫ",
            "likely_hallucination": "ВЫСОКИЙ РИСК НЕДОСТОВЕРНОСТИ",
            "empty": "ПУСТОЙ ОТВЕТ",
        }
        print(f"Вердикт: {verdicts.get(result.verdict, result.verdict)}")
        print(f"Оценка недостоверности: {result.score:.3f} (порог {result.threshold:.3f})")
        print(f"Доля спорного текста: мягко {result.ai_share:.3f}, жёстко {result.ai_share_hard:.3f}")
        print(
            "Оценка доли участия ИИ: "
            f"{result.ai_participation:.3f} "
            "(калибровка на синтетическом корпусе; см. /v1/model)"
        )
        print(f"Режим: {result.mode}; время: {result.latency_ms:.1f} мс")
        if result.warning:
            print(f"ВНИМАНИЕ: {result.warning}")
        if result.spans:
            print("Фрагменты:")
            for span in result.spans:
                print(f"  - {span.start}:{span.end} риск {span.risk:.3f} [{span.label}] {span.text!r}")
        else:
            print("Фрагменты: не найдены")
        if tokens and result.tokens:
            print("Токены:")
            for row in result.tokens:
                print(f"  {row['index']:3d} {row['text']:18s} риск {row['risk']:.3f} " f"метка {row['label']}")
    return EXIT_OK if result.verdict in {"grounded", "empty"} else EXIT_ISSUES


def _load_dataset(path: str | Path, pairs: int = 0, seed: int = 1312) -> list[dict]:
    """Прочитать корпус пар или сгенерировать его, если файла нет."""
    target = Path(path)
    if target.is_file():
        return list(read_pairs(target))
    if pairs > 0:
        generated = generate_pairs(pairs, seed=seed)
        write_pairs(generated, target)
        print(f"Корпус сгенерирован: {target} ({len(generated)} пар, seed={seed})")
        return [pair.to_dict() for pair in generated]
    raise FileNotFoundError(f"корпус {target} не найден: укажите --dataset или добавьте --make-dataset N")


# ------------------------------------------------------------------ команды


def _verifier_from_args(args: argparse.Namespace) -> Verifier:
    """Собрать верификатор по аргументам команды (общий путь для verify/evaluate).

    Флаги additive: без них поведение ровно прежнее — режим из ``--mode``/
    ``--backend``, веса из ``--weights``, покрытие включено, кеш не используется.
    """
    from .features import load_feature_cache

    cache = load_feature_cache(args.features_cache, counting=True) if getattr(args, "features_cache", None) else None
    return Verifier(
        mode=_mode(args),
        model_name=getattr(args, "model", None),
        weights_path=args.weights,
        features_cache=cache,
        coverage=bool(getattr(args, "coverage", True)),
    )


def cmd_verify(args: argparse.Namespace) -> int:
    """Проверить один ответ относительно контекста."""
    answer = args.answer
    if args.answer_file:
        answer = Path(args.answer_file).read_text(encoding="utf-8")
    if not answer:
        print("ошибка: нужен --answer или --answer-file", file=sys.stderr)
        return EXIT_ERROR

    context = args.context
    if args.context_file:
        context = Path(args.context_file).read_text(encoding="utf-8")

    verifier = _verifier_from_args(args)
    result = verifier.verify(answer, context, with_tokens=args.tokens)
    return _print_result(result, as_json=args.json, tokens=args.tokens)


def cmd_train(args: argparse.Namespace) -> int:
    """Обучить веса, порог маски, голову и калибровку."""
    from .train import save_training_artifacts, train

    records = _load_dataset(args.dataset, pairs=args.make_dataset, seed=args.seed)
    print(f"Корпус: {args.dataset} — {len(records)} пар")
    report = train(
        records,
        mode=_mode(args),
        seed=args.seed,
        target_fpr=args.target_fpr,
        test_size=args.test_size,
        folds=args.folds,
        dataset_name=str(args.dataset),
        version=__version__,
    )
    print(report.summary())
    validation = report.validation
    print(
        "Валидация (отложенная часть): "
        f"precision={validation['precision']:.3f} recall={validation['recall']:.3f} "
        f"F1={validation['f1']:.3f} FPR={validation['fpr']:.3f} AUC={validation['auc']:.3f}"
    )
    end_to_end = validation.get("end_to_end", {})
    if end_to_end:
        tokens = end_to_end.get("tokens", {})
        spans = end_to_end.get("spans", {})
        print(
            "Сквозная проверка через verify(): "
            f"token F1={tokens.get('f1', 0):.3f} FPR={tokens.get('fpr', 0):.3f} | "
            f"фрагменты: полнота по покрытию={spans.get('recall_containment', 0):.3f} "
            f"F1 при расширенной разметке={spans.get('f1_expanded_labels', 0):.3f}"
        )
    written = save_training_artifacts(report, args.out, root=Path(args.out).parent.parent or ".")
    print("Записано: " + ", ".join(f"{name}={path}" for name, path in written.items()))
    if args.json:
        _print_json({"validation": report.validation, "folds": report.folds})
    return EXIT_OK


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Пересчитать калибровку и порог решения по размеченному корпусу."""
    from .calibration import IsotonicCalibrator, choose_threshold

    records = _load_dataset(args.dataset, pairs=args.make_dataset, seed=args.seed)
    verifier = Verifier(mode=_mode(args), weights_path=args.weights)
    bundle = verifier.bundle

    raw_scores: list[float] = []
    labels: list[int] = []
    for record in records:
        result = verifier.verify(record.get("answer", ""), record.get("context", ""))
        raw_scores.append(float(result.stats.get("raw_score", 0.0)))
        labels.append(1 if record.get("labels") else 0)

    calibrator = IsotonicCalibrator.fit(raw_scores, labels)
    calibrated = calibrator.transform(raw_scores)
    threshold = choose_threshold(calibrated, labels, max_fpr=args.target_fpr)
    bundle.isotonic = calibrator
    bundle.threshold = float(threshold)
    bundle.target_fpr = args.target_fpr
    bundle.meta = {**(bundle.meta or {}), "calibration_pairs": len(records)}
    bundle.save(args.out)
    print(
        f"Калибровка пересчитана на {len(records)} парах: порог={threshold:.4f} "
        f"(целевой FPR {args.target_fpr:.2f}); записано в {args.out}"
    )
    return EXIT_OK


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Показать метрики конвейера на размеченном корпусе."""
    records = _load_dataset(args.dataset, pairs=args.make_dataset, seed=args.seed)
    verifier = _verifier_from_args(args)
    metrics = verifier.evaluate(records)
    if args.json:
        _print_json(metrics)
        return EXIT_OK
    tokens = metrics["tokens"]
    spans = metrics["spans"]
    answers = metrics["answers"]
    print(f"Пар: {metrics['pairs']} (режим {metrics['mode']})")
    print(
        f"Токены: precision={tokens['precision']:.3f} recall={tokens['recall']:.3f} "
        f"F1={tokens['f1']:.3f} FPR={tokens['fpr']:.3f} AUC={tokens['auc']:.3f}"
    )
    print(
        f"Фрагменты: строгий IoU F1={spans['f1']:.3f}, полнота по покрытию="
        f"{spans['recall_containment']:.3f}, F1 при расширенной разметке="
        f"{spans['f1_expanded_labels']:.3f}, ширина ×{spans['mean_width_ratio']:.1f}"
    )
    print(
        f"Ответы: precision={answers['precision']:.3f} recall={answers['recall']:.3f} "
        f"F1={answers['f1']:.3f} FPR={answers['fpr']:.3f} AUC={answers['auc']:.3f} "
        f"(порог {answers['threshold']:.3f})"
    )
    coverage = metrics.get("coverage") or {}
    if coverage:
        def _fmt(value: object) -> str:
            return "н/д" if value is None else f"{value:.3f}"

        print(
            f"Покрытие фактов (включено: {coverage.get('enabled')}): "
            f"coverage_recall_missing={_fmt(coverage.get('recall_missing'))} "
            f"coverage_precision_missing={_fmt(coverage.get('precision_missing'))} "
            f"coverage_recall_partial={_fmt(coverage.get('recall_partial'))} "
            f"coverage_precision_partial={_fmt(coverage.get('precision_partial'))} "
            f"coverage_fpr_clean={_fmt(coverage.get('fpr_clean'))} "
            f"(пар с контекстом {coverage.get('pairs_with_context')}, чистых {coverage.get('clean_pairs')})"
        )
    gate = tokens["f1"] >= 0.90 and tokens["fpr"] <= 0.10
    print(f"Критерий качества (token F1 ≥ 0.90 при FPR ≤ 0.10): {'выполнен' if gate else 'не выполнен'}")
    if metrics["warning"]:
        print(f"ВНИМАНИЕ: {metrics['warning']}")
    return EXIT_OK


SELFTEST_CASES = (
    (
        "достоверный ответ",
        "Срок хранения первичных документов составляет 10 лет.",
        "Регламент 343: срок хранения первичных документов составляет 10 лет.",
        {"grounded"},
    ),
    (
        "подмена числа",
        "Срок хранения первичных документов составляет 3 года.",
        "Регламент 343: срок хранения первичных документов составляет 10 лет.",
        {"doubtful", "likely_hallucination"},
    ),
    (
        "выдуманное утверждение",
        "Срок хранения составляет 10 лет. Дополнительно требуется согласование с внешним аудитором.",
        "Регламент 343: срок хранения составляет 10 лет.",
        {"doubtful", "likely_hallucination"},
    ),
)


def cmd_selftest(args: argparse.Namespace) -> int:
    """Проверить конвейер на встроенных примерах (без внешних данных)."""
    verifier = Verifier(mode=_mode(args), weights_path=args.weights)
    failures: list[str] = []
    print(f"Само-проверка SpanVerify {__version__} (режим {verifier.mode})")
    for name, answer, context, expected in SELFTEST_CASES:
        result = verifier.verify(answer, context)
        ok = result.verdict in expected
        if not ok:
            failures.append(name)
        print(
            f"  [{'OK ' if ok else 'ПРОВАЛ'}] {name}: вердикт={result.verdict} "
            f"оценка={result.score:.3f} порог={result.threshold:.3f} "
            f"фрагментов={len(result.spans)}"
        )
    stats = verifier.bundle
    print(
        f"  параметры: веса={ {k: round(v, 2) for k, v in stats.weights.items()} }, "
        f"span_z={stats.span_z}, floor={stats.span_floor}, cap={stats.span_cap}"
    )
    print(f"  калибровка: {'есть' if stats.isotonic else 'нет'}; голова: {(stats.head or {}).get('type', 'none')}")
    if verifier.warning:
        print(f"  ВНИМАНИЕ: {verifier.warning}")
    if failures:
        print("ПРОВАЛЕНО: " + ", ".join(failures))
        return EXIT_ERROR
    print("Все проверки пройдены.")
    return EXIT_OK


def cmd_server(args: argparse.Namespace) -> int:
    """Запустить HTTP-сервис (по умолчанию порт 8765)."""
    if not args.no_browser:
        url = f"http://127.0.0.1:{args.port}"
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001 - браузер может отсутствовать
            pass
    serve(
        host=args.host,
        port=args.port,
        mode=_mode(args),
        weights_path=args.weights,
        quiet=args.quiet,
    )
    return EXIT_OK


def cmd_demo(args: argparse.Namespace) -> int:
    """Полный демонстрационный прогон: корпус → обучение → метрики → примеры."""
    from .train import train

    generated = generate_pairs(args.pairs, seed=args.seed)
    path = write_pairs(generated, args.dataset)
    print(f"Корпус: {path}")
    for key, value in corpus_statistics(generated).items():
        print(f"  {key}: {value}")

    report = train(
        [pair.to_dict() for pair in generated],
        mode=_mode(args),
        seed=args.seed,
        folds=args.folds,
        dataset_name=str(path),
        version=__version__,
    )
    print(report.summary())
    verifier = Verifier(mode=_mode(args), weights=report.bundle)
    for name, answer, context, _expected in SELFTEST_CASES:
        result = verifier.verify(answer, context)
        print(f"  {name:24s} → {result.verdict:20s} оценка {result.score:.3f} фрагментов {len(result.spans)}")
    print(f"Версия корпуса: {DATASET_VERSION}. Демо-числа не являются научным результатом.")
    return EXIT_OK


def cmd_config(args: argparse.Namespace) -> int:
    """Напечатать действующие параметры конвейера."""
    verifier = Verifier(mode=_mode(args), weights_path=args.weights)
    payload = verifier.bundle.to_dict()
    payload["runtime"] = {
        "version": __version__,
        "mode": verifier.mode,
        "weights_source": verifier.bundle.source,
        "weights_path": str(args.weights),
        "model_name": verifier.model_name,
    }
    _print_json(payload)
    return EXIT_OK


def cmd_analyze(args: argparse.Namespace) -> int:
    """Историческая ветка: оценка «текст создан ИИ» без контекста (surrogate/hf)."""
    text = getattr(args, "text", None)
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
    if not text:
        print("ошибка: нужен текст или --text-file", file=sys.stderr)
        return EXIT_ERROR
    from .config import Config
    from .detector import Detector

    backend = getattr(args, "backend", None) or getattr(args, "mode", None) or "surrogate"
    detector = Detector(Config.load().with_overrides(backend="surrogate" if backend in {"demo", "stub"} else backend))
    result = detector.analyze(text, threshold=args.threshold)
    if args.json:
        print(
            json.dumps(
                {
                    "verdict": result.verdict,
                    "share": result.share,
                    "ai_fraction_soft": result.ai_fraction_soft,
                    "ai_fraction_tokens": result.ai_fraction_tokens,
                    "backend": result.backend,
                    "calibrated": result.calibrated,
                    "threshold": result.threshold,
                    "spans": [span.to_dict() for span in result.spans],
                    "warnings": result.warnings,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return EXIT_OK
    print(f"Вердикт: {result.verdict} (доля {result.share:.3f}, порог {result.threshold:.3f})")
    print(f"Бэкенд: {result.backend}; калибровка: {'есть' if result.calibrated else 'нет'}")
    for warning in result.warnings:
        print(f"ВНИМАНИЕ: {warning}")
    print(f"Фрагментов: {len(result.spans)}")
    return EXIT_ISSUES if result.verdict in {"likely_ai", "mixed"} else EXIT_OK


# ------------------------------------------------------------------ парсер


def build_parser() -> argparse.ArgumentParser:
    """Собрать парсер аргументов (общий для всех точек входа)."""
    parser = argparse.ArgumentParser(prog="spanverify", description="Проверка ответа относительно контекста")
    parser.add_argument("--version", action="version", version=f"SpanVerify {__version__}")
    # Исторический флаг: раньше бэкенд задавался только глобально. Он оставлен
    # и как глобальный, и у каждой подкоманды, чтобы старые скрипты, в которых
    # флаг стоит и до, и после команды, продолжали работать.
    parser.add_argument("--backend", choices=BACKENDS, default=None, help="историческое имя режима")
    subparsers = parser.add_subparsers(dest="command")

    def add_common(sub: argparse.ArgumentParser) -> None:
        # default=SUPPRESS: если флаг не указан у подкоманды, значение глобального
        # флага не затирается (иначе «--backend hf analyze …» терял бы режим).
        sub.add_argument("--mode", choices=["demo", "surrogate", "hf"], default=argparse.SUPPRESS)
        sub.add_argument("--backend", choices=BACKENDS, default=argparse.SUPPRESS, help="историческое имя режима")
        sub.add_argument("--weights", default=WEIGHTS_FILENAME)
        sub.add_argument(
            "--model",
            default=None,
            help="модель режима hf; по умолчанию берётся из обученного бандла "
            "(meta.model), а затем из config — иначе ключ кеша признаков не совпадёт",
        )
        sub.add_argument(
            "--features-cache",
            action="append",
            default=[],
            metavar="PATH",
            help="каталог или файл кеша предпосчитанных признаков режима hf "
            "(scripts/precompute_features.py); можно указать несколько раз",
        )
        coverage_group = sub.add_mutually_exclusive_group()
        coverage_group.add_argument(
            "--coverage",
            dest="coverage",
            action="store_true",
            default=argparse.SUPPRESS,
            help="включить проверку покрытия фактов (по умолчанию включена)",
        )
        coverage_group.add_argument(
            "--no-coverage",
            dest="coverage",
            action="store_false",
            default=argparse.SUPPRESS,
            help="выключить проверку покрытия фактов (историческое поведение до v1.4)",
        )

    verify = subparsers.add_parser("verify", help="проверить ответ относительно контекста")
    verify.add_argument("--answer", default=None, help="текст ответа")
    verify.add_argument("--answer-file", default=None, help="файл с ответом")
    verify.add_argument("--context", default=None, help="текст контекста (документ-источник)")
    verify.add_argument("--context-file", default=None, help="файл с контекстом")
    verify.add_argument("--json", action="store_true", help="вывести разбор в JSON")
    verify.add_argument("--tokens", action="store_true", help="показать разбор по токенам")
    add_common(verify)
    verify.set_defaults(func=cmd_verify)

    train_parser = subparsers.add_parser("train", help="обучить веса, порог, голову и калибровку")
    train_parser.add_argument("--dataset", default=DEFAULT_DATASET)
    train_parser.add_argument("--out", default=WEIGHTS_FILENAME)
    train_parser.add_argument("--make-dataset", type=int, default=240, help="сгенерировать корпус, если файла нет")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--folds", type=int, default=5)
    train_parser.add_argument("--test-size", type=float, default=0.3)
    train_parser.add_argument("--target-fpr", type=float, default=0.1)
    train_parser.add_argument("--json", action="store_true")
    add_common(train_parser)
    train_parser.set_defaults(func=cmd_train)

    calibrate = subparsers.add_parser("calibrate", help="пересчитать калибровку и порог решения")
    calibrate.add_argument("--dataset", default=DEFAULT_DATASET)
    calibrate.add_argument("--out", default=WEIGHTS_FILENAME)
    calibrate.add_argument("--make-dataset", type=int, default=0)
    calibrate.add_argument("--seed", type=int, default=42)
    calibrate.add_argument("--target-fpr", type=float, default=0.1)
    add_common(calibrate)
    calibrate.set_defaults(func=cmd_calibrate)

    evaluate = subparsers.add_parser("evaluate", help="метрики на размеченном корпусе")
    evaluate.add_argument("--dataset", default=DEFAULT_DATASET)
    evaluate.add_argument("--make-dataset", type=int, default=0)
    evaluate.add_argument("--seed", type=int, default=1312)
    evaluate.add_argument("--json", action="store_true")
    add_common(evaluate)
    evaluate.set_defaults(func=cmd_evaluate)

    selftest = subparsers.add_parser("selftest", help="само-проверка на встроенных примерах")
    add_common(selftest)
    selftest.set_defaults(func=cmd_selftest)

    server = subparsers.add_parser("server", help="запустить HTTP-сервис")
    server.add_argument("--host", default="0.0.0.0")
    server.add_argument("--port", type=int, default=DEFAULT_PORT)
    server.add_argument("--no-browser", action="store_true")
    server.add_argument("--quiet", action="store_true")
    add_common(server)
    server.set_defaults(func=cmd_server)
    serve_alias = subparsers.add_parser("serve", help="псевдоним команды server")
    serve_alias.add_argument("--host", default="0.0.0.0")
    serve_alias.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve_alias.add_argument("--no-browser", action="store_true")
    serve_alias.add_argument("--quiet", action="store_true")
    add_common(serve_alias)
    serve_alias.set_defaults(func=cmd_server)

    demo = subparsers.add_parser("demo", help="сквозной демонстрационный прогон")
    demo.add_argument("--pairs", type=int, default=240)
    demo.add_argument("--dataset", default=DEFAULT_DATASET)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--folds", type=int, default=5)
    add_common(demo)
    demo.set_defaults(func=cmd_demo)

    config = subparsers.add_parser("config", help="показать действующие параметры")
    add_common(config)
    config.set_defaults(func=cmd_config)

    analyze = subparsers.add_parser("analyze", help="историческая ветка: только текст, без контекста")
    analyze.add_argument("text", nargs="?", default=argparse.SUPPRESS)
    # dest="text": флаг и позиционный аргумент пишут в одно поле, поэтому
    # работают обе исторические формы записи («analyze "текст"» и
    # «analyze --text "текст"»). default=SUPPRESS, чтобы необъявленный флаг не
    # затирал значение позиционного аргумента.
    analyze.add_argument("--text", dest="text", default=argparse.SUPPRESS, help="текст (историческое имя)")
    analyze.add_argument("--text-file", default=None)
    analyze.add_argument("--threshold", type=float, default=None)
    analyze.add_argument("--mode", choices=["demo", "surrogate", "hf"], default=argparse.SUPPRESS)
    analyze.add_argument("--backend", choices=BACKENDS, default=argparse.SUPPRESS)
    analyze.add_argument("--json", action="store_true")
    analyze.set_defaults(func=cmd_analyze)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа CLI: возвращает код возврата 0/1/2."""
    configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return EXIT_OK
    try:
        return int(args.func(args))
    except FileNotFoundError as error:
        print(f"ошибка: {error}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("\nпрервано пользователем", file=sys.stderr)
        return EXIT_ERROR
    except DatasetFormatError as error:
        # Схема корпуса: сообщение уже содержит строку, причины и пример.
        print(f"ошибка: {error}", file=sys.stderr)
        return EXIT_ERROR
    except Exception as error:  # noqa: BLE001 - CLI обязан вернуть код, а не трассировку
        print(f"ошибка выполнения: {type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_ERROR


def run() -> None:  # pragma: no cover - обёртка для консольной точки входа
    """Обёртка для ``python -m spanverify``."""
    raise SystemExit(main())


if __name__ == "__main__":  # pragma: no cover
    run()
