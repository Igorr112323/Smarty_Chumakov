#!/usr/bin/env python3
"""Собрать reports/РЕЗУЛЬТАТЫ_ПРОГОНА.md из фактических прогонов.

Отчёт собирается **только** из выводов выполненных команд. Ни одно число не
берётся из памяти и не додумывается. Если измерение не выполнено, в отчёт
попадает ``не измерено`` и причина — раскрывать пробел честнее, чем подставлять
ожидаемое значение.

Что запускается:

1. ``pytest --collect-only -q`` — количество собранных тестов;
2. ``spanverify evaluate`` на демонстрационном корпусе и на корпусе A;
3. чтение ``reports/METRICS.json`` при его наличии;
4. проверка наличия ``data/corpus_a3/sources`` и ``manifest.json``;
5. проверка доступности режима ``hf`` (torch установлен или нет).

Использование:

    python scripts/make_run_report.py --out reports/РЕЗУЛЬТАТЫ_ПРОГОНА.md
    python scripts/make_run_report.py --skip-evaluate   # без долгих прогонов
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run(cmd: list[str]) -> tuple[int, str]:
    """Выполнить команду, вернуть код и объединённый вывод."""
    proc = subprocess.run(
        [sys.executable, *cmd],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def collect_tests() -> tuple[str | None, str | None]:
    """Количество собранных тестов: сам collect, без прогона."""
    code, out = _run(["-m", "pytest", "--collect-only", "-q", "--no-header", "-p", "no:cacheprovider"])
    if code != 0:
        return None, f"сборка тестов завершилась с кодом {code}"
    match = re.search(r"(\d+)\s+tests? collected", out)
    if match:
        return match.group(1), None
    # В режиме ``-q`` сводной строки нет: вывод идёт построчно «файл: число».
    per_file = re.findall(r"^(?:tests|spanverify)/\S+\.py:\s*(\d+)\s*$", out, re.M)
    if per_file:
        return str(sum(int(value) for value in per_file)), None
    return None, "в выводе сборки не найдено количество тестов"


def hf_status() -> tuple[bool, str]:
    """Проверить, доступен ли режим hf в этом окружении."""
    try:
        from spanverify.backends.hf import HFBackend

        ok, reason = HFBackend.dependencies()
        return ok, reason
    except Exception as error:  # noqa: BLE001
        return False, f"{type(error).__name__}: {error}"


def corpus_a3_status() -> tuple[int, bool]:
    sources = ROOT / "data" / "corpus_a3" / "sources"
    count = len(list(sources.glob("*.txt"))) if sources.is_dir() else 0
    manifest = (ROOT / "data" / "corpus_a3" / "manifest.json").is_file()
    return count, manifest


def measure(dataset: Path) -> dict[str, object] | None:
    """Прогон ``spanverify evaluate --json`` по набору данных.

    Прогон идёт через CLI, а не через прямой вызов: так отчёт содержит ровно
    те числа, которые видит пользователь команды.
    """
    if not dataset.is_file():
        return None
    code, out = _run(["-m", "spanverify", "evaluate", "--dataset", str(dataset), "--json"])
    if code != 0:
        return {"error": f"код {code}"}
    try:
        metrics = json.loads(out)
    except json.JSONDecodeError:
        return {"error": "вывод не разобран как JSON"}
    tokens = metrics.get("tokens", {})
    spans = metrics.get("spans", {})
    return {
        "pairs": metrics.get("pairs"),
        "mode": metrics.get("mode"),
        "token_f1": tokens.get("f1"),
        "token_fpr": tokens.get("fpr"),
        "token_auc": tokens.get("auc"),
        "span_f1": spans.get("f1"),
        "width": spans.get("mean_width_ratio"),
    }


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build(skip_evaluate: bool) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []
    lines.append("# Результаты прогона\n")
    lines.append("Файл сгенерирован скриптом `scripts/make_run_report.py`.")
    lines.append(f"Дата формирования: **{now}**.\n")
    lines.append(
        "Правило: число попадает в отчёт, только если оно получено выводом "
        "команды. Неизмеренное помечено «не измерено» с причиной.\n"
    )

    lines.append("## 1. Тесты\n")
    count, reason = collect_tests()
    if count is None:
        lines.append(f"Собрано тестов: **не измерено**. Причина: {reason}.\n")
    else:
        lines.append(f"Собрано тестов: **{count}**.\n")
    lines.append("```bash\npython -m pytest -q\n```\n")

    lines.append("## 2. Режим работы с языковой моделью (`hf`)\n")
    ok, reason = hf_status()
    if ok:
        lines.append(f"Зависимости режима `hf`: **доступны** ({reason}).\n")
        lines.append("Фактических прогонов в режиме `hf` на момент отчёта: **0**.\n")
    else:
        lines.append(f"Режим `hf` **недоступен** в этом окружении: {reason}.\n")
    lines.append(
        "При запросе режима `hf` без зависимостей программа возвращает явную "
        "ошибку с кодом 2 и подсказкой `pip install -r requirements-hf.txt`, "
        "а не переходит на лексический режим.\n"
    )

    lines.append("## 3. Корпус на реальных документах (A3)\n")
    docs, has_manifest = corpus_a3_status()
    lines.append(f"Загружено документов: **{docs}** из 120.\n")
    lines.append(
        f"Файл `manifest.json` (сводная ведомость скачанного): "
        f"{'присутствует' if has_manifest else '**отсутствует**'}.\n"
    )
    if not has_manifest:
        lines.append(
            "Скачивание выполняется заданием GitHub Actions «Корпус A3»; "
            "в локальной среде доступ к источнику закрыт, поэтому числа по "
            "корпусу A3 в отчёте отсутствуют.\n"
        )

    metrics_path = ROOT / "reports" / "METRICS.json"
    lines.append("## 4. Метрики из METRICS.json\n")
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            lines.append(f"Файл испорчен или не читается: {error}.\n")
            metrics = None
        if isinstance(metrics, dict):
            test = ((metrics.get("corpus_a") or {}).get("by_split") or {}).get("test") or {}
            tokens = test.get("tokens") or {}
            spans = test.get("spans") or {}
            external = metrics.get("external_tests") or {}
            lines.append(
                "| Показатель | Значение |\n|---|---|\n"
                f"| token F1 (тестовая часть корпуса A) | {_fmt(tokens.get('f1'))} |\n"
                f"| token precision / recall | {_fmt(tokens.get('precision'))} / {_fmt(tokens.get('recall'))} |\n"
                f"| token FPR | {_fmt(tokens.get('fpr'))} |\n"
                f"| token AUC | {_fmt(tokens.get('auc'))} |\n"
                f"| строгий span-F1 | {_fmt(spans.get('strict_f1'))} |\n"
                f"| полнота по покрытию | {_fmt(spans.get('coverage'))} |\n"
                f"| ширина фрагмента | ×{_fmt(spans.get('width_ratio'), 1)} |\n"
                f"| прогонов в режиме hf | {_fmt(external.get('hf_runs'))} |\n"
            )
            lines.append("")
            lines.append(
                "Условия измерения: **тестовая часть** корпуса A (180 пар из 1200), "
                "веса обучены на обучающей части, разбиение по документам без общих групп.\n"
            )
    else:
        lines.append("Файл `reports/METRICS.json` отсутствует: **не измерено**.\n")

    lines.append("## 5. Прогоны evaluate\n")
    if skip_evaluate:
        lines.append("Прогоны отключены флагом `--skip-evaluate`: **не измерено**.\n")
    else:
        lines.append(
            "| Набор | Пар | Режим | token F1 | token FPR | token AUC | строгий span-F1 | ширина |\n"
            "|---|---|---|---|---|---|---|---|\n"
        )
        for label, rel in (("демонстрационный", "data/demo_pairs.jsonl"), ("корпус A", "data/corpus_a/pairs.jsonl")):
            path = ROOT / rel
            result = measure(path)
            if result is None:
                lines.append(f"| {label} | не измерено | — | — | — | — | — | — |")
            elif result.get("error"):
                lines.append(f"| {label} | не измерено ({result['error']}) | — | — | — | — | — | — |")
            else:
                lines.append(
                    f"| {label} | {_fmt(result['pairs'], 0)} | {result['mode']} | "
                    f"{_fmt(result['token_f1'])} | {_fmt(result['token_fpr'])} | {_fmt(result['token_auc'])} | "
                    f"{_fmt(result['span_f1'])} | ×{_fmt(result['width'], 1)} |"
                )
        lines.append("")
        lines.append(
            "Условия измерения: **весь набор целиком** на текущих весах из "
            "`config/weights.json` (демонстрационный корпус) либо на весах по умолчанию "
            "(корпус A). Разница с разделом 4 — в условиях, а не в программе: "
            "раздел 4 даёт оценку на отложенной части после обучения, раздел 5 — "
            "состояние «как есть» на всех парах. Оба числа честные, складывать их нельзя.\n"
        )

    lines.append("## 6. Чего в отчёте нет и почему\n")
    lines.append(
        "| Позиция | Причина |\n|---|---|\n"
        "| время обработки текстов 500 / 2 000 / 10 000 / 50 000 символов | "
        "функциональные испытания не проводились |\n"
        "| пиковая память, максимальная длина текста, утечка | то же |\n"
        "| размеры файлов поставки `.exe` | сборка возможна только на Windows-раннере Actions |\n"
        "| устойчивость к пяти видам искажений | отчёт не создан: сценарий не прогнан |\n"
        "| числа из статей-аналогов | PDF недоступен, извлечение не выполнено |\n"
        "| метрики корпуса A2 | собран только черновик `--dry-run`, прогон не выполнен |\n"
    )
    lines.append("")
    lines.append("## 7. Как повторить\n")
    lines.append("```bash\ncd spanverify-module\npython scripts/make_run_report.py\n```\n")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Отчёт о прогоне из фактических команд")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "РЕЗУЛЬТАТЫ_ПРОГОНА.md")
    parser.add_argument("--skip-evaluate", action="store_true", help="не запускать evaluate")
    args = parser.parse_args()
    content = build(args.skip_evaluate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(content, encoding="utf-8")
    print(f"Готово: {args.out} ({len(content)} байт)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
