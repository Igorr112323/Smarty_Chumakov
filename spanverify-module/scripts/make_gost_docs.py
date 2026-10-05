"""Отчётные документы по ГОСТ 7.32: протокол, акт, ведомость (пункт 2.5 промта).

Документы **не являются** официальными бланками: это машиночитаемые заготовки,
которые пользователь при необходимости переносит в свой шаблон. Ничего не
подписывается и не подаётся от имени пользователя — в каждом документе это
написано прямым текстом.

Числа берутся только из файлов репозитория (reports/METRICS.json,
reports/functional_tests.json, data/*/manifest.json). Если файла нет, в документе
стоит ``нет данных`` — выдуманных значений не бывает.

Запуск::

    python scripts/make_gost_docs.py --out reports/gost
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spanverify import __version__  # noqa: E402
from spanverify.config import read_runtime_text, runtime_roots  # noqa: E402

DISCLAIMER = (
    "Документ подготовлен автоматически как рабочий материал. "
    "Он не является официальной формой, не содержит подписей, печатей и номеров "
    "сертификатов и не подаётся от имени пользователя."
)


def load_json(path: Path) -> dict[str, Any] | None:
    """Прочитать JSON или вернуть ``None`` (молча ничего не подставляем)."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def value(data: dict[str, Any] | None, *path: str, default: str = "нет данных") -> Any:
    """Значение по пути в словаре (``нет данных`` вместо выдумки)."""
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def metrics_rows(metrics: dict[str, Any] | None) -> list[tuple[str, str, str]]:
    """Строки таблицы метрик: (показатель, значение, источник)."""
    if not metrics:
        return [("Метрики", "нет данных", "reports/METRICS.json отсутствует")]
    rows: list[tuple[str, str, str]] = []
    for corpus, payload in sorted(metrics.get("corpora", {}).items()):
        for mode, values in sorted((payload or {}).get("modes", {}).items()):
            f1 = value(values, "verdicts", "f1", default="—")
            fpr = value(values, "verdicts", "fpr", default="—")
            rows.append((f"{corpus} / {mode}", f"вердикт F1 {f1}, FPR {fpr}", "reports/METRICS.json"))
    return rows or [("Метрики", "нет данных", "reports/METRICS.json пуст")]


def protocol_text(version: str, metrics: dict[str, Any] | None, functional: dict[str, Any] | None) -> str:
    """Протокол испытаний: что, на чём и с каким результатом проверялось."""
    rows = metrics_rows(metrics)
    lines = [
        "# Протокол испытаний лабораторного образца SpanVerify",
        "",
        f"Версия ПО: {version}. Дата формирования: {date.today().isoformat()}.",
        "",
        DISCLAIMER,
        "",
        "## 1. Объект испытаний",
        "",
        "Программный модуль проверки достоверности ответа относительно документа-контекста",
        "и оценки доли участия ИИ (SpanVerify). Способ поставки: исходный код, zipapp (.pyz),",
        "сборка Windows (.exe).",
        "",
        "## 2. Условия испытаний",
        "",
        f"ОС и оборудование: {value(functional, 'environment', 'os')}, "
        f"Python {value(functional, 'environment', 'python')}, "
        f"CPU {value(functional, 'environment', 'cpu')} "
        f"({value(functional, 'environment', 'cpu_count')} ядер), "
        f"RAM {value(functional, 'environment', 'ram_total_mb')} МБ.",
        "",
        "## 3. Результаты измерений",
        "",
        "| Показатель | Значение | Источник |",
        "|---|---|---|",
    ]
    lines += [f"| {name} | {result} | {source} |" for name, result, source in rows]
    lines += [
        "",
        "## 4. Функциональные проверки",
        "",
        "| Проверка | Результат | Источник |",
        "|---|---|---|",
        f"| Холодный запуск до GET /health | {value(functional, 'smoke', 'cold_start_seconds')} с | reports/functional_tests.json |",
        f"| Поля контракта POST /v1/verify | {value(functional, 'smoke', 'fields_present')} | reports/functional_tests.json |",
        f"| Максимальная длина текста | {value(functional, 'max_length', 'max_ok_chars')} символов | reports/functional_tests.json |",
        f"| Рост памяти на цикле | {value(functional, 'leak', 'growth_mb_per_minute')} МБ/мин | reports/functional_tests.json |",
        "",
        "## 5. Вывод",
        "",
        "Измеренные значения соответствуют перечисленным выше файлам-источникам;",
        "не измерявшиеся показатели в таблицах отсутствуют (в отчётах помечены `null`",
        "с указанием причины). Протокол не подписывается автоматически.",
    ]
    return "\n".join(lines) + "\n"


def act_text(version: str) -> str:
    """Акт готовности: перечень того, что фактически реализовано в поставке."""
    return (
        "\n".join(
            [
                "# Акт готовности лабораторного образца SpanVerify",
                "",
                f"Версия ПО: {version}. Дата формирования: {date.today().isoformat()}.",
                "",
                DISCLAIMER,
                "",
                "## Состав поставки",
                "",
                "| Компонент | Состояние | Подтверждение |",
                "|---|---|---|",
                "| Ядро проверки (признаки → риск → фрагменты → вердикт) | реализовано | пакет `spanverify/`, тесты `tests/` |",
                "| Режим `demo` (лексические суррогаты) | реализовано | CLI `--mode demo` |",
                "| Режим `hf` (реальная модель transformers) | реализовано | `spanverify/hfmodel.py`, пилот в CI |",
                "| Оценка покрытия фактов документа (missing/partial) | реализовано | `spanverify/coverage.py` |",
                "| Точные границы фрагментов (narrow/expanded) | реализовано | `spanverify/bounds.py` |",
                "| Оценка доли участия ИИ | реализовано | `spanverify/participation.py`, `config/participation.json` |",
                "| HTTP API (`/health`, `/v1/verify`) | реализовано | `spanverify/server.py`, смоук-тест в CI |",
                "| Сборка .pyz / .exe | выполняется в CI | `.github/workflows/build-and-release-exe.yml` |",
                "",
                "## Ограничения",
                "",
                "* Числа на демонстрационном режиме не переносятся на режим `hf` (разные признаки);",
                "  в каждом отчёте указан режим.",
                "* Внешние наборы данных используются только как тест, не для обучения.",
                "* Веса моделей в репозиторий не коммитятся (скачиваются на месте прогона).",
            ]
        )
        + "\n"
    )


def vedomost_text(version: str) -> str:
    """Ведомость документов и артефактов поставки."""
    files: list[str] = []
    for root in runtime_roots():
        for relative in ("README.md", "spanverify-module/README.md", "reports/METRICS.json", "reports/TODO_AUDIT.md"):
            path = root / relative
            if path.is_file():
                files.append(f"| {relative} | {path.stat().st_size} |")
    lines = [
        "# Ведомость документов SpanVerify",
        "",
        f"Версия ПО: {version}. Дата: {date.today().isoformat()}.",
        "",
        DISCLAIMER,
        "",
        "| Документ | Размер, байт |",
        "|---|---|",
        *files,
        "",
        "## Наличие обязательных отчётов",
        "",
        "| Отчёт | Файл | Есть |",
        "|---|---|---|",
    ]
    for name, relative in (
        ("Аудит незавершённых мест", "spanverify-module/reports/TODO_AUDIT.md"),
        ("Отчёт по корпусу A3", "spanverify-module/reports/CORPUS_A3.md"),
        ("Внешние наборы", "spanverify-module/reports/EXTERNAL_TESTS.md"),
        ("Устойчивость к искажениям", "spanverify-module/reports/robustness.md"),
        ("Функциональные испытания", "spanverify-module/reports/FUNCTIONAL_TESTS.md"),
        ("Итоги прогона", "spanverify-module/reports/РЕЗУЛЬТАТЫ_ПРОГОНА.md"),
    ):
        exists = any((root / relative).is_file() for root in runtime_roots())
        lines.append(f"| {name} | {relative} | {'да' if exists else 'нет'} |")
    return "\n".join(lines) + "\n"


def write_docx(markdown: str, path: Path) -> str:
    """Сохранить .docx, если установлен python-docx (иначе честно сообщить)."""
    try:
        import docx  # noqa: PLC0415
    except ImportError:
        return "python-docx не установлен — .docx не создан"
    document = docx.Document()
    for line in markdown.splitlines():
        if line.startswith("# "):
            document.add_heading(line[2:], level=1)
        elif line.startswith("## "):
            document.add_heading(line[3:], level=2)
        elif line.startswith("|"):
            document.add_paragraph(line)
        else:
            document.add_paragraph(line)
    document.save(str(path))
    return f"создан {path.name}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ГОСТ-документы по итогам прогонов")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "gost")
    parser.add_argument("--metrics", type=Path, default=ROOT / "reports" / "METRICS.json")
    parser.add_argument("--functional", type=Path, default=ROOT / "reports" / "functional_tests.json")
    args = parser.parse_args(argv)

    _ = read_runtime_text  # интерфейс поставки доступен, но здесь не требуется
    metrics = load_json(args.metrics)
    functional = load_json(args.functional)
    args.out.mkdir(parents=True, exist_ok=True)
    documents = {
        "PROTOKOL_ispytaniy.md": protocol_text(__version__, metrics, functional),
        "AKT_gotovnosti.md": act_text(__version__),
        "VEDOMOST_dokumentov.md": vedomost_text(__version__),
    }
    notes: list[str] = []
    for name, text in documents.items():
        (args.out / name).write_text(text, encoding="utf-8")
        notes.append(write_docx(text, args.out / name.replace(".md", ".docx")))
        print(f"записан {args.out / name}")
    (args.out / "README.txt").write_text(
        DISCLAIMER + "\n\n" + "\n".join(notes) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
