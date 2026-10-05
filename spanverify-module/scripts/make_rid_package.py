"""РИД-комплект: заявки на программу для ЭВМ и базу данных (пункт 2.5 промта).

Готовятся **заготовки** документов для государственной регистрации:

* заявка на программу для ЭВМ (Роспатент, форма с рефератом и списком файлов);
* заявка на базу данных (структура, объём, источники данных);
* опись депонируемых материалов;
* лист сведений об авторе/правообладателе — **не заполняется автоматически**:
  персональные и правовые данные вносит пользователь.

Ничего не подаётся от имени пользователя: все документы содержат пометку
«черновик, не подано», а поля, требующие решения человека (автор, правообладатель,
дата приоритета), оставлены пустыми.

Запуск::

    python scripts/make_rid_package.py --out reports/rid
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

DRAFT = "ЧЕРНОВИК. НЕ ПОДАНО. Документ подготовлен автоматически для ручного заполнения."


def module_stats() -> dict[str, Any]:
    """Фактические размеры кода: модули, строки, тесты."""
    files = sorted((ROOT / "spanverify").glob("*.py")) + sorted((ROOT / "spanverify" / "backends").glob("*.py"))
    tests = sorted((ROOT / "tests").glob("test_*.py"))
    scripts = sorted((ROOT / "scripts").glob("*.py"))
    lines = sum(len(path.read_text(encoding="utf-8").splitlines()) for path in files)
    return {
        "modules": [path.relative_to(ROOT).as_posix() for path in files],
        "module_count": len(files),
        "lines": lines,
        "test_files": len(tests),
        "script_files": len(scripts),
    }


def load_metrics() -> dict[str, Any] | None:
    path = ROOT / "reports" / "METRICS.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def program_application(stats: dict[str, Any], metrics: dict[str, Any] | None) -> str:
    """Заявка на программу для ЭВМ (заготовка под форму Роспатента)."""
    return (
        "\n".join(
            [
                "# Заявка на государственную регистрацию программы для ЭВМ (заготовка)",
                "",
                DRAFT,
                "",
                "## 1. Название программы",
                "",
                "SpanVerify — модуль проверки достоверности ответа относительно документа-контекста",
                "и оценки доли участия искусственного интеллекта.",
                "",
                "## 2. Правообладатель и автор",
                "",
                "Заполняет пользователь (автоматически не подставляется):",
                "",
                "| Поле | Значение |",
                "|---|---|",
                "| Правообладатель | ______________________ |",
                "| Автор(ы) | ______________________ |",
                "| Дата приоритета | ______________________ |",
                "",
                "## 3. Реферат программы",
                "",
                f"Программа реализована на Python {stats['lines']} строками в {stats['module_count']} модулях.",
                "Назначение: проверка фрагментов ответа (в том числе сгенерированного языковой моделью)",
                "на подтверждаемость документом-источником с указанием точных границ сомнительных",
                "фрагментов и оценкой доли машинного текста.",
                "",
                "Функции: расчёт признаков (энтропия внимания и масса внимания на контекст реальной",
                "модели transformers, плотность представлений), свёртка признаков в риск, выделение",
                "посимвольных фрагментов, покрытие фактов документа (пропуск/частичность), калибровка",
                "порога, HTTP API, CLI, веб-интерфейс, zipapp-сборка.",
                "",
                "## 4. Тип ЭВМ и язык",
                "",
                "IBM PC-совместимые, ОС Windows/Linux, Python 3.10+. Режим `demo` работает без",
                "сторонних библиотек; режим `hf` требует torch и transformers.",
                "",
                "## 5. Объём программы",
                "",
                f"| Показатель | Значение | Источник |",
                f"|---|---|---|",
                f"| Модулей | {stats['module_count']} | `spanverify/` |",
                f"| Строк | {stats['lines']} | подсчёт по файлам |",
                f"| Файлов тестов | {stats['test_files']} | `tests/` |",
                f"| Вспомогательных скриптов | {stats['script_files']} | `scripts/` |",
                "",
                "## 6. Заимствованные компоненты",
                "",
                "Внешние библиотеки не включены в поставку ядра: используются только для тестов",
                "(pytest, ruff, black) и режима `hf` (torch, transformers). Веса сторонних моделей",
                "в поставку не входят и в репозитории не хранятся.",
                "",
                f"## 7. Сведения о метриках (из reports/METRICS.json: {'есть' if metrics else 'нет данных'})",
                "",
                "Конкретные значения приведены в отчёте о прогоне; здесь они не дублируются,",
                "чтобы заявка не расходилась с измерениями.",
            ]
        )
        + "\n"
    )


def database_application(stats: dict[str, Any]) -> str:
    """Заявка на базу данных: состав корпусов и структура записей."""
    return (
        "\n".join(
            [
                "# Заявка на государственную регистрацию базы данных (заготовка)",
                "",
                DRAFT,
                "",
                "## 1. Название базы данных",
                "",
                "Корпус размеченных пар «документ — ответ» для оценки достоверности (SpanVerify Corpus).",
                "",
                "## 2. Структура записей",
                "",
                "| Поле | Тип | Описание |",
                "|---|---|---|",
                "| `id` | строка | идентификатор пары |",
                "| `context` | текст | фрагмент документа-источника |",
                "| `answer` | текст | проверяемый ответ |",
                "| `labels` | список | посимвольные метки (start, end, type) |",
                "| `meta` | объект | документ, группа, режим искажения, происхождение ответа |",
                "",
                "## 3. Источники данных",
                "",
                "Корпус A3 собран из официально опубликованных нормативных правовых актов",
                "(официальные документы государственных органов не являются объектами авторских",
                "прав, п. 6 ст. 1259 ГК РФ); персональные данные и закрытые документы не включаются.",
                "",
                "## 4. Объём",
                "",
                f"| Компонент | Значение |",
                f"|---|---|",
                f"| Модулей обработки | {stats['module_count']} |",
                f"| Тестовых файлов | {stats['test_files']} |",
                "| Пар в корпусах | см. `data/*/manifest.json` (числа не дублируются) |",
                "",
                "## 5. Правообладатель",
                "",
                "Заполняет пользователь (автоматически не подставляется): ______________________",
            ]
        )
        + "\n"
    )


def deposit_inventory(stats: dict[str, Any]) -> str:
    """Опись депонируемых материалов."""
    rows = ["| Файл | Назначение |", "|---|---|"]
    for name in stats["modules"]:
        rows.append(f"| `{name}` | исходный модуль программы |")
    rows += [
        "| `tests/` | автоматические тесты |",
        "| `scripts/` | сборка корпусов, эксперименты, отчёты |",
        "| `config/` | параметры, калибровка, веса голов |",
        "| `data/` | корпуса (тексты актов и пары) |",
        "| `reports/` | отчётные материалы прогонов |",
    ]
    return (
        "\n".join(
            [
                "# Опись депонируемых материалов (заготовка)",
                "",
                DRAFT,
                "",
                f"Дата формирования: {date.today().isoformat()}. Версия ПО: {__version__}.",
                "",
                *rows,
                "",
                "Материалы передаются в электронном виде; состав и хеши фиксируются в",
                "`reports/CHECKS.sha256` (создаётся скриптом сборки поставки).",
            ]
        )
        + "\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="РИД-комплект (заявки и описи)")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "rid")
    args = parser.parse_args(argv)
    stats = module_stats()
    metrics = load_metrics()
    args.out.mkdir(parents=True, exist_ok=True)
    documents = {
        "ZAYAVKA_programma_EVM.md": program_application(stats, metrics),
        "ZAYAVKA_baza_dannyh.md": database_application(stats),
        "OPIS_deponiruemyh_materialov.md": deposit_inventory(stats),
    }
    for name, text in documents.items():
        (args.out / name).write_text(text, encoding="utf-8")
        print(f"записан {args.out / name}")
    index = {
        "status": "draft_not_submitted",
        "note": DRAFT,
        "documents": sorted(documents),
        "stats": stats,
        "human_fields_required": ["правообладатель", "автор(ы)", "дата приоритета"],
    }
    (args.out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"индекс: {args.out / 'index.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
