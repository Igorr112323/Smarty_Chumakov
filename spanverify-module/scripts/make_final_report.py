#!/usr/bin/env python3
"""Сборка итогового файла ``reports/РЕЗУЛЬТАТЫ_ПРОГОНА.md``.

Правило, которое определяет устройство скрипта: **ни одно число в отчёте не пишется
руками**. Каждая цифра читается из файла, созданного реально выполненной командой, и
рядом с ней указывается, из какого файла она взята. Если числа нет — в таблице стоит
``null`` и причина, а не правдоподобная величина.

Источники чисел:

* ``reports/METRICS.json`` — единый файл чисел (создаётся ``scripts/collect_metrics.py``);
* ``reports/TODO_AUDIT.json`` — реестр работ (``scripts/make_todo_audit.py``);
* ``data/corpus_a3/sources/sources.json`` и ``data/corpus_a3/manifest.json`` — корпус
  на реальных актах;
* ``reports/OCR_BENCH.json`` — замер скорости распознавания;
* ``reports/a3_test_hf.json``, ``reports/a3_test_demo.json``,
  ``reports/ext_*_hf.json``, ``reports/pilot/pilot.json`` — прогоны в режиме hf;
* ``reports/robustness.json``, ``reports/FUNCTIONAL_TESTS.json`` — устойчивость и
  функциональные замеры.

Запуск::

    python scripts/make_final_report.py
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent

NO = "null"


def read_json(path: Path) -> dict | None:
    """Прочитать JSON или вернуть ``None`` (отсутствие файла — это тоже факт)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def get(data: Any, path: str, default: Any = None) -> Any:
    """Значение по пути ``a.b.c``."""
    current = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def num(value: Any, digits: int = 4) -> str:
    """Число для таблицы; отсутствие — явный ``null``."""
    if value is None:
        return NO
    if isinstance(value, float):
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def git_info(root: Path) -> dict:
    """Коммит, ветка и дата последнего коммита — для воспроизводимости."""

    def run(args: list[str]) -> str:
        try:
            result = subprocess.run(args, cwd=root, check=False, capture_output=True)  # noqa: S603
            return result.stdout.decode("utf-8", "replace").strip()
        except OSError:
            return ""

    return {
        "commit": run(["git", "rev-parse", "HEAD"]) or NO,
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or NO,
        "committed_at": run(["git", "log", "-1", "--format=%cI"]) or NO,
        "describe": run(["git", "describe", "--tags", "--always"]) or NO,
    }


def hardware() -> dict:
    """Фактическое железо и ОС среды, в которой собран отчёт."""
    info: dict = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "processor": platform.processor() or platform.machine(),
    }
    try:
        import os

        info["cpu_count"] = os.cpu_count()
    except Exception:  # noqa: BLE001 - сведения о среде не должны ронять отчёт
        info["cpu_count"] = None
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
        for line in meminfo.splitlines():
            if line.startswith("MemTotal:"):
                info["memory_gb"] = round(int(line.split()[1]) / 1024 / 1024, 1)
                break
    except OSError:
        info["memory_gb"] = None
    return info


# ---------------------------------------------------------------------------
# Целевые показатели: цель — факт — достигнут ли
# ---------------------------------------------------------------------------

TARGETS: tuple[dict, ...] = (
    {
        "name": "Документов реальных опубликованных актов",
        "target": "≥ 120",
        "path": "a3_documents",
        "cmp": lambda value: value is not None and value >= 120,
        "source": "data/corpus_a3/sources/sources.json → documents_total",
    },
    {
        "name": "из них федеральных и ведомственных",
        "target": "≥ 60",
        "path": "a3_federal",
        "cmp": lambda value: value is not None and value >= 60,
        "source": "data/corpus_a3/sources/sources.json → by_level",
    },
    {
        "name": "из них региональных",
        "target": "≥ 40",
        "path": "a3_regional",
        "cmp": lambda value: value is not None and value >= 40,
        "source": "data/corpus_a3/sources/sources.json → by_level",
    },
    {
        "name": "Атомарных фактов, извлечённых из документов",
        "target": "≥ 1200",
        "path": "a3_facts",
        "cmp": lambda value: value is not None and value >= 1200,
        "source": "data/corpus_a3/sources/sources.json → facts_found_total",
    },
    {
        "name": "Пар «ответ — документ» в корпусе A3",
        "target": "≥ 1200",
        "path": "a3_pairs",
        "cmp": lambda value: value is not None and value >= 1200,
        "source": "data/corpus_a3/manifest.json → pairs",
    },
    {
        "name": "Доля чистых пар",
        "target": "0.40–0.50",
        "path": "a3_clean_share",
        "cmp": lambda value: value is not None and 0.40 <= value <= 0.50,
        "source": "data/corpus_a3/manifest.json → balance.clean_share",
    },
    {
        "name": "Минимум пар на каждый тип расхождения",
        "target": "≥ 40",
        "path": "a3_min_per_mode",
        "cmp": lambda value: value is not None and value >= 40,
        "source": "data/corpus_a3/manifest.json → balance.counts",
    },
    {
        "name": "Общих документов между частями train/dev/test",
        "target": "0",
        "path": "a3_shared_groups",
        "cmp": lambda value: value == 0,
        "source": "data/corpus_a3/manifest.json → shared_groups",
    },
    {
        "name": "Покрытие кода тестами",
        "target": "≥ 85 %",
        "path": "coverage",
        "cmp": lambda value: value is not None and value >= 85,
        "source": "reports/METRICS.json → tests.coverage_percent",
    },
    {
        "name": "Ложные срабатывания на чистых парах (корпус A1)",
        "target": "≤ 0.05",
        "path": "false_flags",
        "cmp": lambda value: value is not None and value <= 0.05,
        "source": "reports/METRICS.json → corpus_a.by_mode.faithful.verdict_fpr",
    },
    {
        "name": "Строгий span-F1 при IoU ≥ 0.5 (узкая разметка, отложенная часть)",
        "target": "≥ 0.50",
        "path": "narrow_f1",
        "cmp": lambda value: value is not None and value >= 0.50,
        "source": "reports/METRICS.json → corpus_a.by_split.test.spans.narrow.f1_iou_0_5",
    },
    {
        "name": "Ширина найденного фрагмента относительно эталона",
        "target": "≤ ×2",
        "path": "width_ratio",
        "cmp": lambda value: value is not None and value <= 2.0,
        "source": "reports/METRICS.json → corpus_a.by_split.test.spans.narrow.width_ratio",
    },
    {
        "name": "Прогонов в режиме hf",
        "target": "≥ 1",
        "path": "hf_runs",
        "cmp": lambda value: value is not None and value >= 1,
        "source": "reports/METRICS.json → external_tests.hf_runs",
    },
)


def collect_values(context: dict) -> dict:
    """Фактические значения целевых показателей."""
    metrics = context["metrics"] or {}
    a3 = context["a3"] or {}
    a3_manifest = context["a3_manifest"] or {}
    by_level = a3.get("by_level") or {}
    counts = get(a3_manifest, "balance.counts") or {}
    non_clean = {name: value for name, value in counts.items() if name != "faithful"}
    return {
        "a3_documents": a3.get("documents_total"),
        "a3_federal": by_level.get("федеральный"),
        "a3_regional": by_level.get("региональный"),
        "a3_facts": a3.get("facts_found_total"),
        "a3_pairs": a3_manifest.get("pairs"),
        "a3_clean_share": get(a3_manifest, "balance.clean_share"),
        "a3_min_per_mode": min(non_clean.values()) if non_clean else None,
        "a3_shared_groups": a3_manifest.get("shared_groups"),
        "coverage": get(metrics, "tests.coverage_percent"),
        "false_flags": get(metrics, "corpus_a.by_mode.faithful.verdict_fpr"),
        "narrow_f1": get(metrics, "corpus_a.by_split.test.spans.narrow.f1_iou_0_5"),
        "width_ratio": get(metrics, "corpus_a.by_split.test.spans.narrow.width_ratio"),
        "hf_runs": get(metrics, "external_tests.hf_runs"),
    }


# ---------------------------------------------------------------------------
# Комплект документов
# ---------------------------------------------------------------------------

DELIVERABLES: tuple[tuple[str, str, str], ...] = (
    ("Реестр незакрытых работ", "spanverify-module/reports/TODO_AUDIT.md", "генерируется скриптом"),
    ("Единый файл чисел", "spanverify-module/reports/METRICS.json", "генерируется скриптом"),
    ("Отчёт по корпусу A3", "spanverify-module/reports/CORPUS_A3.md", "генерируется скриптом"),
    ("Сводный отчёт по корпусам", "spanverify-module/reports/CORPUS_REPORT.md", "генерируется скриптом"),
    ("Доступность источников", "spanverify-module/reports/SOURCES_AVAILABILITY.json", "генерируется скриптом"),
    ("Замер скорости распознавания", "spanverify-module/reports/OCR_BENCH.json", "генерируется скриптом"),
    ("Отчёт по внешним наборам", "spanverify-module/reports/EXTERNAL_TESTS.md", "генерируется скриптом"),
    ("Устойчивость к искажениям", "spanverify-module/reports/robustness.md", "генерируется скриптом"),
    ("Функциональное тестирование", "spanverify-module/reports/FUNCTIONAL_TESTS.md", "генерируется скриптом"),
    ("Пилот на реальной модели", "spanverify-module/reports/pilot/pilot.md", "генерируется скриптом"),
    ("Техническое задание", "docs/ТЗ.md", "нужны данные человека: реквизиты договора"),
    ("Описание программы (ГОСТ 19.404)", "docs/ОПИСАНИЕ_ПРОГРАММЫ.md", "нужны данные человека: реквизиты"),
    ("Руководство пользователя (ГОСТ 19.505)", "docs/РУКОВОДСТВО_ПОЛЬЗОВАТЕЛЯ.md", "генерируется скриптом"),
    ("Руководство программиста (ГОСТ 19.504)", "docs/РУКОВОДСТВО_ПРОГРАММИСТА.md", "генерируется скриптом"),
    (
        "Руководство системного программиста (ГОСТ 19.503)",
        "docs/РУКОВОДСТВО_СИСТЕМНОГО_ПРОГРАММИСТА.md",
        "генерируется скриптом",
    ),
    ("Пояснительная записка", "docs/ПОЯСНИТЕЛЬНАЯ_ЗАПИСКА.md", "нужны данные человека: реквизиты"),
    ("Отчёт о НИР (ГОСТ 7.32-2017)", "docs/ОТЧЁТ_о_НИР_шаблон.md", "нужны данные человека: реквизиты, исполнители"),
    ("Материалы РИД: программа для ЭВМ", "docs/РИД_программа/", "нужны данные человека: правообладатель, авторы"),
    ("Материалы РИД: полезная модель", "docs/РИД_полезная_модель/", "нужны данные человека: авторы, заявитель"),
    ("Служебные сведения по РИД", "docs/РИД_СЛУЖЕБНЫЕ_СВЕДЕНИЯ.md", "нужны данные человека"),
)


def deliverable_rows(repo_root: Path) -> list[dict]:
    """Состояние каждого документа комплекта: есть файл или нет."""
    rows = []
    for title, relative, note in DELIVERABLES:
        path = repo_root / relative
        exists = path.exists()
        if not exists:
            status = "нужны данные человека" if "человека" in note else "не готово"
        elif "человека" in note:
            status = "готово, но нужны данные человека"
        else:
            status = "готово"
        size = None
        if path.is_file():
            size = path.stat().st_size
        elif path.is_dir():
            size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        rows.append(
            {
                "title": title,
                "path": relative,
                "exists": exists,
                "status": status,
                "note": note,
                "bytes": size,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Сборка текста
# ---------------------------------------------------------------------------


def load_context(root: Path) -> dict:
    """Собрать все файлы-источники чисел."""
    return {
        "root": root,
        "metrics": read_json(root / "reports" / "METRICS.json"),
        "audit": read_json(root / "reports" / "TODO_AUDIT.json"),
        "a3": read_json(root / "data" / "corpus_a3" / "sources" / "sources.json"),
        "a3_manifest": read_json(root / "data" / "corpus_a3" / "manifest.json"),
        "ocr_bench": read_json(root / "reports" / "OCR_BENCH.json"),
        "availability": read_json(root / "reports" / "SOURCES_AVAILABILITY.json"),
        "a3_hf": read_json(root / "reports" / "a3_test_hf.json"),
        "a3_demo": read_json(root / "reports" / "a3_test_demo.json"),
        "pilot": read_json(root / "reports" / "pilot" / "pilot.json"),
        "ext_qa_hf": read_json(root / "reports" / "ext_ragtruth_qa_hf.json"),
        "ext_rushallu_hf": read_json(root / "reports" / "ext_rushallu_hf.json"),
        "robustness": read_json(root / "reports" / "robustness.json"),
        "functional": read_json(root / "reports" / "FUNCTIONAL_TESTS.json"),
    }


def section_metrics_table(context: dict) -> list[str]:
    """Таблица метрик: корпус × режим × часть выборки."""
    metrics = context["metrics"] or {}
    lines = [
        "| Корпус | Часть | Режим | token P | token R | token F1 | token FPR | token AUC | вердикт F1 | вердикт FPR |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    def row(corpus: str, split: str, mode: str, block: dict | None) -> str:
        tokens = (block or {}).get("tokens") or {}
        verdicts = (block or {}).get("verdicts") or {}
        return (
            f"| {corpus} | {split} | {mode} | {num(tokens.get('precision'))} | {num(tokens.get('recall'))} "
            f"| {num(tokens.get('f1'))} | {num(tokens.get('fpr'))} | {num(tokens.get('auc'))} "
            f"| {num(verdicts.get('f1'))} | {num(verdicts.get('fpr'))} |"
        )

    lines.append(row("A1 (синтетика)", "test", "demo", get(metrics, "corpus_a.by_split.test")))
    lines.append(row("A1 (синтетика)", "dev", "demo", get(metrics, "corpus_a.by_split.dev")))
    lines.append(row("A3 (реальные акты)", "test", "demo", get(metrics, "corpus_a3_real.by_split.test")))
    lines.append(row("A3 (реальные акты)", "dev", "demo", get(metrics, "corpus_a3_real.by_split.dev")))
    if context["a3_hf"]:
        lines.append(row("A3 (реальные акты)", "test", "**hf**", context["a3_hf"]))
    else:
        lines.append("| A3 (реальные акты) | test | **hf** | null | null | null | null | null | null | null |")
    for name, run in (get(metrics, "external_tests.runs") or {}).items():
        lines.append(row(name, str(run.get("split", "test")), str(run.get("mode", "demo")), run))
    return lines


def section_spans(context: dict) -> list[str]:
    """Раздельные метрики узкой и расширенной разметки."""
    metrics = context["metrics"] or {}
    lines = [
        "| Корпус | Часть | Разметка | F1 при IoU ≥ 0.5 | precision | recall | накрытие | ширина к эталону |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for corpus, base in (
        ("A1 (синтетика)", "corpus_a.by_split.test.spans"),
        ("A3 (реальные акты)", "corpus_a3_real.by_split.test.spans"),
    ):
        for markup, title in (("narrow", "узкая"), ("expanded", "расширенная")):
            block = get(metrics, f"{base}.{markup}")
            if not block:
                lines.append(f"| {corpus} | test | {title} | null | null | null | null | null |")
                continue
            lines.append(
                f"| {corpus} | test | {title} | {num(block.get('f1_iou_0_5'))} | {num(block.get('precision'))} "
                f"| {num(block.get('recall'))} | {num(block.get('coverage'))} | ×{num(block.get('width_ratio'), 2)} |"
            )
    return lines


def section_by_kind(context: dict) -> list[str]:
    """Разбивка по типам расхождений."""
    metrics = context["metrics"] or {}
    lines = [
        "| Тип расхождения | Пар | Полнота вердикта | FPR вердикта | token F1 | накрытие фрагмента |",
        "|---|---|---|---|---|---|",
    ]
    by_mode = get(metrics, "corpus_a3_real.by_mode") or get(metrics, "corpus_a.by_mode") or {}
    for name, block in sorted(by_mode.items()):
        lines.append(
            f"| {name} | {num(block.get('pairs'))} | {num(block.get('verdict_recall'))} "
            f"| {num(block.get('verdict_fpr'))} | {num(block.get('token_f1'))} | {num(block.get('span_coverage'))} |"
        )
    if len(lines) == 2:
        lines.append("| null | null | null | null | null | null |")
    return lines


def render(context: dict) -> str:
    """Собрать итоговый отчёт."""
    metrics = context["metrics"] or {}
    audit = context["audit"] or {}
    a3 = context["a3"] or {}
    a3_manifest = context["a3_manifest"] or {}
    git = git_info(context["root"])
    env = hardware()
    values = collect_values(context)
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")

    out: list[str] = []
    add = out.append

    add("# Результаты прогона")
    add("")
    add(
        "Файл собран скриптом `scripts/make_final_report.py` и **не правится руками**. "
        "Каждое число прочитано из файла, созданного реально выполненной командой; рядом с "
        "таблицами указано, из какого файла взяты значения. Там, где измерения нет, стоит "
        "`null` и причина — величина не подставляется."
    )
    add("")

    # 1
    add("## 1. Что это за файл и как он получен")
    add("")
    add(f"* Собран: {generated}")
    add(f"* Коммит: `{git['commit']}`")
    add(f"* Ветка: `{git['branch']}`")
    add(f"* Дата коммита: {git['committed_at']}")
    add(f"* Версия модуля: {get(metrics, 'meta.version', NO)}")
    add(f"* Единый файл чисел создан: {get(metrics, 'meta.generated_at', NO)}")
    add(f"* Зерно обучения: {get(metrics, 'meta.seed', NO)}")
    add("")
    add("Команда сборки отчёта: `python scripts/make_final_report.py`")
    add("")

    # 2
    add("## 2. Среда и воспроизводимость")
    add("")
    add("| Параметр | Значение |")
    add("|---|---|")
    add(f"| Операционная система | {env['platform']} |")
    add(f"| Python | {env['python']} |")
    add(f"| Процессор | {env['processor']} |")
    add(f"| Логических ядер | {num(env.get('cpu_count'))} |")
    add(f"| Оперативная память, ГБ | {num(env.get('memory_gb'), 1)} |")
    add(f"| Тестов собрано | {num(get(metrics, 'tests.collected'))} |")
    add(f"| Покрытие, % | {num(get(metrics, 'tests.coverage_percent'), 2)} |")
    add("")
    add(
        "Важно о среде: загрузка документов и прогоны на реальной модели выполняются на "
        "раннерах GitHub, потому что среда разработки в интернет не выпускается. Числа "
        "режима `hf` получены на процессоре раннера `ubuntu-latest`, а не на видеокарте; "
        "это влияет на время, но не на значения метрик."
    )
    add("")

    # 3
    add("## 3. Таблица «цель — факт — достигнут ли»")
    add("")
    add("| Показатель | Цель | Факт | Достигнут | Откуда число |")
    add("|---|---|---|---|---|")
    reached = 0
    for target in TARGETS:
        value = values.get(str(target["path"]))
        ok = bool(target["cmp"](value))
        reached += int(ok)
        mark = "да" if ok else ("нет данных" if value is None else "нет")
        add(f"| {target['name']} | {target['target']} | {num(value)} | {mark} | `{target['source']}` |")
    add("")
    add(f"Достигнуто показателей: **{reached} из {len(TARGETS)}**.")
    add("")

    # 4
    add("## 4. Корпус на реальных опубликованных актах")
    add("")
    add("| Показатель | Значение |")
    add("|---|---|")
    add(f"| Документов | {num(a3.get('documents_total'))} |")
    add(f"| Знаков текста | {num(a3.get('text_chars_total'))} |")
    add(f"| Атомарных фактов | {num(a3.get('facts_found_total'))} |")
    add(f"| Пар | {num(a3_manifest.get('pairs'))} |")
    add(f"| Доля чистых пар | {num(get(a3_manifest, 'balance.clean_share'))} |")
    add(f"| Сплиты train/dev/test | {json.dumps(a3_manifest.get('splits'), ensure_ascii=False)} |")
    add(f"| Общих документов между частями | {num(a3_manifest.get('shared_groups'))} |")
    add(f"| Способы извлечения текста | {json.dumps(a3.get('extraction_methods'), ensure_ascii=False)} |")
    add("")
    add("Уровни и регионы:")
    add("")
    add("| Уровень / субъект | Документов |")
    add("|---|---|")
    for name, count in sorted((a3.get("by_level") or {}).items(), key=lambda item: -item[1]):
        add(f"| **{name}** | {count} |")
    for name, count in sorted((a3.get("by_region") or {}).items(), key=lambda item: -item[1]):
        add(f"| {name} | {count} |")
    add("")
    add("Виды актов:")
    add("")
    add("| Вид акта | Документов |")
    add("|---|---|")
    for name, count in sorted((a3.get("by_type") or {}).items(), key=lambda item: -item[1]):
        add(f"| {name} | {count} |")
    add("")
    add("Контрольные суммы корпуса:")
    add("")
    add("| Файл | SHA256 |")
    add("|---|---|")
    for name, digest in (a3_manifest.get("sha256") or {}).items():
        add(f"| `{name}` | `{digest}` |")
    if not (a3_manifest.get("sha256") or {}):
        add("| null | null |")
    add("")

    # 5
    add("## 5. Метрики по корпусам, режимам и частям выборки")
    add("")
    add(
        "Столбец «режим» обязателен: `demo` — лексические признаки без весов модели, "
        "`hf` — реальные веса языковой модели. Значения `demo` никогда не выдаются за `hf`."
    )
    add("")
    out.extend(section_metrics_table(context))
    add("")
    add("Источник чисел: `reports/METRICS.json`, `reports/a3_test_hf.json`.")
    add("")

    # 6
    add("## 6. Качество границ фрагментов")
    add("")
    add(
        "Узкая и расширенная разметка показаны раздельно. Это принципиально: узкий "
        "найденный фрагмент не даёт пересечения IoU ≥ 0.5 с целым предложением, поэтому "
        "сравнивать между собой можно только однородные пары."
    )
    add("")
    out.extend(section_spans(context))
    add("")

    # 7
    add("## 7. Разбивка по типам расхождений")
    add("")
    out.extend(section_by_kind(context))
    add("")

    # 8
    add("## 8. Пилот на реальной модели")
    add("")
    pilot = context["pilot"]
    if not pilot:
        add(
            "Пилот не выполнен в этом состоянии репозитория: файла `reports/pilot/pilot.json` нет. "
            "Команда: `python scripts/pilot_rugpt3small.py --pairs 24 --bootstrap 5000 --out reports/pilot`."
        )
    else:
        add(f"* Статус: {pilot.get('status', NO)}")
        add(f"* Модель: {get(pilot, 'environment.model', pilot.get('model', NO))}")
        add(f"* Пар: {num(pilot.get('pairs'))}, токенов: {num(pilot.get('tokens'))}")
        add(f"* Итераций бутстрэпа: {num(pilot.get('bootstrap_iterations'))}")
        add(f"* Длительность, с: {num(pilot.get('duration_s'), 1)}")
        add("")
        add("| Слой | Признак | AUC | 95 % ДИ |")
        add("|---|---|---|---|")
        for layer, features in (pilot.get("auc") or {}).items():
            for feature, block in features.items():
                add(
                    f"| {layer} | {feature} | {num(block.get('auc_oriented'))} "
                    f"| [{num(block.get('ci_low'))}; {num(block.get('ci_high'))}] |"
                )
    add("")

    # 9
    add("## 9. Устойчивость к искажениям")
    add("")
    robustness = context["robustness"]
    if not robustness:
        add("Замер не выполнен: файла `reports/robustness.json` нет. `null` вместо чисел.")
    else:
        add("| Искажение | Пар | token F1 | Δ к исходному | FPR |")
        add("|---|---|---|---|---|")
        for name, block in (robustness.get("distortions") or {}).items():
            add(
                f"| {name} | {num(block.get('pairs'))} | {num(block.get('f1'))} "
                f"| {num(block.get('delta'))} | {num(block.get('fpr'))} |"
            )
    add("")

    # 10
    add("## 10. Производительность и функциональные замеры")
    add("")
    functional = context["functional"]
    if not functional:
        add("Функциональные замеры не выполнены: файла `reports/FUNCTIONAL_TESTS.json` нет.")
    else:
        add("| Замер | Значение |")
        add("|---|---|")
        for name, value in functional.items():
            add(f"| {name} | {json.dumps(value, ensure_ascii=False)[:120]} |")
    add("")
    bench = context["ocr_bench"]
    add("Скорость распознавания официальных сканов (это самая дорогая часть сборки корпуса):")
    add("")
    if not bench:
        add("Замер не выполнен: файла `reports/OCR_BENCH.json` нет.")
    else:
        add(f"* Вывод замера: {bench.get('conclusion', NO)}")
        add(f"* Ядер на машине замера: {num(get(bench, 'environment.processor_count'))}")
        add("")
        add("| Настройка | Страниц | Рендер, с | Распознавание, с | Секунд на страницу | Знаков |")
        add("|---|---|---|---|---|---|")
        for variant in bench.get("variants") or []:
            add(
                f"| {variant.get('name')} | {num(variant.get('pages'))} | {num(variant.get('render_seconds'), 2)} "
                f"| {num(variant.get('ocr_seconds'), 2)} | {num(variant.get('seconds_per_page'), 2)} "
                f"| {num(variant.get('chars'))} |"
            )
    add("")

    # 11
    add("## 11. Сравнение с опубликованными работами")
    add("")
    comparison = get(metrics, "corpus_b.baseline_comparison")
    add(f"* Набор: {get(metrics, 'corpus_b.dataset', NO)}")
    add(f"* Источник: {get(metrics, 'corpus_b.citation', NO)}")
    if comparison:
        add(f"* Статья: {comparison.get('article_url', NO)}")
        add(f"* Как получены числа: {comparison.get('note', NO)}")
        add(f"* Извлечённых таблиц: {len(comparison.get('tables') or {})}")
    else:
        add(f"* Сравнение: `null`. Причина: {get(metrics, 'corpus_b.baseline_note', NO)}")
    add("")
    add("Наши числа на этом наборе (режим указан явно):")
    add("")
    add("| Показатель | Значение | Режим |")
    add("|---|---|---|")
    mode = get(metrics, "corpus_b.mode", NO)
    for key in ("f1", "precision", "recall", "fpr", "auc"):
        add(f"| token {key} | {num(get(metrics, f'corpus_b.our_metrics.tokens.{key}'))} | {mode} |")
    for key in ("accuracy", "jaccard_score", "rougeL"):
        add(f"| их {key} | {num(get(metrics, f'corpus_b.their_metrics.{key}'))} | {mode} |")
    add("")

    # 12
    add("## 12. Правовые основания и правила сбора данных")
    add("")
    add(
        "* Правовое основание использования текстов: **п. 6 ст. 1259 ГК РФ** — официальные "
        "документы государственных органов не являются объектами авторских прав."
    )
    add("* Источники — только официальные публикации из белого списка `config/sources_whitelist.json`.")
    add("* Персональные данные, закрытые и платные источники не скачивались.")
    add("* `robots.txt` каждого хоста читается до запросов; запрещённые пути не запрашиваются.")
    add("* Частота запросов — не чаще одного в секунду; User-Agent содержит контакт.")
    add(
        "* У каждого документа в `sources.json` записаны URL, вид, номер, дата акта, "
        "SHA256 текста и способ извлечения."
    )
    add("")
    availability = context["availability"]
    if availability:
        sources = availability.get("sources") or []
        ok = sum(1 for item in sources if item.get("available"))
        add(
            f"Доступность источников из среды разработки измерена: доступно {ok} из {len(sources)}. "
            "Поэтому загрузка выполняется на раннерах GitHub — это зафиксировано, а не обойдено молча."
        )
        add("")
    add(
        "**Отдельно о robots.txt портала опубликования.** В нём есть `Disallow: /File` с "
        "заглавной буквы, а адрес выгрузки — `/file/pdf?eoNumber=…` со строчной. RFC 9309 "
        "(п. 2.2.2) определяет сопоставление путей как чувствительное к регистру, и так же "
        "ведут себя `urllib.robotparser` и разборщик Google. Решение работать по "
        "регистрозависимому прочтению принято самостоятельно и **вынесено человеку на "
        "подтверждение**: это толкование, а не несомненность. Путь `/Search` запрещён и не "
        "запрашивается (проверяется тестом)."
    )
    add("")

    # 13
    add("## 13. Реестр незакрытых работ")
    add("")
    items = audit.get("items") or []
    if not items:
        add("Реестр не собран: файла `reports/TODO_AUDIT.json` нет.")
    else:
        closed = sum(1 for item in items if item["status"] == "закрыт")
        openned = sum(1 for item in items if item["status"] == "не закрыт")
        nodata = sum(1 for item in items if item["status"] == "нет данных")
        add(f"Всего пунктов {len(items)}: закрыто {closed}, не закрыто {openned}, нет данных {nodata}.")
        add("")
        add("| № | Приоритет | Пункт | Состояние | Измеренный факт |")
        add("|---|---|---|---|---|")
        for item in items:
            add(f"| {item['key']} | {item['priority']} | {item['title']} | {item['status']} | {item['detail']} |")
        add("")
        add("Подробности и команды закрытия — в `reports/TODO_AUDIT.md`.")
    add("")

    # 14
    add("## 14. Комплект документов и что нужно от человека")
    add("")
    add("| Документ | Путь | Состояние | Размер, байт |")
    add("|---|---|---|---|")
    rows = deliverable_rows(REPO_ROOT)
    for row in rows:
        add(f"| {row['title']} | `{row['path']}` | {row['status']} | {num(row['bytes'])} |")
    add("")
    need_human = [row for row in rows if "человека" in row["status"]]
    add("### Что нужно от человека")
    add("")
    if need_human:
        for row in need_human:
            add(f"* **{row['title']}** (`{row['path']}`) — {row['note']}.")
    else:
        add("* Дополнительных данных от человека для перечисленных документов не требуется.")
    add(
        "* Подтвердить толкование `robots.txt` портала опубликования (раздел 12): "
        "продолжать выгрузку по пути `/file/pdf` или сменить источник."
    )
    add(
        "* Решить по пунктам реестра, помеченным «не закрыт»: доделывать или зафиксировать "
        "фактическое значение в договорных документах."
    )
    add("")
    add("### Одна команда, которая воспроизводит всё локально")
    add("")
    add("```bash")
    add("cd spanverify-module && \\")
    add("  python -m pytest --cov=spanverify --cov-report=json:reports/coverage.json -q && \\")
    add("  python scripts/build_corpus_a.py --docs data/corpus_a3/sources --out data/corpus_a3 \\")
    add("      --target 1400 --seed 43 --real && \\")
    add("  python scripts/make_corpus_a3_report.py && \\")
    add("  python scripts/collect_metrics.py && \\")
    add("  python scripts/make_todo_audit.py && \\")
    add("  python scripts/make_final_report.py && \\")
    add("  python scripts/check_numbers.py")
    add("```")
    add("")
    add(
        "Скачивание самих актов в эту команду не входит: оно требует доступа в интернет и "
        "выполняется рабочим процессом `.github/workflows/corpus-a3.yml` "
        "(или локально: `python scripts/fetch_npa_corpus.py --out data/corpus_a3 --all-themes`)."
    )
    add("")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Сборка итогового файла результатов прогона")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "РЕЗУЛЬТАТЫ_ПРОГОНА.md")
    args = parser.parse_args()

    context = load_context(args.root)
    text = render(context)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")

    values = collect_values(context)
    reached = sum(1 for target in TARGETS if target["cmp"](values.get(str(target["path"]))))
    print(f"целевых показателей достигнуто: {reached} из {len(TARGETS)}")
    for target in TARGETS:
        value = values.get(str(target["path"]))
        if not target["cmp"](value):
            print(f"  не достигнут: {target['name']} — цель {target['target']}, факт {num(value)}")
    print(f"Отчёт: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
