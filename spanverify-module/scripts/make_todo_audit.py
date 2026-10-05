#!/usr/bin/env python3
"""Генератор реестра незакрытых работ ``reports/TODO_AUDIT.md`` (часть 1 задания).

Принцип
-------

Отчёт **не пишется руками**. Каждый пункт реестра — это правило, которое читает
фактические числа из ``reports/METRICS.json`` и сопутствующих файлов отчётов и
само решает, закрыт пункт или нет. Если нужного числа нет, пункт помечается
«нет данных» с указанием, какой командой это число получается, — но никогда не
помечается закрытым «на глаз».

Поэтому запуск скрипта на другом состоянии репозитория даст другой отчёт, и
расхождение между текстом и числами невозможно по построению.

Приоритеты
----------

* **P0** — мешает честно отчитаться по договору: заявленный режим не работает,
  число не измерено, корпус не собран.
* **P1** — качество ниже заявленного или метрика непригодна.
* **P2** — важно для продукта и сопровождения, но не влияет на отчётные числа.

Запуск::

    python scripts/make_todo_audit.py
    python scripts/make_todo_audit.py --out reports/TODO_AUDIT.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent

CLOSED = "закрыт"
OPEN = "не закрыт"
NO_DATA = "нет данных"


@dataclass
class Item:
    """Пункт реестра: как он проверяется и чем подтверждается."""

    key: str
    priority: str
    title: str
    was: str
    criterion: str
    check: Callable[[dict], tuple[str, str]]
    evidence: str
    command: str
    tags: list[str] = field(default_factory=list)


def _get(data: Any, path: str, default: Any = None) -> Any:
    """Значение по пути ``a.b.c``; ``default`` — если по дороге ничего нет."""
    current = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _fmt(value: Any) -> str:
    """Число для текста отчёта: без выдумывания точности."""
    if value is None:
        return "нет"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def load_context(root: Path) -> dict:
    """Собрать все факты, по которым оцениваются пункты реестра."""
    context: dict = {"root": root}

    metrics_path = root / "reports" / "METRICS.json"
    context["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    context["metrics_path"] = metrics_path

    manifest_path = root / "data" / "corpus_a3" / "sources" / "sources.json"
    context["a3"] = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    context["a3_path"] = manifest_path

    pairs_path = root / "data" / "corpus_a3" / "pairs.jsonl"
    context["a3_pairs"] = sum(1 for _ in pairs_path.open(encoding="utf-8")) if pairs_path.is_file() else 0

    bench_path = root / "reports" / "OCR_BENCH.json"
    context["ocr_bench"] = json.loads(bench_path.read_text(encoding="utf-8")) if bench_path.is_file() else {}

    availability = root / "reports" / "SOURCES_AVAILABILITY.json"
    context["availability"] = json.loads(availability.read_text(encoding="utf-8")) if availability.is_file() else {}

    review_path = root / "reports" / "review_queue.json"
    context["review"] = json.loads(review_path.read_text(encoding="utf-8")) if review_path.is_file() else {}

    return context


# ---------------------------------------------------------------------------
# Правила проверки пунктов
# ---------------------------------------------------------------------------


def check_hf_runs(context: dict) -> tuple[str, str]:
    """Режим hf: сколько прогонов на внешних наборах выполнено в нём."""
    runs = _get(context["metrics"], "external_tests.hf_runs")
    demo = _get(context["metrics"], "external_tests.demo_runs")
    if runs is None:
        return NO_DATA, "в METRICS.json нет поля external_tests.hf_runs"
    if int(runs) > 0:
        return CLOSED, f"прогонов в режиме hf: {runs} (в режиме demo: {demo})"
    return OPEN, f"прогонов в режиме hf: 0, все {demo} прогона выполнены в режиме demo"


def check_pilot_auc(context: dict) -> tuple[str, str]:
    """Пилот на реальной модели: отличается ли AUC признаков от случая."""
    pilot = _get(context["metrics"], "pilot")
    if not pilot:
        return NO_DATA, "в METRICS.json нет раздела pilot (артефакт CI не подложен)"
    best = None
    for features in (pilot.get("auc") or {}).values():
        for values in features.values():
            auc = values.get("auc_oriented")
            low = values.get("ci_low")
            if auc is None:
                continue
            if best is None or auc > best[0]:
                best = (auc, low, values.get("ci_high"))
    if best is None:
        return NO_DATA, "в разделе pilot нет значений AUC"
    auc, low, high = best
    if low is not None and low > 0.5:
        return CLOSED, f"лучший AUC признака {auc:.3f}, 95 % ДИ [{low:.3f}; {high:.3f}] — нижняя граница выше 0.5"
    return OPEN, f"лучший AUC признака {auc:.3f}, 95 % ДИ [{low}; {high}] — ноль-гипотеза не отвергнута"


def check_span_quality(context: dict) -> tuple[str, str]:
    """Точность границ фрагментов: строгий F1 и во сколько раз шире эталона."""
    narrow = _get(context["metrics"], "corpus_a.by_split.test.spans.narrow")
    if not narrow:
        return NO_DATA, "в METRICS.json нет corpus_a.by_split.test.spans.narrow"
    f1 = float(narrow.get("f1_iou_0_5", 0.0))
    width = float(narrow.get("width_ratio", 0.0))
    if f1 >= 0.5 and width <= 2.0:
        return CLOSED, f"узкая разметка на отложенной части: строгий F1 при IoU≥0.5 = {f1:.4f}, ширина ×{width:.2f}"
    return OPEN, f"узкая разметка: строгий F1 = {f1:.4f}, ширина ×{width:.2f} (цель: F1 ≥ 0.5 и ширина ≤ ×2)"


def _by_mode(context: dict, mode: str, field_name: str) -> float | None:
    value = _get(context["metrics"], f"corpus_a.by_mode.{mode}.{field_name}")
    return float(value) if value is not None else None


def check_missing(context: dict) -> tuple[str, str]:
    """Пропущенный факт документа: ловится ли он вообще."""
    recall = _by_mode(context, "missing", "verdict_recall")
    pairs = _get(context["metrics"], "corpus_a.by_mode.missing.pairs")
    if recall is None:
        return NO_DATA, "в METRICS.json нет corpus_a.by_mode.missing"
    if recall >= 0.7:
        return CLOSED, f"полнота по вердикту на типе «missing» = {recall:.4f} ({pairs} пар)"
    return OPEN, f"полнота по вердикту на типе «missing» = {recall:.4f} ({pairs} пар), цель ≥ 0.7"


def check_partial(context: dict) -> tuple[str, str]:
    """Частично подтверждённый ответ."""
    recall = _by_mode(context, "partial", "verdict_recall")
    pairs = _get(context["metrics"], "corpus_a.by_mode.partial.pairs")
    if recall is None:
        return NO_DATA, "в METRICS.json нет corpus_a.by_mode.partial"
    if recall >= 0.7:
        return CLOSED, f"полнота по вердикту на типе «partial» = {recall:.4f} ({pairs} пар)"
    return OPEN, f"полнота по вердикту на типе «partial» = {recall:.4f} ({pairs} пар), цель ≥ 0.7"


def check_false_flags(context: dict) -> tuple[str, str]:
    """Ложные срабатывания на чистых парах."""
    fpr = _by_mode(context, "faithful", "verdict_fpr")
    pairs = _get(context["metrics"], "corpus_a.by_mode.faithful.pairs")
    if fpr is None:
        return NO_DATA, "в METRICS.json нет corpus_a.by_mode.faithful"
    if fpr <= 0.05:
        return CLOSED, f"доля ложных срабатываний на чистых парах = {fpr:.4f} ({pairs} пар), порог 0.05"
    return OPEN, f"доля ложных срабатываний на чистых парах = {fpr:.4f} ({pairs} пар), цель ≤ 0.05"


def check_corpus_a3(context: dict) -> tuple[str, str]:
    """Корпус на реальных актах: сколько документов, уровней, регионов, пар."""
    documents = (context["a3"].get("documents") or {}) if context["a3"] else {}
    if not documents:
        return OPEN, "корпус A3 не собран: в data/corpus_a3/sources/sources.json нет документов"
    total = len(documents)
    by_level = context["a3"].get("by_level") or {}
    federal = int(by_level.get("федеральный", 0))
    regional = int(by_level.get("региональный", 0))
    facts = sum(int(item.get("facts_found") or 0) for item in documents.values())
    pairs = context["a3_pairs"]
    detail = (
        f"документов {total} (федеральных {federal}, региональных {regional}), "
        f"атомарных фактов {facts}, пар {pairs}"
    )
    if total >= 120 and federal >= 60 and regional >= 40 and facts >= 1200 and pairs >= 1200:
        return CLOSED, detail
    return OPEN, detail + " — цель: ≥120 документов (≥60 федеральных, ≥40 региональных), ≥1200 фактов, ≥1200 пар"


def check_corpus_a2(context: dict) -> tuple[str, str]:
    """Корпус A2: собран ли он по-настоящему или остался сухим прогоном."""
    available = _get(context["metrics"], "corpus_a2.available")
    if available is None:
        return OPEN, "в METRICS.json нет раздела corpus_a2: набор остаётся сухим прогоном"
    if available:
        return CLOSED, f"corpus_a2 собран: пар {_get(context['metrics'], 'corpus_a2.pairs')}"
    return OPEN, f"corpus_a2 недоступен: {_get(context['metrics'], 'corpus_a2.note')}"


def check_participation(context: dict) -> tuple[str, str]:
    """Доля участия ИИ: на каких текстах откалибрована."""
    calibrated = _get(context["metrics"], "demo.participation.calibrated_on")
    if calibrated is None:
        return NO_DATA, "в METRICS.json нет demo.participation.calibrated_on"
    if "synthetic" in str(calibrated).lower():
        return OPEN, f"калибровка выполнена на синтетике: «{calibrated}»"
    return CLOSED, f"калибровка выполнена на реальных текстах: «{calibrated}»"


def check_baseline(context: dict) -> tuple[str, str]:
    """Сравнение с опубликованными базовыми значениями."""
    comparison = _get(context["metrics"], "corpus_b.baseline_comparison")
    note = _get(context["metrics"], "corpus_b.baseline_note")
    if comparison:
        return CLOSED, f"сравнение с базовыми значениями выполнено: {json.dumps(comparison, ensure_ascii=False)[:200]}"
    return OPEN, f"базовые значения не извлечены: {note}"


def check_review_queue(context: dict) -> tuple[str, str]:
    """Очередь ручной проверки пар."""
    review = context["review"]
    if not review:
        return NO_DATA, "файла reports/review_queue.json нет: очередь ручной проверки не зафиксирована"
    total = int(review.get("total") or 0)
    checked = int(review.get("checked") or 0)
    if total and checked >= total:
        return CLOSED, f"проверено {checked} из {total} пар очереди"
    return OPEN, f"проверено {checked} из {total} пар очереди"


def check_transfer(context: dict) -> tuple[str, str]:
    """Падение качества при переносе на чужой корпус."""
    cross = _get(context["metrics"], "cross_corpus")
    if not cross:
        return NO_DATA, "в METRICS.json нет раздела cross_corpus"
    drop = float(cross.get("drop", 0.0))
    external = [
        (name, _get(run, "tokens.f1")) for name, run in (_get(context["metrics"], "external_tests.runs") or {}).items()
    ]
    worst = min((value for _name, value in external if value is not None), default=None)
    detail = (
        f"свой корпус F1 {cross.get('in_corpus_f1')}, перенос {cross.get('cross_corpus_f1')} "
        f"(падение {drop:.4f}), FPR на чужом корпусе {cross.get('cross_fpr')}"
    )
    if worst is not None:
        detail += f"; худший token F1 на внешних наборах {worst}"
    if worst is not None and worst >= 0.5:
        return CLOSED, detail
    return OPEN, detail + " — перенос остаётся низким"


def check_sources_available(context: dict) -> tuple[str, str]:
    """Доступность официальных источников из среды сборки."""
    availability = context["availability"]
    if not availability:
        return NO_DATA, "файла reports/SOURCES_AVAILABILITY.json нет"
    sources = availability.get("sources") or []
    total = len(sources)
    ok = sum(1 for item in sources if item.get("available"))
    if total and ok == total:
        return CLOSED, f"доступны все {total} источников белого списка"
    return OPEN, f"из {total} источников белого списка доступны {ok}: загрузка выполняется на раннере GitHub"


def check_ocr_cost(context: dict) -> tuple[str, str]:
    """Стоимость распознавания: измерена ли она и выбраны ли настройки по замеру."""
    bench = context["ocr_bench"]
    if not bench:
        return NO_DATA, "замер reports/OCR_BENCH.json не выполнен"
    conclusion = bench.get("conclusion")
    if bench.get("fastest"):
        return CLOSED, str(conclusion)
    return OPEN, str(conclusion)


def check_coverage(context: dict) -> tuple[str, str]:
    """Покрытие тестами."""
    percent = _get(context["metrics"], "tests.coverage_percent")
    gate = _get(context["metrics"], "tests.coverage_gate_percent")
    collected = _get(context["metrics"], "tests.collected")
    if percent is None:
        return NO_DATA, "в METRICS.json нет tests.coverage_percent"
    if float(percent) >= float(gate or 85):
        return CLOSED, f"покрытие {percent} % при пороге {gate} %, тестов собрано {collected}"
    return OPEN, f"покрытие {percent} % ниже порога {gate} %"


REGISTRY: tuple[Item, ...] = (
    Item(
        key="P0-1",
        priority="P0",
        title="Режим hf не прогонялся на внешних наборах",
        was="external_tests.hf_runs = 0, все числа получены в режиме demo",
        criterion="хотя бы один прогон внешнего набора выполнен в режиме hf",
        check=check_hf_runs,
        evidence="reports/METRICS.json → external_tests",
        command="python scripts/external_eval.py --mode hf",
        tags=["режим", "договор"],
    ),
    Item(
        key="P0-2",
        priority="P0",
        title="Пилот на реальной модели: AUC признаков неотличим от случайного",
        was="AUC ≈ 0.50, доверительный интервал накрывает 0.5",
        criterion="нижняя граница 95 % ДИ хотя бы одного признака выше 0.5",
        check=check_pilot_auc,
        evidence="reports/pilot/pilot.json (артефакт job «Пилот» в CI)",
        command="python scripts/pilot_rugpt3small.py --pairs 24 --bootstrap 5000 --out reports/pilot",
        tags=["признаки", "hf"],
    ),
    Item(
        key="P0-3",
        priority="P0",
        title="Корпус A3 на реальных опубликованных актах не собран",
        was="корпус отсутствовал; числа считались на синтетике",
        criterion="≥120 документов (≥60 федеральных, ≥40 региональных), ≥1200 фактов, ≥1200 пар",
        check=check_corpus_a3,
        evidence="data/corpus_a3/sources/sources.json, data/corpus_a3/pairs.jsonl",
        command="изменить .github/corpus-a3.trigger → рабочий процесс corpus-a3",
        tags=["корпус", "данные"],
    ),
    Item(
        key="P0-4",
        priority="P0",
        title="Базовые значения из опубликованных работ не извлечены",
        was="baseline = null, сравнивать не с чем",
        criterion="значения базовых работ извлечены из PDF либо явно помечены null с указанием файла",
        check=check_baseline,
        evidence="reports/METRICS.json → corpus_b.baseline_note",
        command="python scripts/extract_baseline_pdf.py",
        tags=["сравнение"],
    ),
    Item(
        key="P0-5",
        priority="P0",
        title="Очередь ручной проверки пар не пройдена",
        was="18 пар в очереди ручной проверки, отметок о проверке нет",
        criterion="все пары очереди имеют отметку проверки",
        check=check_review_queue,
        evidence="reports/review_queue.json",
        command="python scripts/review_queue.py --check",
        tags=["разметка"],
    ),
    Item(
        key="P1-1",
        priority="P1",
        title="Границы фрагментов шире эталона, строгий span-F1 низкий",
        was="span-F1 0.098–0.283 при ширине ×24 от эталона",
        criterion="строгий F1 при IoU ≥ 0.5 не ниже 0.5 и ширина не больше ×2 на отложенной части",
        check=check_span_quality,
        evidence="reports/METRICS.json → corpus_a.by_split.test.spans.narrow",
        command="python -m spanverify evaluate --dataset data/corpus_a/pairs.jsonl",
        tags=["фрагменты"],
    ),
    Item(
        key="P1-2",
        priority="P1",
        title="Пропущенный факт документа не обнаруживается",
        was="полнота на типе «missing» = 0.0",
        criterion="полнота по вердикту на типе «missing» не ниже 0.7",
        check=check_missing,
        evidence="reports/METRICS.json → corpus_a.by_mode.missing",
        command="python -m spanverify evaluate --dataset data/corpus_a/pairs.jsonl",
        tags=["покрытие фактов"],
    ),
    Item(
        key="P1-3",
        priority="P1",
        title="Частично подтверждённый ответ почти не обнаруживается",
        was="полнота на типе «partial» = 0.2667",
        criterion="полнота по вердикту на типе «partial» не ниже 0.7",
        check=check_partial,
        evidence="reports/METRICS.json → corpus_a.by_mode.partial",
        command="python -m spanverify evaluate --dataset data/corpus_a/pairs.jsonl",
        tags=["покрытие фактов"],
    ),
    Item(
        key="P1-4",
        priority="P1",
        title="Ложные срабатывания на чистых парах",
        was="14 % чистых пар помечались как сомнительные",
        criterion="доля ложных срабатываний на чистых парах не выше 5 %",
        check=check_false_flags,
        evidence="reports/METRICS.json → corpus_a.by_mode.faithful",
        command="python -m spanverify evaluate --dataset data/corpus_a/pairs.jsonl",
        tags=["ложные тревоги"],
    ),
    Item(
        key="P1-5",
        priority="P1",
        title="Доля участия ИИ откалибрована только на синтетике",
        was="calibrated_on = synthetic-participation-corpus",
        criterion="калибровка выполнена на реальных текстах со смесями 0/0.25/0.5/0.75/1",
        check=check_participation,
        evidence="reports/METRICS.json → demo.participation.calibrated_on",
        command="python scripts/calibrate_participation.py --real",
        tags=["доля ИИ"],
    ),
    Item(
        key="P1-6",
        priority="P1",
        title="Качество не переносится на чужие данные",
        was="0.953 на своём корпусе → 0.717 при переносе → 0.10–0.20 на внешних наборах",
        criterion="token F1 на внешних наборах не ниже 0.5",
        check=check_transfer,
        evidence="reports/METRICS.json → cross_corpus, external_tests.runs",
        command="python scripts/cross_corpus.py --out reports/cross_corpus.json",
        tags=["перенос"],
    ),
    Item(
        key="P1-7",
        priority="P1",
        title="Корпус A2 остался сухим прогоном",
        was="A2 не собирался, числа по нему отсутствуют",
        criterion="раздел corpus_a2 в METRICS.json содержит собранный набор",
        check=check_corpus_a2,
        evidence="reports/METRICS.json → corpus_a2",
        command="python scripts/build_corpus_a2.py",
        tags=["корпус"],
    ),
    Item(
        key="P2-1",
        priority="P2",
        title="Официальные источники недоступны из среды сборки",
        was="попытки загрузки молча не удавались",
        criterion="доступность каждого источника белого списка измерена и записана",
        check=check_sources_available,
        evidence="reports/SOURCES_AVAILABILITY.json",
        command="python scripts/probe_sources_availability.py --out reports/SOURCES_AVAILABILITY.json",
        tags=["источники"],
    ),
    Item(
        key="P2-2",
        priority="P2",
        title="Стоимость распознавания сканов не измерена",
        was="настройки OCR выбраны по умолчанию, время прогона не укладывалось в лимит",
        criterion="время распознавания измерено на реальном файле и настройки выбраны по замеру",
        check=check_ocr_cost,
        evidence="reports/OCR_BENCH.json",
        command="python scripts/bench_ocr.py --out reports/OCR_BENCH.json --pages 2",
        tags=["производительность"],
    ),
    Item(
        key="P2-3",
        priority="P2",
        title="Покрытие тестами ниже порога",
        was="порог 85 % не подтверждался числом в едином файле",
        criterion="покрытие не ниже порога, число записано в METRICS.json",
        check=check_coverage,
        evidence="reports/METRICS.json → tests",
        command="python -m pytest --cov=spanverify --cov-fail-under=85",
        tags=["качество кода"],
    ),
)


def git_commit(root: Path) -> str:
    """Текущий коммит (для воспроизводимости отчёта)."""
    try:
        result = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
        )
        return result.stdout.decode().strip() or "неизвестен"
    except OSError:
        return "неизвестен"


def evaluate(context: dict) -> list[dict]:
    """Прогнать все правила и вернуть состояние каждого пункта."""
    rows = []
    for item in REGISTRY:
        status, detail = item.check(context)
        rows.append(
            {
                "key": item.key,
                "priority": item.priority,
                "title": item.title,
                "was": item.was,
                "criterion": item.criterion,
                "status": status,
                "detail": detail,
                "evidence": item.evidence,
                "command": item.command,
                "tags": item.tags,
            }
        )
    return rows


def render(rows: list[dict], context: dict) -> str:
    """Собрать текст отчёта из состояний пунктов."""
    closed = [row for row in rows if row["status"] == CLOSED]
    openned = [row for row in rows if row["status"] == OPEN]
    nodata = [row for row in rows if row["status"] == NO_DATA]
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    commit = git_commit(context["root"])
    meta = _get(context["metrics"], "meta", {}) or {}

    lines: list[str] = []
    lines.append("# Реестр незакрытых работ (аудит)")
    lines.append("")
    lines.append(
        "Файл создаётся скриптом `scripts/make_todo_audit.py` и **не правится руками**. "
        "Каждый пункт закрывается не мнением, а числом: правило проверки читает "
        "`reports/METRICS.json` и сопутствующие файлы и само ставит отметку. "
        "Если числа нет, пункт помечается «нет данных» и указывается команда, которая его даёт."
    )
    lines.append("")
    lines.append(f"* Отчёт собран: {generated}")
    lines.append(f"* Коммит: `{commit}`")
    lines.append(f"* Версия модуля: {meta.get('version', 'нет')}")
    lines.append(f"* Файл чисел: `reports/METRICS.json` (создан {meta.get('generated_at', 'нет данных')})")
    lines.append(f"* Зерно обучения: {meta.get('seed', 'нет')}")
    lines.append("")
    lines.append("## Сводка")
    lines.append("")
    lines.append("| Состояние | Пунктов |")
    lines.append("|---|---|")
    lines.append(f"| Закрыто | {len(closed)} |")
    lines.append(f"| Не закрыто | {len(openned)} |")
    lines.append(f"| Нет данных | {len(nodata)} |")
    lines.append(f"| **Всего** | **{len(rows)}** |")
    lines.append("")
    lines.append("По приоритетам:")
    lines.append("")
    lines.append("| Приоритет | Закрыто | Не закрыто | Нет данных |")
    lines.append("|---|---|---|---|")
    for priority in ("P0", "P1", "P2"):
        subset = [row for row in rows if row["priority"] == priority]
        lines.append(
            f"| {priority} | {sum(1 for row in subset if row['status'] == CLOSED)} "
            f"| {sum(1 for row in subset if row['status'] == OPEN)} "
            f"| {sum(1 for row in subset if row['status'] == NO_DATA)} |"
        )
    lines.append("")
    lines.append("## Реестр")
    lines.append("")
    lines.append("| № | Приоритет | Что было | Критерий закрытия | Состояние | Измеренный факт |")
    lines.append("|---|---|---|---|---|---|")
    for row in rows:
        mark = {CLOSED: "✅ закрыт", OPEN: "❌ не закрыт", NO_DATA: "⚠️ нет данных"}[row["status"]]
        lines.append(
            f"| {row['key']} | {row['priority']} | {row['was']} | {row['criterion']} " f"| {mark} | {row['detail']} |"
        )
    lines.append("")
    lines.append("## Подробно по пунктам")
    lines.append("")
    for row in rows:
        mark = {CLOSED: "закрыт", OPEN: "не закрыт", NO_DATA: "нет данных"}[row["status"]]
        lines.append(f"### {row['key']} ({row['priority']}) — {row['title']}")
        lines.append("")
        lines.append(f"* **Состояние:** {mark}")
        lines.append(f"* **Было:** {row['was']}")
        lines.append(f"* **Критерий закрытия:** {row['criterion']}")
        lines.append(f"* **Измеренный факт:** {row['detail']}")
        lines.append(f"* **Чем подтверждается:** {row['evidence']}")
        lines.append(f"* **Команда:** `{row['command']}`")
        lines.append("")
    lines.append("## Как читать отметки")
    lines.append("")
    lines.append(
        "* **закрыт** — правило нашло число, и число удовлетворяет критерию. "
        "Не «сделано», а «измерено и соответствует».\n"
        "* **не закрыт** — число найдено и критерию не удовлетворяет. "
        "Это не оценка усилий, а факт.\n"
        "* **нет данных** — число не получено. Пункт нельзя считать ни закрытым, ни проваленным; "
        "в строке указана команда, которой число добывается."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Сборка реестра незакрытых работ")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "TODO_AUDIT.md")
    parser.add_argument("--json", type=Path, default=ROOT / "reports" / "TODO_AUDIT.json")
    args = parser.parse_args()

    context = load_context(args.root)
    rows = evaluate(context)
    text = render(rows, context)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    args.json.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "commit": git_commit(args.root),
                "items": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    closed = sum(1 for row in rows if row["status"] == CLOSED)
    openned = sum(1 for row in rows if row["status"] == OPEN)
    nodata = sum(1 for row in rows if row["status"] == NO_DATA)
    print(f"пунктов: {len(rows)}; закрыто {closed}; не закрыто {openned}; нет данных {nodata}")
    for row in rows:
        if row["status"] != CLOSED:
            print(f"  {row['key']} [{row['status']}] {row['title']}: {row['detail']}")
    print(f"Отчёт: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
