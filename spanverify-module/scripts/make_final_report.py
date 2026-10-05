"""Итоговый отчёт о прогоне: 12 разделов (пункт 5 промта).

Отчёт **генерируется скриптом** из файлов результатов; ни одно число не
вписывается руками. Если файла нет, в разделе стоит «нет данных» с указанием,
какого файла не хватает — выдуманных значений не бывает.

Разделы:

1. что было не доделано и что стало (по журналу ``reports/audit_fixes.json``);
2. состояние кода (модули, тесты, линтеры, сборки);
3. реальные акты: уровни, органы, темы, факты, SHA256, недоступные;
4. корпус: пары, режимы, чистота, разбиения, очередь проверки;
5. метрики (режим, seed, n, команда, железо);
6. пилот: ДИ, гипотеза, контроль контраста, базовые уровни;
7. устойчивость к искажениям;
8. ресурсы и производительность;
9. внешние наборы и аналоги;
10. цель/факт по каждой целевой метрике;
11. честный список неизмеренного;
12. команды воспроизведения.

Запуск::

    python scripts/make_final_report.py --out reports/РЕЗУЛЬТАТЫ_ПРОГОНА.md
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spanverify import __version__  # noqa: E402

NO_DATA = "нет данных"


def read_json(path: Path) -> dict[str, Any] | None:
    """Прочитать JSON или вернуть ``None`` (без подстановок)."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def get(data: Any, *path: str, default: Any = None) -> Any:
    """Значение по пути; ``default`` вместо исключения."""
    current = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def fmt(value: Any, digits: int = 4) -> str:
    """Число в отчёт: без выдумок, с разумным округлением."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def missing(path: Path) -> str:
    """Пометка об отсутствии файла (в отчёт, а не в консоль)."""
    return f"{NO_DATA} (нет файла `{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}`)"


def git_log(limit: int = 25) -> list[str]:
    """Последние коммиты ветки (для раздела «что сделано»)."""
    try:
        result = subprocess.run(
            ["git", "log", f"-{limit}", "--pretty=%h %ad %s", "--date=short"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        return [line for line in result.stdout.splitlines() if line.strip()]
    except OSError:  # pragma: no cover - git есть в CI и локально
        return []


def section_1(audit: dict[str, Any] | None) -> list[str]:
    lines = ["## 1. Незавершённые места: было → стало", ""]
    if not audit:
        lines += [
            missing(ROOT / "reports" / "audit_fixes.json"),
            "",
            "Журнал соответствия «находка аудита → исправление» не найден; перечень находок —",
            "в `reports/TODO_AUDIT.md`.",
            "",
        ]
        return lines
    lines += [
        "| № | Находка аудита | Приоритет | Статус | Что сделано (проверка) | Коммит |",
        "|---|---|---|---|---|---|",
    ]
    for item in audit.get("items", []):
        evidence = str(item.get("evidence", "—")).replace("|", "/").replace("\n", " ")
        lines.append(
            "| {id} | {title} | {priority} | {status} | {evidence} | {commit} |".format(
                id=item.get("id", "—"),
                title=item.get("title", "—"),
                priority=item.get("priority", "—"),
                status=item.get("status", "—"),
                evidence=evidence,
                commit=f"`{item.get('commit', '—')}`",
            )
        )
    if audit.get("unfinished"):
        lines += ["", "Осталось незавершённым: " + "; ".join(audit["unfinished"]) + ".", ""]
    lines += ["", "Последние коммиты:", "", "```", *git_log(), "```", ""]
    return lines


def section_2(code: dict[str, Any] | None) -> list[str]:
    lines = ["## 2. Состояние кода", ""]
    if not code:
        lines += [missing(ROOT / "reports" / "code_state.json"), ""]
        return lines
    lines += [
        f"Версия: {code.get('version', __version__)}. Python: {code.get('python', platform.python_version())}.",
        "",
        "| Проверка | Результат |",
        "|---|---|",
        f"| Тесты | {code.get('tests_passed', NO_DATA)} из {code.get('tests_total', NO_DATA)} "
        f"(пропущено {code.get('tests_skipped', 0)}) |",
        f"| Покрытие | {fmt(code.get('coverage_percent'))} % |",
        f"| ruff | {code.get('ruff', NO_DATA)} |",
        f"| black | {code.get('black', NO_DATA)} |",
        f"| Модулей в пакете | {code.get('modules', NO_DATA)} |",
        f"| Строк кода | {code.get('lines', NO_DATA)} |",
        "",
    ]
    builds = code.get("builds") or {}
    if builds:
        lines += ["| Сборка | Размер, байт | Проверка |", "|---|---|---|"]
        for name, payload in builds.items():
            lines.append(f"| {name} | {payload.get('bytes', 'null')} | {payload.get('check', NO_DATA)} |")
        lines.append("")
    return lines


def section_3(sources: dict[str, Any] | None) -> list[str]:
    lines = ["## 3. Реальные акты", ""]
    if not sources:
        lines += [missing(ROOT / "data" / "corpus_a3" / "sources" / "sources.json"), ""]
        return lines
    documents = sources.get("documents") or {}
    by_level = sources.get("by_level") or {}
    by_region = sources.get("by_region") or {}
    region_status = sources.get("region_status") or {}
    lines += [
        f"Всего актов с текстом: {len(documents)} (цель — не меньше 120).",
        f"Уровни: {', '.join(f'{name} — {count}' for name, count in sorted(by_level.items())) or NO_DATA}.",
        f"Регионы: {', '.join(f'{name} — {count}' for name, count in sorted(by_region.items())) or NO_DATA}.",
        f"Темы: {', '.join(f'{name} — {count}' for name, count in sorted((sources.get('by_theme') or {}).items())) or NO_DATA}.",
        f"Виды: {', '.join(f'{name} — {count}' for name, count in sorted((sources.get('by_type') or {}).items())) or NO_DATA}.",
        f"Способы извлечения текста: {sources.get('extraction_methods') or NO_DATA}.",
        "",
        "Квоты регионов:",
        "",
        "| Код | Регион | Требовалось | Найдено | Выполнено |",
        "|---|---|---|---|---|",
    ]
    for code, payload in sorted(region_status.items()):
        lines.append(
            f"| {code} | {payload.get('name') or '—'} | {payload.get('wanted')} | {payload.get('found')} | "
            f"{'да' if payload.get('available') else 'нет'} |"
        )
    unavailable = sources.get("unavailable") or []
    lines += [
        "",
        f"Недоступные документы (с фактической причиной): {len(unavailable)} из "
        f"{sources.get('skipped_total', 0)} пропущенных записей; в отчёте показаны первые 15.",
        "",
        "| Актов | SHA256 PDF (первые 12 символов) | Текст, знаков | Способ |",
        "|---|---|---|---|",
    ]
    for doc_id, payload in sorted(documents.items())[:15]:
        sha = str(payload.get("pdf_sha256") or "")[:12]
        lines.append(
            f"| `{doc_id}` ({payload.get('act_type', '—')} {payload.get('act_number', '')}) | {sha} | "
            f"{payload.get('text_chars', '—')} | {payload.get('extraction', '—')} |"
        )
    if unavailable:
        lines += ["", "| Пропущенный акт | Причина |", "|---|---|"]
        for item in unavailable[:15]:
            lines.append(f"| `{item.get('doc_id') or '—'}` | {item.get('reason') or '—'} |")
    lines += [
        "",
        f"Правовое основание: {sources.get('legal_basis', NO_DATA)}.",
        f"Политика по персональным данным: {sources.get('personal_data_policy', NO_DATA)}.",
        f"robots.txt учитывался: пауза {sources.get('pause_seconds', NO_DATA)} с, "
        f"белый список {sources.get('whitelist', NO_DATA)}.",
        "",
    ]
    return lines


def section_4(manifest: dict[str, Any] | None, summary: dict[str, Any] | None) -> list[str]:
    lines = ["## 4. Корпус", ""]
    if not manifest:
        lines += [missing(ROOT / "data" / "corpus_a3" / "manifest.json"), ""]
    else:
        counts = get(manifest, "balance", "counts", default={})
        lines += [
            f"Пар: {manifest.get('pairs', NO_DATA)} (цель 1200), документов: {get(manifest, 'documents', 'count', default=NO_DATA)}.",
            f"Чистых: {get(manifest, 'balance', 'clean', default=NO_DATA)} "
            f"({fmt(get(manifest, 'balance', 'clean_share'), 3)} от всех).",
            f"Разбиения: {manifest.get('splits', NO_DATA)}; общих групп: {manifest.get('shared_groups', NO_DATA)}.",
            f"Проверено дословности контекстов: {manifest.get('contexts_checked', NO_DATA)}, "
            f"проблем: {len(manifest.get('contexts_problems') or [])}.",
            f"Проверено атрибуции чисел: {manifest.get('number_attribution_checked', NO_DATA)}, "
            f"проблем: {len(manifest.get('number_attribution_problems') or [])}.",
            "",
            "| Тип пары | Пар |",
            "|---|---|",
        ]
        for name, count in sorted((counts or {}).items()):
            lines.append(f"| {name} | {count} |")
        lines.append("")
    if summary:
        review = summary.get("review_queue") or {}
        if review:
            lines += [
                f"Очередь ручной проверки: {review.get('total', NO_DATA)} пар "
                f"(фактически проверено человеком: {review.get('checked', 0)}).",
                "",
            ]
    return lines


def section_5(metrics: dict[str, Any] | None) -> list[str]:
    lines = ["## 5. Метрики", ""]
    if not metrics:
        lines += [missing(ROOT / "reports" / "METRICS.json"), ""]
        return lines
    meta = metrics.get("meta") or {}
    lines += [
        f"Seed: {meta.get('seed', NO_DATA)}; n: {meta.get('pairs', NO_DATA)} пар; "
        f"железо: {meta.get('hardware', NO_DATA)}; ОС: {meta.get('os', NO_DATA)}.",
        f"Команда: `{meta.get('command', NO_DATA)}`.",
        "",
        "| Корпус | Режим | Модель | Вердикт F1 | FPR | token F1 | Строгий span-F1 | width (narrow/expanded) | AUC |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for corpus, payload in sorted((metrics.get("corpora") or {}).items()):
        for mode, values in sorted((payload.get("modes") or {}).items()):
            spans = values.get("spans") or {}
            lines.append(
                f"| {corpus} | {mode} | {values.get('model') or '—'} | {fmt(get(values, 'verdicts', 'f1'))} | "
                f"{fmt(get(values, 'verdicts', 'fpr'))} | {fmt(get(values, 'tokens', 'f1'))} | "
                f"{fmt(spans.get('f1'))} | {fmt(spans.get('mean_width_ratio'), 2)} / "
                f"{fmt(spans.get('mean_width_ratio_expanded'), 2)} | {fmt(get(values, 'tokens', 'auc'))} |"
            )
    lines += ["", "Метрики по типам ошибок (recall):", "", "| Корпус | Режим |", "|---|---|"]
    for corpus, payload in sorted((metrics.get("corpora") or {}).items()):
        for mode, values in sorted((payload.get("modes") or {}).items()):
            by_type = values.get("by_type") or {}
            cells = ", ".join(f"{name}: {fmt(item.get('recall'), 3)}" for name, item in sorted(by_type.items()))
            lines.append(f"| {corpus} | {mode} | {cells or NO_DATA} |")
    lines.append("")
    return lines


def section_6(pilot: dict[str, Any] | None, iterations: dict[str, Any] | None) -> list[str]:
    lines = ["## 6. Пилот", ""]
    if not pilot:
        lines += [missing(ROOT / "reports" / "pilot" / "pilot.json"), ""]
    elif pilot.get("status") == "model_unavailable":
        lines += [
            f"Пилот не выполнен: модель `{pilot.get('model')}` недоступна.",
            f"Причина: `{pilot.get('error')}`.",
            "",
        ]
    else:
        contrast = pilot.get("contrast") or {}
        hypothesis = pilot.get("hypothesis") or {}
        lines += [
            f"Модель: `{pilot.get('model') or (pilot.get('environment') or {}).get('model')}`; "
            f"пар: {pilot.get('pairs', NO_DATA)}; "
            f"бутстрэп {pilot.get('bootstrap_iterations', NO_DATA)} итераций, seed {pilot.get('seed', NO_DATA)}.",
            f"Слои: {', '.join(pilot.get('layers', []))}.",
            "",
            f"Контроль контраста: доля слов с опорой {fmt(contrast.get('grounded_value_copy_rate'), 3)} "
            f"(нужно ≥ 0.80), без опоры {fmt(contrast.get('unsupported_value_copy_rate'), 3)} (нужно ≤ 0.30), "
            f"контраст {'выдержан' if contrast.get('contrast_ok') else 'НЕ выдержан'}.",
            "",
            "| Слой | Признак | AUC | 95 % ДИ |",
            "|---|---|---|---|",
        ]
        for layer, features in sorted((pilot.get("auc") or {}).items()):
            for name, values in sorted(features.items()):
                lines.append(
                    f"| {layer} | {name} | {fmt(values.get('auc_oriented'))} | "
                    f"[{fmt(values.get('ci_low'))}; {fmt(values.get('ci_high'))}] |"
                )
        if hypothesis:
            verdict = "подтверждена" if hypothesis.get("confirmed") else "не подтверждена"
            lines += [
                "",
                f"Гипотеза «масса внимания на контекст информативнее сырой энтропии»: {verdict}. "
                f"AUC массы {fmt(hypothesis.get('mass_auc'))}, энтропии {fmt(hypothesis.get('entropy_auc'))}, "
                f"разность {fmt(hypothesis.get('difference'))} "
                f"[{fmt(hypothesis.get('difference_ci_low'))}; {fmt(hypothesis.get('difference_ci_high'))}], "
                f"n = {hypothesis.get('n', NO_DATA)}.",
                "",
            ]
    if not iterations:
        lines += [missing(ROOT / "reports" / "experiments" / "features_hf.json"), ""]
    else:
        lines += ["Итерации доработки признаков:", "", "| Итерация | AUC | 95 % ДИ | Признаки |", "|---|---|---|---|"]
        for item in iterations.get("iterations", []):
            label = item.get("title") or item.get("iteration") or item.get("name")
            added = item.get("features") or item.get("added_features") or []
            lines.append(
                f"| {label} | {fmt(item.get('auc'))} | "
                f"[{fmt(item.get('ci_low'))}; {fmt(item.get('ci_high'))}] | "
                f"{', '.join(added) if added else 'базовые признаки (12)'} |"
            )
        if iterations.get("note"):
            lines += ["", iterations["note"]]
        lines.append("")
    return lines


def section_7(robustness: dict[str, Any] | None) -> list[str]:
    lines = ["## 7. Устойчивость к искажениям", ""]
    if not robustness:
        lines += [missing(ROOT / "reports" / "robustness_hf.md"), ""]
        return lines
    lines += [
        f"Искажений: {len(robustness.get('distortions', []))}; пар на искажение: "
        f"{get(robustness, 'distortions', default=[{}])[0].get('pairs', NO_DATA) if robustness.get('distortions') else NO_DATA}; "
        f"режим {robustness.get('mode', NO_DATA)}.",
        "",
        "| Искажение | Изменено ответов | Не-«grounded» до | после | Изменение |",
        "|---|---|---|---|---|",
    ]
    for item in robustness.get("distortions", []):
        lines.append(
            f"| {item.get('name')} | {item.get('changed_answers')} | {fmt(item.get('flagged_share_before'), 3)} | "
            f"{fmt(item.get('flagged_share_after'), 3)} | {fmt(item.get('delta'), 3)} |"
        )
    lines.append("")
    return lines


def section_8(functional: dict[str, Any] | None, code: dict[str, Any] | None) -> list[str]:
    lines = ["## 8. Ресурсы и производительность", ""]
    if not functional:
        lines += [missing(ROOT / "reports" / "functional_tests.json"), ""]
        return lines
    env = functional.get("environment") or {}
    lines += [
        f"Среда: {env.get('os', NO_DATA)}, Python {env.get('python', NO_DATA)}, CPU {env.get('cpu', NO_DATA)} "
        f"({env.get('cpu_count', NO_DATA)} ядер), RAM {env.get('ram_total_mb', NO_DATA)} МБ.",
        "",
        "| Длина, символов | Секунд | Вердикт | Фрагментов | Пик памяти, МБ |",
        "|---|---|---|---|---|",
    ]
    for row in functional.get("processing", []):
        lines.append(
            f"| {row.get('chars')} | {row.get('seconds')} | {row.get('verdict')} | {row.get('spans')} | "
            f"{row.get('peak_memory_mb')} |"
        )
    smoke = functional.get("smoke") or {}
    leak = functional.get("leak") or {}
    lines += [
        "",
        f"Максимальная обработанная длина: {get(functional, 'max_length', 'max_ok_chars', default=NO_DATA)} символов; "
        f"поведение при превышении: {json.dumps(get(functional, 'max_length', 'behavior_on_excess'), ensure_ascii=False)}.",
        f"Холодный запуск до `GET /health`: {smoke.get('cold_start_seconds', NO_DATA)} с; "
        f"поля контракта: {smoke.get('fields_present', NO_DATA)}.",
        f"Цикл {leak.get('duration_seconds', NO_DATA)} с, итераций {leak.get('iterations', NO_DATA)}, "
        f"рост памяти {leak.get('growth_mb_per_minute', NO_DATA)} МБ/мин (фактическая длительность вместо 8 часов).",
    ]
    sizes = functional.get("sizes") or {}
    if sizes:
        lines += ["", "| Артефакт | Байт | Причина отсутствия |", "|---|---|---|"]
        for name, payload in sizes.items():
            lines.append(
                f"| {payload.get('path', name)} | {payload.get('bytes', 'null')} | {payload.get('reason') or '—'} |"
            )
    if code and code.get("builds"):
        lines.append("")
    lines.append("")
    return lines


def section_9(external: dict[str, Any] | None, prior: dict[str, Any] | None) -> list[str]:
    lines = ["## 9. Внешние наборы и аналоги", ""]
    if external:
        lines += [
            f"Внешние наборы: {', '.join(sorted((external.get('datasets') or {}).keys())) or NO_DATA}.",
            "",
            "| Набор, задача | Пар | token F1 | FPR | AUC | Их метрика | Режим |",
            "|---|---|---|---|---|---|---|",
        ]
        rows = external.get("results") or external.get("rows") or []
        for row in rows:
            lines.append(
                f"| {row.get('dataset')} / {row.get('task')} | {row.get('pairs')} | {fmt(row.get('token_f1'))} | "
                f"{fmt(row.get('fpr'))} | {fmt(row.get('auc'))} | {fmt(row.get('their_metric'))} | {row.get('mode')} |"
            )
        lines.append("")
    else:
        lines += [missing(ROOT / "reports" / "external_tests.json"), ""]
    if prior:
        topics = prior.get("topics") or {}
        found = sum(
            len(entry.get(source, {}).get("found") or [])
            for entry in topics.values()
            for source in ("openalex", "crossref", "github", "patents")
        )
        lines += [
            f"Обзор аналогов и патентный поиск: тем {len(topics)}, найдено записей {found} "
            f"(дата запроса {prior.get('generated_at', NO_DATA)}; подробности — `reports/prior_art.md`).",
            "",
        ]
    else:
        lines += [missing(ROOT / "reports" / "prior_art.json"), ""]
    return lines


def section_10(targets: dict[str, Any] | None) -> list[str]:
    lines = ["## 10. Цель / факт", ""]
    if not targets:
        lines += [missing(ROOT / "reports" / "targets.json"), ""]
        return lines
    lines += ["| Показатель | Цель | Факт | Достигнуто | Комментарий |", "|---|---|---|---|---|"]
    for item in targets.get("items", []):
        lines.append(
            f"| {item.get('name')} | {item.get('target')} | {item.get('fact')} | "
            f"{'да' if item.get('achieved') else 'нет'} | {item.get('comment', '')} |"
        )
    lines.append("")
    return lines


def section_11(audit: dict[str, Any] | None, functional: dict[str, Any] | None) -> list[str]:
    lines = ["## 11. Что не измерено (честный список)", ""]
    items: list[str] = []
    if audit:
        items += [f"{item}" for item in audit.get("unmeasured", [])]
    if functional:
        for name, payload in (functional.get("sizes") or {}).items():
            if payload.get("bytes") is None:
                items.append(f"{payload.get('path', name)}: {payload.get('reason')}")
        if not (functional.get("smoke") or {}).get("cold_start_seconds"):
            items.append("холодный запуск сервиса не измерялся (смоук-тест пропущен)")
    if not items:
        lines += ["Все заявленные показатели измерены; список пуст.", ""]
    else:
        lines += [f"* {item}" for item in items] + [""]
    return lines


def section_12() -> list[str]:
    return [
        "## 12. Команды воспроизведения",
        "",
        "```bash",
        "# проверки и метрики",
        "python -m pytest -q",
        "python scripts/collect_metrics.py",
        "python scripts/check_numbers.py",
        "",
        "# реальные акты и корпус (нужна сеть; учитываются robots.txt и пауза больше секунды)",
        "python scripts/fetch_npa_corpus.py --target-docs 130 --regions 23:18,77:8,50:6,78:6,61:6,26:6,16:6,66:6",
        "python scripts/fetch_npa_corpus.py --verify",
        "python scripts/build_corpus_a.py --real --docs data/corpus_a3/sources --out data/corpus_a3 --target 1200 --seed 42",
        "python scripts/make_corpus_a3_report.py",
        "",
        "# режим hf (реальная модель; скачивание весов с HuggingFace)",
        "python scripts/pilot_rugpt3small.py --model ai-forever/rugpt3small_based_on_gpt2 --pairs 48 --bootstrap 5000",
        "python scripts/feature_iterations.py --mode hf --model ai-forever/rugpt3small_based_on_gpt2 "
        "--dataset data/corpus_a3/pairs.jsonl --layers first,middle,-4,last --limit 240 --bootstrap 5000",
        "python scripts/run_experiments.py --mode hf --model ai-forever/rugpt3small_based_on_gpt2 "
        "--dataset data/corpus_a/splits/train.jsonl --folds 5",
        "python scripts/robustness.py --mode hf --dataset data/corpus_a3/pairs.jsonl --limit 60",
        "python scripts/calibrate_participation_real.py --mode hf --sources data/corpus_a3/sources "
        "--pairs data/corpus_a3/pairs.jsonl",
        "",
        "# функциональные испытания и отчётные документы",
        "python scripts/functional_tests.py --mode demo --lengths 500,2000,10000,50000",
        "python scripts/make_gost_docs.py --out reports/gost",
        "python scripts/make_rid_package.py --out reports/rid",
        "python scripts/prior_art_search.py --out reports/prior_art.json",
        "python scripts/make_final_report.py --out reports/РЕЗУЛЬТАТЫ_ПРОГОНА.md",
        "```",
        "",
        "Полный конвейер выполняется в GitHub Actions: `.github/workflows/arena-pipeline.yml`.",
        "",
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Итоговый отчёт о прогоне (12 разделов)")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "РЕЗУЛЬТАТЫ_ПРОГОНА.md")
    parser.add_argument("--reports", type=Path, default=ROOT / "reports")
    args = parser.parse_args(argv)

    reports = args.reports
    audit = read_json(reports / "audit_fixes.json")
    code = read_json(reports / "code_state.json")
    sources = read_json(ROOT / "data" / "corpus_a3" / "sources" / "sources.json")
    manifest = read_json(ROOT / "data" / "corpus_a3" / "manifest.json")
    summary = read_json(reports / "review_queue.json")
    metrics = read_json(reports / "METRICS.json")
    pilot = read_json(reports / "pilot" / "pilot.json")
    if not pilot:
        # Локальный пилот на доступной реальной модели (MiniLM-L6) — тот же формат,
        # получается scripts/pilot_rugpt3small.py --model <путь> --out reports/pilot_local.
        pilot = read_json(reports / "pilot_local" / "pilot.json")
    iterations = read_json(reports / "experiments" / "features_hf.json")
    robustness_data = read_json(reports / "robustness_hf.json")
    functional = read_json(reports / "functional_tests.json")
    external = read_json(reports / "external_tests.json")
    prior = read_json(reports / "prior_art.json")
    targets = read_json(reports / "targets.json")

    lines = [
        "# Результаты прогона SpanVerify",
        "",
        f"Отчёт сформирован {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} скриптом "
        "`scripts/make_final_report.py` из файлов результатов. Числа не вписываются руками: "
        "если источника нет, в таблице стоит `null` или «нет данных» с указанием файла.",
        "",
    ]
    lines += section_1(audit)
    lines += section_2(code)
    lines += section_3(sources)
    lines += section_4(manifest, summary)
    lines += section_5(metrics)
    lines += section_6(pilot, iterations)
    lines += section_7(robustness_data)
    lines += section_8(functional, code)
    lines += section_9(external, prior)
    lines += section_10(targets)
    lines += section_11(audit, functional)
    lines += section_12()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"отчёт: {args.out} ({len(lines)} строк)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
