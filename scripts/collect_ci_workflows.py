#!/usr/bin/env python3
"""Задания CI в один файл: `reports/CI_WORKFLOWS.json` и документ `docs/ЗАДАНИЯ_CI.md`.

Зачем это нужно. Заданий в репозитории много (основной CI, сборка .exe, замеры
объёмов, пилот на реальной модели и исторические одноразовые прогоны), и по
файлам не видно, какие из них обязаны быть зелёными, какие запускаются вручную,
а какие не запускаются вовсе. Правило проекта — «не обещать, а показывать»:
поэтому документ собирается из двух источников фактов:

* `.github/workflows/*.yml` — что задание умеет: триггеры, задания, лимиты
  времени, файлы-триггеры, шаги загрузки артефактов;
* `reports/CI_RUNS.json` — что задание делало в последний раз (номер прогона и
  итог; `None` означает «прогон ещё идёт» и пишется как «идёт»).

Скрипт ничего не придумывает: если workflow не запускался ни разу, в документе
стоит «прогонов нет», а не «успешно». Запуск::

    python scripts/collect_ci_workflows.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
OUT_JSON = ROOT / "reports" / "CI_WORKFLOWS.json"
OUT_DOC = ROOT / "docs" / "ЗАДАНИЯ_CI.md"
RUNS = ROOT / "reports" / "CI_RUNS.json"

#: Расписание: что задание делает само, а что только по просьбе человека.
SCHEDULE_WORDS = {
    "schedule": "по расписанию",
    "workflow_dispatch": "вручную",
    "push": "на пуш",
    "pull_request": "на pull request",
    "repository_dispatch": "по внешнему событию",
}


def _triggers(data: dict) -> list[str]:
    on = data.get(True, data.get("on")) or {}
    if isinstance(on, list):
        return [str(item) for item in on]
    return [str(key) for key in on]


def _workflow(name: str, path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    on = data.get(True, data.get("on")) or {}
    triggers = _triggers(data)
    push = on.get("push") if isinstance(on, dict) else None
    push_paths: list[str] = []
    push_branches: list[str] = []
    push_tags: list[str] = []
    if isinstance(push, dict):
        push_paths = [str(item) for item in push.get("paths") or []]
        push_branches = [str(item) for item in push.get("branches") or []]
        push_tags = [str(item) for item in push.get("tags") or []]
    schedule = []
    if isinstance(on, dict) and on.get("schedule"):
        schedule = [str(item.get("cron", "")) for item in on["schedule"]]

    jobs: list[dict] = []
    for job_name, job in (data.get("jobs") or {}).items():
        steps = job.get("steps") or []
        jobs.append(
            {
                "name": str(job.get("name") or job_name),
                "timeout_minutes": job.get("timeout-minutes"),
                "steps": len(steps),
                "uploads_artifacts": any(
                    str(step.get("uses") or "").startswith("actions/upload-artifact") for step in steps
                ),
                "runs_python": str(job.get("runs-on") or ""),
            }
        )

    return {
        "file": path.relative_to(ROOT).as_posix(),
        "name": str(data.get("name") or name),
        "triggers": triggers,
        "schedule": schedule,
        "push_paths": push_paths,
        "push_branches": push_branches,
        "push_tags": push_tags,
        "jobs": jobs,
        "manual_only": triggers == ["workflow_dispatch"] or set(triggers) == {"push", "workflow_dispatch"},
        "never_runs_by_itself": "schedule" not in triggers and "push" not in triggers and "pull_request" not in triggers,
    }


def _last_runs() -> dict[str, dict]:
    """Последний прогон каждого задания по данным reports/CI_RUNS.json."""
    if not RUNS.is_file():
        return {}
    payload = json.loads(RUNS.read_text(encoding="utf-8"))
    last: dict[str, dict] = {}
    for run in payload.get("runs") or []:
        name = str(run.get("name") or "")
        created = str(run.get("created_at") or "")
        current = last.get(name)
        if current is None or created > str(current.get("created_at") or ""):
            last[name] = run
    return last


def _status(run: dict | None) -> str:
    if run is None:
        return "прогонов нет"
    verdict = run.get("conclusion")
    if verdict in (None, "null", ""):
        return f"идёт (прогон {run['id']})"
    if verdict == "success":
        return f"успех (прогон {run['id']})"
    if verdict == "skipped":
        return f"пропущен (прогон {run['id']})"
    return f"{verdict} (прогон {run['id']})"


def render_doc(payload: dict, last_runs: dict[str, dict]) -> str:
    lines = [
        "# Задания CI: что запускается само, а что вручную",
        "",
        "Документ собран скриптом `scripts/collect_ci_workflows.py` из файлов",
        "`.github/workflows/*.yml` (что задание умеет) и `reports/CI_RUNS.json`",
        "(что оно делало в последний раз). Числа и номера прогонов не вписаны",
        "руками: их источник — реестр прогонов.",
        "",
        f"Команда: `python scripts/collect_ci_workflows.py` (запуск: {payload['generated_at']}).",
        "",
        "| Задание | Запуск | Последний прогон | Артефакты | Числа в реестре |",
        "| --- | --- | --- | --- | --- |",
    ]
    registry = payload["registry"]
    for item in payload["workflows"]:
        how = ", ".join(SCHEDULE_WORDS.get(trigger, trigger) for trigger in item["triggers"])
        if item["schedule"]:
            how += " (" + ", ".join(item["schedule"]) + " UTC)"
        if item["push_paths"]:
            how += "; пуш " + ", ".join(f"`{path}`" for path in item["push_paths"])
        if item["push_tags"]:
            how += "; теги " + ", ".join(f"`{tag}`" for tag in item["push_tags"])
        if item["push_branches"]:
            how += "; ветки " + ", ".join(f"`{branch}`" for branch in item["push_branches"])
        artifacts = "да" if any(job["uploads_artifacts"] for job in item["jobs"]) else "нет"
        run = last_runs.get(item["name"])
        if item["name"] in registry:
            numbers = registry[item["name"]]
        elif run is not None and run.get("conclusion") == "success":
            # Задание отработало, но числа в реестр не переносятся: так и написано,
            # а не спрятано за прочерком — иначе «запись в реестре» выглядела бы
            # выполненной для всех заданий одинаково.
            numbers = "в артефакте прогона (в реестр не переносится)"
        else:
            numbers = "—"
        lines.append(
            f"| `{item['name']}` | {how} | {_status(last_runs.get(item['name']))} | "
            f"{artifacts} | {numbers} |"
        )

    lines += [
        "",
        f"Источник колонки «Последний прогон» — `reports/CI_RUNS.json`: последние "
        f"{payload['runs_considered']} прогонов репозитория. Поэтому «прогонов нет» "
        "означает «нет в этом списке», а не «задание никогда не запускалось»; "
        "пересобрать список — `python scripts/collect_ci_runs.py --limit "
        f"{payload['runs_considered']}`.",
        "",
        "## Что это значит для приёмки",
        "",
        "Обязательными считаются задания, которые запускаются на пуш в `main` и в",
        "зону разработки: основной CI (тесты, линтер, контрактное задание, сверка",
        "чисел) и сборка `.exe`. Замер объёмов режима `hf` запускается пушем",
        "файла-триггера: он отвечает требованиям ТЗ п. 4.2 треб. 5 и 6, а его числа",
        "попадают в реестр `reports/NUMBERS.json` тем же прогоном.",
        "",
        "Задания, которые не запускаются сами (их триггер — только правка",
        "файла-триггера или ручной запуск), помечены в колонке «Запуск»: это не",
        "забытые задания, а измерения, которые делают один раз по событию. Ни одно",
        "из них не выдаёт числа «на глаз»: результат каждого — артефакт с числом и",
        "записью в реестре, либо честное «прогонов нет».",
        "",
        "Ручной запуск из командной строки в этой работе недоступен (интеграция",
        "GitHub не имеет права `Actions: write`, ответ API — 403), поэтому",
        "одноразовые задания запускаются пушем файла-триггера вида",
        "`.github/<имя>.trigger`; это записано в комментариях к каждому workflow.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Собрать документ о заданиях CI")
    parser.add_argument("--out-json", default=str(OUT_JSON.relative_to(ROOT)))
    parser.add_argument("--out-doc", default=str(OUT_DOC.relative_to(ROOT)))
    args = parser.parse_args()

    paths = sorted(WORKFLOWS.glob("*.yml"))
    if not paths:
        print(f"в {WORKFLOWS} нет workflow — собирать нечего")
        return 1
    workflows = [_workflow(path.stem, path) for path in paths]
    runs = _last_runs()
    payload = {
        "kind": "ci_workflows",
        "runs_considered": len(json.loads(RUNS.read_text(encoding="utf-8")).get("runs") or [])
        if RUNS.is_file()
        else 0,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": "python scripts/collect_ci_workflows.py",
        "source": ".github/workflows/*.yml + reports/CI_RUNS.json",
        "registry": {
            "ci": "reports/METRICS.json, reports/contract/*.json",
            "build-and-release-exe": "reports/release_sizes.json (spanverify.exe, spanverify.pyz)",
            "hf-footprint": "reports/contract/4.2.5-hf-footprint.json, reports/contract/4.2.6-storage.json",
            "hf-pilot": "spanverify-module/reports/pilot/pilot.json",
            "corpus-a3": "reports/ext_*.json, spanverify-module/data/corpus_a3/manifest.json",
        },
        "workflows": workflows,
    }
    out_json = ROOT / args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    doc = render_doc(payload, runs)
    out_doc = ROOT / args.out_doc
    out_doc.parent.mkdir(parents=True, exist_ok=True)
    out_doc.write_text(doc, encoding="utf-8")
    print(f"записано: {out_json}")
    print(f"записано: {out_doc} ({len(workflows)} заданий)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
