"""Номера прогонов CI в один файл — `reports/CI_RUNS.json`.

Правило проекта: числа берутся из прогонов. Номер прогона — тоже факт о проекте, и
он должен быть зафиксирован, а не пересказан по памяти. Скрипт спрашивает у
GitHub API список последних прогонов и их заданий и складывает результат в
`reports/CI_RUNS.json`; оттуда его читает `docs/ЗАКРЫТИЕ_ДОГОВОРА_ПО_КОДУ.md`.

Запуск (нужен `gh` с доступом к репозиторию)::

    python scripts/collect_ci_runs.py --limit 30

Скрипт не выдумывает значения: если `gh` недоступен или отвечает ошибкой, файл не
перезаписывается, а команда завершается кодом 1 с объяснением.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "CI_RUNS.json"
DEFAULT_REPO = "Igorr112323/Smarty_Chumakov"


def gh_json(args: list[str]) -> object:
    proc = subprocess.run(
        ["gh", "api", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"gh api {args} завершился кодом {proc.returncode}")
    return json.loads(proc.stdout)


def jobs_summary(repo: str, run_id: int) -> list[dict]:
    data = gh_json([f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100"])
    jobs = []
    for job in data.get("jobs", []):
        jobs.append({"name": job.get("name"), "conclusion": job.get("conclusion")})
    return jobs


def collect(repo: str, limit: int, branch: str | None) -> dict:
    query = f"repos/{repo}/actions/runs?per_page={limit}"
    data = gh_json([query])
    runs = []
    for run in data.get("workflow_runs", []):
        if branch and run.get("head_branch") != branch and run.get("name") != "build-and-release-exe":
            continue
        runs.append(
            {
                "id": run.get("id"),
                "name": run.get("name"),
                "event": run.get("event"),
                "head_branch": run.get("head_branch"),
                "head_sha": run.get("head_sha"),
                "conclusion": run.get("conclusion"),
                "created_at": run.get("created_at"),
                "url": run.get("html_url"),
                "jobs": jobs_summary(repo, int(run["id"])),
            }
        )
    return {
        "kind": "ci_runs",
        "repo": repo,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": f"python scripts/collect_ci_runs.py --limit {limit}"
        + (f" --branch {branch}" if branch else ""),
        "note": (
            "список прогонов получен из GitHub API (gh api); заключения заданий — "
            "как их отдаёт API на момент сбора"
        ),
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Собрать номера прогонов CI")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--branch", default=None, help="оставить только прогоны этой ветки")
    parser.add_argument("--out", default=str(OUT.relative_to(ROOT)))
    args = parser.parse_args()
    try:
        payload = collect(args.repo, args.limit, args.branch)
    except (RuntimeError, json.JSONDecodeError) as error:
        print(f"прогоны не собраны: {error}", file=sys.stderr)
        return 1
    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{path}: прогонов {len(payload['runs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
