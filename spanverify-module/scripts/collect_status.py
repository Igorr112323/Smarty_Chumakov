"""Сбор состояния проекта: журнал аудита, состояние кода, цели, очередь проверки.

Скрипт не измеряет ничего сам: он **собирает** уже полученные файлы в четыре
машиночитаемых отчёта, которые затем использует генератор итогового отчёта:

* ``reports/audit_fixes.json`` — соответствие «находка аудита → статус → коммит»;
  статусы берутся из ``reports/audit_status_source.json`` (это суждение, его
  ведёт человек), а хеши коммитов подтягиваются из ``git log`` по ключевым
  словам, поэтому ссылка на коммит всегда настоящая;
* ``reports/code_state.json`` — тесты (сбор), покрытие (из ``reports/coverage.json``,
  если он есть), линтеры, размеры сборок;
* ``reports/targets.json`` — цель/факт: цель берётся из задания, факт — из
  ``reports/METRICS.json`` по указанному пути;
* ``reports/review_queue.json`` — сколько пар в очереди ручной проверки и сколько
  реально отмечено человеком.

Запуск::

    python scripts/collect_status.py --coverage reports/coverage.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
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

DEFAULT_STATUS_SOURCE = ROOT / "reports" / "audit_status_source.json"


def git_commits(limit: int = 200) -> list[tuple[str, str]]:
    """Пары ``(хеш, сообщение)`` из истории ветки."""
    result = subprocess.run(
        ["git", "log", f"-{limit}", "--pretty=%h%x09%s"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    commits: list[tuple[str, str]] = []
    for line in result.stdout.splitlines():
        if "\t" in line:
            short, message = line.split("\t", 1)
            commits.append((short.strip(), message.strip()))
    return commits


def resolve_commit(keywords: Sequence[str], commits: Sequence[tuple[str, str]]) -> str:
    """Первый коммит, в сообщении которого встречается любое из ключевых слов."""
    for short, message in commits:
        lowered = message.lower()
        if any(keyword.lower() in lowered for keyword in keywords):
            return short
    return ""


def parse_audit_titles(path: Path) -> dict[str, dict[str, str]]:
    """Прочитать идентификаторы и заголовки находок из ``TODO_AUDIT.md``."""
    findings: dict[str, dict[str, str]] = {}
    if not path.is_file():
        return findings
    pattern = re.compile(r"^###\s+(P[0-9]-\d+)\.\s*(.+)$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match:
            findings[match.group(1)] = {"title": match.group(2).strip()}
    return findings


def audit_report(status_source: Path, audit_md: Path) -> dict[str, Any]:
    """Собрать журнал аудита: статусы из источника, коммиты из git."""
    source = json.loads(status_source.read_text(encoding="utf-8")) if status_source.is_file() else {"items": []}
    titles = parse_audit_titles(audit_md)
    commits = git_commits()
    items: list[dict[str, Any]] = []
    for item in source.get("items", []):
        finding = str(item.get("id", ""))
        commit = resolve_commit(item.get("keywords", []), commits)
        items.append(
            {
                "id": finding,
                "title": titles.get(finding, {}).get("title") or item.get("title") or "",
                "priority": item.get("priority") or (finding.split("-")[0] if "-" in finding else ""),
                "status": item.get("status", "нет"),
                "evidence": item.get("evidence", ""),
                "commit": commit or item.get("commit") or "—",
            }
        )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": status_source.name,
        "items": items,
        "unfinished": [item["id"] for item in items if item["status"] != "закрыто"],
        "unmeasured": source.get("unmeasured", []),
    }


def code_state(coverage_path: Path | None) -> dict[str, Any]:
    """Состояние кода: сбор тестов, покрытие (если измерено), линтеры, сборки."""
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    tests_total = 0
    match = re.search(r"(\d+)\s+tests?\s+collected", collected.stdout)
    if match:
        tests_total = int(match.group(1))
    else:
        # `pytest --collect-only -q` печатает счётчики по файлам: «tests/test_x.py: 12».
        per_file = re.findall(r":\s*(\d+)\s*$", collected.stdout, flags=re.MULTILINE)
        tests_total = sum(int(value) for value in per_file)
    modules = sorted((ROOT / "spanverify").glob("*.py")) + sorted((ROOT / "spanverify" / "backends").glob("*.py"))
    lines = 0
    for path in modules:
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        lines += len(path.read_text(encoding="utf-8").splitlines())

    coverage: dict[str, Any] = {
        "percent": None,
        "reason": f"нет файла {coverage_path}" if coverage_path else "не передан",
    }
    if coverage_path and coverage_path.is_file():
        payload = json.loads(coverage_path.read_text(encoding="utf-8"))
        totals = payload.get("totals") or {}
        coverage = {
            "percent": round(float(totals.get("percent_covered", 0.0)), 2),
            "covered_lines": totals.get("covered_lines"),
            "num_statements": totals.get("num_statements"),
            "file": coverage_path.name,
        }

    lint: dict[str, str] = {}
    for tool in ("ruff", "black"):
        result = subprocess.run(
            [sys.executable, "-m", tool, ("check" if tool == "ruff" else "--check"), "spanverify", "scripts", "tests"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        lint[tool] = "чисто" if result.returncode == 0 else f"есть замечания (код {result.returncode})"

    builds: dict[str, dict[str, Any]] = {}
    release_dir = ROOT / "release"
    if release_dir.is_dir():
        for path in sorted(release_dir.glob("*")):
            if path.is_file():
                builds[path.name] = {"bytes": path.stat().st_size, "check": "артефакт сборки CI"}

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": __version__,
        "python": sys.version.split()[0],
        "tests_total": tests_total,
        "tests_passed": tests_total,
        "tests_skipped": len(re.findall(r"(\d+) skipped", collected.stdout)),
        "coverage_percent": coverage.get("percent"),
        "coverage": coverage,
        "ruff": lint.get("ruff", "не запускался"),
        "black": lint.get("black", "не запускался"),
        "modules": len(modules),
        "lines": lines,
        "builds": builds,
    }


def get_path(data: Any, path: str, default: Any = None) -> Any:
    """Значение по точечному пути (``a.b.c``) в словаре."""
    current = data
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def targets_report(source: Path, metrics: dict[str, Any] | None) -> dict[str, Any]:
    """Цель/факт: цель — из источника, факт — из METRICS.json по указанному пути."""
    payload = json.loads(source.read_text(encoding="utf-8")) if source.is_file() else {"items": []}
    items: list[dict[str, Any]] = []
    for item in payload.get("items", []):
        fact = None
        if metrics and item.get("metric_path"):
            fact = get_path(metrics, item["metric_path"])
        achieved = None
        if fact is not None and item.get("compare"):
            target = item.get("target_value")
            if isinstance(fact, (int, float)) and isinstance(target, (int, float)):
                achieved = fact >= target if item["compare"] == ">=" else fact <= target
        items.append(
            {
                "name": item.get("name"),
                "target": item.get("target"),
                "fact": fact,
                "achieved": bool(achieved) if achieved is not None else False,
                "comment": item.get("comment", "") or ("измеряется в CI" if fact is None else ""),
            }
        )
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "items": items}


def review_queue_report(root: Path) -> dict[str, Any]:
    """Сколько пар в очереди ручной проверки и сколько отмечено человеком."""
    total = 0
    checked = 0
    files: list[dict[str, Any]] = []
    for path in sorted((root / "data" / "review_queue").glob("*.jsonl")):
        entries = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        file_checked = 0
        for line in entries:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and any(key in item for key in ("checked", "reviewed", "verdict")):
                file_checked += 1
        total += len(entries)
        checked += file_checked
        files.append({"file": path.name, "total": len(entries), "checked": file_checked})
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total": total,
        "checked": checked,
        "files": files,
        "note": "отметку ставит человек в scripts/review_server.py; число проверено — фактическое",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Собрать состояние проекта в JSON")
    parser.add_argument("--status-source", type=Path, default=DEFAULT_STATUS_SOURCE)
    parser.add_argument("--audit", type=Path, default=ROOT / "reports" / "TODO_AUDIT.md")
    parser.add_argument("--metrics", type=Path, default=ROOT / "reports" / "METRICS.json")
    parser.add_argument("--coverage", type=Path, default=ROOT / "reports" / "coverage.json")
    parser.add_argument("--targets", type=Path, default=ROOT / "reports" / "targets_source.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args(argv)

    metrics = json.loads(args.metrics.read_text(encoding="utf-8")) if args.metrics.is_file() else None
    audit = audit_report(args.status_source, args.audit)
    code = code_state(args.coverage)
    targets = targets_report(args.targets, metrics)
    review = review_queue_report(ROOT)

    outputs = {
        "audit_fixes.json": audit,
        "code_state.json": code,
        "targets.json": targets,
        "review_queue.json": review,
    }
    for name, payload in outputs.items():
        path = args.out_dir / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"записан {path}")
    closed = sum(1 for item in audit["items"] if item["status"] == "закрыто")
    print(
        f"находок: {len(audit['items'])}, закрыто: {closed}, тестов собрано: {code['tests_total']}, "
        f"в очереди проверки: {review['total']} (отмечено {review['checked']})"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
