"""Collect read-only GitHub workflow receipts; never evaluate/fill missing evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

REPO = "Igorr112323/Smarty_Chumakov"
ROOT = Path(__file__).resolve().parent.parent


def gh(*args):
    return json.loads(subprocess.check_output(["gh", *map(str, args)], text=True))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def collect(run_id, target, require_success):
    run = gh(
        "run",
        "view",
        run_id,
        "--json",
        "headSha,url,status,conclusion,jobs,name,createdAt,updatedAt",
    )
    if require_success:
        assert run["status"] == "completed" and run["conclusion"] == "success", run
    save(target / "run.json", run)
    artifacts = gh("api", f"repos/{REPO}/actions/runs/{run_id}/artifacts?per_page=100")
    save(target / "artifacts.json", artifacts)
    summary = {"run": run, "artifacts": artifacts, "notices": []}
    for job in run["jobs"]:
        annotations = gh(
            "api",
            f"repos/{REPO}/check-runs/{job['databaseId']}/annotations?per_page=100",
        )
        save(target / f"job-{job['databaseId']}-annotations.json", annotations)
        for a in annotations:
            title, message = a.get("title", ""), a.get("message", "")
            if not title.startswith("P7 ") and title not in (
                "P6 parity",
                "размеры поставки",
            ):
                continue
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                if title == "P6 parity" and ', "environment":' in message:
                    data = json.loads(message.split(', "environment":', 1)[0] + "}")
                    data["environment_omitted_from_notice"] = True
                else:
                    data = message
            summary["notices"].append(
                {
                    "job": job["name"],
                    "job_id": job["databaseId"],
                    "title": title,
                    "data": data,
                }
            )
    save(target / "summary.json", summary)
    return summary


def derived(summary, out):
    tables, profiles, calibrators = [], [], {}
    for note in summary["notices"]:
        data, title = note["data"], note["title"]
        if title == "P7 benchmark table":
            rows = [
                dict(
                    zip(data["columns"], r, strict=True),
                    mode=data["mode"],
                    phase=data.get("phase", "after"),
                )
                for r in data["rows"]
            ]
            assert len(rows) == 12
            tables += rows
        elif title == "P7 benchmark profiles":
            profiles.append(data)
        elif title == "P7 calibrator artifact":
            artifact = data["artifact"]
            encoded = (
                json.dumps(artifact, ensure_ascii=False, indent=2) + "\n"
            ).encode()
            assert hashlib.sha256(encoded).hexdigest() == data["sha256"]
            save(out / f"ai_share_calibrator_v1.{data['mode']}.json", artifact)
            calibrators[data["mode"]] = data
    if tables:
        save(out / "native-benchmark.json", tables)
    if profiles:
        save(out / "native-profiles.json", profiles)
    if calibrators:
        save(out / "calibrator-provenance.json", calibrators)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--acceptance", type=int, required=True)
    p.add_argument("--windows", type=int, required=True)
    p.add_argument("--ci", type=int, required=True)
    p.add_argument("--paired", type=int)
    p.add_argument("--out", type=Path, default=ROOT / "reports/p7/final-receipts")
    p.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Diagnostic snapshot only; never acceptance.",
    )
    a = p.parse_args()
    for role, run_id in (
        ("acceptance", a.acceptance),
        ("windows", a.windows),
        ("ci", a.ci),
    ):
        summary = collect(run_id, a.out / role, not a.allow_incomplete)
        if role == "acceptance":
            derived(summary, a.out)
    if a.paired:
        paired = collect(a.paired, a.out / "paired", not a.allow_incomplete)
        derived(paired, a.out / "paired")
    save(
        a.out / "collection.json",
        {
            "require_success": not a.allow_incomplete,
            "acceptance": a.acceptance,
            "windows": a.windows,
            "ci": a.ci,
            "tests_excluded": False,
        },
    )


if __name__ == "__main__":
    main()
