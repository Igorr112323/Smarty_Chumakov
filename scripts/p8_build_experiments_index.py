"""Inventory measurement workflows from GitHub metadata; no data/model evaluation."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "Igorr112323/Smarty_Chumakov"


def api(path):
    return json.loads(
        subprocess.check_output(["gh", "api", f"repos/{REPO}/{path}"], text=True)
    )


def main():
    rows = []
    paths = [
        p.name
        for p in (ROOT / ".github/workflows").glob("*.yml")
        if p.name not in ("ci.yml", "p8-receipts.yml")
    ]
    for workflow in paths:
        page = 1
        while True:
            batch = api(f"actions/workflows/{workflow}/runs?per_page=100&page={page}")[
                "workflow_runs"
            ]
            for r in batch:
                if workflow.startswith("p") and workflow[1:2].isdigit():
                    family = "P" + workflow[1]
                elif "postprocess" in workflow:
                    family = "P0"
                elif "recover-p1" in workflow:
                    family = "P1"
                elif workflow == "build-and-release-exe.yml":
                    family = (
                        "P7"
                        if r["id"] >= 39236657579
                        else ("P6" if r["id"] >= 39184437524 else "P0")
                    )
                else:
                    family = "P0"
                canonical = sorted(
                    {
                        str(p.relative_to(ROOT))
                        for p in (ROOT / "spanverify-module/reports/experiments").rglob(
                            "*"
                        )
                        if p.is_dir() and p.name == f"run{r['id']}"
                    }
                )
                if workflow in (
                    "build-and-release-exe.yml",
                    "p7-publish-acceptance.yml",
                ) and r.get("head_commit", {}).get("message", "").startswith("P8:"):
                    family = "P8"
                if (ROOT / f"reports/p8/run{r['id']}").is_dir():
                    canonical.append(f"reports/p8/run{r['id']}")
                receipt = f"reports/measurements/run{r['id']}/receipt.json"
                record = dict(
                    run_id=str(r["id"]),
                    date=r["created_at"],
                    commit=r["head_sha"],
                    family=family,
                    workflow=workflow,
                    record_kind=(
                        "publication/recovery"
                        if any(
                            x in workflow
                            for x in ("publish", "collect", "recover", "salvage")
                        )
                        or workflow == "hf-runs.yml"
                        else "measurement/delivery"
                    ),
                    status=r["status"],
                    outcome=r["conclusion"],
                    url=r["html_url"],
                    canonical_directories=canonical,
                    receipt=receipt,
                )
                dest = ROOT / receipt
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
                rows.append(record)
            if len(batch) < 100:
                break
            page += 1
    incident = api("actions/runs/39266027509")
    record = dict(
        run_id=str(incident["id"]),
        date=incident["created_at"],
        commit=incident["head_sha"],
        family="P8",
        workflow="ci.yml",
        record_kind="process incident: incidental synthetic diagnostic",
        status=incident["status"],
        outcome=incident["conclusion"],
        url=incident["html_url"],
        canonical_directories=["reports/p8/infra-freeze-barrier"],
        receipt="reports/p8/infra-freeze-barrier/run.json",
    )
    rows.append(record)
    rows.sort(key=lambda r: (r["date"], r["run_id"]))
    latest = {
        f: max(
            (
                r
                for r in rows
                if r["family"] == f and r["record_kind"] == "measurement/delivery"
            ),
            key=lambda r: r["date"],
        )["run_id"]
        for f in sorted({r["family"] for r in rows})
    }
    dest = ROOT / "reports/p8/measurement-inventory.json"
    dest.write_text(
        json.dumps(
            dict(
                run_id="P8-index-audit",
                date="2026-10-19",
                latest_by_family=latest,
                runs=rows,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    text = (
        "# EXPERIMENTS INDEX — единственный текущий индекс измерений\n\nВторой и единственный текущий индекс требований: [contract/coverage-matrix.json](contract/coverage-matrix.json). Все прежние сводки — исторические/неавторитетные. Метаданные всех запусков измерительных/delivery workflow получены из API; failure/cancelled не скрыты. Receipt — канонический каталог provenance; native каталоги содержат собственно измерения. Отсутствие native evidence не превращает успешный workflow в доказанную метрику.\n\nПоследний run каждой семьи: "
        + ", ".join(f"{f}: **{r}**" for f, r in latest.items())
        + ".\n\n| UTC дата | Семья | Run | Commit | Результат | Канонические evidence/receipt |\n|---|---|---|---|---|---|\n"
    )
    for r in rows:
        links = (
            "; ".join(f"`{d}`" for d in r["canonical_directories"])
            + f"; [{r['receipt']}](../{r['receipt']})"
        )
        text += f"| {r['date']} | {r['family']} | [{r['run_id']}]({r['url']}) | `{r['commit']}` | {r['outcome'] or r['status']} | {links} |\n"
    text += "\n## P8 final evidence\n\nOfficial and B: `reports/p8/run39266451529/`, original per-pair streams in immutable workflow artifacts (SHA256/row IDs in manifests); final release archive preserves them. Official successful measurement source dd4454b, exact product b9cf9ac. First attempt39266027598 produced ZERO predictions; traceback retained in `reports/p8/infra-first-attempt`. Historical local summaries with no recorded remote run explicitly say `historical-local-unrecorded`, never invented scientific provenance. See [FINAL_REPORT](FINAL_REPORT.md).\n"
    text += "\n## Critical P8 execution-freeze breach\n\nCI39266027509 was cancelled, but always-contract ran a temporary synthetic ParticipationModel.fit before seal (artifact12192328736 created14:12:03Z). [Immutable original run/artifact metadata](p8/infra-freeze-barrier/). No official/B fit data or product change; nevertheless ANY-fit prohibition was violated. Inference-worker fit_calls=[] is process-scoped. Guard now requires successful barrier; no official retry. Numeric joint gate PASS, process freeze NOT achieved, conservative RC NOTlatest. Ordinary CI synthetic diagnostics are regressions, not candidate comparisons; this exceptional incident is explicitly indexed chronologically above.\n"
    (ROOT / "reports/EXPERIMENTS_INDEX.md").write_text(text)


if __name__ == "__main__":
    main()
