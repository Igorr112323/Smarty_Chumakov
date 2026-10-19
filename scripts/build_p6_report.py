"""Assemble P6 acceptance from actual receipts; missing evidence never becomes success."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/acceptance"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    rows = []
    benchmarks = []
    for mode in ("demo", "hf"):
        candidates = sorted(
            OUT.glob(f"run*/p6-product-{mode}/benchmark.json"), reverse=True
        )
        selected = None
        for p in candidates:
            data = load(p)
            if len(data) == 12 and all(
                s.get("profile_enabled") is False
                for r in data
                for s in r["repeats"]
                if s["status"] == "completed"
            ):
                selected = p
                break
        if selected is None:
            rows.append(
                {
                    "id": f"speed-{mode}",
                    "status": "не измерено",
                    "reason": "complete native benchmark receipt absent",
                }
            )
            continue
        folder = selected.parent
        run = load(folder.parent / "receipt.json")
        for r in load(selected):
            benchmarks.append(
                {
                    "mode": mode,
                    "scenario": r["scenario"],
                    "chars": r["chars"],
                    "thermal": r["thermal"],
                    "median_seconds": r["median_seconds"],
                    "max_seconds": r["max_seconds"],
                    "status": r["status"],
                    "receipt": str(selected.relative_to(ROOT)),
                    "run_url": run["url"],
                }
            )
        rows.append(
            {
                "id": f"speed-{mode}",
                "status": (
                    "подтверждено"
                    if all(r["status"] == "достигнут" for r in load(selected))
                    else "не достигнут"
                ),
                "receipt": str(selected.relative_to(ROOT)),
                "run_url": run["url"],
                "source_sha": run["code_sha"],
            }
        )
        for name, file in [
            ("interfaces", "interfaces/interfaces.json"),
            ("ai-share", "ai-share.json"),
        ]:
            p = folder / file
            if not p.exists():
                rows.append(
                    {
                        "id": f"{name}-{mode}",
                        "status": "не измерено",
                        "run_url": run["url"],
                    }
                )
            else:
                fact = load(p)
                rows.append(
                    {
                        "id": f"{name}-{mode}",
                        "status": (
                            "подтверждено"
                            if name == "interfaces" and fact.get("passed")
                            else "измерено, не доказательство калибровки"
                        ),
                        "receipt": str(p.relative_to(ROOT)),
                        "run_url": run["url"],
                        "fact": fact,
                    }
                )
    parities = sorted(OUT.glob("run*/p6-parity/parity.json"), reverse=True)
    if parities:
        p = parities[0]
        fact = load(p)
        rows.append(
            {
                "id": "frozen-public-parity",
                "status": "подтверждено" if fact["passed"] else "не достигнут",
                "fact": fact,
                "receipt": str(p.relative_to(ROOT)),
                "run_url": load(p.parent.parent / "receipt.json")["url"],
            }
        )
    suite = OUT / "full-suite.xml"
    if suite.exists():
        doc = ET.parse(suite).getroot()
        suites = list(doc.iter("testsuite"))
        totals = {
            k: sum(int(s.get(k, 0)) for s in suites)
            for k in ("tests", "failures", "errors", "skipped")
        }
        rows.append(
            {
                "id": "local-suite",
                "status": (
                    "подтверждено"
                    if totals["failures"] == totals["errors"] == 0
                    else "не достигнут"
                ),
                "fact": totals,
                "tests_excluded": False,
                "receipt": str(suite.relative_to(ROOT)),
            }
        )
    for name in ("ordinary-ci", "windows-rc"):
        p = OUT / (name + ".json")
        if p.exists():
            fact = load(p)
            accepted = fact.get("conclusion") == "success"
            if name == "windows-rc":
                release_path = OUT / "rc-release.json"
                release = load(release_path) if release_path.exists() else {}
                sizes = {a["name"]: a["size"] for a in release.get("assets", [])}
                fact["measured_size_gate"] = {
                    "exe_bytes": sizes.get("spanverify.exe"),
                    "pyz_bytes": sizes.get("spanverify.pyz"),
                    "exe_ok": sizes.get("spanverify.exe", 10**12) <= 9000000,
                    "pyz_ok": sizes.get("spanverify.pyz", 10**12) <= 600000,
                }
                accepted = (
                    accepted
                    and release.get("isPrerelease", False)
                    and release.get("targetCommitish") == fact.get("headSha")
                    and fact["measured_size_gate"]["exe_ok"]
                    and fact["measured_size_gate"]["pyz_ok"]
                )
            rows.append(
                {
                    "id": name,
                    "status": ("подтверждено" if accepted else "не достигнут"),
                    "fact": fact,
                    "run_url": fact.get("url"),
                }
            )
        else:
            rows.append({"id": name, "status": "не измерено"})
    functional = OUT / "functional-2M-demo.json"
    if functional.exists():
        fact = load(functional)
        rows.append(
            {
                "id": "full-2M-demo-answer",
                "status": "подтверждено функционально; скорость не достигнута",
                "fact": {
                    k: v for k, v in fact.items() if k not in ("environment", "stats")
                },
                "receipt": str(functional.relative_to(ROOT)),
            }
        )
    rows.append(
        {
            "id": "native-HF-long-answer-without-truncation",
            "status": "не достигнут",
            "receipt": "reports/acceptance/long-answer-limitation.json",
        }
    )
    rows.extend(
        [
            {
                "id": "robustness",
                "status": "измерено на saved OOF/dev, не новая независимая проверка",
                "receipt": "reports/acceptance/robustness.json",
            },
            {
                "id": "official-test177",
                "status": "не запускался",
                "incident": "Initial old chunking test parsed unsplit corpus; see protocol incident. No model evaluation or selection on test177.",
            },
            {
                "id": "whole-contract",
                "status": "не принят",
                "reason": "Dev success and interface smoke are not full contract acceptance; speed failures, censored full-text requests and external Windows HF runtime need explicit accounting.",
            },
        ]
    )
    result = {
        "stage": "P6",
        "tests_excluded": False,
        "token_convention": "P5 kept spans + mandatory rules; historical baseline counts unchanged",
        "requirements": rows,
        "benchmarks": benchmarks,
    }
    (OUT / "coverage-matrix.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    text = [
        "# P6: requirement → measured evidence",
        "",
        "Не заменяет историческую матрицу договора. Нет доказательства — нет повышения статуса.",
        "",
        "| Проверка | Статус | Прямой источник |",
        "|---|---|---|",
    ]
    for r in rows:
        source = r.get("run_url", r.get("receipt", "protocol.json"))
        text.append(f"| {r['id']} | {r['status']} | {source} |")
    text += [
        "",
        "## Native product timing (three repeats)",
        "",
        "| mode | role | chars | state | median s | max s | status |",
        "|---|---|---:|---|---:|---:|---|",
    ]
    for r in benchmarks:
        text.append(
            f"| {r['mode']} | {r['scenario']} | {r['chars']} | {r['thermal']} | {r['median_seconds']} | {r['max_seconds']} | {r['status']} |"
        )
    (OUT / "coverage-matrix.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    inventory = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            inventory.append(
                {
                    "path": str(p.relative_to(ROOT)),
                    "bytes": p.stat().st_size,
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                }
            )
    (OUT / "manifest.json").write_text(
        json.dumps({"files": inventory}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
