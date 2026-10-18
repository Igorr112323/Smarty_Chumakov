#!/usr/bin/env python3
"""Documentary evidence map, not a substitute for official acceptance."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = "39164044941"
P4 = f"spanverify-module/reports/experiments/p4_context_head/run{RUN}"
URL = "https://github.com/Igorr112323/Smarty_Chumakov"


def main():
    results = ROOT / P4 / "all_family_metrics.json"
    measured = json.loads(results.read_text()) if results.exists() else None
    rows = []

    def row(id_, requirement, status, note, *files):
        rows.append(
            {
                "id": id_,
                "requirement": requirement,
                "status": status,
                "scope": "current P4 candidate; historical evidence is explicitly qualified",
                "note": note,
                "evidence": [{"path": f, "url": f"{URL}/blob/arena/c5f69767-smarty-chumakov/{f}"} for f in files],
            }
        )

    row(
        "input",
        "UTF-8 input up to 2,000,000 characters",
        "частично",
        "Historical synthetic demo processed 2M chars; no current P4 HF 2M acceptance measurement.",
        "reports/contract/4.1.1-text-limit.json",
    )
    row(
        "risk",
        "Output risk in [0,1]",
        "частично",
        "Real public dev evidence and finite-risk checks; no all-input/packaged API acceptance.",
        P4 + "/dev_evidence.jsonl.gz",
        P4 + "/runtime_parity.json",
    )
    row(
        "spans",
        "Fragments with coordinates, label and confidence",
        "частично",
        "Real dev spans and exact text slices; existing API exposes confidence. Current P4 packaged API is not separately measured.",
        P4 + "/dev_evidence.jsonl.gz",
        "reports/contract/4.1.2-api-schema.json",
    )
    row(
        "features",
        "Five features per token",
        "достигнут" if measured else "не достигнут",
        "Original five HF columns remain unchanged; P4 56 inputs are deterministic aggregates of the same five plus unchanged position/length/digit, not new HF measurements.",
        P4 + "/feature_specs.json",
        P4 + "/manifest.json",
    )
    row(
        "participation",
        "AI participation [0,1], calibrated at step 0.25",
        "частично",
        "Historical 240 synthetic calibration pairs at 0/.25/.5/.75/1, MAE .0352; not proof of P4 real-text acceptance.",
        "reports/contract/4.1.4-ai-participation.json",
    )
    row(
        "metric_groups",
        "Token/answer/span metrics, strict IoU >= 0.5",
        "достигнут" if measured else "не достигнут",
        "Real frozen public dev, all families; span FPR remains declared token proxy, not invented TN.",
        P4 + "/all_family_metrics.json",
        P4 + "/historical_evidence_parity.json",
    )
    passed = (
        measured
        and all(measured["p4"][f]["f1"] >= 0.50 for f in ("tokens", "spans", "answers"))
        and measured["p4"]["tokens"]["fpr"] <= 0.40
        and measured["p4"]["answers"]["fpr"] <= 0.40
    )
    row(
        "quality_gate",
        "F1 >= .50 jointly with FPR <= .40 on document-held-out part of 1200 pairs",
        "частично" if passed else "не достигнут",
        "A successful dev gate would still be partial contractual evidence: official test177 remains CLOSED. No status based on FPR alone.",
        P4 + "/all_family_metrics.json",
        P4 + "/oof_validation.json",
    )
    for id_, text in [
        ("speed_per_1000", "<= .03 seconds per 1000 characters"),
        ("speed_50k", "50,000 characters <= 1 second"),
        ("speed_2m", "2,000,000 characters <= 60 seconds"),
    ]:
        row(
            id_,
            text,
            "не достигнут",
            "Only historical demo timing available; training/replay timings are not product speed benchmarks. No P4 HF acceptance artifact.",
            "reports/contract/4.1.7-performance.json",
            P4 + "/manifest.json",
        )
    for id_, text in [("cli", "CLI interface"), ("web", "Web interface on port8765"), ("http", "HTTP API")]:
        row(
            id_,
            text,
            "частично",
            "Historical demo health/index/verify HTTP200 at port8765; CLI exit1 in old smoke. Current P4 interfaces not separately accepted.",
            "reports/contract/4.1.8-interfaces.json",
        )
    row(
        "exe",
        "Executable <=9 MB",
        "частично",
        "Historical v1.3.1 exe8,969,008B; not a P4 build and not full HF runtime footprint.",
        "reports/contract/4.1.9-delivery-artifacts.json",
        "reports/contract/4.2.5-hf-footprint.json",
    )
    row(
        "package",
        "Package <=0.6 MB",
        "частично",
        "Historical pyz174,634B; no rebuilt P4 delivery/size measurement.",
        "reports/contract/4.2.1-delivery-size.json",
    )
    row(
        "build",
        "Automatic source build with SHA-256",
        "частично",
        "Historical release workflow38582542536 and published SHA256SUMS; release_checksum_verified=null; no P4 executable checksum acceptance.",
        "reports/contract/4.2.4-build-sha256.json",
    )
    for id_, requirement, artifact, note in [
        (
            "no_install",
            "4.2.3 Launch without installation",
            "4.2.3-no-install",
            "Historical copied demo returned exit1, although no writes outside copy were recorded. P4 no-install acceptance is not established.",
        ),
        (
            "hf_footprint",
            "4.2.5 HF runtime and separately delivered model footprint",
            "4.2.5-hf-footprint",
            "Historical measured HF runtime libraries1,053,454,603B are NOT the tiny exe/pyz size. P4 does not rebuild or certify that delivery.",
        ),
        (
            "storage",
            "4.2.6 Storage requirements",
            "4.2.6-storage",
            "Historical HF footprint/storage inventory exists; no current P4 installed storage acceptance.",
        ),
        (
            "portable_copy",
            "4.2.7 Portable copying and integrity",
            "4.2.7-portable-copy",
            "Historical copied pyz hashes matched, but published release verification was null; no P4 packaged-copy acceptance.",
        ),
        (
            "runtime_config",
            "4.2.8 Reference machine / RAM / CPU / offline runtime",
            "4.2.8-runtime-config",
            "Historical demo peak261MB on3.85GB host; specified reference machine not tested. P4 cache-only CPU replay is not a memory/offline-delivery benchmark.",
        ),
        (
            "updates",
            "4.2.9 Whole-package replacement / update procedure",
            "4.2.9-update-procedure",
            "Historical operating guide contains replacement/previous-version procedure; no measured P4 packaged upgrade.",
        ),
        (
            "qualification",
            "4.2.10 Operation without specialist preparation",
            "4.2.10-qualification",
            "Historical guide sections checked; no P4 end-user usability acceptance artifact.",
        ),
    ]:
        row(id_, requirement, "частично", note, "reports/contract/" + artifact + ".json")
    for id_, text in [
        ("number", "Robustness: numeric substitutions"),
        ("contradiction", "Robustness: contradiction"),
        ("paraphrase", "Robustness: paraphrase"),
        ("omission", "Robustness: omission"),
    ]:
        row(
            id_,
            text,
            "частично",
            "A3 corpus modes and real pooled dev metrics exist, but no separate passing P4 per-case robustness acceptance artifact.",
            "spanverify-module/data/corpus_a3/manifest.json",
            P4 + "/dev_error_decomposition.json",
        )
    row(
        "control_sample",
        "Control sample / corpus1200 / held-out document split",
        "частично",
        "Frozen A3 manifest1200, train842/dev181. Original fold membership validated; test177 not read, no full official acceptance.",
        "spanverify-module/data/corpus_a3/manifest.json",
        P4 + "/source_provenance.json",
        P4 + "/oof_validation.json",
    )
    assert len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        missing = [e["path"] for e in r["evidence"] if not (ROOT / e["path"]).is_file()]
        if missing:
            r["status"] = "не достигнут"
            r["note"] += " Missing evidence: " + ", ".join(missing)
        r["current_run"] = f"{URL}/actions/runs/{RUN}" if any(e["path"].startswith(P4) for e in r["evidence"]) else None
    payload = {
        "as_of": "2026-10-18",
        "source": "User-specified Appendix3 requirements; evidence only, no invented acceptance",
        "current_measurement_run": f"{URL}/actions/runs/{RUN}",
        "official_test_read": False,
        "overall": "не достигнут",
        "no_artifact_rule": "No current proof => not achieved; historical/demo proof cannot establish P4 speed/delivery/official quality",
        "requirements": rows,
    }
    out = ROOT / "reports/contract"
    (out / "coverage-matrix.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    text = "# Appendix3: evidence coverage matrix\n\nAs of 2026-10-18. **Overall: НЕ ДОСТИГНУТ.** Official test177 remains closed.\n\nHistorical/demo artifacts are not current P4 HF acceptance. A dev gate cannot establish speed, delivery or the whole contract.\n\n| Requirement | Status | Scope / limitation | Real evidence |\n|---|---|---|---|\n"
    for r in rows:
        links = "; ".join(f"[{e['path'].split('/')[-1]}]({e['url']})" for e in r["evidence"])
        text += f"| {r['requirement']} | {r['status']} | {r['note']} | {links} |\n"
    (out / "coverage-matrix.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
