#!/usr/bin/env python3
"""Documentary evidence map, not a substitute for official acceptance."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = "39170981235"
P5 = f"spanverify-module/reports/experiments/p5_span_filter/run{RUN}"
URL = "https://github.com/Igorr112323/Smarty_Chumakov"


def main():
    results = ROOT / P5 / "all_family_metrics.json"
    measured = json.loads(results.read_text()) if results.exists() else None
    rows = []

    def row(id_, requirement, status, note, *files):
        rows.append(
            {
                "id": id_,
                "requirement": requirement,
                "status": status,
                "scope": "current P5 candidate; historical evidence is explicitly qualified",
                "note": note,
                "evidence": [{"path": f, "url": f"{URL}/blob/arena/c5f69767-smarty-chumakov/{f}"} for f in files],
            }
        )

    row(
        "input",
        "UTF-8 input up to 2,000,000 characters",
        "частично",
        "Historical synthetic demo processed 2M chars; no current P5 HF 2M acceptance measurement.",
        "reports/contract/4.1.1-text-limit.json",
    )
    row(
        "risk",
        "Output risk in [0,1]",
        "частично",
        "Real public dev evidence and finite-risk checks; no all-input/packaged API acceptance.",
        P5 + "/dev_evidence.jsonl.gz",
        P5 + "/parity.json",
    )
    row(
        "spans",
        "Fragments with coordinates, label and confidence",
        "частично",
        "Real dev spans and exact text slices; existing API exposes confidence. Current P5 packaged API is not separately measured.",
        P5 + "/dev_evidence.jsonl.gz",
        "reports/contract/4.1.2-api-schema.json",
    )
    row(
        "features",
        "Five features per token",
        "достигнут" if measured else "не достигнут",
        "Frozen P4 retains the original five HF columns and 56 derived token inputs. P5 adds only 19 span-level derivatives; neither token risks nor HF measurements are retrained.",
        "spanverify-module/reports/experiments/p4_context_head/run39164044941/feature_specs.json",
        P5 + "/span_feature_spec.json",
        P5 + "/manifest.json",
    )
    row(
        "participation",
        "AI participation [0,1], calibrated at step 0.25",
        "частично",
        "Historical 240 synthetic calibration pairs at 0/.25/.5/.75/1, MAE .0352; not proof of P5 real-text acceptance.",
        "reports/contract/4.1.4-ai-participation.json",
    )
    row(
        "metric_groups",
        "Token/answer/span metrics, strict IoU >= 0.5",
        "достигнут" if measured else "не достигнут",
        "Real frozen public dev, all families; span FPR remains declared token proxy, not invented TN.",
        P5 + "/all_family_metrics.json",
        P5 + "/baseline_parity.json",
    )
    passed = (
        measured
        and all(measured["p5"][f]["f1"] >= 0.50 for f in ("tokens", "spans", "answers"))
        and measured["p5"]["tokens"]["fpr"] <= 0.40
        and measured["p5"]["answers"]["fpr"] <= 0.40
    )
    row(
        "quality_gate",
        "F1 >= .50 jointly with FPR <= .40 on document-held-out part of 1200 pairs",
        "частично" if passed else "не достигнут",
        "A successful dev gate would still be partial contractual evidence: official test177 remains CLOSED. No status based on FPR alone.",
        P5 + "/all_family_metrics.json",
        P5 + "/oof_validation.json",
    )
    for family in ("tokens", "spans"):
        f = measured["p5"][family] if measured else {}
        fpr = measured["p5"]["tokens"]["fpr"] if measured else None
        achieved = bool(f and f["f1"] >= 0.5 and fpr <= 0.4)
        row(
            "dev_" + family + "_gate",
            "P5 independent public dev " + family + " quality gate",
            "достигнут" if achieved else "не достигнут",
            f"Dev only: F1={f.get('f1')}, FPR={fpr}; strict spans use existing token proxy. This is not official test or whole-contract acceptance.",
            P5 + "/all_family_metrics.json",
            P5 + "/parity.json",
        )
    row(
        "official_test",
        "Official test177 acceptance",
        "не достигнут",
        "Official test177 remains CLOSED; no metric or result is asserted for it.",
        P5 + "/manifest.json",
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
            "Only historical demo timing available; training/replay timings are not product speed benchmarks. No P5 HF acceptance artifact.",
            "reports/contract/4.1.7-performance.json",
            P5 + "/manifest.json",
        )
    for id_, text in [("cli", "CLI interface"), ("web", "Web interface on port8765"), ("http", "HTTP API")]:
        row(
            id_,
            text,
            "частично",
            "Historical demo health/index/verify HTTP200 at port8765; CLI exit1 in old smoke. Current P5 interfaces not separately accepted.",
            "reports/contract/4.1.8-interfaces.json",
        )
    row(
        "exe",
        "Executable <=9 MB",
        "частично",
        "Historical v1.3.1 exe8,969,008B; not a P5 build and not full HF runtime footprint.",
        "reports/contract/4.1.9-delivery-artifacts.json",
        "reports/contract/4.2.5-hf-footprint.json",
    )
    row(
        "package",
        "Package <=0.6 MB",
        "частично",
        "Historical pyz174,634B; no rebuilt P5 delivery/size measurement.",
        "reports/contract/4.2.1-delivery-size.json",
    )
    row(
        "build",
        "Automatic source build with SHA-256",
        "частично",
        "Historical release workflow38582542536 and published SHA256SUMS; release_checksum_verified=null; no P5 executable checksum acceptance.",
        "reports/contract/4.2.4-build-sha256.json",
    )
    for id_, requirement, artifact, note in [
        (
            "no_install",
            "4.2.3 Launch without installation",
            "4.2.3-no-install",
            "Historical copied demo returned exit1, although no writes outside copy were recorded. P5 no-install acceptance is not established.",
        ),
        (
            "hf_footprint",
            "4.2.5 HF runtime and separately delivered model footprint",
            "4.2.5-hf-footprint",
            "Historical measured HF runtime libraries1,053,454,603B are NOT the tiny exe/pyz size. P5 does not rebuild or certify that delivery.",
        ),
        (
            "storage",
            "4.2.6 Storage requirements",
            "4.2.6-storage",
            "Historical HF footprint/storage inventory exists; no current P5 installed storage acceptance.",
        ),
        (
            "portable_copy",
            "4.2.7 Portable copying and integrity",
            "4.2.7-portable-copy",
            "Historical copied pyz hashes matched, but published release verification was null; no P5 packaged-copy acceptance.",
        ),
        (
            "runtime_config",
            "4.2.8 Reference machine / RAM / CPU / offline runtime",
            "4.2.8-runtime-config",
            "Historical demo peak261MB on3.85GB host; specified reference machine not tested. P5 cache-only CPU replay is not a memory/offline-delivery benchmark.",
        ),
        (
            "updates",
            "4.2.9 Whole-package replacement / update procedure",
            "4.2.9-update-procedure",
            "Historical operating guide contains replacement/previous-version procedure; no measured P5 packaged upgrade.",
        ),
        (
            "qualification",
            "4.2.10 Operation without specialist preparation",
            "4.2.10-qualification",
            "Historical guide sections checked; no P5 end-user usability acceptance artifact.",
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
            "не достигнут",
            "Pooled A3 dev metrics are not dedicated robustness acceptance: no separately measured passing P5 challenge/control artifact. Paraphrase is not an explicit A3 mode.",
            "spanverify-module/data/corpus_a3/manifest.json",
            P5 + "/dev_error_decomposition.json",
        )
    row(
        "control_sample",
        "Control sample / corpus1200 / held-out document split",
        "частично",
        "Frozen A3 manifest1200, train842/dev181. Original folds unchanged; official177 not evaluated. Legacy unsplit read disclosed in reports/p7/freeze-and-incident.json; no new P7 reads.",
        "spanverify-module/data/corpus_a3/manifest.json",
        P5 + "/manifest.json",
        P5 + "/oof_validation.json",
    )
    assert len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        missing = [e["path"] for e in r["evidence"] if not (ROOT / e["path"]).is_file()]
        if missing:
            r["status"] = "не достигнут"
            r["note"] += " Missing evidence: " + ", ".join(missing)
        r["current_run"] = f"{URL}/actions/runs/{RUN}" if any(e["path"].startswith(P5) for e in r["evidence"]) else None
    for r in rows:
        if r["id"] in {"metric_groups", "quality_gate", "dev_tokens_gate", "dev_spans_gate"}:
            r["scope"] = "historical frozen P5 legacy512; not re-evaluated P7 long-v1 quality"
            r["note"] += " Historical P5 metric, not a new P7 long-path quality evaluation."
    payload = {
        "as_of": "2026-10-19",
        "source": "User-specified Appendix3 requirements; evidence only, no invented acceptance",
        "current_measurement_run": f"{URL}/actions/runs/{RUN}",
        "official_test_read": True,
        "official_test_evaluated": False,
        "official_test_read_scope": "legacy P6 chunking-test unsplit read, not new P7 evaluation",
        "incident_evidence": "reports/p7/freeze-and-incident.json",
        "overall": "не достигнут",
        "no_artifact_rule": "No current proof => not achieved; historical/demo proof cannot establish P5 speed/delivery/official quality",
        "requirements": rows,
    }
    # A versioned evidence overlay updates ONLY explicitly measured requirements.
    # Historical P5 quality rows/denominators are retained, never recomputed here.
    overlay_path = ROOT / "reports/p7/contract-overlay.json"
    if overlay_path.exists():
        overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
        known = {r["id"]: r for r in rows}
        assert len({r["id"] for r in overlay["requirements"]}) == len(overlay["requirements"])
        for update in overlay["requirements"]:
            assert update["id"] in known
            assert update["evidence"] and update["note"] and update["current_run"]
            assert all((ROOT / e["path"]).is_file() for e in update["evidence"])
            known[update["id"]].update(update)
        payload["current_measurement_run"] = overlay["current_measurement_run"]
        payload["p7_evidence_overlay"] = str(overlay_path.relative_to(ROOT))
    p8 = ROOT / "reports/p8/run39266451529/official/result.json"
    if p8.exists():
        official = json.loads(p8.read_text())
        evidence_path = str(p8.relative_to(ROOT))
        run_url = f"{URL}/actions/runs/39266451529"
        known = {r["id"]: r for r in rows}
        for id_, note in {
            "quality_gate": "P8 official177 joint PASS: tokensF1 .6252045826513911, strict spansF1 .5164835164835164, token-FPR proxy .015466785356626237; internal .90/.10 NOT achieved.",
            "official_test": "177/177 once through public Verifier; 7 over512/max1266, full long-v1; empty cache0hits177misses,381.292202784s, no fits. Zero-output infrastructure abort preserved, not a second quality chance.",
            "control_sample": "A3 1200 pairs: immutable train842/dev181/test177 document split; official177 now evaluated. Historical unsplit fixture-read incident and train-only guard retained; no claim test was never exposed.",
            "spans": "Official strict kept-spans+rules:47TP38FP50FN, IoU>=.5 unchanged, containment .2886597938, mean contained width1.33926249.",
        }.items():
            known[id_].update(status="достигнут" if official["joint_gate"] else "не достигнут",scope="P8 official held-out measurement; frozen P7 product",note=note,evidence=[dict(path=evidence_path,url=f"{URL}/blob/arena/c5f69767-smarty-chumakov/{evidence_path}"),dict(path="reports/FINAL_REPORT.md",url=f"{URL}/blob/arena/c5f69767-smarty-chumakov/reports/FINAL_REPORT.md")],current_run=run_url)
        payload.update(run_id="39266451529",date="2026-10-19",current_measurement_run=run_url,official_test_evaluated=True,official_test_read_scope="P8 authorized final evaluation177; prior legacy read and zero-output infrastructure abort explicitly retained",tests_excluded=False,p8_official_metrics=official["metrics"],p8_joint_gate=official["joint_gate"],current_indices=["reports/EXPERIMENTS_INDEX.md","reports/contract/coverage-matrix.json"])
        external = ROOT / "reports/p8/run39266451529/external/result.json"
        if external.exists():
            payload["external_validation"] = dict(pairs=1000,contract_gate=False,metrics=json.loads(external.read_text())["metrics"],evidence=str(external.relative_to(ROOT)),limitation="423 human exact spans/333positive answers; one first-match ambiguity; raw license unconfirmed; severe transfer drop published")
        delivery = ROOT / "reports/p8/DELIVERY.json"
        if delivery.exists():
            payload["delivery"] = json.loads(delivery.read_text())
            delivery_info = payload["delivery"]
            for id_ in ("exe", "package", "build", "no_install", "portable_copy"):
                known[id_].update(scope="P8 actual Windows build artifacts; publication-only source failure preserved",note=f"P8 build39271014161/source654a48e: exe8981233B, pyz232874B, strict gates PASS; three complete suites and smoke/offline validations PASS. Original workflow publication-only FAIL; recovery reuses exact checksummed artifacts, no rebuild/scoring. {delivery_info.get('publication_recovery', '')}",evidence=[dict(path="reports/p8/DELIVERY.json",url=f"{URL}/blob/arena/c5f69767-smarty-chumakov/reports/p8/DELIVERY.json"),dict(path="reports/p8/run39271014161/windows/release/SHA256SUMS.txt",url=f"{URL}/blob/arena/c5f69767-smarty-chumakov/reports/p8/run39271014161/windows/release/SHA256SUMS.txt")],current_run=f"{URL}/actions/runs/39271014161")
            for id_ in ("hf_footprint", "storage"):
                known[id_].update(status="частично",scope="P8 measured directory/file footprint, not complete installed SDK",note="Actual weight file551290714B; full model/tokenizer directory554277308B; libs870588148B; directory+libs1424865456B PLUS Python/app/filesystem overhead. Complete installed SDK total unmeasured; compressed portable637048763B is not installed storage.",evidence=[dict(path="reports/p8/run39271014161/hf-autonomy/windows-hf-autonomy.json",url=f"{URL}/blob/arena/c5f69767-smarty-chumakov/reports/p8/run39271014161/hf-autonomy/windows-hf-autonomy.json")],current_run=f"{URL}/actions/runs/39271014161")
        payload["evidence_context"] = dict(run_id="39266451529",date="2026-10-19",status="CURRENT authoritative requirements index",successor="reports/EXPERIMENTS_INDEX.md")
    out = ROOT / "reports/contract"
    (out / "coverage-matrix.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    text = "# Appendix3: evidence coverage matrix\n\nAs of 2026-10-19. **Overall: НЕ ДОСТИГНУТ.** Official test177 evaluated once in P8; joint quality gate PASS, overall contract remains NOT achieved; legacy read and infrastructure incident disclosed.\n\nHistorical/demo artifacts are not current P5 HF acceptance. A dev gate cannot establish speed, delivery or the whole contract.\n\n| Requirement | Status | Scope / limitation | Real evidence |\n|---|---|---|---|\n"
    for r in rows:
        links = "; ".join(f"[{e['path'].split('/')[-1]}]({e['url']})" for e in r["evidence"])
        text += f"| {r['requirement']} | {r['status']} | {r['note']} | {links} |\n"
    (out / "coverage-matrix.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
