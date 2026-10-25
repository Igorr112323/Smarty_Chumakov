"""P8 provenance regression: no metric or historical number quietly changes roles."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("p8_hygiene", ROOT / "scripts/p8_evidence_hygiene.py")
hygiene = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hygiene)


def test_all_non_run_metric_documents_have_provenance():
    found = 0
    for folder in (ROOT / "reports", ROOT / "spanverify-module/reports"):
        for path in folder.rglob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            if hygiene.is_run(path) or not hygiene.metric_file(path, data):
                continue
            found += 1
            assert hygiene.provenance_problems(path, data) == [], path
    assert found >= 20
    for name in ("reports/FINAL_REPORT.md", "reports/p8/RELEASE_NOTES.md"):
        # Catch SHA/year concatenation without altering immutable raw snapshots.
        assert not re.search(
            r"(?<![0-9a-f])[0-9a-f]{41,63}(?![0-9a-f])",
            (ROOT / name).read_text(encoding="utf-8"),
        ), name
    path = Path("arbitrary-new-name.json")
    assert hygiene.metric_file(path, {"auc": 0.75})
    assert hygiene.metric_file(Path("model_metrics.json"), {"auc": 0.75})
    assert hygiene.provenance_problems(path, {"auc": 0.75})
    broken = {
        "f1": 0.5,
        "run_id": "local-probe",
        "date": "2026-10-19",
        "_status": "SUPERSEDED",
    }
    assert any("successor" in p for p in hygiene.provenance_problems(path, broken))
    broken["successor"] = "reports/EXPERIMENTS_INDEX.md"
    assert hygiene.provenance_problems(path, broken) == []
    broken["date"] = "not-a-date"
    assert any("date" in p for p in hygiene.provenance_problems(path, broken))


def test_legacy_numbers_immutable_and_superseded():
    fixture = json.loads((ROOT / "reports/p8/legacy-payload-hashes.json").read_text(encoding="utf-8"))
    for name, record in fixture["files"].items():
        data = json.loads((ROOT / name).read_text(encoding="utf-8"))
        assert next(iter(data)) == "_status"
        assert data["_status"].startswith("SUPERSEDED")
        assert data["successor"] == "reports/EXPERIMENTS_INDEX.md"
        old = {key: data[key] for key in record["original_keys"]}
        digest = hashlib.sha256(json.dumps(old, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        assert digest == record["semantic_sha256"], name


def test_index_covers_latest_run_in_every_family():
    inventory = json.loads((ROOT / "reports/p8/measurement-inventory.json").read_text(encoding="utf-8"))
    text = (ROOT / "reports/EXPERIMENTS_INDEX.md").read_text(encoding="utf-8")
    assert set(inventory["latest_by_family"]) == {f"P{i}" for i in range(9)}
    for family, run in inventory["latest_by_family"].items():
        records = [r for r in inventory["runs"] if r["family"] == family and r["record_kind"] == "measurement/delivery"]
        assert run == max(records, key=lambda r: r["date"])["run_id"]
        assert run in text
        assert all((ROOT / r["receipt"]).exists() and r["commit"] and r["date"] for r in records)


def test_frozen_prediction_files_byte_identical():
    fixture = json.loads((ROOT / "reports/p8/frozen-production-sha256.json").read_text(encoding="utf-8"))
    for name, expected in fixture["files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name


def test_ci_collector_only_copies_sealed_a3_receipt():
    script = ROOT / "spanverify-module/scripts/collect_metrics.py"
    module_spec = importlib.util.spec_from_file_location("p8_collect_metrics", script)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    class ForbiddenEvaluation:
        def evaluate(self, *args, **kwargs):
            raise AssertionError("official177 must never be evaluated by ordinary CI")

    import yaml

    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    assert workflow["jobs"]["contract"]["if"] == "always() && needs.p8-freeze-barrier.result == 'success'"
    block = module._corpus_a3_block(ForbiddenEvaluation())
    assert "SUPERSEDED" in block["scope"]
    assert block["pairs"] == 1200
