"""P8 provenance regression: no metric or historical number quietly changes roles."""

from __future__ import annotations

import hashlib
import importlib.util
import json
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
            for item in hygiene.objects(data):
                context = item.get("evidence_context", item)
                assert context.get("run_id") and context.get("date"), path
                if "unrecorded" in str(context["run_id"]):
                    assert context.get("limitation"), path
                if "SUPERSEDED" in str(item.get("_status", context.get("status", ""))):
                    assert item.get("successor", context.get("successor")), path
    assert found >= 20


def test_legacy_numbers_immutable_and_superseded():
    fixture = json.loads((ROOT / "reports/p8/legacy-payload-hashes.json").read_text())
    for name, record in fixture["files"].items():
        data = json.loads((ROOT / name).read_text())
        assert next(iter(data)) == "_status"
        assert data["_status"].startswith("SUPERSEDED")
        assert data["successor"] == "reports/EXPERIMENTS_INDEX.md"
        old = {key: data[key] for key in record["original_keys"]}
        digest = hashlib.sha256(json.dumps(old, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        assert digest == record["semantic_sha256"], name


def test_index_covers_latest_run_in_every_family():
    inventory = json.loads((ROOT / "reports/p8/measurement-inventory.json").read_text())
    text = (ROOT / "reports/EXPERIMENTS_INDEX.md").read_text()
    assert set(inventory["latest_by_family"]) == {f"P{i}" for i in range(9)}
    for family, run in inventory["latest_by_family"].items():
        records = [r for r in inventory["runs"] if r["family"] == family and r["record_kind"] == "measurement/delivery"]
        assert run == max(records, key=lambda r: r["date"])["run_id"]
        assert run in text
        assert all((ROOT / r["receipt"]).exists() and r["commit"] and r["date"] for r in records)


def test_frozen_prediction_files_byte_identical():
    fixture = json.loads((ROOT / "reports/p8/frozen-production-sha256.json").read_text())
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

    block = module._corpus_a3_block(ForbiddenEvaluation())
    assert "SUPERSEDED" in block["scope"]
    assert block["pairs"] == 1200
