"""REALGEN-R1 freeze assertions that do not open sealed evaluation inputs.

Ordinary CI must never read test_realgen answers, test labels/spans, or their
answer-context rows. Their committed hashes remain metadata in the freeze
manifest; byte-level verification happens only inside the authorized run.
The permanently locked real_test split is never opened by these tests.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parent
REAL = REPO_ROOT / "reports" / "science" / "real"
SEALED_CONTENT_FILES = {
    "reports/science/real/REALGEN_ANSWERS.jsonl",
    "reports/science/real/REALGEN_ANSWERS_TEST_REALGEN.jsonl",
    "reports/science/real/REALGEN_LABELS.jsonl",
    "reports/science/real/REALGEN_SPANS.jsonl",
    "reports/science/real/REALGEN_DOCS.jsonl",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(name: str) -> dict:
    return json.loads((REAL / name).read_text(encoding="utf-8"))


def test_frozen_quotas_and_sealed_split_are_preserved() -> None:
    """400+ answers, ≥40% erroneous, 300 dev, test_realgen sealed."""
    phase_a = _load("REALGEN_PHASE_A.json")
    frozen = _load("REALGEN_FROZEN.json")
    assert phase_a["quotas"]["n_answers"] >= 400
    assert phase_a["quotas"]["erroneous_share"] >= 0.40
    assert phase_a["quotas"]["rules_changed"] is False
    assert frozen["n_answers"] == phase_a["quotas"]["n_answers"]
    assert frozen["splits"]["dev"]["n_answers"] == 300
    assert frozen["splits"]["test_realgen"]["sealed"] is True
    assert frozen["real_test_viewed"] is False
    assert set(frozen["rules"]) == {"a", "b", "c", "d"}
    assert frozen["rule_c_mode"] in ("any_window", "best_window")


def test_dev_inputs_match_the_freeze_without_reading_sealed_files() -> None:
    """Verify only dev/code bytes; sealed input hashes are metadata-only here."""
    frozen = _load("REALGEN_FROZEN.json")
    dev_path = REAL / "REALGEN_ANSWERS_DEV.jsonl"
    assert _sha256(dev_path) == frozen["files"][str(dev_path.relative_to(REPO_ROOT))]["sha256"]

    assert SEALED_CONTENT_FILES <= set(frozen["files"])
    assert (
        frozen["splits"]["test_realgen"]["sha256"]
        == frozen["files"]["reports/science/real/REALGEN_ANSWERS_TEST_REALGEN.jsonl"]["sha256"]
    )

    # Invoke only the control-plane guard. authorize(False, None) must refuse
    # before reading REALGEN_TEST_PREDEPLOY.json or any evaluation input.
    spec = importlib.util.spec_from_file_location("realgen_test_official", REAL / "realgen_test_official.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(module.Refusal, match="explicit --run-now is required"):
        module.authorize(False, None)


def test_background_has_zero_overlap_with_test_hosting_docs() -> None:
    """Background contains 6510 documents and no 1200 test-hosting doc_id."""
    background = _load("REALGEN_BACKGROUND.json")
    manifest = _load("CORPUS_MANIFEST.json")
    ids = [entry["id"] for entry in manifest["entries"]]
    assert background["n_documents"] == 6510 == len(background["doc_ids"])
    assert background["asserts"]["overlap_with_test_hosting_docs"] == 0
    assert background["asserts"]["real_test_locked_file_read"] is False
    assert set(background["doc_ids"]) <= set(ids), "фон содержит id вне манифеста"
    selected = sorted(ids, key=lambda x: hashlib.sha256(f"42:{x}".encode()).hexdigest())[:1500]
    ordered = sorted(selected, key=lambda x: hashlib.sha256(f"42:split:{x}".encode()).hexdigest())
    test_ids = set(ordered[300:1500])
    assert not test_ids & set(background["doc_ids"]), "фон содержит test-hosting документы"


def test_dev_and_test_realgen_splits_do_not_intersect() -> None:
    """The dev payload is checked directly; sealed counts come from metadata."""
    splits = _load("REALGEN_SPLITS.json")
    frozen = _load("REALGEN_FROZEN.json")
    assert splits["dev_answers"] == 300
    assert splits["overlap"] == 0
    assert splits["test_realgen_sealed"] is True
    assert not set(splits["dev_docs"]) & set(splits["test_realgen_docs"])
    rows = [json.loads(line) for line in (REAL / "REALGEN_ANSWERS_DEV.jsonl").open(encoding="utf-8") if line.strip()]
    assert len(rows) == 300
    assert all(row["split"] == "dev" for row in rows)
    assert frozen["splits"]["test_realgen"]["n_answers"] == splits["test_realgen_answers"]


def test_phase_b_reports_only_dev_and_honest_gates() -> None:
    """Phase B records 300 dev answers and never claims test data access."""
    phase_b = _load("REALGEN_PHASE_B.json")
    assert phase_b["data"]["dev_answers"] == 300
    assert phase_b["data"]["smoke_test"] is False
    assert phase_b["real_test_viewed"] is False
    assert phase_b["provenance"]["test_realgen_file_read"] is False
    assert phase_b["provenance"]["real_test_locked_file_read"] is False
    assert len(phase_b["per_feature_AUC"]) == 27
    assert phase_b["gates"]["token_AUC_min"] == 0.70
    assert phase_b["gates"]["span_AUC_min"] == 0.55
    assert phase_b["gates"]["verdict"] in ("PASS", "FAIL")
    for key in ("composite_OOF", "rubert_tiny2_finetune"):
        block = phase_b[key]
        assert 0.0 <= block["token_AUC"] <= 1.0
        assert 0.0 <= block["span_AUC"] <= 1.0
        assert len(block["token_ci95"]) == len(block["span_ci95"]) == 2


def test_phase_b_oof_rows_never_touch_test_realgen_docs() -> None:
    """OOF rows are dev-only and cover all five fixed folds."""
    splits = _load("REALGEN_SPLITS.json")
    test_docs = set(splits["test_realgen_docs"])
    seen, folds, labels = 0, set(), set()
    with (REAL / "REALGEN_PHASE_B_OOF.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            assert row["doc_id"] not in test_docs
            folds.add(row["fold"])
            labels.add(row["label"])
            seen += 1
    assert seen == _load("REALGEN_PHASE_B.json")["data"]["n_tokens"]
    assert folds == {0, 1, 2, 3, 4}
    assert labels <= {0, 1}


def test_phase_a_reports_run_id_provenance() -> None:
    """Generation provenance contains a verified Actions run ID."""
    phase_a = _load("REALGEN_PHASE_A.json")
    audit = _load("RUN_ID_AUDIT.json")
    assert phase_a["provenance"]["real_test_viewed"] is False
    assert phase_a["provenance"]["ci_run_ids"], "нет проверяемого CI run-id"
    assert phase_a["provenance"]["run_id"] in ("local-no-ci-evidence", *phase_a["provenance"]["ci_run_ids"])
    assert any(entry.get("kind") == "verified-actions-run" for entry in audit["entries"])
