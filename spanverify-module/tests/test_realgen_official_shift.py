"""Fail-closed curator shift gate of the one-shot REALGEN official evaluator.

Ordinary CI must prove that without REALGEN_OFFICIAL_SHIFT=1 the evaluator
refuses with exit 3 and does not open a single sealed byte (100 test_realgen
answers, gold labels/spans, test contexts or the locked real_test 1200 pairs).
The tests also pin the dev-only provenance of the frozen full-dev bundle.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parent
REAL = REPO_ROOT / "reports" / "science" / "real"
EVALUATOR = REAL / "realgen_test_official.py"
PREDEPLOY = REAL / "REALGEN_TEST_PREDEPLOY.json"
SEALED = (
    REAL / "REALGEN_ANSWERS_TEST_REALGEN.jsonl",
    REAL / "REALGEN_ANSWERS.jsonl",
    REAL / "REALGEN_LABELS.jsonl",
    REAL / "REALGEN_SPANS.jsonl",
    REAL / "REALGEN_DOCS.jsonl",
    REAL / "real_test_pairs.jsonl",
    REAL / "real_test_spans.jsonl",
)


def _evaluator():
    spec = importlib.util.spec_from_file_location("realgen_official_shift_under_test", EVALUATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("realgen_official_shift_under_test", module)
    spec.loader.exec_module(module)
    return module


def test_official_evaluator_refuses_without_shift_env(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """No REALGEN_OFFICIAL_SHIFT=1 → exit 3, zero sealed opens, explicit reason."""
    module = _evaluator()
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_RUN_ID", "40398753669")
    monkeypatch.setenv("REALGEN_SHIFT_CODE", "FIRE-REALGEN-1")
    monkeypatch.delenv("REALGEN_OFFICIAL_SHIFT", raising=False)
    monkeypatch.setattr(sys, "argv", ["realgen_test_official.py", "--run-now"])

    forbidden = {path.resolve() for path in SEALED}
    opened: list[str] = []
    original_open = Path.open

    def guarded_open(self: Path, *args, **kwargs):
        if self.resolve() in forbidden:
            opened.append(self.name)
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    exit_code = module.main()
    assert exit_code == 3, "missing shift authorization must be exit 3"
    assert opened == [], f"sealed payload opened without shift: {opened}"
    assert "REALGEN_OFFICIAL_SHIFT" in capsys.readouterr().err


def test_shift_gate_is_first_and_sealed_reads_are_last() -> None:
    """The gate precedes all metadata reads; sealed reads live only in evaluation."""
    script = EVALUATOR.read_text(encoding="utf-8")
    gate = script.index("hmac.compare_digest(os.environ.get(OFFICIAL_SHIFT_ENV")
    assert script.index("if not run_now:") < gate
    assert gate < script.index("PREDEPLOY.read_text")
    assert script.index("PREDEPLOY.read_text") < script.index("records = _safe_read_jsonl(TEST_ANSWERS)")
    assert script.index("except ShiftRefusal") < script.index("except Refusal")
    assert "return 3" in script
    assert script.index("def run_evaluation") < script.index("records = _safe_read_jsonl(TEST_ANSWERS)")


def test_frozen_bundle_is_dev_only_and_pinned() -> None:
    """The preregistration pins a dev-only bundle and contains no test metrics."""
    record = json.loads(PREDEPLOY.read_text(encoding="utf-8"))
    bundle = record["frozen_bundle"]
    assert bundle["trained_on"] == "realgen_dev_300_only"
    assert bundle["seed"] == 42
    assert bundle["token_threshold"] == record["policies"]["token_threshold"] == 0.5
    assert record["reported_metrics"]["official_test_values"] is None
    if bundle["status"] == "FROZEN":
        assert len(bundle["bundle"]["sha256"]) == 64
        assert str(bundle["train_run"]["run_id"]).isdecimal()
        assert len(bundle["weights"]["sha256"]) == 64
        assert bundle["weights"]["mb"] > 0 and bundle["bundle"]["mb"] > 0
        assert record["model_bundle"]["sha256"] == bundle["bundle"]["sha256"]
        assert record["model_bundle"]["actions_run_id"] == bundle["train_run"]["run_id"]
        assert {"composite", "r2_only", "rubert_tiny2_ft"} <= set(bundle["models"])
