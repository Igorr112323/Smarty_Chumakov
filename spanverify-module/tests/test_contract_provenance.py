"""Test for CONTRACT.json provenance validation.

Ensures that if provenance is "ci", the artifact must have source_artifact
and commit matching the artifact.
"""

import json
from pathlib import Path

import pytest


def test_contract_provenance_local_or_ci():
    """Provenance must be 'local' or 'ci' with proper artifact reference."""
    contract_path = Path(__file__).parent.parent.parent / "reports" / "CONTRACT.json"
    repo_root = Path(__file__).parent.parent.parent

    if not contract_path.exists():
        pytest.skip("CONTRACT.json not found")

    with open(contract_path, encoding="utf-8") as f:
        contract = json.load(f)

    provenance = contract.get("provenance")

    if provenance == "ci":
        # If provenance is "ci", must have ci_run_id and source_artifact
        assert "ci_run_id" in contract, "CONTRACT.json with provenance='ci' must have 'ci_run_id' field"
        assert "commit" in contract, "CONTRACT.json with provenance='ci' must have 'commit' field"

        run_id = contract["ci_run_id"]
        artifact_path = repo_root / "reports" / "ci" / f"CONTRACT-{run_id}.json"

        assert artifact_path.is_file(), f"нет артефакта {artifact_path} для прогона {run_id}"

        # Verify commit matches the artifact
        with open(artifact_path, encoding="utf-8") as f:
            artifact = json.load(f)

        assert (
            artifact["commit"] == contract["commit"]
        ), "commit в CONTRACT.json не совпадает с commit в артефакте прогона"

        # Verify source_artifact is specified
        assert contract.get("source_artifact"), "не указан id артефакта в source_artifact"

    elif provenance == "local":
        # Local provenance is always valid
        pass

    else:
        pytest.fail(f"CONTRACT.json provenance must be 'local' or 'ci', got: {provenance}")


def test_contract_provenance_ci_without_artifact_fails():
    """Trap test: provenance='ci' without artifact must fail."""
    contract_path = Path(__file__).parent.parent.parent / "reports" / "CONTRACT.json"

    if not contract_path.exists():
        pytest.skip("CONTRACT.json not found")

    with open(contract_path, encoding="utf-8") as f:
        contract = json.load(f)

    provenance = contract.get("provenance")

    # If current provenance is 'ci', the artifact must exist
    if provenance == "ci":
        run_id = contract.get("ci_run_id")
        assert run_id, "ci_run_id должен быть указан при provenance='ci'"

        repo_root = Path(__file__).parent.parent.parent
        artifact_path = repo_root / "reports" / "ci" / f"CONTRACT-{run_id}.json"

        # If artifact is missing, this assertion will fail (as it should)
        assert artifact_path.is_file(), f"АРТЕФАКТ ОТСУТСТВУЕТ: {artifact_path} - тест обязан упасть!"
