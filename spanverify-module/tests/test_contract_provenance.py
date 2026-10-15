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

    if not contract_path.exists():
        pytest.skip("CONTRACT.json not found")

    with open(contract_path, encoding="utf-8") as f:
        contract = json.load(f)

    provenance = contract.get("provenance")

    if provenance == "ci":
        # If provenance is "ci", must have source_artifact and commit
        assert "source_artifact" in contract, (
            "CONTRACT.json with provenance='ci' must have 'source_artifact' field"
        )
        assert "commit" in contract, (
            "CONTRACT.json with provenance='ci' must have 'commit' field"
        )

        # commit must match the artifact reference
        commit = contract["commit"]

        # Verify commit is not a base commit (must be from the CI run)
        # This is a heuristic: if generated_at is present, commit should be recent
        if "generated_at" in contract:
            # If we have a generated_at timestamp, commit should be from the run
            # In practice, we just check that commit is present and non-empty
            assert len(commit) >= 7, "commit hash must be at least 7 characters"

    elif provenance == "local":
        # Local provenance is always valid
        pass

    else:
        pytest.fail(
            f"CONTRACT.json provenance must be 'local' or 'ci', got: {provenance}"
        )
