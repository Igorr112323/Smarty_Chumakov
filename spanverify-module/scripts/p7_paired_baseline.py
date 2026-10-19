"""Materialize P6 executable source only; same-host train-only before benchmark."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]
REPO = MODULE.parent
TARGET = REPO / ".arena/p7/paired-before"
SOURCE = "14a018a1b90b51cc47cd8baf628147ac22e1d8b1"


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    archive = subprocess.check_output(
        [
            "git",
            "archive",
            SOURCE,
            "spanverify-module/spanverify",
            "spanverify-module/config",
        ],
        cwd=REPO,
    )
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        members = tar.getmembers()
        for member in members:
            if (
                member.issym()
                or member.islnk()
                or not (TARGET / member.name).resolve().is_relative_to(TARGET.resolve())
            ):
                raise ValueError("Unexpected snapshot member")
        tar.extractall(TARGET, members=members)
    root = TARGET / "spanverify-module"
    (root / "scripts").mkdir(exist_ok=True)
    # Same measurement harness/start/stop/repeats, with absolute paths and
    # timeout-bound instrumentation. Scored application is unmodified P6.
    shutil.copy2(MODULE / "scripts/p6_acceptance.py", root / "scripts/p6_acceptance.py")
    for name in ("data", "reports"):
        link = root / name
        if not link.exists():
            link.symlink_to(MODULE / name, target_is_directory=True)
        elif link.resolve() != (MODULE / name).resolve():
            raise ValueError("Snapshot input link mismatch")
    receipt = dict(
        program_source=SOURCE,
        harness_source=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        training=False,
        official_test_read=False,
        inputs="Only document(n,scenario) from the first train row, identical P6/P7 sizes",
        purpose="Paired same-host measurement, not a rerun or rewrite of historical P6 metrics",
    )
    (TARGET / "snapshot-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
