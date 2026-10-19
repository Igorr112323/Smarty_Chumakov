"""Publish final documentation over verified measurement artifacts; never measure/fit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
import zipfile
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
SCIENCE_PATHS = [
    "spanverify-module/spanverify",
    "spanverify-module/scripts",
    "spanverify-module/config",
    "spanverify-module/spanverify.spec",
    ".github/workflows/p7-acceptance.yml",
    ".github/workflows/build-and-release-exe.yml",
    ".github/workflows/p7-paired-benchmark.yml",
]


def validate() -> dict:
    collection = json.loads(
        (ROOT / "reports/p7/final-receipts/collection.json").read_text()
    )
    assert (
        collection["require_success"] is True and collection["tests_excluded"] is False
    )
    runs = {}
    for role in ("acceptance", "windows", "ci"):
        run = json.loads(
            subprocess.check_output(
                [
                    "gh",
                    "run",
                    "view",
                    str(collection[role]),
                    "--json",
                    "headSha,status,conclusion,url",
                ],
                text=True,
            )
        )
        assert run["status"] == "completed" and run["conclusion"] == "success", (
            role,
            run,
        )
        subprocess.run(
            [
                "git",
                "diff",
                "--exit-code",
                run["headSha"],
                "HEAD",
                "--",
                *SCIENCE_PATHS,
            ],
            cwd=ROOT,
            check=True,
        )
        runs[role] = run
    assert runs["acceptance"]["headSha"] == runs["windows"]["headSha"]
    return {
        "kind": "artifact-publication-only; no new measurements or fitting",
        "measurement_source": runs["acceptance"]["headSha"],
        "documentation_source": os.environ["GITHUB_SHA"],
        "runs": runs,
        "acceptance_run": collection["acceptance"],
        "tests_excluded": False,
    }


def assemble(original: Path, overlay: Path, out: Path, metadata: dict) -> str:
    archive = original / "p7-acceptance.zip"
    expected = (original / "p7-acceptance.zip.sha256").read_text().split()[0]
    assert (
        hashlib.sha256(archive.read_bytes()).hexdigest() == expected
    ), "source archive checksum mismatch"
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as temporary:
        stage = Path(temporary)
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                path = Path(member.filename)
                target = stage / path
                if (
                    PureWindowsPath(member.filename).drive
                    or path.is_absolute()
                    or ".." in path.parts
                    or "\\" in member.filename
                    or stat.S_ISLNK(member.external_attr >> 16)
                    or not target.resolve().is_relative_to(stage.resolve())
                ):
                    raise ValueError("unsafe archive member")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.open(member) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
        shutil.copytree(overlay, stage, dirs_exist_ok=True)
        (stage / "FINAL-PUBLICATION.json").write_text(
            json.dumps(
                dict(metadata, original_archive_sha256=expected),
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        shutil.make_archive(str(out.with_suffix("")), "zip", stage)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_suffix(out.suffix + ".sha256").write_text(f"{digest}  {out.name}\n")
    return digest


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    metadata = validate()
    if args.prepare:
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            output.write(f"acceptance_run={metadata['acceptance_run']}\n")
        return
    digest = assemble(
        ROOT / ".arena/p7-publication/original",
        ROOT / "reports/p7",
        ROOT / ".arena/p7-publication/p7-final-acceptance.zip",
        metadata,
    )
    print(
        "::notice title=P7 final acceptance archive::"
        + json.dumps(dict(metadata, sha256=digest))
    )


if __name__ == "__main__":
    main()
