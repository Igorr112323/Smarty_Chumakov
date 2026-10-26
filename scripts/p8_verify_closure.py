"""Read-only P8 seal: hash existing bytes and saved results; NEVER import a model.

Only --output is written. Git history, datasets, evidence and assets are inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "Igorr112323/Smarty_Chumakov"
ACCEPTANCE = "9be619031a6cd0d47e826e7ea63208d87f42315e"
RC = "v1.3.1-rc.p8.20261019"
STABLE = "p8-stable-20261019"
BUNDLE = "e0b13c693a058dfd66ed24a52fc14e9c70964e95885358e7b99cc1f328ad6892"
DATASET = "232d2583480f2237fc48d10eb4413043bf35f59be5a10138fbf53235ce83ba88"
EVIDENCE = "7607c513a760b68860e89aa3255d7f44942cc6a678c83bca3c641128067b738f"
ARCHIVE = "58143baac4a39d86aeaa46879c497d5195f0202f58174c91af746d94dfe2f158"


def gh(*args):
    return subprocess.check_output(["gh", *args], cwd=ROOT, text=True)


def api(path):
    return json.loads(gh("api", f"repos/{REPO}/{path}"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(assets, release_tag):
    checks = {}
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    checks["no_staged_diff"] = (
        subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0
    )
    checks["acceptance_ancestor"] = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ACCEPTANCE, "HEAD"], cwd=ROOT
        ).returncode
        == 0
    )
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", ACCEPTANCE, "HEAD"], cwd=ROOT, text=True
    ).splitlines()
    import re

    checks["historical_run_catalogs_unchanged"] = not any(
        re.search(r"(^|/)run\d+(/|$)", p) for p in changed
    )
    frozen = json.loads(
        (ROOT / "reports/p8/frozen-production-sha256.json").read_text()
    )["files"]
    checks["frozen_prediction_files_identical"] = all(
        sha(ROOT / name) == digest for name, digest in frozen.items()
    )
    bundle = (
        ROOT
        / "spanverify-module/reports/experiments/p5_span_filter/run39170981235/combined_bundle.json"
    )
    checks["bundle_sha256"] = sha(bundle) == BUNDLE
    checks["bundle_size_122874"] = bundle.stat().st_size == 122874
    checks["test_dataset_sha256"] = (
        sha(ROOT / "spanverify-module/data/corpus_a3/splits/test.jsonl") == DATASET
    )
    result = json.loads(
        (ROOT / "reports/p8/run39266451529/official/result.json").read_text()
    )
    checks["joint_gate_true"] = result["joint_gate"] is True
    checks["saved_result_hashes_match"] = (
        result["bundle_sha256"] == BUNDLE and result["dataset_sha256"] == DATASET
    )
    manifest = json.loads(
        (ROOT / "reports/p8/run39266451529/official/artifact-manifest.json").read_text()
    )
    checks["official_evidence_manifest_sha256"] = (
        manifest["manifest"]["evidence.jsonl"]["sha256"] == EVIDENCE
    )
    archive = assets / "p8-final-acceptance.zip"
    checks["release_archive_sha256"] = sha(archive) == ARCHIVE
    with zipfile.ZipFile(archive) as z:
        member = "evidence/p8-official177/evidence.jsonl"
        h = hashlib.sha256()
        with z.open(member) as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        checks["official_evidence_bytes_sha256"] = h.hexdigest() == EVIDENCE
        checks["official_evidence_rows_177"] = (
            manifest["manifest"]["evidence.jsonl"]["rows"] == 177
        )
        checks["archive_test_dataset_sha256"] = (
            hashlib.sha256(
                z.read("source/spanverify-module/data/corpus_a3/splits/test.jsonl")
            ).hexdigest()
            == DATASET
        )
    sums = {}
    for line in (
        (assets / "SHA256SUMS.txt").read_text(encoding="utf-8-sig").splitlines()
    ):
        digest, name = line.split(maxsplit=1)
        sums[name.strip().lstrip("*")] = digest
    checks["all_SHA256SUMS_files_match"] = all(
        sha(assets / name) == digest for name, digest in sums.items()
    )
    for name, size in [("spanverify.exe", 8981233), ("spanverify.pyz", 232874)]:
        checks[name + "_size"] = (assets / name).stat().st_size == size
        checks[name + "_sha256"] = sha(assets / name) == sums[name]
    rc = api("releases/tags/" + RC)
    release = api("releases/tags/" + release_tag)
    expected = json.loads(
        (ROOT / "reports/p8/run39290438219/release.json").read_text()
    )["assets"]
    hashes = {}
    for item in expected:
        name = item["name"]
        digest = sha(assets / name)
        hashes[name] = digest
        remote = next(a for a in release["assets"] if a["name"] == name)
        original = next(a for a in rc["assets"] if a["name"] == name)
        checks["asset_digest_" + name] = (
            item["digest"]
            == original["digest"]
            == remote["digest"]
            == "sha256:" + digest
            and item["size"] == remote["size"] == (assets / name).stat().st_size
        )
    checks["stable_published"] = not release["draft"] and not release["prerelease"]
    checks["stable_tag_acceptance_HEAD"] = (
        api("git/ref/tags/" + release_tag)["object"]["sha"] == ACCEPTANCE
    )
    checks["latest_points_to_stable"] = api("releases/latest")["id"] == release["id"]
    return dict(
        run_id="P8-close-verify:" + os.getenv("GITHUB_RUN_ID", "local"),
        date=datetime.now(UTC).date().isoformat(),
        execution_head=head,
        acceptance_head=ACCEPTANCE,
        read_only_inputs=True,
        tests_excluded=False,
        checks=checks,
        all_true=all(checks.values()),
        file_sha256=hashes,
        stable_release=release["html_url"],
        receipt_scope="Actual execution HEAD; later receipt-only commit is independently reverified by Final CI job. No scoring, model import, fit, calibration or rebuild.",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=Path, default=ROOT / ".arena/closure-assets")
    p.add_argument("--release", default=STABLE)
    p.add_argument("--output", type=Path, default=ROOT / "reports/p8/VERIFICATION.json")
    args = p.parse_args()
    if not args.assets.exists():
        args.assets.mkdir(parents=True)
        gh(
            "release",
            "download",
            args.release,
            "--repo",
            REPO,
            "--dir",
            str(args.assets),
        )
    result = verify(args.assets, args.release)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        "Final CI verify-pass: "
        + ("ALL TRUE" if result["all_true"] else "FAILED")
        + " HEAD="
        + result["execution_head"]
    )
    if not result["all_true"]:
        raise SystemExit(
            json.dumps({k: v for k, v in result["checks"].items() if not v})
        )


if __name__ == "__main__":
    main()
