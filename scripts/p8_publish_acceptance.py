"""Publish already-built frozen binaries and original evidence: NO fits/inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPO = "Igorr112323/Smarty_Chumakov"


def api(path):
    for attempt in range(3):
        try:
            return json.loads(subprocess.check_output(["gh", "api", f"repos/{REPO}/{path}"], text=True))
        except subprocess.CalledProcessError:
            if attempt == 2:
                raise
            time.sleep(3)
    raise RuntimeError("unreachable metadata retry")


def run_info(number):
    return json.loads(subprocess.check_output(["gh", "run", "view", str(number), "--json", "status,conclusion,headSha,url"], text=True))


def current_ci():
    """Wait for ALL ordinary CI runs on this exact publication commit."""
    sha = os.environ["GITHUB_SHA"]
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        runs = api("actions/workflows/ci.yml/runs?branch=arena%2Fc5f69767-smarty-chumakov&per_page=100")["workflow_runs"]
        matches = [r for r in runs if r["head_sha"] == sha]
        if matches and all(r["status"] == "completed" for r in matches):
            assert all(r["conclusion"] == "success" for r in matches), [(r["id"], r["conclusion"]) for r in matches]
            return [r["id"] for r in matches]
        print("Await exact-head full ordinary CI:", [(r["id"], r["status"]) for r in matches], flush=True)
        time.sleep(30)
    raise TimeoutError("Full ordinary CI has not sealed this exact publication HEAD")


def validate():
    delivery = json.loads((ROOT / "reports/p8/DELIVERY.json").read_text())
    ci_ids = current_ci()
    runs = {role: run_info(number) for role, number in {"evaluation": 39266451529, "windows": delivery["windows_run"], "ci": ci_ids[0]}.items()}
    windows_jobs = api(f"actions/runs/{delivery['windows_run']}/jobs?per_page=100")["jobs"]
    # Original build succeeded completely; ONLY PS release argument splatting failed.
    # Keep the failure, never relabel the source workflow as SUCCESS.
    for role, info in runs.items():
        assert info["status"] == "completed", (role, info)
        if role == "windows" and info["conclusion"] == "failure":
            failures = [s["name"] for j in windows_jobs for s in j["steps"] if s["conclusion"] == "failure"]
            assert len(failures) == 1 and failures[0].startswith("Публикация релиза"), failures
            assert all(j["conclusion"] == "success" for j in windows_jobs if j["name"].startswith("Тесты"))
            build = next(j for j in windows_jobs if j["name"] == "Сборка и проверка .exe")
            for fragment in ("Проверка размеров поставки", "Windows HF autonomy", "Артефакты сборки", "Смоук-тест zipapp", "Проверка автономности"):
                step = next(s for s in build["steps"] if fragment in s["name"])
                assert step["conclusion"] == "success", step
        else:
            assert info["conclusion"] == "success", (role, info)
        subprocess.run(["git", "diff", "--exit-code", info["headSha"], "HEAD", "--", "spanverify-module/spanverify", "spanverify-module/config", "spanverify-module/reports/experiments/p5_span_filter"], check=True)
    quality = json.loads((ROOT / "reports/p8/run39266451529/official/result.json").read_text())
    assert quality["joint_gate"] is True
    return dict(runs=runs, ci_runs=ci_ids, windows_run=delivery["windows_run"], tests_excluded=False, documentation_source=os.environ["GITHUB_SHA"], release_tag=delivery["release_tag"], stable_promotion=delivery["stable_promotion"], execution_freeze_status=delivery["execution_freeze_status"], windows_source_failure_scope="Publication only; all build/test/size/offline checks SUCCESS, original failure retained", publication_run=os.environ["GITHUB_RUN_ID"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    meta = validate()
    if args.prepare:
        (ROOT / ".arena").mkdir(exist_ok=True)
        (ROOT / ".arena/p8-publication-metadata.json").write_text(json.dumps(meta))
        with open(os.environ["GITHUB_OUTPUT"], "a") as handle:
            handle.write(f"windows_run={meta['windows_run']}\n")
        return
    binaries = ROOT / ".arena/p8-publication/windows/release"
    for line in (binaries / "SHA256SUMS.txt").read_text(encoding="utf-8-sig").splitlines():
        expected, name = line.split(maxsplit=1)
        assert hashlib.sha256((binaries / name.strip()).read_bytes()).hexdigest() == expected
    for folder, role in [("p8-official177", "official"), *[(f"p8-external-{i}", f"external-{i}") for i in range(4)]]:
        manifest = json.loads((ROOT / f"reports/p8/run39266451529/{role}/artifact-manifest.json").read_text())
        path = ROOT / ".arena/p8-publication/evidence" / folder / "evidence.jsonl"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["manifest"]["evidence.jsonl"]["sha256"]
    branch = api("git/ref/heads/arena/c5f69767-smarty-chumakov")
    assert branch["object"]["sha"] == os.environ["GITHUB_SHA"], "Superseded publication source; refuse publishing"
    assets = [str(binaries / name) for name in ("spanverify.exe", "spanverify-win.zip", "spanverify.pyz", "frozen-p5.json", "p5-runtime.json", "SHA256SUMS.txt")]
    # Real argv list, not PowerShell scalar-string splatting. No rebuild/re-inference.
    subprocess.run(["gh", "release", "create", meta["release_tag"], *assets, "--repo", REPO, "--target", meta["documentation_source"], "--title", "SpanVerify 1.3.1 — P8 final evaluation", "--notes-file", str(ROOT / "reports/p8/RELEASE_NOTES.md"), *(["--latest=true"] if meta["stable_promotion"] else ["--prerelease", "--latest=false"])], check=True, capture_output=True, text=True)
    out = ROOT / ".arena/p8-publication/p8-final-acceptance.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    checks = {}
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0"):
            path = ROOT / name
            if name and path.is_file():
                archive.write(path, "source/" + name)
        for folder, prefix in [(ROOT / ".arena/p8-publication/windows", "delivery"), (ROOT / ".arena/p8-publication/evidence", "evidence")]:
            for path in sorted(folder.rglob("*")):
                if path.is_file():
                    name = prefix + "/" + str(path.relative_to(folder))
                    archive.write(path, name)
                    checks[name] = dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        archive.writestr("FINAL-PUBLICATION.json", json.dumps(dict(meta, files=checks), ensure_ascii=False, indent=2))
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    sha = out.with_suffix(".zip.sha256")
    sha.write_text(f"{digest}  {out.name}\n")
    subprocess.run(["gh", "release", "upload", meta["release_tag"], str(out), str(sha)], check=True, capture_output=True, text=True)
    print("::notice title=P8 final archive::" + json.dumps(dict(meta, sha256=digest, bytes=out.stat().st_size)))


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if isinstance(exc, subprocess.CalledProcessError):
            print("::error title=P8 command stderr::" + ((exc.stderr or "") + (exc.stdout or "")).replace("%", "%25").replace("\n", "%0A").replace("\r", "%0D"))
        print("::error title=P8 publication infrastructure::" + traceback.format_exc().replace("\n", "%0A"))
        raise
