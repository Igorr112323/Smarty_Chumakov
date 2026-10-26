"""Copy original RC bytes to curator-authorized stable; NO rebuild/inference."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "Igorr112323/Smarty_Chumakov"
TAG = "p8-stable-20261019"
assets = ROOT / ".arena/closure-assets"
assets.mkdir(parents=True, exist_ok=True)
subprocess.run(
    [
        "gh",
        "release",
        "download",
        "v1.3.1-rc.p8.20261019",
        "--repo",
        REPO,
        "--dir",
        str(assets),
    ],
    check=True,
)
expected = json.loads((ROOT / "reports/p8/run39290438219/release.json").read_text())[
    "assets"
]
for item in expected:
    path = assets / item["name"]
    assert path.stat().st_size == item["size"], path
    assert (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() == item["digest"]
    ), path
subprocess.run(
    [
        "gh",
        "release",
        "upload",
        TAG,
        *[str(assets / a["name"]) for a in expected],
        "--repo",
        REPO,
    ],
    check=True,
)
subprocess.run(
    [
        "gh",
        "release",
        "edit",
        TAG,
        "--repo",
        REPO,
        "--draft=false",
        "--prerelease=false",
        "--latest=true",
        "--notes-file",
        str(ROOT / "reports/p8/closure/STABLE_NOTES.md"),
    ],
    check=True,
)
subprocess.run(
    ["python", str(ROOT / "scripts/p8_verify_closure.py"), "--assets", str(assets)],
    check=True,
)
