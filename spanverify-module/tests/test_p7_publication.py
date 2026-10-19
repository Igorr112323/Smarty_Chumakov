"""Final artifact publication must retain raw proof and reject corrupt/unsafe ZIPs."""

from __future__ import annotations

import hashlib
import importlib.util
import zipfile
from pathlib import Path

import pytest


def publisher():
    path = Path(__file__).resolve().parents[2] / "scripts/p7_publish_acceptance.py"
    spec = importlib.util.spec_from_file_location("p7_publication_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original(path, member="REPORT.md"):
    path.mkdir(exist_ok=True)
    archive = path / "p7-acceptance.zip"
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr(member, "original evidence")
        target.writestr("raw/profile.prof", b"raw unchanged")
    (path / "p7-acceptance.zip.sha256").write_text(hashlib.sha256(archive.read_bytes()).hexdigest())


def test_final_publication_keeps_raw_and_overlays_final_report(tmp_path):
    original(tmp_path / "old")
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "REPORT.md").write_text("final report")
    out = tmp_path / "out/final.zip"
    digest = publisher().assemble(tmp_path / "old", overlay, out, {"kind": "publication only"})
    assert digest == hashlib.sha256(out.read_bytes()).hexdigest()
    with zipfile.ZipFile(out) as result:
        assert result.read("REPORT.md") == b"final report"
        assert result.read("raw/profile.prof") == b"raw unchanged"
        assert b"publication only" in result.read("FINAL-PUBLICATION.json")


def test_final_publication_rejects_path_escape_and_wrong_checksum(tmp_path):
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    for member in ("../escape", "/absolute", "..\\windows-escape", "C:/escape"):
        original(tmp_path / "old", member)
        with pytest.raises(ValueError, match="unsafe archive"):
            publisher().assemble(tmp_path / "old", overlay, tmp_path / "out/final.zip", {})
    original(tmp_path / "old")
    (tmp_path / "old/p7-acceptance.zip.sha256").write_text("0" * 64)
    with pytest.raises(AssertionError, match="checksum"):
        publisher().assemble(tmp_path / "old", overlay, tmp_path / "out/final.zip", {})
