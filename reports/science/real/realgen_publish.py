#!/usr/bin/env python3
"""Мост GITHUB_STEP_SUMMARY → check-runs output.text (разрешён протоколом).

Sandbox не имеет доступа к blob-логам и Azure-хранилищу артефактов, но имеет
api.github.com.Workflow НЕ делает коммитов: он только публикует check-runs с
base64(gzip(данных)) в output.text; человек/агент вручную извлекает и коммитит
файлы с полем extraction в RUN_ID_AUDIT.json.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import subprocess
from pathlib import Path

CHUNK = 60000  # лимит output.text = 65535 символов


def _api_post(repo: str, payload: dict) -> None:
    proc = subprocess.run(
        [
            "gh",
            "api",
            "--method",
            "POST",
            f"repos/{repo}/check-runs",
            "--input",
            "-",
        ],
        input=json.dumps(payload, ensure_ascii=False),
        text=True,
        capture_output=True,
        env={**os.environ, "GH_TOKEN": os.environ.get("GH_TOKEN", "")},
    )
    if proc.returncode != 0:
        raise RuntimeError(f"check-run POST failed: {proc.stderr[:500]}")


def publish_bytes(data: bytes, name_prefix: str, head_sha: str, repo: str, extra: dict | None = None) -> dict:
    packed = gzip.compress(data, compresslevel=9, mtime=0)
    encoded = base64.b64encode(packed).decode("ascii")
    chunks = [encoded[i : i + CHUNK] for i in range(0, len(encoded), CHUNK)]
    sha = hashlib.sha256(data).hexdigest()
    meta = {
        "kind": "REALGEN_CHECKRUN_PAYLOAD",
        "name_prefix": name_prefix,
        "sha256": sha,
        "raw_bytes": len(data),
        "gzip_bytes": len(packed),
        "base64_chars": len(encoded),
        "chunks": len(chunks),
        "encoding": "base64(gzip(jsonl, mtime=0))",
        "head_sha": head_sha,
    }
    if extra:
        meta.update(extra)
    summary = "```json\n" + json.dumps(meta, ensure_ascii=False, indent=1) + "\n```"
    _api_post(
        repo,
        {
            "name": f"{name_prefix}/meta",
            "head_sha": head_sha,
            "status": "completed",
            "conclusion": "success",
            "output": {
                "title": f"REALGEN {name_prefix} — манифест",
                "summary": summary[:60000],
                "text": f"meta={json.dumps(meta, ensure_ascii=False, sort_keys=True)}",
            },
        },
    )
    for index, chunk in enumerate(chunks):
        _api_post(
            repo,
            {
                "name": f"{name_prefix}/chunk-{index:04d}",
                "head_sha": head_sha,
                "status": "completed",
                "conclusion": "success",
                "output": {
                    "title": f"REALGEN {name_prefix} chunk {index + 1}/{len(chunks)}",
                    "summary": f"chunk {index + 1}/{len(chunks)} of {name_prefix}; sha256={sha}",
                    "text": f"REALGEN-CHUNK-v1|{name_prefix}|{index:04d}|{len(chunks):04d}|{sha}|{chunk}",
                },
            },
        )
    return meta


def publish_file(path: Path, name_prefix: str, head_sha: str, repo: str, extra: dict | None = None) -> dict:
    return publish_bytes(Path(path).read_bytes(), name_prefix, head_sha, repo, extra)


def step_summary(text: str) -> None:
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)
