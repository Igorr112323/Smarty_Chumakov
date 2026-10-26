#!/usr/bin/env python3
"""Локальная ручная экстракция REALGEN-данных из check-runs output.text.

Никакого auto-commit: скрипт только кладёт файлы в .arena/realgen_extract/,
дальше человек/агент коммитит вручную с полем extraction в RUN_ID_AUDIT.json.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import realgen_common as rc  # noqa: E402

REPO = "Igorr112323/Smarty_Chumakov"
PREFIX_MARK = "REALGEN-CHUNK-v1|"


def list_check_runs(head_sha: str, name_prefix: str) -> list[dict]:
    found: dict[str, tuple[int, dict]] = {}
    for page in range(1, 11):
        proc = subprocess.run(
            [
                "gh",
                "api",
                "-X",
                "GET",
                f"repos/{REPO}/commits/{head_sha}/check-runs",
                "-f",
                "per_page=100",
                "-f",
                f"page={page}",
                "--jq",
                '[.check_runs[] | select(.name|startswith("' + name_prefix + '")) | {id, name, summary: .output.summary, text: .output.text}]',
            ],
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"gh api check-runs failed: {proc.stderr[:400]}")
        rows = json.loads(proc.stdout or "[]")
        if not rows:
            break
        for row in rows:
            ident = int(row["id"])
            if row["name"] not in found or ident > found[row["name"]][0]:
                found[row["name"]] = (ident, row)
        if len(rows) < 100:
            break
    return [item[1] for item in found.values()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--name-prefix", default="REALGEN-A/")
    parser.add_argument("--out-dir", default=".arena/realgen_extract")
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    runs = list_check_runs(args.head_sha, args.name_prefix)
    chunks: dict[str, dict[int, str]] = {}
    metas: dict[str, dict] = {}
    for row in runs:
        text = row.get("text") or ""
        name = row["name"]
        if name.endswith("/meta"):
            summary = row.get("summary") or ""
            if "```json" in summary:
                try:
                    metas[name[: -len("/meta")]] = json.loads(summary.split("```json", 1)[1].split("```", 1)[0])
                except json.JSONDecodeError:
                    pass
            continue
        if not text.startswith(PREFIX_MARK):
            continue
        _, prefix, index, total, sha, payload = text.split("|", 5)
        chunks.setdefault(prefix, {})[int(index)] = payload
        metas.setdefault(prefix, {}).update({"expected_chunks": int(total), "sha256": sha})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for prefix, parts in sorted(chunks.items()):
        meta = metas.get(prefix, {})
        expected = meta.get("expected_chunks", max(parts) + 1)
        if sorted(parts) != list(range(expected)):
            raise RuntimeError(f"{prefix}: неполный набор чанков {sorted(parts)} != 0..{expected - 1}")
        raw = gzip.decompress(base64.b64decode("".join(parts[i] for i in range(expected))))
        sha = hashlib.sha256(raw).hexdigest()
        if sha != meta.get("sha256", sha):
            raise RuntimeError(f"{prefix}: sha256 не совпал {sha} != {meta['sha256']}")
        safe = prefix.replace("/", "__") + ".jsonl"
        path = out_dir / safe
        path.write_bytes(raw)
        report.append({"prefix": prefix, "file": str(path), "bytes": len(raw), "sha256": sha, "chunks": expected})
        print(f"{prefix}: {len(raw)} bytes sha256={sha} chunks={expected}")

    (out_dir / "EXTRACTION_META.json").write_text(
        json.dumps({"kind": "REALGEN_EXTRACTION_META", "head_sha": args.head_sha, "metas": metas}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "EXTRACTION.json").write_text(
        json.dumps(
            {
                "kind": "REALGEN_EXTRACTION",
                "extraction": "manual: gh api check-runs output.text (no workflow autocommit)",
                "head_sha": args.head_sha,
                "run_id": args.run_id or "local-no-ci-evidence",
                "files": report,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"итого файлов: {len(report)}")


if __name__ == "__main__":
    main()
