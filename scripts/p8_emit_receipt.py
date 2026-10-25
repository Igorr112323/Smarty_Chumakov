"""Transfer existing artifact metadata, never invoke the verifier or training."""

import base64
import gzip
import hashlib
import json
import sys
from pathlib import Path

folder = Path(sys.argv[1])
role = sys.argv[2]
payload = dict(
    role=role,
    run_id=sys.argv[3] if len(sys.argv) > 3 else "39266451529",
    date="2026-10-19",
    files={},
    manifest={},
)
for path in sorted(folder.rglob("*")):
    if not path.is_file():
        continue
    data = path.read_bytes()
    name = str(path.relative_to(folder))
    payload["manifest"][name] = dict(
        bytes=len(data), sha256=hashlib.sha256(data).hexdigest()
    )
    if name.endswith("evidence.jsonl"):
        rows = data.splitlines()
        payload["manifest"][name]["rows"] = len(rows)
        payload["manifest"][name]["ids"] = [json.loads(row)["id"] for row in rows]
    elif (
        path.suffix in (".json", ".txt", ".md")
        and path.name != "frozen-p5.json"
        and len(data) < 1000000
    ):
        payload["files"][name] = base64.b64encode(data).decode()
encoded = base64.b64encode(
    gzip.compress(json.dumps(payload).encode(), mtime=0)
).decode()
for i in range(0, len(encoded), 3000):
    print(f"::notice title=P8 receipt {role} {i//3000:04d}::{encoded[i:i+3000]}")
