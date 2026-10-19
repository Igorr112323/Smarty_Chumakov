"""P6 isolation: full regression must not open the reserved official test177."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESERVED = {
    str((ROOT / "data/corpus_a3/splits/test.jsonl").resolve()),
    str((ROOT / "data/corpus_a3/pairs.jsonl").resolve()),
}


def _protect_reserved(event, args):
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        if os.path.realpath(os.fsdecode(args[0])) in RESERVED:
            raise PermissionError("P6: official test177 and the unsplit corpus are reserved")


sys.addaudithook(_protect_reserved)
