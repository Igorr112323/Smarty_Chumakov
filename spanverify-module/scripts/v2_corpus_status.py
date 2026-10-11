#!/usr/bin/env python3
"""Состояние корпуса v2 в ветке данных: сколько документов, в каком режиме, что требует подтверждения.

Зачем
-----

Загрузка остановлена по решению заказчика (режим `--text-only`, а цепочка волн — на паузе, см.
`.github/corpus-v2.mode`). Уже скачанное не удаляется, но должно быть помечено: документы,
полученные распознаванием, лежат в ветке данных как `ocr_pending_approval` — они не участвуют в
заморозке шага 0, пока заказчик явно не подтвердит режим.

Что считается
-------------

Только то, что видно по файлам: число `.txt` в `acts/`, число непустых, распределение длины,
число отчётов волн. Никаких оценок «сколько ещё надо» скрипт не выдаёт.

Воспроизведение (локально, по FETCH_HEAD после `git fetch origin data/corpus-v2-raw`):

    git fetch -q origin data/corpus-v2-raw
    python3 scripts/v2_corpus_status.py --git-ref FETCH_HEAD --json reports/v2/corpus_status.json

или по каталогу сборки волны:

    python3 scripts/v2_corpus_status.py --dir /tmp/v2/wave --json reports/v2/corpus_status.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

MODE_FILE = Path(__file__).resolve().parents[2] / ".github" / "corpus-v2.mode"
MIN_CHARS_USABLE = 800  # столько же, сколько `--min-facts` не гарантирует, но текст должен быть не пустой


APPROVAL_RE = re.compile(r"\bapproved\b\s+\d{4}-\d{2}-\d{2}", re.IGNORECASE)


def read_mode(path: Path = MODE_FILE) -> str:
    """Режим загрузки из файла-переключателя (пустой файл или отсутствие = text-only)."""
    return mode_and_approval(path)[0]


def mode_and_approval(path: Path = MODE_FILE) -> tuple[str, bool]:
    """(режим, подтверждён ли он заказчиком) из файла-переключателя режима.

    Файл читается двумя словами: режим (`text-only` | `ocr` | `paused`) и — только для
    `ocr` — отметка подтверждения `approved YYYY-MM-DD`. Распознавание не считается
    выбранным режимом, пока в файле нет отметки: файл меняется коммитом в рабочую ветку,
    и «тихий» переход `paused → ocr` без записи о решении заказчика тесту виден.
    """
    try:
        line = path.read_text(encoding="utf-8").strip()
    except OSError:
        return ("text-only", True)
    if not line:
        return ("text-only", True)
    mode = line.split()[0]
    approved = bool(APPROVAL_RE.search(line)) if mode == "ocr" else mode == "text-only"
    return (mode, approved)


REPO_ROOT = Path(__file__).resolve().parents[2]


def list_blobs(git_ref: str, repo_root: Path = REPO_ROOT) -> list[tuple[str, bytes]]:
    """(путь, содержимое) каждого файла ветки — читается `git show`, без рабочей копии.

    `-C` обязателен: `git ls-tree` из подкаталога берёт текущий каталог как префикс пути и
    для ветки данных (где `acts/` лежит в корне репозитория) вернул бы пустой список.
    """
    git = ["git", "-C", str(repo_root)]
    listing = subprocess.run(
        git + ["ls-tree", "-r", "--name-only", git_ref],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    out: list[tuple[str, bytes]] = []
    for name in listing:
        blob = subprocess.run(git + ["show", f"{git_ref}:{name}"], capture_output=True, check=True).stdout
        out.append((name, blob))
    return out


def summarize(files: list[tuple[str, bytes]]) -> dict[str, Any]:
    """Счётчики по содержимому: документы в `acts/`, непустые, годные по длине, отчёты волн."""
    acts = [(name, blob) for name, blob in files if "/acts/" in f"/{name}" and name.endswith(".txt")]
    reports = [name for name, _ in files if name.startswith("reports-wave")]
    lengths = sorted(len(blob.decode("utf-8", "replace").strip()) for _, blob in acts)
    nonempty = sum(1 for value in lengths if value > 0)
    usable = sum(1 for value in lengths if value >= MIN_CHARS_USABLE)

    def quantile(q: float) -> int:
        if not lengths:
            return 0
        index = min(len(lengths) - 1, max(0, int(round(q * (len(lengths) - 1)))))
        return lengths[index]

    return {
        "documents_in_acts": len(acts),
        "documents_nonempty": nonempty,
        "documents_at_least_min_chars": usable,
        "min_chars_usable": MIN_CHARS_USABLE,
        "chars_median": quantile(0.5),
        "chars_p10": quantile(0.1),
        "chars_p90": quantile(0.9),
        "wave_reports": len(reports),
        "report_files": sorted(reports),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--git-ref", default="", help="реф, из которого читаются файлы (например FETCH_HEAD)")
    ap.add_argument("--dir", default="", help="или каталог сборки волны (подкаталог acts/)")
    ap.add_argument("--data-branch", default="data/corpus-v2-raw", help="имя ветки данных для отчёта")
    ap.add_argument("--json", dest="json_out", default="", help="куда записать отчёт")
    args = ap.parse_args(argv)

    if bool(args.git_ref) == bool(args.dir):
        raise SystemExit("нужно ровно одно из: --git-ref <реф> или --dir <каталог>")

    if args.git_ref:
        files = list_blobs(args.git_ref)
        origin = f"git:{args.git_ref}"
    else:
        root = Path(args.dir)
        files = []
        for path in sorted(root.rglob("*.txt")):
            files.append((str(path.relative_to(root)), path.read_bytes()))
        origin = f"dir:{root}"

    mode, approved = mode_and_approval()
    payload: dict[str, Any] = {
        "data_branch": args.data_branch,
        "read_from": origin,
        "load_mode": mode,
        "mode_approved_by_customer": approved,
        # Режим, в котором эти файлы получены, — распознавание; заказчик его не подтверждал.
        # Флаг читается при заморозке шага 0: документы с этим флагом в неё не входят.
        "ocr_pending_approval": not approved,
        "note": (
            "Документы получены в режиме распознавания сканов. Заказчик выбрал --text-only, "
            "смена режима не подтверждена: файлы сохранены, но в размечаемую выборку не включаются, "
            "пока заказчик не ответит. Уже скачанное не удаляется."
        ),
        "counts": summarize(files),
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    counts = payload["counts"]
    print(
        f"режим загрузки: {mode} | ветка данных: {args.data_branch} | документов в acts/: "
        f"{counts['documents_in_acts']} (непустых {counts['documents_nonempty']}, "
        f"не короче {counts['min_chars_usable']} знаков {counts['documents_at_least_min_chars']}), "
        f"медиана длины {counts['chars_median']} знаков | отчётов волн: {counts['wave_reports']}"
    )
    print(f"ocr_pending_approval: {payload['ocr_pending_approval']}")
    if args.json_out:
        print(f"json: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
