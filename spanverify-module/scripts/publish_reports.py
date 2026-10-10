#!/usr/bin/env python3
"""Опубликовать небольшие файлы прогона в ветку через Contents API (без checkout и без rebase).

Зачем заменять `git checkout -B` на API. Прошлый прогон (`38043189316`) потерял
результаты так: шаги `grid` и `sweep` отработали, `upload-artifact` прошёл, а
шаг «Опубликовать в ветку данных» упал — он переключал рабочее дерево CI на
другую ветку, и любое совпадение путей между ветками (или гонка двух job'ов)
ломает такое переключение. Артефакты при этом читаются с blob-хоста, который
песочнице недоступен, — то есть числа оказывались невидимыми для того, кто
протокол проверяет.

Contents API не трогает рабочее дерево: файл читается, сравнивается, и при
отличии записывается одним запросом с ожидаемым `sha` (опечатка в `sha` даёт
409, а не тихую перезапись чужого коммита) — гонка двух job'ов разрешается
повтором. Запрос идёт через `gh api --input -` со stdin: у аргумента `execve`
лимит 128 КБ, а манифест с предсказаниями больше.

Использование::

    python scripts/publish_reports.py --branch data/metrics --dest hf-protocol \\
        --report reports/hf_protocol/sweep.json --report reports/hf_protocol/manifest_val.json \\
        --message "hf протокол: шаг 0-2, прогон $GITHUB_RUN_ID"
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MAX_ATTEMPTS = 4


def repo_slug() -> str:
    """`owner/repo`: из окружения CI, иначе из удалённого адреса git."""
    from_env = os.environ.get("GITHUB_REPOSITORY", "").strip()
    if from_env:
        return from_env
    try:
        url = subprocess.run(
            ["git", "-C", str(ROOT), "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover - нет git в окружении
        raise SystemExit(f"не могу определить репозиторий: {exc}") from exc
    tail = url.rsplit("/", 1)[-1]
    name = tail.removesuffix(".git")
    owner = url.split("/")[-2] if url.count("/") >= 1 else ""
    if not owner or not name:
        raise SystemExit(f"не могу разобрать адрес удалённого репозитория: {url}")
    return f"{owner}/{name}"


def existing_sha(slug: str, path: str, branch: str) -> str | None:
    """sha уже лежащего в ветке файла (для аккуратной перезаписи) или None."""
    completed = subprocess.run(
        [
            "gh",
            "api",
            f"repos/{slug}/contents/{path}?ref={branch}",
            "--jq",
            ".sha",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    value = completed.stdout.strip()
    return value or None


def current_bytes(slug: str, path: str, branch: str) -> bytes | None:
    """Содержимое файла в ветке — чтобы не создавать коммит «изменений нет»."""
    completed = subprocess.run(
        ["gh", "api", f"repos/{slug}/contents/{path}?ref={branch}", "--jq", ".content"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    try:
        return base64.b64decode("".join(completed.stdout.split()))
    except (ValueError, TypeError):  # pragma: no cover - ответ повреждён
        return None


def build_payload(*, message: str, branch: str, content: bytes, sha: str | None) -> str:
    """Тело запроса Contents API: контент всегда base64, `sha` — только при перезаписи."""
    body: dict[str, Any] = {"message": message, "branch": branch, "content": base64.b64encode(content).decode("ascii")}
    if sha:
        body["sha"] = sha
    return json.dumps(body, ensure_ascii=False)


def put(slug: str, path: str, payload: str) -> tuple[int, str]:
    completed = subprocess.run(
        ["gh", "api", "--method", "PUT", f"repos/{slug}/contents/{path}", "--input", "-"],
        input=payload,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode, (completed.stderr or completed.stdout).strip()


def publish(local: Path, slug: str, dest_path: str, branch: str, message: str) -> str:
    """Опубликовать один файл; возвращает вердикт для журнала."""
    data = local.read_bytes()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        remote = current_bytes(slug, dest_path, branch)
        if remote is not None and remote == data:
            return "без изменений"
        sha = existing_sha(slug, dest_path, branch)
        code, output = put(slug, dest_path, build_payload(message=message, branch=branch, content=data, sha=sha))
        if code == 0:
            return "записано"
        # 409/422 — файл успел измениться между чтением и записью: перечитать и повторить.
        if attempt < MAX_ATTEMPTS and ("409" in output or "422" in output):
            time.sleep(2 * attempt)
            continue
        raise SystemExit(f"{dest_path}: не удалось записать ({output[:400]})")
    return "не записано"  # pragma: no cover - недостижимо: выход из цикла либо return, либо SystemExit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--branch", default="data/metrics", help="ветка для записи")
    parser.add_argument("--dest", default="", help="префикс каталога в ветке (например hf-protocol)")
    parser.add_argument("--report", action="append", default=[], help="файл или каталог (повторяемо)")
    parser.add_argument("--message", default="reports: обновление артефактов [skip ci]")
    parser.add_argument("--dry-run", action="store_true", help="показать план и ничего не отправлять")
    parser.add_argument(
        "--keep-tree",
        action="store_true",
        help="класть в ветку с относительным путём; по умолчанию — плоско: dest/имя файла",
    )
    parser.add_argument("--out-dir", default="", help="заодно скопировать файлы в локальный каталог")
    args = parser.parse_args(argv)

    files: list[Path] = []
    for item in args.report:
        path = Path(item)
        if not path.is_absolute():
            path = ROOT / path
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*") if p.is_file()))
        elif path.is_file():
            files.append(path)
        else:
            print(f"пропущено (нет файла): {path}", file=sys.stderr)
    if not files:
        raise SystemExit("нечего публиковать: --report не указал ни одного файла")
    if args.out_dir:
        target = Path(args.out_dir)
        for path in files:
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target / path.name)
        print(f"копия в {target}: файлов {len(files)}")

    # Черновик не отправляет запросов, поэтому ни `gh`, ни адрес репозитория для
    # него не нужны: план можно посмотреть и в дереве без git-удалёнки.
    if args.dry_run:
        slug = "owner/repo"
    else:
        if shutil.which("gh") is None:
            raise SystemExit("нужен `gh` в PATH: публикация идёт через Contents API")
        slug = repo_slug()
    names = [path.name for path in files]
    if not args.keep_tree and len(set(names)) != len(names):
        # Плоская раскладка — как в ветке данных сейчас (`hf-protocol/sweep.json`);
        # два `manifest.json` в один каталог легли бы один поверх другого.
        raise SystemExit(f"в плоской раскладке имена файлов не должны повторяться: {sorted(names)}")
    for path in files:
        relative = path.relative_to(ROOT) if path.is_relative_to(ROOT) else Path(path.name)
        if args.keep_tree:
            parts = list(relative.parts)
            if parts[:1] == ["reports"]:
                parts = parts[1:]
            inner = "/".join(parts)
        else:
            inner = path.name
        dest_path = f"{args.dest}/{inner}" if args.dest else inner
        if args.dry_run:
            print(f"[dry-run] {path} -> {slug}@{args.branch}:{dest_path} ({path.stat().st_size} байт)")
            continue
        verdict = publish(path, slug, dest_path, args.branch, args.message)
        print(f"{dest_path}: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
