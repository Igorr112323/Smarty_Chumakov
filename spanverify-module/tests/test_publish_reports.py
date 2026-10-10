"""Тесты публикационного помощника `scripts/publish_reports.py`.

Сеть и `gh` не затрагиваются: проверяются форма тела запроса Contents API,
раскладка путей в ветке, отказ на повторяющихся именах и поведение при гонке
двух job'ов (409 → перечитать и повторить). Это тот канал, которым числа
протокола доезжают до проверяющего, поэтому он обязан быть предсказуемым.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("publish_reports_for_test", ROOT / "scripts" / "publish_reports.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _module()


def test_payload_carries_base64_content_and_sha_only_when_replacing() -> None:
    """Тело запроса: контент — base64, `sha` — только при перезаписи существующего файла."""
    body = json.loads(module.build_payload(message="m", branch="data/metrics", content=b"ab\xff", sha=None))
    assert body["branch"] == "data/metrics" and body["message"] == "m"
    assert base64.b64decode(body["content"]) == b"ab\xff", "двоичные файлы (.jsonl.gz) должны проходить как есть"
    assert "sha" not in body, "для нового файла sha указывать нельзя — API ответит 422"
    replaced = json.loads(module.build_payload(message="m", branch="b", content=b"x", sha="deadbeef"))
    assert replaced["sha"] == "deadbeef"


def test_repo_slug_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """В CI репозиторий берётся из окружения — без вызова git."""
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    assert module.repo_slug() == "owner/repo"


def test_repo_slug_from_git_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    """Вне CI — из адреса удалённого репозитория (включая форму с .git)."""
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)

    def fake_run(command: list[str], **_kwargs: Any) -> Any:
        class Result:
            stdout = "https://github.com/Igorr112323/Smarty_Chumakov.git\n"
            returncode = 0

        assert command[:3] == ["git", "-C", str(ROOT)]
        return Result()

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module.repo_slug() == "Igorr112323/Smarty_Chumakov"


def test_dry_run_paths(tmp_path: Path, capsys: pytest.CaptureContext[str]) -> None:
    """Плоская раскладка (`dest/имя`) и `--keep-tree` (путь относительно корня модуля)."""
    reports = tmp_path / "spanverify-module" / "reports" / "hf_protocol"
    reports.mkdir(parents=True)
    (reports / "sweep.json").write_text(json.dumps({"experiments": []}), encoding="utf-8")
    # dry-run ничего не отправляет — проверяем по напечатанному плану
    module.ROOT = tmp_path / "spanverify-module"
    try:
        module.main(["--dry-run", "--dest", "hf-protocol", "--report", str(reports / "sweep.json")])
    finally:
        module.ROOT = ROOT
    printed = capsys.readouterr().out
    assert "hf-protocol/sweep.json" in printed, printed


def test_duplicate_basenames_are_rejected(tmp_path: Path) -> None:
    """Два `manifest.json` в один каталог ветки данных — тихая потеря файла."""
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()
    (first / "manifest.json").write_text("{}", encoding="utf-8")
    (second / "manifest.json").write_text("{}", encoding="utf-8")
    saved = module.ROOT
    module.ROOT = tmp_path
    try:
        with pytest.raises(SystemExit, match="не должны повторяться"):
            module.main(
                [
                    "--dry-run",
                    "--dest",
                    "hf",
                    "--report",
                    str(first / "manifest.json"),
                    "--report",
                    str(second / "manifest.json"),
                ]
            )
    finally:
        module.ROOT = saved


def test_unchanged_file_is_not_committed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Содержимое совпадает — коммита нет: история ветки данных остаётся осмысленной."""
    target = tmp_path / "sweep.json"
    target.write_text('{"a": 1}', encoding="utf-8")
    monkeypatch.setattr(module, "current_bytes", lambda *args: b'{"a": 1}')
    monkeypatch.setattr(module, "put", lambda *args, **kwargs: pytest.fail("запроса быть не должно"))
    assert module.publish(target, "o/r", "hf-protocol/sweep.json", "data/metrics", "m") == "без изменений"


def test_conflict_is_retried_after_reread(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """409 (файл изменился между чтением и записью) — перечитать sha и повторить."""
    target = tmp_path / "sweep.json"
    target.write_text("новое", encoding="utf-8")
    seen: list[tuple[Any, ...]] = []

    def fake_existing_sha(*args: Any) -> str:
        seen.append(args)
        return f"sha{len(seen)}"

    monkeypatch.setattr(module, "current_bytes", lambda *args: b"chuzhoe")
    monkeypatch.setattr(module, "existing_sha", fake_existing_sha)
    answers = iter([(1, "HTTP 409: sha mismatch"), (0, "{}")])
    monkeypatch.setattr(module, "put", lambda *args, **kwargs: next(answers))
    monkeypatch.setattr(module.time, "sleep", lambda _delay: None)
    assert module.publish(target, "o/r", "hf-protocol/sweep.json", "data/metrics", "m") == "записано"
    assert len(seen) == 2, "второй заход обязан перечитать sha после 409"
    assert all(args[1] == "hf-protocol/sweep.json" and args[2] == "data/metrics" for args in seen)


def test_hard_failure_is_reported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ошибка, которую нельзя разрешить повтором, — SystemExit с текстом ответа."""
    target = tmp_path / "sweep.json"
    target.write_text("x", encoding="utf-8")
    monkeypatch.setattr(module, "current_bytes", lambda *args: None)
    monkeypatch.setattr(module, "existing_sha", lambda *args: None)
    monkeypatch.setattr(module, "put", lambda *args, **kwargs: (1, "HTTP 403: Resource not accessible by integration"))
    monkeypatch.setattr(module.time, "sleep", lambda _delay: None)
    with pytest.raises(SystemExit, match="403"):
        module.publish(target, "o/r", "hf-protocol/sweep.json", "data/metrics", "m")
