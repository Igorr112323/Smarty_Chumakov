"""Тесты сервера ручной проверки корпуса.

Сервер пишет файл решений человека — то есть влияет на то, что попадёт в отчёт о
качестве корпуса. Поэтому проверяется не «страница открывается», а: решения
дописываются и не теряются, чужой id и неверный вердикт отвергаются, прогресс
считается по последнему решению, а выборка берётся случайно с заданным seed.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from scripts.review_server import decisions_index, load_jsonl, make_handler, make_sample

PAIRS = [
    {
        "id": "corpus-a-00001",
        "context": "Для первичных документов срок хранения составляет пять лет.",
        "answer": "Срок хранения первичных документов — десять лет.",
        "labels": [[31, 41, 1]],
        "meta": {"mode": "number_attribution", "taxonomy": "Contradiction"},
    },
    {
        "id": "corpus-a-00002",
        "context": "Для вторичных документов срок хранения составляет десять лет.",
        "answer": "Срок хранения вторичных документов — десять лет.",
        "labels": [],
        "meta": {"mode": "faithful", "taxonomy": "faithful"},
    },
]


@pytest.fixture
def server(tmp_path: Path):
    """Поднять сервер проверки в отдельном потоке на свободном порту."""
    queue = tmp_path / "queue.jsonl"
    queue.write_text("".join(json.dumps(pair, ensure_ascii=False) + "\n" for pair in PAIRS), encoding="utf-8")
    out = tmp_path / "decisions.jsonl"
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(queue, out))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", out
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _post(url: str, payload: dict) -> dict:
    """Отправить решение на сервер и вернуть ответ как JSON."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310 - адрес своего же сервера
        return json.loads(response.read().decode("utf-8"))


def test_page_shows_first_pending_pair(server) -> None:
    """Страница показывает первую непроверенную пару и прогресс."""
    base, _out = server
    with urllib.request.urlopen(f"{base}/", timeout=10) as response:  # noqa: S310
        page = response.read().decode("utf-8")
    assert "corpus-a-00001" in page
    assert "осталось 2" in page
    assert "<mark>" in page, "размеченный фрагмент должен быть подсвечен"


def test_decision_is_appended_and_next_pair_shown(server) -> None:
    """После решения пара уходит из очереди, а решение сохраняется в файл."""
    base, out = server
    saved = _post(f"{base}/decision", {"id": "corpus-a-00001", "verdict": "ok", "comment": "метка верна"})
    assert saved["saved"]["verdict"] == "ok"
    assert saved["left"] == 1
    assert decisions_index(out)["corpus-a-00001"]["comment"] == "метка верна"
    with urllib.request.urlopen(f"{base}/", timeout=10) as response:  # noqa: S310
        page = response.read().decode("utf-8")
    assert "corpus-a-00002" in page


def test_stats_count_decisions(server) -> None:
    """Статистика считает проверенные, оставшиеся и разбивку по решениям."""
    base, _out = server
    _post(f"{base}/decision", {"id": "corpus-a-00001", "verdict": "wrong"})
    _post(f"{base}/decision", {"id": "corpus-a-00002", "verdict": "skip"})
    with urllib.request.urlopen(f"{base}/api/stats", timeout=10) as response:  # noqa: S310
        stats = json.loads(response.read().decode("utf-8"))
    assert stats == {"queue": 2, "checked": 2, "left": 0, "decisions": {"wrong": 1, "skip": 1}}


def test_unknown_id_and_bad_verdict_are_rejected(server) -> None:
    """Чужой id и недопустимый вердикт не попадают в файл решений."""
    base, out = server
    with pytest.raises(urllib.error.HTTPError) as unknown:
        _post(f"{base}/decision", {"id": "нет-такой", "verdict": "ok"})
    assert unknown.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as bad:
        _post(f"{base}/decision", {"id": "corpus-a-00001", "verdict": "наверное"})
    assert bad.value.code == 400
    assert not out.exists(), "после отвергнутых решений файл создавать не нужно"


def test_second_decision_overwrites_in_index(tmp_path: Path) -> None:
    """Перепроверка пары заменяет прежнее решение, а не удваивает его."""
    path = tmp_path / "decisions.jsonl"
    path.write_text('{"id": "p1", "verdict": "ok"}\n{"id": "p1", "verdict": "wrong"}\n', encoding="utf-8")
    index = decisions_index(path)
    assert len(index) == 1
    assert index["p1"]["verdict"] == "wrong"
    assert len(load_jsonl(path)) == 2, "обе записи остаются в файле как история"


def test_make_sample_is_deterministic_and_takes_share(tmp_path: Path) -> None:
    """Выборка на проверку детерминирована и берёт заданную долю пар."""
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(json.dumps({"id": f"p{index}"}, ensure_ascii=False) + "\n" for index in range(100)),
        encoding="utf-8",
    )
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    count = make_sample(source, first, share=0.10, seed=42)
    make_sample(source, second, share=0.10, seed=42)
    assert count == 10
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


def test_make_sample_fails_on_empty_source(tmp_path: Path) -> None:
    """Пустой источник — понятная ошибка, а не пустая очередь."""
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="нет пар"):
        make_sample(empty, tmp_path / "queue.jsonl", share=0.10, seed=1)
