"""Дефект E (N1): чужой формат датасета должен отвергаться, а не считаться молча.

Проверяющий воспроизвёл на релизе v1.2.0: файл с полями ``question/label`` вместо
``context/labels`` принимался, метрики печатались как ``F1 0.000, AUC=nan``, код
возврата был 0. Эти тесты падают на v1.2.0 и проходят после валидации схемы.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spanverify.cli import EXIT_ERROR, EXIT_OK, main
from spanverify.dataset import read_pairs


def _format_error():
    """Класс ошибки схемы (на v1.2.0 его ещё нет — тест падает по существу)."""
    from spanverify.dataset import DatasetFormatError

    return DatasetFormatError


FOREIGN_LINES = (
    '{"id":"x1","question":"тест","answer":"тест","label":1}',
    '{"id":"x2","question":"тест","answer":"тест","label":0}',
)


def _write(path: Path, lines: tuple[str, ...]) -> Path:
    """Записать JSONL-файл из готовых строк."""
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_foreign_schema_is_rejected(tmp_path: Path) -> None:
    """Чужой формат: ошибка схемы с перечнем отсутствующих полей и примером строки."""
    path = _write(tmp_path / "foreign.jsonl", FOREIGN_LINES)
    with pytest.raises(_format_error()) as excinfo:
        list(read_pairs(path))
    message = str(excinfo.value)
    assert "context" in message
    assert "labels" in message
    assert "строк" in message or "строка" in message
    assert "Неизвестные поля" in message or "question" in message
    assert "example" in message.lower() or "пример" in message.lower()


def test_missing_required_key_reports_line_number(tmp_path: Path) -> None:
    """Нет обязательного ключа ``context`` — в сообщении есть номер строки."""
    path = _write(
        tmp_path / "broken.jsonl",
        (
            '{"id":"ok","context":"Регламент: срок 10 лет.","answer":"Срок 10 лет.","labels":[[6,8,1]]}',
            '{"id":"bad","answer":"Срок 10 лет.","labels":[]}',
        ),
    )
    with pytest.raises(_format_error()) as excinfo:
        list(read_pairs(path))
    assert "строк" in str(excinfo.value)
    assert "2" in str(excinfo.value)


def test_label_bounds_are_checked(tmp_path: Path) -> None:
    """Метка за границами ответа — ошибка схемы, а не молчаливый ноль в метриках."""
    path = _write(
        tmp_path / "bounds.jsonl",
        ('{"id":"p","context":"Регламент: срок 10 лет.","answer":"Срок 10 лет.","labels":[[0,999,1]]}',),
    )
    with pytest.raises(_format_error()) as excinfo:
        list(read_pairs(path))
    assert "границ" in str(excinfo.value).lower() or "end" in str(excinfo.value)


def test_broken_json_reports_line_number(tmp_path: Path) -> None:
    """Битая JSON-строка — ошибка с номером строки, а не трассировка парсера."""
    path = _write(
        tmp_path / "json.jsonl",
        (
            '{"id":"ok","context":"Регламент: срок 10 лет.","answer":"Срок 10 лет.","labels":[]}',
            '{"id":"bad", ',
        ),
    )
    with pytest.raises(_format_error()) as excinfo:
        list(read_pairs(path))
    assert "2" in str(excinfo.value)


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    """Пустой файл — ошибка: считать метрики не на чем."""
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(_format_error()):
        list(read_pairs(path))


def test_valid_corpus_still_reads(tmp_path: Path) -> None:
    """Корректный файл читается без изменений (совместимость со старым форматом)."""
    path = _write(
        tmp_path / "ok.jsonl",
        (
            '{"id":"p","context":"Регламент: срок 10 лет.","answer":"Срок 10 лет.","labels":[[6,8,1]],"meta":{"kind":"x"}}',
        ),
    )
    pairs = list(read_pairs(path))
    assert len(pairs) == 1
    assert pairs[0]["answer"] == "Срок 10 лет."


def test_cli_evaluate_rejects_foreign_dataset(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """CLI: чужой формат → код 2 и понятное сообщение (а не F1 0.000 с кодом 0)."""
    path = _write(tmp_path / "foreign.jsonl", FOREIGN_LINES)
    code = main(["evaluate", "--dataset", str(path)])
    captured = capsys.readouterr()
    assert code == EXIT_ERROR, code
    assert "ожидалось" in captured.err.lower() or "схема" in captured.err.lower()


def test_cli_evaluate_accepts_valid_dataset(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """CLI: корректный формат → код 0 и метрики как раньше."""
    path = _write(
        tmp_path / "ok.jsonl",
        (
            '{"id":"p1","context":"Регламент 343: срок хранения документов составляет 10 лет.",'
            '"answer":"Срок хранения документов составляет 10 лет.","labels":[],"meta":{}}',
            '{"id":"p2","context":"Регламент 343: срок хранения документов составляет 10 лет.",'
            '"answer":"Срок хранения документов составляет 3 года.","labels":[[33,39,1]],"meta":{}}',
        ),
    )
    code = main(["evaluate", "--dataset", str(path)])
    captured = capsys.readouterr()
    assert code == EXIT_OK, captured.err
    assert "Токены:" in captured.out


def test_cli_evaluate_json_has_no_nan(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """В JSON-выводе нет NaN: по RFC 8259 это невалидный JSON."""
    path = _write(
        tmp_path / "ok.jsonl",
        (
            '{"id":"p1","context":"Регламент: срок 10 лет.","answer":"Срок 10 лет.","labels":[],"meta":{}}',
            '{"id":"p2","context":"Регламент: срок 10 лет.","answer":"Срок 3 года.","labels":[[6,11,1]],"meta":{}}',
        ),
    )
    code = main(["evaluate", "--dataset", str(path), "--json"])
    captured = capsys.readouterr()
    assert code == EXIT_OK, captured.err
    assert "NaN" not in captured.out and "nan" not in captured.out
    payload = json.loads(captured.out)
    assert payload["tokens"]["auc"] is None or isinstance(payload["tokens"]["auc"], float)


def test_zipapp_propagates_exit_codes(tmp_path: Path) -> None:
    """Релизный .pyz возвращает 0 / 1 / 2, а не всегда 0 (дефект E на практике).

    Проверяющий запускал команды воспроизведения именно на релизном `.pyz`, и там
    код возврата терялся: zipapp собирался с точкой входа ``cli:main`` вместо
    ``cli:run``, поэтому даже отказ схемы давал код 0. Тест собирает архив и
    проверяет все три исхода на реальном файле.
    """
    import subprocess
    import sys

    from scripts.make_zipapp import build

    pyz = build(tmp_path / "release", name="spanverify-test.pyz")

    foreign = _write(tmp_path / "foreign.jsonl", FOREIGN_LINES)
    doubtful = "Срок хранения первичных документов составляет десять лет."
    context = (
        "Согласно регламенту, срок хранения первичных документов составляет пять лет. "
        "Срок хранения вторичных документов составляет десять лет."
    )
    valid = _write(
        tmp_path / "ok.jsonl",
        ('{"id":"p","context":"Регламент: срок 10 лет.","answer":"Срок 10 лет.",' '"labels":[[6,8,1]],"meta":{}}',),
    )

    def run(*args: str) -> int:
        return subprocess.run([sys.executable, str(pyz), *args], capture_output=True, text=True, timeout=120).returncode

    assert run("evaluate", "--dataset", str(foreign)) == EXIT_ERROR
    assert run("verify", "--answer", doubtful, "--context", context) == 1
    assert run("evaluate", "--dataset", str(valid)) == EXIT_OK
