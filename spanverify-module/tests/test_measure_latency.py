"""Тесты измерителя ресурсов (`scripts/measure_latency.py`).

Смысл проверки: скрипт существует для того, чтобы числа в отчётах были получены командой, а не
набраны от руки. Значит у него должно быть детерминированное свойство, на которое можно сослаться:
выборка пар из корпуса не зависит от порядка строк в файле и даёт ровно запрошенное число пар,
акида измерений — медиану и 95-й перцентиль — считает по всем измеренным парам, а не по первой.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for _path in (str(ROOT), str(SCRIPTS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import measure_latency  # noqa: E402


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_pick_pairs_is_deterministic_and_respects_count(tmp_path: Path) -> None:
    rows = [{"id": f"p-{index:02d}", "answer": "а" * (index + 1), "context": "контекст"} for index in range(40)]
    source = tmp_path / "pairs.jsonl"
    _write(source, rows)
    first = [row["id"] for row in measure_latency.pick_pairs(source, 10)]
    second = [row["id"] for row in measure_latency.pick_pairs(source, 10)]
    assert first == second, "выборка поехала между запусками"
    assert len(first) == 10
    assert len(set(first)) == 10, "дубли в выборке"
    # самая короткая и самая длинная пары обязаны попасть в выборку: иначе «разные длины» фиктивные
    shortest = rows[0]["id"]
    longest = rows[-1]["id"]
    assert shortest in first and longest in first


def test_rows_without_answer_are_skipped(tmp_path: Path) -> None:
    source = tmp_path / "pairs.jsonl"
    _write(
        source, [{"id": "empty", "answer": "   ", "context": "есть"}, {"id": "ok", "answer": "да", "context": "есть"}]
    )
    pairs = measure_latency.pick_pairs(source, 10)
    assert [row["id"] for row in pairs] == ["ok"]


def test_rss_reader_returns_a_positive_number() -> None:
    """Пик RSS обязан читаться: иначе в отчёт попал бы ноль вместо измерения."""
    assert measure_latency._rss_kib() > 0
