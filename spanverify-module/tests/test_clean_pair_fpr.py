"""Доля ложных замечаний на корректных ответах (дефект B4 реестра).

Смысл проверки: программа, которая обвиняет корректный ответ, хуже программы,
которая ничего не находит. Обвинение без основания подрывает доверие к каждому
выданному фрагменту, поэтому доля ложных замечаний на чистых парах — отдельный
показатель, а не побочный продукт точности.

Критерий «готово» из реестра: доля помеченных чистых пар ≤ 0,10 без подбора
порога по тестовой части. Проверяется на отложенной части корпуса A.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spanverify.engine import Verifier

SPLIT = Path(__file__).resolve().parents[1] / "data" / "corpus_a" / "splits" / "test.jsonl"
MAX_FALSE_MARK_RATE = 0.10


def _clean_pairs() -> list[tuple[str, str]]:
    """Пары без расхождений: ответ полностью следует документу."""
    if not SPLIT.is_file():
        pytest.skip(f"нет файла разбиения: {SPLIT}")
    pairs: list[tuple[str, str]] = []
    with SPLIT.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("meta", {}).get("mode") == "faithful":
                pairs.append((record["answer"], record["context"]))
    return pairs


def test_clean_split_is_not_empty() -> None:
    """Проверка имеет смысл, только если чистые пары действительно есть."""
    assert len(_clean_pairs()) >= 20, "чистых пар слишком мало для вывода о доле"


def test_false_mark_rate_on_clean_pairs_is_low() -> None:
    """Корректный ответ не получает замечаний сверх допустимой доли."""
    verifier = Verifier()
    pairs = _clean_pairs()
    flagged = sum(1 for answer, context in pairs if verifier.verify(answer, context).verdict != "grounded")
    rate = flagged / len(pairs)
    assert rate <= MAX_FALSE_MARK_RATE, (
        f"ложных замечаний {flagged} из {len(pairs)} (доля {rate:.3f}), "
        f"допустимо не более {MAX_FALSE_MARK_RATE:.2f}"
    )


def test_clean_pairs_have_no_spans() -> None:
    """Корректный ответ не содержит ни одного спорного фрагмента."""
    verifier = Verifier()
    with_spans = [answer for answer, context in _clean_pairs() if verifier.verify(answer, context).spans]
    assert not with_spans, f"фрагменты выданы на {len(with_spans)} корректных ответах: {with_spans[:2]}"
