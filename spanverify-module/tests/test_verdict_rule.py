"""Правило вердикта: одиночный токен маски не обвиняет весь ответ.

Причина проверки: историческое правило «любой фрагмент — ответ спорный» на
корпусе A3 (прогон 37532050295) пометило 1047 ответов из 1200 (вердикт
FPR 0.893) — один случайный токен маски делал спорным весь ответ. Новое
правило требует хотя бы два помеченных токена маски, но фрагменты текстовых
правил (привязка числа к объекту) по-прежнему решают и в одиночку.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spanverify.core import SpanResult
from spanverify.dataset import read_pairs
from spanverify.engine import (
    VERDICT_ANY_SPAN,
    VERDICT_MIN_TWO_TOKENS,
    Verifier,
    _verdict,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "demo_pairs.jsonl"


def _span(n_tokens: int, source: str, risk: float = 0.5) -> SpanResult:
    return SpanResult(start=0, end=10, text="фрагмент", risk=risk, label="doubtful", n_tokens=n_tokens, source=source)


def test_single_mask_token_does_not_flip_verdict() -> None:
    """Один помеченный токен маски — ещё не основание считать ответ спорным."""
    assert _verdict(0.1, 0.5, [_span(1, "mask")], VERDICT_MIN_TWO_TOKENS) == "grounded"


def test_two_mask_tokens_flip_verdict() -> None:
    """Два помеченных токена маски уже делают ответ спорным."""
    assert _verdict(0.1, 0.5, [_span(2, "mask")], VERDICT_MIN_TWO_TOKENS) == "doubtful"
    assert _verdict(0.1, 0.5, [_span(1, "mask"), _span(1, "mask")], VERDICT_MIN_TWO_TOKENS) == "doubtful"


def test_attribution_span_is_decisive_alone() -> None:
    """Фрагмент правила привязки числа решает даже в одиночку (дефект D)."""
    assert _verdict(0.1, 0.5, [_span(1, "attribution")], VERDICT_MIN_TWO_TOKENS) == "doubtful"
    # Склейка маски и правила наследует решающий статус.
    assert _verdict(0.1, 0.5, [_span(1, "rule")], VERDICT_MIN_TWO_TOKENS) == "doubtful"


def test_score_above_threshold_still_means_hallucination() -> None:
    """Скор ответа выше порога — недостоверно независимо от числа токенов."""
    assert _verdict(0.9, 0.5, [], VERDICT_MIN_TWO_TOKENS) == "likely_hallucination"


def test_legacy_rule_reproduces_old_behaviour() -> None:
    """Старое правило (для сравнения до/после) помечает любой фрагмент."""
    assert _verdict(0.1, 0.5, [_span(1, "mask")], VERDICT_ANY_SPAN) == "doubtful"
    assert _verdict(0.1, 0.5, [], VERDICT_ANY_SPAN) == "grounded"


def test_unknown_rule_is_rejected() -> None:
    """Неизвестное правило вердикта — ошибка, а не молчаливое поведение."""
    with pytest.raises(ValueError):
        Verifier(mode="demo", verdict_rule="как-нибудь")


def _pair_00014() -> dict:
    if not DATA.is_file():
        pytest.skip("нет демо-корпуса")
    for pair in read_pairs(DATA):
        if pair["id"] == "pair-00014":
            return pair
    pytest.skip("pair-00014 отсутствует в демо-корпусе")


def test_demo_pair_single_token_verdict_before_after() -> None:
    """Реальный случай из демо-корпуса: один токен маски.

    Старое правило делает ответ спорным, новое — нет; скор ответа ниже порога,
    поэтому решение зависит именно от числа помеченных токенов.
    """
    pair = _pair_00014()
    legacy = Verifier(mode="demo", verdict_rule=VERDICT_ANY_SPAN)
    current = Verifier(mode="demo", verdict_rule=VERDICT_MIN_TWO_TOKENS)
    legacy_result = legacy.verify(pair["answer"], pair["context"], with_tokens=True)
    current_result = current.verify(pair["answer"], pair["context"], with_tokens=True)
    flagged = [token for token in current_result.tokens if token.get("flagged")]
    assert len(flagged) == 1, "кейс обязан быть ровно с одним помеченным токеном маски"
    assert current_result.score < current_result.threshold
    assert legacy_result.verdict == "doubtful"
    assert current_result.verdict == "grounded"
    # Токенная маска от правила вердикта не меняется.
    assert [token["risk"] for token in legacy_result.tokens] == [token["risk"] for token in current_result.tokens]


def test_evaluate_tokens_identical_under_both_rules() -> None:
    """В оценке до/после токены и фрагменты совпадают, различаются только вердикты."""
    pair = _pair_00014()
    legacy = Verifier(mode="demo", verdict_rule=VERDICT_ANY_SPAN).evaluate([pair])
    current = Verifier(mode="demo").evaluate([pair])
    for key in ("tp", "fp", "fn", "tn", "f1", "fpr"):
        assert legacy["tokens"][key] == current["tokens"][key], key
    assert (
        legacy["verdicts"]["fpr"] != current["verdicts"]["fpr"] or legacy["verdicts"]["f1"] != current["verdicts"]["f1"]
    ), "правила обязаны различаться хотя бы на одном вердикте этого кейса"
