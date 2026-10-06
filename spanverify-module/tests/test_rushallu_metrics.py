"""Тесты перевода метрик RusHallu-RAG на stdlib.

Смысл: внешний бенчмарк оценивается нашим кодом, но их метриками. Значит формулы
обязаны совпадать с оригиналом (``metrics/span_metrics.py`` закреплённого коммита),
иначе сравнение с публикацией некорректно. Здесь проверяется совпадение с
библиотекой ``rouge_score`` (если установлена), разумность формул совпадения и
аккуратная обработка пустой разметки (у них там получается ``nan``).
"""

from __future__ import annotations

import importlib.util
import math

import pytest

from spanverify.rushallu_metrics import (
    calculate_match_metrics,
    calculate_rouge_scores,
    char_tokens,
    evaluate_spans,
    rouge_l,
    rouge_n,
    score_pair,
)

HAS_ROUGE_SCORE = importlib.util.find_spec("rouge_score") is not None


def test_tokenization_is_per_character() -> None:
    """Их токенизатор режет текст по символам — без слов и пунктуации отдельно."""
    assert char_tokens("аб, в") == ["а", "б", ",", " ", "в"]
    assert char_tokens("") == []


def test_rouge_n_counts_repeated_ngrams() -> None:
    """ROUGE-N считается по мультимножествам: повтор символа учитывается один раз."""
    assert rouge_n("ааа", "ааа", 1) == pytest.approx(1.0)
    assert rouge_n("аб", "аб", 1) == pytest.approx(1.0)
    assert rouge_n("абв", "аб", 1) == pytest.approx(2 * (2 / 2) * (2 / 3) / (2 / 2 + 2 / 3))
    assert rouge_n("", "аб") == pytest.approx(0.0)


def test_rouge_l_is_lcs_f_measure() -> None:
    """ROUGE-L — F-measure на длине наибольшей общей подпоследовательности."""
    assert rouge_l("абвгд", "абвгд") == pytest.approx(1.0)
    assert rouge_l("", "абв") == pytest.approx(0.0)
    value = rouge_l("абвгд", "абхгд")
    assert 0.5 < value < 1.0


@pytest.mark.skipif(not HAS_ROUGE_SCORE, reason="библиотека rouge_score не установлена")
def test_rouge_matches_rouge_score_library() -> None:
    """Наши ROUGE совпадают с оригинальной библиотекой (символьная токенизация)."""
    from rouge_score.rouge_scorer import RougeScorer

    class Characters:
        """Токенизатор-заглушка: список символов, как у авторов бенчмарка."""

        def tokenize(self, text: str) -> list[str]:
            """Разбить строку на символы."""
            return list(text)

    scorer = RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False, tokenizer=Characters())
    cases = [
        ("тяжело избежать использования", "тяжело избежать использования потенциально"),
        ("срок пять лет", "срок десять лет"),
        ("короткий", "совсем другое"),
        ("", "непусто"),
    ]
    for reference, predicted in cases:
        ours = score_pair(reference, predicted)
        for metric in ("rouge1", "rouge2", "rougeL"):
            theirs = scorer.score(reference, predicted)[metric].fmeasure
            assert ours[metric] == pytest.approx(theirs, abs=1e-9), (reference, predicted, metric)


def test_match_metrics_known_values() -> None:
    """accuracy / Jaccard / hamming считаются по множествам спанов одной пары."""
    reference = [["спан1", "спан2"], ["другой"]]
    predicted = [["спан1"], ["другой"]]
    result = calculate_match_metrics(predicted, reference)
    assert result["accuracy"] == pytest.approx(0.5)  # первая пара не совпала, вторая да
    assert result["jaccard_score"] == pytest.approx((1 / 2 + 1 / 1) / 2)
    assert result["hamming_loss"] == pytest.approx((1 / 2 + 0 / 1) / 2)
    assert result["empty_reference"] == 0


def test_match_metrics_do_not_return_nan_on_empty_reference() -> None:
    """Пустая разметка не даёт nan: у авторов там деление на ноль, у нас 1.0 и 0.0."""
    result = calculate_match_metrics([["лишний"]], [[]])
    assert result["jaccard_score"] == pytest.approx(0.0) or result["jaccard_score"] == pytest.approx(1.0)
    assert not math.isnan(result["jaccard_score"])
    assert result["hamming_loss"] == pytest.approx(0.0)
    assert result["empty_reference"] == 1


def test_rouge_scores_for_identical_batches_is_one() -> None:
    """Если предсказания совпали с разметкой целиком — ROUGE равен 1.0 (как у них)."""
    batches = [["спан"], ["два", "спана"]]
    result = calculate_rouge_scores(batches, batches)
    assert result["rouge1"] == pytest.approx(1.0)
    assert result["rougeL"] == pytest.approx(1.0)


def test_evaluate_spans_returns_both_groups() -> None:
    """Полный вызов отдаёт ROUGE, метрики совпадения и число пар."""
    result = evaluate_spans([["а"]], [["а"]])
    assert result["pairs"] == 1
    for key in ("rouge1", "rouge2", "rougeL", "accuracy", "jaccard_score", "hamming_loss"):
        assert key in result
