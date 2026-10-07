"""Фильтр коротких фрагментов маски (вариант сравнения маски).

Гипотеза из разбора чистых пар (``data/metrics/clean_false_flags_a3.md``,
кеш прогона 37609202780): ложные пометки — длинные канцелярские обороты и
заголовки, а не одиночные токены. Вариант маски: фрагмент короче порога
знаков, в котором нет числа, даты или денежной суммы, не помечается. Здесь
проверяется только механика фильтра в движке и в подборе на обучающей части;
порог фиксирован снаружи и по тесту не подбирается.
"""

from __future__ import annotations

from spanverify.core import SpanResult, tokenize_with_offsets
from spanverify.engine import Verifier, WeightsBundle, _token_payload
from spanverify.features import FeatureMatrix
from spanverify.normalize import contains_number_date_or_amount
from spanverify.train import TokenSample, _filter_short_flagged_groups, _score_pairs


def _bundle() -> WeightsBundle:
    """Бандл, в котором риск даёт только энтропия внимания."""
    return WeightsBundle(
        weights={"attention_entropy": 1.0, "ctx_attention_mass": 0.0, "embedding_density": 0.0},
        threshold=0.9,
        span_z=0.0,
        span_floor=0.3,
        span_cap=0.5,
        mode="demo",
    )


def _verifier(span_filter_min_chars: int, monkeypatch, answer: str, hot: set[int]) -> Verifier:
    """Верификатор с подменёнными признаками: риск 1.0 на токенах ``hot``."""
    verifier = Verifier(mode="demo", weights=_bundle(), span_filter_min_chars=span_filter_min_chars)
    tokens = tokenize_with_offsets(answer)
    features = FeatureMatrix(
        attention_entropy=[1.0 if i in hot else 0.0 for i in range(len(tokens))],
        ctx_attention_mass=[0.0] * len(tokens),
        embedding_density=[0.0] * len(tokens),
    )
    monkeypatch.setattr(verifier, "_features", lambda *args, **kwargs: features)
    return verifier


# ------------------------------------------------------------- распознаватель


def test_contains_number_date_or_amount():
    """Число, дата и сумма распознаются; чистый текст — нет."""
    assert contains_number_date_or_amount("5")
    assert contains_number_date_or_amount("01.01.2026")
    assert contains_number_date_or_amount("1 250 000 рублей")
    assert contains_number_date_or_amount("код целевой статьи 1250065420")
    assert not contains_number_date_or_amount("Институт языка")
    assert not contains_number_date_or_amount("пункт изложить в новой редакции")
    assert not contains_number_date_or_amount("")


# -------------------------------------------------------------------- движок


def test_filter_short_mask_spans_drops_short_text_without_numbers():
    """Короткий фрагмент без числовой записи снимается; фрагмент с числом — нет."""
    verifier = Verifier(mode="demo", weights=_bundle(), span_filter_min_chars=40)
    short_text_only = SpanResult(
        start=0,
        end=14,
        text="Институт языка",
        risk=0.9,
        label="doubtful",
        n_tokens=2,
        source="mask",
        token_indices=(3, 4),
    )
    short_with_number = SpanResult(
        start=20,
        end=37,
        text="на 2023-2030 годы",
        risk=0.9,
        label="doubtful",
        n_tokens=3,
        source="mask",
        token_indices=(6, 7, 8),
    )
    long_text_only = SpanResult(
        start=40,
        end=120,
        text="ведение архива технической документации на объекты государственного учета",
        risk=0.9,
        label="doubtful",
        n_tokens=8,
        source="mask",
        token_indices=(10, 11, 12, 13, 14, 15, 16, 17),
    )
    kept, kept_indices = verifier._filter_short_mask_spans([short_text_only, short_with_number, long_text_only])
    assert [span.text for span in kept] == [
        "на 2023-2030 годы",
        "ведение архива технической документации на объекты государственного учета",
    ]
    assert kept_indices == {6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17}


def test_verify_filter_removes_short_span_and_token_flags(monkeypatch):
    """С фильтром короткий фрагмент без числа не помечается: ни спан, ни токены."""
    answer = "Альфа Бета Гамма Дельта Эпсилон Зета"
    plain = _verifier(0, monkeypatch, answer, hot={1, 2})
    result_plain = plain.verify(answer, with_tokens=True)
    assert result_plain.spans, "без фильтра фрагмент должен остаться"
    flagged_plain = [row["index"] for row in result_plain.tokens if row["flagged"]]
    assert 1 in flagged_plain and 2 in flagged_plain

    filtered = _verifier(40, monkeypatch, answer, hot={1, 2})
    result_filtered = filtered.verify(answer, with_tokens=True)
    assert result_filtered.spans == []
    flagged_filtered = [row["index"] for row in result_filtered.tokens if row["flagged"]]
    assert flagged_filtered == []
    assert result_filtered.verdict == "grounded"


def test_verify_filter_keeps_short_span_with_number(monkeypatch):
    """Короткий фрагмент с числом фильтр не снимает."""
    answer = "Альфа Бета 2023 Гамма Дельта Эпсилон"
    filtered = _verifier(40, monkeypatch, answer, hot={1, 2, 3})
    result = filtered.verify(answer, with_tokens=True)
    assert result.spans, "фрагмент с числом обязан остаться"
    assert "2023" in result.spans[0].text


def test_token_payload_flagged_override():
    """Переопределение пометок токенов работает независимо от риска."""
    answer = "Альфа Бета Гамма"
    tokens = tokenize_with_offsets(answer)
    features = FeatureMatrix(
        attention_entropy=[0.5] * len(tokens),
        ctx_attention_mass=[0.0] * len(tokens),
        embedding_density=[0.0] * len(tokens),
    )
    rows = _token_payload(tokens, features, [0.5] * len(tokens), 0.3, flagged_override={1})
    assert [row["flagged"] for row in rows] == [False, True, False]
    assert [row["label"] for row in rows] == ["ok", "doubtful", "ok"]


# ------------------------------------------------------------- сторона выбора


def _samples(answer: str, hot: set[int]) -> list[TokenSample]:
    """Сэмплы всех токенов ответа: риск 1.0 на ``hot``, смещения реальные."""
    tokens = tokenize_with_offsets(answer)
    samples = []
    for index, token in enumerate(tokens):
        samples.append(
            TokenSample(
                pair_id="t1",
                index=index,
                total=len(tokens),
                text=token.text,
                features={
                    "attention_entropy": 1.0 if index in hot else 0.0,
                    "ctx_attention_mass": 0.0,
                    "embedding_density": 0.0,
                    "risk": 1.0 if index in hot else 0.0,
                },
                label=0,
                start=token.start,
                end=token.end,
            )
        )
    return samples


def test_filter_short_flagged_groups_directly():
    """Группы: короткая без числа снимается, с числом и длинная — остаются."""
    answer = "Альфа Бета Гамма Дельта 2023 Эпсилон"
    samples = _samples(answer, hot=set())
    by_index = {sample.index: sample for sample in samples}
    kept = _filter_short_flagged_groups({1, 2, 4, 5}, by_index, answer, 40)
    assert 1 not in kept and 2 not in kept, "короткая группа без числа снимается"
    assert 4 in kept and 5 in kept, "группа с числом остаётся"


def test_score_pairs_filter_unflags_short_group():
    """Подбор на обучающей части видит тот же фильтр, что движок."""
    answer = "Альфа Бета Гамма Дельта 2023 Эпсилон"
    verifier_off = Verifier(mode="demo", weights=_bundle(), span_filter_min_chars=0)
    verifier_on = Verifier(mode="demo", weights=_bundle(), span_filter_min_chars=40)
    samples = _samples(answer, hot={1, 2, 4, 5})
    pair = {"id": "t1", "answer": answer, "context": "", "labels": []}
    risk_fn = lambda group: [sample.features["risk"] for sample in group]  # noqa: E731

    metrics_off = _score_pairs(verifier_off, [pair], risk_fn, (0.0, 0.3, 0.5), {"t1": samples})
    metrics_on = _score_pairs(verifier_on, [pair], risk_fn, (0.0, 0.3, 0.5), {"t1": samples})
    # Без фильтра помечены все четыре токена двух групп; с фильтром группа
    # «Бета Гамма» (без числа, короче 40 знаков) снята, группа с «2023» — нет.
    assert metrics_off["fp"] == 4
    assert metrics_on["fp"] == 2


def test_score_pairs_filter_keeps_long_group_without_numbers():
    """Длинная группа без числа остаётся помеченной и при фильтре."""
    answer = "ведение архива технической документации на объекты государственного учета жилищного фонда"
    verifier = Verifier(mode="demo", weights=_bundle(), span_filter_min_chars=40)
    samples = _samples(answer, hot={0, 1, 2, 3, 4, 5, 6, 7})
    pair = {"id": "t1", "answer": answer, "context": "", "labels": []}
    risk_fn = lambda group: [sample.features["risk"] for sample in group]  # noqa: E731
    metrics = _score_pairs(verifier, [pair], risk_fn, (0.0, 0.3, 0.5), {"t1": samples})
    assert metrics["fp"] == 8
