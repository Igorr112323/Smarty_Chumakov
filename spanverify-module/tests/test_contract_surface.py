"""Поверхность требований договора: признаки, лимиты, критерии, уверенность.

Эти тесты проверяют то, что дальше проверяет контрактный прогон
(``scripts/contract_check.py``), но на уровне модуля — быстро и без сети:

* ТЗ п. 4.1 треб. 2 — у фрагмента есть номера символов, метка и уверенность 0…1;
* ТЗ п. 4.1 треб. 3 — ровно пять именованных признаков на токен, числа конечны;
* ТЗ п. 4.1 треб. 1 — лимит 2 000 000 знаков объявлен и исполняется;
* ТЗ п. 4.1 треб. 6 — критерии считаются по одному источнику (``spanverify.gates``);
* ТЗ п. 4.1 треб. 8 — HTTP-маршрут дампа признаков отвечает пятью признаками.
"""

from __future__ import annotations

import json
import math
import threading
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

import pytest

from spanverify.cli import main
from spanverify.core import SpanResult
from spanverify.engine import Verifier
from spanverify.features import (
    DEMO_DENSITY_REFERENCE,
    FEATURE_NAMES,
    combine,
    feature_vectors,
)
from spanverify.gates import CONTRACT_GATE, INTERNAL_TARGET, gate_report
from spanverify.limits import (
    MAX_REQUEST_BYTES,
    MAX_TEXT_CHARS,
    TextTooLongError,
    check_text_length,
)
from spanverify.server import MAX_BODY, Service, make_handler

ANSWER = "Срок хранения первичных документов составляет 3 года."
CONTEXT = "Регламент 343: срок хранения первичных документов составляет 10 лет."


# --------------------------------------------------------------- признаки


def test_exactly_five_named_features_per_token() -> None:
    """ТЗ п. 4.1 треб. 3: ровно пять признаков на токен, все значения конечны."""
    assert len(FEATURE_NAMES) == 5
    verifier = Verifier(mode="demo")
    result = verifier.verify(ANSWER, CONTEXT, with_tokens=True)
    assert result.tokens, "дамп токенов пуст"
    for row in result.tokens:
        for name in FEATURE_NAMES:
            assert name in row, f"нет признака {name}"
            value = float(row[name])
            assert math.isfinite(value), f"{name} = {value}"
            assert value >= 0.0


def test_feature_vectors_cover_all_five_names() -> None:
    """Единая точка выдачи признаков возвращает все пять имён и одну длину."""
    verifier = Verifier(mode="demo")
    matrix = verifier.features_for(ANSWER, CONTEXT)
    vectors = feature_vectors(matrix)
    assert set(vectors) == set(FEATURE_NAMES)
    lengths = {len(values) for values in vectors.values()}
    assert lengths == {len(matrix)}


def test_demo_density_scale_reference_is_positive() -> None:
    """Нормировка шкалы плотности объявлена и положительна."""
    assert 0.0 < DEMO_DENSITY_REFERENCE < 1.0


def test_combine_keeps_five_feature_signs() -> None:
    """Признаки опоры снижают риск, энтропия — повышает (проверка знаков)."""
    supported = combine([0.0], [1.0], [1.0], mass_norm=[1.0], max_similarity=[1.0])[0]
    unsupported = combine([0.0], [0.0], [0.0], mass_norm=[0.0], max_similarity=[0.0])[0]
    assert supported < unsupported
    assert combine([1.0], [1.0], [1.0], mass_norm=[1.0], max_similarity=[1.0])[0] > supported


# --------------------------------------------------------------- фрагменты


def test_span_has_offsets_label_and_confidence() -> None:
    """ТЗ п. 4.1 треб. 2: границы, метка и уверенность 0…1 у фрагмента."""
    verifier = Verifier(mode="demo")
    result = verifier.verify(ANSWER, CONTEXT)
    assert result.spans, "подмена числа не найдена"
    for span in result.spans:
        payload = span.to_dict()
        assert 0 <= payload["start"] < payload["end"] <= len(ANSWER)
        assert ANSWER[span.start : span.end] == span.text
        assert payload["label"] in {"likely_hallucination", "doubtful"}
        assert 0.0 <= payload["risk"] <= 1.0
        assert 0.0 <= payload["confidence"] <= 1.0
        assert payload["confidence"] == pytest.approx(round(1.0 - payload["risk"], 4))


def test_span_confidence_is_complement_of_risk() -> None:
    """Уверенность — производная риска, а не второй независимый скор."""
    assert SpanResult(start=0, end=1, text=",", risk=0.25, label="doubtful").confidence == 0.75
    assert SpanResult(start=0, end=1, text=",", risk=2.0, label="doubtful").confidence == 0.0
    assert SpanResult(start=0, end=1, text=",", risk=-1.0, label="doubtful").confidence == 1.0


# --------------------------------------------------------------- лимиты


def test_text_limit_is_declared_and_enforced() -> None:
    """ТЗ п. 4.1 треб. 1: лимит 2 000 000 знаков объявлен и исполняется."""
    assert MAX_TEXT_CHARS == 2_000_000
    check_text_length("короткий текст")
    check_text_length("x" * MAX_TEXT_CHARS)
    with pytest.raises(TextTooLongError) as error:
        check_text_length("x" * (MAX_TEXT_CHARS + 1))
    assert "2000000" in str(error.value)


def test_request_body_limit_covers_two_texts() -> None:
    """ТЗ п. 4.1 треб. 8: тело запроса вмещает два предельных текста в UTF-8."""
    assert MAX_BODY == MAX_REQUEST_BYTES
    assert MAX_BODY >= 2 * MAX_TEXT_CHARS * 2  # два текста, по 2 байта на букву


def test_engine_rejects_too_long_text() -> None:
    """Движок отказывает на тексте длиннее лимита, а не обрезает его молча."""
    verifier = Verifier(mode="demo")
    with pytest.raises(TextTooLongError):
        verifier.verify("ответ", "я" * (MAX_TEXT_CHARS + 1))


# --------------------------------------------------------------- критерии


def test_gates_use_contract_numbers() -> None:
    """ТЗ п. 4.1 треб. 6: договорный ориентир — F1 ≥ 0.50 при FPR ≤ 0.40."""
    assert CONTRACT_GATE == {"f1_min": 0.50, "fpr_max": 0.40}
    assert INTERNAL_TARGET == {"f1_min": 0.90, "fpr_max": 0.10}


def test_gate_report_statuses() -> None:
    """Статусы критериев считаются от измеренных значений, а не от ожиданий."""
    reached = gate_report({"f1": 0.6, "fpr": 0.3}, "tokens")
    assert reached["contract"]["status"] == "достигнут"
    assert reached["internal"]["status"] == "НЕ достигнут"
    missed = gate_report({"f1": 0.4, "fpr": 0.3}, "tokens")
    assert missed["contract"]["status"] == "НЕ достигнут"
    unknown = gate_report({"f1": None, "fpr": 0.3}, "tokens")
    assert unknown["contract"]["status"] == "не измерено"
    spans = gate_report({"f1": 0.7, "fpr_tokens": 0.01}, "spans")
    assert spans["contract"]["status"] == "достигнут"


def test_evaluate_reports_gates_for_all_families() -> None:
    """Оценка возвращает критерии по токенам, ответам и фрагментам."""
    verifier = Verifier(mode="demo")
    metrics = verifier.evaluate([{"context": CONTEXT, "answer": ANSWER, "labels": [[38, 44, 1]]}])
    assert set(metrics["gates"]) == {"tokens", "answers", "spans"}
    for block in metrics["gates"].values():
        assert block["contract"]["gate"] == CONTRACT_GATE


# --------------------------------------------------------------- интерфейсы


def test_cli_features_prints_five_features(capsys) -> None:
    """CLI-команда ``features`` печатает дамп признаков и возвращает 0."""
    code = main(["features", "--answer", ANSWER, "--context", CONTEXT, "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 5
    assert payload["features"] == list(FEATURE_NAMES)
    assert payload["tokens"]
    for row in payload["tokens"]:
        for name in FEATURE_NAMES:
            assert math.isfinite(float(row[name]))


def test_http_features_route_returns_five_features() -> None:
    """ТЗ п. 4.1 треб. 8: POST /v1/features отдаёт пять признаков по токенам."""
    service = Service(mode="demo")
    handler = make_handler(service)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = Request(
            f"http://127.0.0.1:{server.server_address[1]}/v1/features",
            data=json.dumps({"answer": ANSWER, "context": CONTEXT}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["count"] == 5
        assert payload["features"] == list(FEATURE_NAMES)
        assert payload["tokens"]
    finally:
        server.shutdown()
        server.server_close()
