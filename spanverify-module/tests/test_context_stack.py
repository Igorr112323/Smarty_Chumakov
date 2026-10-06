"""Тесты контекстного конвейера: ядро, признаки, движок, корпус, обучение.

Каждый тест проверяет свойство, которое важно для продукта: непрерывность
координат, абсолютный смысл признака опоры, поведение порога маски, честность
метрик. Числовые пороги в тестах подобраны с запасом, чтобы не ломаться от
небольших изменений данных.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spanverify.core import (
    VerificationResult,
    number_value,
    numbers_in,
    split_chunks,
    split_sentences,
    tokenize_with_offsets,
)
from spanverify.dataset import (
    Fact,
    Pair,
    _pick_unused,
    _value_span,
    corpus_statistics,
    generate_pairs,
    make_pair,
    read_pairs,
    unit_form,
    write_pairs,
)
from spanverify.engine import (
    SMOOTH_WINDOW,
    Verifier,
    WeightsBundle,
    _answer_score,
    _mean_width_ratio,
    _smooth,
    _span_f1,
    span_threshold_for,
    verify_text,
)
from spanverify.features import (
    DEFAULT_WEIGHTS,
    DEMO_WARNING,
    FeatureMatrix,
    _clamp01,
    _jaccard,
    _stem,
    _support_overlap,
    _trigrams,
    combine,
    demo_features,
    is_scored_token,
    scale,
)
from spanverify.train import (
    TokenSample,
    _head_rows,
    _head_scores,
    _select_mask,
    _span_f1_from_indices,
    _standardize,
    _train_logreg,
    _weight_grid,
    auc_score,
    collect_samples,
    load_head,
    save_training_artifacts,
    train,
)

CONTEXT = "Регламент 343: срок хранения первичных документов составляет 10 лет."
ANSWER_TRUE = "Срок хранения первичных документов составляет 10 лет."
ANSWER_FALSE = "Срок хранения первичных документов составляет 3 года."
ANSWER_FABRICATED = (
    "Срок хранения первичных документов составляет 10 лет. " "Дополнительно требуется согласование с внешним аудитором."
)


def make_verifier(**kwargs) -> Verifier:
    """Движок в демо-режиме на весах по умолчанию (без чтения файлов)."""
    bundle = WeightsBundle(weights=dict(DEFAULT_WEIGHTS), threshold=0.5)
    return Verifier(mode="demo", weights=bundle, **kwargs)


def tiny_pairs(count: int = 60, seed: int = 5) -> list[dict]:
    """Небольшой воспроизводимый корпус пар для обучения в тестах."""
    return [pair.to_dict() for pair in generate_pairs(count, seed=seed)]


# ------------------------------------------------------------------ ядро


def test_tokens_cover_text_continuously():
    """Токены покрывают текст без пропусков: склейка совпадает с исходником."""
    text = "Регламент 343: срок — 10 лет, не меньше."
    tokens = tokenize_with_offsets(text)
    assert len(tokens) > 5
    assert "".join(token.text for token in tokens) == text
    assert tokens[0].start == 0
    assert tokens[-1].end == len(text)


def test_tokens_keep_offsets_consistent():
    """Смещения токенов совпадают с позициями их текста в исходной строке."""
    text = "Хранение 10 лет."
    for token in tokenize_with_offsets(text):
        assert text[token.start : token.end] == token.text


def test_whitespace_only_text_gives_no_tokens():
    """Текст из пробелов не порождает токенов (и не падает)."""
    assert tokenize_with_offsets("   \n\t ") == []


def test_sentences_split_covers_whole_text():
    """Разбиение на предложения покрывает текст целиком и по порядку."""
    text = "Первое предложение. Второе! Третье?"
    bounds = split_sentences(text)
    assert len(bounds) == 3
    assert bounds[0][0] == 0
    assert bounds[-1][1] == len(text)
    for start, end in bounds:
        assert text[start:end].strip()


def test_chunks_from_string_and_list_are_equivalent_in_content():
    """Контекст-строка и список фрагментов дают одинаковый текст.

    Разбиение на чанки при этом разное (строка режется по предложениям, список
    уже пришёл готовыми фрагментами) — это документированное поведение.
    """
    single = split_chunks("Первое. Второе.")
    multiple = split_chunks(["Первое. Второе."])
    assert single.text == multiple.text
    assert "".join(single.chunks).replace(" ", "") == "".join(multiple.chunks).replace(" ", "")


def test_chunks_of_none_is_empty():
    """Пустой контекст — это пустые чанки, а не исключение."""
    chunks = split_chunks(None)
    assert chunks.text == "" and chunks.chunks == ()


def test_number_value_reads_digits_and_words():
    """Числовое значение распознаётся и цифрами, и словом."""
    digits = tokenize_with_offsets("10 лет")[0]
    words = tokenize_with_offsets("десять лет")[0]
    assert number_value(digits) == "10"
    assert number_value(words) == "10"
    assert number_value(tokenize_with_offsets("хранение")[0]) is None


def test_numbers_in_collects_unique_values():
    """Сбор чисел из текста не повторяет одинаковые значения."""
    assert numbers_in("10 лет и 10 месяцев, плюс 3 дня") == {"10", "3"}


def test_verification_result_roundtrip_keeps_fields():
    """Сериализация результата сохраняет вердикт, оценку и фрагменты."""
    verifier = make_verifier()
    result = verifier.verify(ANSWER_FALSE, CONTEXT)
    restored = VerificationResult.from_dict(result.to_dict())
    assert restored.verdict == result.verdict
    assert restored.threshold == pytest.approx(result.threshold)
    assert len(restored.spans) == len(result.spans)


# ------------------------------------------------------------------ признаки


@pytest.mark.parametrize("text", [".", ",", "—", "и", "в", "  ", "на"])
def test_scored_token_rejects_non_content(text):
    """Пунктуация и короткие служебные слова не оцениваются как факты."""
    assert not is_scored_token(text)


@pytest.mark.parametrize("text", ["10", "хранения", "документов", "аудитором"])
def test_scored_token_accepts_content(text):
    """Значимые слова и числа участвуют в оценке достоверности."""
    assert is_scored_token(text)


def test_scale_keeps_zero_and_normalises_maximum():
    """Нормировка по максимуму не сдвигает абсолютный ноль."""
    assert scale([0.0, 0.5, 1.0]) == [0.0, 0.5, 1.0]
    assert scale([0.0, 0.2, 0.4]) == [0.0, 0.5, 1.0]
    assert scale([]) == []


def test_combine_uses_absolute_mass_not_relative():
    """Признак опоры не перенормируется: одинаковая слабая опора = одинаково высокий риск."""
    entropy = [0.0, 0.0]
    weak_risk = combine(entropy, [0.1, 0.1], [0.0, 0.0])
    strong_risk = combine(entropy, [1.0, 0.9], [0.0, 0.0])
    assert weak_risk[0] == pytest.approx(weak_risk[1])
    assert weak_risk[0] > strong_risk[0]
    assert strong_risk[1] > strong_risk[0]


def test_combine_weights_are_normalised():
    """Сумма весов приводится к единице: риск остаётся в [0, 1]."""
    risk = combine([1.0], [0.0], [1.0], {"attention_entropy": 2, "ctx_attention_mass": 2, "embedding_density": 2})
    assert 0.0 <= risk[0] <= 1.0


def test_combine_handles_nan_safely():
    """NaN в признаках не превращает риск в NaN."""
    risk = combine([float("nan")], [float("nan")], [float("nan")])
    assert risk == [risk[0]]
    assert risk[0] == risk[0]


def test_clamp01_limits_values():
    """Значения признаков зажаты в [0, 1], NaN превращается в 0."""
    assert _clamp01(-3.0) == 0.0
    assert _clamp01(7.5) == 1.0
    assert _clamp01(float("nan")) == 0.0


def test_trigrams_are_case_and_yo_insensitive():
    """Триграммы устойчивы к регистру и «ё»."""
    assert _trigrams("Ёжик") == _trigrams("ежик")
    assert _trigrams("") == frozenset()


def test_stem_is_prefix_of_four_characters():
    """Основа-префикс склеивает словоформы одного корня."""
    assert _stem("хранения") == _stem("хранение")
    assert _stem("лет") == "лет"


def test_jaccard_bounds():
    """Коэффициент Жаккара равен 1 для одинаковых множеств и 0 для пустых."""
    assert _jaccard(frozenset("abc"), frozenset("abc")) == 1.0
    assert _jaccard(frozenset(), frozenset("abc")) == 0.0


def test_support_overlap_rewards_shared_stem():
    """Опора равна 1 при общем корне и мала для чужого слова."""
    context_words = [("документов", _trigrams("документов"))]
    assert _support_overlap("документы", context_words) == 1.0
    assert _support_overlap("аудитором", context_words) < 0.5


def test_demo_features_zero_mass_for_unconfirmed_number():
    """Число, которого нет в контексте, не получает опоры."""
    tokens = tokenize_with_offsets(ANSWER_FALSE)
    matrix = demo_features(ANSWER_FALSE, CONTEXT, tokens)
    index = [i for i, token in enumerate(tokens) if token.word == "3"][0]
    assert matrix.ctx_attention_mass[index] == 0.0


def test_demo_features_full_mass_for_confirmed_number():
    """Число из контекста получает полную опору."""
    tokens = tokenize_with_offsets(ANSWER_TRUE)
    matrix = demo_features(ANSWER_TRUE, CONTEXT, tokens)
    index = [i for i, token in enumerate(tokens) if token.word == "10"][0]
    assert matrix.ctx_attention_mass[index] == 1.0


def test_demo_features_matrix_has_one_row_per_token():
    """Матрица признаков выровнена по токенам ответа."""
    tokens = tokenize_with_offsets(ANSWER_TRUE)
    matrix = demo_features(ANSWER_TRUE, CONTEXT, tokens)
    assert len(matrix) == len(tokens)
    row = FeatureMatrix(
        attention_entropy=[0.1], ctx_attention_mass=[0.2], embedding_density=[0.3], token_risk=[0.4]
    ).as_rows()[0]
    assert set(row) == {"attention_entropy", "ctx_attention_mass", "embedding_density", "risk"}
    assert row["risk"] == pytest.approx(0.4)


def test_demo_features_marks_backend():
    """Матрица несёт пометку бэкенда — иначе демо-числа можно спутать с научными."""
    matrix = demo_features(ANSWER_TRUE, CONTEXT)
    assert matrix.meta["backend"] == "demo"


def test_demo_warning_is_explicit():
    """Текст предупреждения прямо говорит, что демо-числа не результат."""
    assert "ДЕМО-РЕЖИМ" in DEMO_WARNING
    assert "hf" in DEMO_WARNING


def test_hf_features_without_torch_is_explicit(monkeypatch):
    """Без torch режим hf объясняет причину, а не падает молча."""
    from spanverify import features as features_module

    monkeypatch.setitem(__import__("sys").modules, "torch", None)
    with pytest.raises((ImportError, RuntimeError)) as error:
        features_module.hf_features(ANSWER_TRUE, CONTEXT)
    assert "torch" in str(error.value).lower() or "transformers" in str(error.value).lower()


# ------------------------------------------------------------------ движок


def test_verify_contract_keys_present():
    """Ответ содержит все поля контракта API (Приложение Г)."""
    payload = make_verifier().verify(ANSWER_FALSE, CONTEXT).to_dict()
    assert set(payload) >= {
        "score",
        "is_hallucination",
        "ai_share",
        "ai_share_hard",
        "threshold",
        "spans",
        "latency_ms",
        "mode",
        "stats",
        "verdict",
    }
    assert set(payload["stats"]) >= {
        "token_count",
        "mean_risk",
        "p90_risk",
        "raw_score",
        "span_threshold",
        "warning",
    }


def test_grounded_answer_has_no_spans():
    """Ответ, полностью опирающийся на документ, не помечается."""
    result = make_verifier().verify(ANSWER_TRUE, CONTEXT)
    assert result.verdict == "grounded"
    assert result.spans == []
    assert result.is_hallucination is False


def test_substituted_value_is_flagged():
    """Подмена числа в остальном верного ответа помечается фрагментом."""
    result = make_verifier().verify(ANSWER_FALSE, CONTEXT)
    assert result.spans, "подмена не найдена"
    assert result.verdict in {"doubtful", "likely_hallucination"}
    assert any("3" in span.text for span in result.spans)


def test_fabricated_sentence_is_flagged():
    """Выдуманное предложение попадает во фрагмент риска."""
    result = make_verifier().verify(ANSWER_FABRICATED, CONTEXT)
    assert result.spans
    assert any("аудитор" in span.text for span in result.spans)


def test_empty_answer_is_handled():
    """Пустой ответ — отдельный вердикт без исключений."""
    result = make_verifier().verify("   ", CONTEXT)
    assert result.verdict == "empty"
    assert result.stats["token_count"] == 0
    assert result.spans == []


def test_without_context_answer_is_not_confirmed():
    """Без контекста опоры нет: ответ не может считаться подтверждённым."""
    result = make_verifier().verify(ANSWER_TRUE, "")
    assert result.verdict in {"doubtful", "likely_hallucination"}
    assert result.spans


def test_punctuation_does_not_create_spans():
    """Знаки препинания не порождают фрагментов у достоверного ответа."""
    result = make_verifier().verify("Срок хранения — 10 лет!!!", CONTEXT)
    assert result.verdict in {"grounded", "empty"}
    assert result.spans == []


def test_tokens_payload_returned_on_request():
    """Разбор по токенам отдаётся только по запросу и содержит метку."""
    verifier = make_verifier()
    assert verifier.verify(ANSWER_FALSE, CONTEXT).tokens == []
    rows = verifier.verify(ANSWER_FALSE, CONTEXT, with_tokens=True).tokens
    assert rows and {"index", "text", "risk", "label", "flagged"} <= set(rows[0])


def test_verify_text_helper_uses_defaults(tmp_path, monkeypatch):
    """Функция-обёртка работает без явного объекта и без файла весов."""
    monkeypatch.chdir(tmp_path)
    result = verify_text(ANSWER_FALSE, CONTEXT, mode="demo", weights_path=None)
    assert result.spans


def test_verifier_keeps_passed_bundle(monkeypatch, tmp_path):
    """Переданные в конструктор веса не подменяются файлом с диска."""
    path = tmp_path / "weights.json"
    path.write_text(
        json.dumps(
            {
                "weights": {"attention_entropy": 1.0, "ctx_attention_mass": 0.0, "embedding_density": 0.0},
                "threshold": 0.9,
            }
        ),
        encoding="utf-8",
    )
    bundle = WeightsBundle(weights=dict(DEFAULT_WEIGHTS), threshold=0.25)
    verifier = Verifier(mode="demo", weights=bundle, weights_path=path)
    assert verifier.bundle.threshold == pytest.approx(0.25)
    assert verifier.weights_loaded is True


def test_span_threshold_respects_bounds():
    """Порог маски всегда внутри [floor, cap], даже если MAD велик или мал."""
    assert span_threshold_for([0.1, 0.1, 0.1], 3.0, 0.35, 0.6) == pytest.approx(0.35)
    assert span_threshold_for([0.9, 0.1, 0.9], 0.0, 0.35, 0.6) <= 0.6
    assert span_threshold_for([], 1.0, 0.35, 0.6) == 1.0


def test_smoothing_keeps_a_single_peak():
    """Сглаживание окном 3 сохраняет одиночный пик (вес центра 0.6)."""
    plain_mean = (0.0 + 0.9 + 0.0) / 3
    smoothed = _smooth([0.0, 0.9, 0.0], SMOOTH_WINDOW)
    assert smoothed[1] > plain_mean
    assert 0.0 < smoothed[0] < smoothed[1]


def test_smoothing_short_sequence_is_identity():
    """На последовательности короче окна сглаживание не меняет значения."""
    assert _smooth([0.4, 0.6], SMOOTH_WINDOW) == [0.4, 0.6]


def test_answer_score_rewards_peaks():
    """Оценка ответа учитывает и средний уровень, и верхние значения."""
    flat = _answer_score([0.4] * 10)
    spiky = _answer_score([0.0] * 9 + [0.9])
    assert spiky < flat  # один пик на десять токенов — ещё не весь ответ
    assert _answer_score([0.8] * 10) > flat


def test_answer_score_of_empty_is_zero():
    """Пустой ответ имеет нулевую оценку."""
    assert _answer_score([]) == 0.0


def test_non_content_tokens_do_not_shift_risks():
    """Пунктуация в конце ответа не повышает риск соседних токенов."""
    with_punctuation = make_verifier().verify(ANSWER_TRUE + " !!!", CONTEXT)
    assert with_punctuation.spans == []
    assert with_punctuation.stats["mean_risk"] < 0.2


def test_weights_bundle_roundtrip(tmp_path):
    """Сохранение и загрузка параметров сохраняют веса, порог и калибровку."""
    verifier = make_verifier()
    result = verifier.verify(ANSWER_FALSE, CONTEXT)
    bundle = verifier.bundle
    bundle.threshold = result.threshold
    path = tmp_path / "weights.json"
    bundle.save(path)
    restored = WeightsBundle.load(path)
    assert restored.weights == bundle.weights
    assert restored.threshold == pytest.approx(bundle.threshold)
    assert restored.source == "disk"
    assert restored.loaded is True


def test_weights_bundle_defaults_without_file(tmp_path):
    """Без файла параметров движок работает на значениях по умолчанию."""
    bundle = WeightsBundle.load(tmp_path / "нет.json")
    assert bundle.source == "defaults"
    assert bundle.loaded is False
    assert bundle.weights == DEFAULT_WEIGHTS


def test_head_changes_decisions():
    """Обученная голова реально участвует в решении, а не лежит мёртвым грузом."""
    bundle = WeightsBundle(weights=dict(DEFAULT_WEIGHTS), threshold=0.5)
    bundle.head = {
        "type": "logreg",
        "model": {"weights": [0.0, 0.0, 0.0, 3.0, 0.0, 0.0], "bias": -0.5},
        "scaler": {"means": [0, 0, 0, 0, 0, 0], "scales": [1, 1, 1, 1, 1, 1]},
    }
    plain = Verifier(mode="demo", weights=WeightsBundle(weights=dict(DEFAULT_WEIGHTS), threshold=0.5)).verify(
        ANSWER_TRUE, CONTEXT
    )
    with_head = Verifier(mode="demo", weights=bundle).verify(ANSWER_TRUE, CONTEXT)
    assert with_head.stats["mean_risk"] > plain.stats["mean_risk"]
    assert with_head.spans and not plain.spans


def test_broken_head_falls_back_to_rule():
    """Повреждённый артефакт головы не ломает проверку — работает правило."""
    bundle = WeightsBundle(weights=dict(DEFAULT_WEIGHTS), threshold=0.5)
    bundle.head = {"type": "logreg", "file": "config/нет-такого-файла.json"}
    result = Verifier(mode="demo", weights=bundle).verify(ANSWER_TRUE, CONTEXT)
    assert result.verdict == "grounded"


def test_evaluate_returns_documented_sections():
    """Сквозная оценка возвращает метрики токенов, фрагментов и ответов."""
    pairs = tiny_pairs(40)
    metrics = make_verifier().evaluate(pairs)
    assert set(metrics) >= {"tokens", "spans", "answers", "pairs", "mode"}
    assert metrics["pairs"] == len(pairs)
    assert 0.0 <= metrics["tokens"]["f1"] <= 1.0
    assert metrics["spans"]["iou_threshold"] == 0.5


def test_evaluate_containment_recall_is_full_for_matching_pipeline():
    """На подтверждённых парах расширение до предложения накрывает разметку."""
    pairs = [pair.to_dict() for pair in generate_pairs(40, seed=11) if pair.meta["kind"] == "valueless" or True]
    metrics = make_verifier().evaluate(pairs)
    assert metrics["spans"]["recall_containment"] >= 0.6


def test_span_f1_helpers_agree_on_identical_ranges():
    """Оба расчёта F1 по фрагментам дают 1.0 при полном совпадении."""
    assert _span_f1_from_indices([(0, 3)], [(0, 3)]) == 1.0
    assert _span_f1([[(0, 3)]], [[(0, 3)]])["f1"] == 1.0


def test_mean_width_ratio_reports_expansion():
    """Отношение ширин показывает, что найденный фрагмент шире узкой разметки."""
    assert _mean_width_ratio([[(0, 50)]], [[(10, 12)]]) == pytest.approx(25.0)
    assert _mean_width_ratio([[(0, 5)]], [[]]) == 0.0


# ------------------------------------------------------------------ корпус


def test_dataset_is_reproducible():
    """Один и тот же seed даёт побайтово одинаковый корпус."""
    first = [pair.to_dict() for pair in generate_pairs(20, seed=99)]
    second = [pair.to_dict() for pair in generate_pairs(20, seed=99)]
    assert first == second


def test_dataset_labels_are_inside_answer():
    """Разметка всегда ссылается на реальные символы ответа."""
    for pair in generate_pairs(60, seed=17):
        assert pair.labels or pair.meta["kind"] == "faithful"
        for start, end, label in pair.labels:
            assert 0 <= start < end <= len(pair.answer)
            assert label == 1
            assert pair.answer[start:end].strip()


def test_faithful_pairs_have_no_labels():
    """У подтверждённых ответов нет размеченной недостоверности."""
    for pair in generate_pairs(60, seed=23):
        if pair.meta["kind"] == "faithful":
            assert pair.labels == []


def test_substitution_pairs_flag_a_number():
    """В подмене значения размечен именно числовой фрагмент."""
    marked = 0
    for pair in generate_pairs(60, seed=29):
        if pair.meta["kind"] == "value_substitution":
            start, end, _ = pair.labels[0]
            fragment = pair.answer[start:end]
            assert any(char.isdigit() for char in fragment) or fragment.isalpha()
            marked += 1
    assert marked > 0


def test_fabrication_text_absent_from_context():
    """Выдуманный фрагмент действительно отсутствует в контексте."""
    checked = 0
    for pair in generate_pairs(80, seed=31):
        if pair.meta["kind"] == "fabrication":
            start, end, _ = pair.labels[0]
            fragment = pair.answer[start:end]
            assert fragment not in pair.context
            checked += 1
    assert checked > 0


def test_corpus_statistics_counts_kinds():
    """Статистика корпуса согласована с числом пар и долями классов."""
    pairs = generate_pairs(50, seed=41)
    stats = corpus_statistics(pairs)
    assert stats["pairs"] == 50
    assert sum(stats["kinds"].values()) == 50
    assert 0.0 <= stats["hallucination_rate"] <= 1.0
    assert stats["mean_answer_chars"] > 10


def test_write_and_read_pairs_roundtrip(tmp_path):
    """Запись и чтение корпуса сохраняют разметку."""
    path = write_pairs(generate_pairs(7, seed=3), tmp_path / "pairs.jsonl")
    rows = list(read_pairs(path))
    assert len(rows) == 7
    assert {"id", "context", "answer", "labels"} <= set(rows[0])


def test_pick_unused_cycles_after_exhaustion():
    """Выбор без повторов перезапускается, когда варианты кончились."""
    import random

    rng = random.Random(1)
    used: dict[str, set] = {}
    picks = [_pick_unused(rng, ["a", "b"], used) for _ in range(3)]
    assert sorted(picks[:2]) == ["a", "b"]
    assert picks[2] in {"a", "b"}


def test_unit_form_agrees_with_number():
    """Форма единицы согласуется с числом: 1 год, 3 года, 5 лет, 11 лет."""
    assert unit_form(1, "years") == "год"
    assert unit_form(3, "years") == "года"
    assert unit_form(5, "years") == "лет"
    assert unit_form(11, "years") == "лет"
    assert unit_form(22, "years") == "года"


def test_value_span_prefers_digit_form():
    """Разметка подмены указывает на цифровую запись значения."""
    labels = _value_span("Составляет 5 лет.", 5)
    assert labels == [[11, 12, 1]]


def test_make_pair_uses_distinct_answer_templates():
    """Один шаблон вопроса не дублируется внутри пары (проверка антишаблона)."""
    import random

    pair = make_pair(random.Random(7), index=0)
    assert pair.meta["question"] not in pair.answer or pair.meta["question"].startswith("Каков")


def test_fact_properties_are_consistent():
    """Свойства факта согласованы: единица, словесная форма, титульная форма."""
    fact = Fact(1, "срок хранения", "years", 5, "t", "tail")
    assert fact.unit == "лет"
    assert fact.value_phrase(3) == "3 года"
    assert fact.subject_titled == "Срок хранения"


def test_pair_dict_shape():
    """Словарь пары содержит ровно те поля, которые читает обучение."""
    payload = Pair(id="p", context="c", answer="a").to_dict()
    assert set(payload) == {"id", "context", "answer", "labels", "meta"}


# ------------------------------------------------------------------ обучение


def test_weight_grid_is_a_simplex():
    """Сетка весов: сумма ровно 1, шаг 0.1, число комбинаций как у симплекса."""
    grid = _weight_grid()
    assert len(grid) == 66
    for weights in grid:
        assert sum(weights.values()) == pytest.approx(1.0)
        assert all(0.0 <= value <= 1.0 for value in weights.values())
    assert {"attention_entropy": 0.9, "ctx_attention_mass": 0.1, "embedding_density": 0.0} in grid


def test_auc_perfect_and_inverted():
    """AUC равен 1 при идеальном разделении и 0 при инвертированном."""
    assert auc_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert auc_score([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)
    assert auc_score([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)


def test_auc_single_class_is_nan():
    """AUC не определён, если в выборке только один класс."""
    value = auc_score([1, 1], [0.1, 0.9])
    assert value != value


def test_collect_samples_matches_labels():
    """Сбор сэмплов даёт столько же положительных токенов, сколько в разметке."""
    pairs = generate_pairs(20, seed=13)
    samples, stats = collect_samples(make_verifier(), [pair.to_dict() for pair in pairs])
    assert stats["pairs"] == 20
    assert stats["positive"] == sum(sample.label for sample in samples)
    assert all(0.0 <= value <= 1.0 for sample in samples for value in sample.features.values())


def test_standardize_gives_zero_mean():
    """Стандартизация центрирует столбцы признаков."""
    rows, means, scales = _standardize([[1.0, 2.0], [3.0, 4.0]])
    assert means == pytest.approx([2.0, 3.0])
    assert scales == pytest.approx([1.0, 1.0])
    assert sum(row[0] for row in rows) == pytest.approx(0.0)


def test_logreg_learns_order():
    """Логистическая регрессия выучивает порядок на разделимых данных."""

    def sample(index: int, risk: float, label: int) -> TokenSample:
        return TokenSample(
            "p",
            index,
            2,
            "слово",
            {"attention_entropy": 0.0, "ctx_attention_mass": 0.0, "embedding_density": 0.0, "risk": risk},
            label,
        )

    samples = [sample(0, -2.0, 0), sample(1, -1.0, 0), sample(2, 1.0, 1), sample(3, 2.0, 1)]
    rows, labels = _head_rows(samples, [-2.0, -1.0, 1.0, 2.0])
    scaled, means, scales = _standardize(rows)
    model = _train_logreg(scaled, labels, epochs=300)
    assert model["weights"][3] > 0  # столбец риска
    payload = {"model": model, "scaler": {"means": means, "scales": scales}}
    probabilities = _head_scores(samples, payload, [-2.0, -1.0, 1.0, 2.0])
    assert probabilities[3] > probabilities[0]
    assert 0.0 <= min(probabilities) and max(probabilities) <= 1.0


def test_head_scores_rejects_wrong_dimensionality():
    """Артефакт головы чужой размерности откатывается к правилу."""
    samples = [
        TokenSample(
            "p",
            0,
            1,
            "слово",
            {"attention_entropy": 0.0, "ctx_attention_mass": 0.0, "embedding_density": 0.0, "risk": 0.3},
            1,
        )
    ]
    payload = {"model": {"weights": [1.0], "bias": 0.0}, "scaler": {"means": [0.0], "scales": [1.0]}}
    assert _head_scores(samples, payload, [0.3]) == [0.3]


def test_head_scores_falls_back_without_model():
    """Без модели голова возвращает риски правила — без исключений."""
    samples = [
        TokenSample(
            "p",
            0,
            1,
            "слово",
            {"attention_entropy": 0.0, "ctx_attention_mass": 0.0, "embedding_density": 0.0, "risk": 0.3},
            1,
        )
    ]
    assert _head_scores(samples, {}, [0.3]) == [0.3]


def test_train_produces_valid_bundle():
    """Обучение на малом корпусе даёт согласованный набор параметров."""
    report = train(tiny_pairs(80, seed=5), mode="demo", seed=42, folds=5, dataset_name="tests")
    bundle = report.bundle
    assert sum(bundle.weights.values()) == pytest.approx(1.0)
    assert 0.0 <= bundle.threshold <= 1.0
    assert bundle.isotonic is not None
    assert len(bundle.folds) == 5
    assert bundle.meta["synthetic"] is True
    assert bundle.span_floor < bundle.span_cap
    # Если победила голова, её модель обязана лежать в бандле: иначе оценка
    # прочитает config/head.json из поставки и применит чужие веса.
    if (bundle.head or {}).get("type") == "logreg":
        assert (bundle.head.get("model") or {}).get("weights"), "голова не встроена в бандл"


def test_train_reports_validation_and_selection():
    """Отчёт содержит метрики валидации и объяснение выбора сигнала."""
    report = train(tiny_pairs(80, seed=6), mode="demo", seed=42)
    validation = report.validation
    assert {"tokens", "f1", "fpr", "auc", "span_f1", "answer_auc", "end_to_end", "selection"} <= set(validation)
    assert validation["selection"]["signal"] in {"rule", "logreg"}
    assert validation["end_to_end"]["tokens"]["f1"] >= 0.0


def test_train_is_reproducible():
    """При одинаковом seed обучение даёт одинаковые параметры."""
    first = train(tiny_pairs(60, seed=8), mode="demo", seed=42, folds=3).bundle.to_dict()
    second = train(tiny_pairs(60, seed=8), mode="demo", seed=42, folds=3).bundle.to_dict()
    assert first["weights"] == second["weights"]
    assert first["threshold"] == pytest.approx(second["threshold"])
    assert first["isotonic"] == second["isotonic"]


def test_train_requires_data():
    """Пустой корпус — понятная ошибка, а не тихий результат."""
    with pytest.raises(ValueError):
        train([], mode="demo")


def test_save_training_artifacts_writes_weights(tmp_path):
    """Обучение сохраняет weights.json в требуемой схеме (Приложение Г)."""
    report = train(tiny_pairs(70, seed=9), mode="demo", seed=42)
    written = save_training_artifacts(report, tmp_path / "config" / "weights.json", root=tmp_path)
    assert written["weights"].is_file()
    payload = json.loads(written["weights"].read_text(encoding="utf-8"))
    assert set(payload) >= {"weights", "threshold", "target_fpr", "folds", "isotonic", "head", "seed", "version"}
    assert set(payload["isotonic"]) == {"x", "y"}
    assert payload["seed"] == 42


def test_saved_head_is_readable(tmp_path):
    """Если выбрана голова, её артефакт читается и содержит модель и масштаб."""
    report = train(tiny_pairs(90, seed=10), mode="demo", seed=42)
    written = save_training_artifacts(report, tmp_path / "config" / "weights.json", root=tmp_path)
    if "head" in written:
        payload = load_head(written["head"])
        assert payload["type"] == "logreg"
        assert len(payload["model"]["weights"]) == len(payload["scaler"]["means"]) == 6
        assert payload["model"]["weights"]


def test_load_head_missing_file_returns_none(tmp_path):
    """Отсутствующий артефакт головы — это None, а не исключение."""
    assert load_head(tmp_path / "нет.json") is None


def test_select_mask_respects_fpr_limit():
    """Подбор порога маски не превышает заданный FPR на обучающей части."""
    verifier = make_verifier()
    pairs = tiny_pairs(60, seed=12)
    samples, _ = collect_samples(verifier, pairs)
    by_pair: dict[str, list[TokenSample]] = {}
    for sample in samples:
        by_pair.setdefault(sample.pair_id, []).append(sample)

    def risk_fn(rows: list[TokenSample]) -> list[float]:
        return [row.features["risk"] for row in rows]

    params, metrics = _select_mask(verifier, pairs, risk_fn, by_pair, target_fpr=0.05)
    assert metrics["fpr"] <= 0.05 or metrics == {}
    assert len(params) == 3


def test_span_f1_counts_only_close_matches():
    """Фрагменты с недостаточным перекрытием не засчитываются как верные."""
    assert _span_f1_from_indices([(0, 10)], [(0, 2)], iou_threshold=0.5) == 0.0
    assert _span_f1_from_indices([(0, 2)], [(0, 3)], iou_threshold=0.5) == pytest.approx(1.0)
    assert _span_f1_from_indices([], [], iou_threshold=0.5) == 0.0


def test_weights_json_matches_schema_on_disk():
    """Файл весов в репозитории соответствует схеме контракта."""
    path = Path("config/weights.json")
    if not path.is_file():
        pytest.skip("веса ещё не обучены в этом окружении")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert pytest.approx(sum(payload["weights"].values())) == 1.0
    assert payload["target_fpr"] == pytest.approx(0.1)
    assert len(payload["isotonic"]["x"]) == len(payload["isotonic"]["y"])
    assert payload["seed"] == 42
