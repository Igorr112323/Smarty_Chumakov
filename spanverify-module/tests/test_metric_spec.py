"""Определения метрик из ``docs/METRIC_SPEC.md`` сверяются с кодом.

Тест — часть шага 0 протокола: спецификация заморожена, и её формулы обязаны
совпадать с реализацией продукта, а не «примерно». Проверка сделана на
синтетических фикстурах, где TP/FP/FN/TN посчитаны руками, чтобы ловить не
только опечатку в коде, но и расхождение кода со спецификацией.

Отдельные проверки касаются харнесса ``scripts/hf_protocol.py``: сглаживание,
выбор порога и групповая CV — то, чем нельзя «подкрутить» результат.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spanverify.engine import Verifier, _answer_metrics, _span_f1, _token_metrics  # noqa: E402
from spanverify.features import is_scored_token  # noqa: E402

SPEC_PATH = ROOT / "docs" / "METRIC_SPEC.md"


def _harness():
    script = ROOT / "scripts" / "hf_protocol.py"
    spec = importlib.util.spec_from_file_location("hf_protocol_under_test", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------- уровень токенов


def test_token_formulas_match_spec_on_hand_counted_fixture() -> None:
    """TP/FP/FN/TN, precision, recall, F1 и FPR — как записано в спецификации.

    Фикстура: пять токенов, из них два gold, помечены три (один gold пропущен,
    два ложных). Руками: TP=1, FP=2, FN=1, TN=1 → precision=1/3, recall=1/2,
    F1=2·(1/3)·(1/2)/(1/3+1/2)=0.4, FPR=2/3.
    """
    labels = [1, 1, 0, 0, 0]
    flags = [True, False, True, True, False]
    risks = [0.9, 0.2, 0.8, 0.7, 0.1]
    numbers = _token_metrics(labels, flags, risks)
    precision = 1 / 3
    recall = 1 / 2
    assert numbers["tp"] == 1 and numbers["fp"] == 2 and numbers["fn"] == 1 and numbers["tn"] == 1
    assert numbers["precision"] == pytest.approx(precision)
    assert numbers["recall"] == pytest.approx(recall)
    assert numbers["f1"] == pytest.approx(2 * precision * recall / (precision + recall))
    assert numbers["fpr"] == pytest.approx(2 / 3)
    # AUC — доля пар «gold-токен против не-gold», где оценка gold выше:
    # positives {0,9; 0,2} против negatives {0,8; 0,7; 0,1} → 4 из 6.
    assert numbers["auc"] == pytest.approx(4 / 6)


def test_fpr_denominator_is_all_non_gold_tokens() -> None:
    """FPR = FP/(FP+TN): знаменатель — ВСЕ не-gold токены, а не только gold.

    Формулировка в ТЗ двусмысленна («по токенам, которые в gold поддержаны»),
    спецификация фиксирует прочтение по формуле; тест не даёт подменить одно
    другим: при делении на число gold-токенов значение было бы другим.
    """
    labels = [0, 0, 0, 0, 1, 1]
    flags = [True, False, False, False, True, False]
    numbers = _token_metrics(labels, flags, [0.9, 0.1, 0.1, 0.1, 0.9, 0.2])
    assert numbers["fp"] == 1 and numbers["tn"] == 3
    assert numbers["fpr"] == pytest.approx(1 / 4)
    assert numbers["fpr"] != pytest.approx(1 / 2)


def test_zero_denominators_do_not_drop_cases() -> None:
    """Ни один случай не выбрасывается: пустые знаменатели дают 0.0, как в спеке.

    «Ничего не помечено» → precision и F1 равны 0.0 (не nan, не пропуск), «нет
    не-gold токенов» → FPR 0.0. Тихо исключить такой случай из микросреднего —
    значит улучшить числу без всякого улучшения качества.
    """
    nothing = _token_metrics([1, 1], [False, False], [0.1, 0.2])
    assert nothing["precision"] == 0.0 and nothing["f1"] == 0.0
    assert nothing["fp"] == 0 and nothing["tn"] == 0 and nothing["fpr"] == 0.0
    all_positive = _token_metrics([1, 1], [True, True], [0.9, 0.8])
    assert all_positive["fpr"] == 0.0 and all_positive["precision"] == 1.0 and all_positive["f1"] == 1.0


def test_scored_token_filter_is_the_product_one() -> None:
    """Оцениваются только содержательные токены: иначе это другая метрика."""
    assert is_scored_token("договор")
    assert not is_scored_token(",")
    assert not is_scored_token("—")


# --------------------------------------------------------------- уровень ответов


def test_answer_metrics_use_the_verdict_threshold_rule() -> None:
    """Ответ помечен, если оценка не ниже порога; метрики — микросредние."""
    labels = [1, 1, 0, 0]
    scores = [0.6, 0.4, 0.55, 0.05]
    numbers = _answer_metrics(labels, scores, 0.5)
    assert numbers["threshold"] == 0.5
    assert numbers["tp"] == 1 and numbers["fp"] == 1 and numbers["fn"] == 1 and numbers["tn"] == 1
    assert numbers["precision"] == pytest.approx(0.5) and numbers["f1"] == pytest.approx(0.5)
    assert numbers["fpr"] == pytest.approx(0.5)


def test_verdict_flag_expression_is_frozen() -> None:
    """Правило вердикта зафиксировано в спеке — сверяем его формулировку с кодом.

    Спецификация цитирует выражение продукта; тест проверяет обе части: что в
    тексте спеки стоит именно это множество вердиктов и что «grounded»/«empty»
    в продукте не считаются нарушением.
    """
    text = SPEC_PATH.read_text(encoding="utf-8")
    assert re.search(r"grounded[^\n]{0,40}empty", text), "в спецификации нет зафиксированного множества вердиктов"
    verifier = Verifier(mode="demo")
    context = "Срок хранения первичных документов составляет десять лет."
    grounded = verifier.verify("Срок хранения документов — десять лет.", context, with_tokens=True)
    distorted = verifier.verify("Срок хранения документов — три года.", context, with_tokens=True)
    assert grounded.verdict in {"grounded", "empty"}
    assert distorted.verdict not in {"grounded", "empty"}
    assert (distorted.verdict not in {"grounded", "empty"}) is True


# ------------------------------------------------------------------ уровень фрагментов


def test_span_f1_matches_manual_iou_matching() -> None:
    """Greedy-сопоставление по IoU ≥ 0,5, каждый фрагмент используется один раз.

    Первый ответ: (0,10) попадает на gold (0,12) с IoU 10/12 ≈ 0,833 → TP; (10,14)
    на тот же gold ставить нельзя (уже использовано), своего нет → FP. Второй
    ответ: gold (0,8), предсказание (6,20) даёт IoU 2/22 ≈ 0,09 < 0,5 → ещё FP и
    FN. Итог: TP=1, FP=2, FN=1 → precision 1/3, recall 1/2, F1 0,4.
    """
    predicted = [[(0, 10), (10, 14)], [(6, 20)]]
    truth = [[(0, 12)], [(0, 8)]]
    numbers = _span_f1(predicted, truth, iou_threshold=0.5)
    assert numbers["tp"] == 1 and numbers["fp"] == 2 and numbers["fn"] == 1
    assert numbers["precision"] == pytest.approx(1 / 3) and numbers["recall"] == pytest.approx(0.5)
    assert numbers["f1"] == pytest.approx(0.4)
    assert numbers["iou_threshold"] == 0.5


def test_span_f1_at_and_below_iou_threshold() -> None:
    """На границе IoU = 0,5 совпадение принимается, ниже — нет (спека: ≥ 0,5)."""
    boundary = _span_f1([[(0, 10)]], [[(0, 20)]], iou_threshold=0.5)
    assert boundary["tp"] == 1 and boundary["fp"] == 0 and boundary["f1"] == pytest.approx(1.0)
    below = _span_f1([[(0, 10)]], [[(6, 16)]], iou_threshold=0.5)
    assert below["tp"] == 0 and below["fp"] == 1 and below["fn"] == 1


# ------------------------------------------------------ спецификация ↔ реализация


@pytest.mark.parametrize(
    "fragment",
    [
        r"FPR\s*=\s*FP\s*/\s*\(FP\s*\+\s*TN\)",
        r"IoU\s*[≥>=]{1,2}\s*0[,.]5",
        r"token",
        r"group CV|CV по документам|группов\w+ CV",
    ],
)
def test_metric_spec_contains_frozen_definitions(fragment: str) -> None:
    """Ключевые формулы и правила обязаны быть в замороженном файле.

    Смысл проверки: спецификация — источник определений, и если её правят,
    убрав формулу, тест падает, а не остаётся «документ про что-то».
    """
    text = SPEC_PATH.read_text(encoding="utf-8")
    assert re.search(fragment, text, flags=re.IGNORECASE), f"в METRIC_SPEC.md не найдено: {fragment}"


def test_metric_spec_declares_primary_level_and_freeze_order() -> None:
    text = SPEC_PATH.read_text(encoding="utf-8")
    assert re.search(r"первичн\w* уровня.{0,60}токен", text, flags=re.IGNORECASE | re.DOTALL)
    assert re.search(r"до [^\n]{0,40}(test|шага 3)", text, flags=re.IGNORECASE)


# ------------------------------------------------------------------ харнесс протокола


def test_harness_metrics_are_the_product_functions() -> None:
    """Харнесс обязан считать метрики функциями продукта, а не своей копией."""
    harness = _harness()
    assert harness._token_metrics is _token_metrics
    assert harness._answer_metrics is _answer_metrics
    assert harness._span_f1 is _span_f1


def test_harness_threshold_selection_respects_fpr_limit() -> None:
    """Порог: максимум F1 при FPR ≤ target; кандидаты — значения оценок.

    Оценки: четыре отрицательных (0,9; 0,8; 0,2; 0,1) и два положительных
    (0,95; 0,3). При target FPR 0,5 допустимо не более двух ложных: максимум F1
    достигается на пороге 0,95? нет — на 0,8: TP=1, FP=1, FN=1 → F1≈0,5 при
    FPR=0,25. Проверка фиксирует, что ограничение по FPR действительно режет
    более «смелые» пороги, а не игнорируется.
    """
    harness = _harness()
    scores = [0.95, 0.9, 0.8, 0.3, 0.2, 0.1]
    labels = [1, 0, 0, 1, 0, 0]
    cut = harness.choose_threshold(scores, labels, target_fpr=0.5)
    assert 0.0 < cut["threshold"] <= 0.95
    assert cut["fpr"] <= 0.5
    strict = harness.choose_threshold(scores, labels, target_fpr=0.0)
    # Более жёсткое ограничение не может дать более смелый порог.
    assert strict["threshold"] >= cut["threshold"]
    assert strict["fpr"] <= cut["fpr"]


def test_harness_group_folds_keep_documents_together() -> None:
    harness = _harness()
    keys = ["doc-a", "doc-a", "doc-a", "doc-b", "doc-b", "doc-c", "doc-d", "doc-d"]
    folds = harness.group_folds(keys, folds=2, seed=42)
    assert sum(len(fold) for fold in folds) == len(keys)
    positions: dict[str, set[int]] = {}
    for number, fold in enumerate(folds):
        for index in fold:
            positions.setdefault(keys[index], set()).add(number)
    assert all(len(where) == 1 for where in positions.values()), "документ разъехался по фолдам"
    # Детерминизм: тот же seed — те же фолды.
    assert folds == harness.group_folds(keys, folds=2, seed=42)


def test_harness_smoothing_and_merge_are_deterministic() -> None:
    harness = _harness()
    assert harness._smooth_within([0.2, 0.8, 0.4], ["p", "p", "p"], 1) == pytest.approx(
        [0.5, (0.2 + 0.8 + 0.4) / 3, 0.6]
    )
    assert harness._smooth_within([0.2, 0.8, 0.4], ["p", "p", "p"], 0) == [0.2, 0.8, 0.4]
    merged = harness.merge_spans(
        [
            {"start": 0, "end": 3, "score": 0.9},
            {"start": 4, "end": 7, "score": 0.9},
            {"start": 40, "end": 44, "score": 0.9},
        ],
        threshold=0.5,
        gap=2,
    )
    # Зазор 4−3 = 1 ≤ 2 → первый интервал склеивается со вторым.
    assert merged == [(0, 7), (40, 44)]
    assert harness.merge_spans(
        [{"start": 0, "end": 3, "score": 0.9}, {"start": 4, "end": 7, "score": 0.2}], 0.5, 2
    ) == [(0, 3)]


def test_harness_boost_matches_monotone_signal() -> None:
    """Бустинг пней обязан различать разделяющий признак (проверка реализации)."""
    harness = _harness()
    rows = [[float(index % 7) - 3.0] for index in range(700)]
    labels = [1 if row[0] > 0.5 else 0 for row in rows]
    keys = [f"doc-{index // 70}" for index in range(700)]
    model, threshold, details = harness.fit_with_cv(
        rows,
        labels,
        keys,
        [f"pair-{index}" for index in range(700)],
        {"classifier": "boost", "rounds": 12, "folds": 4},
        42,
    )
    scores = model.scores(rows)
    separated = sum(1 for score, label in zip(scores, labels, strict=False) if (score >= threshold) == bool(label))
    assert separated / len(labels) > 0.9, f"бустинг не разделил тривиальный признак: {separated} из {len(labels)}"
    assert details["folds"] >= 2
    metrics = harness._token_metrics(labels, [score >= threshold for score in scores], scores)
    assert metrics["f1"] > 0.9, f"бустинг не научился разделять линейно разделимый признак: {metrics}"


def test_harness_criterion_status_is_honest() -> None:
    harness = _harness()
    passed = harness.criterion_status({"f1": 0.61, "fpr": 0.39})
    assert passed["status"] == "PASS"
    for numbers, expected in (
        ({"f1": 0.61, "fpr": 0.41}, "NOT MET"),
        ({"f1": 0.59, "fpr": 0.0}, "NOT MET"),
        ({"f1": 0.0, "fpr": 0.0}, "NOT MET"),
        ({}, "NOT MET"),
    ):
        assert harness.criterion_status(numbers)["status"] == expected, numbers


def test_harness_aggregates_seeds_without_choosing_best() -> None:
    """mean ± std по всем seed'ам: лучший не выбирается (протокол это запрещает)."""
    harness = _harness()
    per_seed = [
        {
            "seed": 42,
            "threshold": 0.5,
            "tokens": {"f1": 0.6, "fpr": 0.3, "auc": 0.9},
            "answers": {"f1": 0.5},
            "spans": {"f1": 0.2},
        },
        {
            "seed": 43,
            "threshold": 0.5,
            "tokens": {"f1": 0.7, "fpr": 0.2, "auc": 0.95},
            "answers": {"f1": 0.6},
            "spans": {"f1": 0.4},
        },
    ]
    aggregate = harness.aggregate_seeds(per_seed)
    assert aggregate["f1"] == pytest.approx(0.65)
    assert aggregate["f1_std"] == pytest.approx(0.05)
    assert aggregate["f1_per_seed"] == [0.6, 0.7]
    assert aggregate["fpr"] == pytest.approx(0.25)
    assert math.isclose(aggregate["span_f1"], 0.3)


# ------------------------------------------------- сетка признаков (pure python)


def test_grid_feature_names_are_unique_and_frozen() -> None:
    """Имена признаков сетки — зафиксированный кортеж: по нему сверяется кеш."""
    from spanverify.hf_grid import GRID_CACHE_FORMAT, GRID_FEATURE_NAMES

    assert GRID_CACHE_FORMAT == 3
    assert isinstance(GRID_FEATURE_NAMES, tuple) and len(GRID_FEATURE_NAMES) == 36
    assert len(set(GRID_FEATURE_NAMES)) == len(GRID_FEATURE_NAMES)
    grid = json.loads((ROOT / "config" / "hf_experiments.json").read_text(encoding="utf-8"))
    assert len(grid["experiments"]) <= 10, "лимит экспериментов превышен уже в сетке"
    used = {name for item in grid["experiments"] for name in item["features"]}
    assert used <= set(GRID_FEATURE_NAMES), f"в сетке признаки вне hf_grid: {sorted(used - set(GRID_FEATURE_NAMES))}"


def test_grid_lexicon_window_and_trigram() -> None:
    """Лексико-числовые признаки считаются по документу; сглаживание — по окну."""
    from spanverify.hf_grid import _covers, _document_lexicon, _trigram_match, _window

    assert _window([0.0, 0.3, 0.9], 1) == pytest.approx([0.15, 0.4, 0.6])
    assert _window([0.4], 1) == [0.4]
    assert _window([], 2) == []
    lexicon = _document_lexicon("Срок хранения первичных документов составляет десять лет.")
    assert "10" in lexicon["numbers"], "число словами не распознаётся — признак num_match был бы слепым"
    assert lexicon["stems"] and lexicon["sentence_stems"]
    assert _trigram_match("документов", lexicon["stems"]) > _trigram_match("самолётом", lexicon["stems"])
    assert _covers([(2, 6)], type("T", (), {"start": 4, "end": 5})())
    assert not _covers([(2, 4)], type("T", (), {"start": 4, "end": 5})())


def test_grid_sentence_support_prefers_confirmed_sentence() -> None:
    """Поддержка предложения: подтверждённое предложение документа > выдуманное."""
    from spanverify.core import split_sentences
    from spanverify.hf_grid import _document_lexicon, _sentence_support

    context = "Срок хранения первичных документов составляет десять лет. Акт подписан 12 мая."
    answer = "Срок хранения документов составляет десять лет. Договор расторгнут комиссией."
    support = _sentence_support(answer, list(split_sentences(answer)), _document_lexicon(context))
    assert len(support) == 2
    assert support[0] > support[1], f"первое предложение подтверждено документом сильнее: {support}"
    assert 0.0 <= support[0] <= 1.0


def test_grid_alignment_offsets_with_a_fake_tokenizer() -> None:
    """Восстановление смещений подслов работает и без offset_mapping (случай GPT-2)."""
    from spanverify.hf_grid import _offsets_by_alignment

    class FakeTokenizer:
        words = ["Срок", "хранения", "десять", "лет"]

        def __call__(self, text, **_kwargs):  # noqa: N802 - повторяем контракт hf-токенизатора
            return {"input_ids": list(range(len(self.words)))}

        def decode(self, ids, **_kwargs):
            return self.words[ids[0]]

    prompt = "Документ. Срок хранения десять лет"
    pairs = _offsets_by_alignment(FakeTokenizer(), prompt, 1024)
    assert [prompt[start:end] for start, end in pairs] == ["Срок", "хранения", "десять", "лет"]


def test_grid_row_writer_contract() -> None:
    """Строка кеша v3: 36 массивов, токены ответа и метаданные — контракт precompute."""
    import importlib.util

    script = ROOT / "scripts" / "precompute_features.py"
    spec = importlib.util.spec_from_file_location("precompute_for_grid", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from spanverify.hf_grid import GRID_FEATURE_NAMES

    result = {
        "arrays": {name: [0.25, 0.5] for name in GRID_FEATURE_NAMES},
        "tokens": [{"text": "срок", "start": 0, "end": 4, "scored": 1}],
        "meta": {"unmatched_tokens": 0, "seq_len": 2},
    }
    row = module.dump_grid_row("key-1", "pair-1", result, "model-x")
    assert row["grid_format"] == 3 and row["cache_format"] == 3
    assert row["pair_id"] == "pair-1" and row["model"] == "model-x"
    # Модель и её revision — часть определения признака: шаг 3 weights не грузит,
    # поэтому ревизия обязана читаться из самой строки кеша.
    assert row["revision"] == "", "без ревизии поле пустое: выдумывать её нельзя"
    with_revision = module.dump_grid_row("key-1", "pair-1", result, "model-x", "rev-77")
    assert with_revision["revision"] == "rev-77", "revision весов обязана попадать в строку кеша"
    assert set(row["arrays"]) == set(GRID_FEATURE_NAMES)
    assert row["arrays"]["mass_last"] == [0.25, 0.5]
    assert row["tokens"][0]["text"] == "срок"
