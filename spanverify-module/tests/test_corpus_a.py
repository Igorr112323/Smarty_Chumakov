"""Тесты корпуса A: детерминированность, баланс классов и корректность разметки.

Корпус A — обучающий корпус, собранный скриптом. Ошибка в нём (перепутанные метки,
утечка между сплитами, слишком мало чистых пар) незаметна в метриках, зато портит
все выводы. Поэтому проверяются именно эти свойства, а не числа на выходе модели.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from scripts.build_corpus_a import (
    DATASET_VERSION,
    MODES,
    TAXONOMY,
    Fact,
    build,
    build_corpus,
    build_variant,
    check_balance,
    check_shared_groups,
    check_spans,
    generate_documents,
    split_pairs,
)
from spanverify.dataset import read_pairs

TARGET = 1200
SEED = 42


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Собрать корпус A в временный каталог и вернуть пары, сплиты и манифест."""
    out = tmp_path_factory.mktemp("corpus_a")
    manifest = build(out / "docs", out, target=TARGET, seed=SEED)
    pairs = list(read_pairs(out / "pairs.jsonl"))
    splits = {name: list(read_pairs(out / "splits" / f"{name}.jsonl")) for name in ("train", "dev", "test")}
    return {"manifest": manifest, "pairs": pairs, "splits": splits}


def test_target_and_balance_are_met(corpus: dict) -> None:
    """Корпус набирает цель, ≥40 % чистых пар и ≥40 пар на каждый тип ошибки."""
    balance = corpus["manifest"]["balance"]
    assert len(corpus["pairs"]) == TARGET
    assert balance["clean_share"] >= 0.40
    assert balance["problems"] == []
    for mode in MODES:
        assert balance["counts"][mode] >= (40 if mode != "faithful" else 1), mode


def test_taxonomy_types_are_all_present(corpus: dict) -> None:
    """Покрыты все шесть типов таксономии бенчмарка, включая подтип «атрибуция числа»."""
    taxonomy = {pair["meta"]["taxonomy"] for pair in corpus["pairs"]}
    assert {"Contradiction", "Unconfirmed", "Missing", "Oversight", "Partial", "Excess"} <= taxonomy
    attribution = [pair for pair in corpus["pairs"] if pair["meta"]["mode"] == "number_attribution"]
    assert len(attribution) >= 40


def test_every_span_matches_answer_slice(corpus: dict) -> None:
    """Для каждой метки срез ответа равен записанному тексту спана (проверка не по кругу)."""
    checked = 0
    for pair in corpus["pairs"]:
        texts = pair["meta"]["span_texts"]
        assert len(texts) == len(pair["labels"]), pair["id"]
        for index, (start, end, label) in enumerate(pair["labels"]):
            assert label == 1
            assert 0 <= start < end <= len(pair["answer"]), pair["id"]
            assert pair["answer"][start:end] == texts[index], pair["id"]
            checked += 1
    assert checked == corpus["manifest"]["spans_checked"]
    assert corpus["manifest"]["spans_problems"] == []


def test_splits_share_no_documents(corpus: dict) -> None:
    """Сплиты 70/15/15 не пересекаются по документам: иначе метрики завышены."""
    assert corpus["manifest"]["shared_groups"] == 0
    for name, pairs_block in corpus["splits"].items():
        assert pairs_block, name
        assert all(pair["meta"]["dataset_version"] == DATASET_VERSION for pair in pairs_block)
    assert sum(len(block) for block in corpus["splits"].values()) == len(corpus["pairs"])


def test_generation_is_deterministic(tmp_path: Path) -> None:
    """Один и тот же seed даёт одинаковые пары: корпус воспроизводим."""
    documents = generate_documents(10, random.Random(1))
    first = build_corpus(documents, target=120, seed=7)
    second = build_corpus(documents, target=120, seed=7)
    assert [pair["answer"] for pair in first] == [pair["answer"] for pair in second]
    assert [pair["labels"] for pair in first] == [pair["labels"] for pair in second]
    other_seed = build_corpus(documents, target=120, seed=8)
    assert [pair["answer"] for pair in first] != [pair["answer"] for pair in other_seed]


def test_number_attribution_uses_other_subject_value() -> None:
    """Подтип «атрибуция числа» берёт значение другого субъекта и размечает именно его."""
    rng = random.Random(3)
    first = Fact("первичные учётные документы", "срок хранения", "пять лет")
    second = Fact("вторичные учётные документы", "срок хранения", "десять лет")
    variant = build_variant(first, second, "number_attribution", rng)
    assert "десять лет" in variant["answer"]
    assert "пять лет" not in variant["answer"]
    start, end, label = variant["labels"][0]
    assert label == 1
    assert variant["answer"][start:end] == variant["spans_text"][0] == "десять лет"


def test_mode_requires_matching_feature_for_attribution() -> None:
    """Без второго факта с тем же признаком подтип атрибуции не строится (не подменяется молча)."""
    rng = random.Random(4)
    first = Fact("акты", "срок хранения", "пять лет")
    other = Fact("счета", "периодичность резервного копирования", "раз в сутки")
    with pytest.raises(ValueError, match="атрибуция числа"):
        build_variant(first, other, "number_attribution", rng)


def test_clean_pairs_have_no_labels(corpus: dict) -> None:
    """Чистые пары (faithful) идут без разметки: ложные метки испортили бы обучение."""
    for pair in corpus["pairs"]:
        if pair["meta"]["mode"] == "faithful":
            assert pair["labels"] == []
            assert pair["meta"]["clean"] is True
        else:
            assert pair["labels"], pair["id"]
            assert pair["meta"]["clean"] is False


def test_balance_check_reports_shortfall() -> None:
    """Проверка состава честно сообщает о недоборе, а не молчит."""
    pairs = [
        {"meta": {"mode": "faithful"}},
        {"meta": {"mode": "contradiction"}},
    ]
    balance = check_balance(pairs)
    assert balance["problems"]
    assert any("contradiction" in problem for problem in balance["problems"])


def test_shared_groups_detects_leak() -> None:
    """Если документ попал в две части, проверка это находит (счётчик не всегда 0)."""
    pair = {"meta": {"group": "doc-1"}}
    leaks = check_shared_groups({"train": [pair], "test": [pair]})
    assert leaks == 1


def test_check_spans_detects_broken_labels() -> None:
    """Испорченная метка (срез не совпадает с текстом) не проходит проверку."""
    pair = {
        "id": "corpus-a-1",
        "answer": "срок пять лет",
        "labels": [[5, 9, 1]],
        "meta": {"span_texts": ["десять"], "group": "doc-1"},
    }
    result = check_spans([pair])
    assert result["problems"]


def test_split_pairs_keeps_groups_together() -> None:
    """Разбиение режется по документам целиком: ни одна группа не разрывается."""
    pairs = [{"meta": {"group": f"doc-{index // 3}"}, "id": f"p{index}"} for index in range(30)]
    splits = split_pairs(pairs, seed=5)
    assert sum(len(block) for block in splits.values()) == 30
    for name, block in splits.items():
        groups = {pair["meta"]["group"] for pair in block}
        for other_name, other_block in splits.items():
            if other_name == name:
                continue
            assert not groups & {pair["meta"]["group"] for pair in other_block}


def test_manifest_hashes_are_stable(corpus: dict) -> None:
    """Манифест содержит SHA256 файлов и признак синтетических документов."""
    assert corpus["manifest"]["sha256"]["pairs.jsonl"]
    assert corpus["manifest"]["documents"]["synthetic"] is True
    assert corpus["manifest"]["documents"]["count"] >= 60


def test_taxonomy_mapping_covers_all_modes() -> None:
    """Для каждого режима есть тип таксономии — иначе разметка потеряла бы класс ошибки."""
    assert set(TAXONOMY) == set(MODES)
    assert TAXONOMY["number_attribution"] == "Contradiction"
