"""Тесты генератора демонстрационного корпуса."""

from __future__ import annotations

import json
import random

from spanverify.demo_data import (
    dataset_statistics,
    generate_dataset,
    make_ai_paragraph,
    make_human_paragraph,
    make_mixed_document,
    read_dataset,
    write_dataset,
)


def test_dataset_generation_is_reproducible():
    first = generate_dataset(5, seed=99)
    second = generate_dataset(5, seed=99)
    assert [d["text"] for d in first] == [d["text"] for d in second]


def test_different_seeds_give_different_texts():
    assert generate_dataset(3, seed=1)[0]["text"] != generate_dataset(3, seed=2)[0]["text"]


def test_labels_are_ordered_and_inside_text():
    doc = make_mixed_document(random.Random(4))
    text = doc["text"]
    previous_end = 0
    for start, end, label in doc["labels"]:
        assert 0 <= start < end <= len(text)
        assert start >= previous_end
        assert label in (0, 1)
        previous_end = end


def test_labels_alternate_between_classes():
    doc = make_mixed_document(random.Random(5), segments=6)
    labels = [label for _, _, label in doc["labels"]]
    assert len(labels) >= 5
    assert all(a != b for a, b in zip(labels, labels[1:], strict=False))


def test_ai_and_human_paragraphs_differ_in_vocabulary():
    rng = random.Random(6)
    ai = make_ai_paragraph(rng, 4).lower().split()
    human = make_human_paragraph(rng, 4).lower().split()
    assert len(set(ai)) / len(ai) < len(set(human)) / len(human)


def test_write_and_read_dataset_roundtrip(tmp_path):
    documents = generate_dataset(4, seed=42)
    path = tmp_path / "data" / "demo.jsonl"
    write_dataset(documents, path)
    loaded = list(read_dataset(path))
    assert len(loaded) == 4
    assert loaded[0]["text"] == documents[0]["text"]
    assert first_line_is_json(path)


def first_line_is_json(path) -> bool:
    line = path.read_text(encoding="utf-8").splitlines()[0]
    return isinstance(json.loads(line), dict)


def test_dataset_statistics_shape():
    documents = generate_dataset(10, seed=5)
    stats = dataset_statistics(documents)
    assert stats["documents"] == 10
    assert 0.0 < stats["ai_share_chars"] < 1.0
    assert stats["vocabulary"] > 50
    assert stats["dataset_version"].endswith("synthetic")
