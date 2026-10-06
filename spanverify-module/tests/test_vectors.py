"""Тесты векторизации и контекстной плотности."""

from __future__ import annotations

from spanverify.vectors import (
    cosine,
    hash_index,
    knn_density,
    normalize,
    token_vector,
    vectorize,
)


def test_hash_index_is_deterministic_and_in_range():
    first = hash_index("слово", 1024)
    second = hash_index("слово", 1024)
    assert first == second
    assert 0 <= first < 1024
    assert 0 <= hash_index("другое", 1024) < 1024


def test_token_vector_is_normalized():
    vec = token_vector("метод", dim=512, prev_word="данный", next_word="обеспечивает")
    norm = sum(v * v for v in vec.values()) ** 0.5
    assert norm == 1.0


def test_cosine_identical_vectors_is_one():
    vec = token_vector("метод", dim=512)
    assert cosine(vec, vec) == 1.0


def test_cosine_range_for_unrelated_words():
    a = token_vector("метод", dim=4096)
    b = token_vector("яблоко", dim=4096)
    assert 0.0 <= cosine(a, b) < 1.0


def test_normalize_handles_empty_vector():
    assert normalize({}) == {}


def test_vectorize_uses_neighbours_as_context():
    words = ["Первый", "Второй", "Третий"]
    vectors = vectorize(words, dim=256)
    standalone = token_vector("Второй", dim=256)
    assert vectors[1] != standalone  # контекст учтён
    assert len(vectors) == len(words)


def test_knn_density_is_higher_for_repeated_words():
    words = ["метод", "метод", "метод", "яблоко", "груша", "слива"]
    vectors = vectorize(words, dim=1024)
    density = knn_density(vectors, k=2)
    assert density[0] > density[-1]


def test_knn_density_handles_single_token():
    assert knn_density(vectorize(["один"], dim=128)) == [0.0]


def test_knn_density_mask_excludes_neighbours():
    words = ["метод", "метод", "метод"]
    vectors = vectorize(words, dim=256)
    masked = knn_density(vectors, k=2, mask=[True, False, False])
    assert masked[0] == 0.0
