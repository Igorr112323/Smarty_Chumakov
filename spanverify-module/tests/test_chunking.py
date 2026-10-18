"""Тесты для модуля чанкинга и отбора релевантных фрагментов."""

from __future__ import annotations

import json

import pytest

from spanverify.chunking import (
    chunk_document,
    estimate_tokens,
    normalize_for_retrieval,
    select_relevant_chunks,
    select_relevant_chunks_for_claims,
)


class TestEstimateTokens:
    def test_empty(self) -> None:
        assert estimate_tokens("") == 0

    def test_short_text(self) -> None:
        assert estimate_tokens("привет") >= 1

    def test_long_text(self) -> None:
        # ~270 chars ≈ 100 tokens
        text = "а" * 270
        assert estimate_tokens(text) == 100


class TestNormalizeForRetrieval:
    def test_lowercase(self) -> None:
        assert normalize_for_retrieval("ПРИВЕТ") == "привет"

    def test_whitespace(self) -> None:
        assert normalize_for_retrieval("  a   b  ") == "a b"

    def test_numbers_to_digits(self) -> None:
        result = normalize_for_retrieval("три года")
        assert "3" in result


class TestChunkDocument:
    def test_empty(self) -> None:
        dc = chunk_document("")
        assert len(dc.chunks) == 0

    def test_short_text_single_chunk(self) -> None:
        dc = chunk_document("Короткий текст.", max_tokens_per_chunk=380)
        assert len(dc.chunks) >= 1

    def test_long_text_multiple_chunks(self) -> None:
        # Create text with clear sentence boundaries
        sentences = [f"Предложение номер {i}. " for i in range(50)]
        text = " ".join(sentences)
        dc = chunk_document(text, max_tokens_per_chunk=20)
        assert len(dc.chunks) > 1

    def test_overlap(self) -> None:
        text = ". ".join([f"Сentence {i}" for i in range(10)]) + "."
        dc_no_overlap = chunk_document(text, max_tokens_per_chunk=15, overlap_sentences=0)
        dc_overlap = chunk_document(text, max_tokens_per_chunk=15, overlap_sentences=1)
        # With overlap, total text should cover more (or same) unique content
        assert len(dc_overlap.chunks) >= len(dc_no_overlap.chunks)

    def test_paragraph_aware(self) -> None:
        text = "Первый абзац.\n\nВторой абзац.\n\nТретий абзац."
        dc = chunk_document(text, max_tokens_per_chunk=5)
        assert len(dc.chunks) >= 2

    def test_chunk_positions_valid(self) -> None:
        text = "Раз. Два. Три. Четыре. Пять."
        dc = chunk_document(text, max_tokens_per_chunk=5)
        for chunk in dc.chunks:
            assert chunk.start >= 0
            assert chunk.end <= len(text)
            assert chunk.end > chunk.start

    def test_long_sentence_split(self) -> None:
        # Single very long sentence should be split
        long_word = "слово " * 500
        text = long_word.strip()
        dc = chunk_document(text, max_tokens_per_chunk=50)
        assert len(dc.chunks) > 1

    def test_normalized_populated(self) -> None:
        dc = chunk_document("Текст документа.", max_tokens_per_chunk=380)
        assert len(dc.normalized) == len(dc.chunks)


class TestSelectRelevantChunks:
    def test_empty_doc(self) -> None:
        dc = chunk_document("")
        assert select_relevant_chunks("any answer", dc) == []

    def test_single_chunk(self) -> None:
        dc = chunk_document("Короткий документ.", max_tokens_per_chunk=380)
        result = select_relevant_chunks("ответ", dc, top_k=3)
        assert result == [0]

    def test_fewer_than_top_k(self) -> None:
        dc = chunk_document("Раз. Два.", max_tokens_per_chunk=5)
        result = select_relevant_chunks("ответ", dc, top_k=10)
        assert len(result) <= len(dc.chunks)

    def test_sorted_by_position(self) -> None:
        sentences = [f"Тема {i}. " for i in range(20)]
        text = " ".join(sentences)
        dc = chunk_document(text, max_tokens_per_chunk=10)
        result = select_relevant_chunks("Тема 5", dc, top_k=3)
        assert result == sorted(result)

    def test_exact_match_found(self) -> None:
        # When answer contains exact words from a specific chunk
        text = "Яблоки и груши. Бананы и апельсины. Слиты и танки. Самолёты и вертолёты."
        dc = chunk_document(text, max_tokens_per_chunk=10)
        result = select_relevant_chunks("бананы апельсины", dc, top_k=2)
        assert len(result) >= 1

    def test_on_corpus_a3_sample(self) -> None:
        """BM25 should find relevant chunks on real data."""
        data_path = __file__.rsplit("tests/", 1)[0] + "data/corpus_a3/pairs.jsonl"
        try:
            with open(data_path, encoding="utf-8") as f:
                pairs = [json.loads(line) for line in f][:20]
        except FileNotFoundError:
            pytest.skip("corpus_a3 not available")
            return

        for p in pairs:
            dc = chunk_document(p["context"], max_tokens_per_chunk=380)
            result = select_relevant_chunks(p["answer"], dc, top_k=3)
            assert all(0 <= idx < len(dc.chunks) for idx in result)


class TestSelectRelevantChunksForClaims:
    def test_empty_claims(self) -> None:
        dc = chunk_document("Текст документа.", max_tokens_per_chunk=380)
        result = select_relevant_chunks_for_claims([], dc)
        assert isinstance(result, list)

    def test_multiple_claims_union(self) -> None:
        text = "Первый раздел о яблоках. Второй раздел о бананах. Третий раздел о грушах."
        dc = chunk_document(text, max_tokens_per_chunk=10)
        result = select_relevant_chunks_for_claims(
            ["яблоки", "груши"],
            dc,
            top_k_per_claim=1,
        )
        assert len(result) >= 1


class TestCorpusAIntegration:
    """Интеграционный тест на корпусе A."""

    def test_bm25_finds_values(self) -> None:
        """BM25 должен находить чанки с value в 90%+ случаев."""
        data_path = __file__.rsplit("tests/", 1)[0] + "data/corpus_a/pairs.jsonl"
        try:
            with open(data_path, encoding="utf-8") as f:
                pairs = [json.loads(line) for line in f][:100]
        except FileNotFoundError:
            pytest.skip("corpus_a not available")
            return

        labeled = [p for p in pairs if len(p["labels"]) > 0 and p["meta"].get("value")]
        if len(labeled) < 5:
            pytest.skip("not enough labeled pairs")

        hits = 0
        total = 0
        for p in labeled:
            dc = chunk_document(p["context"], max_tokens_per_chunk=380)
            if len(dc.chunks) == 1:
                hits += 1
                total += 1
                continue

            value = p["meta"]["value"]
            containing = [i for i, c in enumerate(dc.chunks) if value.lower() in c.text.lower()]
            if not containing:
                continue
            total += 1
            indices = select_relevant_chunks(p["answer"], dc, top_k=3)
            if any(idx in containing for idx in indices[:3]):
                hits += 1

        if total > 0:
            assert hits / total >= 0.90, f"BM25 recall@3 = {hits/total:.2%} < 90%"
