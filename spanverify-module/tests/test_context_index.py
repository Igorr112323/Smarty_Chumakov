"""Индекс слов контекста считает ровно то же, что и полный перебор.

Ускорение признаков (ТЗ п. 4.1 треб. 7 — быстродействие) сделано индексом:
слова документа сравниваются только с теми словами ответа, у которых есть общая
символьная триграмма. Это не приближение, а тот же расчёт, поэтому тесты ниже
сверяют быстрый путь с прямым перебором на реальных словах:

* :meth:`ContextWords.support` против прежней формулы «основа совпала → 1.0,
  иначе 0.7 × Jaccard по всем словам документа»;
* :meth:`ContextWords.similarities` против «доля по ``k`` ближайшим» и
  «ближайшее слово»;
* значения признаков ``embedding_density`` и ``ctx_max_similarity`` из
  :func:`demo_features` против тех же величин, посчитанных в лоб.

Если кто-то поменяет индекс так, что признаки разойдутся с прежним расчётом,
тесты упадут — тогда качество на корпусах изменилось бы незаметно.
"""

from __future__ import annotations

from spanverify.features import (
    DEMO_DENSITY_REFERENCE,
    ContextWords,
    _jaccard,
    _stem,
    _trigrams,
    demo_features,
)

DOCUMENT_WORDS = (
    "срок хранения первичных документов составляет десять лет для счета-фактуры "
    "предельный объём одного вложения составляет десять мегабайт регламент учёта "
    "вторичные документы и регистры хранятся в архиве организации"
).split()
ANSWER_WORDS = "срок хранения первичных документов три года аудитором ведомость".split()


def _brute_support(word: str, context_words: list[tuple[str, frozenset[str]]]) -> float:
    """Прежняя формула без индекса: перебор всех слов документа."""
    stem = _stem(word)
    grams: frozenset[str] | None = None
    best = 0.0
    for context_word, context_grams in context_words:
        if _stem(context_word) == stem:
            return 1.0
        if grams is None:
            grams = _trigrams(word)
        best = max(best, 0.7 * _jaccard(grams, context_grams))
    return best


def _brute_similarities(word: str, context_words: list[tuple[str, frozenset[str]]], k: int) -> tuple[float, float]:
    """Прежняя формула: полный список похожестей, сортировка, первые k."""
    grams = _trigrams(word)
    if not grams or not context_words:
        return 0.0, 0.0
    similarities = sorted((_jaccard(grams, grams_item) for _, grams_item in context_words), reverse=True)
    top = similarities[: max(1, k)]
    return sum(top) / len(top), similarities[0]


def _pairs() -> list[tuple[str, frozenset[str]]]:
    return [(word, _trigrams(word)) for word in DOCUMENT_WORDS]


def test_support_matches_brute_force() -> None:
    """Опора через индекс совпадает с перебором по всем словам документа."""
    pairs = _pairs()
    index = ContextWords(pairs)
    for word in ANSWER_WORDS + ["сторонний", "контрагент", "0012345"]:
        assert index.support(word) == _brute_support(word, pairs), word


def test_similarities_match_brute_force() -> None:
    """Доля по k ближайшим и ближайшее слово совпадают с перебором."""
    pairs = _pairs()
    index = ContextWords(pairs)
    for word in ANSWER_WORDS + ["архивный", "регистр", "сторонний"]:
        for k in (1, 3, 5, 8):
            assert index.similarities(word, k) == _brute_similarities(word, pairs, k), (
                word,
                k,
            )


def test_index_keeps_duplicate_words() -> None:
    """Повторы слов в документе учитываются так же, как при переборе."""
    pairs = [("документов", _trigrams("документов")) for _ in range(4)]
    index = ContextWords(pairs)
    assert index.similarities("документ", 5) == _brute_similarities("документ", pairs, 5)
    assert len(index) == 4


def test_empty_context_is_falsy_and_zero() -> None:
    """Пустой документ: индексировать нечего, значения нулевые."""
    index = ContextWords([])
    assert not index
    assert index.support("документов") == 0.0
    assert index.similarities("документов", 5) == (0.0, 0.0)


def test_demo_features_density_matches_manual_calculation() -> None:
    """Признаки плотности и близости совпадают с перебором по словам документа."""
    from spanverify.core import tokenize_with_offsets
    from spanverify.features import _is_content

    answer = "Срок хранения первичных документов составляет три года."
    context = "Регламент: срок хранения первичных документов составляет десять лет."
    matrix = demo_features(answer, context)
    tokens = tokenize_with_offsets(answer)
    pairs = [(token.word, _trigrams(token.word)) for token in tokenize_with_offsets(context) if _is_content(token)]
    assert pairs, "в документе должны быть содержательные слова"

    checked = 0
    for token, density, best in zip(tokens, matrix.embedding_density, matrix.ctx_max_similarity, strict=True):
        if not _is_content(token) or not _trigrams(token.word):
            assert density == 0.0 and best == 0.0, token.text
            continue
        raw, closest = _brute_similarities(token.word, pairs, 5)
        assert density == min(1.0, max(0.0, raw / DEMO_DENSITY_REFERENCE)), token.text
        assert best == min(1.0, max(0.0, closest)), token.text
        checked += 1
    assert checked >= 5, "проверка должна была коснуться содержательных слов ответа"
