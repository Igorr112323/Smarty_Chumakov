"""Метрики бенчмарка RusHallu-RAG (перевод их ``metrics/span_metrics.py`` на stdlib).

Зачем копия, а не импорт
------------------------

Их код требует ``numpy`` и ``rouge_score``. Чтобы оценка внешнего бенчмарка шла в CI
без тяжёлых зависимостей, метрики повторены здесь на стандартной библиотеке —
**дословно по смыслу** их файла ``metrics/span_metrics.py`` (закреплённый коммит
``345907f9``): та же токенизация по символам, тот же выбор лучшего предсказания по
среднему F-measure, те же формулы accuracy / Jaccard / hamming.

Отличия (и почему)
------------------

* Деление на ноль: у них при пустом истинном наборе спанов получается ``nan``.
  Здесь такие случаи посчитаны отдельно: Jaccard = 1.0, hamming = 0.0, а число
  таких пар попадает в отчёт (``empty_reference``).
* ROUGE: своя реализация ROUGE-1 / ROUGE-2 / ROUGE-L с символьной токенизацией.
  Совпадение с ``rouge_score`` (``use_stemmer=False``, ``tokenizer=list``) проверено
  тестом ``test_rouge_matches_rouge_score_library`` — тест включается, если библиотека
  установлена.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence


def char_tokens(text: str) -> list[str]:
    """Токенизация как в их ``Tokenizer``: каждый символ — токен."""
    return list(text)


def _ngrams(tokens: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    """Счётчик n-грамм (для ROUGE-N)."""
    return Counter(tuple(tokens[index : index + n]) for index in range(max(0, len(tokens) - n + 1)))


def _f_measure(overlap: float, reference_len: float, predicted_len: float) -> float:
    """F-measure как в ROUGE: гармоническое среднее точности и полноты."""
    if overlap <= 0 or reference_len <= 0 or predicted_len <= 0:
        return 0.0
    precision = overlap / predicted_len
    recall = overlap / reference_len
    return 2 * precision * recall / (precision + recall)


def rouge_n(reference: str, predicted: str, n: int = 1) -> float:
    """ROUGE-N по символам (n = 1, 2)."""
    reference_grams = _ngrams(char_tokens(reference), n)
    predicted_grams = _ngrams(char_tokens(predicted), n)
    overlap = sum((reference_grams & predicted_grams).values())
    return _f_measure(overlap, sum(reference_grams.values()), sum(predicted_grams.values()))


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    """Длина наибольшей общей подпоследовательности (две строки памяти)."""
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(reference: str, predicted: str) -> float:
    """ROUGE-L по символам (F-measure на длине НОП)."""
    overlap = _lcs_length(char_tokens(reference), char_tokens(predicted))
    return _f_measure(overlap, len(reference), len(predicted))


def score_pair(reference: str, predicted: str) -> dict[str, float]:
    """Все три ROUGE для пары «истинный спан — предсказанный спан»."""
    return {
        "rouge1": rouge_n(reference, predicted, 1),
        "rouge2": rouge_n(reference, predicted, 2),
        "rougeL": rouge_l(reference, predicted),
    }


ROUGE_METRICS = ("rouge1", "rouge2", "rougeL")


def best_rouge_for(reference: str, candidates: Sequence[str]) -> dict[str, float]:
    """Лучшее по среднему F-measure предсказание для одного истинного спана."""
    if not candidates:
        return dict.fromkeys(ROUGE_METRICS, 0.0)
    scored = [score_pair(reference, candidate) for candidate in candidates]
    means = [sum(item.values()) / len(item) for item in scored]
    return scored[means.index(max(means))]


def calculate_rouge_scores(
    first_batch_list: Sequence[Sequence[str]], second_batch_list: Sequence[Sequence[str]]
) -> dict:
    """ROUGE по образцу их функции ``calculate_rouge_scores``.

    В их ``evaluate_json`` вызов такой: ``calculate_rouge_scores(pred_spans,
    true_spans)``, а внутри — «для каждого элемента первого списка выбрать лучшее
    совпадение во втором». Порядок сохранён дословно, имена параметров нейтральные.

    Особый случай из их кода: если два списка совпали целиком — ROUGE = 1.0.
    """
    per_pair: list[dict[str, float]] = []
    for first_batch, second_batch in zip(first_batch_list, second_batch_list, strict=False):
        if list(first_batch) == list(second_batch):
            per_pair.append(dict.fromkeys(ROUGE_METRICS, 1.0))
            continue
        items = [best_rouge_for(reference, second_batch) for reference in first_batch]
        if not items:
            per_pair.append(dict.fromkeys(ROUGE_METRICS, 0.0))
            continue
        per_pair.append({metric: sum(item[metric] for item in items) / len(items) for metric in ROUGE_METRICS})
    if not per_pair:
        return dict.fromkeys(ROUGE_METRICS, 0.0)
    return {metric: sum(item[metric] for item in per_pair) / len(per_pair) for metric in ROUGE_METRICS}


def calculate_match_metrics(predicted_spans: Sequence[Sequence[str]], reference_spans: Sequence[Sequence[str]]) -> dict:
    """accuracy / Jaccard / hamming — как в их ``calculate_match_metrics``."""
    accuracy: list[float] = []
    jaccard: list[float] = []
    hamming: list[float] = []
    empty_reference = 0
    for true_batch, pred_batch in zip(reference_spans, predicted_spans, strict=False):
        true_set, pred_set = set(true_batch), set(pred_batch)
        accuracy.append(1.0 if true_set == pred_set else 0.0)
        union = true_set | pred_set
        if not union:
            jaccard.append(1.0)
        else:
            jaccard.append(len(true_set & pred_set) / len(union))
        if not true_set:
            empty_reference += 1
            hamming.append(0.0)
        else:
            hamming.append((len(true_set) - len(true_set & pred_set)) / len(true_set))
    count = len(accuracy) or 1
    return {
        "accuracy": sum(accuracy) / count,
        "jaccard_score": sum(jaccard) / count,
        "hamming_loss": sum(hamming) / count,
        "empty_reference": empty_reference,
    }


def evaluate_spans(reference: Iterable[Iterable[str]], predictions: Iterable[Iterable[str]]) -> dict:
    """Полный набор их метрик: ROUGE + совпадения (порядок аргументов как в их коде)."""
    reference_spans = [list(batch) for batch in reference]
    predicted_spans = [list(batch) for batch in predictions]
    result = calculate_rouge_scores(predicted_spans, reference_spans)
    result.update(calculate_match_metrics(predicted_spans, reference_spans))
    result["pairs"] = len(reference_spans)
    return result
