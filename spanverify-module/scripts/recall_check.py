#!/usr/bin/env python3
"""Воспроизведение чисел chunk_selection.json.

Этот скрипт вычисляет recall@k для отбора релевантных чанков BM25 на корпусах A и A3.

Запуск:
    cd spanverify-module
    python3 scripts/recall_check.py

Результат:
    Печатает recall@1, recall@3, recall@5 для обоих корпусов с двумя критериями:
    - soft: любое пересечение чанка с релевантным регионом
    - coverage_50: чанк покрывает ≥50% релевантного региона
"""
import json
import sys
from pathlib import Path

# Добавляем текущую директорию в путь для импорта spanverify
sys.path.insert(0, str(Path(__file__).parent.parent))

from spanverify.chunking import chunk_document, select_relevant_chunks
from spanverify.dataset import read_pairs


def calc_recall(pairs, corpus_name, criteria="soft"):
    """Вычисляет recall@k для заданного корпуса и критерия попадания."""
    results = {"recall@1": 0, "recall@3": 0, "recall@5": 0, "n": 0}

    for p in pairs:
        if not p.get("labels"):
            continue

        context = p.get("context", "")
        answer = p.get("answer", "")
        meta = p.get("meta", {})

        dc = chunk_document(context, max_tokens_per_chunk=380)

        for label in p["labels"]:
            start, end, cls = label
            if cls == 0:
                continue

            if corpus_name == "a3":
                fact_span = meta.get("fact_span")
                context_span = meta.get("context_span")

                if not fact_span or not context_span:
                    continue

                # Релевантный регион: fact_span относительно начала контекста
                rel_start = max(0, fact_span[0] - context_span[0])
                rel_end = min(len(context), fact_span[1] - context_span[0])

                if rel_start >= rel_end or rel_start >= len(context):
                    continue

                region_length = rel_end - rel_start

                # Определяем чанки, попадающие в регион
                containing = []
                for i, c in enumerate(dc.chunks):
                    if criteria == "soft":
                        # Любое пересечение
                        if c.start < rel_end and c.end > rel_start:
                            containing.append(i)
                    elif criteria == "coverage_50":
                        # Покрытие ≥50% региона
                        overlap_start = max(c.start, rel_start)
                        overlap_end = min(c.end, rel_end)
                        if overlap_end > overlap_start:
                            overlap_length = overlap_end - overlap_start
                            if overlap_length / region_length >= 0.5:
                                containing.append(i)

                if not containing:
                    continue

                results["n"] += 1

                # Для recall@1 используем top_k=1 (истинный топ-1)
                indices_1 = select_relevant_chunks(answer, dc, top_k=1)
                if indices_1 and indices_1[0] in containing:
                    results["recall@1"] += 1

                # Для recall@3 используем top_k=3
                indices_3 = select_relevant_chunks(answer, dc, top_k=3)
                if any(idx in containing for idx in indices_3):
                    results["recall@3"] += 1

                # Для recall@5 используем top_k=5
                indices_5 = select_relevant_chunks(answer, dc, top_k=5)
                if any(idx in containing for idx in indices_5):
                    results["recall@5"] += 1

            else:  # corpus_a
                value = meta.get("value", "")
                if not value or len(value) < 3:
                    continue

                value_lower = value.lower()
                containing = []
                for match_start in range(len(context)):
                    idx = context.lower().find(value_lower, match_start)
                    if idx == -1:
                        break

                    for i, c in enumerate(dc.chunks):
                        if criteria == "soft":
                            if c.start <= idx < c.end and i not in containing:
                                containing.append(i)
                        elif criteria == "coverage_50":
                            if c.start <= idx < c.end:
                                region_length = len(value)
                                overlap_start = max(c.start, idx)
                                overlap_end = min(c.end, idx + region_length)
                                if overlap_end > overlap_start:
                                    overlap_length = overlap_end - overlap_start
                                    if (
                                        overlap_length / region_length >= 0.5
                                        and i not in containing
                                    ):
                                        containing.append(i)

                    match_start = idx + 1

                if not containing:
                    continue

                results["n"] += 1

                # Для recall@1 используем top_k=1 (истинный топ-1)
                indices_1 = select_relevant_chunks(answer, dc, top_k=1)
                if indices_1 and indices_1[0] in containing:
                    results["recall@1"] += 1

                # Для recall@3 используем top_k=3
                indices_3 = select_relevant_chunks(answer, dc, top_k=3)
                if any(idx in containing for idx in indices_3):
                    results["recall@3"] += 1

                # Для recall@5 используем top_k=5
                indices_5 = select_relevant_chunks(answer, dc, top_k=5)
                if any(idx in containing for idx in indices_5):
                    results["recall@5"] += 1

    if results["n"] > 0:
        return {
            "n": results["n"],
            "recall@1": round(results["recall@1"] / results["n"], 4),
            "recall@3": round(results["recall@3"] / results["n"], 4),
            "recall@5": round(results["recall@5"] / results["n"], 4),
        }
    return {"n": 0}


def main():
    """Главная функция."""
    data_root = Path("data")

    # Корпус A3
    print("=== Корпус A3 (реальные акты) ===")
    pairs_a3 = list(read_pairs(data_root / "corpus_a3" / "pairs.jsonl"))
    labeled_a3 = [p for p in pairs_a3 if p.get("labels")]
    print(f"Размеченных пар: {len(labeled_a3)}")

    a3_soft = calc_recall(labeled_a3, "a3", "soft")
    a3_cov50 = calc_recall(labeled_a3, "a3", "coverage_50")

    print("Soft (любое пересечение):")
    print(
        f"  n={a3_soft['n']}, recall@1={a3_soft['recall@1']}, "
        f"recall@3={a3_soft['recall@3']}, recall@5={a3_soft['recall@5']}"
    )
    print("Coverage ≥50%:")
    print(
        f"  n={a3_cov50['n']}, recall@1={a3_cov50['recall@1']}, "
        f"recall@3={a3_cov50['recall@3']}, recall@5={a3_cov50['recall@5']}"
    )

    # Корпус A
    print("\n=== Корпус A (шаблонные документы) ===")
    pairs_a = list(read_pairs(data_root / "corpus_a" / "pairs.jsonl"))
    labeled_a = [p for p in pairs_a if p.get("labels")]
    print(f"Размеченных пар: {len(labeled_a)}")

    a_soft = calc_recall(labeled_a, "a", "soft")
    a_cov50 = calc_recall(labeled_a, "a", "coverage_50")

    print("Soft (любое пересечение):")
    print(
        f"  n={a_soft['n']}, recall@1={a_soft['recall@1']}, "
        f"recall@3={a_soft['recall@3']}, recall@5={a_soft['recall@5']}"
    )
    print("Coverage ≥50%:")
    print(
        f"  n={a_cov50['n']}, recall@1={a_cov50['recall@1']}, "
        f"recall@3={a_cov50['recall@3']}, recall@5={a_cov50['recall@5']}"
    )

    # Проверка ворот G1
    print("\n=== Ворота G1 (recall@5 ≥ 95% с покрытием ≥50%) ===")
    a3_pass = a3_cov50["recall@5"] >= 0.95
    a_pass = a_cov50["recall@5"] >= 0.95
    print(f"A3: {a3_cov50['recall@5']} {'✓' if a3_pass else '✗'}")
    print(f"A:  {a_cov50['recall@5']} {'✓' if a_pass else '✗'}")

    if a3_pass and a_pass:
        print("\n✓ Ворота G1 взяты с запасом на обоих корпусах")
    else:
        print("\n✗ Ворота G1 НЕ взяты")


if __name__ == "__main__":
    main()
