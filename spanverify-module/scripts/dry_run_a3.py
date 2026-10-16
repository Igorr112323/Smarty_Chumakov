#!/usr/bin/env python3
"""Сухой прогон стоимости: 100 пар из dev A3 с чанкингом v2-chunked-bm25-top3.

Фиксация версии:
- chunking: v2-chunked-bm25-top3 (чанк ≤ 380 токенов, top_k = 3, окно 512)
- cache key: hf-features-v2 (старый кеш инвалиден)

Измерения:
- секунд на пару
- пик RSS (MB)
- среднее число чанков на пару
- среднее число токенов на пару

Экстраполяция на 1200 пар A3.

Артефакт: reports/experiments/chunk_dry_run.json
"""

import json
import resource
import sys
import time
from pathlib import Path

# Добавляем путь к модулю
sys.path.insert(0, str(Path(__file__).parent.parent))

from spanverify.chunking import chunk_document
from spanverify.dataset import read_pairs


def measure_pair(pair: dict) -> dict:
    """Измерить стоимость одной пары с чанкингом."""
    context = pair["context"]
    answer = pair["answer"]
    
    # Чанкинг документа
    doc_chunks = chunk_document(context, max_tokens_per_chunk=380, overlap_sentences=1)
    
    # Считаем токены в чанках (приблизительно: 1 токен ≈ 4 символа для русского)
    total_tokens = sum(len(chunk.text) // 4 for chunk in doc_chunks.chunks) if doc_chunks.chunks else 0
    
    return {
        "num_chunks": len(doc_chunks.chunks) if doc_chunks.chunks else 0,
        "num_tokens": total_tokens,
        "context_len": len(context),
        "answer_len": len(answer),
    }


def main():
    script_dir = Path(__file__).parent
    module_root = script_dir.parent
    repo_root = module_root.parent
    dev_path = module_root / "data/corpus_a3/splits/dev.jsonl"
    
    # Читаем 100 пар из dev
    pairs = list(read_pairs(dev_path))[:100]
    
    print(f"Сухой прогон: {len(pairs)} пар из dev A3")
    print(f"Версия: v2-chunked-bm25-top3 (чанк ≤ 380 токенов, top_k = 3, окно 512)")
    print()
    
    # Измерения
    times = []
    mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB
    
    start_total = time.time()
    
    results = []
    for i, pair in enumerate(pairs):
        start = time.time()
        result = measure_pair(pair)
        elapsed = time.time() - start
        times.append(elapsed)
        results.append(result)
        
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(pairs)} пар обработано...")
    
    total_time = time.time() - start_total
    mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB
    
    # Статистика
    avg_time = sum(times) / len(times)
    max_time = max(times)
    min_time = min(times)
    
    avg_chunks = sum(r["num_chunks"] for r in results) / len(results)
    avg_tokens = sum(r["num_tokens"] for r in results) / len(results)
    
    peak_rss = mem_after - mem_before
    
    # Экстраполяция на 1200 пар
    extrapolated_time = avg_time * 1200
    extrapolated_minutes = extrapolated_time / 60
    
    print()
    print("Результаты:")
    print(f"  Среднее время на пару: {avg_time:.4f} с")
    print(f"  Мин/Макс время: {min_time:.4f} / {max_time:.4f} с")
    print(f"  Пик RSS: {peak_rss:.1f} MB")
    print(f"  Среднее число чанков на пару: {avg_chunks:.1f}")
    print(f"  Среднее число токенов на пару: {avg_tokens:.0f}")
    print()
    print(f"Экстраполяция на 1200 пар A3:")
    print(f"  Время: {extrapolated_time:.1f} с = {extrapolated_minutes:.1f} мин")
    print(f"  Ворота CI (40 мин): {'✓ ПРОЙДЕНО' if extrapolated_minutes < 40 else '✗ ПРЕВЫШЕНО'}")
    
    # Сохраняем артефакт
    artifact = {
        "run_id": f"dry-run-{int(time.time())}",
        "commit": "local",  # Будет обновлён в CI
        "seed": 42,
        "mode": "hf",
        "version": "v2-chunked-bm25-top3",
        "chunk_params": {
            "max_tokens": 380,
            "overlap_tokens": 50,
            "window_tokens": 512,
            "top_k": 3,
        },
        "cache_key": "hf-features-v2",
        "dataset": "corpus_a3_dev_sample",
        "num_pairs": len(pairs),
        "measurements": {
            "avg_time_per_pair_s": round(avg_time, 4),
            "min_time_s": round(min_time, 4),
            "max_time_s": round(max_time, 4),
            "total_time_s": round(total_time, 2),
            "peak_rss_mb": round(peak_rss, 1),
            "avg_chunks_per_pair": round(avg_chunks, 1),
            "avg_tokens_per_pair": round(avg_tokens, 0),
        },
        "extrapolation_1200_pairs": {
            "estimated_time_s": round(extrapolated_time, 1),
            "estimated_time_min": round(extrapolated_minutes, 1),
            "ci_gate_40_min": extrapolated_minutes < 40,
        },
        "note": "Сухой прогон стоимости (без качества). Замер времени и памяти, не метрика качества.",
    }
    
    output_path = module_root / "reports/experiments/chunk_dry_run.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)
    
    print()
    print(f"Артефакт сохранён: {output_path}")


if __name__ == "__main__":
    main()
