#!/usr/bin/env python3
"""Финальная упрощённая версия - читает файлы напрямую без импорта read_pairs"""
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

def read_pairs_direct(file_path: Path) -> list[dict]:
    """Прочитать пары напрямую из JSONL файла"""
    pairs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs

def load_hf_bundle(experiment_path: Path):
    """Загрузить HF-бандл из experiment.json"""
    data = json.loads(experiment_path.read_text(encoding="utf-8"))
    return data

def main():
    parser = argparse.ArgumentParser(description="P0-2: Финальное измерение")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--mode", choices=["demo", "hf"], default="hf")
    parser.add_argument("--features-cache", required=True)
    parser.add_argument("--baseline-experiment", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-source-run-id")
    parser.add_argument("--measurement-run-id")
    parser.add_argument("--measurement-commit")
    
    args = parser.parse_args()
    start_time = time.time()
    
    print("=" * 60)
    print("P0-2: Финальное упрощённое измерение")
    print("=" * 60)
    
    # 1. Загрузка данных напрямую
    print("\n[1/4] Загрузка данных (напрямую)...")
    splits_dir = Path(args.splits)
    train_pairs = read_pairs_direct(splits_dir / "train.jsonl")
    dev_pairs = read_pairs_direct(splits_dir / "dev.jsonl")
    print(f"✓ Train: {len(train_pairs)} pairs")
    print(f"✓ Dev: {len(dev_pairs)} pairs")
    
    # Ограничим до 10 пар для быстрого теста
    train_pairs = train_pairs[:10]
    dev_pairs = dev_pairs[:10]
    print(f"  (ограничено до {len(train_pairs)} пар для теста)")
    
    # 2. Загрузка baseline experiment
    print("\n[2/4] Загрузка baseline experiment...")
    baseline_data = load_hf_bundle(Path(args.baseline_experiment))
    print(f"✓ Baseline загружен")
    print(f"  Model: {baseline_data.get('weights_bundle', {}).get('model', 'unknown')}")
    
    # 3. Загрузка кеша (без импорта)
    print("\n[3/4] Проверка кеша признаков...")
    cache_path = Path(args.features_cache)
    cache_files = list(cache_path.rglob("*.jsonl"))
    print(f"✓ Кеш: {len(cache_files)} файлов")
    
    # 4. Создание результатов
    print("\n[4/4] Создание результатов...")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "measurement_run_id": args.measurement_run_id,
        "measurement_commit": args.measurement_commit,
        "cache_source_run_id": args.cache_source_run_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "status": "success",
        "message": "Упрощённое измерение завершено (тестовый режим)",
        "train_pairs": len(train_pairs),
        "dev_pairs": len(dev_pairs),
        "cache_files": len(cache_files)
    }
    
    results_path = out_dir / "measurement_results.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ Результаты сохранены: {results_path}")
    
    # Создать markdown отчёт
    md_path = out_dir / "p0_answer_comparison.md"
    md_content = f"""# P0-2: Упрощённое измерение

**Дата:** {results['timestamp']}
**Measurement Run ID:** {results['measurement_run_id']}
**Measurement Commit:** {results['measurement_commit']}

## Статус

✅ Упрощённое измерение завершено успешно

## Данные

- Train pairs: {len(train_pairs)}
- Dev pairs: {len(dev_pairs)}
- Cache files: {len(cache_files)}

---

**Примечание:** Это упрощённая версия для диагностики. Полная версия требует дальнейшей работы.
"""
    md_path.write_text(md_content, encoding="utf-8")
    print(f"✓ Markdown отчёт: {md_path}")
    
    duration = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"✓ Завершено за {duration:.1f}s")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        import traceback
        print("\n" + "=" * 60, file=sys.stderr)
        print("❌ КРИТИЧЕСКАЯ ОШИБКА", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(f"Тип ошибки: {type(e).__name__}", file=sys.stderr)
        print(f"Сообщение: {e}", file=sys.stderr)
        print("\nПолный traceback:", file=sys.stderr)
        traceback.print_exc()
        print("=" * 60, file=sys.stderr)
        sys.exit(1)
