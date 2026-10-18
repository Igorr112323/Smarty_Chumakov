#!/usr/bin/env python3
"""Минимальная версия measure_aggregators для диагностики"""
import sys
import time
from pathlib import Path

def main():
    print("=" * 60)
    print("Минимальная версия measure_aggregators")
    print("=" * 60)
    
    print("\n[1/3] Создание выходной директории...")
    out_dir = Path("reports/p0_answer_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Директория создана: {out_dir.absolute()}")
    
    print("\n[2/3] Создание тестового файла...")
    test_file = out_dir / "test.txt"
    test_file.write_text("Тестовый файл создан успешно\n", encoding="utf-8")
    print(f"✓ Файл создан: {test_file}")
    
    print("\n[3/3] Завершение...")
    print("=" * 60)
    print("✓ Минимальная версия выполнена успешно!")
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
