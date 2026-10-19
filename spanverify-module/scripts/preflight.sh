#!/bin/bash
# Предполётная проверка перед коммитом/пушом
# Запускает collect_metrics и check_numbers, чтобы избежать расхождений чисел

set -e

echo "=== Preflight check ==="

# Переходим в корень модуля
cd "$(dirname "$0")/.."

# 0. Проверка маркеров конфликтов в документах
echo "0. Проверка маркеров конфликтов..."
CONFLICTS=$(grep -rn "^<<<<<<<\|^>>>>>>>\|^=======$" --exclude-dir=.git --exclude-dir=__pycache__ --exclude-dir=.pytest_cache .. 2>/dev/null || true)
if [ -n "$CONFLICTS" ]; then
    echo "ОШИБКА: обнаружены маркеры конфликтов в документах:"
    echo "$CONFLICTS"
    echo "Разрешите конфликты перед коммитом."
    exit 1
fi
echo "   ✓ маркеры конфликтов отсутствуют"

# 1. Запускаем collect_metrics (обновляет METRICS.json)
echo "1. Запуск collect_metrics.py..."
python3 scripts/collect_metrics.py --coverage-json reports/coverage.json 2>&1 | tail -5

# 2. Запускаем check_numbers (проверяет документы на соответствие METRICS.json)
echo ""
echo "2. Запуск check_numbers.py..."
python3 scripts/check_numbers.py --metrics reports/METRICS.json

echo ""
echo "=== Preflight check завершён ==="
echo "Если всё прошло, можно коммитить."
echo "Если были ошибки — исправьте расхождения в документах."
