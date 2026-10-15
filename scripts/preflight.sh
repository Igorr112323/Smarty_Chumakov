#!/bin/bash
# Preflight check перед коммитом: обновление METRICS.json и проверка чисел
#
# Использование:
#   cd spanverify-module
#   ../scripts/preflight.sh
#
# Что делает:
#   1. Запускает pytest с покрытием
#   2. Обновляет METRICS.json из coverage.json
#   3. Запускает check_numbers.py для сверки чисел в документах

set -e

echo "=== Preflight check ==="

# Проверяем, что мы в spanverify-module
if [ ! -f "pyproject.toml" ] || [ ! -d "tests" ]; then
    echo "Ошибка: запустите из директории spanverify-module"
    exit 1
fi

# Шаг 1: pytest с покрытием
echo ""
echo "Шаг 1: pytest с покрытием..."
python3 -m pytest --timeout=60 --cov=spanverify --cov-report=json:reports/coverage.json -q

# Шаг 2: обновление METRICS.json
echo ""
echo "Шаг 2: обновление METRICS.json..."
python3 scripts/collect_metrics.py --coverage-json reports/coverage.json

# Шаг 3: проверка чисел
echo ""
echo "Шаг 3: проверка чисел в документах..."
python3 ../scripts/check_numbers.py --metrics reports/METRICS.json

echo ""
echo "=== Preflight check завершён ==="
echo "Если все шаги прошли — можно коммитить."
