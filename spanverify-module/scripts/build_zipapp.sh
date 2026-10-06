#!/usr/bin/env bash
# Сборка SpanVerify в один переносимый файл (zipapp).
#
# Быстрая альтернатива PyInstaller: получается один файл .pyz, которому
# нужен установленный Python 3.10+ на целевой машине. Конфигурация и
# калибратор упаковываются внутрь архива.
#
#   bash scripts/build_zipapp.sh          -> release/spanverify.pyz

set -euo pipefail
cd "$(dirname "$0")/.."

echo "== SpanVerify: сборка zipapp =="
python3 -m pytest -q

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp -r spanverify "$STAGE/"
cp -r config "$STAGE/"
cp -r data "$STAGE/"
find "$STAGE" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

mkdir -p release
python3 -m zipapp "$STAGE" -m "spanverify.cli:main" -o release/spanverify.pyz -p "/usr/bin/env python3"
chmod +x release/spanverify.pyz

echo "Проверка собранного архива..."
(cd /tmp && "$OLDPWD/release/spanverify.pyz" selftest)

echo "Готово: release/spanverify.pyz ($(du -h release/spanverify.pyz | cut -f1))"
