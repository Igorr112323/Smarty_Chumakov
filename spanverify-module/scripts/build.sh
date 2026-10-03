#!/usr/bin/env bash
# Сборка SpanVerify в один исполняемый файл под Linux/macOS.
#
#   bash scripts/build.sh              # без режима hf
#   bash scripts/build.sh --with-hf    # включить torch/transformers в сборку
#
# Результат: release/spanverify

set -euo pipefail
cd "$(dirname "$0")/.."

WITH_HF=0
SKIP_TESTS=0
for arg in "$@"; do
  case "$arg" in
    --with-hf) WITH_HF=1 ;;
    --skip-tests) SKIP_TESTS=1 ;;
    *) echo "Неизвестный аргумент: $arg"; exit 2 ;;
  esac
done

echo "== SpanVerify: сборка бинарника =="
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet -r requirements.txt
[ "$WITH_HF" = "1" ] && python3 -m pip install --quiet -r requirements-hf.txt

if [ "$SKIP_TESTS" = "0" ]; then
  echo "Прогон тестов..."
  python3 -m pytest -q
fi

if [ ! -f config/calibration.json ]; then
  echo "Калибратор не найден, обучаю на демо-корпусе..."
  python3 -m spanverify demo --n 240 --seed 1312 --out data/demo_dataset.jsonl
  python3 -m spanverify calibrate --dataset data/demo_dataset.jsonl --out config/calibration.json
fi

echo "Сборка PyInstaller..."
python3 -m PyInstaller spanverify.spec --noconfirm --clean

rm -rf release && mkdir -p release
cp dist/spanverify release/spanverify
cp -r config release/config
cp -r data release/data
cp README.md release/README.md 2>/dev/null || true

echo "Проверка собранного файла..."
./release/spanverify selftest

echo "Готово: release/spanverify ($(du -h release/spanverify | cut -f1))"
