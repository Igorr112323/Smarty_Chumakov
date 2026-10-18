# P0 Шаг 0: Инфраструктура для промежуточных прогонов

**Дата:** 2026-10-17  
**Коммит:** d471e45  
**Статус:** ✅ Выполнено

## Что сделано

### 1. Флаг `--no-test` в `run_experiments.py`

**Изменения:**
- Добавлен аргумент `--no-test` в argparse
- Блок оценки официального теста обёрнут в `if not args.no_test`
- При `--no-test` печатается "ТЕСТ НЕ ИЗМЕРЯЛСЯ (--no-test)"
- В `experiment.json` устанавливается `test_official: null`
- В `experiment.md` выводится секция "Официальный тест не измерялся"

**Зачем:**
Промежуточные прогоны P0-шагов 1-3 (границы фрагментов, оценка ответа, калибровка вердикта) не должны трогать официальный тест. Это защищает от случайного подбора параметров под тестовые данные.

**Проверка:**
```bash
python scripts/run_experiments.py --dataset data/corpus_a3/pairs.jsonl \
  --mode hf --features-cache reports/hf-cache-v2 --no-test \
  --out reports/experiments/p0_test
```

Ожидаемый вывод:
```
ТЕСТ НЕ ИЗМЕРЯЛСЯ (--no-test)
```

### 2. Флаг `--features-cache` в `calibrate_verdict.py`

**Изменения:**
- Добавлен аргумент `--features-cache` (action="append", можно указать несколько раз)
- Импортированы `load_feature_cache` и `set_feature_cache` из `spanverify.features`
- Кеш загружается после парсинга аргументов и передаётся в `Verifier` и `train`
- В конце выводится счётчик hits/misses

**Зачем:**
Калибровка вердикта в режиме hf без кеша пересчитывает признаки моделью для каждой пары (~6.4 с/пара × 1200 пар ≈ 2 часа). С кешем — минуты.

**Проверка:**
```bash
python scripts/calibrate_verdict.py \
  --dataset data/corpus_a3/pairs.jsonl --mode hf \
  --features-cache reports/hf-cache-v2 \
  --out reports/verdict_calibration.json \
  --freeze config/verdict.json
```

Ожидаемый вывод:
```
кеш признаков: 1195 ключей
...
кеш признаков: попаданий N, промахов M
```

## Критерии приёмки

✅ **644 теста проходят** (проверено `python -m pytest tests/ -q`)  
✅ **Линтер и форматтер** (проверено `ruff check` и `black --check`)  
✅ **Синтаксис** (проверено `python -m py_compile`)

## Следующий шаг

**P0-1 (Шаг 1): Границы фрагментов**

Цель: сузить фрагменты от 9.1× до ≤2.0× ширины, поднять strict F1 с 0.161 до ≥0.45.

Команда для промежуточных прогонов:
```bash
python scripts/run_experiments.py \
  --dataset data/corpus_a3/pairs.jsonl \
  --mode hf \
  --features-cache reports/hf-cache-v2 \
  --no-test \
  --out reports/experiments/p0_step1_vARIANT
```

Метрики для выбора варианта (по dev):
- strict F1 фрагментов (IoU ≥ 0.5) — основная
- mean_width_ratio — должна упасть с 9.1×
- recall_containment — не должна просесть ниже 0.45
- токены идентичны (tokens_identical: true)

Варианты:
1. `clause` (текущий) — расширение до границ клаузы
2. `mask` — границы ровно по помеченным токенам
3. `clause_capped` — клауза, но не шире max(40 знаков, 2 × ширина маски)
4. `mask_padded` — маска + по одному соседнему токену слева и справа

## Файлы

- `spanverify-module/scripts/run_experiments.py` — добавлен `--no-test`
- `spanverify-module/scripts/calibrate_verdict.py` — добавлен `--features-cache`
