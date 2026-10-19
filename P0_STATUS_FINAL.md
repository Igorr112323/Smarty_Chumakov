# P0-2: Статус сравнения агрегаторов ответов

**Дата:** 2026-10-18  
**Статус:** ⚠️ Частично завершено (технические ограничения CI)

## Что было сделано

### ✅ Выполнено

1. **Исправлен баг в `_compress()`** (`calibration.py`)
   - Проблема: функция схлопывала кривую к последней точке
   - Решение: отслеживание tolerance от последней сохранённой точки
   - Коммит: `b603f8c`

2. **Исправлен тест `test_compress_exact_mode`**
   - Проблема: неправильное ожидание для scores=[.1,.2,.3], labels=[0,1,1]
   - Решение: правильные ожидания thresholds=[0.1,0.2,0.3], values=[0,1,1,1.0]
   - Коммит: `b603f8c`

3. **Создан workflow для измерения агрегаторов**
   - Файл: `.github/workflows/postprocess-p0-answer.yml`
   - Загружает кеш признаков из run 39060670054
   - Загружает baseline experiment (clause_capped)
   - Выполняет регрессионные тесты
   - Сохраняет результаты в уникальную директорию

4. **Создана рабочая основа для измерения**
   - Файл: `spanverify-module/scripts/measure_final.py`
   - Читает данные напрямую (без проблемных импортов)
   - Создаёт базовые результаты и отчёт
   - Работает в CI среде успешно

5. **Полная версия скрипта сравнения**
   - Файл: `spanverify-module/scripts/measure_aggregators.py`
   - Реализует сравнение 3 агрегаторов (current, max, mean_top3)
   - Использует документ-group OOF калибровку
   - Выбирает порог с ограничением FPR ≤ 0.30
   - Сохраняет полные результаты и evidence

### ⚠️ Проблемы

**Критическая проблема CI среды:**
- Шаг измерения падает за 0 секунд при использовании импортов из `spanverify.dataset`, `spanverify.engine`, `spanverify.features`, `spanverify.calibration`
- Причина не может быть определена из-за ограничений sandbox (недоступны логи)
- Минимальные импорты работают (проверено в `test_minimal.py`)
- Сложные импорты вызывают немедленное падение шага

**Диагностика:**
- Создан `test_minimal.py` - проходит успешно
- Создан `measure_aggregators_diag.py` - падает при добавлении импортов
- Создан `measure_final.py` - работает без сложных импортов

## Что требуется для завершения P0-2

### Вариант 1: Локальное выполнение

Полная версия `measure_aggregators.py` может быть выполнена локально:

```bash
cd spanverify-module

python scripts/measure_aggregators.py \
  --dataset data/corpus_a3/pairs.jsonl \
  --splits data/corpus_a3/splits \
  --mode hf \
  --features-cache reports/hf-cache-v2 \
  --baseline-experiment reports/experiments/p0_span/run39060670054/clause_capped/experiment.json \
  --out reports/p0_answer_comparison \
  --cache-source-run-id 39060670054 \
  --measurement-run-id $(date +%s) \
  --measurement-commit $(git rev-parse HEAD)
```

Результаты будут сохранены в `reports/p0_answer_comparison/`:
- `aggregator_comparison.json` - полные результаты
- `p0_answer_comparison.md` - markdown отчёт
- `evidence_*.jsonl` - evidence для каждого агрегатора

### Вариант 2: Исправление CI среды

Требуется investigate почему импорты из spanverify.* вызывают падение шага в GitHub Actions. Возможные причины:
- Проблемы с зависимостями (torch, transformers)
- Проблемы с путями импорта
- Проблемы с инициализацией модулей
- Проблемы с памятью или ресурсами

## Результаты (упрощённая версия)

**Run ID:** 39081416935  
**Commit:** 7d3aa94246a04ece5dedbce17a0b4df764f3f694  
**Дата:** 2026-10-18T00:05:11.985211Z

### Данные
- Train pairs: 10 (ограничено для теста)
- Dev pairs: 10 (ограничено для теста)
- Cache files: 8

### Статус
✅ Упрощённое измерение завершено успешно

**Примечание:** Это упрощённая версия для диагностики. Полная версия требует выполнения локально или исправления CI среды.

## Файлы

### Созданные/изменённые файлы

1. `spanverify-module/scripts/measure_aggregators.py` - полная версия (требует локального выполнения)
2. `spanverify-module/scripts/measure_final.py` - упрощённая версия (работает в CI)
3. `spanverify-module/scripts/test_minimal.py` - минимальный диагностический тест
4. `spanverify-module/scripts/measure_aggregators_diag.py` - диагностическая версия
5. `spanverify-module/scripts/measure_aggregators_simple.py` - упрощённая версия с импортами
6. `spanverify-module/scripts/measure_aggregators_minimal.py` - минимальная версия
7. `spanverify-module/spanverify/calibration.py` - исправлен _compress
8. `spanverify-module/tests/test_calibration_fix.py` - исправлен тест
9. `.github/workflows/postprocess-p0-answer.yml` - workflow для измерения
10. `spanverify-module/reports/experiments/p0_answer/run39081416935/` - результаты упрощённой версии

### Результаты из предыдущих runs

**Run 39060670054 (baseline):**
- Token control: PASSED (TP=138, FP=636, FN=149, TN=7046)
- Answer F1=0.6099, FPR=0.6941, AUC=0.5047, threshold=0.595122
- Это был single-aggregator measurement (не полное сравнение)

## Следующие шаги

1. **Выполнить полную версию локально** для получения реальных результатов сравнения 3 агрегаторов
2. **Или investigate и исправить** проблемы с импортами в CI среде
3. **Сравнить результаты** 3 агрегаторов (current, max, mean_top3)
4. **Выбрать лучший агрегатор** на основе FPR ≤ 0.30 constraint
5. **Сохранить evidence** и обновить документацию

## Выводы

P0-2 частично завершено:
- ✅ Исправлены баги в калибровке
- ✅ Создан workflow для измерения
- ✅ Создана рабочая основа
- ⚠️ Полное сравнение 3 агрегаторов требует локального выполнения или исправления CI

Технический долг: необходимо investigate проблемы с импортами в GitHub Actions environment.
