# P0: Исправление изоляции теста и подготовка к dev-сравнению

**Дата:** 2026-10-17  
**Коммит:** (будет заполнен после commit)  
**Статус:** ✅ Исправления выполнены, готовы к запуску

## Критические дефекты, исправленные в этом коммите

### 1. Изоляция официального теста при `--no-test`

**Проблема:**  
В `scripts/run_experiments.py` при использовании флага `--no-test` код всё равно оценивал производительность на тестовых данных:

```python
# ДО ИСПРАВЛЕНИЯ:
if not args.no_test:
    # ... оценка на тесте ...
else:
    # Пустой блок, но verifier уже был оценён на records (включая test)
    pass
```

**Почему это критично:**
- `--no-test` должен гарантировать, что тестовые данные **никогда** не передаются в `verifier.evaluate()`
- Без этого любой подбор параметров (пороги, веса) косвенно использует тест через общие метрики
- Это нарушает принцип "тест измеряется один раз после заморозки"

**Исправление:**

1. **`run_experiments.py` (строки 536-598):**
   - Добавлена явная проверка: если `--no-test` установлен, но нет official splits → ошибка (код 3)
   - При наличии splits: создаётся `records_no_test` (исключая test IDs)
   - `verifier.evaluate()` вызывается только на `records_no_test`
   - `dev_official` заполняется, `test_official` остаётся `null`
   - Добавлен вывод: "ТЕСТ ИЗОЛИРОВАН (--no-test): оцениваем N пар (train+dev), исключаем M тестовых пар"

2. **`calibrate_verdict.py` (строки 321-325, 410-440):**
   - Добавлен флаг `--no-test` в argparse
   - При `--no-test`: `test_rows = None`, `test_rules = None`
   - Блок оценки теста обёрнут в `if not args.no_test:`
   - Контроль на втором корпусе (`args.also`) тоже пропускается при `--no-test`
   - Добавлен вывод: "ТЕСТ НЕ ОЦЕНИВАЛСЯ (--no-test): промежуточный прогон P0"

3. **`spanverify/train.py` (строки 837-844):**
   - Исправлена передача параметров в `end_to_end = Verifier(...)`
   - Теперь передаются: `span_expansion`, `span_filter_min_chars`, `verdict_rule` из исходного `verifier`
   - Раньше эти параметры терялись, и сквозная проверка использовала значения по умолчанию

**Регрессионные тесты (`tests/test_no_test_isolation.py`):**

1. `test_no_test_without_splits_errors`:
   - Проверяет, что при `--no-test` без splits возвращается код ошибки 3
   - Подтверждает: без official splits невозможно изолировать тест

2. `test_no_test_with_splits_excludes_test`:
   - Перехватывает `Verifier.evaluate()` и запоминает все переданные пары
   - Проверяет: test IDs **никогда** не появляются в вызовах `evaluate()`
   - Проверяет: `test_official == null` в JSON отчёте
   - Проверяет: `dev_official` заполнен корректно

**Результат:** ✅ 2 теста пройдены, 45 тестов в ключевых модулях (api, backends, calibration, config, no_test_isolation) — все зелёные.

---

### 2. Сведение результатов после перехода к матрице

**Проблема:**  
После добавления матрицы `span_expansion` (clause, mask, clause_capped, mask_padded) workflow сломался:

1. **Неправильное имя лога (строка 194):**
   ```yaml
   # БЫЛО:
   grep -E "кеш признаков: попаданий|промахов" reports/experiments-a3.log
   # ДОЛЖНО БЫТЬ:
   grep -E "кеш признаков: попаданий|промахов" reports/experiments-a3-${{ matrix.span_expansion }}.log
   ```

2. **Неправильное имя артефакта в publish job (строки 297-303):**
   ```yaml
   # БЫЛО:
   - uses: actions/download-artifact@v4
     with:
       name: hf-experiment-a3  # ← Такого артефакта нет!
       path: spanverify-module/reports/experiments/a3
   # ДОЛЖНО БЫТЬ:
   - uses: actions/download-artifact@v4
     with:
       pattern: hf-experiment-a3-*  # ← Скачиваем все 4 варианта
       path: spanverify-module/reports/experiments/a3-variants
   ```

**Исправление:**

1. **Исправлено имя лога** — теперь использует `${{ matrix.span_expansion }}`

2. **Обновлён publish job:**
   - Скачивает все артефакты `hf-experiment-a3-*` через `pattern`
   - Добавлен шаг "Разложить варианты по папкам": перекладывает из `a3-variants/hf-experiment-a3-{variant}/` в `a3/{variant}/`
   - Добавлен шаг "Сравнить варианты span_expansion": запускает `scripts/compare_span_expansion.py`

3. **Создан `scripts/compare_span_expansion.py`:**
   - Читает `experiment.json` из каждого варианта
   - Создаёт таблицу сравнения в `reports/experiments/a3/comparison.md`
   - Колонки: Strict Span F1, Containment, Width Ratio, Token F1, Token FPR, Answer F1, Verdict F1
   - Дополнительно: Token Precision/Recall/AUC, Answer AUC, Verdict FPR
   - Определяет лучший вариант по strict span F1
   - Добавляет оговорку: "Test не измерялся (--no-test)"

---

## Что было сделано до этого коммита

### Коммит `860f95e`: Параметр `span_expansion`
- Добавлен параметр `span_expansion` в `Verifier.__init__()` (4 варианта)
- Модифицирован `_build_spans()` для выбора стратегии расширения границ
- Добавлен `--span-expansion` в `run_experiments.py` и `calibrate_verdict.py`

### Коммит `717c9dc`: Workflow сканирует 4 варианта
- Преобразован `experiment-a3` в матрицу с 4 вариантами `span_expansion`
- Каждый вариант запускается параллельно с `--no-test`

### Коммит `087ffc9`: Триггер workflow
- Добавлена строка в `.github/hf-experiments.trigger`
- Запущен workflow 39053281367

### Прогон 39053281367: Результаты
- ✅ 4 matrix-job A3 (clause, mask, clause_capped, mask_padded) — success
- ✅ 16 precompute-job A/A3 — success
- ❌ Job "Свести числа и сохранить" — failure (артефакт `hf-experiment-a3` не найден)

**Проблема:** Результаты 4 вариантов недоступны для анализа из-за сломанного сведения.

---

## Что нужно сделать дальше

### 1. Запустить workflow с исправлениями

```bash
echo "P0: fix test isolation $(date -Iseconds)" >> .github/hf-experiments.trigger
git add .github/hf-experiments.trigger
git commit -m "trigger: повторный запуск после исправления --no-test и сведения"
git push origin arena/c5f69767-smarty-chumakov
```

**Ожидаемое время:** ~40 минут (precompute ~20 мин, experiment ~30 мин, publish ~5 мин)

### 2. Проверить результаты

После завершения workflow:

1. Скачать артефакт `hf-metrics-summary`
2. Открыть `reports/experiments/a3/comparison.md`
3. Проверить таблицу сравнения 4 вариантов

**Критерии выбора (на dev):**
- Strict span F1 ≥ 0.45 (IoU ≥ 0.5)
- Width ≤ 2.0× (средняя ширина фрагмента)
- Containment ≥ 0.45 (покрытие разметки)
- Tokens identical: true (токенные метрики не должны меняться)

### 3. Выбрать лучший вариант

Приоритет:
1.Strict span F1 (главная метрика)
2. При равенстве — меньшая ширина (width)
3. Проверить: containment ≥ 0.45

### 4. Зафиксировать выбор

```python
# В config/weights.json или отдельном конфиге:
{
  "span_expansion": "clause",  # или mask, clause_capped, mask_padded
  ...
}
```

После reload `Verifier` должен использовать выбранную стратегию автоматически, без ручного `--span-expansion`.

### 5. Продолжить P0 в правильном порядке

**P0 шаг 2: Агрегация ответов**
- Сравнить агрегаторы: текущий (mean), max, mean_top3
- Выбор на dev по answer F1
- Сохранить в конфиг

**P0 шаг 3: Калибровка вердикта**
- Запустить `calibrate_verdict.py` с `--no-test` и `--features-cache`
- Расширенная сетка: `min_flagged_share ∈ {0.0, 0.05, 0.10, 0.15, 0.20, 0.25}`, `min_mask_tokens ∈ {2, 3, 4, 5, 6}`, `min_fragment_chars ∈ {0, 20, 40}`
- Выбор на dev по verdict F1 при FPR ≤ target

**P0 шаг 4: Один замер теста**
- После полной заморозки всех параметров
- Один прогон на тесте
- Сравнить с договорными ориентирами: F1 ≥ 0.50, FPR ≤ 0.40

---

## Договорные ориентиры (не меняются)

- **Token F1 ≥ 0.90 при FPR ≤ 0.10** — критерий качества мастер-промта
- **Вердикт F1 ≥ 0.50 при FPR ≤ 0.40** — договорное качество
- **1200 пар** — контрактный объём
- **Test измеряется один раз** после полной заморозки

---

## Что НЕ было сделано и почему

### 1. Не скачаны артефакты из прогона 39053281367

**Причина:** Песочница Arena не имеет доступа к Azure Blob Storage (где хранятся артефакты GitHub Actions). Попытки скачать через `gh run download` и `gh api ... /zip` возвращают EOF.

**Решение:** Перезапустить workflow с исправлениями. Это быстрее и надёжнее, чем пытаться восстановить старые артефакты.

### 2. Не измерены dev-метрики для 4 вариантов

**Причина:** Результаты прогона 39053281367 недоступны для анализа (см. выше).

**Решение:** Запустить workflow заново после этого коммита.

### 3. Не выбран лучший вариант `span_expansion`

**Причина:** Нет dev-метрик для сравнения.

**Решение:** После получения результатов из нового прогона — анализ и выбор.

---

## Файлы, изменённые в этом коммите

```
M .github/workflows/hf-experiments.yml           # Исправление сведения
M spanverify-module/scripts/calibrate_verdict.py # --no-test изоляция
M spanverify-module/scripts/run_experiments.py   # --no-test изоляция
M spanverify-module/spanverify/train.py          # Передача span_expansion
A spanverify-module/scripts/compare_span_expansion.py  # Сравнение вариантов
A spanverify-module/tests/test_no_test_isolation.py    # Регрессионные тесты
```

---

## Проверки

```bash
# Синтаксис
python -m py_compile spanverify-module/scripts/run_experiments.py
python -m py_compile spanverify-module/scripts/calibrate_verdict.py
python -m py_compile spanverify-module/spanverify/train.py
python -m py_compile spanverify-module/scripts/compare_span_expansion.py
python -c "import yaml; yaml.safe_load(open('.github/workflows/hf-experiments.yml'))"

# Тесты
python -m pytest tests/test_no_test_isolation.py -v
python -m pytest tests/test_api.py tests/test_backends.py tests/test_calibration.py tests/test_config.py tests/test_no_test_isolation.py -v
```

**Результат:** ✅ Все проверки пройдены.

---

## Статус

- ✅ Дефект `--no-test` исправлен и протестирован
- ✅ Сведение результатов исправлено
- ✅ Скрипт сравнения вариантов создан
- ✅ Регрессионные тесты добавлены
- ⏳ Ожидается запуск workflow с исправлениями
- ⏳ Ожидается получение dev-метрик для 4 вариантов
- ⏳ Ожидается выбор лучшего варианта
- ⏳ Ожидается продолжение P0 (шаги 2-4)
