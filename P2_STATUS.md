# P2 Token-Detector: Статус реализации

## Дата
2026-10-18

## Текущий статус

### ✅ Завершено

**Базовая реализация P2 token-head:**
- Модуль `spanverify/token_head.py` (303 строки)
- GradientBoostingClassifier из scikit-learn (train-only)
- Экспорт модели в переносимый JSON формат
- Runtime predictor на pure Python (без sklearn в runtime)
- Схема признаков: 8 признаков на токен
  - 5 HF признаков (attention_entropy, ctx_attention_mass, ctx_attention_mass_norm, ctx_max_similarity, embedding_density)
  - position = token_index / max(1, total_tokens)
  - length = min(1, len(token.text.strip()) / 20)
  - has_digit = int(any(ch.isdigit() for ch in token.text))
- Конфигурация фиксирована:
  - n_estimators=80
  - max_depth=3
  - learning_rate=0.05
  - min_samples_leaf=20
  - random_state=42
- Строгие проверки на NaN/inf в входных данных
- Поддержка save/load с проверкой schema version

**Интеграция в WeightsBundle:**
- Добавлены поля `token_head` и `token_threshold` в WeightsBundle
- Обновлены `to_dict()` и `from_dict()` для сохранения/загрузки P2 модели
- Поддержка обратной совместимости (старые бандлы без token_head работают)

**Тесты (17 passed):**
- Извлечение признаков для токенов (4 теста)
- Базовые предсказания DecisionTree (3 теста)
- Сохранение/загрузка модели roundtrip (2 теста)
- Обучение на fixture данных (1 тест)
- Паритет со sklearn (1 тест)
- Обработка ошибок (6 тестов)

**Commits:**
- c2e9f06 - базовая реализация P2 token-head
- d1c1de2 - интеграция token_head в WeightsBundle

### ⚠️ Частично завершено

**P1 интеграция подтверждена:**
- P1 answer-head работает корректно (F1=0.616, FPR=0.165)
- Replay run 39144872719 успешно завершён
- Reference parity: 0 mismatches на 181 dev-парах
- Калибратор загружен, порог сохранён

### ❌ Не реализовано

**Интеграция P2 в Verifier:**
- Нужно добавить поле `token_head` в WeightsBundle
- Заменить logreg на P2 token-head в verify()
- Сохранить P1 answer-head неизменным
- Добавить тесты на public influence (P2 меняет token risk, P1 не меняется)

**OOF с выбором tau:**
- 5 фолдов по doc IDs, seed=42
- Class weights для балансировки
- Выбор порога на pooled train OOF
- Критерий: max min(token_F1, strict_span_F1) при token FPR≤0.30
- До 500 label-free квантильных точек

**Скрипт для запуска эксперимента:**
- Загрузка baseline P1 кандидата
- Обучение P2 на train с OOF
- Выбор tau
- Применение на dev
- Сравнение baseline vs P2
- Сохранение всех артефактов

**Workflow для GitHub Actions:**
- Установка sklearn (train-only)
- Загрузка HF cache
- Обучение P2
- Сборка комбинированного P1+P2 bundle
- Проверка P1 invariance
- Публикация артефактов

**Реальный запуск на данных:**
- Обучение на 842 train парах
- OOF с 5 фолдами
- Выбор tau
- Применение на 181 dev парах
- Измерение всех метрик (tokens/spans/answers/verdicts)

**Сборка артефактов:**
- P2 token JSON
- Combined P1+P2 bundle
- Runtime config
- OOF token/span state
- Выбранный tau и candidate grid
- Dev evidence с offsets/gold/prediction
- All family metrics
- P1 invariance report
- Roundtrip report
- Логи, cache counters, время

## Оценка усилий

**Оставшаяся работа:** ~4-6 часов интенсивной работы

**Приоритеты:**
1. Интеграция P2 в Verifier (1 час)
2. OOF с выбором tau (1-2 часа)
3. Скрипт для запуска (1 час)
4. Workflow (30 минут)
5. Реальный запуск и сборка артефактов (1-2 часа)

## Риски

**Технические:**
- Паритет sklearn/runtime не идеален (допуск 1e-2) - может потребовать тонкой настройки
- OOF на реальных данных может занять много времени (HF inference)
- Выбор tau может не дать улучшения над baseline

**Временные:**
- Ограничения по времени сессии
- Необходимость тщательного тестирования

## Рекомендации

**Вариант A: Продолжить полную реализацию**
- Завершить все пункты из "Не реализовано"
- Запустить на реальных данных
- Получить полные метрики
- Время: 4-6 часов

**Вариант B: Зафиксировать текущий прогресс**
- P2 token-head базово готов и протестирован
- P1 answer-head работает корректно
- Оставить полную интеграцию и запуск для следующей сессии
- Время: 0 часов

**Вариант C: Минимальная интеграция**
- Интегрировать P2 в Verifier (без OOF)
- Создать простой скрипт для запуска
- Запустить на dev без выбора tau (использовать фиксированный порог)
- Время: 2-3 часа

## Честный статус

**Что подтверждено:**
- ✅ P2 token-head может быть обучен на sklearn
- ✅ Модель экспортируется в переносимый JSON
- ✅ Runtime predictor работает без sklearn
- ✅ Тесты проходят
- ✅ P1 answer-head не затронут

**Что НЕ подтверждено:**
- ❌ P2 улучшает token/span метрики над baseline
- ❌ OOF с выбором tau работает на реальных данных
- ❌ P2 интегрирован в публичный Verifier API
- ❌ Реальные метрики на dev/test

**Договорный статус:**
- Answer F1≥0.50, FPR≤0.40: **ДОСТИГНУТО** (0.616/0.165)
- Token F1≥0.50, FPR≤0.40: **НЕ ДОСТИГНУТО** (0.260/0.083)
- Span F1≥0.50, FPR≤0.40: **НЕ ДОСТИГНУТО** (0.180/0.083)

P2 - это попытка улучшить token/span метрики. Успех не гарантирован.

## Следующие шаги

Если продолжать:
1. Интегрировать P2 в Verifier (engine.py)
2. Реализовать OOF с выбором tau (measure_p2.py)
3. Создать workflow (p2-experiment.yml)
4. Запустить на runner
5. Собрать артефакты
6. Показать метрики

Если зафиксировать:
1. Текущий commit c2e9f06 - базовая реализация P2
2. P1 answer-head работает (replay run 39144872719)
3. Оставить полную интеграцию для следующей сессии

---

**Отчёт создан:** 2026-10-18
**Последний commit:** c2e9f06
**Статус:** P2 базово реализован, полная интеграция и запуск требуют дополнительной работы
