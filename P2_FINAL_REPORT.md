# P2 Token-Head: Финальный отчёт

**Дата:** 2026-10-18  
**Ветка:** `arena/c5f69767-smarty-chumakov`  
**Последний коммит:** `6539a3b` — P2: минимальный эксперимент (интеграция проверена)

---

## Статус: БАЗОВЫЙ МОДУЛЬ ГОТОВ (50%)

### ✅ Выполнено

#### 1. Базовый модуль P2 (коммит 0eb6c29)
- **Архитектура:** GradientBoostingClassifier на 8 признаках
  - 5 HF: `attention_entropy`, `ctx_attention_mass`, `embedding_density`, `ctx_attention_mass_norm`, `ctx_max_similarity`
  - 3 геометрических: `position` (нормализованная), `length` (логарифм), `has_digit`
- **Экспорт:** JSON с trees, thresholds, leaf values
- **Runtime:** pure-Python модуль (sklearn только при обучении)
- **init=zero:** `initial_prediction=0.0` (НЕ вычисляется из невзвешенных y)
- **Float32 конвертация:** через `struct.pack/unpack` для порогового паритета
- **Stable sigmoid:** clamp ±30 перед `exp(-x)`
- **Guards:** ValueError на NaN/inf, missing children, feature index out of range, wrong schema/n_features

#### 2. Тесты (39 прошли)
- **TestExtractTokenFeatures** (4): базовая экстракция, нормализация position, log length, has_digit
- **TestToFloat32** (3): базовая конвертация, boundary значения, threshold boundary
- **TestTokenGBDTModel** (10): predict_one/predict_many, save/load, guards
- **TestDecisionTree** (5): базовый проход, save/load, missing children
- **TestTrainTokenGBDT** (7): parity с sklearn, max_delta ≤ 1e-12 на 7 fixtures
- **TestP2PublicInfluence** (9): P2 меняет token risk, не влияет на P1 answer score

#### 3. Интеграция в Verifier.verify() (коммит 0eb6c29)
```python
if token_head_model is not None:
    # P2: GradientBoosting на 8 признаках
    token_features_matrix = [extract_token_features(i, total_tokens, token.text, hf_features) for i, token in enumerate(tokens)]
    risk = token_head_model.predict_many(token_features_matrix)
else:
    # Legacy: combine() + head
    risk = combine(...)
```

**Цепочка:**
1. HF matrix + token geometry → 8 признаков
2. P2 token-head → `predict_many()` → token risk
3. Zero-mask unscored tokens
4. Smoothing
5. Span threshold → mask spans
6. P1 answer-head → raw score → calibrated score → threshold → prediction

#### 4. Минимальный эксперимент (коммит 6539a3b)
- ✅ P1 invariance: OK (0 drifts на 10 dev парах)
- ✅ P2 влияет на token risk (разные risks с P2 vs без P2)
- ✅ Интеграция работает

**Отчёт:** `spanverify-module/reports/experiments/p2_experiment/run1/experiment_report.json`

---

### ❌ Не выполнено

#### 1. Реальный эксперимент на контроле
**Требуется:**
- GBDT: `loss=log_loss, n_estimators=80, lr=0.05, max_depth=3, min_samples_leaf=20, subsample=1.0, init=zero, random_state=42`
- 5 document folds, seed=42
- 842 train + 181 dev полных пар
- Class/sample weights вычисляются только по fit-части фолда
- Извлечение 8 признаков для каждого токена из HF features
- OOF predictions для всех фолдов
- Выбор tau: `max min(token_F1, strict_span_F1)` при `token FPR ≤ 0.30`, `strict IoU ≥ 0.5`
- До 500 label-free квантильных точек OOF smoothed-risk
- Финальная P2-head на всём train
- Применение на dev с frozen tau

**Статус:** НЕ ЗАПУЩЕН

**Причина:** Требует реализации:
- Извлечения token-level labels из span labels
- Извлечения HF features для каждого токена (требует HF backend)
- 5 document folds с balanced weights
- Вычисления span metrics (не только token)
- Оптимизации tau на OOF

#### 2. Метрики на dev
**Требуется:**
- Token F1, FPR (цель: F1 ≥ 0.50, FPR ≤ 0.40)
- Span F1, FPR (цель: F1 ≥ 0.50, FPR ≤ 0.40)
- Answer F1, FPR (должны совпасть с baseline)
- P1 invariance на 181 dev парах

**Статус:** НЕ ВЫЧИСЛЕНЫ

#### 3. GitHub Actions workflow
**Требуется:**
- Workflow `.github/workflows/p2-experiment.yml`
- Запуск эксперимента на GitHub Actions
- Публикация артефактов:
  - P2 JSON (финальная модель)
  - Combined bundle (P1 + P2)
  - OOF state
  - Tau
  - Dev evidence
  - All-family metrics
  - P1 invariance report

**Статус:** НЕ СОЗДАН

#### 4. Публичный run ID
**Требуется:**
- GitHub Actions run ID
- Commit SHA
- Публикация в `spanverify-module/reports/experiments/p2_experiment/run<id>/`

**Статус:** НЕ ПОЛУЧЕН

---

## Критические замечания (исправлены)

### 1. init="zero" вычислял prior из невзвешенных y
**Проблема:** При balanced sample weights sklearn использует weighted prior [0.5, 0.5] → log-odds 0.0, а код давал log(prior_pos / (1-prior_pos)) → max_delta≈0.138

**Исправление:** `train_token_gbdt` всегда использует `init="zero"` + `initial_prediction=0.0`. Экспорт через `export_token_gbdt_from_sklearn(gbdt)` проверяет `init=="zero"` и берёт `initial_prediction=0.0`.

### 2. Runtime не конвертировал входы в float32
**Проблема:** `1.0 + 1e-9 > 1.0` в float64, но `== 1.0` в float32 → разные ветки деревьев → max_delta≈0.462

**Исправление:** `_to_float32(value)` через `struct.pack/unpack` в `predict_one`. Тест `TestToFloat32.test_threshold_boundary` проверяет boundary case.

### 3. Parity-test обучал две разные модели
**Проблема:** sklearn без balanced weights, export-helper с balanced → две разные модели → допуск 1e-2 был костылём

**Исправление:** тесты используют `export_token_gbdt_from_sklearn(gbdt)` на том же sklearn-объекте. Допуск: `max_delta ≤ 1e-12`.

### 4. Guards валидации
**Добавлены:**
- ValueError на NaN/inf в features/threshold/leaf/init/rate
- ValueError на missing children у internal node
- ValueError на feature index out of range `[0, N_TOKEN_FEATURES)`
- ValueError на `n_features != 8` при загрузке
- ValueError на unknown `schema_version` или `type`

---

## Артефакты

### Коммиты
- `0eb6c29` — P2: базовый модуль token-head (GBDT, 8 признаков, pure-Python runtime)
- `6539a3b` — P2: минимальный эксперимент (интеграция проверена)

### Файлы
- `spanverify-module/spanverify/token_head.py` — P2 модуль (TokenGBDTModel, DecisionTree, extract_token_features)
- `spanverify-module/tests/test_token_head.py` — 30 тестов token_head
- `spanverify-module/tests/test_p2_public_influence.py` — 9 тестов public influence
- `spanverify-module/scripts/p2_experiment.py` — скрипт минимального эксперимента
- `spanverify-module/reports/experiments/p2_experiment/run1/` — отчёт минимального эксперимента
- `P2_STATUS.md` — статус разработки

### Тесты
- **39 тестов прошли** (30 token_head + 9 public_influence)
- Все parity-тесты: max_delta ≤ 1e-12
- P1 invariance: ✅ OK
- P2 influence: ✅ OK

---

## Итог

**Готовность P2: 50%**

✅ **Готово:**
- Базовый модуль (GBDT, 8 признаков, pure-Python runtime)
- Интеграция в Verifier.verify()
- Тесты (39 прошли)
- Критические замечания исправлены
- Минимальный эксперимент (интеграция проверена)

❌ **Не готово:**
- Реальный эксперимент на контроле (5 doc-folds, 842+181 пар)
- Метрики на dev (token/span/answer F1, FPR)
- GitHub Actions workflow
- Публичный run ID
- Публикация артефактов

**Следующие шаги:**
1. Реализовать полный эксперимент с 5 doc-folds
2. Запустить на GitHub Actions
3. Опубликовать метрики
4. **Честный отчёт:** если P2 не достиг норм (token F1 ≥ 0.50, FPR ≤ 0.40; span F1 ≥ 0.50, FPR ≤ 0.40) — написать «НЕ ДОСТИГНУТО»

---

## Ответ на вопрос пользователя

**Вопрос:** Готов ли P2?

**Ответ:** Базовый модуль готов, интеграция проверена, тесты прошли. Но реальный эксперимент на контроле НЕ выполнен, метрики НЕ опубликованы. Готовность 50%.

**P2 НЕ достиг норм** (потому что эксперимент не запущен).
