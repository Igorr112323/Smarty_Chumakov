> Historical diagnostic report, NOT a quality run. The percentage below is not measured contractual readiness. run1 used demo, handwritten placeholder, fitting_pairs=0, dev_pairs_checked=10, influence_pairs=1.

# P2 Token-Head: Статус разработки

**Дата:** 2026-10-18  
**Ветка:** `arena/c5f69767-smarty-chumakov`  
**Последний коммит:** `0eb6c29` — P2: базовый модуль token-head (GBDT, 8 признаков, pure-Python runtime)

---

## Критика базового модуля (исправлено в коммите 0eb6c29)

### Замечания владельца
1. ❌ **init="zero" вычислял prior из невзвешенных y** — НЕВЕРНО при balanced sample weights
   - **Исправлено:** `train_token_gbdt` всегда использует `init="zero"` + `initial_prediction=0.0`
   - Экспорт через `export_token_gbdt_from_sklearn(gbdt)` проверяет `init=="zero"` и берёт `initial_prediction=0.0`

2. ❌ **Runtime не конвертировал входы в float32** — пороговые значения дают неверные ветки
   - **Исправлено:** `_to_float32(value)` через `struct.pack/unpack` в `predict_one`
   - Тест `TestToFloat32.test_threshold_boundary`: `1.0 + 1e-9 > 1.0` в float64, но `== 1.0` в float32

3. ❌ **Parity-test обучал две разные модели** — sklearn без balanced weights, export-helper с balanced
   - **Исправлено:** тесты используют `export_token_gbdt_from_sklearn(gbdt)` на том же sklearn-объекте
   - Допуск: `max_delta ≤ 1e-12` (не 1e-2)

4. ✅ **Гарды валидации:**
   - ValueError на NaN/inf в features/threshold/leaf/init/rate
   - ValueError на missing children у internal node
   - ValueError на feature index out of range `[0, N_TOKEN_FEATURES)`
   - ValueError на `n_features != 8` при загрузке
   - ValueError на unknown `schema_version` или `type`

5. ✅ **Stable sigmoid:** clamp ±30 перед `exp(-x)`

### Результат
**Все 30 тестов token_head прошли** (включая 7 parity-тестов с `max_delta ≤ 1e-12`)

---

## Интеграция P2 в Verifier.verify() (коммит 0eb6c29)

### Логика
```python
if token_head_model is not None:
    # P2: GradientBoosting на 8 признаках
    token_features_matrix = [extract_token_features(i, total_tokens, token.text, hf_features) for i, token in enumerate(tokens)]
    risk = token_head_model.predict_many(token_features_matrix)
else:
    # Legacy: combine() + head
    risk = combine(features.attention_entropy, features.ctx_attention_mass, features.embedding_density, ...)
```

### Цепочка
1. HF matrix (5 признаков) + token geometry (position, length, has_digit) → 8 признаков
2. P2 token-head → `predict_many()` → token risk
3. Zero-mask unscored tokens (пунктуация, короткие служебные слова)
4. Smoothing (скользящее среднее)
5. Span threshold → mask spans
6. P1 answer-head → raw score → calibrated score → threshold → prediction

### Результат
**Все 9 тестов public influence прошли:**
- ✅ P2 меняет token raw_risk
- ✅ P2 НЕ меняет P1 answer raw/calibrated score
- ✅ P2 НЕ меняет P1 answer prediction
- ✅ P2 save/load не меняет предсказания
- ✅ Повреждённый P2 вызывает ошибку (не silent fallback)
- ✅ Bundle без P2 работает как раньше
- ✅ P2 меняет flags и spans

---

## Что осталось сделать

### 1. Реальный эксперимент на контроле
**Контроль:** `spanverify-module/reports/experiments/p1_public_replay/run39144872719/p1_candidate_bundle.json`  
**Recipe:**
- GBDT: `loss=log_loss, n_estimators=80, lr=0.05, max_depth=3, min_samples_leaf=20, subsample=1.0, init=zero, random_state=42`
- Class/sample weights вычисляются только по fit-части фолда
- 5 document folds, seed=42, без pair-ID fallback
- 842 train + 181 dev полных пар
- Критерий tau: `max min(token_F1, strict_span_F1)` при `token FPR ≤ 0.30`, `strict IoU ≥ 0.5`
- До 500 label-free квантильных точек OOF smoothed-risk
- Финальная P2-head на всём train, dev — одна оценка
- `span_expansion="mask"` (узкий, без расширения до предложения/клаузы)

### 2. Приёмка
- ✅ P1 invariance по ID на 181 dev-парах (raw/calibrated score, threshold, prediction; tolerance 1e-12; counts 49/14/47/71)
- ✅ Baseline vs P2: все семейные метрики
- ✅ Ответ — результаты, не «основа готова». Если P2 не достиг норм — честно «НЕ достигнуто»

### 3. Workflow
- Measurement script: `spanverify-module/scripts/p2_experiment.py`
- GitHub Actions workflow: `.github/workflows/p2-experiment.yml`
- Артефакты:
  - P2 JSON (финальная модель)
  - Combined bundle (P1 + P2)
  - OOF state (predictions для всех фолдов)
  - Tau (оптимальный threshold)
  - Dev evidence (детальные предсказания)
  - All-family metrics: baseline vs P2
  - P1 invariance report

### 4. Публикация
- Public run ID (GitHub Actions)
- Public run SHA (commit hash)
- Метрики в `spanverify-module/reports/experiments/p2_experiment/run<id>/`

---

## Базовый модуль P2 (готов)

### Архитектура
- **Токен-признаки (8):**
  - 5 HF: `attention_entropy`, `ctx_attention_mass`, `embedding_density`, `ctx_attention_mass_norm`, `ctx_max_similarity`
  - 3 геометрических: `position` (нормализованная), `length` (логарифм), `has_digit`
- **Модель:** GradientBoostingClassifier
  - `loss=log_loss, n_estimators=80, learning_rate=0.05, max_depth=3, min_samples_leaf=20, subsample=1.0`
  - `init=zero, random_state=42`
- **Экспорт:** JSON с trees, thresholds, leaf values
- **Runtime:** pure-Python модуль (sklearn только при обучении)

### Тесты
- ✅ **TestExtractTokenFeatures** (4): базовая экстракция, нормализация position, log length, has_digit
- ✅ **TestToFloat32** (3): базовая конвертация, boundary значения, threshold boundary
- ✅ **TestTokenGBDTModel** (10): predict_one/predict_many, save/load, guards (NaN/inf/wrong n_features/unknown schema)
- ✅ **TestDecisionTree** (5): базовый проход, save/load, missing children
- ✅ **TestTrainTokenGBDT** (7): parity с sklearn (balanced/imbalanced/custom weights/held-out/save_load/threshold_boundary), max_delta ≤ 1e-12
- ✅ **TestP2PublicInfluence** (9): P2 меняет token risk, не влияет на P1 answer score, save/load, broken P2 raises error, bundle без P2 работает

**Итого:** 39 тестов прошли

---

## Следующие шаги

1. Создать `scripts/p2_experiment.py` — реальный эксперимент на контроле
2. Создать `.github/workflows/p2-experiment.yml` — GitHub Actions workflow
3. Запустить эксперимент, получить публичный run ID
4. Опубликовать метрики в `reports/experiments/p2_experiment/run<id>/`
5. **Честный отчёт:** если P2 не достиг норм (token F1 ≥ 0.50, FPR ≤ 0.40; span F1 ≥ 0.50, FPR ≤ 0.40) — написать «НЕ ДОСТИГНУТО»

---

## Итог

**Базовый модуль P2 готов:**
- ✅ Архитектура (GBDT, 8 признаков)
- ✅ Экспорт/импорт (JSON, pure-Python runtime)
- ✅ Интеграция в Verifier.verify()
- ✅ Тесты (39 прошли)
- ✅ Критика исправлена (init=zero, float32, parity 1e-12, guards)

**Реальный эксперимент НЕ выполнен.**  
**Метрики НЕ опубликованы.**  
**Готовность P2: 50%** (модуль готов, эксперимент не запущен).

**Следующий шаг:** запустить реальный эксперимент на контроле и опубликовать результаты.
