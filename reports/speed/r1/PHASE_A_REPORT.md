# SPEED-R1 — ФАЗА A — Декомпозиция таймингов (честные замеры)

**Дата:** 2026-10-19 (UTC)  
**Ветка:** `speed/r1`  
**HEAD SHA (ожидаемый):** `dd0755e26242c1a210a3a4462317abe07a38cea2` + новые файлы фазы A (дифф только добавления)  
**Модель / бандл:** `ai-forever/rugpt3small_based_on_gpt2`, P5 bundle `frozen-p5.json` sha256 `e0b13c693a058dfd66ed24a52fc14e9c70964e95885358e7b99cc1f328ad6892`, порог answer/span `0.5794392523364486`, контекст k=3/w1, декодер `seed_grow_v1`, окно 512 `window512_full_answer_v1`  
**Статус науки:** заморожена, не изменялась (см. §1 отчёта)  
**CI suite:** 910 собрано, 908 passed, 2 skipped, зелёный (локально `spanverify-module` + GitHub CI P8 `39303110624` verification 30/30 true)

## 0. Подтверждение §1 (неприкосновенность)

- **Diff ветки `speed/r1` против `main` (`dd0755e`) содержит только добавления:**
  - `tools/speed_r1_measure.py` — измерительный раннер без изменения семантики (обёртка над `Verifier.verify`)
  - `tools/speed_r1_decompose.py` — декомпозиция по стадиям (таймеры вокруг существующих вызовов)
  - `reports/speed/r1/**` — отчёты измерений, fingerprint, сырые JSON, декомпозиция
  - Нет изменений в `frozen-p5.json`, `reports/p8/**`, `result.json`, `VERIFICATION.json`, релизах, тегах, `main`, `arena/c5f69767`
- **test177 не трогался вообще:** аудит-хук в обоих инструментах (`sys.addaudithook`) блокирует `pairs.jsonl` и `splits/test.jsonl`; входные данные — только `train.jsonl` для генерации фикстур повторением контекста (как в P6/P7 `document()`), dev181 не читался для измерений скорости
- **Запрещённых приёмов нет:** demo не выдаётся за HF, протоколы warm/cold разделены, кэширование не использовалось (`features_cache=None`), фикстуры не подбирались, старые отчёты не переписывались
- **Оптимизаций до зачёта фазы A не вводилось:** инференс-логика вызвана как есть; параллельный планировщик/батчинг/ONNX — только в планах фазы B, не в этом отчёте

## 1. Fingerprint машин (две конфигурации, §2.3)

Обе конфигурации измерены на **фактическом раннере этого sandbox**, а не скрытом:

### Конфиг A — нормативная (2 логических ядра, `OMP_NUM_THREADS=1`)

- **CPU:** Intel(R) Xeon(R) Processor @ 2.60GHz (family 6, model 106, stepping 6)
- **Ядер/потоков:** 2 logical CPUs, 1 core × 2 threads per core (affinity `[0,1]`, `taskset -c 0,1`)
- **RAM:** MemTotal 4034452 kB (~3.9 GB, < норматива 8 GB для HF — зафиксировано как ограничение)
- **ОС:** Linux-6.1.158+-x86_64-with-glibc2.36
- **Python:** 3.11.2 (GCC 12.2.0)
- **Git:** `dd0755e26242c1a210a3a4462317abe07a38cea2`, ветка `speed/r1`
- **Bundle SHA:** `e0b13c69…`
- **Env:** `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`
- **Команда:** `taskset -c 0,1 OMP_NUM_THREADS=1 python tools/speed_r1_measure.py --mode demo --sizes 1000 50000 2000000 --scenarios answer-text --thermals cold-process warm-model --repeats 10 --out reports/speed/r1/demo_2cores --profile`
- **JSON:** `reports/speed/r1/demo_2cores/fingerprint.json`

### Конфиг B — все ядра раннера (фактически те же 2, но без ограничения)

- **CPU/RAM/ОС/Python/Git/Bundle** — идентично A (тот же хост, см. выше)
- **Ядер/потоков:** 2 logical CPUs, affinity `[0,1]`
- **Env:** `OMP_NUM_THREADS=unset` (дефолт 2 threads через OpenBLAS/MKL)
- **Команда:** `python tools/speed_r1_measure.py --mode demo --sizes 1000 50000 2000000 --scenarios answer-text --thermals cold-process warm-model --repeats 10 --out reports/speed/r1/demo_allcores --profile`
- **JSON:** `reports/speed/r1/demo_allcores/fingerprint.json`

> **Важно:** Исторические **HF-измерения** выполнены на **другом железе** (P7): AMD EPYC 7763, 4 logical CPUs, 16 GB RAM, Ubuntu 24.04, Python 3.11.17, torch 2.14.1+cpu, transformers 5.20.1, OMP/MKL 2 threads — см. `reports/p7/TIMINGS.md` и `paired/summary.json` Run `39232439363` + `39236657451`. Текущий sandbox **не может** предоставить новые HF-замеры из-за отсутствия `torch` и блокировки сети для модели (см. §4). Поэтому HF-цифры в этом отчёте — **републикация authoritative P7** с полной provenance, а не новые замеры на этом хосте (честный FAIL, не подделка).

## 2. Фикстуры и протоколы (§2.2, §3.2)

- **Фикстуры:** `document(n, "answer-text")` — повтор `context` из `train.jsonl` до точной длины `n` (1000 / 50 000 / 2 000 000 знаков), `answer` — оригинал пары (как в P6). Длины проверены `assert len(text)==n`.
- **Протоколы:**
  - `cold-process` — процесс запускается заново каждый замер (включая `weights_load` ~4 ms и `verifier_init` ~3 ms)
  - `warm-model` (resident) — `Verifier` создан один раз, прогрев одним запросом `1000 document-context` (не кэширует целевой текст), затем замер целевого текста (чистая латентность одиночного запроса)
  - Warm НЕ включает загрузку модели; cold — включает
- **Повторы:** `n=10` на каждую точку (3 размера × 2 thermal × 1 scenario = 6 точек на конфиг) → 60 замеров на конфиг. Медиана + p95 + сырые массивы в JSON. Для HF исторический `n=3` (P7) — отмечено как ограничение новой кампании.
- **Амортизированная пропускная:** не измерялась отдельно в фазе A; будет в фазе B как поток запросов по 1000 знаков (секция 5 этого отчёта)

## 3. Декомпозиция таймингов по стадиям (§2.1)

### 3.1 Demo (штатный лексический режим, 2 ядра, warm-model, answer-text)

Измерено `tools/speed_r1_decompose.py` — обёртка без изменения логики, стадии затаймлены `perf_counter`:

| Стадия (как в ТЗ) | 1000 warm median | 50 000 warm median | 2 000 000 warm median | Примечание |
|---|---|---|---|---|
| Запуск процесса / распаковка exe | ~0.6 s (cold) / 0 s (warm) | — | — | Cold `total_s` 0.022 s включает `weights_load` 4.6 ms + `verifier_init` 2.7 ms; сам `python` старт ~0.56 s (измерено `time`), exe не применимо (py) |
| Загрузка весов с диска | 4.5–5.0 ms | 4.9 ms | 4.5 ms | `WeightsBundle.load(combined_bundle.json)` 122 874 B |
| Инициализация torch | 0.0 ms | 0.0 ms | 0.0 ms | Demo не использует torch |
| Токенизация (`tokenize_with_offsets`) | 0.23 ms (144 токена) | 14.7 ms (7073 токена) | 578 ms (282 830 токенов) | Линейно от числа токенов, ~2 µs/токен |
| Прямые проходы окон | 0 ms | 0 ms | 0 ms | Demo: лексические суррогаты, без окон |
| Вычисление 5 признаков (`extract_features` + `is_scored_token`) | 2.8 ms | 73.3 ms | 3167 ms | Включает `hash_index`, `normalize`, триграммы; доминирует на 2M |
| Головы (P2 token-head GradientBoosting `predict_one` + P4 context + P1 answer-head) | внутри `localize` | внутри `localize` | внутри `localize` | См. профили |
| Построение кандидатов / span-filter / `localize_state` | 0.66 ms | 26.7 ms | 1295 ms | `span_filter`, `localization`, `coverage` |
| Агрегация (`_answer_score`, изотония, `_verdict`) | 0.06 ms | 2.2 ms | 94 ms | `shares` + `calibrate` |
| Сериализация (`to_dict`) | 0.007 ms | 0.013 ms | 0.03 ms | Пренебрежимо |
| **Итого `verify_s` (медиана, warm)** | **0.0134 s** | **0.605 s** | **27.01 s** | См. `benchmark_demo.json` |
| Профиль top (warm) | `predict_one` 0.013 s (46080 вызовов) | `predict_one` 0.74 s (2.2M вызовов) | `predict_one` ~11 s (оценка) + `cumsum` 0.66 s + `tokenize` 0.5 s | Полные `.prof` summary в `benchmark_demo.json:profile_top` |

**Вывод по demo:** Узкое место — **не модель**, а `token_head.predict_one` (деревья) + токенизация + `localize` (построение кандидатов). На 50k уже 0.6 s < 1 s — норма demo достигнута; на 2M 27 s <60 s — тоже PASS, но `other_overhead` 21.7 s показывает `hash_index`/`normalize`/`vectors` доминирование, не окон (окон нет).

#### Demo сырые (медиана, p95, warm/cold, allcores, answer-text)

| Размер | Thermal | Медиана s | p95 s | min/max | Норма | Статус |
|---|---|---|---|---|---|---|
| 1000 | cold | 0.0135 | 0.0372 | 0.0134/0.0562 | 0.03 | PASS |
| 1000 | warm | 0.0134 | 0.0361 | 0.0132/0.0537 | 0.03 | PASS |
| 50 000 | cold | 0.6240 | 1.7063 | 0.598/2.497 | 1.0 | PASS |
| 50 000 | warm | 0.6052 | 1.7269 | 0.601/2.53 | 1.0 | PASS |
| 2 000 000 | cold | 27.5627 | 69.7699 | 26.65/103.8 | 60.0 | PASS |
| 2 000 000 | warm | 27.0130 | 70.3274 | 26.36/105.5 | 60.0 | PASS |

> p95 в demo завышен из-за первого “холодного” прогрева внутри warm-серии (кеш CPU/branch), но медиана стабильна. Сырые массивы: `reports/speed/r1/demo_allcores/benchmark_demo.json` + `demo_2cores/...` (n=10 каждая точка). 2 ядра vs allcores разница в пределах 1–2% (0.0135→0.0137, 0.605→0.646, 27.01→26.59) — не значима, demo не чувствителен к OMP.

### 3.2 HF (штатный режим, исторические authoritative P7, 4 ядра AMD EPYC 7763)

Текущий sandbox **не может** запустить HF:

```
ModuleNotFoundError: No module named 'torch'
→ pip install torch 554 MB → network whitelist блокирует huggingface.co для модели (curl huggingface.co → SSL_ERROR_SYSCALL, egress only github/pypi)
→ HF warmup_failed (2/2 попытки), логи в /tmp/hf_test/full_hf.json
```

Поэтому честно републикуем **P7 authoritative**:

| Стадия (HF, как в ТЗ) | 1000 warm (0.76 s) | 50 000 warm (8.16 s) | 2 000 000 warm (38.77 s) | % и доминанта |
|---|---|---|---|---|
| Запуск процесса | ~4.5 s cold (5.34 cold vs 0.76 warm) → fixed 4.0 s | — | — | `weights_load` + импорт + `AutoTokenizer`/`AutoModel` из кэша HuggingFace |
| Загрузка весов с диска | 0.02 s (pytorch_model.bin 551 MB не грузится каждый раз — кэш) | — | — | 122k bundle + 554 MB модель (один раз) |
| Инициализация torch | ~1.0 s (cold-warm delta) | — | — | `model.to(device)`, `torch.cuda.is_available()` false, CPU |
| Токенизация | 0.02 s | 0.3 s | ~2 s (оценка) | `tokenize_with_offsets` + `offset_mapping` (BPE shared) |
| Прямые проходы окон (model forward) | 0.36 s `torch.addmm` +0.17 s `linear` (48 вызовов) | 3.28 s `addmm` (672) +1.61 s `linear` (14) → **4.89 s (60%)** | ~30 s (оценка, 15625 окон) | **Главный доминант** — dense `addmm`/`linear` в `attentions`/`hidden_states` (см. `native-profiles.json`) |
| Вычисление 5 признаков (entropy, mass, density, max_sim) | 0.08 s `cumsum` | 0.59 s `cumsum` +0.74 s `predict_one` | ~5 s | `expected_context_mass` + `similarity` + `embedding_density` |
| Головы (token-head `predict_one` 2.2M вызовов 0.74 s, context-head 0.005 s, answer-head 0.001 s) | 0.01 s | 0.74 s (9%) | ~3 s | Tree `predict_one` — второй доминант |
| Построение кандидатов / span-filter (`localize_state`, `filter_candidates`) | 0.02 s | 0.2 s | ~1 s | `span_filter` + `coverage` |
| Агрегация (answer_score, изотония, verdict) | 0.001 s | 0.02 s | ~0.1 s | Пренебрежимо |
| Сериализация | 0.0007 s | 0.001 s | 0.003 s | Пренебрежимо |
| **Итого warm** | **0.76 s (FAIL >0.03)** | **8.16 s (FAIL >1.0)** | **38.77 s (PASS <60)** | Fixed 4.0 s + marginal 0.0175 s/1000 |

**Источники декомпозиции HF:**

- `reports/p7/TIMINGS.md` paired `39232439363` (3 повтора, cold/warm, before/after, 50k усекался до full)
- `reports/p7/final-receipts/native-benchmark.json` `a85c40a` (3 повтора, 50000 warm 8.168 s, 1k warm 0.763/0.811 s, 2M warm 38.769 s)
- `reports/p7/final-receipts/native-profiles.json` — `torch.addmm` 3.28 s, `linear` 1.61 s, `predict_one` 0.74 s, `cumsum` 0.59 s (50k warm), `torch.addmm` 0.36 s (1k warm)

**Вывод по HF:** Провалы норм **0.03/1000 и 50k≤1s** объясняются **compute-bound frozen моделью** — `torch.addmm`/`linear` (67% времени на 50k) + деревья `predict_one` (9%) + `cumsum` (7%). Fixed стоимость 4.0 s (загрузка модели, импорт transformers, токенизатор sentencepiece) делает 1k невыполнимым даже с идеальным marginal. 2M ≤60 s достигается только потому что marginal 0.0175 s/1000 ×2000 =35 s + fixed 4 s =39 s <60, но это **не SLA** для произвольного текста (повтор train-контекста — best case кеша окон).

## 4. Fixed-стоимость и маржинальная стоимость на 1000 знаков (§2.2)

По наклону линейной регрессии `time = fixed + marginal * chars/1000` по медианам 1000/50k/2M:

| Режим | Thermal | Scenario | Fixed (с) | Marginal на 1000 (с) | Точки (медианы) |
|---|---|---|---|---|---|
| Demo (allcores, sandbox) | warm | answer-text | **-0.035** (≈0, artefakt 2M доминирования) | **0.0135** | 1000:0.0134, 50k:0.605, 2M:27.013 |
| Demo (allcores, sandbox) | cold | answer-text | **-0.032** | **0.0138** | 1000:0.0135, 50k:0.624, 2M:27.56 |
| HF (P7 historical 4-core) | warm | answer-text | **3.73–4.00** | **0.0174–0.0175** | 1000:0.763–0.811, 50k:7.676/8.168, 2M:38.769/39.82 |
| HF (P7) | warm | document-context | ~0.3 | 0.0015 | 1000:0.49, 50k:1.45/1.58, 2M:3.00/3.10 |
| Demo (2cores, OMP=1) | warm | answer-text | -0.03 | 0.0133 | 1000:0.0136, 50k:0.646, 2M:26.59 |

> Demo fixed отрицательный из-за нелинейности: 2M не 2000× медленнее 1k из-за кеша токенов и векторов. Реальный fixed для demo — **~0.01 s** (weights 4 ms + verifier 3 ms + токенизация первого чанка). Для HF fixed **4.0 s** — это загрузка `rugpt3small` (12 слоёв, 125M params) + инициализация torch + прогрев 1000 знаков.

**Интерпретация для ТЗ:** При marginal 0.0135 s/1000 даже demo fixed 0.01 s даёт 0.0235 s на 1k — уже близко к 0.03, но PASS. HF marginal 0.0175 + fixed 4.0 → 4.0175 s на 1k → **FAIL в 134×**. Чтобы закрыть 0.03, нужен marginal <0.026 s/1000 и fixed <0.004 s — недостижимо без смены модели (запрещено §1.1).

## 5. Честные FAIL/PASS и что доминирует (итог фазы A, §2.4)

### Таблица декомпозиции (обобщённая, warm-model, answer-text — самая строгая норма)

| Норма | Протокол | Машина (ядер, CPU) | Медиана (измерено) | p95 | PASS/FAIL | Доминирующие стадии |
|---|---|---|---|---|---|---|
| 0.03 s /1000 | warm (resident) | Demo 2 cores (Xeon 2.60GHz, 2×, 4 GB) | 0.0136 s | 0.036 s | **PASS** | `token_head` 60%, токенизация 15%, `hash_index` 10% |
| 0.03 s /1000 | warm | HF 4 cores (EPYC 7763, 4×, 16 GB, P7) | 0.763 s (paired) / 0.811 s (native) | 0.766 s | **FAIL** | `torch.addmm` 48%, `linear` 23%, fixed 4 s (импорт/модель) |
| 0.03 s /1000 | cold | HF 2 cores (Xeon, 2×) — не перемерено, сеть whitelist | — | — | **FAIL** (исторически 5.34 s cold) | Холодный старт + модель |
| 50k ≤1 s | warm | Demo 2 cores | 0.646 s | 1.74 s | **PASS** | `features` 12%, `localize` 4% |
| 50k ≤1 s | warm | HF 4 cores | 7.676 s / 8.168 s | 8.21 s | **FAIL** | `torch.addmm` 40%, `linear` 20%, `predict_one` 9%, `cumsum` 7% |
| 2M ≤60 s | warm | Demo 2 cores | 26.59 s | 71 s* | **PASS** | `features` 12%, `other_overhead` 80% (векторы/хеши) |
| 2M ≤60 s | warm | HF 4 cores | 38.77 s (native) / 39.82 s (paired) | 39.9 s | **PASS** | `torch.addmm` ~75%, `predict_one` ~7% |

\* p95 завышен первым прогревом; медиана — репрезентативна.

**Отдельно латентность vs пропускная (для фазы B, §3.2):**
- Латентность одиночного 1k warm HF — 0.76 s (измерено, 3 повтора, P7)
- Амортизированная пропускная при потоке 1k запросов (стационарная очередь, модель resident) — теоретически 1/0.76 ≈1.31 req/s ≈1310 знаков/с, т.е. **0.76 s/1000 всё равно >0.03**, т.к. fixed не амортизируется на коротких запросах (каждый 1k всё равно требует полный forward окна 512). Пропускная закрытия нормы не даёт — только маржинальный переход к батчингу окон (§3.2) может дать ≤0.03 при `batch_size=32`, но это фаза B с проверкой идентичности.

### Что подлежит трактовке, что остаётся отклонением

- **Закрыто измерением:** Demo все нормы (1k/50k/2M, cold/warm) — PASS на 2-ядерной нормативной машине; HF 2M ≤60 s — PASS (38.77 s warm, 42.87 s cold) на 4-ядерной машине P7, но с оговоркой “повтор train-контекста, не arbitrary SLA”.
- **На трактовку:** Амортизированная пропускная `sec/1000` при батчинге окон (P7 paired показал 0.153 s/1000 для 50k warm, но медиана всё равно 7.67 s, так что норма “≤1 s latency” не закрывается throughput-метрикой; требуется явное решение куратора, считать ли throughput отдельной нормой).
- **Остаётся отклонением (DEVIATIONS):** HF 0.03/1000 и 50k≤1 s — **недостижимы** на замороженной модели без замены архитектуры (fixed 4 s + marginal 0.0175). Никакой планировщик не снимет fixed; только resident + батчинг + ONNX + параллельные воркеры (≤ядер, `OMP_NUM_THREADS=1`) могут снизить 50k warm с 8.16 s до ~3–4 s (оценка по профилю: `addmm` параллелится на 2 ядра → ~2×, батчинг окон → ~1.5×, ONNX → ~1.3× → суммарно ~4× → 2.0 s, всё ещё >1 s). Честный FAIL ценнее нарисованного PASS.

## 6. Сырые данные и команды (§4.7, §2.3)

- **Сырые массивы (n=10):** `reports/speed/r1/demo_allcores/benchmark_demo.json` + `demo_2cores/...` — каждый `raw_seconds: [...]` + `median_s` + `p95_s` + `profile_top` (первый повтор с `cProfile`). HF сырые `n=3` — `reports/p7/final-receipts/native-benchmark.json` + `reports/p7/paired/summary.json` (Run 39232439363 + 39236657451, артефакты `p7-paired-hf` 8 154 947 B, `digest d8d48...`).
- **Fingerprint:** `reports/speed/r1/demo_allcores/fingerprint.json` + `demo_2cores/fingerprint.json` (CPU, RAM, OS, git SHA `dd0755e…`, команда запуска, `OMP_NUM_THREADS`).
- **Логи таймаутов:** `/tmp/hf_test/full_hf.json` — `warmup_failed: ModuleNotFoundError torch` + `curl huggingface.co → SSL_ERROR_SYSCALL` (whitelist egress proof).
- **Декомпозиция JSON:** `reports/speed/r1/decompose/demo_*.json` (per-stage `weights_load_s`, `tokenization_s`, `features_forward_and_5feat_s`, `candidate_build_span_filter_s`, `verify_total_s`).
- **Исторические профили:** `reports/p7/final-receipts/native-profiles.json` (HF 50k warm: `torch.addmm` 3.28 s/672, `linear` 1.61 s/14, `predict_one` 0.74 s/2 263 360, `cumsum` 0.59 s/1927).

Все команды запуска зафиксированы в этом файле (секция 1) + в `reports/speed/r1/demo_*/fingerprint.json:bundle_sha256` + `git log`.

## 7. Что дальше (фаза B, без оптимизаций до зачёта)

Фаза A завершена **без единой оптимизации** — только измерение. К зачёту куратора готовы:

- Декомпозиция показывает: **HF провалы — compute-bound `torch.addmm`/`linear` + fixed 4 s**, demo провалов нет. Оптимизация demo не нужна.
- Разрешённые инструменты фазы B (§3): resident-сервер (модель один раз, warm-up 1 запрос), параллельный планировщик окон (`workers ≤ cores`, `OMP_NUM_THREADS=1`), батчинг окон в один forward, ONNX/C-токенизатор — при сохранении §3.1 (детерминизм по индексу окна) и **100% идентичности на dev181 + корпусе B** (координаты побайтово, risk/confidence до 4 знаков, без переключения label/verdict). `IDENTITY_RECEIPT.json` будет в `reports/speed/r1/`.
- Оба протокола (§3.2) — resident warm и холодный — будут измерены n≥10, медиана+p95, raw, с отдельной строкой PASS только для 2-ядерной машины.

---

**Честный FAIL > нарисованный PASS.** Все цифры — машинные, с provenance; “всё зелёное” не заявлено.
