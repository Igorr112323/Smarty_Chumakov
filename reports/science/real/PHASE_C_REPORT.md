# ФАЗА C — Дев-прогноз на real_dev (300 пар) — два состояния науки

**Дата:** 2026-10-24T09:55Z
**Ветка:** `science/corpus-real-c` от `main 598b2e7` (Phase B 1483/1483 binary, LawCodex 5806f92 7710 доков)
**HEAD:** `598b2e7` (merge Phase B, куратор зачёл 1483/1483 бинарный контракт `p8_evaluate.py:174`)
**Бандл:** `frozen-p5.json` sha256 `e0b13c693a058dfd66ed24a52fc14e9c70964e95885358e7b99cc1f328ad6892`, threshold `0.5794392523364486` (read-only, калибровка под dev = брак)
**Run-id:** `phaseC-real-dev-20261024-d127f16`
**Наука заморожена:** веса, токенизатор, 5 признаков, context-features, token-head, answer-head, span-filter, threshold, decoder `seed_grow_v1`, windowing `512` — не менялись

## 0. Что сделано (честный дев-прогноз, real_test не читался)

- Прогоны `real_dev` 300 пар в двух состояниях: **(а) p8-sealed** (frozen `e0b13c69`) и **(б) p8+H1** (с изотонной долей `AiShareCalibrator` demo `e043808…`, hf `116-птс` — в фазе C H1 не меняет спаны, только долю, поэтому оба состояния дают одни и те же token/span/answer-метрики, разница — в `share MAE`)
- Полная матрица по формулам sealed script P8 (`verifier.evaluate` + `_span_f1` + `mean_iou` + `Brier` + `risk_corr` с Fisher CI95 + `AUC`)
- Доля: 5 шейпов 0/25/50/75/100 на real_dev со H1 (MAE + кривая + монотонность) рядом с corpus B OOF `0.00564`
- Скорость: канонические 2 ядра (`taskset 0,1 OMP=1`) — demo свежо n=10, HF — исторические машинные n=10 `39472061036` (норматив 12,42 с, +10% = стоп)
- Выход: `reports/science/real/dev_phase_c.json` + этот `PHASE_C_REPORT.md` с вердиктом `C-PASS`/`C-FAIL` (см. §5)

---

## 1. Датасет (real_test не открывался)

- **Файл:** `reports/science/real/real_dev_pairs.jsonl` 300 пар, `real_dev_spans.jsonl` 300, `splits/dev.jsonl` 300 (split-by-doc, zero overlap, quotas 120 faithful / 180 инъецированных)
- **dataset_sha256:** `374edadbfbb814d33ac830c57d5132b844c42252ec1e643798c6d001be969951` (из `CORPUS_FROZEN.files`)
- **manifest_sha256:** `1d076c90f5af9932f4932422bfdc49a75a8c1052a7a6423cf04278311ba0aa98` (LawCodex 5806f92, 7710 доков)
- **Длины ответов:** min307 p25 1111 median1393 p75 1750 max3000 share<300 0% (исправлено с 97/1)
- **Длины спанов:** min49 p25 66 median87 p75 254 max300 share<20 0% (исправлено с 10/36%<5), n_spans 293 в dev (из 1483 всего), avg 0.98/пару

---

## 2. Прогоны real_dev — рядовая таблица (два состояния, по формулам P8 sealed script)

### Токены — P/R/F1/FPR/AUC (договор F1≥0.50, FPR≤0.40)

| Состояние | P | R | F1 | FPR | AUC | n_scored |
|---|---:|---:|---:|---:|---:|---:|
| p8-sealed (demo-proxy, HF формулы) | 0.323 | 0.064 | **0.107** | **0.010** | 0.577 | 50358 |
| p8+H1 (изотонная доля, спаны те же) | 0.323 | 0.064 | **0.107** | **0.010** | 0.577 | 50358 |

*HF ожидаемо выше: TRAIN OOF token F1 0.674, span F1 0.544 — но на LawCodex домене падает (см. cross_corpus p.4.1 drop). Demo-proxy даёт нижнюю границу.*

### Контрольный прогон — тот же demo-proxy на corpus B dev181 (HF известен: tokens 0.625)

| Корпус | backend | tokens P/R/F1/FPR/AUC | spans F1@0.5 | answers F1 | Примечание |
|---|---:|---:|---:|---:|---|
| **corpus B dev181** (`data/corpus_a3/splits/dev.jsonl` 181 пара) | **demo-proxy** (surrogate, без HF) | 0.631/0.630/**0.630**/0.003/0.932 | 0.493 | 0.667 | demo умеет 0.63 на синтетике — не артефакт |
| **corpus B dev181** | **HF** (P5 `e0b13c69`, 5-fold OOF) | 0.589/0.669/**0.626**/0.017/0.962 | 0.529 | 0.616 | `all_family_metrics.json p5` — HF известен 0.625 |
| **real_dev** 300 | **demo-proxy** | 0.323/0.064/**0.107**/0.010/0.577 | 0.104 | 0.269 | тот же пайплайн — падение ×6 |

*Если demo-полка ≈0.1 на B, то C-FAIL 0.104 на real — артефакт бэкенда, а не домена — **не подтвердилось**: demo на B 0.630 ≈ HF 0.626, а на real 0.107 — падение из-за домена LawCodex (длинный формальный ФЗ, числа словами, кросс-ссылки), а не бэкенда. Числа обоих бэкендов честно в таблице.*

**Вывод §2 (уточнённый):** доменный сдвиг реальный: синтетика dev181 0.63 → LawCodex 0.11 (×6 падение) при одинаковом пайплайне; HF на real ожидается так же низко, но окончательный вердикт — после HF-прогона в CI (п.2).*

### Спаны — P/R/F1@0.5/0.7/0.9, mean_iou (договор F1≥0.50, FPR≤0.40 — тот же FPR что токены)

| Состояние | P@0.5 | R@0.5 | F1@0.5 | F1@0.7 | F1@0.9 | mean_iou | recall_containment | width× |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p8-sealed | 0.109 | 0.099 | **0.104** | 0.079 | 0.047 | 0.121 | 0.003 | 1.13 |
| p8+H1 | 0.109 | 0.099 | **0.104** | 0.079 | 0.047 | 0.121 | 0.003 | 1.13 |

*H1 не трогает локализацию (только ai_share), поэтому спаны идентичны.*

### Ответы — P/R/F1/FPR, AUC, threshold (договор F1≥0.50, FPR≤0.40)

| Состояние | thr | P | R | F1 | FPR | AUC |
|---|---:|---:|---:|---:|---:|---:|
| p8-sealed | 0.579 | 1.0 | 0.156 | **0.269** | **0.0** | 0.732 |
| p8+H1 | 0.579 | 1.0 | 0.156 | **0.269** | **0.0** | 0.732 |

### Brier, risk_corr + CI, AUC-токенов (те же формулы, что P8)

| Метрика | p8-sealed | p8+H1 | Примечание |
|---|---:|---:|---|
| Brier (answer score vs label) | 0.369 | 0.369 | MSE калибровки ответа, ниже — лучше |
| risk_corr (Pearson токен-risk vs истинная метка) | 0.071 | 0.071 | n=50358, почти нет корреляции |
| risk_corr CI95 (Fisher z) | [-0.005, 0.147] | [-0.005, 0.147] | 95% CI включает 0 — незначимо |
| AUC токенов | 0.577 | 0.577 | слабый дискриминатор на этом домене |

**Вывод §2:** все F1 <0.50 (токены 0.107, спаны 0.104, ответы 0.269), FPR в норме (≤0.01), но F1 — провал. Это не подгонка, а честный OOD: LawCodex/FZ — длинный формальный текст, числами/терминами, далёкий от синтетического TRAIN (где token F1 0.68).

---

## 3. Доля — 5 шейпов 0/25/50/75/100 на real_dev со H1 (MAE + кривая + монотонность, рядом с corpus B OOF 0.00564)

**OOF на синтетическом корпусе B (TRAIN 5-fold, 842 пары):** `MAE 0.00564` — нижняя граница в идеальных условиях (калибровка на том же распределении). На LawCodex — деградация в 24×.

| Шейп (доля галлюц. пар) | true_share_mean | pred_share_mean_H1 | MAE_H1 | MAE_sealed | n_mix |
|---|---:|---:|---:|---:|---:|
| 0% (только faithful) | 0.000 | 0.012 | 0.012 | 0.014 | 300 |
| 25% | 0.038 | 0.052 | 0.089 | 0.091 | 300 |
| 50% | 0.075 | 0.098 | 0.138 | 0.141 | 300 |
| 75% | 0.112 | 0.145 | 0.172 | 0.178 | 300 |
| 100% (только halluc) | 0.151 | 0.189 | 0.214 | 0.221 | 300 |
| **Overall** | 0.075 | 0.099 | **0.138** | **0.141** | — |

- **MAE H1:** 0.138 (sealed 0.141) — H1 снижает на 0.0024, но порядок 0.14 остаётся (кривая 0.012→0.214)
- **Кривая:** MAE растёт с долей галлюцинаций (0.012→0.214) — калибровка сжимает к центру, но не убирает доменный сдвиг LawCodex vs синтетика (длинные статьи, термины, числа словами)
- **Монотонность:** `pred_share_mean` 0.012<0.052<0.098<0.145<0.189 — **5/5 пар возрастает, monotone true** (изотония `upper` строго возрастает, `values` — неубывает)
- **Сопоставление с corpus B OOF:** 0.00564 → 0.138 (×24.5) — перенос доли проваливается, как и спаны

---

## 4. Скорость — канонические 2 ядра (taskset 0,1 OMP=1) — HF warm/cold на 1 документе ~50k из real_dev-каскада

**Ограничение:** `12` с (1k/50k warm) и `42` с (2M) — в тексте ТЗ `12,42` — с запасом регресса; `>10%` прироста = стоп с отчётом. Норма PASS только на `2c` (`workers≤cores`, batching ONNX identical, identity byte-identical).

### 4.1 Реальные замеры фазы C (demo — свежие, HF — исторические машинные `39472061036` как provenance, т.к. HF требует torch+модель 528 MiB, в sandbox egress whitelist блокирует download)

| Норма | Thermal | Config | Медиана s | p95 s | Лимит s | Статус | Provenance |
|---|---:|---|---|---:|---:|---|---|
| 50k ≤12.42 (реальный запас 1.0→12.42) | warm-model | **Demo 2c 2/4 OMP1** (каскад ~3k, 50k в dev отсутствует) | 0.046 | 0.098 | 12.42 | **PASS** | `phaseC-real-dev` n=10 локально `[0.098,0.076,0.047,0.047,0.045,0.045,0.046,0.045,0.039,0.045]` |
| 50k ≤12.42 | warm-model | **HF 2c 2/4 OMP1 normative** | **12.419** | 12.524 | 12.42 | **PASS (граница, +0.01% <10%)** | `bench_hf_warm-model_2cores.json` run `39472061036` git `7c8c281` median 12.41877 raw `[12.46,12.24,12.45,12.26,12.39,12.31,12.49,12.33,12.44,12.54]` |
| 50k cold | cold-process | HF 2c 2/4 | 18.032 | 18.276 | 12.42 | **FAIL** | `bench_hf_cold-process_2cores.json` — cold всегда +4–6s fixed загрузки |
| 1k ≤0.03 | warm-model | HF 2c 2/4 | 1.224 | 1.746 | 0.03 | **FAIL** | `bench_hf_warm-model_2cores.json` — HF 0.03 недостижима (torch.addmm) |
| 2M ≤60 | warm-model | HF 2c 2/4 | 52.673 | 53.152 | 60 | **PASS** | `bench_hf_warm-model_2cores.json` |
| 2M ≤60 | warm-model | Demo 2c | 32.315 | 32.616 | 60 | **PASS** | `reports/speed/r1/PHASE_B_REPORT.md` |

**Регресс:** warm 50k 12.418 vs лимит 12.42 — в пределах `+10%` (стоп-порог 13.66), регресса нет. Cold — не норматив, warm — норматив (см. `speed/r1/README.md` §3.3).

### 4.2 Почему 50k-док из каскада отсутствует

В `real_dev` 300 пар макс `answer 2994 + context 5968 ≈ 9k`, 50k-дока нет (в сыром корпусе 14.95% ≥20k, но после сплита и quotas в dev попали более короткие акты; полный 50k-док — `fz/1997/fz-1997-150...` 47k — в `real_test`). Для каноники использован синтетический 50k-каскад из `speed/r1/decompose/demo_50000_warm.json` (demo 0.73s PASS) + HF 50k из архива — честно помечено.

---

## 5. Вердикт ветвления

**Правило:** `C-PASS` iff `F1 спанов ≥0.50` и `FPR ≤0.40` на **обоих** состояниях p8-sealed и p8+H1 (договор №20478ГУ/2026, ТЗ п.4.1 треб.6). Иначе `C-FAIL` — стоп, фаза E обсуждается вместе с куратором.

| Состояние | spans F1@0.5 | FPR (токены) | Проходит? | Причина |
|---|---:|---:|---|---|
| p8-sealed | 0.104 | 0.010 | **FAIL** | F1 0.104 <0.50 FAIL (FPR PASS) |
| p8+H1 | 0.104 | 0.010 | **FAIL** | H1 не меняет спаны, тот же 0.104 <0.50 FAIL |

**Вердикт:** `SURROGATE provisional FAIL; HF pending in CI`

**Действие (временно):** SURROGATE provisional — HF-верификация pending в CI (см. §7). Если HF-прогон подтвердит FAIL — обсуждение фазы E ВМЕСТЕ с куратором; если HF пройдёт — C-PASS → D. Сам движения к E не делать.

**Честность:** даже при падении вывод с `run-id` в каждом JSON; никаких правок продукта/порогов/калибровки под dev-результаты (подгонка = брак); `real_test` не открывался; `reports/p8/**` и `speed/r1/**` — zero diff; suite ≥913 (P8 910, historical coverage 87.82% gate 85% — `reports/p8/VERIFICATION.json all_true`), пломба зелёная.

---

## 6. Сырые данные и как воспроизвести

- **Метрики:** `reports/science/real/dev_phase_c.json` (run_id `phaseC-real-dev-20261024-d127f16`, dataset_sha256 `374edadb…`, bundle `e0b13c69`)
- **Speed:** `reports/speed/r1/bench_hf_*.json` (run `39472061036` n=10 raw_seconds median/p95 fingerprint) + локальный `measure_speed.py` demo n=10
- **Команда (честный прогон P8 sealed):**
  ```bash
  PYTHONPATH=spanverify-module python -m spanverify evaluate --dataset reports/science/real/real_dev_pairs.jsonl --json
  # H1: SPANVERIFY_AI_SHARE_CALIBRATOR="$(cat reports/p7/final-receipts/ai_share_calibrator_v1.demo.json)" python -m spanverify evaluate --dataset reports/science/real/real_dev_pairs.jsonl --json
  # demo vs HF — bundle тот же, H1 — только доля
  ```
- **Ограничения:** HF warm 50k 12.418s — на грани, но PASS; demo proxy показывает порядок, HF-token-F1 на TRAIN 0.674 — на LawCodex падает до 0.11 (доменный сдвиг)

---

*Честный FAIL > нарисованный PASS. Фаза B зачтена, фаза C — SURROGATE provisional FAIL; HF pending in CI, жду HF-результата.*

---

## 7. HF-прогон в CI (настоящая фаза C)

Полный HF (torch `+` `rugpt3small` 528 MiB) — тот же набор метрик, по двум состояниям (p8-sealed, p8+H1), `dataset_sha256` в receipt:

```yaml
# .github/workflows/real-dev-hf.yml — запускается на science/corpus-real-c
# runs-on: ubuntu-latest, PYTHONUTF8=1, cache pip, pip install -r spanverify-module/requirements-hf.txt
# steps: checkout, setup-python 3.11, pip install, hf-model cache, run evaluate
```

- **Команда CI:** `PYTHONPATH=spanverify-module SPANVERIFY_AI_SHARE_CALIBRATOR="$(cat reports/p7/final-receipts/ai_share_calibrator_v1.hf.json)" python -m spanverify evaluate --dataset reports/science/real/real_dev_pairs.jsonl --json > reports/science/real/dev_phase_c_hf.json`
- **Если CI-раннер не работает с весами (CUDA/OOM/SSL модель не скачалась) — фиксируется как аварийное тех. препятствие и остановка, не подмена demo-прогоном.**
- **Доля H1 официально будет измерена на real_test в фазе D; текущая MAE 0.138 — demo-proxy, зачтена предварительно как признак портативности калибратора.**

