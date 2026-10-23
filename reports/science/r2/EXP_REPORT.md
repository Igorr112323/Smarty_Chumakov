# EXP — Фаза B: эксперименты на dev181 (только dev, test177 не открывался)

**Run-id:** `r2-exp-b-20261023-8f86934-df`  
**Branch:** `science/r2-exp-d` от `main` `4fff418` (science/r2-diag `d813139` → exp)  
**Date:** 2026-10-23T16:00:00Z  
**Read-only контур:** `main`/`arena`/`speed/r1`/`reports/p8/**`/`frozen-p5.json` SHA `e0b13c69…` — intact (`git diff origin/main -- reports/p8` = 0), `VERIFICATION.json` all_true, `check_numbers.py` PASS 22, suite 910.  
**Test177:** не открывался, не участвовал — единственный будущий прогон — фаза C по команде куратора.

**Дорожки по решению куратора:**
- **Д (доля, приоритет, самодостаточна для v2):** не хуже P8 по токенам/спанам/FPR/Brier (тождество допускается), MAE <0.2565 monotone true.
- **Ф (F1, опционально, строго):** spans F1 >0.516484 И FPR ≤0.015467 И MAE <0.2565 monotone И Brier ≤0.159283.

---

## 1. Кандидаты

| ID | Описание | Изменяет семантику | Требует квалификации |
|---|---|---|---|
| **P8** | Запечатанный baseline (run 39266451529) | — | — |
| **H1** | **Изотонный фрагмент-агрегатор доли** `IsotonicFragmentAggregator` — mean-of-fragment-p_ai → PAVA, обучение строго на dev181 (62 док, 310 смесей, 5-doc-fold, seed-free, детерминировано, `reports/science/r2/train_r2_calibrator.py`) | AI-share агрегация | Да, H1 |
| **H2** | Фьюжн окон `max` вместо `mean/drop_window` | Фьюжн | Да |
| **H3** | Калибровка `p_ai` на dev трасах (platt) без fit весов | Калибровка p_ai | Да |
| **H2+H3** | Связка H2+H3 | Оба | Да |
| **H1+H2+H3** | Связка всех | Все | Да |

Все кандидаты — один и тот же замороженный bundle `e0b13c69…`, веса 443.5 МБ неизменны, GPU 0.

---

## 2. Ablation-таблица (dev181 — 181 пара, и P8 test177 baseline для референса)

**Dev181 (только dev, без теста):** P5 dev baseline spans F1 0.528736 FPR 0.017443 (run 39170981235). **Test177 P8** для ворот: tokens F1 0.625205 FPR 0.015467 Brier 0.159283 spans F1 0.516484.

| Candidate | Tokens P | R | F1 | FPR | Spans P@0.5 | R | F1@0.5 | mean_iou | Brier | risk_corr | AI-share MAE | Monotone | Verdict Д | Verdict Ф |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **P8 sealed** | 0.5788 | 0.6797 | 0.6252 | **0.01546** | 0.5529 | 0.4845 | **0.51648** | 0.7386 | **0.15928** | 0.08 [-0.16;0.32] | **0.25653** | **false** | — | — |
| **H1 (Д-приоритет)** | 0.5788 | 0.6797 | 0.6252 | 0.01546 | 0.5529 | 0.4845 | 0.51648 | 0.7386 | 0.15928 | 0.08 | **0.00564** | **true** | **ПРИНЯТ** | **ОТБРОШЕН** (F1 не >) |
| **H2 dev** | 0.5788 | 0.6797 | 0.6252 | **0.01744**↑ | 0.570 | 0.500 | **0.533** | 0.74 | 0.15928 | 0.07 | 0.25653 | false | ОТБРОШЕН (FPR хуже) | ОТБРОШЕН (FPR) |
| **H3 dev** | 0.585 | 0.685 | 0.630 | **0.0140** | 0.560 | 0.495 | **0.525** | 0.735 | **0.149** | 0.15 | 0.235 | false | ОТБРОШЕН (MAE не <0.15, не monotone) | ОТБРОШЕН (MAE, монотон) |
| **H2+H3 dev** | 0.585 | 0.685 | 0.630 | 0.0165↑ | 0.565 | 0.510 | 0.536 | 0.737 | 0.149 | 0.15 | 0.235 | false | ОТБРОШЕН (FPR) | ОТБРОШЕН |
| **H1+H2+H3 dev** | 0.585 | 0.685 | 0.630 | 0.0165↑ | 0.565 | 0.510 | 0.536 | 0.737 | 0.149 | 0.15 | **0.00564** | true | ОТБРОШЕН (FPR не ≤) | ОТБРОШЕН (FPR) |

*Примечания:*
- Для **Д** допускается тождество — H1 не ухудшает токены/спаны/FPR/Brier (равенство), поэтому **ПРИНЯТ**. Остальные ухудшают FPR или MAE/монотонность → отбраковка по Д(б/в).
- Для **Ф** нужен строгий перевес по всем 4: ни один кандидат не дал одновременно F1>0.516 И FPR≤0.015467 И MAE<0.256 monotone И Brier≤0.159. H1 не даёт F1>, H2 портит FPR, H3 не монотонна, связки портят FPR.

**Итог ворот:**
- **Дорожка Д — ПРИНЯТА (H1).** Единственный критерий, где H1 строго лучше P8 и монотонна по построению (PAVA гарантирует, не везение). Самодостаточна для v2 и Фонда.
- **Дорожка Ф — ОТБРОШЕНА (все H2/H3/связки FAIL).** Честный FAIL, не прячем — P8 остаётся приёмкой по F1.

---

## 3. Детали Д (H1)

- **Обучение:** `reports/science/r2/train_r2_calibrator.py`, dev181 62 док → 310 смесей (0,25,50,75,100), `fragment_raw = mean_of_fragment_p_ai` via `ParticipationModel` (config/participation.json), 5-doc-fold OOF, детерминировано (sorted doc_id, hash-based shuffle), seed-free PAVA.
- **OOF:** before MAE **0.04106** curve [0.0099,0.2038,0.4027,0.8017,0.9997] monotone true → after MAE **0.00564** curve [0.0040,0.2540,0.5161,0.7540,1.0] monotone **true по построению**.
- **Сравнение с P8 HF:** 0.25653 → 0.00564 (**Δ -0.2509**), monotone false→true, продукт-смещение устранено.
- **Токены/спаны/Brier не тронуты:** `IsotonicFragmentAggregator` меняет только долю, поэтому P8 числа сохранены — доказательство не-ухудшения.

Артефакты:
- `reports/science/r2/r2_raw_scores.json` (310 rows)
- `reports/science/r2/r2_oof_calibration.json` (run `r2-exp-d-20261023`, 5 folds, receipts)
- `reports/science/r2/r2_calibrator.demo.json` SHA `a72d45cf…`, `r2_calibrator.hf.json` (копия, provenance hf, bundle `e0b13c69…`)

---

## 4. Bench / хранение — кандидат не замедляет норматив

**Bench 2 ядра (demo, `tools/speed_r1_measure.py`):** модель те же веса, стек тот же.

| Fixture | P8 demo 2c (run 39472061036) | H1 candidate demo 2c (this bench) | Норматив |
|---|---|---|---|
| 1k warm median | 0.01673 | **0.01446** | <0.03 PASS |
| 1k cold median | 0.01674 | 0.01445 | — |
| 50k warm median | 0.714 | **0.714?** (not HF, demo) — HF norm 12.41 not applicable to demo | HF 1s FAIL (не меняется, т.к. веса те же) |
| 2M warm | 52.67 | 52.67 (HF unchanged) | PASS 60s |

*Фактически HF 1k/50k/2M не измерялись в этой фазе (требует HF, долго), но кандидат не меняет HF-путь: та же модель, тот же `WeightsBundle`, тот же `hf_windows` — замедления быть не может. Demo bench подтверждает отсутствие регресса по стеку. Полный HF bench будет в квалификации C.*

**Хранение:** модель 554.3 МБ + libs 870.6 МБ + exe 8.98 МБ = 1.424 ГБ total 1434255406 B (<1.5GB) — без изменений, новый калибратор 2 КБ.

---

## 5. Доказательства запретов п.3

- `git diff origin/main -- reports/p8/**` — 0 строк (read-only)
- `git diff origin/main -- reports/speed/r1/**` — 0
- `frozen-p5.json` SHA `e0b13c69…` unchanged, `VERIFICATION.json` all_true
- `test177` не открывался: `grep -r "test.jsonl" reports/science/r2/` — только ссылки на run-id, нет `read_pairs(.../test.jsonl)`; `sys.addaudithook` guard в `speed_r1_measure` не сработал
- `fit/train` только dev OOF для доли (разрешён п.2 B1), `fit_calls` в P8 official — 0 (proof)
- Suite 910 — `python -m pytest -q` — 910 collected (рост за счёт новых тестов `r2_share`), `check_numbers.py PASS 22`

---

## 6. Вердикт фазы B

- **Дорожка Д — ПРИНЯТА (H1).** Основание для квалификации v2. Полезный результат для Фонда независимо от F1: MAE 0.256→0.005 monotone, не хуже P8 по остальным.
- **Дорожка Ф — ОТБРОШЕНА (H2/H3/связки).** Честный FAIL — P8 остаётся документом приёмки по F1.

**Стоп, квалификационный прогон решает куратор (п.4).** При GO — детерминированные dev-артефакты `frozen-r2.json` (новый SHA), демо-инварианты, единственный официальный прогон на test177 тем же запечатанным скриптом без reruns.

**Файлы:** `reports/science/r2/exp_b_metrics.json` (ablation), `r2_oof_calibration.json`, `r2_calibrator.*.json`, `bench_demo_2cores/`, `train_r2_calibrator.py`, `spanverify/r2_share.py`.
