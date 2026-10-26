# REALGEN-R1 — итоговый отчёт (фазы A и B)

Ветка `arena/233cd01f-smarty-chumakov`, база `main de11ee9`. Дата 2026-10-26.

## Протокол

- `real_test` 1200 (sha f40c737c…) НЕ читался ни в одной роли; `real_test_viewed:false` в обоих фазах.
- Фон: 6510 документов LawCodex/ru@5806f92 минус 1200 test-hosting, overlap=0 (ассерт в тестах).
- main не тронут напрямую; пломба P8, reports/p8/**, reports/speed/**, VERIFICATION.json, релизы/теги — не изменялись.
- run-id: только верифицируемые Actions-прогоны + local-no-ci-evidence с provenance; журнал — RUN_ID_AUDIT.json.
- Экстракция данных из CI — вручную через check-runs output.text; auto-commit из workflow отсутствует.
- Suite: 967 собранных nodes (было 950), покрытие ≥85%, ruff/black зелёные, CI-матрица не сужалась.

## Фаза A

- Ответов: **400**; ошибочных **169 (0.4225)** при квоте ≥40%; чистых 71, неясных 160.
- `atom_unclear_rate = 0.4324`; спанов 1373: a=402, b=303, c=668.
- Длины ответов min/med/max: 26/774/1954; спанов: 1/4/67 символов.
- Темп: 526.29 ответов/час в пересчёте на один поток; 8 шардов CI параллельно, wall-clock ≈ 382.6 с на раунд.
- Remedy: не потребовался: R1 уже даёт ≥40% ошибочных ответов. R2 сгенерирован как contingency (400 ответов), в заморозку не вошёл.
- Чувствительность правила (c): заморозка `any_window` → 0.4225; строже `best_window` → 0.3800; без (c) → 0.2375. Правила (a)/(b)/(d) не менялись.
- Сплиты: dev 300 / test_realgen 100 (ЗАПЕЧАТАН, доступ по шифт-коду), overlap=0.
- Отклонение бюджета: pytorch_model.bin = 551.291 МБ (директива 443.5 МБ, бюджет 550 МБ → +1.291 МБ). Модель research-only в CI, продуктовая пломба не тронута.
- Префикс sha f8198329… фона из директивы не воспроизведён: готового файла нет ни в одной ветке; опубликован полный sha нашей канонической карты 6510 id: `48190fffd0deb02ec6df71e865ff3cf9d581ab00fb6831d7d5c014232023bbdb`.

## Фаза B (только dev 300)

- Токенов: 51472 (позитивных 1828); span-окон: 2180 (позитивных 437); OOF 5-fold по sha256(doc_id), bootstrap 5000.
- Composite OOF: token-AUC **0.954731** CI95 [0.942672, 0.96432], span-AUC **0.725425** CI95 [0.699055, 0.75082].
- rubert-tiny2 fine-tune (batch 8, lr 2e-5, 3 ep, веса 117.53 МБ, 2748 шагов): token-AUC **0.808308** CI95 [0.778615, 0.835505], span-AUC **0.741393** CI95 [0.718138, 0.764238].
- Ворота (≥0.70 / ≥0.55 / CI>0.5, обе модели): **PASS**; composite_pass=True, rubert_pass=True.

### per_feature_AUC (все 27)

| признак | семейство | AUC | ориентированный |
| --- | --- | ---: | ---: |
| `attention_entropy` | old5 | 0.541875 | 0.541875 |
| `ctx_attention_mass` | old5 | 0.469829 | 0.530171 |
| `ctx_attention_mass_norm` | old5 | 0.47552 | 0.52448 |
| `ctx_max_similarity` | old5 | 0.451263 | 0.548737 |
| `embedding_density` | old5 | 0.540021 | 0.540021 |
| `sentence_cosine_answer_doc` | E1 | 0.403672 | 0.596328 |
| `sentence_cosine_answer_chunk_max` | E1 | 0.411182 | 0.588818 |
| `contrast_doc_chunk` | E1 | 0.528469 | 0.528469 |
| `rolling_logprob_next_token` | E1 | 0.468078 | 0.531922 |
| `lexical_number_attribution_window` | E1 | 0.640861 | 0.640861 |
| `stopword_norm` | E1 | 0.350313 | 0.649687 |
| `digit_density_answer` | R2 | 0.782091 | 0.782091 |
| `citation_density_answer` | R2 | 0.727022 | 0.727022 |
| `token_is_numeric` | R2 | 0.794094 | 0.794094 |
| `token_is_citation` | R2 | 0.585824 | 0.585824 |
| `token_len_chars` | R2 | 0.346258 | 0.653742 |
| `token_letters` | R2 | 0.25822 | 0.74178 |
| `position_ratio` | R2 | 0.528939 | 0.528939 |
| `sentence_length_chars` | R2 | 0.428081 | 0.571919 |
| `local_number_density` | R2 | 0.83201 | 0.83201 |
| `local_stopword_share` | R2 | 0.294479 | 0.705521 |
| `doc_coverage` | R2 | 0.534342 | 0.534342 |
| `distinct_digits` | R2 | 0.792092 | 0.792092 |
| `uppercase_ratio` | R2 | 0.434457 | 0.565543 |
| `punctuation_ratio` | R2 | 0.713626 | 0.713626 |
| `unit_flag` | R2 | 0.537067 | 0.537067 |
| `type_token_ratio_answer` | R2 | 0.453609 | 0.546391 |

> Метки REALGEN получены детерминированными правилами по числам/ссылкам, поэтому лексические признаки (token_is_numeric, local_number_density, citation_density) ожидаемо дают высокий AUC. Это НЕ доказательство обнаружения галлюцинаций в свободной генерации и НЕ перенос на real_test 1200; ворота измеряют только внутреннюю separability этого корпуса.

> пересчитаны на ai-forever/rugpt3small_based_on_gpt2 (last-layer attention / hidden states) — не вызов замороженного hf_features_chunked, определения сохранены

> E1/R2 признаки — детерминированная реимплементация по опубликованным именам в E0_DIAG/E1_SIGNAL; исходный код E1HUNT_R2/R3 в этом чекауте отсутствует (проверено по trees веток)

## Файлы

| файл | байт | sha256 (12) |
| --- | ---: | --- |
| `reports/science/real/REALGEN_PLAN.json` | 8036 | `99855041101a…` |
| `reports/science/real/RUN_ID_AUDIT.json` | 7707 | `c6019aeb1dc7…` |
| `reports/science/real/REALGEN_BACKGROUND.json` | 589570 | `c720c549f3fc…` |
| `reports/science/real/REALGEN_SELECTION.json` | 284991 | `db98cc713331…` |
| `reports/science/real/REALGEN_TEMPLATES.json` | 3037 | `7851061136b2…` |
| `reports/science/real/REALGEN_DOCS.jsonl` | 7114244 | `809b1375b4f5…` |
| `reports/science/real/REALGEN_ANSWERS.jsonl` | 2111547 | `9a98ae3affc3…` |
| `reports/science/real/REALGEN_ANSWERS_DEV.jsonl` | 1593048 | `cd902ac8c33d…` |
| `reports/science/real/REALGEN_ANSWERS_TEST_REALGEN.jsonl` | 518499 | `8aac288649bd…` |
| `reports/science/real/REALGEN_LABELS.jsonl` | 503601 | `550b9a162c52…` |
| `reports/science/real/REALGEN_SPANS.jsonl` | 376471 | `e431a1538636…` |
| `reports/science/real/REALGEN_SPLITS.json` | 36714 | `d2d83ca61150…` |
| `reports/science/real/REALGEN_FROZEN.json` | 181201 | `372ff6b88204…` |
| `reports/science/real/REALGEN_PHASE_A.json` | 13686 | `3e5872e2cd43…` |
| `reports/science/real/REALGEN_PHASE_A.md` | 3876 | `6f1ce990faaf…` |
| `reports/science/real/REALGEN_PHASE_B.json` | 10184 | `35ff350f9be2…` |
| `reports/science/real/REALGEN_PHASE_B_OOF.jsonl` | 11166726 | `5f445aeb8c11…` |
| `reports/science/real/realgen_common.py` | 8121 | `949662c7dc7a…` |
| `reports/science/real/realgen_build_background.py` | 8313 | `df22f7ead1c9…` |
| `reports/science/real/realgen_generate.py` | 10925 | `1962a760ddbe…` |
| `reports/science/real/realgen_publish.py` | 3634 | `744175965ee7…` |
| `reports/science/real/realgen_extract.py` | 4828 | `278b61950561…` |
| `reports/science/real/realgen_label.py` | 36857 | `061e7f5a67e4…` |
| `reports/science/real/realgen_phase_b.py` | 32866 | `bc7a061e572a…` |
| `reports/science/real/realgen_final_report.py` | 10385 | `ee5dea7b2bf4…` |

## Run-id / provenance

- seq 1: `local-no-ci-evidence` — ревизия состояния репозитория, проверка доступности хостов и реконструкция background 6510
- seq 2: `verified-actions-run` [40288730753](https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/40288730753) — генерация R1 (400) + R2 (400) ответов моделью ai-forever/rugpt3small_based_on_gpt2, 8 шаблонов × 2 режима, seed от doc_id
- seq 3: `verified-actions-run` [40288730574](https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/40288730574) — первые два новых workflow-файла не были зарегистрированы в reports/CI_WORKFLOWS.json / docs/ЗАДАНИЯ_CI.md и имели timeout-minutes > 40 → test_ci_workflows.py + test_workflow_limits.py
- seq 4: `local-no-ci-evidence` — детерминированная авторазметка атомов, проверка квот, заморозка REALGEN_FROZEN.json и отчёт фазы A
- seq 5: `verified-actions-run` [40298420234](https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/40298420234) — smoke-тест фазы B на 30 dev-ответах (диагностика: eager attention вместо SDPA)
- seq 6: `verified-actions-run` [40298872296](https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/40298872296) — полная батарея фазы B: old5 (eager attention GPT-2), E1/rubert sentence cosine, R2 16 признаков, composite LogReg OOF 5-fold, rubert-tiny2 FT OOF (batch 8 / lr 2e-5 / 3 ep), bootstrap 5000; 300 dev, 51472 токена
- seq 7: `verified-actions-run` [40298872199](https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/40298872199) — причина: Windows CRLF-конвертация замороженных REALGEN-файлов ломала sha-тест; диагноз по зелёным 5 jobs + локальной репродукции
- seq 8: `verified-actions-run` [40302780074](https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/40302780074) — зелёный suite на финальной голове (push-матрица 2 jobs + линтер/покрытие/контракт/пломба)
- seq 9: `verified-actions-run` [40302871659](https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/40302871659) — подтверждение зелени suite на PR #24; pilot/external jobs skipped по триггеру pull_request (не сужение матрицы: полная матрица тестов 6/6 success)

## CI / suite

- PR #24: https://github.com/Igorr112323/Smarty_Chumakov/pull/24 (merge НЕ выполнялся).
- Полная матрица CI на PR: run 40302871659 — 6/6 тестовых jobs success, линтер/покрытие success, контракт success, Final CI read-only P8 seal success; pilot/external skipped по триггеру, не по правке матрицы.
- Push-CI на той же голове: run 40302780074 — success.
- Промежуточный провал Windows (run 40298872199) и его причина (CRLF) зафиксированы в RUN_ID_AUDIT.json seq 7.
- Локально: 967 collected, 964 passed, 3 skipped; ruff/black: All checks passed.

## Известные ограничения (честно)

1. HF-веса недоступны из sandbox: генерация и все HF-измерения выполнены в GitHub Actions, данные извлечены вручную через check-runs. Локальная репродукция без HF невозможна.
2. Марковская модель ruGPT3small без instruction tuning часто продолжает/копирует текст акта; это естественная генерация, но ответы не похожи на ответы современного ассистента.
3. Метки получены правилами по числам/ссылкам → высокая лексическая separability. Положительные ворота НЕ означают обобщения на реальные галлюцитации и НЕ снимают исторический STOP.
4. E1/R2 признаки — реимплементация по именам из E0_DIAG/E1_SIGNAL: исходный код E1HUNT_R2/R3 в этом чекауте отсутствует (проверено по trees всех веток).
5. test_realgen запечатан и не использован; real_test 1200 не открывался.
