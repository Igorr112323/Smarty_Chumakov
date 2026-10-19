# FINAL REPORT — P8, 19 октября 2026

## Вывод

**Численный официальный quality gate PASS; строгая execution freeze НЕ соблюдена (pre-seal synthetic fit в отменённом CI). Поставка — RC, НЕ latest. Весь договор НЕ выполнен.**
Замороженный продукт P7 (`b9cf9acdab1ca7bf082d03629ea7e599597f3fa8`) один раз успешно измерен на всех 177 test-парах через публичный `Verifier`. Никакого обучения/подбора по test или B. Bundle 122874 B, SHA256 `e0b13c693a058dfd66ed24a52fc14e9c70964e95885358e7b99cc1f328ad6892`, tau 0.5794392523364486, P1/P4/decoder/span_filter/калибраторы неизменны. Побайтовый regression manifest: [frozen-production-sha256](p8/frozen-production-sha256.json).

Только два текущих индекса: [EXPERIMENTS_INDEX](EXPERIMENTS_INDEX.md) и [contract/coverage-matrix.json](contract/coverage-matrix.json). Остальные сводки — evidence или явно исторические, не конкурирующие индексы.

## 1. Официальный test177

[Run39266451529](https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/39266451529), commit **dd4454b05867495018abe5a2f657fc4d6d0fd9a6** (полный фактический SHA также записан в [result.json](p8/run39266451529/official/result.json)); продукт — точно b9cf9ac. Числа ниже — TEST, не dev.

| Семья | TP / FP / FN / TN | Precision | Recall | F1 | FPR | AUC |
|---|---|---:|---:|---:|---:|---:|
| Tokens | 191 / 139 / 90 / 8848 | 0.578788 | 0.679715 | **0.625205** | **0.015467** | 0.971717 |
| Strict spans | 47 / 38 / 50 / нет span-TN | 0.552941 | 0.484536 | **0.516484** | token proxy 0.015467 | не определён |
| Answers, фиксированный tau | 51 / 15 / 46 / 65 | 0.772727 | 0.525773 | 0.625767 | 0.187500 | raw 0.697423 / calibrated 0.684407 |
| Verdicts | 62 / 19 / 35 / 61 | 0.7654 | 0.6392 | 0.6966 | 0.2375 | не определён |

Знаменатели: 9268 scored tokens (281 positive / 8987 negative); 97 gold spans, 85 predicted; 177 answers (97 positive / 80 negative). Gold, scored predicate, tokenization, сглаживание и kept-spans+rules сохранены. Strict matching: прежний greedy IoU≥0.5. Containment recall **0.288660**, mean width ratio **1.339262**, только по contained gold. Не выдумывается span TN/FPR. Verdicts сохраняют исходное округление frozen metric function.

Реально исполненный gate: token F1≥0.50, strict span F1≥0.50, token-FPR proxy≤0.40 **одновременно PASS**; внутренний ориентир 0.90/0.10 не достигнут. [Gate code](../spanverify-module/scripts/p8_evaluate.py), полный gate_report в result.json.

Предварительный metadata-only аудит до labels: **177, 7 >512, max1266**; единый `window512_full_answer_v1`, без усечённого legacy cache. [Все позиции по ID](p8/run39266451529/official/pre-label-long-audit.json). Пустой feature cache: **0 hits /177 misses**, свежая extraction+public inference **381.292202784 с**, fit_calls=[]; это не inference из старого dev-cache. Счётчики относятся к pair feature cache; внутренний window memoization отдельно не инструментировался. Per-pair JSONL SHA256 **7607c513a760b68860e89aa3255d7f44942cc6a678c83bca3c641128067b738f**. Точный исправный hash и размеры — в [artifact-manifest](p8/run39266451529/official/artifact-manifest.json), оригинальный поток — в linked artifact и финальном архиве. Dataset test.jsonl SHA256 **232d2583480f2237fc48d10eb4413043bf35f59be5a10138fbf53235ce83ba88**.

### Разрешённое инфраструктурное восстановление, не вторая quality-попытка

Первый run39266027598, commit4b28132, остановился **до первого завершённого предсказания**: wrapper ошибочно запрещал `transformers.train(False)` при model.eval(). Полный traceback сохранён в [infra-first-attempt](p8/infra-first-attempt/annotations.json), исходный artifact сохранён. В recovery проверены нулевой evidence.jsonl и отсутствие result.json. Исправлен только wrapper, допускающий False eval-mode для torch/transformers; fit/train(True) запрещены. Исходная consumed-tag не удалялась; отдельная atomic recovery-1 tag использована один раз; run_attempt=1. Плохих чисел до восстановления не было; повторного измерения после полученных метрик нет. Labels первого abort-run успели загрузиться после metadata-аудита, но ни одного scoring result не возникло; это не скрывается.

## 2. P2→P3→P4→P5: только исторический DEV-reference

181 dev, legacy512, исходные знаменатели и числа НЕ переписаны. Не сравнивать их как независимый test и не смешивать с P8 test.

| Этап / run | Token F1 | Token FPR | Strict span F1 | Answers TP/FP/FN/TN |
|---|---:|---:|---:|---|
| P2 /39155368906 | 0.446941 | 0.043608 | 0.232143 | 49/14/47/71 |
| P3 /39160817890 | 0.460251 | 0.034496 | 0.350515 | 49/14/47/71 |
| P4 /39164044941 | 0.522137 | 0.025644 | 0.395722 | 49/14/47/71 |
| P5 /39170981235 | 0.626427 | 0.017443 | 0.528736 | 49/14/47/71 |

[Каноническая неизменённая таблица](../spanverify-module/reports/experiments/p5_span_filter/run39170981235/all_family_metrics.json). P5 dev tokens192/134/95/7548; spans46/32/50. Отдельный P7 long-v1 reference answers52/14/44/71 — **тоже не test**; 178 short byte-identical, 3 long изменились по заранее утверждённому контракту.

## 3. Внешний corpus B, test-only, 1000/1000

RuHallu-RAG, pinned upstream commit `345907f983da91e1b7720b5944d982ea9c9aeb83`; два CSV проверены SHA256 (`f1e9043bb092eeef2c765e52652d40839aeab84268ed3badb0d417fb987d8f6f`, `f44f10e574ce547f832f2215189e40e158d06be3dbeec712926f1c59df60375e`). Адаптированный pairs SHA256 **a520f5fcab3cd9d3988e6b0039efe733153f40a0808dc2b7c5ff5ad7613c70d3**. Четыре непересекающихся шарда по250, aggregate только объединяет существующие1000 evidence, не повторяет inference.

Реальная human-разметка: **423 точных spans /333 positive answers**; поэтому поддерживаются все семьи, а не выдуманные answer-to-span offsets. Одна upstream неоднозначность сохранена как первый exact occurrence, лицензия raw-корпуса не подтверждена. Raw source не добавлен в Git, provenance/ограничение лицензии сохранены. Ни fit, ни калибровка, ни подбор thresholds, ни SIRIN/AlignRuScore comparator не выполнялись.

| Семья | TP / FP / FN / TN | Precision | Recall | F1 | FPR | AUC |
|---|---|---:|---:|---:|---:|---:|
| Tokens | 152 / 510 / 8867 /18559 | 0.229607 | 0.016853 | **0.031402** | 0.026745 | 0.437462 |
| Strict spans | 6 /182 /417 / нет TN | 0.031915 | 0.014184 | **0.019640** | token proxy0.026745 | не определён |
| Answers | 203 /271 /130 /396 | 0.428270 | 0.609610 | **0.503098** | **0.406297** | raw0.621505 / calibrated0.616485 |
| Verdicts | 221 /319 /112 /348 | 0.4093 | 0.6637 | 0.5063 | 0.4783 | не определён |

28088 scored tokens (9019 positive/19069 negative);423 gold/188 predicted spans;1000 answers333positive/667negative. Containment0, width ratio0 (нет contained gold; frozen function's empty convention, не свидетельство хорошей ширины). Во всём B **0 >512**; pair-feature cache **0hits/1000misses**, shard elapsed sum **1856.906875125с** (parallel wall time не равен сумме); fit_calls=[] во всех четырёх. Падение переноса большое и опубликовано без корректировки модели. **B НЕ является контрактным gate и не отменяет/не подменяет официальный test gate.** [Полные метрики](p8/run39266451529/external/result.json), исходные per-shard reports/manifests рядом; artifacts и финальный архив содержат все per-pair JSONL.

## 4. Скорость, масштаб и честные отклонения

Контролируемый paired-run39232439363, один CPU7763,48 rows; полный [P7 TIMINGS](p7/TIMINGS.md), [P7 REPORT](p7/REPORT.md).

| Fixture | Before warm | After warm | Интерпретация |
|---|---:|---:|---|
| demo1k | см.48 raw rows | 0.016517с | не ухудшён по заданному абсолютному допуску |
| demo50k | 1.04183с | 0.714563с | ≤1с достигнуто |
| demo2M | timeout75с | 30.090007с | ≤60с достигнуто |
| HF1k | 3.57060с | 0.763317с | улучшение, но малый норматив не достигнут |
| HF50k | 3.89361с **усечённый old path** | 7.676519с full | НЕ выдаётся за speedup сопоставимой семантики |
| HF2M | timeout75с | 39.824549с | full mapping ≤60с |

Финальный трёхкратный P7 HF50k warm median **8.168092351с**, max8.213496: норматив ≤1с **НЕ достигнут**, compute-bound frozen model (addmm/linear/treepredict), не основание менять модель после test. HF2M warm median38.769417с; отдельный индексный before91.417021→after39.301874с в одном процессе с **полным public JSON parity**, кроме latency_ms;2M chars/282830 native tokens/last end2M/all finite/coverage PASS. Повторяемый train fixture — лучший случай window cache, не гарантия произвольного уникального текста.

AI-share: train-only original5doc-fold OOF PAVA HF MAE **0.406824→0.2565365102**, demo0.449732→0.298901; pooled q-curves **nonmonotone**, статус частичный. Синтетическая доля символов НЕ доказательство реального LLM-authorship; opt-in отдельный artifact, default raw неизменён. [CALIBRATION](p7/CALIBRATION.md).

## 5. Evidence hygiene и инциденты

Полный хронологический индекс включает SUCCESS/failure/cancelled и SHA/date/native evidence/receipt. Legacy `reports/experiments/a3/experiment.json` (37609202780) и `reports/METRICS.json` (header14Oct), а также реальные module-relative originals, начинаются с **SUPERSEDED / historical baseline before P1–P5**, successor=EXPERIMENTS_INDEX. Числа старых payload не исправлялись: [семантические hashes](p8/legacy-payload-hashes.json), regression проверяет исходные ключи/значения. Mixed legacy METRICS содержит более поздние append-only compatibility поля; старый header не делает всё тело одним измерением14Oct. Новый CURRENT_METRICS — лишь совместимый числовой payload для прежних проверок, НЕ третий индекс; его прежние dataset-блоки исторические.

[Регрессии](../spanverify-module/tests/test_p8_evidence_hygiene.py): provenance всех aggregate metric documents вне per-run каталогов, successor у SUPERSEDED, последний run каждой семьи в индексе, неизменность legacy payload и всех prediction files. Неизвестные historical local run явно обозначены unrecorded, не выданы за выдуманный GitHub run. Старые canonical per-run P2/P3/P4/P5 сохранены побайтово.

Исторический test-read incident НЕ скрыт: offending `TestSelectRelevantChunks::test_on_corpus_a3_sample` читал unsplit corpus, потом first20; точное время чтения неизвестно. Это был aggregate/fixture read, не прежняя официальная оценка. Guardfix **5ff79f9a7bed33c00c6278f67d68ddff73f8325c2026**,18Oct23:34:59Z, train-only; conftest запрещает reserved177 во всех тестах. Source pairs SHA **b9a4cf05a9078644dd539470481a1b8079aed2d9201e195c3ecd901276ff7c8e**. Ранее заявленная гарантия отсутствия всех pre-seal CI fits оказалась НЕВЕРНОЙ: always-contract bypass описан ниже; workflow barrier, dispatch403 и инфраструктурное восстановление описаны в [PROTOCOL](p8/PROTOCOL.md).

## 6. Поставка и полная проверка

Финальный Windows build выполняет полные тесты3.10–3.12, exe/pyz smoke, строгие лимиты9,000,000/600,000B и реальную offline HF autonomy; training fallback запрещён. Публикация stable/latest разрешается **только** совместным официальным gate; tag `p8-final-20261019` — проверочный release-envelope, frozen runtime version остаётся1.3.1. При FAIL — только RC/prerelease, latest не меняется. Notes обязательно сохраняют все указанные отклонения, даже при PASS качества.

Проверенные P7 ориентиры до новой сборки: exe8,979,956B /pyz232,874B; отдельно модель **551,290,714B**, библиотеки **870,588,148B** (sum1,421,878,862B **плюс Python/application**, полный installed SDK total не измерен). Это не «всё HF внутри9MB», не прежняя неточная оценка ~500MB libs. Итоговые P8 build/release/CI receipts и контрольные суммы находятся в `reports/p8/DELIVERY.json` и linked artifact manifests; новый архив сохраняет source, binary, исходные per-pair доказательства. Публикационный receipt создаётся позже самого архива, без рекурсивного перепаковывания.

Полный локальный pytest без exclusions и ordinary CI finalHEAD отражаются фактическими JUnit/log/run receipts: [full-suite.log](p8/full-suite.log), [full-suite.xml](p8/full-suite.xml). Локально: **910 collected /908 passed /2 pre-existing skipped,0 failures,407.793с**, JUnit подтверждает. Два старых optional skips сохраняются, новые skips не добавлены. `tests_excluded=false`. Переоценки official177 ради сборки/CI не выполняется.

### P8 actual Windows receipt and publication recovery

Build-source **654a48e3f074654d62fb4adb38a6c37555c53681**, run **39271014161**: all three full Python suites SUCCESS, all build/smoke/size/offline steps SUCCESS. **exe8,981,233B /pyz232,874B /onedir ZIP9,148,872B**. exe SHA256 `93f354eeea49633b72df0c8d3548ba912465c3b3309d9e72452fed815f274362`; pyz `4006d891adfb9cd0c8554178b0c1d505aaf5b21b8b78bf72e800faddd33ae76f`. [Raw sizes/checksums and manifests](p8/run39271014161/windows/), [actual HF offline receipts](p8/run39271014161/hf-autonomy/windows-hf-autonomy.json). Both exe/pyz HTTP200/backendhf, offline/firewall block true; product registry keys unchanged, not a global OS registry trace.

The SOURCE workflow conclusion remains **FAILURE, publication only**: PS scalar-string flag splat produced `no matches found for '-'` before creating a release. [Failure step log and raw hash](p8/infra-windows-publication/). Do not call that source workflow SUCCESS. Recovery publishes the SAME already-validated bytes from artifact12197265568 (18,159,922B), using a real argv list; no rebuild, no new177/B scoring. Future PS flag is explicitly string-array typed. Final publication waits for all ordinary CI on its exact documentation HEAD; resolved IDs are included in archive FINAL-PUBLICATION.json and the later committed publication receipt.

Actual P8 weight file551,290,714B; entire six-file model/tokenizer directory **554,277,308B**; libraries **870,588,148B**. Model-directory+libs **1,424,865,456B**, PLUS Python/app/filesystem overhead. Complete installed SDK total remains unmeasured/partial. Separate compressed portable artifact12195949376 is **637,048,763B**, not the installed footprint and not the small exe size.

Metadata-only log transfer initially refused ANSI escape output; recovered with explicit `--allow-escape-sequences` (receipt workflow39276221841). This did not run any verifier or fit. Historical CI cancellations track normal successive pushes, not excluded tests or hidden quality reruns. All final source/workflow checks remain enabled.

Infrastructure queue failover: hosted x64 jobs remained queued without steps for >25minutes on19Oct; Linux ordinary CI and metadata-only publisher moved to ubuntu-24.04-arm. ALL suites, Python versions, coverage85%, Windows matrix and existing checks retained, no new skips/exclusions. This does NOT rerun official/B scoring, change the model, rebuild the accepted Windows bytes, or replace P7 AMD speed measurements with ARM timings.

## Final process incident — execution freeze NOT ACHIEVED; conservative RC

Late audit found an EARLY ordinary-CI cancellation bypass: run39266027509/source4b28132, contract job118107779326 used `if: always()` and ran despite cancelled barrier. Its contract artifact12192328736 was created **2026-10-19T14:12:03Z**, before successful official job completed14:26:51Z. `scripts/contract_check.py::Benchmark.participation` invokes a temporary `ParticipationModel.fit` on synthetic240 examples. This violates the requested ban on ANY pre-seal fit, including temporary diagnostics. This is NOT excused as “only a test”. The frozen product/P1/P4/bundle/tau were NOT altered, and official/B data were NOT used for that fit; recorded inference-worker fit_calls=[] does not describe separate CI processes.

[Incident metadata, original run/artifact provenance](p8/infra-freeze-barrier/). Job cancellation timestamps are internally inconsistent (completedAt precedes startedAt); artifact creation timestamp is retained instead of inventing a precise fit start time. The actual fit instant is not independently reconstructed. The executed standard contract code includes the fit; resulting artifact is preserved permanently in final archive.

Fix: even always-contract now REQUIRES barrier success; regression checks exact condition. All existing post-seal checks remain enabled. **No repeat official measurement** and no model tuning. Numeric joint gate remains PASS, but clean execution-freeze compliance is **NOT achieved**. Conservative final release **v1.3.1-rc.p8.20261019**, prerelease/latest=false; stable/latest **v1.3.1 remains unchanged**. Earlier conditional stable-publication descriptions above describe authorization policy, NOT an executed promotion. Full original failures/cancellations remain visible.

Queue failover outcome: the temporary ARM pool attempt started metadata publishers but did NOT clear ordinary CI's runner/concurrency queue. Final ordinary CI is restored to **x64 ubuntu-24.04**, pinned rather than ubuntu-latest, with a fresh **ci-p8-acceptance** concurrency namespace (same workflow, all checks, all Python versions, Windows matrix and85% coverage). This isolates acceptance from stale cancelled runs, NOT from any test or failing metric. Publisher also uses known-working ubuntu-24.04 x64. First publication39278757995 failed CI-wait timeout without publishing; original failure preserved. No source-code/model changes, exclusions or test weakening.

Publication provenance: acceptance source fd1b863 ordinary push39287371393 and PR39287382275 BOTH SUCCESS (including full6-combination PR matrix). Metadata publication39287371656 passed all frozen-file/binary/evidence validation, then `gh release create` failed; NO release/tag created, latest unchanged. Original traceback/log preserved. Recovery targets the **exact acceptance/publication GITHUB_SHA**, not older Windows-build654 source. Binaries remain the identical SHA-validated original Windows bytes, no rebuild or inference. CLI stderr captured explicitly for audit; release tag represents acceptance source, Windows build source remains separately attributed.
