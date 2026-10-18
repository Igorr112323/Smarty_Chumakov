# P4: context-enhanced token-head — фактический CPU experiment

**Результат: улучшение подтверждено на dev, совместный quality gate НЕ ДОСТИГНУТ.**
Token F1 впервые выше .50; strict span F1 остаётся .39572. Это не полная приёмка договора.

## Run, протокол и неизменность источников

- Настоящий успешный [run39164044941](https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/39164044941), code SHA `4f0099384e165bae7006a64fbca52e2440cc4c18`; публикация `3432ab6c015b1befcf5f3206826aeee4ced23763`.
- [Artifact12151395245](https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/39164044941/artifacts/12151395245), `p4-context-39164044941`, 8,840,515 compressed bytes. Основные runner evidence — в нём и в [каталоге run](spanverify-module/reports/experiments/p4_context_head/run39164044941/). Дополнительные post-hoc анализ, аудит и локальные regression/CI receipts опубликованы отдельно в этом каталоге ветки, а не задним числом в исходном artifact.
- Первый технический run39164033456 остановился **до обучения** из-за ошибочного имени исходного artifact. Исправлен на `p3-postprocess-39160817890`; failure receipt сохранён. Это не второй научный подбор.
- Train842, dev181. **Official test177 не читался.** `pairs.jsonl` также не читался: его hash `b9a4cf05a9078644dd539470481a1b8079aed2d9201e195c3ecd901276ff7c8e` проверен как frozen manifest declaration, не выдаётся за повторное хеширование всех1200 строк.
- Фактически прочитанные train SHA256 `75c2504acefb6a3c325f8b3b94c1e4da9ec441d9440c1cfd07c16eb8e49040a2`; dev `cba705f836d9c25f28fc162b5cbf89b52bfabe7c47c56e2a4d6d652cded2c2dd`.
- Source OOF:842 pairs,265 documents,5 исходных folds,46607 scored tokens; missing0, shared documents0. Fold membership не генерировался заново. Дополнительный actual train/dev audit:265/62 documents, shared docs/groups0; test не открывался. Все50 fit/held document receipts повторно проверены на отсутствие пересечения для P4 matrix.
- Original P2 model:97494B, SHA256 `33112af1997e12afe9aee07a6f84eb29760c9165e1a928938b2979110d822e9e`; combined115068B, `87298b01ec157b20c6a06cee9244d75ce1d7882a30f20abddbb6e9781bbf46ab`. Исторические P2/P3 evidence не изменялись.
- Сохранены обе исходные gzip-копии. Распакованный OOF5742956B SHA256 `5ca0ff5121707e950a4a710c8d733c092aeca477857ca146847c5de0ef98959b`; gzip `96facdab18c9cb01c7a53a523c700c1b829c0105714fdc68a2cddf12ff01297d`. Распакованный source dev3932617B `e54d290b4894246e61a256af5568af500bc979997d7ab72e7f6ea237129d31ce`; gzip `926ca9d19ca100017df9349af2f8c4c69c6adc56ee5a4baca0dea4c0e15d1465`.
- Строго8 cache shards,1195 keys, **1928 hits/0 misses**; любой miss прекращает запуск. Fresh HF inference/API/GPU — нет. Backbone/window512/chunking, corpus, tokenization, scored predicate, smoothing и исходные8 features прежние.

## До обучения: контекстная гипотеза

Диагностика выполнена до любого fit и до dev replay (1.995s validation/diagnostics). Она относится к **исходным P2 OOF risks при историческом tau**, не к будущей модели. Окна — соседние scored tokens внутри одного ответа; self исключён, на краях clamp. Полные min/Q25/Q50/Q75/Q90/Q95/Q99/max соседних risks и **всех8 base features** для каждого k=1/2/3 и TP/FN/FP-clean/FP-dirty сохранены в `context_diagnostics.json` (84144B).

Пример при k=3:

| Исходная группа | neighbor max risk Q25 / Q50 / Q75 | neighbor mean risk Q50 |
|---|---|---|
| TP | .71981 / .85380 / .91184 | .57652 |
| FN | .47333 / .53501 / .63830 | .35120 |
| FP clean | .61320 / .67291 / .74086 | .43485 |
| FP dirty | .60927 / .66533 / .74404 | .43044 |

FN median neighbor-max для k1/2/3 = .51047/.53202/.53501; clean-FP = .66192/.66787/.67291. У FN есть сильный верхний хвост (k3 Q90=.83332), но **простая версия «вокруг FN риск выше, а вокруг clean-FP низкий» не подтвердилась**. Ожидать исправления всех FN одним расширением нельзя.

Исходные ошибки: FN617 =506 edge +111 interior;217 полностью пропущенных gold из467;54 gold фрагментированы; FP tokens1664 =913 clean +751 dirty. Категории wholly-missed/fragmented относятся к gold spans и не складываются с token counts.

Практическая польза производных соседних **признаков**, а не простого соседнего risk, подтверждена ниже: при одинаковом P3 decoder OOF objective вырос .36073→.43077, независимо на dev — token F1 .46025→.52214 и strict span F1 .35052→.39572. Это частичное подтверждение полезности контекста, не доказательство чистой разделимости классов по рискам.

## Модель: зарегистрированные10 кандидатов

Pure/versioned `context_features_schema_v1`:8 base +6 derivatives на каждый base =56. Derivatives: previous, next, left-k max, right-k max, bilateral neighbor mean, self−mean; k1/2/3; scored subsequence within pair, clamp, finite. Пять договорных HF features остаются исходными пятью: attention_entropy, ctx_attention_mass, ctx_attention_mass_norm, ctx_max_similarity, embedding_density. Position/length/digit не новые HF measurements; все aggregates явно производные.

Тот же sklearn CPU GBDT: log_loss,80 trees,learning_rate .05,depth3,min_leaf20,subsample1,init zero,seed42. Python3.11.17,numpy1.26.4,sklearn1.5.2. Balanced sample weights вычисляются **только на fit fold**, положительный вес умножается на1/2/4. Полные defaults реконструированы без fit в `full_estimator_params.json`; counts/weights/fold/model hashes — `fit_receipts.json` и `sample_weight_receipts.json`.

Контроль base8,w1 использовал исходный trainer. **Max risk delta к P2 OOF =0.0**. Исторические token TP767/FP1664/FN617/TN43559,F1=.4020969856,FPR=.0367954360; span190/1052/277,F1=.2223522528 совпали. STOP guard не сработал, поскольку воспроизведение точное.

Первая стадия: все10 моделей сравниваются с **одним frozen P3 decoder**, без индивидуальной подгонки порогов:

| k | w | OOF token F1 | strict span F1 = objective | token FPR |
|---|---|---|---|---|
| base8 | 1 | .485024 | .360731 | .021405 |
| 1 | 1 | .506203 | .387733 | .022643 |
| 1 | 2 | .411727 | .337621 | .052650 |
| 1 | 4 | .309170 | .252806 | .097826 |
| 2 | 1 | .546819 | .423121 | .018641 |
| 2 | 2 | .428859 | .348797 | .047166 |
| 2 | 4 | .328670 | .270236 | .089600 |
| **3** | **1** | **.568507** | **.430769** | **.015833** |
| 3 | 2 | .456876 | .378426 | .042589 |
| 3 | 4 | .357647 | .293999 | .077438 |

Критерий: max min(token F1,strict span F1), FPR≤.30; tie меньший FPR, затем фиксированный порядок. Выбран **k3,w1**. Нельзя заключать, что w2/4 хуже при любых возможных порогах: здесь именно заранее зарегистрированный fixed-decoder model comparison.

Вторая стадия: **1300 прежних P3 configs на НОВЫХ OOF risks победителя**, не повторный старый frozen-risk scan. Выбран low=`.46823304343873473`, high=`.6287422757811378`, bridge1,min_seed2, attribution exact. OOF token853/671/531/44552,P=.559711,R=.616329,F1=.5866575,FPR=.01483758; strict spans191/203/276,P=.484772,R=.408994,F1/objective=.44367015. Полные grids раздельно: `model_grid_scores.json`, `grid_scores.jsonl.gz`; решения — `selected_model.json`, `selected_decoder.json`. Это train selection, не независимая приёмка.

50 OOF fits +1 final fit на842 train pairs. Final46607 tokens:1384 positive/45223 negative. Только после freeze сделан один public dev replay победителя; ни k/w, ни decoder, ни P1 tau по dev не менялись.

## Независимый public dev181: P2 → P3 → P4

Counts tokens/answers/verdicts: TP/FP/FN/TN. Для spans: TP/FP/FN; строгий IoU≥.5, все rule/evidence/number_attribution spans включены. У spans нет самостоятельного TN: FPR в существующем gate — явно token proxy, не выдуманный span TN. AUC — для непрерывных token risks и calibrated answer score, не для произвольного множества отрицательных spans.

| Метрика | P2 | P3 | P4 |
|---|---|---|---|
| Token counts |179/335/108/7347|165/265/122/7417|171/197/116/7485|
| Token precision |.348249|.383721|.464674|
| Token recall |.623693|.574913|.595819|
| Token F1 |.446941|.460251|**.522137**|
| Token FPR |.043608|.034496|**.025644**|
| Token AUC |.946305|.946305|.961514|
| Strict span counts |39/201/57|34/64/62|37/54/59|
| Strict span precision |.162500|.346939|.406593|
| Strict span recall |.406250|.354167|.385417|
| Strict span F1 |.232143|.350515|**.395722**|
| Containment |17/96=.177083|13/96=.135417|16/96=.166667|
| Mean width ratio, **только contained gold** |14.209686|5.239085|2.530743|
| Expanded-label F1, diagnostic only |.077381|.175258|.160428|
| Answer counts |49/14/47/71|49/14/47/71|49/14/47/71|
| Answer P/R |.777778/.510417|.777778/.510417|.777778/.510417|
| Answer F1/FPR |.616352/.164706|.616352/.164706|.616352/.164706|
| Calibrated answer AUC |.733824|.733824|.733824|
| Verdict counts |58/18/38/67|64/23/32/62|58/18/38/67|
| Verdict P/R |.7632/.6042|.7356/.6667|.7632/.6042|
| Verdict F1/FPR |.6744/.2118|.6995/.2706|.6744/.2118|
| Verdict flagged share |.4199|.4807|.4199|

P1 frozen threshold `.5794392523364486`, raw/calibrated scores, scaler/calibrator, original fields и решения не изменились. Verdict logic прежняя; числа verdict изменились относительно P3 вследствие новой локализации, не новых правил. Watch verdict FPR≤.30 соблюдён. Совпадение verdict counts P2 и P4 не означает идентичность всех per-pair verdicts.

P2/P3 baseline metrics заново рассчитаны **из сохранённого evidence**, не новой HF inference; все сравнения совпали (`historical_evidence_parity.json`). Public P4 vs save/load:181 pairs, mismatch0; pure builder+decoder vs public:0; finite raw risks, token flags, hard-share/spans и точные text slices проверены. P1 vs frozen source и reload — delta0.0/mismatch0. Исходные P1 fields отдельно сравнены целиком с исключением только разрешённых token_head/context_spec/decoder.

## Почему общий gate всё ещё отрицательный

Dev token gate .50/.40 пройден, answer gate пройден, но strict span F1=.39572<.50. FPR не компенсирует F1. Резерв .60/.30 совместно также не достигнут. AUC .96151 не делает локализацию автоматически хорошей.

Дополнительный **post-hoc**, без fit/search/tuning, разбор тех же сохранённых dev states (`posthoc_plateau_diagnostics.json`) воспроизвёл public spans/flags с0 mismatches:

- Token FP197 =146 clean +51 dirty; FN116 =92 edge +24 interior.
- Final strict spans37TP/54FP/59FN; FP39 clean +15 dirty. **51 predictions вообще не пересекают gold**,3 пересекают с IoU<.5. Поэтому основные остаточные FP — не только «слишком широкая attribution».
- Mask-only:35TP/52FP/61FN,F1=.382514; attribution-only2TP/6FP/94FN; unchanged coverage0 spans. После неизменённого merge итог37/54/59. Rules не выключались и не исключались из principal metric.
- Из96 gold:37 имеют strict match;17 не имеют eligible scored token при frozen low;40 имеют eligible token, но не retained mask overlap;2 имеют retained overlap без final strict match. Это геометрические post-hoc категории, не новые варианты decoder. Fragmented gold с несколькими final predictions:0.
- Исторические final FP clean/dirty: P2 102/99 → P3 43/21 → P4 39/15. Span FN57→62→59: recall остаётся серьёзным ограничением. P4 лучше P3, но не превосходит P2 по span recall .38542 против .40625.

OOF oracle, **не кандидат и не вход в selection**:
- Лучший любой monotone scalar threshold на новых OOF risks:tau=.6561100225145975,token F1=.5526002382,696TP/439FP/688FN.
- Ideal merge/gold snap при этом tau:200TP/0FP/267FN,F1=.5997001499; oracle удаляет FP и знает истинные границы — непригоден для deployment.
- .55260 — потолок только семейства одного монотонного порога, не hysteresis (зарегистрированный structured decoder дал .58666). .59970 — **условная оценка при token-oracle tau**, не глобальный span optimum и не достижимое обещание .60. Вывод — сохраняются трудные ошибки детекции/покрытия и отбора компонент; одной чисткой широких границ plateau не объясняется.

## Реализация, аудит, тесты и CI — раздельно

- `spanverify/context_head.py`: pure builder, strict versioned portable56-column head/export; runtime без numpy/sklearn. `engine.py`: versioned dispatch, fail-closed malformed/missing/inconsistent context spec, legacy path сохранён. Legacy8 tree-index guard **не ослаблялся**; для56 создан отдельный exporter.
- Workflow `.github/workflows/p4-context-head.yml`: dispatch + явный `.github/p4-context.trigger`; P2/P3 triggers и обычный CI не отключались. Failure evidence сохраняется.
- Runner preflight: **87 targeted passed**, в том числе14 P4 cases (clamp first/last/short, pair/doc isolation, determinism, malformed/schema/NaN, export/reload, public P1 parity, unchanged source features).
- Full local suite:816 passed /22 failed /2 skipped за325.12s; те же22 исторических failing nodes, new0/resolved0 относительно P3. Полные traceback — `local_full_suite.log`, machine-readable statuses — `test_status.json`, JUnit — `full_suite_junit.xml`. Ничего не исключалось для зелёного результата. Диагноз исторических групп: calibration/sentinel; decimal coverage; workflow registry/provenance; cache max_length; stale METRICS/docs; verdict fixtures; unsupported --no-test; attribution/number words/CLI/selftest; два старых60min workflow limits.
- Обычный CI run39164048308 **FAILED**, не green. Quality job остановился на глобальном Ruff; contract job failed; test jobs exit2. Обнаружена и исправлена новая collection-проблема: P4 tests импортируют numpy, а обычный test job устанавливал только pytest/PyYAML, а jobs с полным registry также не устанавливали CPU training stack. Добавлен отдельный `requirements-token-head-tests.txt` (numpy1.26.4/scipy1.13.1/sklearn1.5.2) в test/quality/contract/numbers CI jobs. Обычный `requirements.txt` и delivery/build dependencies оставлены прежними; тесты не пропускаются, runtime deps/HF model не менялись. Исходный runner scientific run уже имел эти же pins, поэтому measurement не переобучается из-за исправления test environment. `ci_status.json` содержит фактические annotations; прямой download CI traceback дал EOF, причина не выдаётся за скачанный traceback. Новый ordinary CI после финального push отслеживается отдельно.
- Ruff/Black для затронутых runtime/script/tests проверены из `spanverify-module` (правильный first-party import context); documentary scripts — из корня с тем же config. Это не утверждение, что весь исторический repository lint green.
- Новый token head100127B; combined120455B, SHA256 **`e6c96ea4390dbf124540c413594110180f84d9be9d5c13ad1577232b70ee9e48`**. Full file hashes/sizes — `published_files_sha256.json`; исходный runner manifest не переписывался. Единственное расхождение pre-stdout manifest size — measurement.log, потому что tee дописывает финальный stdout после manifest (8906→12667B); отмечено в `evidence_audit.json`.
- CPU measurement2110.545s:10-model OOF2014.246s, decoder28.404s, final fit54.645s, dev/parity6.559s. Это **не** speed benchmark продукта.

## Договор

[coverage-matrix.json](reports/contract/coverage-matrix.json) / [coverage-matrix.md](reports/contract/coverage-matrix.md) — построчно достигнут/частично/не достигнут с реальными ссылками. Исторический demo/release не считается текущей P4 HF приёмкой. Official test, производительность .03s/1000,50k≤1s/2M≤60s, интерфейсы и поставка exe≤9MB/package≤.6MB остаются отдельными неподтверждёнными гейтами; SHA нового bundle не является SHA готового exe. Общая готовность договора **НЕ ДОСТИГНУТА**.
