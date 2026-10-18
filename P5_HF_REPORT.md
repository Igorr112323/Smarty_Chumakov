# P5 — span keep/drop на замороженных P4 risks

**Совместный quality gate .50/.40 на независимом public dev181 ДОСТИГНУТ:**
token F1 **.6264274062**, FPR **.0174433741**; strict span F1 **.5287356322**.
Это не official-test приёмка и не подтверждение скорости, поставки или всего договора.
Резерв .60/.30 совместно НЕ достигнут: strict span F1 ниже .60.

## 1. Реальный run и порядок стадий

- Единственный P5 [CPU run39170981235](https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/39170981235) — **success**.
- Измерительный commit **`977e9dc81b378cd2b775b815864898aad92496d4`**, публикация evidence **`1a450c225419d86e9c253eee33db75223676ab1c`**.
- [Artifact12152925261](https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/39170981235/artifacts/12152925261): `p5-span-39170981235`, **20,107,334 compressed bytes**.
- Основные runner files одновременно в artifact и [каталоге run](spanverify-module/reports/experiments/p5_span_filter/run39170981235/). Отдельные post-hoc source/pool audits, полный локальный suite и CI annotations дополнены в ветке; они не выдаются за задним числом добавленные файлы исходного artifact.
- Remote/local HEAD перед работой совпали с `93bf462aa0e1be344ffa33f21f5259a29368b581`. История сохранена, push обычный; P2/P3/P4 каталоги не изменены.
- До fit: OOF validation →282 генератора →195124 candidate rows →coverage/oracle/feature diagnostics (**18.313s**). Затем5 cross-fit span heads →pooled OOF probabilities →4794-point search →final head на всех train candidates →bundle freeze (**179.625s от начала**) →единственный winner public dev replay с обязательной reload parity.
- Всего **182.132s**, training/search/final-fit после diagnostics вместе161.311s, dev/parity/baseline/final logging около2.507s. Это не speed benchmark продукта.
- **6 span-classifier fits, 0 token-classifier fits**. P4 head, risks, P1, backbone,5 HF columns, tokenization/scored predicate, smoothing, window512, corpus/folds и rules logic не менялись. Fresh HF inference/API/GPU — нет.
- Cache1195 keys,8 shards, **1204 hits/0 misses**. Любой отсутствующий train/dev key прекращает запуск до вызова backend.
- **Official test177 и pairs.jsonl не читались.** Train842/dev181; train265/dev62 документов без пересечения. Pooled span folds взяты из source состояний; каждый документ только в одном held fold.

Проверка opaque SHA legacy source_dev gzip была частью source integrity preflight; его строки/метки/метрики до freeze не разбирались и не использовались. P4 dev evidence разбирается только после записи выбранного config, final model и freeze receipt.

## 2. Пререгистрация: генератор, признаки, голова, s

До запуска зафиксированы:

- Low: train-OOF risk quantiles [.50,.60,.70,.75,.80,.85,.90] плюс P4 low.
- High: [.90,.94,.96,.975,.985] плюс P4 high; low≤high; bridge0/1/2; min_seed1/2. Получилось **282 допустимые cells**, без последующего расширения.
- В запросе не было численного списка keep-квантилей; до fit публично и в коде зафиксированы **[0,.05,.10,.20,.30,.40,.50,.60,.70,.80,.90,.95,.975,.99,1]**, плюс all-keep0 и all-drop nextafter(1,2). Получилось17 различных s.
- **Одна pooled span-голова, общая для всех cells**, а не отдельный подбор типа/параметров на каждую cell: StandardScaler + logistic regression, L2,C1,balanced class weights по fit-fold,lbfgs,max_iter1000,tol1e-6,seed42. Пять scaler/model fits исключают held документы полностью; один финальный fit повторяет тот же pooled training protocol.
-19 зарегистрированных входов: scored_count,seed_count,max/mean/min risk,risk_mass,low/high fractions,char/token width,bridged gaps/tokens,left/right gradients,exact attribution anchor,rule overlap tag,relative start/end,P1 calibrated answer score.
- Gold/clean/dirty/doc identity не входят в признаки. Метка хранится отдельно: прежний greedy IoU≥.5 **внутри decoder cell**, с прежним порядком и без snap границ.
- Candidate dedup по pair+cell+offsets:0 duplicates.195124 строк,53146 positive/141978 negative. Все OOF probability indices выровнены, отсутствующих состояний/NaN нет.
- Final class weights: negative .6871627999,positive1.8357355210. Fold weights/counts/документы/coefficients/scales — `fit_receipts.json`; финальная голова и полный feature order — `combined_bundle.json`. Все fits сошлись за93–120 iterations; export parity ≤**1.50e-15**, проверялась на всех fit rows того же fitted object, не отдельном переобучении.

## 3. OOF coverage и oracle — диагностика, не кандидаты

Все282 cell diagnostics записаны **до fitting/selection и до dev** в `candidate_diagnostics.json`. Gold denominator —467. Примеры из зарегистрированной сетки:

| Cell | low / high; bridge / seeds | Any-overlap coverage | IoU≥.5 coverage | Без какого-либо candidate | Ideal keep F1, только candidates | Ideal keep + обязательные rules F1 |
|---|---|---|---|---|---|---|
|267, P4 generator|.468233/.628742;1/2|188/467|185/467|279|.567485|.551227|
|0, liberal Q50/Q90|.072223/.418658;0/1|410/467|258/467|57|.711724|.689384|
|218, выбранный|.418658/.418658;1/1|410/467|354/467|57|.862363|.835681|

**Никакого gold-snap:** oracle оставляет greedy-matched candidates с их реальными boundaries. В отдельной колонке обязательные unchanged rules добавляются и проходят обычный merge; их FP не скрываются. Oracle использует gold и не является deployable head или критерием выбора. Это условный label-oracle, не поиск глобального optimum по всем подмножествам после rule merge.

У selected generator число кандидатов на ответ Q25/median/Q75 =1/2/3,p95=6,max17; на375 clean train answers =1/2/3,p95=6,max14. На gold число пересекающих candidates median1,p951,max1 (включая нули для57 пропусков).

Все19 features имеют matched/unmatched quantiles в `span_feature_spec.json`. Примеры pooled median matched / unmatched: max risk **.89799/.64093**, mean risk **.72649/.52906**, scored count **4/3**, char width **32/18**, P1 calibrated score **.57944/.51887**. Это описательные распределения, не причинная feature importance; одинаковые offsets разных cells остаются разными зарегистрированными наблюдениями.

Дополнительный read-only audit уже замороженного train pool:8407 уникальных pair+offsets без cell, union any coverage415/467,strict392/467;52 gold без любого overlap и75 без strict candidate. Median уникальных overlaps на gold4,p958; на clean answer median7,p9527. **Union разных cells не является разрешённым новым генератором и не участвовал в выборе.** Этот дополнительный audit не выдаётся за pre-fit файл.

P4 oracle .552600 token / .599700 ideal snap сохранён как исторический факт, не пересчитывался. Более высокий P5 label-oracle не опровергает его: здесь другой либеральный candidate pool и покрытие, а старый .599700 был условной оценкой при другом token-oracle tau, не глобальным потолком любых двухстадийных схем.

## 4. Выбор — только train-OOF

Критерий **max min(token F1,strict span F1), token FPR≤.30**, tie меньший FPR, затем фиксированный порядок.

Выбраны:

```text
cell = 218
low = high = 0.4186579753072923  # оба train-risk Q90
bridge = 1
min_seed = 1
head = pooled standardized logreg, C=1
s = 0.5081811955031305          # Q70 pooled cross-fitted span probabilities
```

`min_seed=1` выбран **по полному OOF grid**, не из известных P4 dev FN. Ни одного добавления/tau изменения после dev не было.

OOF: tokens **919TP/422FP/465FN/44801TN**,P=.68530947,R=.66401734,F1=.67449541,FPR=.00933153; strict spans **214TP/105FP/253FN**,P=.67084639,R=.45824411,F1/objective=.54452926. Это selection metrics, не независимая приёмка.

Полные **282 ×1 head ×17 s =4794** records — [`grid_scores.jsonl.gz`](spanverify-module/reports/experiments/p5_span_filter/run39170981235/grid_scores.jsonl.gz); `selected_config.json`, `generator_cells.json`, `keep_thresholds.json`, `oof_span_probabilities.jsonl.gz` позволяют воспроизвести выбор.

## 5. Независимый public dev181: P2 → P3 → P4 → P5

Сохранённые historical baselines воспроизведены из evidence с0 расхождений, не новой HF inference и не иным decoder. Counts tokens/answers/verdicts = TP/FP/FN/TN; spans = TP/FP/FN. Gold, denominator и strict IoU≥.5 не менялись. В P5 token flags — токены внутри **kept spans + rules**, как задано; исторические token flags не пересчитывались под новую конвенцию. Span FPR — существующий token proxy, не выдуманный span TN.

| Метрика |P2|P3|P4|P5|
|---|---|---|---|---|
|Token counts|179/335/108/7347|165/265/122/7417|171/197/116/7485|**192/134/95/7548**|
|Token precision|.348249|.383721|.464674|**.588957**|
|Token recall|.623693|.574913|.595819|**.668990**|
|Token F1|.446941|.460251|.522137|**.626427**|
|Token FPR|.043608|.034496|.025644|**.017443**|
|Token AUC|.946305|.946305|.961514|**.961514**|
|Strict span counts|39/201/57|34/64/62|37/54/59|**46/32/50**|
|Strict span precision|.162500|.346939|.406593|**.589744**|
|Strict span recall|.406250|.354167|.385417|**.479167**|
|Strict span F1|.232143|.350515|.395722|**.528736**|
|Containment|17/96|13/96|16/96|**25/96=.260417**|
|Mean width ratio, только contained gold|14.209686|5.239085|2.530743|**2.098564**|
|Answer counts|49/14/47/71|49/14/47/71|49/14/47/71|**49/14/47/71**|
|Answer P/R|.777778/.510417|.777778/.510417|.777778/.510417|**.777778/.510417**|
|Answer F1/FPR|.616352/.164706|.616352/.164706|.616352/.164706|**.616352/.164706**|
|Calibrated answer AUC|.733824|.733824|.733824|**.733824**|
|Verdict counts|58/18/38/67|64/23/32/62|58/18/38/67|**58/20/38/65**|
|Verdict P/R|.7632/.6042|.7356/.6667|.7632/.6042|**.7436/.6042**|
|Verdict F1/FPR|.6744/.2118|.6995/.2706|.6744/.2118|**.6667/.2353**|
|Verdict flagged share|.4199|.4807|.4199|**.4309**|

P1 threshold **.5794392523364486**, raw/calibrated score, scaler/calibrator, answer fields и per-pair decisions прежние. Answer metrics вычислены из actual replay rows, не подставлены из baseline. Verdict logic не менялась; verdict FPR ухудшился относительно P4, но watch≤.30 соблюдён. Все rule evidence включены в основную strict-метрику.

## 6. Parity и остаточные ошибки

- P1:181 pairs,**0 mismatches**, raw/calibrated/threshold max delta **0.0**.
- P4 full risks:**13150 tokens, drift0.0**, finite; AUC поэтому прежний, риски не заменялись filtered scores или нулями.
- Reload/pure builder+filter/public spans+flags: **0 mismatches**. `span.text == answer[start:end]` для всех.
- Direct raw JSON audit: P4 bundle и P5 bundle **идентичны во всех полях кроме нового span_filter**, а не только похожи по итоговым метрикам.
- Baseline parity P2/P3/P4: matched true, mismatches[] (`baseline_parity.json`).

Даже после dev gate pass ошибки не скрыты: FP spans32 =**22 clean +10 dirty**;29 не пересекают gold,3 имеют overlap ниже IoU.50FN:10 без generator/final overlap,37 с кандидатом до keep/drop, но без final overlap,3 с overlap ниже strict IoU. До фильтра generator имеет any coverage85/96,strict71/96;11 gold без generator candidate (один из них имеет лишь частичный rule overlap, поэтому эти11 нельзя прибавлять к37 и3 как disjoint FN).

Token FP134 =92 clean +42 dirty;FN95 =70 edge +25 interior. Kept-mask-only strict46/30/50,F1=.534884; unchanged rules-only2/6/94; final **46/32/50,F1=.528736**. Для gate используется именно final со всеми rules, не более красивая mask-only таблица. Дополнительный post-hoc audit делает только saved-state replay, без нового fit/search/HF inference.

## 7. Integrity, файлы и размеры

Frozen source P4 combined120455B SHA256 `e6c96ea4390dbf124540c413594110180f84d9be9d5c13ad1577232b70ee9e48`;
P4 winner OOF gzip618264B `77a4e10edf171a84b2ef8f9be5fe93249f551e2afdc0770049e5960c379b6c8f`;
P4 dev gzip575319B `0133c1607baa666c7296361ee2ecdb5baaf417eb0f3d5369601e3d8d76dd81e5`.

P2 references после распаковки: OOF5742956B SHA256 `5ca0ff5121707e950a4a710c8d733c092aeca477857ca146847c5de0ef98959b`;dev3932617B `e54d290b4894246e61a256af5568af500bc979997d7ab72e7f6ea237129d31ce`. Весь published P4 inventory повторно совпал; эталонный control max delta0.0 принят из прежнего receipt, не переобучался.

Новый combined bundle **122874B**, SHA256 **`e0b13c693a058dfd66ed24a52fc14e9c70964e95885358e7b99cc1f328ad6892`**. Дополнение к исходному bundle2419B не является размером всей HF поставки.

Основные файлы:
- `candidate_pool.jsonl.gz`17078199B — полный cell-specific pool с отдельными labels/19 features;
- `oof_span_probabilities.jsonl.gz`2084233B;
- `grid_scores.jsonl.gz`299824B;
- `candidate_diagnostics.json`415409B;
- `span_feature_spec.json`12641B;
- `dev_evidence.jsonl.gz`571592B.

Полные source hashes,8 cache shard hashes и versions — `manifest.json`; raw source/candidate alignment/doc-fold audits — `provenance_audit.json`; все опубликованные hashes/sizes — `published_files_sha256.json`. Первоначальный runner inventory совпал без расхождений и не переписывался.

## 8. Runtime, тесты и CI отдельно

`spanverify/span_filter.py` содержит одну pure candidate-builder и одну pure filter/merge функцию, общий offline/public путь. Никакого numpy/sklearn в новом runtime-модуле. Версия `span_filter_schema_v1`, strict field/vector/NaN validation; legacy без поля сохраняет старое поведение. Empty kept set авторитетен, rules остаются, fallback к P4 tau отсутствует. Bridge2 реализован только в P5: старый P3 validator не ослаблялся.

Workflow `.github/workflows/p5-span-filter.yml`: явный `.github/p5-span.trigger` +dispatch,35min measurement-step timeout с резервом для always upload/branch publication. P2/P3/P4 explicit triggers и обычный CI не ослаблялись.

- **108 targeted passed**, включая21 P5 case: short/one/no eligible,clamp/gaps,finite/deterministic features,forbidden clean/gold inputs,document fold guard,empty/rules,malformed schemas,roundtrip,legacy,P1/risk/pure-public parity.
- **Полный локальный suite:842 passed /22 failed /2 skipped,866 nodes,302.25s (0:05:02).** Новых failing nodes0,resolved0 относительно P4. Дополнительные workflow-parametrized checks также выполнены. Ни один тест не исключался. Exact traceback: `full_suite_traceback.log`; JUnit:`full_suite_junit.xml`; summary:`test_status.json`.
- Исторические22 группы не объявляются исправленными: calibration/sentinel;decimal coverage;registry/provenance;cache max_length;stale METRICS/docs;verdict fixtures;unsupported --no-test;attribution/number words/CLI/selftest;два прежних60min workflow limits.
- Ruff/Black затронутых runtime/script/tests — passed из `spanverify-module`; documentary scripts — passed из корня с module config. Это не весь исторический repository lint.
- Обычный [CI39170981159](https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/39170981159), code977e9dc — **FAILED**: global Ruff,test jobs и contract job exit1. Подробные реальные job/annotation snapshots — `ordinary_ci_status.json`. Точные866/22 counts выше — локальный полный прогон, не числа, выдуманные по CI annotations. P5 measurement success не называется green full CI.

## 9. Матрица договора

Обновлены [coverage-matrix.json](reports/contract/coverage-matrix.json) и [coverage-matrix.md](reports/contract/coverage-matrix.md): dev token и strict span gates **достигнуты**; контрактный quality evidence **частичный**, так как official test177 закрыт. Полная готовность договора **НЕ ДОСТИГНУТА**: скорость .03s/1000,50k≤1s/2M≤60s,текущие CLI/web/HTTP acceptance,exe≤9MB/package≤.6MB и SHA поставки не измерялись этим experiment. Исторические demo/release не подменяют текущую P5 HF приёмку.
