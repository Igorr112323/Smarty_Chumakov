# P3: структурная локализация на замороженных P1/P2

## Итог: НЕ ДОСТИГНУТО

[Реальный run 39160817890](https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/39160817890) завершён success: расчёт и проверки выполнены. Это **не** PASSED quality gate.

- Measurement code: `36d92c5524c96a7ea7e05ce8490aff94e507d69a`.
- Runner publication: `b406a825db0db6e4658976eb6b0dc90c8e12d15e`.
- Последующая чистка стиля: `3601b9966f4091ce0b8fa39de5d7f8643b467413`, без смены decoder/config. После неё pure replay по всем 181 сохранённым dev-state: 0 mismatches flags/spans, риски не мутируются.
- [Опубликованный каталог измерения](https://github.com/Igorr112323/Smarty_Chumakov/tree/b406a82/spanverify-module/reports/experiments/p3_postprocess/run39160817890).
- [Полный artifact](https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/39160817890/artifacts/12149230019): `p3-postprocess-39160817890`, **2 684 039 bytes** compressed.

Перед работой remote HEAD сверен с `cff6f62425a739475c65248d204bd804009db6b0`. Восстановленный workspace сохранён в stash перед fast-forward; четыре изменённых tracked-файла совпали с remote. История не удалялась.

## Протокол и отсутствие переобучения

- Использованы исходные train-OOF states из run39155368906/artifact12146627081; скачивание выполнено runner, не sandbox.
- 842 уникальных train ID, 265 документов, 5 source folds, 46607 scored tokens. Corpus content, doc/fold membership, receipt SHA256, offsets, scored mask, gold и finite full risks проверены; missing states=0, shared_documents=0, каждый held document/pair ровно один раз.
- Classifier fits в measurement: **0**. HF/precompute/API/GPU/backbone/P1/P2 training/official test не запускались. Unit parity fixtures в preflight не являются обучением quality-кандидата.
- Не менялись tokenization/scored predicate, восемь token features, smoothing, model/window512/chunking.
- Только после сохранения победителя читались dev corpus и reference evidence.
- Frozen HF cache source39060670054: восемь distinct shards, 1195 keys, **1267 hits / 0 misses**.
- Measurement duration **36.950 s** (не benchmark продуктовой скорости; setup/download не включены). Python 3.11.17, numpy1.26.4. Стек preflight: scipy1.13.1, sklearn1.5.2, pytest8.3.5, PyYAML6.0.2.
- P2-training push trigger теперь только `.github/p2-training.trigger` + workflow_dispatch; новый P3 workflow после единственного измерения также запускается явным `.github/p3-postprocess.trigger` или dispatch. Обычный CI не отключён. Нового P2 run нет.

## 1. OOF-диагностика причин низкого strict span F1

Все числа ниже — **train OOF**, до выбора P3, исторический P2 tau=0.5653513416423852. Gold не расширялись, greedy strict IoU≥0.5, 467 gold spans.

| Источник | TP | FP | FN | F1 | FP clean | FP dirty | No overlap | Overlap, IoU<0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mask only | 194 | 1021 | 273 | 0.230678 | 495 | 526 | 913 | 108 |
| Attribution, clause output | 4 | 53 | 463 | 0.015267 | 11 | 42 | 17 | 35 |
| Coverage | 0 | 0 | 467 | 0 | 0 | 0 | 0 | 0 |
| Final merged P2 | 190 | 1052 | 277 | 0.222352 | 505 | 547 | 921 | 131 |
| Exact attribution only, diagnostic | 6 | 51 | 461 | 0.022901 | 11 | 40 | 21 | 30 |
| Same P2 mask + exact anchors, diagnostic | 199 | 1057 | 268 | 0.230992 | 505 | 552 | 931 | 126 |

Coverage оставлена с прежними условиями: в frozen baseline она выключена, поэтому её фактический output равен нулю; P3 её не включала/не отключала по dev.

**Интерпретация:**

- Главная масса mask FP не перекрывает gold вообще: 913 из 1021. Одной коррекции ширины недостаточно.
- 54 gold пересекаются с несколькими mask predictions; после final merge — 53. Это диагностика fragmentation, не дополнительные TP.
- **5 корректных mask-spans теряют match после final merge**; список конкретных случаев сохранён в `merge_lost_mask_spans.json`. Поэтому расширение attribution до clause действительно вредит, но оно не единственная/главная причина всех FP.
- Число attribution outputs осталось 57; медиана ширины clause output **70** символов, exact anchor **2** (max211→7). Условие number_attribution не менялось.
- Только exact anchors при прежней маске дают joint/span F1 около0.231, а не0.50. Это диагностическая абляция, не новый победитель и не dev-подбор.
- Mask widths: median9, p90=26, max54 символа; token-components median2, p95=4. Неверно говорить, что все маски широкие.
- Final merged P2 widths: median10, p95=32, max211. OOF mean_width_ratio=9.023 считается только по полностью накрытым gold. Dev14.210 также не является средней шириной каждого span.
- Scored risks внутри gold: median0.59124, p25=0.49663; снаружи: median0.11406, p99=0.68694. Есть и полезное ранжирование, и пересечение хвостов. Это не гарантия высокого F1.
- Полные распределения ширин, числа компонентов, token gaps и рисков: `oof_diagnostics.json`; pair-level evidence: `oof_diagnostic_evidence.jsonl`.

## 2. Train-only выбор decoder config

Один заранее заданный recipe: seed/grow components + exact attribution anchors, coverage без изменений. 22 заданных квантиля (`method=linear`), прежний tau, границы всё/ничего → 25 numeric cutoffs, **1300** конфигураций low≤high × bridge{0,1} × min_seeds{1,2}.

```json
{
  "type": "seed_grow_v1",
  "schema_version": 1,
  "tau_low": 0.382602175322462,
  "tau_high": 0.7038362509143306,
  "bridge_unscored_tokens": 1,
  "min_seed_tokens": 1,
  "attribution_boundary": "exact"
}
```

Low соответствует train risk quantile0.85, high —0.9825. Не уточнялись по dev.

Критерий: max min(token F1, strict span F1), token FPR≤0.30; tie-break меньший FPR; оставшиеся точные ties — фиксированный порядок обхода.

Selected OOF: token P/R/F1/FPR = **0.437536 / 0.544075 / 0.485024 / 0.021405**; TP/FP/FN/TN=753/968/631/44255. Strict spans TP/FP/FN=158/251/309, F1=**0.360731**. Это обучающие selection-метрики postprocessor, не независимая приёмка.

Numeric grid, все scores, criterion/winner сохранены в `numeric_candidates.json`, `grid_scores.jsonl.gz`, `selected_decoder.json`. Никаких повторных HF/tree проходов в grid нет.

## 3. Реальный публичный dev replay: P2 → P3

Все 181 пары, один frozen winner. Baseline P2 измерена заново и воспроизводит source metrics. Все mask/rule spans включены в главную strict метрику.

### Tokens

| Метрика | P2 | P3 |
|---|---:|---:|
| Precision | 0.348249027 | 0.383720930 |
| Recall | 0.623693380 | 0.574912892 |
| F1 | 0.446941323 | **0.460251046** |
| FPR | 0.043608435 | **0.034496225** |
| AUC full risk | 0.946304634 | **0.946304634** |
| TP/FP/FN/TN | 179/335/108/7347 | **165/265/122/7417** |

### Strict spans, IoU≥0.5

| Метрика | P2 | P3 |
|---|---:|---:|
| Precision | 0.162500000 | 0.346938776 |
| Recall | 0.406250000 | 0.354166667 |
| F1 | 0.232142857 | **0.350515464** |
| TP/FP/FN | 39/201/57 | **34/64/62** |
| Containment | 17/96 =0.177083333 | **13/96 =0.135416667** |
| Mean width ratio, contained gold only | 14.209685980 | **5.239085239** |
| FPR token-proxy | 0.043608435 | **0.034496225** |

У spans нет самостоятельно определённой TN-вселенной. FPR не подменён отношением FP/predictions: используется прежний token proxy.

### Answers и rule-based verdict — разные решения

| Семейство | P2 P/R/F1/FPR | P3 P/R/F1/FPR | P2→P3 TP/FP/FN/TN |
|---|---|---|---|
| Answer score-threshold decision | 0.777777778 / 0.510416667 / 0.616352201 / 0.164705882 | **без изменений** | **49/14/47/71 → 49/14/47/71** |
| Rule-based verdict | 0.7632 / 0.6042 / 0.6744 / 0.2118 | **0.7356 / 0.6667 / 0.6995 / 0.2706** | **58/18/38/67 → 64/23/32/62** |

Answer threshold=0.5794392523364486. Calibrated answer AUC=0.7338235294117647 без изменений. Вердикт улучшил recall/F1, но ухудшил precision/FPR — это не скрыто.

**Договорные token/span F1≥0.50 и FPR≤0.40 совместно НЕ достигнуты:** FPR проходит, оба F1 ниже0.50. Рабочий запас≥0.60/≤0.30 также не достигнут. Ни официальный test, ни поставка, ни скорость, ни весь договор этим dev-run не закрыты.

## 4. Инвариантность, единый runtime и fail-closed

- P1 raw/calibrated score, threshold, decisions на **181 ID** против source P1 и source P2: **0 mismatches, max score delta0.0**. Baseline replay и reload также совпали.
- P2 full smoothed raw risks: все **13 150 tokens**, **0 mismatches, max delta0.0** против original dev_evidence. AUC не повышалась обнулением рисков.
- Reload: tokens/spans/verdict/scores одинаковы; pure-state decoder == public Verifier на181 парах, 0 mismatches. `span.text == answer[start:end]` проверено для каждого публичного span.
- Pure `localize_state()` вызывается selection и public `_localize`. Low-risk scored gap не перескакивается; unscored gaps не получают labels; retained set не пересекается со старым tau. Empty set — авторитетный результат, не fallback.
- P3 hard-share использует ровно retained scored mask. Full risks и FeatureMatrix не мутируются.
- Merge обновляет offsets/text, token indices/counts и sources. Attribution условие и confidence сохранены, меняются только границы; исходное объяснение number_attribution сохраняется в stats.
- P3 config — отдельное `localization_decoder` в bundle/runtime. Неизвестная schema/type, NaN/inf и malformed config дают ValueError. Legacy bundle без P3 сохраняет прежний global/adaptive путь.
- `frozen_fields_parity.json`: все поля исходного combined bundle идентичны, единственное добавленное поле — localization_decoder.

## 5. Реальные файлы и hashes

Весь каталог, включая source gzip и новые dev evidence, опубликован **в working branch**, не только в Azure artifact.

| Файл | Bytes |
|---|---:|
| source_oof_states.jsonl.gz | 1 379 585 |
| source_dev_evidence.jsonl.gz | 570 090 |
| dev_evidence.jsonl.gz (P3) | 572 683 |
| oof_diagnostic_evidence.jsonl | 239 949 |
| grid_scores.jsonl.gz | 102 461 |
| p2_token_head.json (неизменённый) | 97 494 |
| combined_bundle.json (P3) | 115 319 |
| runtime_config.json | 618 |

Frozen P2 model SHA256: `33112af1997e12afe9aee07a6f84eb29760c9165e1a928938b2979110d822e9e`.
Frozen source combined SHA256: `87298b01ec157b20c6a06cee9244d75ce1d7882a30f20abddbb6e9781bbf46ab`.
P3 combined SHA256: `bc343e3fcfbfb4ee7056f5f50b016a657ade0a080d8c99b6392d1f99a3ad1e13`.

Распакованные source gzip побайтно проверены повторно локально:
- OOF: 5 742 956 bytes, SHA256 `5ca0ff5121707e950a4a710c8d733c092aeca477857ca146847c5de0ef98959b`.
- Dev source: 3 932 617 bytes, SHA256 `e54d290b4894246e61a256af5568af500bc979997d7ab72e7f6ea237129d31ce`.

`source_provenance.json` содержит source run/artifact/code SHA и raw/gzip hashes. `manifest.json` — actual run/SHA, frozen hashes, versions, cache counts, timings, sizes. `measurement.log` содержит stdout/stderr. Размер лога в manifest — снимок до финального print; финальные sizes/hashes отдельно в `publication_files.json`.

## 6. Тесты и общий CI — отдельный статус

- **73 target tests passed** локально и на runner; из них17 новых P3 cases. Проверяются growth/support, weak rejection, unscored/scored gaps, empty set, punctuation/number, exact anchor/merge, pure/public, P1/risk invariance, roundtrip, malformed schema/NaN.
- Ruff и Black по всем затронутым Python-файлам проходят после чистки стиля. Она не меняет выбранный config/математику. Дополнительный pure replay по181 source states после форматирования:0 mismatches.
- Полный suite действительно выполнен без исключений: **22 failures**, набор failing nodes совпал с P2, **новых failing nodes нет**. Полные traceback в `local_full_suite.log`, сравнение в `test_status.json`.
- Общий GitHub CI **не зелёный**: code runs39160821143/39160817770 при снимке ещё выполняли test matrix; lint и contract jobs уже имели failure. Runner-publication CI39160894265 имеет action_required. Статусы не заменены успехом P3 measurement.

### Разбор 22 существующих failures по фактическим traceback

| Группа | Что именно ломается |
|---|---|
| Calibration/context train | Ожидание cross-validation F1>0.8 при actual0.6; тест запрещает answer threshold1.000000001 (sentinel). P1 для этого не ретюнилась. |
| Number/report checks | Regex не распознаёт `покрытие 58.89 %`, и проверка неверного coverage не находит проблему. |
| Registries/provenance | CI registry содержит12 workflows при20; METRICS606 при821 collected; документы заявляют606; CONTRACT provenance=None. |
| Cache fixtures | Два теста передают dump_matrix(max_length=...), но helper не принимает этот keyword. |
| no_test CLI | Два теста ожидают --no-test, существующий run_experiments parser возвращает SystemExit2: unrecognized arguments. P3 этим CLI не пользуется и official test не запускала. |
| Demo/verdict/CLI expectations | Метрики verdict TN0 вместо1; number-attribution/number-words ожидают grounded, получают likely_hallucination; CLI exit1/2 вместо0; один verdict fixture имеет score=threshold=1; два rule modes не различаются на нём. Логика classifiers/number_attribution не менялась ради этих tests. |
| Workflow limits | Два исторических postprocess workflow задают60 минут при лимите40. Новый P3 имеет40. |

Проверки не удалялись и failures не скрывались. Новые P3 style regressions исправлены; исправление всей старой несвязанной ветки не подменяет отрицательный результат локализации.
