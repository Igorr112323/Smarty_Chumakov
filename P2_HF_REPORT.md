# P2: реальный cached-HF эксперимент

## Результат: НЕ ДОСТИГНУТО

[Run 39155368906](https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/39155368906) — completed / success (расчёт выполнен; это НЕ означает прохождение quality gates).

- Code SHA: `7e0ed4454180c9c1ff0bf39ffbec5b993e41ade5`.
- Публикация runner: `444ff0c`; [каталог результатов](https://github.com/Igorr112323/Smarty_Chumakov/tree/444ff0c/spanverify-module/reports/experiments/p2_hf/run39155368906).
- Исходный HEAD перед работой сверен с remote: `6fc975fc929388df12d2bdb609d635843519137c`. Восстановленный workspace сохранён в git stash перед fast-forward; уникальных изменений относительно remote не оказалось.
- Python 3.11.17; numpy 1.26.4; sklearn 1.5.2; scipy 1.13.1.
- HF cache source 39060670054, восемь distinct shards, 1195 ключей. 1928 hits / 0 misses.
- Backbone `ai-forever/rugpt3small_based_on_gpt2`, window512, chunking=true. Никакого fresh inference/demo/official test/P1 training.
- 842 train, 181 dev; финальный fit: 46607 scored tokens (1384 positive / 45223 negative).
- 5 document folds, seed42; реальные doc sets, SHA256 и model hashes в fold_receipts.json. shared_documents=0 в каждом фолде.
- GBDT: log_loss, 80 trees, lr=.05, depth3, min_leaf20, subsample1, init=zero, seed42; balanced weights только по fit-части.
- Длительность measurement: 141.205 s (без setup/download).

## Dev: штатный Verifier.evaluate(), все семейства

| Семейство | Baseline F1 | Trained P2 F1 | Baseline FPR | P2 FPR | Baseline → P2 TP/FP/FN/TN |
|---|---:|---:|---:|---:|---|
| tokens | 0.260131951 | 0.446941323 | 0.082790940 | 0.043608435 | 138/636/149/7046 → 179/335/108/7347 |
| spans | 0.179775281 | 0.232142857 | 0.082790940 | 0.043608435 | 40/309/56/— → 39/201/57/— |
| answers | 0.616352201 | 0.616352201 | 0.164705882 | 0.164705882 | 49/14/47/71 → 49/14/47/71 |
| verdicts | 0.711300000 | 0.674400000 | 0.682400000 | 0.211800000 | 85/58/11/27 → 58/18/38/67 |

Strict spans: IoU≥0.5, gold не расширялись, включён весь публичный вывод (mask + attribution + coverage). У span-FPR нет TN-вселенной; указан штатный FPR_tokens, не выдуманный span TN.

Token precision/recall: baseline 0.178294574 / 0.480836237 → P2 0.348249027 / 0.623693380. Token AUC по полному smoothed raw_risk: 0.848850700 → 0.946304634.

Answer AUC: raw 0.7321078431372549, calibrated 0.7338235294117647. Answer score-threshold decision и rule-based verdict — разные строки таблицы.

**Совместные договорные token/span F1≥0.50 и FPR≤0.40 НЕ достигнуты:** FPR проходит, оба F1 не проходят. Запас ≥0.60/≤0.30 также не достигнут. Dev не использовался для ретюнинга.

## Global tau и P1 invariance

- Frozen tau = **0.5653513416423852**. Выбран только на общем train document-OOF.
- 500 label-free quantile points + две границы = 502 distinct candidates.
- Критерий max min(token F1, strict span F1), token FPR≤0.30, tie-break меньший FPR.
- Выбранные OOF selection metrics: token F1=0.4020969855832241, FPR=0.036795435950733035; strict span F1=0.2223522527794032. Это обучающие метрики cutoff, не независимая приёмка.
- Одинаковые shared extractor, full-sequence zero-mask/smoothing и _localize используются runtime/OOF; поиск переиспользует готовые smoothed risks, без новых HF/tree проходов.
- Runtime tau не проходит через legacy cap; stats показывает полный использованный float; legacy bundle сохраняет adaptive mask.
- Combined bundle построен dataclasses.replace от полного frozen P1; isotonic, scaler/schema, answer head, answer-threshold 0.5794392523364486 сохранены.
- Все 181 dev ID: raw/calibrated/threshold max delta **0.0**, prediction exact, **0 mismatches**.
- Reload: те же 181 ID, max delta **0.0**, **0 mismatches**. Tokens/spans также сравнены после reload; каждый span.text проверен по исходному slice.
- Answer TP/FP/FN/TN **49/14/47/71**.

## Артефакты

[Полный artifact p2-hf-39155368906](https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/39155368906/artifacts/12146627081): **2 035 693 bytes** (compressed). Включает OOF states и dev evidence; большие JSONL не внесены в git.

| Файл | Размер, bytes | Где |
|---|---:|---|
| p2_token_head.json | 97494 | [p2_token_head.json](https://github.com/Igorr112323/Smarty_Chumakov/tree/444ff0c/spanverify-module/reports/experiments/p2_hf/run39155368906/p2_token_head.json) |
| combined_bundle.json | 115068 | [combined_bundle.json](https://github.com/Igorr112323/Smarty_Chumakov/tree/444ff0c/spanverify-module/reports/experiments/p2_hf/run39155368906/combined_bundle.json) |
| runtime_config.json | 367 | [runtime_config.json](https://github.com/Igorr112323/Smarty_Chumakov/tree/444ff0c/spanverify-module/reports/experiments/p2_hf/run39155368906/runtime_config.json) |
| oof_states.jsonl | 5742956 | artifact |
| dev_evidence.jsonl | 3932617 | artifact |
| dev_answers.jsonl | 34717 | [dev_answers.jsonl](https://github.com/Igorr112323/Smarty_Chumakov/tree/444ff0c/spanverify-module/reports/experiments/p2_hf/run39155368906/dev_answers.jsonl) |
| fold_receipts.json | 40562 | [fold_receipts.json](https://github.com/Igorr112323/Smarty_Chumakov/tree/444ff0c/spanverify-module/reports/experiments/p2_hf/run39155368906/fold_receipts.json) |
| tau_selection.json | 301453 | [tau_selection.json](https://github.com/Igorr112323/Smarty_Chumakov/tree/444ff0c/spanverify-module/reports/experiments/p2_hf/run39155368906/tau_selection.json) |
| all_family_metrics.json | 6962 | [all_family_metrics.json](https://github.com/Igorr112323/Smarty_Chumakov/tree/444ff0c/spanverify-module/reports/experiments/p2_hf/run39155368906/all_family_metrics.json) |
| p1_invariance.json | 379 | [p1_invariance.json](https://github.com/Igorr112323/Smarty_Chumakov/tree/444ff0c/spanverify-module/reports/experiments/p2_hf/run39155368906/p1_invariance.json) |

[Manifest](https://github.com/Igorr112323/Smarty_Chumakov/tree/444ff0c/spanverify-module/reports/experiments/p2_hf/run39155368906/manifest.json), [stdout/stderr log](https://github.com/Igorr112323/Smarty_Chumakov/tree/444ff0c/spanverify-module/reports/experiments/p2_hf/run39155368906/measurement.log), [runner preflight](https://github.com/Igorr112323/Smarty_Chumakov/tree/444ff0c/spanverify-module/reports/experiments/p2_hf/run39155368906/tests.log).

## Тестовый статус

- Targeted preflight: **56 passed**, локально и на runner; actual tau low/high flags/spans, full bundle/isotonic roundtrip, document-filtered fit call, sklearn/export/runtime parity, public influence, backward compatibility.
- Полный tests/ действительно запущен без исключений: **22 failures**. Не объявляется зелёным. Полный traceback — local_full_suite.log, перечень — local_test_status.json в этом каталоге.
- PyYAML 6.0.2 установлен; прежний import YAML blocker не воспроизводится. Конкретные workflow failures сейчас: два исторических postprocess workflow имеют timeout 60 при лимите40; новый P2 timeout=40. Остальные failures включают calibration expectation, number attribution/coverage, cache roundtrip, устаревшие CI/metrics registries, no_test isolation, demo CLI/verdict expectations. Ради зелёного теста модель/калибровка/guards не изменялись.
- Исторический run1 явно diagnostic_only / quality_run=false, mode=demo, handwritten placeholder, fitting_pairs=0, checked_dev_pairs=10, influence_pairs=1. Его история сохранена. Проценты «готовности» из старого отчёта не используются как измеренная приёмка.
