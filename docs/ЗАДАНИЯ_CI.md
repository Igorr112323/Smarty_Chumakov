# Задания CI: что запускается само, а что вручную

Документ собран скриптом `scripts/collect_ci_workflows.py` из файлов
`.github/workflows/*.yml` (что задание умеет) и `reports/CI_RUNS.json`
(что оно делало в последний раз). Числа и номера прогонов не вписаны
руками: их источник — реестр прогонов.

Команда: `python scripts/collect_ci_workflows.py` (запуск: 2026-10-19T00:19:52Z).

| Задание | Запуск | Последний прогон | Артефакты | Числа в реестре |
| --- | --- | --- | --- | --- |
| `build-and-release-exe` | вручную, на пуш; пуш `.github/exe-build.trigger`; теги `v*`; ветки `main`, `arena/**` | успех (прогон 38481064209) | да | reports/release_sizes.json (spanverify.exe, spanverify.pyz) |
| `ci` | на пуш, на pull request, вручную, по расписанию (30 2 * * * UTC); ветки `main`, `arena/**` | успех (прогон 38489996351) | да | reports/METRICS.json, reports/contract/*.json |
| `Корпус A3 (реальные акты, срезы)` | вручную, на пуш; пуш `.github/corpus-a3.trigger`; ветки `arena/**` | прогонов нет | да | — |
| `Разбор ложных пометок по кешу` | вручную, на пуш; пуш `.github/hf-analysis.trigger`; ветки `arena/**` | прогонов нет | да | — |
| `Свести числа экспериментов hf` | на пуш; пуш `.github/hf-collect.trigger`; ветки `arena/**` | прогонов нет | да | — |
| `Эксперименты hf (корпуса A3 и A)` | вручную, на пуш; пуш `.github/hf-experiments.trigger`; ветки `arena/**` | прогонов нет | да | — |
| `hf-footprint` | вручную, на пуш; пуш `.github/hf-footprint.trigger`; ветки `arena/**` | успех (прогон 38438650943) | да | reports/contract/4.2.5-hf-footprint.json, reports/contract/4.2.6-storage.json |
| `Пилот hf (признаки, реальная модель)` | вручную, на пуш; пуш `.github/hf-pilot.trigger`; ветки `arena/**` | прогонов нет | да | — |
| `Пересчёт hf на кеше признаков` | на пуш; пуш `.github/hf-remeasure.trigger`; ветки `arena/**` | прогонов нет | да | — |
| `Тяжёлые прогоны hf` | вручную, на пуш; пуш `.github/hf-runs.trigger`; ветки `arena/**` | прогонов нет | да | — |
| `Забрать числа hf` | вручную, на пуш; пуш `.github/hf-salvage.trigger`; ветки `arena/**` | прогонов нет | нет | — |
| `Маска: сравнение варианта с фильтром коротких фрагментов` | вручную, на пуш; пуш `.github/hf-span-filter.trigger`; ветки `arena/**` | прогонов нет | да | — |
| `P1 Build Test` | на пуш; пуш `.github/workflows/p1-build-test.yml`, `spanverify-module/scripts/build_p1_candidate.py`, `test_p1_build.py`; ветки `arena/c5f69767-smarty-chumakov` | прогонов нет | нет | — |
| `P1 Public Replay` | на пуш, вручную; пуш `.github/workflows/p1-public-replay.yml`, `spanverify-module/scripts/build_p1_candidate.py`, `spanverify-module/scripts/replay_p1_dev.py`, `spanverify-module/tests/test_p1_replay_contract.py`; ветки `arena/c5f69767-smarty-chumakov` | прогонов нет | да | — |
| `P2 cached HF experiment` | на пуш, вручную; пуш `.github/p2-training.trigger`; ветки `arena/c5f69767-smarty-chumakov` | прогонов нет | да | — |
| `P3 frozen-model structural localization` | на пуш, вручную; пуш `.github/p3-postprocess.trigger`; ветки `arena/c5f69767-smarty-chumakov` | прогонов нет | да | — |
| `P4 context token head` | на пуш, вручную; пуш `.github/p4-context.trigger`; ветки `arena/c5f69767-smarty-chumakov` | прогонов нет | да | — |
| `P5 span keep/drop filter` | на пуш, вручную; пуш `.github/p5-span.trigger`; ветки `arena/c5f69767-smarty-chumakov` | прогонов нет | да | — |
| `P6 frozen product acceptance` | вручную, на пуш; пуш `.github/p6-acceptance.trigger`; ветки `arena/c5f69767-smarty-chumakov` | прогонов нет | да | — |
| `P0-3 P1 Answer Head` | на пуш, вручную; пуш `spanverify-module/spanverify/answer_head.py`, `spanverify-module/spanverify/oof.py`, `spanverify-module/scripts/measure_p1.py`, `spanverify-module/tests/test_answer_head.py`, `.github/workflows/postprocess-p0-answer-p1.yml`; ветки `arena/c5f69767-smarty-chumakov` | прогонов нет | да | — |
| `P0-2 Ответы` | на пуш, вручную; пуш `.github/postprocess-p0-answer.trigger`; ветки `arena/c5f69767-smarty-chumakov` | прогонов нет | да | — |
| `Постобработка span_expansion` | на пуш, вручную; пуш `.github/postprocess-span-comparison.trigger`; ветки `arena/c5f69767-smarty-chumakov` | прогонов нет | да | — |
| `Export existing P1 raw files` | на пуш, вручную; пуш `.github/workflows/recover-p1-results.yml`; ветки `arena/c5f69767-smarty-chumakov` | прогонов нет | да | — |

Источник колонки «Последний прогон» — `reports/CI_RUNS.json`: последние 100 прогонов репозитория. Поэтому «прогонов нет» означает «нет в этом списке», а не «задание никогда не запускалось»; пересобрать список — `python scripts/collect_ci_runs.py --limit 100`.

## Что это значит для приёмки

Обязательными считаются задания, которые запускаются на пуш в `main` и в
зону разработки: основной CI (тесты, линтер, контрактное задание, сверка
чисел) и сборка `.exe`. Замер объёмов режима `hf` запускается пушем
файла-триггера: он отвечает требованиям ТЗ п. 4.2 треб. 5 и 6, а его числа
попадают в реестр `reports/NUMBERS.json` тем же прогоном.

Задания, которые не запускаются сами (их триггер — только правка
файла-триггера или ручной запуск), помечены в колонке «Запуск»: это не
забытые задания, а измерения, которые делают один раз по событию. Ни одно
из них не выдаёт числа «на глаз»: результат каждого — артефакт с числом и
записью в реестре, либо честное «прогонов нет».

Ручной запуск из командной строки в этой работе недоступен (интеграция
GitHub не имеет права `Actions: write`, ответ API — 403), поэтому
одноразовые задания запускаются пушем файла-триггера вида
`.github/<имя>.trigger`; это записано в комментариях к каждому workflow.
