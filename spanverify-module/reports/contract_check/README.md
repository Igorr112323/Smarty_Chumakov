# Каталог результатов сверки с договором 117812 (версия 19)

## Что здесь и зачем

Это отчёт независимой проверки (роль «инженер-проверяющий») о том, что из обязательств
Приложения 3 и Приложения 4 к договору действительно лежит в коде и в артефактах сборки `main`
на коммите `d39c95c`, а что отсутствует. Проверка ничего не докручивает и ничего не чинит:
пороги 0,50 / 0,40, IoU 0,5 и FPR 0,10 не изменялись, гиперпараметры не подбирались, текст
договора не правился. Где договор и код расходятся — записано в разделе «Расхождения» основного
файла, решение по каждому пункту остаётся за человеком.

## Из какого состояния собраны числа

`main` = `d39c95c` (Merge pull request #2 from `arena/316d8fab-smarty-chumakov`, 2026-10-07 18:33 +0300).
Проверка выполнена по снапшоту `git -C /home/user/Smarty_Chumakov archive d39c95c | tar -x -C /home/user/main_d39c95c`
(2026-10-12), то есть по дереву ровно этого коммита, без изменений. Боевое дерево и `origin` не тронуты;
отчёты закоммичены в ветку сессии `arena/cd51890b-smarty-chumakov` (так требует окружение Arena),
код — отдельным коммитом `3d5dc87`.

Машина измерения: Linux, Python 3.11.2, 2 ядра Intel Xeon @2,60GHz, 3,9 ГБ ОЗУ, GPU нет,
`torch`/`transformers`/`streamlit` не установлены. Поэтому числа режима `hf` взяты из закоммиченных
артефактов CI-прогонов, а не пересчитаны здесь: пересечь их живым прогоном в этой среде невозможно.

## Состав

| Файл | Как получен |
|---|---|
| `CONTRACT_CHECK.md` | разбор этого каталога: отметка по каждой строке (4.1 (1)–(10), 4.2, задачи 1–5, этап 1 п. 1–5, этап 2 п. 1–5, тесты, 1.6 а)–г), раздел 5) + расхождения + сводка |
| `quality_A.json` | `python3 -c "import json; …"` по `spanverify-module/reports/experiments/a/experiment.json` (снапшот `d39c95c`) |
| `quality_A3.json` | то же по `reports/experiments/a3/experiment.json` |
| `latency.json` | `python3 scripts/measure_latency.py --pairs 50 --out reports/contract_check/latency.json` |
| `tests.md` | `python3 -m pytest tests/ -q --tb=line -p no:cacheprovider --junitxml=…` + `parse_tests.py` (парсерJUnit, без правок результатов) |
| `quality_summary.md` | рукописная сводка этих же чисел, при каждом числе — путь к артефакту |
| `resources.md` | разбор измерений памяти и времени: команда, машина, что числа значат и чего они не значат |

## Команды для повтора

```bash
cd spanverify-module
python3 -m pytest tests/ -q --tb=line -p no:cacheprovider
python3 scripts/measure_latency.py --pairs 50 --out reports/contract_check/latency.json
python3 -m spanverify evaluate --dataset data/demo_pairs.jsonl
python3 -m spanverify selftest
python3 scripts/hf_protocol.py verify-manifest \
  --manifest reports/experiments/a3/manifest.json --experiments-dir reports/experiments
python3 scripts/check_numbers.py
bash scripts/build_zipapp.sh        # 40 470 729 Б, release/spanverify.pyz
gh run view 38127357903 --json jobs --jq '.jobs[] | [.name,.conclusion] | @tsv'
gh run view 37141538627 --json jobs --jq '.jobs[] | [.name,.conclusion] | @tsv'
```

## Что подтверждено прогонами (сводко, детали — в `CONTRACT_CHECK.md`)

- Тесты дерева `d39c95c`: 515, 0 падений, 2 пропущено (`tests.md`); `ruff check .` и `black --check .` чистые.
- Качество на реальном корпусе A3: F1 0,2029 и FPR 0,0653 на отложенной части (0,50 не достигнуто);
  устойчивость по типам расхождений отсутствует; вердикт-ложняк на чистых парах FPR 0,775.
- Качество на синтетическом A1: F1 0,6083 при FPR 0,0237 (достигнуто, но корпус не реальный).
- Ресурсы: медиана времени ответа 0,0505 с, p95 1,5203 с, пик RSS 36 312 КиБ на 2 ядрах CPU без GPU;
  в руководстве этих чисел нет.
- Поставка: `.pyz` собран из `d39c95c` (40 470 729 Б, `--version` и `selftest` зелёные); опубликованный
  `.exe` (9 134 317 Б, sha256 `d32ef36b264858df…`) относится к тегу `v1.3.0` → `8e27b66`, который не
  является предком `main`.
- Отметки: 14 ВЫПОЛНЕНО, 15 ЧАСТИЧНО, 8 НЕ ВЫПОЛНЕНО из 37 договорных строк (плюс 7 вспомогательных
  проверок: 5 ВЫПОЛНЕНО, 2 ЧАСТИЧНО).

## Что здесь не проверено (а не «не выполнено»)

- Точное значение покрытия ≥ 85 %: порог подтверждён зелёным шагом CI, самого процента в `d39c95c` нет
  (`coverage.json` не закоммичен, `METRICS.json → tests.coverage_percent = null`); локальный прогон под
  `--cov` за 30 минут не завершился (2 ядра, 515 тестов) и был остановлен — цифра не приводится.
- Прямой пересчёт hf-метрик из `d39c95c`: требует `torch`, `transformers` и весов `rugpt3small`, их нет
  ни в дереве, ни в сети песочницы (`pypi` доступен, huggingface.co — нет).
- Сборка `.exe` на `main`: `gh workflow run build-and-release-exe.yml --ref main -f publish_release=false`
  → HTTP 403, а `gh release upload/delete` тем же правом не даны; артефакты CI не скачиваются (0 байт
  после редиректа на `productionresultssa17.blob.core.windows.net`).
- Скачивание и сверка sha256 опубликованных бинарников: хеши взяты из `SHA256SUMS.txt` релиза и из
  аннотаций job'а CI; сама загрузка в песочницу не прошла (0 байт), поэтому «файл = хеш» здесь не
  подтверждено, подтверждено «хеш артефакта = хеш в SHA256SUMS.txt».
