# REALGEN-R1 — Фаза A: естественные галлюцинации на реальных актах

Дата: 2026-10-26T09:01:57.319363+00:00. Ветка: `arena/233cd01f-smarty-chumakov`. Прогон: `local-no-ci-evidence`.

## Квоты

- Ответов: **400** (минимум 400) — PASS
- Ошибочных: **169 / 400 = 0.4225** (минимум 40%) — PASS
- Remedy: не потребовался: R1 уже даёт ≥40% ошибочных ответов
- Правила разметки изменялись: **нет**. Замороженная трактовка правила (c): `any_window`.
- Раунды: сгенерировано R1=400, R2=400 (R2 — contingency, в заморозку не вошёл).
- Sensitivity: stricter `best_window` → R1 0.3800; только правила (a)+(b) без (c) → 0.2375.

## Распределение

- чистые: 71; ошибочные: 169; неясные: 160
- атомов всего: 4720; неясных атомов: 2041; `atom_unclear_rate = 0.4324`
- спанов: 1373; по правилам: {'a': 402, 'b': 303, 'c': 668}
- длины ответов (символы): min/median/max = 26 / 774 / 1954
- длины спанов (символы): min/median/max = 1 / 4 / 67
- темп генерации (один поток, сумма времён): 526.29 ответов/час; всего 2736.1 с

## Разбиение

- dev: 300 ответов / 300 документов; test_realgen: 100 ответов / 100 документов; overlap=0.
- test_realgen ЗАПЕЧАТАН, доступ только по шифт-коду куратора; фаза B читает только dev.

## SHA-карта (первые 12 символов)

| файл | sha256 | байт |
| --- | --- | --- |
| `reports/science/real/realgen_common.py` | `949662c7dc7a…` | 8121 |
| `reports/science/real/realgen_build_background.py` | `df22f7ead1c9…` | 8313 |
| `reports/science/real/realgen_generate.py` | `1962a760ddbe…` | 10925 |
| `reports/science/real/realgen_publish.py` | `744175965ee7…` | 3634 |
| `reports/science/real/realgen_extract.py` | `278b61950561…` | 4828 |
| `reports/science/real/realgen_label.py` | `061e7f5a67e4…` | 36857 |
| `reports/science/real/REALGEN_BACKGROUND.json` | `c720c549f3fc…` | 589570 |
| `reports/science/real/REALGEN_SELECTION.json` | `db98cc713331…` | 284991 |
| `reports/science/real/REALGEN_TEMPLATES.json` | `7851061136b2…` | 3037 |
| `reports/science/real/REALGEN_DOCS.jsonl` | `809b1375b4f5…` | 7114244 |
| `reports/science/real/REALGEN_ANSWERS.jsonl` | `9a98ae3affc3…` | 2111547 |
| `reports/science/real/REALGEN_ANSWERS_DEV.jsonl` | `cd902ac8c33d…` | 1593048 |
| `reports/science/real/REALGEN_ANSWERS_TEST_REALGEN.jsonl` | `8aac288649bd…` | 518499 |
| `reports/science/real/REALGEN_LABELS.jsonl` | `550b9a162c52…` | 503601 |
| `reports/science/real/REALGEN_SPANS.jsonl` | `e431a1538636…` | 376471 |
| `reports/science/real/REALGEN_SPLITS.json` | `d2d83ca61150…` | 36714 |

## Отклонение по бюджету весов (честно)

- Директива: 443.5 МБ ≤ 550 МБ. Фактический `pytorch_model.bin`: 551290714 байт = 551.291 МБ (dec) / 525.752 МиБ, sha256 `0d2a4a387b0430654bb1f6813436bd31f8e10a06e18ea0e5c41b33120b78458e`.
- Отклонение от бюджета 550 МБ: **1.291 МБ**. Модель research-only в CI, продуктовая пломба P8 не изменена.

## Provenance

- Модель: `ai-forever/rugpt3small_based_on_gpt2`, 8 шаблонов, 2 режима, seed=42 от doc_id.
- Фон: 6510 документов LawCodex/ru@5806f92 минус 1200 test-hosting; overlap=0; `real_test_viewed:false`.
- CI-прогоны: 40288730753; экстракция вручную через check-runs output.text.
