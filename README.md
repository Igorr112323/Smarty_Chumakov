# SpanVerify — локальная проверка фрагментов

[![CI](https://github.com/Igorr112323/Smarty_Chumakov/actions/workflows/ci.yml/badge.svg?branch=arena%2Fc5f69767-smarty-chumakov)](https://github.com/Igorr112323/Smarty_Chumakov/actions/workflows/ci.yml) [![Stable publication](https://github.com/Igorr112323/Smarty_Chumakov/actions/workflows/p7-publish-acceptance.yml/badge.svg?branch=arena%2Fc5f69767-smarty-chumakov)](https://github.com/Igorr112323/Smarty_Chumakov/actions/workflows/p7-publish-acceptance.yml)

**Документ + ответ → риск и неподтверждённые фрагменты.** Проверка выполняется локально; решение о достоверности остаётся за человеком. Проект **117812**, договор № 20478ГУ/2026, [ТЗ — Приложение 3](docs/ДОГОВОР_ИЗВЛЕЧЕНИЕ.md).

## Скачать для Windows

| Файл | Назначение |
|---|---|
| **[spanverify.exe](https://github.com/Igorr112323/Smarty_Chumakov/releases/latest/download/spanverify.exe)** | **8 981 233 байт**, Python не нужен |
| **[spanverify-win.zip](https://github.com/Igorr112323/Smarty_Chumakov/releases/latest/download/spanverify-win.zip)** | Переносимая папка: распаковать и запустить EXE |
| **[spanverify.pyz](https://github.com/Igorr112323/Smarty_Chumakov/releases/latest/download/spanverify.pyz)** | Python **≥ 3.10**: `python spanverify.pyz server --port 8765` |
| **[SHA256SUMS.txt](https://github.com/Igorr112323/Smarty_Chumakov/releases/latest/download/SHA256SUMS.txt)** | Контрольные суммы поставки |

**Без установщика:** сохраните EXE в отдельную папку и запустите. Веб-интерфейс: **http://localhost:8765**; если порт занят, приложение сообщает фактический адрес. Перед запуском выполните `certutil -hashfile spanverify.exe SHA256` и сравните результат со строкой `spanverify.exe` в `SHA256SUMS.txt`.

По умолчанию доступен облегчённый **demo**, а не научный HF-режим. Для HF отдельно нужны **веса 554.3 MB и библиотеки 870.6 MB**: [инструкция внешнего HF-runtime](docs/presentation/HF_RUNTIME.md). Небольшой размер EXE не означает поставку модели внутри него.

[Руководство пользователя](docs/РУКОВОДСТВО.md) · [Стабильный релиз](https://github.com/Igorr112323/Smarty_Chumakov/releases/tag/p8-stable-20261019) · [Паспорт приёмки](reports/ACCEPTANCE_PASSPORT.md)

## Что получает пользователь

- Общий **риск 0–1** и вердикт относительно документа-источника.
- Список фрагментов ответа: символьные координаты `start`, `end` (срез `[start:end]`), `label`, `confidence` **0–1** и текст. Это координаты ответа, не документа.
- **Пять признаков на токен:** `attention_entropy`, `ctx_attention_mass`, `ctx_attention_mass_norm`, `ctx_max_similarity`, `embedding_density`.
- **AI-share 0–1** и отдельную экспериментальную оценку `ai_participation`. Они не устанавливают авторство и не являются вероятностью того, что текст написал ИИ.
- Локальный веб-интерфейс, CLI и HTTP API; тексты не отправляются во внешний сервис проверки. Для первоначальной подготовки HF нужны внешние зависимости и модель, затем возможна автономная работа.

### Мини-схема: документ + ответ → фрагмент

```text
Документ: «Первичные документы хранятся пять лет,
           вторичные документы — десять лет».
                       +
Ответ: «Срок хранения первичных документов составляет десять лет».
                       ↓
Фрагмент ответа [46:53] = «десять »
label: doubtful; confidence: 0.4346; общий риск: 0.5794
```

Это сокращение **уже сохранённого**, а не заново рассчитанного примера Windows HF: [запрос](reports/p8/run39271014161/hf-autonomy/exe-request.json) → [полный ответ с токенами](reports/p8/run39271014161/hf-autonomy/exe-response.json). Координаты включают пробел; `confidence` и общий риск — разные поля.

## Подтверждённое качество и статус приёмки

**Официальный отложенный test — 177 пар**, однократное измерение замороженного P5 в [прогоне 39266451529](https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/39266451529). Это не dev и не внутренний demo-набор.

| Единица оценки | F1 | FPR | Уточнение |
|---|---:|---:|---|
| Токены | **0.6252** | **0.0155** | TP/FP/FN/TN = 191/139/90/8848 |
| Строгие фрагменты | **0.5165** | — | IoU ≥ 0.5, greedy matching; TP/FP/FN = 47/38/50 |
| Ответы | **0.6258** | **0.1875** | TP/FP/FN/TN = 51/15/46/65 |

У фрагментов TN не определён; в проверке gate используется **токенный FPR-proxy**, а не выдуманный span-FPR. Источник точных чисел: [official result.json](reports/p8/run39266451529/official/result.json), [матрица требований](reports/contract/coverage-matrix.md) и [паспорт](reports/ACCEPTANCE_PASSPORT.md).

**Норма ТЗ, п. 4.1.6:** F1 ≥ 0.50 при FPR ≤ 0.40 на отложенной по документам части набора из 1200 пар; для строгих фрагментов IoU ≥ 0.5. Численный gate официального test пройден; это **не заявление о выполнении всех требований договора**. Внутренние ориентиры 0.90/0.10 не подменяют договорную норму.

**Исторический dev — 181 пара, не test:** токенный F1 **0.626**, строгих фрагментов **0.529**, ответов **0.616** ([P5 dev-метрики](spanverify-module/reports/experiments/p5_span_filter/run39170981235/all_family_metrics.json)). Выбор варианта выполнялся по dev, не по официальному test.

**Статус:** куратор принял численный результат с документированным процессным исключением P8. До пломбы официального test произошёл синтетический fit в отдельном процессе; доказательств влияния test/B на выбор нет, но чистая историческая заморозка **не заявляется**. Более раннее чтение test также раскрыто. Основание: [решение куратора](reports/p8/closure/CURATOR_DECISION.json), [отклонения](reports/DEVIATIONS.md), [квитанция stable-публикации](reports/p8/closure/STABLE_RECEIPT.json). Stable содержит те же байты, что RC; оформление README не является новым научным этапом или перевыпуском.

## Как устроен конвейер

```text
Документ + ответ
  → пять признаков на токен
  → контекстные признаки соседних токенов + token-head
  → кандидаты фрагментов + span-filter
  → правила атрибуции / anchor
  → фрагменты, риск и итоговый verdict
```

- **demo:** лексические суррогаты признаков без torch; проверка интерфейсов и поставки. Его оценки нельзя выдавать за результат HF.
- **HF:** реальная модель `ai-forever/rugpt3small_based_on_gpt2`, внимание и скрытые состояния; внешний runtime. Замороженные головы и фильтр загружаются из [frozen P5 bundle](spanverify-module/reports/experiments/p5_span_filter/run39170981235/combined_bundle.json); обучение при проверке текста не требуется.
- Предсказания, координаты и границы проверяются регрессиями; длинный ответ обрабатывается окнами, без молчаливого обрезания и заполнения нулями.

### Корпус A3 и путь P1 → P5

**A3: 1200 пар на реальных документах**, document-clean разбиение **842 train / 181 dev / 177 test**, без общих документов между частями. Ответы и разметка получены **синтетически**, поэтому этот корпус не доказывает переносимость на произвольные ответы людей или LLM.

Восемь режимов генерации: `faithful`, `contradiction`, `number_attribution`, `partial`, `unconfirmed`, `missing`, `oversight`, `excess` (включая корректные ответы). [Манифест, split counts и SHA-256](spanverify-module/data/corpus_a3/manifest.json) · [пары с разметкой](spanverify-module/data/corpus_a3/pairs.jsonl).

| Вариант | Изменение научного конвейера |
|---|---|
| P1 | Answer-head и честная базовая точка сравнения |
| P2 | Token-head вместо общей смеси признаков |
| P3 | Локализация seed/grow и границы фрагментов |
| P4 | Контекст соседних токенов |
| P5 | Pure candidate builder и span-filter; итоговый замороженный вариант |

Все источники и роли train/dev/test: [EXPERIMENTS_INDEX](reports/EXPERIMENTS_INDEX.md). Исторические результаты не пересчитывались ради README; сводные legacy-числа находятся в [CURRENT_METRICS.json](spanverify-module/reports/CURRENT_METRICS.json), а актуальный официальный test — в файле прогона выше.

## Четыре важных ограничения

1. **Скорость HF не соответствует двум нормам (канонические нормативные 2 ядра, run 39472061036, `reports/speed/r1/PHASE_B_REPORT.md`):** на 1000 знаков warm median **1.22412 с** p95 **1.74642 с** (cold **6.88333 с**), не ≤ 0,03 с; на 50 000 знаков warm **12.41877 с** p95 **12.52436 с** (cold **18.03188 с**), не ≤ 1 с; на 2 млн знаков warm **52.67317 с** p95 **53.15244 с** (cold **52.41802 с**) — ≤ 60 с выполнено. Исторические 4-ядерные P7 значения: на 1000 знаков **0.763 с**, на 50 000 знаков **8.168 с**, на 2 млн знаков **38.769 с** (runs 39232439363/39236657451, исторические, 4 ядра). Маржинальная пропускная **0.01673 с/1000** < 0,03 и хранение **1434255406 B** < 1,5 ГБ — страница не отменяет FAIL латентности.
2. **AI-share — частичное выполнение.** Синтетическая калибровка даёт MAE **0.2565**, остаётся немонотонность; это не оценка авторства и не доказательство участия модели в произвольном тексте.
3. **Полный HF footprint велик.** Каталог модели **554.3 MB**, внешние библиотеки **870.6 MB**; полный размер SDK неизвестен. Ограничение размера EXE/pyz нельзя переносить на весь HF-runtime.
4. **Перенос на корпус B слабый.** На 1000 внешних парах токенный F1 **0.0314**, строгих фрагментов **0.0196**, ответов **0.5031**. B не является договорным gate; права на корпус окончательно не подтверждены, неоднозначность разметки раскрыта. Нельзя обещать универсальную точность по A3.

Подробности, процессные инциденты и принятые исключения: **[DEVIATIONS.md](reports/DEVIATIONS.md)**. Не достигнутое остаётся не достигнутым и после публикации stable.

## Карта доказательств

| Документ | Что подтверждает |
|---|---|
| [ACCEPTANCE_PASSPORT](reports/ACCEPTANCE_PASSPORT.md) | Итог приёмки, точные источники и границы утверждений |
| [FINAL_REPORT](reports/FINAL_REPORT.md) | Научный и инженерный результат вместе с ограничениями |
| [DEVIATIONS](reports/DEVIATIONS.md) | Невыполненные нормы и процессные исключения |
| [coverage-matrix.md](reports/contract/coverage-matrix.md) / [JSON](reports/contract/coverage-matrix.json) | Пункт ТЗ → проверка → артефакт → статус |
| [EXPERIMENTS_INDEX](reports/EXPERIMENTS_INDEX.md) | Прогоны P1–P8, происхождение и роль метрик |
| [VERIFICATION.json](reports/p8/VERIFICATION.json) | Read-only проверка закрытия и stable-поставки: 30 проверок |
| [TESTS.md](docs/TESTS.md) | Реальные файлы suite, обязанности, node counts и причины skip |

## Для разработчика

Из корня репозитория, Python ≥ 3.10 (для полного QA используются зависимости ниже):

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r spanverify-module/requirements.txt -r spanverify-module/requirements-token-head-tests.txt
cd spanverify-module
python -m pytest
```

Полный базовый suite: **950 тестов, 945 passed, 2 skipped**, без исключения тестов; два необязательных сравнения требуют дополнительных HF/ROUGE-библиотек. Это QA, а не новое обучение production или измерение официального test. Детали и точная коллекция — в [TESTS](docs/TESTS.md).

**Workflows / triggers:** обычный `ci.yml` (quality, metrics, contract и финальная read-only пломба) — push / pull_request / ручной запуск / ночное расписание; научные и релизные workflows — только согласно собственным явным условиям. Полный реестр заданий и статусов: [ЗАДАНИЯ_CI](docs/ЗАДАНИЯ_CI.md); оформление не запускает новые научные замеры или сборку релиза.

[Руководство](docs/РУКОВОДСТВО.md) · [README модуля](spanverify-module/README.md) · [ТЗ и календарный план](docs/ТЗ_и_календарный_план.md) · [Программа приёмки](docs/ПРОГРАММА_И_МЕТОДИКА_ПРИЁМКИ.md) · [Лицензия](LICENSE)
