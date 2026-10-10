# Задача 1 — HF A3: статус выполнения

Дата: 2026-10-10 UTC. Цель качества **не подтверждена**. Статус текущего
локального HF-запуска: `blocked`, метрики `null`, не нулевые результаты модели.
Задачи 2–12 и итоговая приёмка не объявляются выполненными.

## Реальные запуски

Команды выполняются из `spanverify-module`, Python: `../.venv/bin/python`.

- `scripts/train_hf_a3.py`: код 2, `BackendUnavailable`, отсутствуют torch / transformers.
  Лог: `hf_a3_train.log`, машинный отчёт: `hf_a3_run.json`, Markdown: `hf_a3_eval.md`.
- `-m spanverify evaluate --dataset data/corpus_a3/pairs.jsonl --mode hf --weights config/weights_hf.json`:
  явная ошибка HF-зависимостей, `task1_hf_evaluate.log`. Поскольку обучение заблокировано,
  файл weights_hf не создан; CLI legacy при отсутствующем файле берёт defaults,
  поэтому этот запуск НЕ является оценкой обученной модели.
- `-m pytest tests/test_hf_a3_pipeline.py tests/test_feature_cache.py tests/test_workflow_limits.py -q -o addopts=`:
  результат в `task1_focused_tests.log` (это функциональные тесты инфраструктуры, не измерение HF-качества).
- `-m spanverify selftest`: все встроенные проверки пройдены, `task1_selftest.log`.
- `-m ruff check spanverify scripts tests`, `-m black --check spanverify scripts tests`:
  `task1_lint.log`.
- Полная проверка с coverage: `task1_full_tests.log`; пока процесс не завершён,
  процент покрытия и успех всей проверки не утверждаются.
- Сбор `METRICS.json` / сверка чисел: `task1_collect_metrics.log`, `task1_check_numbers.log`.
  Завершённость определяется результатами этих команд, не наличием старого METRICS.json.

## Подтверждённые расхождения с исходными предположениями

Чтение текущих JSONL: A3 содержит 1200 пар, официальные split train/dev/test:
842 / 181 / 177. Старые результаты в `reports/experiments/a3/experiment.json`
относятся к историческому прогону, не к новому коду.

`feature_cache_key()` возвращает SHA256. Проверять `model_name in key` нельзя:
имя модели не присутствует в хеше. Новые строки кеша сохраняют `meta.model`,
`meta.mode`, диагностические массивы. Несовпадение явной модели вызывает
`BackendUnavailable` с `model mismatch`; legacy-кеш читается старыми
потребителями, но новый acceptance-конвейер отклоняет его без provenance.

A3 группируется по документам `meta.doc_id/group`, но legacy train группирует
по `subject/question/template`. Новый скрипт адаптирует **копии** записей для
внутреннего document-disjoint split; данные и существующая схема не меняются.
Legacy внутренний CV головы делит токены и остаётся диагностическим, не
приёмочным. Новый отдельный ablation использует document folds и стандартизацию
только на train каждого fold. Official test labels не используются для отбора.

## Что добавлено

- `scripts/train_hf_a3.py`: существующие `train()` и `Verifier(mode='hf')`,
  строгие official splits, проверка кеша, отдельные имена артефактов, логи,
  checksum корпуса/исходников/артефактов, null при ошибках.
- `scripts/hf_feature_ablation.py`: baseline + каждый из четырёх кандидатов +
  совместный вариант; grouped OOF/dev AUC, paired bootstrap по документам.
  Пороги не меняются, production FEATURE_NAMES не расширяется без доказательства.
- `scripts/run_experiments.py --corpus a3 --mode hf`: новый изолированный путь,
  без дополнительного подбора маски из старого run_experiments.
- Запись/чтение кеша сохраняет диагностические признаки и provenance.
- Сквозная проверка train наследует явную модель/кеш верификатора, не конфиг другой модели.
- `collect_metrics.py`: аддитивный `hf_a3_acceptance`, в том числе blocked/null.
- `tests/test_hf_a3_pipeline.py`: инварианты изоляции артефактов, ошибок кеша,
  сплитов и ablation; прежние тесты сохранены.
- `.github/workflows/hf-a3-acceptance.yml`: шардированный реальный HF-предпосчёт,
  обучение и независимый AUC-ablation, артефакты. Код не пушит чужие ветки.

## Блокеры и ограничения

Локальная сеть не разрешает Hugging Face. `gh run download 37609202780 -n hf-runs-a3`
не получил архив из Azure blob (EOF). Поэтому нужны либо результаты нового CI,
либо предоставленные локальные веса/новый полный кеш с provenance и диагностикой.

Нет данных о приросте AUC/F1 кандидатов: никакие признаки ещё не переведены в
рабочие. Нельзя честно убрать из README ограничение качества. `head_hf.json` и
`participation_hf.json` при успешном запуске могут иметь `type=none`: это явный
статус отсутствия выбранной/обученной головы, не свидетельство качества.

AUC-ablation — проверка информативности, не обещание конечного token F1.
Положительный эффект потребует отдельной backward-compatible интеграции и
повторной сквозной проверки. Цель F1/FPR остаётся критерием, не измерением.
