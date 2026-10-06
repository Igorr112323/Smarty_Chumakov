# Запуск корпуса A2 и режима `hf` на машине с GPU

Документ описывает шаг, который в песочнице и в CI выполнить нельзя: генерация
естественных ответов моделью и оценка в режиме `hf`. Всё, что здесь написано,
выполняется человеком на машине с видеокартой; числа, полученные после этого, —
единственные, которые можно называть «результатом на языковой модели».

## 0. Что понадобится

* Python 3.10–3.12;
* видеокарта NVIDIA с CUDA (хватит 8 ГБ видеопамяти для модели 1–2 B), либо
  терпеливое ожидание на CPU: маленькая модель (`rugpt3small`, ~120 M) считает
  200 ответов за минуты, 1.5 B на CPU — за часы;
* ~6 ГБ на диске под веса модели и результаты;
* доступ в интернет только для скачивания весов (один раз).

## 1. Окружение

```bash
git clone https://github.com/Igorr112323/Smarty_Chumakov.git
cd Smarty_Chumakov/spanverify-module

python -m venv .venv-gpu
source .venv-gpu/bin/activate           # Windows: .venv-gpu\Scripts\activate

# torch ставится отдельной командой под свою CUDA (пример для CUDA 12.4)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install transformers accelerate
```

Проверка, что GPU виден:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# ожидаемо: True NVIDIA GeForce ...       (на CPU будет False — это тоже рабочий вариант)
```

## 2. Корпус A1 (быстро, без GPU)

```bash
python scripts/build_corpus_a.py --docs data/corpus_a/docs --out data/corpus_a --target 1200 --seed 42
```

Проверьте, что повторный запуск даёт тот же `sha256` в `data/corpus_a/manifest.json`:
это признак, что корпус воспроизводим.

## 3. Генерация A2

Сначала сухой прогон (работает без весов — посмотреть вопросы и промты):

```bash
python scripts/build_corpus_a2.py --docs data/corpus_a/docs --out data/corpus_a2 --dry-run
```

Затем генерация. Маленькая модель для быстрой проверки пайплайна:

```bash
python scripts/build_corpus_a2.py --docs data/corpus_a/docs --out data/corpus_a2 \
    --model ai-forever/rugpt3small_based_on_gpt2 --limit 200
```

Для содержательных ответов лучше модель 1.5–3 B (например,
`Qwen/Qwen2.5-1.5B-Instruct`); в коде используется `pipeline("text-generation")`, поэтому
подходит любая совместимая модель. Что получится:

* `data/corpus_a2/questions.jsonl` — задания (документ, вопрос, промт, ожидаемое значение);
* `data/corpus_a2/pairs_draft.jsonl` — ответы модели и **черновая** разметка
  (`"auto": true`, `"needs_expert_review": true`, поля `draft_verdict`, `draft_score`);
* `data/corpus_a2/manifest.json` — модель, устройство (`cpu`/`cuda`), число пар.

Важно: черновая разметка не является истиной. Метки в `pairs_draft.jsonl` — это
предсказания нашего же конвейера; использовать их как эталон нельзя.

## 4. Ручная проверка

```bash
# очередь из всех черновиков A2
python scripts/review_server.py --make-sample --source data/corpus_a2/pairs_draft.jsonl \
    --queue data/review_queue/corpus_a2.jsonl --share 1.0 --seed 1

# 10 % A1 — выборочный контроль управляемых подмен
python scripts/review_server.py --make-sample --source data/corpus_a/splits/test.jsonl \
    --queue data/review_queue/corpus_a_10pct.jsonl --share 0.10 --seed 42

python scripts/review_server.py --queue data/review_queue/corpus_a2.jsonl \
    --out data/review_queue/decisions.jsonl --port 8770
```

Страница показывает контекст, ответ с подсветкой метки и три кнопки: **верно** (1),
**неверно** (2), **пропустить** (3). Решения дописываются в JSONL; повторное решение
по той же паре заменяет прежнее, история остаётся в файле. Итог перенесите в
`reports/CORPUS_REPORT.md`: сколько проверено, сколько подтверждено, что нашлось.

## 5. Режим `hf` (наш детектор на реальной модели)

```bash
python scripts/rus_hallu_eval.py --data data/external/rushallu --mode hf \
    --limit 200 --json reports/rus_hallu_eval_hf.json
```

* `--limit 200` — разумная первая проверка; полный прогон 1000 пар делайте, когда
  убедитесь, что память не переполняется;
* устройство выбирается автоматически (`cuda`, если доступна);
* отчёт складывается в отдельный файл: числа `hf` и `demo` нельзя смешивать в одном
  выводе.

Если GPU нет, а `hf` нужен: запустите с `--limit 50` — на CPU это займёт десятки
минут, но чисел не испортит.

## 6. Что делать после прогона

1. Перенести метрики `hf` в `reports/CORPUS_REPORT.md` **отдельным разделом** и
   рядом указать модель и устройство.
2. Пересобрать единый файл чисел: `python scripts/collect_metrics.py`.
3. `python scripts/check_numbers.py` — проверить, что документы сходятся с числами.
4. Закоммитить отчёты (веса и сырые данные не коммитить) и отметить в `PROGRESS.md`,
   что шаг выполнен и какими командами проверен.

## 7. Частые проблемы

| Симптом | Причина и что делать |
|---|---|
| `torch.cuda.is_available() == False` | драйвер или версия CUDA не совпадают с установленным torch: переустановите torch под свою CUDA |
| `CUDA out of memory` | уменьшите `--limit`, возьмите модель меньше или запустите на CPU |
| модель не скачивается | нет доступа к HuggingFace: скачайте веса вручную и укажите локальный путь в `--model` (папка с `config.json`) |
| ответы пустые или повторяют промт | модель слишком слабая: возьмите 1.5 B+ и оставьте `do_sample=False` |
| `data/external/rushallu` пуст | сначала `python scripts/fetch_rushallu.py --out data/external/rushallu --verify` |
