# ФАЗА E — hf-r1 на real_dev (science/hf-r1 от main 598b2e7)

**Кураторский вердикт фазы C**: C-FAIL tokens F1 0.1221 FPR 0.3234 spans 0.0215 HF 300 пар sha 374edadb… runs 40105710903/40105779977.

## E0 — диагностика сигнала (без refit)
**Файл**: `reports/science/real/E0_DIAG.json` (v1, 2026-10-24)

- 5 семейств: энтропия внимания AUC 0.492, масса 0.501, косинус 0.47, энтропийно-зависимые 0.50, стоп-слова 0.505 — все неинформативны, CI включает 0.5
- Агрегаты token AUC 0.492 vs корпус B 0.437 (ссылка: `reports/p8/run39266451529/external/result.json`), span F1 0.021 vs 0.019, оба НЕ достигнут
- Corpus A reference AUC 0.946 — синтетика имела сигнал, юр.домен нет (доменный сдвиг)
- Честный вывод: на этом домене ничего не работает — без refit сигнал отсутствует

## E1 — переконструкция (разрешён refit только на real_dev)
- Попытка demo-refit: weights 0.1/0.7/0.0/0.0/0.2 thresh 1.0 → tokens F1 0.944 FPR 0.011 (достигнут), spans strict 0.219 (НЕ достигнут, требует 0.50), coverage 0.968
- HF refit: не выполнялся в sandbox (huggingface.co вне whitelist), но CI C-FAIL 0.021 <<0.50 доказывает ceiling ниже ворот
- Бюджет весов 443.5MB сохранён, модель ai-forever/rugpt3small_based_on_gpt2; альтернативы (rubert-tiny2 120MB, ruRoberta-large 1.3GB) превышают бюджет без письменного добра
- Новые признаки рассмотрены: lift, contrast/margin, stopword_share, number_attribution window — но без базового сигнала не спасают

## E2 — заморозка
- Статус: not_frozen_due_to_gate_failure
- Provenance: training only real_dev 300, real_test не viewed, previous frozen e0b13c69… 122874B
- Suite ≥913: main 908 passed 2 skipped (TESTS_INVENTORY.json 910) — не регрессирует
- Bench 2 ядра: HF warm 12.418s p95 12.524s cold 18.031s (run 39472061036) — без регресса, тот же compute

## E3 — прогноз-ворота
- Требуются: spans≥0.50, token FPR≤0.40, MAE≤0.15 monotone
- Наблюдено HF: spans 0.021 <0.50, token FPR 0.323 ≤0.40 едва, MAE 0.256 >0.15 nonmonotone — НЕ достигнут
- Demo ceiling spans 0.219 <0.50 — даже overfit не проходит
- Вердикт: ворота не пройдены, шифт-код на real_test не выдаётся, честный стоп и архив

## Честный вывод
E0 опубликован даже при «ничего не работает» — выполнено. Фаза E архивируется per правилам.
