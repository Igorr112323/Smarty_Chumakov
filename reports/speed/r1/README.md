# SPEED-R1 — Измерительная кампания по быстродействию (только честные замеры)

Этот каталог — результат **Фазы A** (декомпозиция таймингов) кампании SPEED-R1.
Фаза B (честное ускорение) — после отдельного зачёта куратора, оптимизаций до зачёта не вводилось.

## Структура

- `PHASE_A_REPORT.md` — главный отчёт фазы A (таблица декомпозиции, fixed/marginal, fingerprint, выводы)
- `demo_allcores/` — замеры demo n=10 (all cores, OMP unset): `benchmark_demo.json` (сырые `raw_seconds` + median/p95), `fingerprint.json`, `fixed_marginal_demo.json`, `full_demo.json`
- `demo_2cores/` — замеры demo n=10 (2 ядра, OMP=1, taskset): аналогично
- `decompose/` — per-stage декомпозиция demo: `weights_load_s`, `tokenization_s`, `features_forward_and_5feat_s`, `candidate_build_span_filter_s`, `verify_total_s` для 1k/50k/2M warm/cold
- `logs/` — логи прогонов + доказательство невозможности HF в этом sandbox (`hf_attempt_sandbox.json` + `curl huggingface.co` SSL_ERROR)
- Исторические HF-цифры с provenance: `reports/p7/TIMINGS.md` (Run 39232439363 paired 3 повтора), `reports/p7/final-receipts/native-benchmark.json` (Run 39236657451 a85c40a 3 повтора), `reports/p7/final-receipts/native-profiles.json` (torch.addmm 3.28s)

## Как воспроизвести

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r spanverify-module/requirements.txt -r spanverify-module/requirements-token-head-tests.txt
# Demo (без HF)
python tools/speed_r1_measure.py --mode demo --sizes 1000 50000 2000000 --scenarios answer-text --thermals cold-process warm-model --repeats 10 --out reports/speed/r1/demo_allcores --profile
OMP_NUM_THREADS=1 taskset -c 0,1 python tools/speed_r1_measure.py --mode demo --sizes 1000 50000 2000000 --scenarios answer-text --thermals cold-process warm-model --repeats 10 --out reports/speed/r1/demo_2cores --profile

# HF требует torch+transformers+модель rugpt3small (в этом sandbox заблокировано egress whitelist, см. logs/hf_attempt_sandbox.json)
# Исторические HF-замеры: см. PHASE_A_REPORT.md §3.2 и reports/p7/
```

## Честный статус норм (ответ на §3.3 для фазы A — предварительный)

- Demo: 1k≤0.03 PASS (0.013s), 50k≤1s PASS (0.60s), 2M≤60s PASS (27s) — обе машины, оба thermal.
- HF: 1k≤0.03 **FAIL** (0.76s warm, 5.34s cold), 50k≤1s **FAIL** (8.16s warm, 12.67s cold), 2M≤60s PASS (38.77s warm) — P7 authoritative, не sandbox.
- Доминанты: `torch.addmm`/`linear` (60% времени) + fixed 4s загрузки модели → нормы недостижимы без смены модели (запрещено).

См. `PHASE_A_REPORT.md` для полной таблицы декомпозиции и fixed/marginal.
