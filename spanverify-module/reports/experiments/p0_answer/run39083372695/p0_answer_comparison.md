# P0-2: Сравнение трёх агрегаторов

## Runtime manifest

- **Dataset:** data/corpus_a3/pairs.jsonl
- **Train pairs:** 842
- **Dev pairs:** 181
- **Cache source run:** 39060670054
- **Measurement run:** 39083372695
- **Measurement commit:** bdd28007d64fc14504dbc9c3f5d8772d69daf1e6
- **Baseline bundle source run:** 39060670054
- **Baseline bundle source commit:** 14ea3a0792e0ca9502d9b78870dd2caf874f1163
- **Bundle hash:** 09fdf87fd91356aa...
- **Model:** ai-forever/rugpt3small_based_on_gpt2

## Контроль воспроизводимости

**Статус:** ✅ PASSED

## Сравнение агрегаторов

| Агрегатор | Raw AUC | Calibrated AUC | F1 | Recall | FPR | TP/FP/FN/TN | Порог |
|-----------|---------|----------------|----|--------|-----|-------------|-------|
| current | 0.5032 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0/0/96/85 | 0.5758 |
| max | 0.4710 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0/0/96/85 | 0.5719 |
| mean_top3 | 0.5078 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0/0/96/85 | 0.5681 |

**Лучший агрегатор:** current

---

**Статус:** ✅ Сравнение завершено