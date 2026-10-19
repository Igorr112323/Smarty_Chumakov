# P1: Сравнение baseline vs answer-head (cross-fitting OOF)

**Train pairs:** 842
**Dev pairs:** 181
**P1 features:** 37 (schema 1.0)
**OOF method:** 5-fold cross-fitting, seed=42
**Threshold method:** choose_threshold_report(max_fpr=0.30)

## Результаты

| Метрика | Baseline (mean_top3) | P1 (answer-head) | Delta |
|---------|---------------------|-----------------|-------|
| Dev Raw AUC | 0.5083 | 0.7321 | +0.2238 |
| Dev Cal AUC | 0.4838 | 0.7338 | +0.2500 |
| Dev F1 | 0.0000 | 0.6164 | +0.6164 |
| Dev FPR | 0.0000 | 0.1647 | +0.1647 |
| Dev Precision | 0.0000 | 0.7778 | +0.7778 |
| Dev Recall | 0.0000 | 0.5104 | +0.5104 |
| TP/FP/FN/TN | 0/0/96/85 | 49/14/47/71 | |
| Threshold | 0.562254 | 0.579439 | |
| Pos. predictions | 0 | 63 | |

## Fold receipts

| Fold | Fit docs | Fit samples | Held-out docs | Held-out samples | Held-out pos | Held-out neg |
|------|----------|-------------|---------------|------------------|--------------|--------------|
| 0 | 212 | 659 | 53 | 183 | 93 | 90 |
| 1 | 212 | 683 | 53 | 159 | 87 | 72 |
| 2 | 212 | 687 | 53 | 155 | 97 | 58 |
| 3 | 212 | 672 | 53 | 170 | 88 | 82 |
| 4 | 212 | 667 | 53 | 175 | 102 | 73 |

**Длительность:** 858.6s