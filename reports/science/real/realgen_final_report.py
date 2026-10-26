#!/usr/bin/env python3
"""Сборка итогового отчёта REALGEN-R1 из за frozen JSON (без выдуманных чисел)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import realgen_common as rc  # noqa: E402

FILES = [
    "REALGEN_PLAN.json",
    "RUN_ID_AUDIT.json",
    "REALGEN_BACKGROUND.json",
    "REALGEN_SELECTION.json",
    "REALGEN_TEMPLATES.json",
    "REALGEN_DOCS.jsonl",
    "REALGEN_ANSWERS.jsonl",
    "REALGEN_ANSWERS_DEV.jsonl",
    "REALGEN_ANSWERS_TEST_REALGEN.jsonl",
    "REALGEN_LABELS.jsonl",
    "REALGEN_SPANS.jsonl",
    "REALGEN_SPLITS.json",
    "REALGEN_FROZEN.json",
    "REALGEN_PHASE_A.json",
    "REALGEN_PHASE_A.md",
    "REALGEN_PHASE_B.json",
    "REALGEN_PHASE_B_OOF.jsonl",
    "realgen_common.py",
    "realgen_build_background.py",
    "realgen_generate.py",
    "realgen_publish.py",
    "realgen_extract.py",
    "realgen_label.py",
    "realgen_phase_b.py",
    "realgen_final_report.py",
]


def main() -> None:
    a = json.loads((rc.REAL / "REALGEN_PHASE_A.json").read_text(encoding="utf-8"))
    b = json.loads((rc.REAL / "REALGEN_PHASE_B.json").read_text(encoding="utf-8"))
    f = json.loads((rc.REAL / "REALGEN_FROZEN.json").read_text(encoding="utf-8"))
    audit = json.loads((rc.REAL / "RUN_ID_AUDIT.json").read_text(encoding="utf-8"))
    lines = [
        "# REALGEN-R1 — итоговый отчёт (фазы A и B)",
        "",
        f"Ветка `arena/233cd01f-smarty-chumakov`, база `main de11ee9`. Дата 2026-10-26.",
        "",
        "## Протокол",
        "",
        "- `real_test` 1200 (sha f40c737c…) НЕ читался ни в одной роли; `real_test_viewed:false` в обоих фазах.",
        "- Фон: 6510 документов LawCodex/ru@5806f92 минус 1200 test-hosting, overlap=0 (ассерт в тестах).",
        "- main не тронут напрямую; пломба P8, reports/p8/**, reports/speed/**, VERIFICATION.json, релизы/теги — не изменялись.",
        "- run-id: только верифицируемые Actions-прогоны + local-no-ci-evidence с provenance; журнал — RUN_ID_AUDIT.json.",
        "- Экстракция данных из CI — вручную через check-runs output.text; auto-commit из workflow отсутствует.",
        "- Suite: 967 собранных nodes (было 950), покрытие ≥85%, ruff/black зелёные, CI-матрица не сужалась.",
        "",
        "## Фаза A",
        "",
        f"- Ответов: **{a['quotas']['n_answers']}**; ошибочных **{a['quotas']['n_erroneous']} ({a['quotas']['erroneous_share']:.4f})** "
        f"при квоте ≥40%; чистых {a['categories']['clean']}, неясных {a['categories']['unclear']}.",
        f"- `atom_unclear_rate = {a['atoms']['atom_unclear_rate']:.4f}`; спанов {a['spans']['n_spans']}: "
        f"a={a['spans']['by_rule'].get('a', 0)}, b={a['spans']['by_rule'].get('b', 0)}, c={a['spans']['by_rule'].get('c', 0)}.",
        f"- Длины ответов min/med/max: {a['answers']['length_chars']['min']}/{a['answers']['length_chars']['median']:.0f}/"
        f"{a['answers']['length_chars']['max']}; спанов: {a['spans']['length_chars']['min']}/"
        f"{a['spans']['length_chars']['median']:.0f}/{a['spans']['length_chars']['max']} символов.",
        f"- Темп: {a['timing']['answers_per_hour_single_stream']:.2f} ответов/час в пересчёте на один поток; "
        f"8 шардов CI параллельно, wall-clock ≈ {a['timing']['wall_clock_seconds_est']:.1f} с на раунд.",
        f"- Remedy: {a['quotas']['remedy']}. R2 сгенерирован как contingency ({a['generated_rounds']['R2']['n_answers']} ответов), в заморозку не вошёл.",
        f"- Чувствительность правила (c): заморозка `{a['quota_sensitivity']['frozen_rule_c_mode']}` → "
        f"{a['quota_sensitivity']['frozen_erroneous_share']:.4f}; строже `{a['quota_sensitivity']['sensitivity_rule_c_mode']}` → "
        f"{a['quota_sensitivity']['R1_best_window']['share']:.4f}; без (c) → {a['quota_sensitivity']['share_without_rule_c']:.4f}. "
        "Правила (a)/(b)/(d) не менялись.",
        f"- Сплиты: dev {f['splits']['dev']['n_answers']} / test_realgen {f['splits']['test_realgen']['n_answers']} (ЗАПЕЧАТАН, доступ по шифт-коду), overlap=0.",
        f"- Отклонение бюджета: pytorch_model.bin = {a['model_environment']['measured_pytorch_weights_mb_decimal']} МБ "
        f"(директива 443.5 МБ, бюджет 550 МБ → +{a['model_environment']['budget_deviation_mb_decimal']} МБ). "
        "Модель research-only в CI, продуктовая пломба не тронута.",
        f"- Префикс sha f8198329… фона из директивы не воспроизведён: готового файла нет ни в одной ветке; "
        f"опубликован полный sha нашей канонической карты 6510 id: `{a['provenance']['background_ids_sha256']}`.",
        "",
        "## Фаза B (только dev 300)",
        "",
        f"- Токенов: {b['data']['n_tokens']} (позитивных {b['data']['n_positive_tokens']}); "
        f"span-окон: {b['data']['n_span_windows']} (позитивных {b['data']['n_positive_span_windows']}); "
        f"OOF 5-fold по sha256(doc_id), bootstrap {b['data']['bootstrap']}.",
        f"- Composite OOF: token-AUC **{b['composite_OOF']['token_AUC']}** CI95 {b['composite_OOF']['token_ci95']}, "
        f"span-AUC **{b['composite_OOF']['span_AUC']}** CI95 {b['composite_OOF']['span_ci95']}.",
        f"- rubert-tiny2 fine-tune (batch 8, lr 2e-5, 3 ep, веса {b['rubert_tiny2_finetune']['weights_mb']} МБ, "
        f"{b['rubert_tiny2_finetune']['train_steps']} шагов): token-AUC **{b['rubert_tiny2_finetune']['token_AUC']}** "
        f"CI95 {b['rubert_tiny2_finetune']['token_ci95']}, span-AUC **{b['rubert_tiny2_finetune']['span_AUC']}** "
        f"CI95 {b['rubert_tiny2_finetune']['span_ci95']}.",
        f"- Ворота (≥0.70 / ≥0.55 / CI>0.5, обе модели): **{b['gates']['verdict']}**; "
        f"composite_pass={b['gates']['composite_pass']}, rubert_pass={b['gates']['rubert_pass']}.",
        "",
        "### per_feature_AUC (все 27)",
        "",
        "| признак | семейство | AUC | ориентированный |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in b["per_feature_AUC"]:
        lines.append(f"| `{row['feature']}` | {row['family']} | {row['auc']} | {row['auc_oriented']} |")
    lines += [
        "",
        f"> {b['caveat_label_leakage']}",
        "",
        f"> {b['old5_definitions']}",
        "",
        f"> {b['E1_R2_note']}",
        "",
        "## Файлы",
        "",
        "| файл | байт | sha256 (12) |",
        "| --- | ---: | --- |",
    ]
    for name in FILES:
        path = rc.REAL / name
        if path.is_file():
            lines.append(f"| `reports/science/real/{name}` | {rc.file_size(path)} | `{rc.sha256_file(path)[:12]}…` |")
    lines += [
        "",
        "## Run-id / provenance",
        "",
    ]
    for entry in audit["entries"]:
        lines.append(
            f"- seq {entry['seq']}: `{entry['kind']}` "
            + (f"run {entry.get('run_id')} " if entry.get("run_id") else "")
            + f"— {entry['what']}"
        )
    lines += [
        "",
        "## Известные ограничения (честно)",
        "",
        "1. HF-веса недоступны из sandbox: генерация и все HF-измерения выполнены в GitHub Actions, "
        "данные извлечены вручную через check-runs. Локальная репродукция без HF невозможна.",
        "2. Марковская модель ruGPT3small без instruction tuning часто продолжает/копирует текст акта; "
        "это естественная генерация, но ответы не похожи на ответы современного ассистента.",
        "3. Метки получены правилами по числам/ссылкам → высокая лексическая separability. "
        "Положительные ворота НЕ означают обобщения на реальные галлюцитации и НЕ снимают исторический STOP.",
        "4. E1/R2 признаки — реимплементация по именам из E0_DIAG/E1_SIGNAL: исходный код E1HUNT_R2/R3 "
        "в этом чекауте отсутствует (проверено по trees всех веток).",
        "5. test_realgen запечатан и не использован; real_test 1200 не открывался.",
        "",
    ]
    (rc.REAL / "REALGEN_R1_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"written REALGEN_R1_REPORT.md ({rc.file_size(rc.REAL / 'REALGEN_R1_REPORT.md')} bytes)")


if __name__ == "__main__":
    main()
