#!/usr/bin/env python3
"""Сравнение вариантов span_expansion на dev.

Читает experiment.json из reports/experiments/a3/{clause,mask,clause_capped,mask_padded}/
и создаёт таблицу сравнения в reports/experiments/a3/comparison.md
"""

import json
import sys
from pathlib import Path


def load_variant(variant: str, base_dir: Path) -> dict | None:
    """Загрузить experiment.json для варианта."""
    exp_file = base_dir / variant / "experiment.json"
    if not exp_file.exists():
        return None
    return json.loads(exp_file.read_text(encoding="utf-8"))


def format_metric(value, fmt=".3f") -> str:
    """Форматировать метрику."""
    if value is None:
        return "N/A"
    return f"{value:{fmt}}"


def main():
    base_dir = Path("reports/experiments/a3")
    if not base_dir.exists():
        print(f"❌ Директория {base_dir} не найдена", file=sys.stderr)
        return 1

    variants = ["clause", "mask", "clause_capped", "mask_padded"]
    results = {}

    for variant in variants:
        data = load_variant(variant, base_dir)
        if data is None:
            print(f"⚠️  Вариант {variant}: experiment.json не найден", file=sys.stderr)
            continue
        results[variant] = data

    if not results:
        print("❌ Ни один вариант не загружен", file=sys.stderr)
        return 1

    # Создаём таблицу сравнения
    lines = [
        "# Сравнение вариантов span_expansion (P0 шаг 1)",
        "",
        f"**Корпус:** A3 (реальные акты)",
        f"**Режим:** hf",
        f"**Dev pairs:** {results[list(results.keys())[0]].get('dev_official', {}).get('pairs', 'N/A')}",
        "",
        "## Метрики на dev",
        "",
        "| Вариант | Strict Span F1 | Containment | Width Ratio | Token F1 | Token FPR | Answer F1 | Verdict F1 |",
        "|---------|----------------|-------------|-------------|----------|-----------|-----------|------------|",
    ]

    for variant in variants:
        if variant not in results:
            continue
        data = results[variant]
        dev = data.get("dev_official", {})
        spans = dev.get("spans", {})
        tokens = dev.get("tokens", {})
        answers = dev.get("answers", {})
        verdicts = dev.get("verdicts", {})

        row = (
            f"| **{variant}** "
            f"| {format_metric(spans.get('strict_f1_iou_0_5'))} "
            f"| {format_metric(spans.get('coverage'))} "
            f"| {format_metric(spans.get('mean_width_ratio'), '.2f')}× "
            f"| {format_metric(tokens.get('f1'))} "
            f"| {format_metric(tokens.get('fpr'))} "
            f"| {format_metric(answers.get('f1'))} "
            f"| {format_metric(verdicts.get('f1'))} |"
        )
        lines.append(row)

    lines.extend([
        "",
        "## Контрольные метрики на dev",
        "",
        "| Вариант | Token Precision | Token Recall | Token AUC | Answer AUC | Verdict FPR |",
        "|---------|-----------------|--------------|-----------|------------|-------------|",
    ])

    for variant in variants:
        if variant not in results:
            continue
        data = results[variant]
        dev = data.get("dev_official", {})
        tokens = dev.get("tokens", {})
        answers = dev.get("answers", {})
        verdicts = dev.get("verdicts", {})

        row = (
            f"| **{variant}** "
            f"| {format_metric(tokens.get('precision'))} "
            f"| {format_metric(tokens.get('recall'))} "
            f"| {format_metric(tokens.get('auc'))} "
            f"| {format_metric(answers.get('auc'))} "
            f"| {format_metric(verdicts.get('fpr'))} |"
        )
        lines.append(row)

    # Добавляем информацию о тестах
    lines.extend([
        "",
        "## Оговорка",
        "",
        "- **Test не измерялся** (--no-test): все метрики приведены на dev split.",
        "- **Выбор варианта:** по strict span F1 ≥ 0.45, width ≤ 2.0×, containment ≥ 0.45.",
        "- **Токенные метрики** должны быть идентичны across variants (tokens_identical: true), "
        "так как меняется только способ формирования фрагментов, не сами токены.",
        "",
    ])

    # Определяем лучший вариант
    best_variant = None
    best_f1 = -1.0
    for variant, data in results.items():
        dev = data.get("dev_official", {})
        spans = dev.get("spans", {})
        f1 = spans.get("strict_f1_iou_0_5")
        if f1 is not None and f1 > best_f1:
            best_f1 = f1
            best_variant = variant

    if best_variant:
        lines.append(f"**Лучший вариант:** {best_variant} (strict span F1 = {best_f1:.3f})")
        lines.append("")

    # Записываем результат
    output = "\n".join(lines)
    output_path = base_dir / "comparison.md"
    output_path.write_text(output, encoding="utf-8")
    print(f"✓ Сравнение записано в {output_path}")
    print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
