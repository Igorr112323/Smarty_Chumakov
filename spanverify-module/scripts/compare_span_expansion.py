#!/usr/bin/env python3
"""Сравнение вариантов span_expansion на dev.

Читает experiment.json из reports/experiments/a3/{clause,mask,clause_capped,mask_padded}/
и создаёт таблицу сравнения в reports/experiments/a3/comparison.md

Требует все 4 варианта с валидными dev_official.
"""

import argparse
import json
import sys
from pathlib import Path


def load_variant(variant: str, base_dir: Path) -> dict:
    """Загрузить experiment.json для варианта. Ошибка, если не найден или невалиден."""
    # Ищем experiment.json в возможных местах
    candidates = [
        base_dir / variant / "experiment.json",
        base_dir / f"hf-experiment-a3-{variant}" / "experiment.json",
        base_dir / f"hf-experiment-a3-{variant}" / "reports" / "experiments" / "a3" / "experiment.json",
    ]
    
    found_paths = []
    for candidate in candidates:
        if candidate.exists():
            found_paths.append(candidate)
    
    if len(found_paths) == 0:
        raise FileNotFoundError(
            f"Вариант {variant}: experiment.json не найден в {base_dir}. "
            f"Проверены пути: {', '.join(str(p) for p in candidates)}"
        )
    
    if len(found_paths) > 1:
        raise ValueError(
            f"Вариант {variant}: найдено несколько experiment.json: {found_paths}. "
            f"Ожидался ровно один файл."
        )
    
    exp_file = found_paths[0]
    data = json.loads(exp_file.read_text(encoding="utf-8"))
    
    # Проверяем наличие dev_official
    if "dev_official" not in data:
        raise ValueError(f"Вариант {variant}: отсутствует поле dev_official в {exp_file}")
    
    dev = data["dev_official"]
    if dev is None:
        raise ValueError(f"Вариант {variant}: dev_official is null в {exp_file}")
    
    # Проверяем наличие ключевых метрик
    required_fields = ["spans", "tokens", "answers", "verdicts"]
    for field in required_fields:
        if field not in dev:
            raise ValueError(f"Вариант {variant}: отсутствует поле {field} в dev_official")
    
    return data


def format_metric(value, fmt=".3f") -> str:
    """Форматировать метрику."""
    if value is None:
        return "N/A"
    return f"{value:{fmt}}"


def check_tokens_identical(results: dict) -> dict:
    """Проверить, что токены идентичны во всех вариантах.
    
    Возвращает словарь:
    {
        "identical": bool,  # True если все 6 полей совпадают во всех вариантах
        "missing_fields": dict,  # {variant: [отсутствующие поля]}
        "mismatched_fields": dict,  # {field: {variant: value}}
    }
    """
    required_keys = ["tp", "fp", "fn", "tn", "f1", "fpr"]
    
    result = {
        "identical": True,
        "missing_fields": {},
        "mismatched_fields": {},
    }
    
    variants = list(results.keys())
    if len(variants) < 2:
        return result
    
    # Проверяем наличие всех обязательных полей в каждом варианте
    for variant in variants:
        tokens = results[variant]["dev_official"]["tokens"]
        missing = [key for key in required_keys if key not in tokens]
        if missing:
            result["missing_fields"][variant] = missing
            result["identical"] = False
    
    # Если есть отсутствующие поля, не сравниваем значения
    if result["missing_fields"]:
        return result
    
    # Сравниваем значения полей между вариантами
    baseline_variant = variants[0]
    baseline = results[baseline_variant]["dev_official"]["tokens"]
    
    for key in required_keys:
        baseline_value = baseline[key]
        mismatches = {baseline_variant: baseline_value}
        
        for variant in variants[1:]:
            tokens = results[variant]["dev_official"]["tokens"]
            variant_value = tokens[key]
            mismatches[variant] = variant_value
            
            if baseline_value != variant_value:
                result["mismatched_fields"][key] = mismatches
                result["identical"] = False
                break
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Сравнение вариантов span_expansion на dev")
    parser.add_argument("--input-dir", required=True, help="Базовая директория с вариантами")
    parser.add_argument("--out", required=True, help="Выходной файл для Markdown отчёта")
    parser.add_argument("--source-run-id", help="ID исходного workflow run")
    parser.add_argument("--source-commit", help="Commit исходного запуска")
    
    args = parser.parse_args()
    
    base_dir = Path(args.input_dir)
    if not base_dir.exists():
        print(f"❌ Директория {base_dir} не найдена", file=sys.stderr)
        sys.exit(1)

    variants = ["clause", "mask", "clause_capped", "mask_padded"]
    results = {}

    # Загружаем все 4 варианта (ошибка, если хотя бы одного нет)
    for variant in variants:
        try:
            data = load_variant(variant, base_dir)
            results[variant] = data
            print(f"✓ Вариант {variant} загружен")
        except (FileNotFoundError, ValueError) as e:
            print(f"❌ {e}", file=sys.stderr)
            sys.exit(1)

    if len(results) != 4:
        print(f"❌ Загружено только {len(results)} из 4 вариантов", file=sys.stderr)
        sys.exit(1)

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
        "## Проверка неизменности токенов",
        "",
    ])
    
    tokens_check = check_tokens_identical(results)
    
    if tokens_check["missing_fields"]:
        lines.append("**❌ Отсутствуют обязательные поля токенов:**")
        for variant, fields in tokens_check["missing_fields"].items():
            lines.append(f"- {variant}: {', '.join(fields)}")
        lines.append("")
        lines.append("Без полных данных TP/FP/FN/TN/F1/FPR проверка равенства невозможна.")
    elif tokens_check["identical"]:
        lines.append("**✅ Токены идентичны** во всех вариантах (TP/FP/FN/TN/F1/FPR совпадают)")
        lines.append("")
        lines.append("Это подтверждает, что изменение стратегии расширения границ не влияет на токены.")
    else:
        lines.append("**❌ Токены различаются** между вариантами:")
        for field, variant_values in tokens_check["mismatched_fields"].items():
            lines.append(f"- {field}:")
            for variant, value in variant_values.items():
                lines.append(f"  - {variant}: {value}")
        lines.append("")
        lines.append("Это указывает на проблему: токены не должны меняться при изменении стратегии границ.")
    
    # Определяем лучший вариант
    best_variant = None
    best_f1 = -1.0
    best_width = float("inf")
    
    for variant, data in results.items():
        dev = data["dev_official"]
        spans = dev["spans"]
        f1 = spans.get("strict_f1_iou_0_5")
        width = spans.get("mean_width_ratio")
        
        if f1 is not None:
            if f1 > best_f1 or (f1 == best_f1 and width is not None and width < best_width):
                best_f1 = f1
                best_variant = variant
                best_width = width if width is not None else float("inf")

    lines.extend([
        "",
        "## Выбор лучшего варианта",
        "",
    ])

    if best_variant:
        # Проверяем внутренние ориентиры
        best_data = results[best_variant]["dev_official"]
        best_spans = best_data["spans"]
        
        f1_ok = best_spans["strict_f1_iou_0_5"] >= 0.45
        width_ok = best_spans["mean_width_ratio"] <= 2.0
        containment_ok = best_spans["coverage"] >= 0.45
        
        all_ok = f1_ok and width_ok and containment_ok
        
        lines.append(f"**Лучший вариант:** {best_variant}")
        lines.append(f"- Strict span F1: {best_spans['strict_f1_iou_0_5']:.3f} (ориентир ≥ 0.45): **{'✅' if f1_ok else '❌'}**")
        lines.append(f"- Mean width ratio: {best_spans['mean_width_ratio']:.2f}× (ориентир ≤ 2.0×): **{'✅' if width_ok else '❌'}**")
        lines.append(f"- Containment: {best_spans['coverage']:.3f} (ориентир ≥ 0.45): **{'✅' if containment_ok else '❌'}**")
        lines.append("")
        
        if all_ok:
            lines.append("**✅ Все внутренние ориентиры P0-3 выполнены**")
        else:
            lines.append("**❌ Не все внутренние ориентиры выполнены**")
            lines.append("")
            lines.append("Внутренние ориентиры не являются договорными требованиями, а служат для промежуточной оценки.")
        
        # Delta к baseline (clause)
        if best_variant != "clause":
            baseline_f1 = results["clause"]["dev_official"]["spans"]["strict_f1_iou_0_5"]
            delta = best_f1 - baseline_f1
            lines.append("")
            lines.append(f"**Delta к baseline (clause):** {delta:+.3f}")
    
    # Добавляем метаинформацию
    lines.extend([
        "",
        "---",
        "",
        "## Метаинформация",
        "",
    ])
    
    if args.source_run_id:
        lines.append(f"- **Source run:** {args.source_run_id}")
    if args.source_commit:
        lines.append(f"- **Source commit:** {args.source_commit}")
    
    lines.extend([
        f"- **Generated at:** {__import__('datetime').datetime.now().isoformat()}",
        f"- **Script:** compare_span_expansion.py",
        "",
    ])

    # Записываем Markdown
    output = "\n".join(lines)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output, encoding="utf-8")
    print(f"✓ Markdown отчёт записан в {output_path}")
    
    # Записываем JSON
    json_path = output_path.with_suffix(".json")
    json_data = {
        "source_run_id": args.source_run_id,
        "source_commit": args.source_commit,
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "variants": {},
    }
    
    for variant, data in results.items():
        dev = data["dev_official"]
        json_data["variants"][variant] = {
            "pairs": dev.get("pairs"),
            "tokens": dev["tokens"],
            "spans": dev["spans"],
            "answers": dev["answers"],
            "verdicts": dev["verdicts"],
        }
    
    json_data["tokens_identical"] = tokens_check["identical"]
    json_data["tokens_missing_fields"] = tokens_check.get("missing_fields", {})
    json_data["tokens_mismatched_fields"] = list(tokens_check.get("mismatched_fields", {}).keys())
    json_data["best_variant"] = best_variant
    
    if best_variant:
        best_spans = results[best_variant]["dev_official"]["spans"]
        json_data["targets_met"] = {
            "strict_f1_gte_0.45": best_spans["strict_f1_iou_0_5"] >= 0.45,
            "width_lte_2.0": best_spans["mean_width_ratio"] <= 2.0,
            "containment_gte_0.45": best_spans["coverage"] >= 0.45,
        }
        json_data["all_targets_met"] = all(json_data["targets_met"].values())
    
    json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ JSON записан в {json_path}")
    
    print(output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
