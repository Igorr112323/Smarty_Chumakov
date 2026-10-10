"""Таблица критерия ТЗ в README — из манифеста, не от руки.

Причина: по протоколу любое число в README обязано приходить из
``reports/hf_final/manifest.json``. handwritten-таблица в какой-то момент
обязательно разъедётся с манифестом на «0,61» против «0.6083», и это будет
выглядеть как подгонка. Здесь — единственный источник: блок между маркерами
``<!-- HF-CRITERION:BEGIN -->`` и ``<!-- HF-CRITERION:END -->``.

Режимы:

* ``--write`` — вставить отрендеренный блок в файл (после шага 3);
* ``--check`` — сверить блок с манифестом; если манифеста ещё нет, блок обязан
  содержать «не измерено»: правдоподобные числа до измерения — то же зло, что
  и их отсутствие;
* ``--stdout`` — напечатать блок (для отчёта и для ответа человеку).

Формат числа — шесть знаков после запятой, как в манифесте: пересказ «примерно
0,6» не проверяется тестом и потому не допускается.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
BEGIN = "<!-- HF-CRITERION:BEGIN -->"
END = "<!-- HF-CRITERION:END -->"
PLACEHOLDER_TEXT = "не измерено"


def fmt(value: Any) -> str:
    if value is None:
        return PLACEHOLDER_TEXT
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def render(manifest: dict[str, Any] | None, corpus: str) -> str:
    """Строки таблицы: метрика | значение | уровень | выборка | источник | тест."""
    lines = [
        BEGIN,
        "",
        f"Критерий ТЗ (Прил. №3 к договору 0117812, п. 4.1): **F1 ≥ 0,60 при FPR ≤ 0,40**"
        " на отложенной по документам части `test` в режиме `hf`. Числа ниже — из "
        f"`spanverify-module/reports/hf_final/manifest.json` (корпус {corpus}); блок собирает "
        "`scripts/render_hf_table.py`, руками вписывать числа в него запрещено, рассинхрон валит CI "
        "(`tests/test_reported_metrics.py`).",
        "",
    ]
    if manifest is None:
        lines += [
            "| Метрика | Значение | Уровень | Выборка | Файл-источник | Подтверждающий тест |",
            "| --- | --- | --- | --- | --- | --- |",
            f"| F1 | {PLACEHOLDER_TEXT} | токены | test (a3) | `reports/hf_final/manifest.json` | `tests/test_criterion_hf.py` |",
            f"| FPR | {PLACEHOLDER_TEXT} | токены | test (a3) | `reports/hf_final/manifest.json` | `tests/test_criterion_hf.py` |",
            "",
            "Измерение ещё не выполнено: прогон — workflow «Протокол hf (критерий ТЗ)», шаг 3 — "
            "и детали протокола: `spanverify-module/docs/METRIC_SPEC.md`, `spanverify-module/docs/EXPERIMENTS.md`.",
        ]
        lines.append(END)
        return "\n".join(lines)
    numbers = manifest.get("metrics_mean_std") or {}
    criterion = manifest.get("criterion") or {}
    rows = [
        (
            "F1 (mean ± std по 5 seed'ам)",
            f"{fmt(numbers.get('f1'))} ± {fmt(numbers.get('f1_std'))}",
            "токены",
            "test (a3)",
            "metrics_mean_std.f1",
            "test_reported_metrics.py",
        ),
        ("FPR", fmt(numbers.get("fpr")), "токены", "test (a3)", "metrics_mean_std.fpr", "test_reported_metrics.py"),
        (
            "Precision",
            fmt(numbers.get("precision")),
            "токены",
            "test (a3)",
            "metrics_mean_std.precision",
            "test_reported_metrics.py",
        ),
        (
            "Recall",
            fmt(numbers.get("recall")),
            "токены",
            "test (a3)",
            "metrics_mean_std.recall",
            "test_reported_metrics.py",
        ),
        (
            "TP / FP / FN / TN",
            f"{fmt(numbers.get('tp'))} / {fmt(numbers.get('fp'))} / {fmt(numbers.get('fn'))} / {fmt(numbers.get('tn'))}",
            "токены",
            "test (a3)",
            "metrics_mean_std",
            "test_reported_metrics.py",
        ),
        (
            "F1",
            fmt(numbers.get("answer_f1")),
            "ответы",
            "test (a3)",
            "metrics_mean_std.answer_f1",
            "test_reported_metrics.py",
        ),
        (
            "FPR",
            fmt(numbers.get("answer_fpr")),
            "ответы",
            "test (a3)",
            "metrics_mean_std.answer_fpr",
            "test_reported_metrics.py",
        ),
        (
            "F1",
            fmt(numbers.get("span_f1")),
            "фрагменты (IoU ≥ 0,5)",
            "test (a3)",
            "metrics_mean_std.span_f1",
            "test_reported_metrics.py",
        ),
        (
            "Статус критерия",
            f"**{fmt(criterion.get('status'))}**",
            "токены",
            "test (a3)",
            "criterion.status",
            "test_criterion_hf.py",
        ),
    ]
    lines += [
        "| Метрика | Значение | Уровень | Выборка | Файл-источник | Подтверждающий тест |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {name} | {value} | {level} | {sample} | `{field}` | `tests/{test}` |"
        for name, value, level, sample, field, test in rows
    ]
    lines += [
        "",
        f"Время прогона: {fmt(manifest.get('duration_s'))} с; железо: "
        f"{manifest.get('hardware', {}).get('cpu_model') or '—'} · "
        f"{fmt(manifest.get('hardware', {}).get('ram_total_gb'))} ГБ RAM · "
        f"torch {manifest.get('hardware', {}).get('torch_version')}, "
        f"transformers {manifest.get('hardware', {}).get('transformers_version')}.",
        f"Конфигурация: `{manifest.get('config', {}).get('name')}` "
        f"(sha256 `{str(manifest.get('config_sha256'))[:16]}…`), пороги и гиперпараметры — из CV по train; "
        f"отступления: {len(manifest.get('deviations') or [])} (см. docs/EXPERIMENTS.md).",
    ]
    lines.append(END)
    return "\n".join(lines)


def manifest_numbers(manifest: dict[str, Any]) -> set[str]:
    """Все числа, которые манифест разрешает писать в README (в формате таблицы).

    Форматов несколько, потому что таблица печатает шесть знаков, а в манифесте
    число может стоять и как ``144.0``, и как ``144``: сверять надо значение, а не
    способ его записи.
    """
    allowed: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, bool):
            return
        elif isinstance(node, float):
            allowed.add(f"{node:.6f}")
            allowed.add(f"{node:.4f}")
            if node.is_integer():
                allowed.add(str(int(node)))
            allowed.add(str(node))
        elif isinstance(node, int):
            allowed.add(str(node))

    walk(manifest)
    return allowed


def value_cells(block: str) -> list[str]:
    """Только колонка «Значение» таблицы: именно в неё подсовывают числа от руки.

    Проше и заголовки, и строки железа сверяются равенством блока с тем, что
    рендеряется из манифеста (см. ``check_or_write``), а перебор чисел оставлен
    для клеток метрик — там расхождение значит «число придумали».
    """
    cells: list[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or stripped.startswith("| ---"):
            continue
        parts = [item.strip() for item in stripped.strip("|").split("|")]
        if len(parts) >= 2 and parts[0] not in {"Метрика"}:
            cells.append(parts[1])
    return cells


def extract_block(text: str) -> str | None:
    match = re.search(re.escape(BEGIN) + r"(.*?)" + re.escape(END), text, flags=re.DOTALL)
    return match.group(0) if match else None


def check_or_write(target: Path, manifest_path: Path, corpus: str, write: bool) -> int:
    manifest = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    block = render(manifest, corpus)
    text = target.read_text(encoding="utf-8")
    current = extract_block(text)
    if write:
        if current is None:
            raise SystemExit(f"в {target} нет маркеров {BEGIN} … {END}: вставьте их вручную один раз")
        target.write_text(text.replace(current, block, 1), encoding="utf-8")
        print(f"блок обновлён: {target}")
        return 0
    if current is None:
        print(f"РАСХОЖДЕНИЕ: в {target} нет блока критерия (маркеры {BEGIN}/{END})", file=sys.stderr)
        return 1
    if manifest is None:
        if PLACEHOLDER_TEXT not in current:
            print(
                "РАСХОЖДЕНИЕ: манифеста reports/hf_final/manifest.json нет, а в блоке README уже стоят "
                "числа — до шага 3 там обязано быть «не измерено»",
                file=sys.stderr,
            )
            return 1
        print("блок критерия: измерение ещё не выполнено, в README стоит «не измерено» — расхождения нет")
        return 0
    if current.strip() != block.strip():
        print(
            f"РАСХОЖДЕНИЕ: блок в {target.name} не совпадает с тем, что даёт манифест.\n"
            "  Числа в этот блок вписывать руками нельзя — пересоберите: "
            f"python scripts/render_hf_table.py --readme {target} --write",
            file=sys.stderr,
        )
        return 1
    allowed = manifest_numbers(manifest)
    problems: list[str] = []
    for cell in value_cells(current):
        if PLACEHOLDER_TEXT in cell:
            continue
        for value in re.findall(r"-?\d+\.\d+|-?\d+", cell):
            if value in allowed:
                continue
            problems.append(f"{cell}: {value}")
    if problems:
        print(
            f"РАСХОЖДЕНИЕ: в блоке README есть числа, которых нет в манифесте: {sorted(set(problems))}",
            file=sys.stderr,
        )
        return 1
    required = []
    for key in ("f1", "fpr", "precision", "recall"):
        value = (manifest.get("metrics_mean_std") or {}).get(key)
        if isinstance(value, (int, float)):
            required.append(f"{float(value):.6f}")
    missing = [value for value in required if value not in current]
    if missing:
        print(
            f"РАСХОЖДЕНИЕ: в блоке README нет чисел из манифеста {missing} — пересоберите блок: "
            "python scripts/render_hf_table.py --write",
            file=sys.stderr,
        )
        return 1
    if "PASS" in current and str((manifest.get("criterion") or {}).get("status")) != "PASS":
        print("РАСХОЖДЕНИЕ: в README «PASS», а в манифесте статус другой", file=sys.stderr)
        return 1
    print("блок критерия в README сверен с манифестом: расхождений нет")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Таблица критерия hf из манифеста")
    parser.add_argument("--readme", default=str(REPO_ROOT / "README.md"))
    parser.add_argument("--manifest", default=str(ROOT / "reports" / "hf_final" / "manifest.json"))
    parser.add_argument("--corpus", default="a3")
    parser.add_argument("--stdout", action="store_true", help="напечатать блок вместо сверки")
    parser.add_argument("--write", action="store_true", help="переписать блок в README")
    args = parser.parse_args(argv)
    if args.stdout:
        manifest = None
        path = Path(args.manifest)
        if path.is_file():
            manifest = json.loads(path.read_text(encoding="utf-8"))
        print(render(manifest, args.corpus))
        return 0
    return check_or_write(Path(args.readme), Path(args.manifest), args.corpus, args.write)


if __name__ == "__main__":
    raise SystemExit(main())
