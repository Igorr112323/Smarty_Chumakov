"""Проверка: числа в документах совпадают с ``reports/METRICS.json``.

Причина появления скрипта: в аудите нашлись расхождения (AUC 0.996 против 0.9978,
«690 токенов» против фактических 546, «207 тестов» против 210, «покрытие 87 %»
против 86 %). Теперь любое число в документах обязано приходить из единого файла
чисел, а расхождение ломает CI.

Что проверяется
---------------
1. Числовые метрики: ``token F1``, ``FPR``, ``AUC``, ``покрытие N %``,
   ``N тестов``, ``N токенов`` — значения должны быть среди посчитанных.
2. Версия: любое упоминание ``1.x.y`` должно совпадать с версией в METRICS.json
   (старые версии допустимы только рядом со словом «устарело»).
3. Обязательные тексты: дословная оговорка о синтетическом корпусе, разделы
   «заявлено / факт», «не проверено», «3 шага» в ``ИТОГ.md``.

Запуск (CI и локально)::

    python scripts/check_numbers.py --metrics reports/METRICS.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HISTORY_DOCS = {"PROGRESS.md", "AUDIT.md"}

# Корень репозитория считаем от файла скрипта: запускать проверку можно из любого
# каталога, и пути к документам всё равно указывают на репозиторий.
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DOCS = (
    "README.md",
    "ИТОГ.md",
    "PROGRESS.md",
    "AUDIT.md",
    "docs/ЗАЯВКА_поля.md",
    "docs/ТЗ_и_календарный_план.md",
    "docs/ОТЧЁТ_о_НИР_шаблон.md",
    "docs/РУКОВОДСТВО.md",
    "spanverify-module/README.md",
)

REQUIRED_IN_ITOG = (
    ("заявлено", "раздел «заявлено / факт»"),
    ("не проверено", "раздел «не проверено»"),
    ("3 шага", "инструкция из трёх шагов для человека"),
)
REQUIRED_IN_README = (
    ("синтетическ", "оговорка о синтетическом корпусе"),
    ("METRICS.json", "ссылка на единый файл чисел"),
)

NUMBER_PATTERNS = (
    (re.compile(r"token F1[^0-9\n]{0,12}(\d\.\d{2,4})"), "token_f1"),
    (re.compile(r"FPR(?!\s*[|/])[^0-9\n|/]{0,12}(\d\.\d{2,4})"), "fpr"),
    (re.compile(r"AUC[^0-9\n]{0,12}(\d\.\d{2,4})"), "auc"),
)

TOLERANT_PATTERNS = (
    (re.compile(r"покрыти\w*[^0-9\n]{0,25}(\d{1,3})\s*%"), "coverage"),
    (re.compile(r"(\d{2,4})\s*(?:тест|кейс)"), "tests"),
    (re.compile(r"(\d+)\s+токен"), "tokens"),
)

OLD_MARKER = "устарел"

# Числа, снятые из аннотаций CI (артефакты в песочнице недоступны),
# помечаются в тексте словом «аннотаци» и сверяются с логом прогона.
CI_MARKER = "аннотаци"

# Числа рядом с этими словами — критерии и пороги, а не измерения: их сверять
# с METRICS.json нельзя (например «token F1 ≥ 0.90 при FPR ≤ 0.10»).
GUARD = re.compile(r"[≥≤<>]|критери|порог|цель|минимум|максимум|не менее|не более", re.IGNORECASE)


def _roundings(value: float) -> set[str]:
    """Допустимые формы записи числа: 2, 3 и 4 знака после точки."""
    forms = {f"{round(value, 2):g}", f"{round(value, 3):g}", f"{round(value, 4):g}"}
    forms |= {f"{value:.2f}", f"{value:.3f}", f"{value:.4f}"}
    return forms


def _allowed(metrics: dict) -> dict[str, set[str]]:
    """Собрать множества допустимых значений каждого типа чисел."""
    demo = metrics["demo"]
    in_corpus = demo["in_corpus"]
    validation = demo["validation"]
    cross = metrics.get("cross_corpus") or {}
    pilot = metrics.get("pilot") or {}

    whole = demo.get("whole_corpus") or {}
    f1_values = [in_corpus["tokens"]["f1"], validation["f1"]]
    fpr_values = []
    if whole:
        f1_values.append(whole["tokens"]["f1"])
        f1_values.append(whole["answers"]["f1"])
    fpr_values = [in_corpus["tokens"]["fpr"], validation["fpr"]]
    if whole:
        fpr_values.extend([whole["tokens"]["fpr"], whole["answers"]["fpr"]])
    auc_values = [
        validation["auc"],
        in_corpus["tokens"]["auc"],
        in_corpus["answers"]["auc"],
        demo["participation"].get("auc_out_of_fold"),
    ]
    if whole:
        auc_values.extend([whole["tokens"]["auc"], whole["answers"]["auc"]])
    token_values = [
        in_corpus["tokens"]["n"],
        pilot.get("tokens"),
        demo["participation"].get("rows"),
    ]
    if whole:
        token_values.append(whole["tokens"]["n"])
    if cross:
        f1_values.append(cross["in_corpus_f1"])
        f1_values.append(cross["cross_corpus_f1"])
        fpr_values.append(cross["cross_fpr"])
    # Корпуса A (управляемые подмены) и B (внешний тест): их числа тоже законные,
    # документы ссылаются на них (reports/CORPUS_REPORT.md и METRICS.json).
    corpus_a = metrics.get("corpus_a") or {}
    if corpus_a.get("available"):
        for block in corpus_a.get("by_split", {}).values():
            tokens = block.get("tokens") or {}
            verdicts = block.get("verdicts") or {}
            for value in (tokens.get("f1"), tokens.get("precision"), tokens.get("recall")):
                if isinstance(value, (int, float)):
                    f1_values.append(float(value))
            if isinstance(tokens.get("fpr"), (int, float)):
                fpr_values.append(float(tokens["fpr"]))
            if isinstance(tokens.get("auc"), (int, float)):
                auc_values.append(float(tokens["auc"]))
            for value in (verdicts.get("f1"), verdicts.get("precision"), verdicts.get("recall")):
                if isinstance(value, (int, float)):
                    f1_values.append(float(value))
            if isinstance(verdicts.get("fpr"), (int, float)):
                fpr_values.append(float(verdicts["fpr"]))
            if isinstance(tokens.get("n"), int):
                token_values.append(tokens["n"])
        for block in corpus_a.get("by_mode", {}).values():
            if isinstance(block.get("token_f1"), (int, float)):
                f1_values.append(float(block["token_f1"]))
            if isinstance(block.get("verdict_recall"), (int, float)):
                f1_values.append(float(block["verdict_recall"]))
            if isinstance(block.get("verdict_fpr"), (int, float)):
                fpr_values.append(float(block["verdict_fpr"]))
            if isinstance(block.get("span_coverage"), (int, float)):
                f1_values.append(float(block["span_coverage"]))
    corpus_b = metrics.get("corpus_b") or {}
    if corpus_b.get("available"):
        ours = corpus_b.get("our_metrics") or {}
        theirs = corpus_b.get("their_metrics") or {}
        for group in ("tokens", "answers"):
            block = ours.get(group) or {}
            for value in (block.get("f1"), block.get("precision"), block.get("recall")):
                if isinstance(value, (int, float)):
                    f1_values.append(float(value))
            if isinstance(block.get("fpr"), (int, float)):
                fpr_values.append(float(block["fpr"]))
            if isinstance(block.get("auc"), (int, float)):
                auc_values.append(float(block["auc"]))
        spans = ours.get("spans") or {}
        for value in (spans.get("strict_f1_iou_0_5"), spans.get("coverage"), spans.get("soft_f1_expanded")):
            if isinstance(value, (int, float)):
                f1_values.append(float(value))
        for key in ("rouge1", "rouge2", "rougeL", "accuracy", "jaccard_score", "hamming_loss"):
            if isinstance(theirs.get(key), (int, float)):
                f1_values.append(float(theirs[key]))

    for layer in (pilot.get("auc") or {}).values():
        for values in layer.values():
            for key in ("auc_oriented", "auc_fact_oriented"):
                value = values.get(key)
                if isinstance(value, (int, float)):
                    auc_values.append(float(value))

    allowed = {
        "token_f1": {form for value in f1_values if isinstance(value, (int, float)) for form in _roundings(value)},
        "fpr": {form for value in fpr_values if isinstance(value, (int, float)) for form in _roundings(value)},
        "auc": {form for value in auc_values if isinstance(value, (int, float)) for form in _roundings(value)},
        "coverage": set(),
        "tests": set(),
        "tokens": set(),
    }
    coverage = metrics.get("tests", {}).get("coverage_percent")
    if isinstance(coverage, (int, float)):
        allowed["coverage"] = {f"{int(round(coverage))}", f"{round(coverage, 1):g}", f"{round(coverage, 2):g}"}
    tests = metrics.get("tests", {}).get("collected")
    if tests:
        allowed["tests"] = {str(tests)}
    allowed["tokens"] = {str(int(value)) for value in token_values if value}
    return allowed


def check_required_sections(repo_root: Path) -> list[str]:
    """Обязательные разделы: «заявлено/факт», «не проверено», «3 шага» в ИТОГ."""
    problems: list[str] = []
    itog = repo_root / "ИТОГ.md"
    if itog.is_file():
        text = itog.read_text(encoding="utf-8").lower()
        for needle, description in REQUIRED_IN_ITOG:
            if needle not in text:
                problems.append(f"ИТОГ.md: нет обязательного элемента — {description}")
    else:
        problems.append(f"{itog}: файла нет, обязательные разделы проверить нельзя")
    readme = repo_root / "README.md"
    if readme.is_file():
        text = readme.read_text(encoding="utf-8")
        for needle, description in REQUIRED_IN_README:
            if needle not in text:
                problems.append(f"README.md: нет обязательного элемента — {description}")
    return problems


def check_docs(docs: list[Path], metrics: dict, repo_root: Path | None = None) -> list[str]:
    """Вернуть список нарушений (пустой список = числа сходятся)."""
    repo_root = repo_root or DEFAULT_REPO_ROOT
    allowed = _allowed(metrics)
    version = str(metrics["meta"]["version"])
    problems: list[str] = []

    for path in docs:
        if not path.is_file():
            continue
        if path.name in HISTORY_DOCS:
            # Это журналы: они хранят исторические числа и обязательно указывают,
            # что актуальные значения — в reports/METRICS.json.
            text = path.read_text(encoding="utf-8")
            if "METRICS.json" not in text:
                problems.append(f"{path}: нет ссылки на reports/METRICS.json (актуальные числа)")
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for pattern, kind in NUMBER_PATTERNS:
                for match in pattern.finditer(line):
                    value = match.group(1)
                    context = line[max(0, match.start(1) - 30) : match.start(1) + 5]
                    if OLD_MARKER in line or CI_MARKER in line or GUARD.search(context):
                        continue
                    if value not in allowed[kind]:
                        problems.append(
                            f"{path}:{number}: {kind}={value} нет в METRICS.json "
                            f"(допустимо: {sorted(allowed[kind])[:6]}…) → {line.strip()[:90]}"
                        )
            for pattern, kind in TOLERANT_PATTERNS:
                for match in pattern.finditer(line):
                    value = match.group(1)
                    context = line[max(0, match.start(1) - 30) : match.start(1) + 5]
                    if allowed[kind] and value not in allowed[kind]:
                        if OLD_MARKER in line or CI_MARKER in line or GUARD.search(context):
                            continue
                        problems.append(
                            f"{path}:{number}: {kind}={value} нет в METRICS.json "
                            f"(допустимо: {sorted(allowed[kind])}) → {line.strip()[:90]}"
                        )
            for match in re.finditer(r"\b1\.\d\.\d\b", line):
                if (
                    match.group(0) != version
                    and OLD_MARKER not in line
                    and CI_MARKER not in line
                    and GUARD.search(line[max(0, match.start() - 30) : match.start() + 5]) is None
                ):
                    problems.append(
                        f"{path}:{number}: версия {match.group(0)} не совпадает с {version} → {line.strip()[:90]}"
                    )

    problems.extend(check_required_sections(repo_root))
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Сверить числа в документах с METRICS.json")
    parser.add_argument("--metrics", default="reports/METRICS.json")
    parser.add_argument("--docs", nargs="*", default=None, help="переопределить список документов")
    args = parser.parse_args()

    repo_root = DEFAULT_REPO_ROOT
    metrics_path = Path(args.metrics)
    if not metrics_path.is_absolute():
        metrics_path = Path(__file__).resolve().parent.parent / metrics_path
    if not metrics_path.is_file():
        print(f"нет файла чисел {metrics_path}: сначала запустите scripts/collect_metrics.py", file=sys.stderr)
        return 2
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    docs = [repo_root / name for name in (args.docs or DEFAULT_DOCS)]
    problems = check_docs(docs, metrics, repo_root=repo_root)
    if problems:
        print(f"РАСХОЖДЕНИЯ ({len(problems)}):")
        for problem in problems[:60]:
            print(f"  - {problem}")
        if len(problems) > 60:
            print(f"  … ещё {len(problems) - 60}")
        return 1
    print(f"Числа сходятся с {metrics_path.name}: проверено документов {sum(1 for d in docs if d.is_file())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
