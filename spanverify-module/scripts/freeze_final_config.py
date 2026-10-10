#!/usr/bin/env python3
"""Шаг 3 готовится здесь: выбранная на val конфигурация → `config/hf_final_config.json` + запись в журнал.

Зачем отдельный скрипт, а не «дописать руками». Финальный прогон выполняется
ровно один раз, и всё, что он читает, обязано быть зафиксировано ДО запуска:
имя эксперимента, признаки, классификатор, гиперпараметры, окно, зазор
слияния фрагментов и — отдельно — число порога. Любая опечатка в этой цепочке
либо обесценивает измерение (конфигурация «подправлена» после чисел), либо
тратит единственный запуск. Поэтому файл конфигурации и запись в
``docs/EXPERIMENTS.md`` генерируются из свода шага 2 (``sweep.json``), а не
переписываются от руки, а повторяльный запуск скрипта отказывается дописывать
журнал второй раз.

Отбор конфигурации (заморожен шагом 0, ``docs/METRIC_SPEC.md``): максимум
``token_f1`` на val при ``token_fpr ≤ 0,40``; ties → меньший FPR. Если ни одна
конфигурация ограничение FPR не прошла, работает стоп-правило протокола: шаг 3
выполняется с максимальной ``token_f1`` на val, а отступление записывается в
``deviations`` — оно попадает и в манифест финала, и в ответ по задаче.

Порог берётся из той же записи ``sweep.json``: это число, подобранное групповой
CV по документам на обучающей части (seed 42), а не по val и тем более не по
test. Шаг 3 применяет его ко всем пяти seed'ам без переселекции — иначе
фиксировать до запуска было бы нечего.

Команда::

    python scripts/freeze_final_config.py \\
        --sweep reports/hf_protocol/sweep.json \\
        --primary a3 --run-id 38043189316
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from hf_protocol import CRITERION_F1, CRITERION_FPR, sha256_file  # noqa: E402

JOURNAL_MARKER = "### Шаг 3 — конфигурация зафиксирована (до запуска)"
THRESHOLD_LINE = "Порог, зафиксированный до шага 3"


def _ranking_key(item: dict[str, Any]) -> tuple[float, float]:
    """Порядок перебора: максимум F1(val), при равенстве — меньший FPR(val)."""
    selection = item.get("selection") or {}
    f1 = float(selection.get("token_f1_val") or 0.0)
    fpr = selection.get("token_fpr_val")
    return (f1, -(float(fpr) if fpr is not None else float("inf")))


def _splits_dir(corpus: str) -> str:
    """Каталог разбиения: имена каталогов корпусов исторически несимметричны."""
    return {"a3": "data/corpus_a3/splits", "a1": "data/corpus_a/splits"}.get(corpus, f"data/corpus_{corpus}/splits")


def choose_experiment(sweep: dict[str, Any]) -> tuple[dict[str, Any], list[str], str]:
    """Конфигурация для шага 3: (запись эксперимента, отступления, правило отбора)."""
    experiments = [item for item in (sweep.get("experiments") or []) if isinstance(item, dict)]
    if not experiments:
        raise SystemExit("в sweep.json нет записей экспериментов: шаг 2 не выполнен, фиксировать нечего")
    limit = int(sweep.get("limit") or 10)
    if len(experiments) > limit:
        raise SystemExit(f"экспериментов {len(experiments)} > лимита {limit}: нарушение шага 2")
    deviations: list[str] = []
    best_name = sweep.get("best")
    by_name = {str(item.get("name") or ""): item for item in experiments}
    if best_name and str(best_name) in by_name:
        chosen = by_name[str(best_name)]
        rule = str(sweep.get("rule") or f"максимум token_f1 на val при token_fpr ≤ {CRITERION_FPR}; ties → меньший FPR")
    else:
        # Ограничение FPR на val не выполнила ни одна конфигурация. Стоп-правило
        # протокола: идём на шаг 3 с лучшей по F1 и честно пишем «не достигнуто».
        ranked = sorted(experiments, key=_ranking_key, reverse=True)
        chosen = ranked[0]
        selection = chosen.get("selection") or {}
        deviations.append(
            "стоп-правило шага 2: за "
            f"{len(experiments)} эксперимент(а/ов) F1(val) ≥ {CRITERION_F1} при FPR(val) ≤ {CRITERION_FPR} "
            f"не достигнуто (лучшее: F1 {selection.get('token_f1_val')} при FPR {selection.get('token_fpr_val')}); "
            "шаг 3 выполняется с максимальной F1(val) конфигурацией, критерий в отчёте — «не достигнуто»"
        )
        rule = (
            "ни одна конфигурация не прошла ограничение FPR(val): выбрана максимальная token_f1 на val "
            "(стоп-правило протокола, см. docs/EXPERIMENTS.md)"
        )
    errors = {
        corpus: numbers.get("error")
        for corpus, numbers in (chosen.get("per_corpus") or {}).items()
        if isinstance(numbers, dict) and numbers.get("error")
    }
    if errors:
        raise SystemExit(f"выбранный эксперимент «{chosen.get('name')}» имеет ошибки: {errors} — фиксировать нельзя")
    selection = chosen.get("selection") or {}
    if selection.get("token_f1_val") is None:
        raise SystemExit(f"у выбранной конфигурации нет чисел на val: отбор был не по val ({chosen.get('name')})")
    if "test" in str(selection.get("corpus") or "").lower():
        raise SystemExit("отбор configuration шёл по test: протокол нарушен, фиксация запрещена")
    return chosen, deviations, rule


def build_config(
    sweep: dict[str, Any],
    chosen: dict[str, Any],
    deviations: list[str],
    rule: str,
    *,
    primary: str,
    seeds: list[int],
    run_id: str,
    sweep_path: Path,
    extra_deviations: list[str],
) -> dict[str, Any]:
    """Содержимое ``config/hf_final_config.json`` — ровно то, что читает шаг 3."""
    numbers = (chosen.get("per_corpus") or {}).get(primary) or {}
    threshold = numbers.get("threshold")
    if threshold is None:
        raise SystemExit(f"в записи эксперимента нет порога для корпуса {primary}: фиксировать нечего")
    tokens = numbers.get("tokens") or {}
    answers = numbers.get("answers") or {}
    spans = numbers.get("spans") or {}
    deviations = list(deviations) + [str(item) for item in extra_deviations if str(item).strip()]
    deviations.append(
        "порог зафиксирован числом и применён ко всем seed'ам шага 3 без переселекции "
        "(шаг 0 требует «порог — только на train, зафиксирован до шага 3»)"
    )
    return {
        "step": "final-config",
        "protocol": str(sweep.get("protocol") or "hf-criterion-v1"),
        "name": str(chosen.get("name")),
        "number": chosen.get("number"),
        "features": list(chosen.get("features") or []),
        "window": int(chosen.get("window") or 0),
        "merge_gap": int(chosen.get("merge_gap") or 2),
        "classifier": str(chosen.get("classifier") or "logreg"),
        "params": dict(chosen.get("params") or {}),
        "threshold": float(threshold),
        "threshold_source": (
            f"групповая CV по документам на обучающей части, seed {seeds[0]} (sweep.json, корпус {primary})"
        ),
        "selection_rule": rule,
        "selection": dict(chosen.get("selection") or {}),
        "val_numbers": {"tokens": tokens, "answers": answers, "spans": spans, "criterion": numbers.get("criterion")},
        "model": sweep.get("model") or {},
        "primary_corpus": primary,
        "splits_dir": _splits_dir(primary),
        "seeds": list(seeds),
        "deviations": deviations,
        "source": {
            "sweep_file": str(sweep_path),
            "sweep_sha256": sha256_file(sweep_path),
            "grid_config_sha256": sweep.get("grid_sha256") or "",
            "ci_run": str(run_id),
            "sweep_code_commit": sweep.get("code_commit") or "",
            "sweep_duration_s": sweep.get("duration_s"),
        },
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frozen_by": "scripts/freeze_final_config.py",
        "final_command": (
            "python scripts/hf_protocol.py final --config config/hf_final_config.json "
            f"--grid-cache reports/hf-grid --splits {_splits_dir(primary)} --corpus-name {primary} "
            f"--seeds {','.join(str(seed) for seed in seeds)} --include-val --out reports/hf_final"
        ),
    }


def journal_block(config: dict[str, Any], *, config_path: Path, config_sha: str, val_primary: str) -> str:
    """Запись журнала: что зафиксировано, откуда взялось и чем подтверждается."""
    tokens = (config.get("val_numbers") or {}).get("tokens") or {}
    answers = (config.get("val_numbers") or {}).get("answers") or {}
    spans = (config.get("val_numbers") or {}).get("spans") or {}
    lines = [
        JOURNAL_MARKER,
        "",
        f"- Конфигурация: `{config['name']}` (эксперимент №{config.get('number')} из 10), "
        f"классификатор `{config['classifier']}`, признаков {len(config['features'])}, "
        f"окно {config['window']}, зазор слияния {config['merge_gap']}.",
        f"- Правило отбора: {config['selection_rule']}.",
        f"- {THRESHOLD_LINE}: {float(config['threshold']):.6f} — источник: {config['threshold_source']}.",
        f"- Числа на val ({val_primary}, тот же порог): токены F1 {tokens.get('f1')} "
        f"P {tokens.get('precision')} R {tokens.get('recall')} FPR {tokens.get('fpr')} AUC {tokens.get('auc')}; "
        f"ответы F1 {answers.get('f1')} FPR {answers.get('fpr')}; фрагменты F1 {spans.get('f1')}.",
        f"- Seed'ы шага 3: {', '.join(str(seed) for seed in config['seeds'])} (зафиксированы до запуска, "
        "лучший не выбирается).",
        f"- Источник: `{config['source']['sweep_file']}` (sha256 `{config['source']['sweep_sha256']}`), "
        f"прогон CI `{config['source']['ci_run']}`, код `{config['source']['sweep_code_commit']}`.",
        f"- Файл конфигурации: `{config_path}` (sha256 `{config_sha}`).",
        "- Отступления: " + ("; ".join(config["deviations"]) if config["deviations"] else "нет"),
        f"- Команда шага 3: `{config['final_command']}`",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep", default="reports/hf_protocol/sweep.json", help="сводка шага 2")
    parser.add_argument("--config-out", default="config/hf_final_config.json")
    parser.add_argument("--journal", default="docs/EXPERIMENTS.md")
    parser.add_argument("--primary", default="a3", help="корпус, по которому шёл отбор")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--run-id", default="", help="номер прогона CI — для строки происхождения")
    parser.add_argument("--deviation", action="append", default=[], help="дополнительное отступление (повторяемо)")
    parser.add_argument("--dry-run", action="store_true", help="напечатать и ничего не писать")
    parser.add_argument("--force", action="store_true", help="разрешить перезапись уже существующей фиксации")
    args = parser.parse_args(argv)

    sweep_path = Path(args.sweep)
    if not sweep_path.is_absolute():
        sweep_path = ROOT / sweep_path
    if not sweep_path.is_file():
        raise SystemExit(f"нет {sweep_path}: шаг 2 не завершён, фиксировать нечего")
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    if str(sweep.get("step")) != "sweep":
        raise SystemExit(f"{sweep_path}: это не сводка шага 2 (step={sweep.get('step')!r})")

    chosen, deviations, rule = choose_experiment(sweep)
    seeds = [int(value) for value in str(args.seeds).split(",") if value.strip()]
    if len(seeds) != 5:
        raise SystemExit(f"протокол требует ровно 5 seed'ов, получено {len(seeds)}: {seeds}")
    config_path = Path(args.config_out)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if config_path.is_file() and not args.force:
        raise SystemExit(
            f"{config_path} уже существует: конфигурация зафиксирована ранее. "
            "Перезапись после чисел — это подгонка, для неё нужен явный --force."
        )
    journal_path = Path(args.journal)
    if not journal_path.is_absolute():
        journal_path = ROOT / journal_path
    journal_text = journal_path.read_text(encoding="utf-8") if journal_path.is_file() else ""
    if JOURNAL_MARKER in journal_text and not args.force:
        raise SystemExit(
            "в журнале уже есть запись фиксации шага 3: повторная фиксация запрещена (--force — только для черновика)"
        )
    val_primary = str((chosen.get("selection") or {}).get("corpus") or args.primary)
    config = build_config(
        sweep,
        chosen,
        deviations,
        rule,
        primary=val_primary,
        seeds=seeds,
        run_id=args.run_id or "не указан",
        sweep_path=sweep_path.relative_to(ROOT) if sweep_path.is_relative_to(ROOT) else sweep_path,
        extra_deviations=list(args.deviation),
    )
    body = json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    # Хеш считается по содержимому, а не по файлу: в --dry-run файла ещё нет,
    # а в журнале число обязано совпадать с тем, что потом проверит шаг 3.
    config_sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    block = journal_block(
        config,
        config_path=config_path.relative_to(ROOT) if config_path.is_relative_to(ROOT) else config_path,
        config_sha=config_sha,
        val_primary=val_primary,
    )
    if args.dry_run:
        print("=== config/hf_final_config.json ===")
        print(body)
        print("=== запись в журнал ===")
        print(block)
        return 0
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(body, encoding="utf-8")
    # sha256 конфигурации попадает в журнал, поэтому журнал пишется после файла.
    with journal_path.open("a", encoding="utf-8") as handle:
        handle.write("\n" + block)
    print(f"зафиксировано: {config_path} (sha256 {sha256_file(config_path)})")
    print(f"порог: {config['threshold']:.6f} | конфигурация: {config['name']} | признаков: {len(config['features'])}")
    print(f"журнал дополнен: {journal_path}")
    print("шаг 3: зафиксировать коммитом с меткой [hf-final] и прогнать workflow (см. docs/EXPERIMENTS.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
