"""Напечатать ключевые числа METRICS.json аннотациями CI.

Зачем: артефакты из GitHub нельзя скачать в закрытом окружении, а аннотации
доступны через API. Скрипт публикует компактные строки, чтобы числа пилота и
кросс-корпусного теста были читаемы без выгрузки файлов.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Аннотации с числами METRICS.json")
    parser.add_argument("--metrics", default="reports/METRICS.json")
    args = parser.parse_args()
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))

    meta = metrics["meta"]
    tests = metrics["tests"]
    demo = metrics["demo"]
    print(
        "::notice title=метрики::"
        f"версия={meta['version']} коммит={meta['commit'][:12]} тестов={tests['collected']} "
        f"покрытие={tests['coverage_percent']}% "
        f"демо: F1={demo['in_corpus']['tokens']['f1']} FPR={demo['in_corpus']['tokens']['fpr']} "
        f"AUC={demo['validation']['auc']} "
        f"фрагменты: строгий={demo['in_corpus']['spans']['strict_f1']} "
        f"мягкий={demo['in_corpus']['spans']['soft_f1']} "
        f"участие_ИИ_AUC={demo['participation']['auc_out_of_fold']}"
    )

    cross = metrics.get("cross_corpus")
    if cross:
        print(
            "::notice title=кросс-корпус::"
            f"внутри={cross['in_corpus_f1']} кросс={cross['cross_corpus_f1']} "
            f"падение={cross['drop']} FPR_кросс={cross['cross_fpr']}"
        )

    pilot = metrics.get("pilot")
    if pilot:
        print(
            "::notice title=пилот::"
            f"статус={pilot['status']} опубликован={pilot['published']} пар={pilot['pairs']} "
            f"токенов={pilot['tokens']} контроль={json.dumps(pilot.get('control'), ensure_ascii=False)}"
        )
        for layer, features in (pilot.get("auc") or {}).items():
            summary = " ".join(
                f"{name}: auc={values['auc_oriented']} ci=[{values['ci'][0]};{values['ci'][1]}] "
                f"на_факт_токене={values['auc_fact_oriented']} "
                f"delta={values['delta_mean']} значимо={'да' if values['delta_significant'] else 'нет'}"
                for name, values in features.items()
            )
            print(f"::notice title=пилот, слой {layer}::{summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
