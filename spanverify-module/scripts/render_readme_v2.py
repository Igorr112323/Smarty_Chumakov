#!/usr/bin/env python3
"""Блок README про цикл v2: генерируется только из `reports/v2/manifest.json`.

Числа в README не пишутся руками нигде (то же правило, что и для блока v1): этот
скрипт умеет и записывать блок, и сверять его с манифестом (`--check`). Если
манифеста нет, блок в README быть не должен — иначе «результат» появляется раньше
измерения.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

BEGIN = "<!-- HF-CRITERION-V2:BEGIN -->"
END = "<!-- HF-CRITERION-V2:END -->"


def number(value: Any) -> str:
    """Русская десятичная запятая и шесть знаков — как в манифесте, без доокруглений."""
    if value is None:
        return "не вычислено"
    return f"{float(value):.6f}".replace(".", ",")


def render(manifest: dict[str, Any]) -> str:
    aggregate = manifest.get("metrics_mean_std") or {}
    internal = manifest.get("goal_internal") or {}
    contract = manifest.get("goal_contract_min") or {}
    pairs = manifest.get("pairs") or {}
    documents = manifest.get("documents") or {}
    goals = (manifest.get("config") or {}).get("goals") or {}
    internal_min = (goals.get("internal") or {}).get("token_f1_min", 0.70)
    contract_min = (goals.get("contract_min") or {}).get("token_f1_min", 0.60)
    lines = [
        BEGIN,
        "### Цикл v2: измерение на независимой выборке",
        "",
        "Числа взяты из `reports/v2/manifest.json` и восстановимы из `predictions_test.jsonl.gz`",
        "(команда пересчёта — `make hf-verify-v2`). Порог и конфигурация заморожены до прогона.",
        "",
        "| цель | F1 (токены) | FPR (токены) | статус |",
        "|---|---|---|---|",
        f"| внутренняя цель (F1 ≥ {number(internal_min)}) | {number(aggregate.get('f1'))} "
        f"± {number(aggregate.get('f1_std'))} | {number(aggregate.get('fpr'))} | {internal.get('status')} |",
        f"| контрактный минимум (F1 ≥ {number(contract_min)}) | {number(aggregate.get('f1'))} "
        f"± {number(aggregate.get('f1_std'))} | {number(aggregate.get('fpr'))} | {contract.get('status')} |",
        "",
        f"- Уровень ответов: F1 {number(aggregate.get('answer_f1'))}, FPR {number(aggregate.get('answer_fpr'))}. "
        f"Уровень фрагментов (IoU ≥ 0,5): F1 {number(aggregate.get('span_f1'))}.",
        f"- AUC (справочно): {number(aggregate.get('auc'))}. Seed'ов: {len(manifest.get('seeds') or [])} "
        f"({', '.join(str(seed) for seed in manifest.get('seeds') or [])}), std(F1) = {number(aggregate.get('f1_std'))}.",
        f"- Выборка: пар train/val/test {pairs.get('train')}/{pairs.get('val')}/{pairs.get('test')}, "
        f"документов {documents.get('train')}/{documents.get('val')}/{documents.get('test')}, "
        f"строк предсказаний {manifest.get('prediction_rows')}.",
        f"- Прогон: `{manifest.get('code_commit')}` , модель `{(manifest.get('model') or {}).get('id')}`, "
        f"время {manifest.get('duration_s')} с, лимит 16 ГБ CPU.",
        f"- Порог: {number(manifest.get('threshold'))} (правило {manifest.get('threshold_rule')} — выбран на val "
        "до шага 3, по test не переселектировался).",
        END,
    ]
    return "\n".join(lines)


def patch_readme(text: str, block: str) -> str:
    if BEGIN in text and END in text:
        head, rest = text.split(BEGIN, 1)
        _old, tail = rest.split(END, 1)
        return head + block + tail
    return text.rstrip() + "\n\n" + block + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Блок README с числами цикла v2")
    parser.add_argument("--manifest", default="reports/v2/manifest.json")
    parser.add_argument("--readme", default="../README.md")
    parser.add_argument("--check", action="store_true", help="только сверить, ничего не записывая")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    readme_path = Path(args.readme)
    if not manifest_path.is_file():
        text = readme_path.read_text(encoding="utf-8") if readme_path.is_file() else ""
        if BEGIN in text:
            print(f"ПРОБЛЕМА: манифеста {manifest_path} нет, а блок v2 в README есть — числа раньше измерения")
            return 1
        print(f"манифеста {manifest_path} нет — блока v2 в README быть не должно (и его нет)")
        return 0
    block = render(json.loads(manifest_path.read_text(encoding="utf-8")))
    text = readme_path.read_text(encoding="utf-8")
    current = ""
    if BEGIN in text and END in text:
        current = text.split(BEGIN, 1)[1].split(END, 1)[0]
    if args.check:
        if current.strip() != block.replace(BEGIN, "").replace(END, "").strip():
            print("ПРОБЛЕМА: блок HF-CRITERION-V2 в README не совпадает с манифестом")
            print("--- в README ---")
            print(current.strip()[:600])
            print("--- из манифеста ---")
            print(block.replace(BEGIN, "").replace(END, "").strip()[:600])
            return 1
        print("README: блок цикла v2 совпадает с манифестом")
        return 0
    readme_path.write_text(patch_readme(text, block), encoding="utf-8")
    print(f"блок записан в {readme_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
