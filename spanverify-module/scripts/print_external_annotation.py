#!/usr/bin/env python3
"""Аннотации CI с числами внешних наборов и проверкой контрольных значений.

Зачем: артефакты GitHub нельзя скачать в закрытом окружении, а аннотации доступны
через API. Поэтому и контрольные числа загрузки, и результаты оценки публикуются
короткими строками — их можно прочитать без выгрузки файлов.

Скрипт ничего не считает сам: он читает ``data/external/MANIFEST.json`` (что
скачано и сверено) и ``reports/external_tests.json`` (что измерено) и печатает
строки формата ``::notice``, которые GitHub показывает в интерфейсе.

Запуск: python scripts/print_external_annotation.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Аннотации с числами внешних наборов")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "external" / "MANIFEST.json")
    parser.add_argument("--combined", type=Path, default=ROOT / "reports" / "external_tests.json")
    args = parser.parse_args()

    if args.manifest.is_file():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        for name, block in (manifest.get("datasets") or {}).items():
            totals = block.get("totals") or {}
            revision = str(block.get("revision", ""))[:12]
            verified = all(info.get("verified") for info in (block.get("files") or {}).values())
            if name == "ragtruth":
                print(
                    "::notice title=внешний набор RAGTruth::"
                    f"ревизия={revision} ответов={totals.get('responses')} "
                    f"источников={totals.get('sources')} спанов={totals.get('spans')} "
                    f"смещения_совпали={totals.get('verified_spans')} "
                    f"test_QA=875/150/160/235 sha256={'совпали' if verified else 'РАСХОЖДЕНИЕ'}"
                )
            elif name == "rushallu":
                print(
                    "::notice title=внешний набор RusHallu-RAG::"
                    f"ревизия={revision} пар={totals.get('responses')} чистых={totals.get('clean')} "
                    f"с_галлюцинациями={totals.get('with_hallucination')} спанов={totals.get('spans')} "
                    f"sha256={'совпали' if verified else 'РАСХОЖДЕНИЕ'}"
                )
    else:
        print("::warning title=внешние наборы::манифест не найден — загрузка не выполнялась")

    if args.combined.is_file():
        combined = json.loads(args.combined.read_text(encoding="utf-8"))
        for key, run in (combined.get("runs") or {}).items():
            tokens = run["our_metrics"]["tokens"]
            theirs = run["their_metrics"]
            origins = "/".join(sorted(run.get("label_origin") or {}))
            print(
                f"::notice title=оценка {key}::"
                f"пар={run['pairs']} разметка={origins} режим={run['mode']} "
                f"token_F1={tokens['f1']} FPR={tokens['fpr']} AUC={tokens['auc']} "
                f"их_accuracy={theirs.get('accuracy')} их_ROUGE_L={theirs.get('rougeL')}"
            )
    else:
        print("::warning title=оценка внешних наборов::файл прогонов не найден")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
