"""Сборка единого реестра чисел проекта — ``reports/NUMBERS.json``.

Правило проекта: любое число в документах обязано иметь источник — прогон,
записанный в JSON. Реестр не считает ничего сам: он только переносит значения из
артефактов прогонов, сохраняя происхождение (файл, путь внутри файла, команда,
коммит, seed, режим, набор данных, дата).

Источники:

* ``spanverify-module/reports/METRICS.json`` — реестр модуля (числа прогонов CI
  и локальных прогонов корпусов);
* ``reports/contract/*.json`` — артефакты контрактных проверок
  (``scripts/contract_check.py``);
* ``spanverify-module/reports/*.json`` — отдельные отчёты (аудит знаков,
  нормировка плотности, кросс-корпусные прогоны и т. д.).

Что нельзя делать: придумывать значение, если артефакта нет. Тогда в реестре
``value: null`` и ``note`` с причиной — так «не измерено» остаётся честным.

Запуск::

    python scripts/build_numbers.py            # собрать реестр
    python scripts/build_numbers.py --check    # проверить, что реестр свежий
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "spanverify-module"
NUMBERS_PATH = ROOT / "reports" / "NUMBERS.json"

#: Пороговые значения договора. Это НЕ измерения, а требования: переносим их в
#: реестр как ``kind: "contract"``, чтобы документы ссылались на них так же, как
#: на измеренные числа, и их нельзя было случайно «улучшить».
CONTRACT_VALUES: list[dict] = [
    {
        "id": "contract.max_text_chars",
        "value": 2_000_000,
        "unit": "знаков",
        "requirement": "ТЗ п. 4.1 треб. 1 — максимальная длина обрабатываемого текста",
        "source": "spanverify-module/spanverify/limits.py",
    },
    {
        "id": "contract.target_ms_per_1000_chars",
        "value": 30.0,
        "unit": "мс/1000 знаков",
        "requirement": "ТЗ п. 4.1 треб. 7 — быстродействие",
        "source": "spanverify-module/spanverify/limits.py",
    },
    {
        "id": "contract.target_seconds_2m",
        "value": 60.0,
        "unit": "с",
        "requirement": "ТЗ п. 4.1 треб. 7 — предельный текст 2 000 000 знаков",
        "source": "spanverify-module/spanverify/limits.py",
    },
    {
        "id": "contract.gate_f1_min",
        "value": 0.50,
        "unit": "F1",
        "requirement": "ТЗ п. 4.1 треб. 6 — целевой ориентир F1",
        "source": "spanverify-module/spanverify/gates.py",
    },
    {
        "id": "contract.gate_fpr_max",
        "value": 0.40,
        "unit": "FPR",
        "requirement": "ТЗ п. 4.1 треб. 6 — целевой ориентир FPR",
        "source": "spanverify-module/spanverify/gates.py",
    },
    {
        "id": "contract.iou_threshold",
        "value": 0.5,
        "unit": "IoU",
        "requirement": "ТЗ п. 4.1 треб. 5 — совпадение фрагмента при пересечении границ",
        "source": "spanverify-module/spanverify/engine.py",
    },
    {
        "id": "contract.default_port",
        "value": 8765,
        "unit": "порт",
        "requirement": "ТЗ п. 4.2 — локальный веб-интерфейс на localhost:8765",
        "source": "spanverify-module/spanverify/server.py",
    },
    {
        "id": "contract.exe_size_max_mb",
        "value": 9.0,
        "unit": "МБ",
        "requirement": "ТЗ п. 4.2 — поставка одним исполняемым файлом",
        "source": "договор, п. 4.2",
    },
    {
        "id": "contract.package_size_max_mb",
        "value": 0.6,
        "unit": "МБ",
        "requirement": "ТЗ п. 4.2 — переносимый пакет",
        "source": "договор, п. 4.2",
    },
    {
        "id": "contract.storage_without_weights_mb",
        "value": 100.0,
        "unit": "МБ",
        "requirement": "ТЗ п. 4.2 — хранение без весов модели",
        "source": "договор, п. 4.2",
    },
    {
        "id": "contract.storage_with_weights_mb",
        "value": 1500.0,
        "unit": "МБ",
        "requirement": "ТЗ п. 4.2 — хранение с весами модели",
        "source": "договор, п. 4.2",
    },
    {
        "id": "contract.participation_mix_step",
        "value": 0.25,
        "unit": "доля",
        "requirement": "ТЗ п. 4.1 треб. 4 — калибровка на наборе с известной долей смеси, шаг 0.25",
        "source": "договор, п. 4.1 треб. 4",
    },
    {
        "id": "contract.seconds_50k",
        "value": 1.0,
        "unit": "с",
        "requirement": "ТЗ п. 4.2 (стр. 13) — документ 50 000 знаков обрабатывается не более чем за 1 с",
        "source": "договор, п. 4.2",
    },
    {
        "id": "contract.features_per_token",
        "value": 5,
        "unit": "признаков",
        "requirement": "ТЗ п. 4.1 треб. 3 — пять числовых признаков на токен",
        "source": "spanverify-module/spanverify/features.py",
    },
    {
        "id": "contract.corpus_pairs",
        "value": 1200,
        "unit": "пар",
        "requirement": "ТЗ п. 4.1 треб. 6 — отложенная по документам часть набора из 1200 пар",
        "source": "spanverify-module/data/corpus_a/pairs.jsonl",
    },
    {
        "id": "contract.hf_libraries_mb",
        "value": 500.0,
        "unit": "МБ",
        "requirement": "ТЗ п. 4.2 (стр. 12) — библиотеки Python режима признаков около 500 МБ",
        "source": "договор, п. 4.2",
    },
    {
        "id": "contract.hf_weights_mb",
        "value": 530.0,
        "unit": "МБ",
        "requirement": "ТЗ п. 4.2 (стр. 12) — файл весов модели около 530 МБ",
        "source": "договор, п. 4.2",
    },
    {
        "id": "contract.cores_min",
        "value": 2,
        "unit": "ядер",
        "requirement": "ТЗ п. 4.2 (стр. 12) — процессор x86-64, число ядер не менее двух",
        "source": "договор, п. 4.2",
    },
    {
        "id": "contract.ram_gb",
        "value": 4,
        "unit": "ГБ",
        "requirement": "ТЗ п. 4.2 (стр. 12) — оперативная память не менее 4 ГБ",
        "source": "договор, п. 4.2",
    },
    {
        "id": "contract.ram_gb_with_weights",
        "value": 8,
        "unit": "ГБ",
        "requirement": "ТЗ п. 4.2 (стр. 12) — для режима с весами модели не менее 8 ГБ",
        "source": "договор, п. 4.2",
    },
    {
        "id": "contract.disk_gb",
        "value": 1.5,
        "unit": "ГБ",
        "requirement": "ТЗ п. 4.2 (стр. 12) — свободное место на диске не менее 1,5 ГБ",
        "source": "договор, п. 4.2",
    },
    {
        "id": "contract.stages",
        "value": 2,
        "unit": "этапа",
        "requirement": "календарный план (Приложение № 4) — два этапа",
        "source": "договор, Приложение № 4",
    },
    {
        "id": "contract.months_per_stage",
        "value": 6,
        "unit": "месяцев",
        "requirement": "календарный план (Приложение № 4) — срок каждого этапа 6 месяцев",
        "source": "договор, Приложение № 4",
    },
    {
        "id": "contract.months_total",
        "value": 12,
        "unit": "месяцев",
        "requirement": "ТЗ п. 8 и календарный план — срок выполнения работ 12 месяцев",
        "source": "договор, Приложение № 4",
    },
    {
        "id": "contract.stage1_rub",
        "value": 200000,
        "unit": "руб.",
        "requirement": "календарный план (Приложение № 4) — сумма расходов первого этапа",
        "source": "договор, Приложение № 4",
    },
    {
        "id": "contract.stage2_rub",
        "value": 300000,
        "unit": "руб.",
        "requirement": "календарный план (Приложение № 4) — сумма расходов второго этапа",
        "source": "договор, Приложение № 4",
    },
    {
        "id": "contract.total_rub",
        "value": 500000,
        "unit": "руб.",
        "requirement": "п. 3.1 договора и календарный план — общая сумма гранта",
        "source": "договор, п. 3.1",
    },
    {
        "id": "contract.reporting_days_before_stage_end",
        "value": 20,
        "unit": "календарных дней",
        "requirement": "п. 6.1 договора — отчётность не позднее чем за 20 календарных дней до окончания этапа",
        "source": "договор, п. 6.1",
    },
    {
        "id": "contract.originality_percent_min",
        "value": 80,
        "unit": "%",
        "requirement": "Приложение № 10, п. 4 — оригинальность текста отчёта не менее 80 %",
        "source": "договор, Приложение № 10",
    },
    {
        "id": "contract.keywords_min",
        "value": 5,
        "unit": "слов",
        "requirement": "Приложение № 10, п. 2.2 — перечень ключевых слов от 5 до 15",
        "source": "договор, Приложение № 10",
    },
    {
        "id": "contract.keywords_max",
        "value": 15,
        "unit": "слов",
        "requirement": "Приложение № 10, п. 2.2 — перечень ключевых слов от 5 до 15",
        "source": "договор, Приложение № 10",
    },
    {
        "id": "contract.report_elements",
        "value": 10,
        "unit": "элементов",
        "requirement": "Приложение № 10, п. 1 — структурные элементы отчёта",
        "source": "docs/ОТЧЁТ_о_НИР_шаблон.md",
    },
    {
        "id": "contract.business_plan_pages_min",
        "value": 20,
        "unit": "страниц",
        "requirement": "Приложение № 11 — объём бизнес-плана не менее 20 страниц (зона получателя гранта)",
        "source": "договор, Приложение № 11",
    },
]

#: Факты об источнике требований: не измерение и не требование, а привязка версии.
#: Подписанный PDF в публичном репозитории не хранится (персональные данные),
#: поэтому фиксируется его SHA-256 и размер, а сами данные не переносятся.
SOURCE_VALUES: list[dict] = [
    {
        "id": "contract.source.pdf_sha256",
        "value": "e5eb6ad8522fa70135f75bf22b0b5f3464fcc17d9c6f04d81b48976a42e69cd5",
        "unit": "SHA-256",
        "source": "docs/ДОГОВОР_ИЗВЛЕЧЕНИЕ.md",
        "requirement": "подписанный договор № 20478ГУ/2026 (проект 117812), файл Договор_0117812 (25).pdf",
    },
    {
        "id": "contract.source.pdf_bytes",
        "value": 5019351,
        "unit": "байт",
        "source": "docs/ДОГОВОР_ИЗВЛЕЧЕНИЕ.md",
        "requirement": "размер подписанного договора (32 страницы)",
    },
]

#: Что переносим из артефактов: (id, файл, путь внутри файла, единица, описание).
MEASURED_SOURCES: list[dict] = [
    # --- объёмы режима hf (измеряет CI-задание hf-footprint) ---
    {
        "id": "hf_footprint.libs_mb",
        "source": "reports/contract/4.2.5-hf-footprint.json",
        "path": ["metrics", "libs_mb"],
        "unit": "МБ",
        "dataset": "hf/rugpt3small",
        "mode": "hf",
    },
    {
        "id": "hf_footprint.weights_mb",
        "source": "reports/contract/4.2.5-hf-footprint.json",
        "path": ["metrics", "weights_mb"],
        "unit": "МБ",
        "dataset": "hf/rugpt3small",
        "mode": "hf",
    },
    {
        "id": "hf_footprint.storage_with_weights_mb",
        "source": "reports/contract/4.2.6-storage.json",
        "path": ["metrics", "storage_with_weights_mb"],
        "unit": "МБ",
        "dataset": "hf/rugpt3small",
        "mode": "hf",
    },
    # --- размеры собранной поставки (записывает задание build-and-release-exe) ---
    {
        "id": "release.exe_bytes",
        "source": "reports/release_sizes.json",
        "path": ["exe_bytes"],
        "unit": "байт",
        "dataset": "spanverify.exe",
        "mode": "release",
    },
    {
        "id": "release.pyz_bytes",
        "source": "reports/release_sizes.json",
        "path": ["pyz_bytes"],
        "unit": "байт",
        "dataset": "spanverify.pyz",
        "mode": "release",
    },

    # --- прогон на отложенной части корпуса A (договорный ориентир) ---
    {
        "id": "corpus_a.test.tokens.f1",
        "source": "spanverify-module/reports/METRICS.json",
        "path": ["corpus_a", "by_split", "test", "tokens", "f1"],
        "unit": "F1",
        "dataset": "corpus_a1_synthetic",
        "mode": "demo",
        "gate_of": "contract.gate_f1_min",
    },
    {
        "id": "corpus_a.test.tokens.fpr",
        "source": "spanverify-module/reports/METRICS.json",
        "path": ["corpus_a", "by_split", "test", "tokens", "fpr"],
        "unit": "FPR",
        "dataset": "corpus_a1_synthetic",
        "mode": "demo",
        "gate_of": "contract.gate_fpr_max",
    },
    {
        "id": "corpus_a.test.tokens.auc",
        "source": "spanverify-module/reports/METRICS.json",
        "path": ["corpus_a", "by_split", "test", "tokens", "auc"],
        "unit": "AUC",
        "dataset": "corpus_a1_synthetic",
        "mode": "demo",
    },
    {
        "id": "corpus_a.test.spans.strict_f1",
        "source": "spanverify-module/reports/METRICS.json",
        "path": ["corpus_a", "by_split", "test", "spans", "strict_f1"],
        "unit": "F1",
        "dataset": "corpus_a1_synthetic",
        "mode": "demo",
    },
    {
        "id": "corpus_a.test.answers.f1",
        "source": "spanverify-module/reports/METRICS.json",
        "path": ["corpus_a", "by_split", "test", "answers", "f1"],
        "unit": "F1",
        "dataset": "corpus_a1_synthetic",
        "mode": "demo",
    },
    {
        "id": "corpus_a.pairs",
        "source": "spanverify-module/reports/METRICS.json",
        "path": ["corpus_a", "pairs"],
        "unit": "пар",
        "dataset": "corpus_a1_synthetic",
        "mode": "demo",
    },
    {
        "id": "corpus_a.shared_groups",
        "source": "spanverify-module/reports/METRICS.json",
        "path": ["corpus_a", "shared_groups"],
        "unit": "групп",
        "dataset": "corpus_a1_synthetic",
        "mode": "demo",
    },
    # --- реальные документы (демо-режим) ---
    {
        "id": "corpus_a3_real.test.tokens.f1",
        "source": "spanverify-module/reports/METRICS.json",
        "path": ["corpus_a3_real", "by_split", "test", "tokens", "f1"],
        "unit": "F1",
        "dataset": "corpus_a3_real",
        "mode": "demo",
    },
    {
        "id": "corpus_a3_real.test.tokens.fpr",
        "source": "spanverify-module/reports/METRICS.json",
        "path": ["corpus_a3_real", "by_split", "test", "tokens", "fpr"],
        "unit": "FPR",
        "dataset": "corpus_a3_real",
        "mode": "demo",
    },
    # --- демонстрационный корпус (синтетика) ---
    {
        "id": "demo.validation.f1",
        "source": "spanverify-module/reports/METRICS.json",
        "path": ["demo", "validation", "f1"],
        "unit": "F1",
        "dataset": "demo",
        "mode": "demo",
    },
    {
        "id": "demo.validation.fpr",
        "source": "spanverify-module/reports/METRICS.json",
        "path": ["demo", "validation", "fpr"],
        "unit": "FPR",
        "dataset": "demo",
        "mode": "demo",
    },
    # --- аудит знаков признаков ---
    {
        "id": "sign_audit.mass.auc_as_risk",
        "source": "spanverify-module/reports/SIGN_AUDIT.json",
        "path": ["metrics", "per_feature", "ctx_attention_mass", "auc_as_risk"],
        "unit": "AUC",
        "dataset": "corpus_a1_synthetic/train",
        "mode": "demo",
    },
    {
        "id": "sign_audit.mass.auc_as_support",
        "source": "spanverify-module/reports/SIGN_AUDIT.json",
        "path": ["metrics", "per_feature", "ctx_attention_mass", "auc_as_support"],
        "unit": "AUC",
        "dataset": "corpus_a1_synthetic/train",
        "mode": "demo",
    },
    {
        "id": "sign_audit.density.auc_as_risk",
        "source": "spanverify-module/reports/SIGN_AUDIT.json",
        "path": ["metrics", "per_feature", "embedding_density", "auc_as_risk"],
        "unit": "AUC",
        "dataset": "corpus_a1_synthetic/train",
        "mode": "demo",
    },
    {
        "id": "sign_audit.density.auc_as_support",
        "source": "spanverify-module/reports/SIGN_AUDIT.json",
        "path": ["metrics", "per_feature", "embedding_density", "auc_as_support"],
        "unit": "AUC",
        "dataset": "corpus_a1_synthetic/train",
        "mode": "demo",
    },
    {
        "id": "density_reference.supported_median",
        "source": "spanverify-module/reports/DENSITY_REFERENCE.json",
        "path": ["metrics", "supported_median"],
        "unit": "схожесть",
        "dataset": "corpus_a1_synthetic/train",
        "mode": "demo",
    },
    {
        "id": "tests.collected",
        "source": "reports/contract/dod.tests.json",
        "path": ["metrics", "collected"],
        "unit": "тестов",
        "dataset": "spanverify-module/tests",
        "mode": "demo",
    },
    {
        "id": "tests.coverage_percent",
        "source": "reports/contract/dod.tests.json",
        "path": ["metrics", "coverage_percent"],
        "unit": "%",
        "dataset": "spanverify-module/spanverify",
        "mode": "demo",
    },
    # --- контрактные артефакты (заполняются прогоном contract_check.py) ---
    {
        "id": "performance.2000.ms_per_1000_chars",
        "source": "reports/contract/4.1.7-performance.json",
        "path": ["metrics", "by_size", "2000", "median_ms_per_1000_chars"],
        "unit": "мс/1000 знаков",
        "dataset": "synthetic-2000",
        "mode": "demo",
        "gate_of": "contract.target_ms_per_1000_chars",
    },
    {
        "id": "performance.50000.ms_per_1000_chars",
        "source": "reports/contract/4.1.7-performance.json",
        "path": ["metrics", "by_size", "50000", "median_ms_per_1000_chars"],
        "unit": "мс/1000 знаков",
        "dataset": "synthetic-50000",
        "mode": "demo",
        "gate_of": "contract.target_ms_per_1000_chars",
    },
    {
        "id": "performance.limit.seconds",
        "source": "reports/contract/4.1.7-performance.json",
        "path": ["metrics", "limit_run", "seconds"],
        "unit": "с",
        "dataset": "synthetic-2000000",
        "mode": "demo",
        "gate_of": "contract.target_seconds_2m",
    },
    {
        "id": "performance.limit.peak_memory_mb",
        "source": "reports/contract/4.1.7-performance.json",
        "path": ["metrics", "limit_run", "peak_memory_mb"],
        "unit": "МБ",
        "dataset": "synthetic-2000000",
        "mode": "demo",
    },
    {
        "id": "delivery.pyz.size_bytes",
        "source": "reports/contract/4.1.9-delivery-artifacts.json",
        "path": ["metrics", "pyz_size_bytes"],
        "unit": "байт",
        "dataset": "spanverify.pyz",
        "mode": "demo",
        "gate_of": "contract.package_size_max_mb",
    },
    {
        "id": "delivery.module.size_mb",
        "source": "reports/contract/4.1.9-delivery-artifacts.json",
        "path": ["metrics", "module_size_mb"],
        "unit": "МБ",
        "dataset": "spanverify-module",
        "mode": "demo",
        "gate_of": "contract.storage_without_weights_mb",
    },
    {
        "id": "participation.calibration.auc",
        "source": "reports/contract/4.1.4-ai-participation.json",
        "path": ["metrics", "auc_binary_at_0_5"],
        "unit": "AUC",
        "dataset": "synthetic-participation-corpus",
        "mode": "demo",
    },
    {
        "id": "participation.calibration.mae",
        "source": "reports/contract/4.1.4-ai-participation.json",
        "path": ["metrics", "mae"],
        "unit": "MAE",
        "dataset": "synthetic-participation-corpus",
        "mode": "demo",
    },
]


def git_commit(where: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=where,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        return "unknown"


def dig(payload: object, path: list[str]) -> object | None:
    node: object = payload
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:  # pragma: no cover - битый артефакт виден в отчёте
        return None


def build() -> dict:
    entries: list[dict] = []
    date = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    for spec in CONTRACT_VALUES:
        entries.append(
            {
                "id": spec["id"],
                "value": spec["value"],
                "unit": spec["unit"],
                "kind": "contract",
                "dataset": None,
                "mode": None,
                "seed": None,
                "commit": None,
                "command": None,
                "date": date,
                "source": spec["source"],
                "source_path": spec["requirement"],
                "gate": None,
                "note": "требование договора, не измерение",
            }
        )

    for spec in SOURCE_VALUES:
        entries.append(
            {
                "id": spec["id"],
                "value": spec["value"],
                "unit": spec["unit"],
                "kind": "source",
                "dataset": None,
                "mode": None,
                "seed": None,
                "commit": None,
                "command": None,
                "date": date,
                "source": spec["source"],
                "source_path": spec["requirement"],
                "gate": None,
                "note": "версия источника требований: зафиксирован хеш файла, персональные данные не переносятся",
            }
        )

    cache: dict[Path, dict | None] = {}
    for spec in MEASURED_SOURCES:
        source = ROOT / spec["source"]
        if source not in cache:
            cache[source] = load_json(source)
        payload = cache[source]
        value = dig(payload, spec["path"]) if payload is not None else None
        meta = (payload or {}).get("meta") if isinstance(payload, dict) else None
        commit = (payload or {}).get("commit") if isinstance(payload, dict) else None
        if commit is None and isinstance(meta, dict):
            commit = meta.get("commit")
        seed = (payload or {}).get("seed") if isinstance(payload, dict) else None
        if seed is None and isinstance(meta, dict):
            seed = meta.get("seed")
        note = ""
        if payload is None:
            note = f"артефакт отсутствует: {spec['source']} — прогон не выполнен"
        elif value is None:
            note = f"в артефакте нет значения по пути {'/'.join(spec['path'])}"
        entries.append(
            {
                "id": spec["id"],
                "value": value,
                "unit": spec["unit"],
                "kind": "measured",
                "dataset": spec.get("dataset"),
                "mode": spec.get("mode"),
                "seed": seed,
                "commit": commit,
                "command": (
                    (payload or {}).get("command")
                    if isinstance(payload, dict)
                    else None
                ),
                "date": (
                    (payload or {}).get("date", date)
                    if isinstance(payload, dict)
                    else date
                ),
                "source": spec["source"],
                "source_path": "/".join(spec["path"]),
                "gate": None,
                "note": note,
            }
        )

    # Вердикты по целевым показателям — из измеренного значения и требования.
    by_id = {entry["id"]: entry for entry in entries}
    for entry in entries:
        gate_of = next(
            (
                spec.get("gate_of")
                for spec in MEASURED_SOURCES
                if spec["id"] == entry["id"]
            ),
            None,
        )
        if not gate_of:
            continue
        target = by_id.get(gate_of, {}).get("value")
        value = entry["value"]
        if value is None or target is None:
            entry["gate"] = "не измерено"
            continue
        if (
            gate_of.endswith("fpr_max")
            or gate_of.endswith("ms_per_1000_chars")
            or gate_of.endswith("seconds_2m")
        ):
            entry["gate"] = (
                "достигнут" if float(value) <= float(target) else "НЕ достигнут"
            )
        else:
            entry["gate"] = (
                "достигнут" if float(value) >= float(target) else "НЕ достигнут"
            )
        entry["gate_source"] = gate_of

    measured = [entry for entry in entries if entry["kind"] == "measured"]
    missing = [entry["id"] for entry in measured if entry["value"] is None]
    return {
        "kind": "numbers_registry",
        "generated_at": date,
        "commit": git_commit(ROOT),
        "rule": (
            "каждое число в документах обязано быть здесь или в протоколе исключений; "
            "значение null означает «не измерено», а не ноль"
        ),
        "sources": [str(spec["source"]) for spec in MEASURED_SOURCES],
        "measured_total": len(measured),
        "measured_missing": missing,
        "entries": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Собрать reports/NUMBERS.json")
    parser.add_argument("--out", default=str(NUMBERS_PATH.relative_to(ROOT)))
    parser.add_argument(
        "--check",
        action="store_true",
        help="проверить, что реестр совпадает с артефактами",
    )
    args = parser.parse_args()

    payload = build()
    out = ROOT / args.out
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if not out.is_file():
            print(f"нет реестра {out}: запустите python scripts/build_numbers.py")
            return 1
        current = json.loads(out.read_text(encoding="utf-8"))
        same = {e["id"]: e["value"] for e in current["entries"]} == {
            e["id"]: e["value"] for e in payload["entries"]
        }
        print(
            "реестр совпадает с артефактами"
            if same
            else "реестр устарел: пересоберите NUMBERS.json"
        )
        return 0 if same else 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered, encoding="utf-8")
    print(
        f"{out}: записей {len(payload['entries'])}, измеренных {payload['measured_total']}"
    )
    if payload["measured_missing"]:
        print("не измерено:", ", ".join(payload["measured_missing"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
