#!/usr/bin/env python3
"""Заморозка конфигурации шага 3 цикла v2 — до единственного запуска на test.

Скрипт делает ровно то, что запрещает делать руками: берёт отобранную на val
конфигурацию из `reports/v2/sweep.json`, берёт хеши разбиения из
`reports/v2/freeze.json`, записывает `config/hf_final_config_v2.json` (его sha256
потом обязан совпасть с полем `config_sha256` в манифесте измерения) и вставляет
в `docs/EXPERIMENTS_v2.md` блок с числом порога. Если шаг 2 не завершён, отобранной
конфигурации нет, стоп-правило сработало или разброс seed'ов не подтверждён —
заморозка невозможна, и скрипт заканчивается ошибкой, а не «тихим» конфигом.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import hf_protocol as v1  # noqa: E402
import hf_protocol_v2 as v2  # noqa: E402

BEGIN = "<!-- FREEZE-V2:BEGIN -->"
END = "<!-- FREEZE-V2:END -->"
FINAL_COMMAND = (
    "python scripts/hf_protocol_v2.py final --config config/hf_final_config_v2.json "
    "--grid-cache reports/v2-grid --splits data/corpus_v2/splits --corpus-name v2 "
    "--seeds 42,43,44,45,46 --include-val --freeze reports/v2/freeze.json "
    "--sweep reports/v2/sweep.json --seedcheck reports/v2/seedcheck.json --out reports/v2"
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"нет файла {path}: шаг обязан быть выполнен до заморозки конфигурации")
    return json.loads(path.read_text(encoding="utf-8"))


def build_payload(
    sweep: dict[str, Any], freeze: dict[str, Any], experiments: dict[str, Any], seedcheck: dict[str, Any]
) -> dict[str, Any]:
    if (sweep.get("stop_rule") or {}).get("triggered"):
        raise SystemExit(
            "стоп-правило шага 2 сработало: конфигурация для шага 3 не замораживается, "
            "тестовая выборка не открывается (docs/METRIC_SPEC_v2.md, раздел 6)"
        )
    selected = sweep.get("selected")
    if not selected:
        raise SystemExit("на val не отобрана ни одна конфигурация (FPR_val > 0,40 у всех гипотез)")
    entry = next((item for item in sweep.get("results") or [] if item.get("name") == selected), None)
    if entry is None:
        raise SystemExit(f"в sweep.json нет записи об отобранной конфигурации «{selected}»")
    e1 = entry.get("e1_val_chosen") or {}
    if not e1.get("threshold"):
        raise SystemExit("у отобранной конфигурации нет замороженного порога (правило E1 на val)")
    if int(seedcheck.get("distinct_f1_values", 0)) < 2 and not seedcheck.get("accepted_zero_std"):
        raise SystemExit(
            f"seedcheck: F1 на val принимает одно значение на {len(seedcheck.get('seeds') or [])} seed'ах — "
            "seed не влияет на обучение; сначала bootstrap_docs, затем заморозка"
        )
    experiment = next((item for item in experiments.get("experiments") or [] if item.get("name") == selected), None)
    if experiment is None:
        raise SystemExit(f"гипотезы «{selected}» нет в config/hf_experiments_v2.json")
    features = [str(name) for name in v1._experiment_entry(1, experiment)["features"]]
    params = {
        key: value for key, value in experiment.items() if key not in {"name", "hypothesis", "change", "features"}
    }
    params.setdefault("classifier", str(experiment.get("classifier", "logreg")))
    return {
        "step": "final-config-v2",
        "protocol": v2.PROTOCOL,
        "name": selected,
        "number": entry.get("number"),
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frozen_by": "scripts/freeze_final_config_v2.py",
        "classifier": str(experiment.get("classifier", "logreg")),
        "features": features,
        "window": int(experiment.get("window", 0)),
        "merge_gap": int(experiment.get("merge_gap", 2)),
        "params": params,
        "threshold": float(e1["threshold"]),
        "threshold_rule": "E1",
        "threshold_source": (
            "максимум F1(токены) на val среди порогов с FPR(токены) ≤ 0,40 "
            "(правило цикла v2, docs/METRIC_SPEC_v2.md, раздел 3); переселекции по test нет"
        ),
        "seeds": [42, 43, 44, 45, 46],
        "seed_rule": "пять seed'ов, зафиксированы до запуска; лучший не выбирается; std обязан быть > 0",
        "goals": {
            "internal": {"token_f1_min": v2.GOAL_INTERNAL_F1, "token_fpr_max": v2.FPR_MAX},
            "contract_min": {"token_f1_min": v2.GOAL_CONTRACT_F1, "token_fpr_max": v2.FPR_MAX},
        },
        "primary_corpus": "v2",
        "splits_dir": str(experiments.get("splits") or "data/corpus_v2/splits"),
        "splits_sha256": dict(freeze.get("splits_sha256") or {}),
        "selection": {
            "corpus": "v2",
            "token_f1_val": e1.get("f1"),
            "token_fpr_val": e1.get("fpr"),
            "auc_val": e1.get("auc"),
            "answer_f1_val": e1.get("answer_f1"),
            "span_f1_val": e1.get("span_f1"),
            "e0_train_cv": entry.get("e0_train_cv"),
        },
        "selection_rule": str(sweep.get("selection_rule") or ""),
        "seed_variance": {
            "f1_std_on_val": seedcheck.get("f1_std"),
            "distinct_f1_values": seedcheck.get("distinct_f1_values"),
            "bootstrap_docs": seedcheck.get("bootstrap_docs"),
            "accepted_zero_std": seedcheck.get("accepted_zero_std"),
        },
        "source": {
            "sweep_file": "reports/v2/sweep.json",
            "sweep_sha256": v1.sha256_file(ROOT / "reports" / "v2" / "sweep.json"),
            "freeze_file": "reports/v2/freeze.json",
            "freeze_sha256": v1.sha256_file(ROOT / "reports" / "v2" / "freeze.json"),
            "sweep_config_sha256": str(sweep.get("config_sha256") or ""),
            "metric_spec_v2_sha256": str(freeze.get("metric_spec_v2_sha256") or ""),
        },
        "final_command": FINAL_COMMAND,
        "deviations": [],
    }


def render_block(payload: dict[str, Any], config_sha256: str) -> str:
    selection = payload["selection"]
    variance = payload["seed_variance"]
    e0 = selection.get("e0_train_cv") or {}
    return "\n".join(
        [
            BEGIN,
            "## Шаг 3: заморозка конфигурации (записано `scripts/freeze_final_config_v2.py`)",
            "",
            f"* Конфигурация: №{payload['number']} `{payload['name']}`, классификатор `{payload['classifier']}`, "
            f"признаков {len(payload['features'])}, окно {payload['window']}, `merge_gap` {payload['merge_gap']}.",
            f"* Порог, зафиксированный до шага 3 v2: {payload['threshold']}",
            f" (правило {payload['threshold_rule']}: максимум F1 на val при FPR ≤ {v2.FPR_MAX}).",
            "* Числа отбора на val: "
            f"F1 {selection['token_f1_val']}, FPR {selection['token_fpr_val']}, AUC {selection['auc_val']}, "
            f"F1 ответов {selection['answer_f1_val']}, F1 фрагментов {selection['span_f1_val']}.",
            "* Справочная строка E0 (порог из групповой CV на train): "
            + (
                f"порог {e0.get('threshold')}, F1 {e0.get('f1')}, FPR {e0.get('fpr')}"
                if e0
                else "не измерена (гипотеза не была в списке `e0_experiments`)"
            ),
            f"* Разброс seed'ов на val: std(F1) = {variance['f1_std_on_val']}, "
            f"различных значений F1 = {variance['distinct_f1_values']}, "
            f"бутстрап по документам = {variance['bootstrap_docs']}.",
            f"* Seed'ы шага 3: {', '.join(str(seed) for seed in payload['seeds'])} — {payload['seed_rule']}.",
            f"* Цели: внутренняя F1 ≥ {payload['goals']['internal']['token_f1_min']}, "
            f"контрактный минимум F1 ≥ {payload['goals']['contract_min']['token_f1_min']} "
            f"при FPR ≤ {v2.FPR_MAX} (токены, независимый test v2).",
            f"* sha256 `config/hf_final_config_v2.json`: `{config_sha256}`.",
            f"* sha256 сетки признаков (из заморозки шага 0): "
            f"`{payload['source']['sweep_config_sha256'][:16]}…` — файл `sweep.json`, "
            f"`{payload['source']['sweep_sha256'][:16]}…`.",
            "* Хеши разбиения, зафиксированные до теста: "
            + ", ".join(f"`{part}: {digest[:12]}…`" for part, digest in payload["splits_sha256"].items()),
            f"* Единственная команда прогона: `{payload['final_command']}` (она же `make hf-final-v2`).",
            f"* Время заморозки: {payload['frozen_at']}. С этого момента конфигурация, метрики, порог "
            "и правило вердикта не меняются; перезапуск шага 3 с другими параметрами запрещён.",
            END,
        ]
    )


def patch_doc(doc: Path, block: str) -> None:
    """Вставить или обновить блок заморозки в журнале.

    Секция «Шаг 3» генерируется целиком: править её построчно значило бы оставить
    в журнале старые числа рядом с новым конфигом.
    """
    text = doc.read_text(encoding="utf-8")
    if BEGIN in text and END in text:
        head, rest = text.split(BEGIN, 1)
        _old, tail = rest.split(END, 1)
        doc.write_text(head + block + tail, encoding="utf-8")
        return
    heading = "## Шаг 3: заморозка конфигурации"
    if heading in text:
        text = text.split(heading, 1)[0].rstrip() + "\n\n" + block + "\n"
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    doc.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Заморозка конфигурации шага 3 цикла v2")
    parser.add_argument("--sweep", default="reports/v2/sweep.json")
    parser.add_argument("--freeze", default="reports/v2/freeze.json")
    parser.add_argument("--seedcheck", default="reports/v2/seedcheck.json")
    parser.add_argument("--experiments", default="config/hf_experiments_v2.json")
    parser.add_argument("--out", default="config/hf_final_config_v2.json")
    parser.add_argument("--doc", default="docs/EXPERIMENTS_v2.md")
    parser.add_argument("--deviation", action="append", default=[], help="текст отступления (повторяемый)")
    args = parser.parse_args(argv)

    sweep = read_json(Path(args.sweep))
    freeze = read_json(Path(args.freeze))
    seedcheck = read_json(Path(args.seedcheck))
    experiments = read_json(Path(args.experiments))
    payload = build_payload(sweep, freeze, experiments, seedcheck)
    payload["deviations"] = list(args.deviation)
    out = Path(args.out)
    body = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    # Хешируется ровно то, что будет записано: скрипт пишет те же байты, что и
    # посчитал (в v1 эта тонкость стоила расхождения на один хеш).
    out.write_text(body + "\n", encoding="utf-8")
    digest = v1.sha256_file(out)
    patch_doc(Path(args.doc), render_block(payload, digest))
    print(f"заморожено: {out} (sha256 {digest})")
    print(f"порог, зафиксированный до шага 3 v2: {payload['threshold']}")
    print(f"блок журнала обновлён: {args.doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
