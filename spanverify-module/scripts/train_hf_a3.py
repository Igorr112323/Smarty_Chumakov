"""Isolated HF-A3 training: measured artifacts, official splits, no extra mask scan.

Run from spanverify-module. No legacy config file is overwritten. HF dependency
or model failures are written as blocked measurements, never as zero scores.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spanverify.dataset import read_pairs  # noqa: E402
from spanverify.engine import Verifier, WeightsBundle  # noqa: E402
from spanverify.features import (  # noqa: E402
    DEFAULT_WEIGHTS,
    HF_MODEL_DEFAULT,
    extract_features,
    feature_cache_key,
    get_feature_cache,
    load_feature_cache,
    set_feature_cache,
    validate_feature_cache_model,
)
from spanverify.train import _split_pairs, train  # noqa: E402

CANDIDATES = ("ctx_sim_contrast", "ctx_sim_margin", "ctx_support_distance", "ctx_similarity_decay")


class Tee:
    """Flush console and persistent log on every write."""

    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_json(value):
    """Strict JSON: unmeasured / undefined floats are null, never NaN."""
    import math

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(safe_json(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def document_id(pair: dict) -> str:
    meta = pair.get("meta") or {}
    group = meta.get("doc_id") or meta.get("group")
    if not group:
        raise ValueError(f"missing document group: {pair.get('id')}")
    return str(group)


def training_adapter(pairs: list[dict]) -> list[dict]:
    """Map A3 document groups to legacy grouping on COPIES, not the corpus.

    The legacy key is (subject, question or template); both template fields
    must be constant to prevent splitting variants from the same document.
    """
    adapted = copy.deepcopy(pairs)
    for pair in adapted:
        group = document_id(pair)
        pair["meta"].update(subject=group, question="hf-a3-document", template="hf-a3-document")
    return adapted


def load_splits(dataset: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    records = list(read_pairs(dataset))
    by_id = {str(pair["id"]): pair for pair in records}
    if len(by_id) != len(records):
        raise ValueError("duplicate pair id in dataset")
    splits = {}
    used_ids, used_docs = set(), set()
    for name in ("train", "dev", "test"):
        path = dataset.parent / "splits" / f"{name}.jsonl"
        part = list(read_pairs(path))
        if not part:
            raise ValueError(f"empty official split: {path}")
        ids = {str(pair["id"]) for pair in part}
        docs = {document_id(pair) for pair in part}
        if len(ids) != len(part) or ids & used_ids or docs & used_docs:
            raise ValueError(f"official split overlap: {name}")
        for pair in part:
            if by_id.get(str(pair["id"])) != pair:
                raise ValueError(f"official split not identical to dataset: {pair['id']}")
        used_ids.update(ids)
        used_docs.update(docs)
        splits[name] = part
    if used_ids != set(by_id):
        raise ValueError("official splits do not partition the dataset")
    return records, splits


def validate_cache(cache, pairs: list[dict], model: str, diagnostics: bool = False) -> None:
    """Fail closed: incomplete, legacy/unproven, malformed caches are not evidence."""
    validate_feature_cache_model(cache, model)
    for pair in pairs:
        key = feature_cache_key(pair["answer"], pair["context"], "hf", model)
        matrix = cache.get(key)
        if matrix is None:
            raise ValueError(f"incomplete HF cache: {pair['id']}")
        if matrix.meta.get("model") != model or matrix.meta.get("mode") != "hf":
            raise ValueError(f"HF cache model provenance missing: {pair['id']}; recompute with current script")
        from spanverify.core import tokenize_with_offsets

        size = len(tokenize_with_offsets(pair["answer"]))
        names = ("attention_entropy", "ctx_attention_mass", "embedding_density") + (CANDIDATES if diagnostics else ())
        for name in names:
            values = getattr(matrix, name)
            if len(values) != size or any(safe_json(v) is None for v in values):
                raise ValueError(f"invalid HF cache feature {name}: {pair['id']}")


def save_hf_artifacts(report, out: Path) -> dict[str, Path]:
    """Keep the existing artifact structures, but use HF-specific filenames."""
    head_path = out.with_name("head_hf.json")
    part_path = out.with_name("participation_hf.json")
    if report.bundle.head.get("type") == "logreg":
        report.bundle.head["file"] = str(head_path)
    report.bundle.save(out)
    # A non-selected / untrained head is explicit, not a fake trained model.
    head = report.head.get("payload") or {"type": "none"}
    part = report.participation.get("payload") or {
        "type": "none",
        "calibrated_on": report.participation.get("calibrated_on", "not measured"),
    }
    write_json(head_path, {**head, "version": report.bundle.version, "seed": report.seed})
    write_json(part_path, {**part, "version": report.bundle.version, "seed": report.seed})
    return {"weights": out, "head": head_path, "participation": part_path}


def make_report(payload: dict) -> str:
    lines = [
        "# HF A3: воспроизводимый запуск",
        "",
        f"Статус: **{payload['status']}**.",
        f"Дата UTC: {payload['started_at']}. Прогон: {payload.get('run_id') or 'local'}.",
        f"Модель: `{payload['model']}`. SHA256 корпуса: `{payload.get('dataset_sha256')}`.",
        "",
    ]
    if payload.get("error"):
        lines += [f"Блокер: `{payload['error']}`", "", "HF-метрики: null (обучение не завершено)."]
    else:
        lines += [
            "| Срез | Пар | token F1 | token FPR | token AUC | verdict F1 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for name, metrics in payload["evaluations"].items():
            t, v = metrics["tokens"], metrics["verdicts"]
            lines.append(f"| {name} | {metrics['pairs']} | {t['f1']} | {t['fpr']} | {t['auc']} | {v['f1']} |")
        lines += [
            "",
            f"Критерий на официальном test: **{payload['target_met_test']}**.",
            "Весь корпус — описательный срез, содержит обучающие данные и не доказывает обобщение.",
            "Внутренний token-CV legacy train имеет утечку по токенам: диагностический, не приёмочный.",
            "Пороги выбирает существующий train только на train; дополнительного сканирования маски нет.",
            "Диагностические признаки НЕ включены в рабочие без отдельного доказательства улучшения AUC.",
            "participation_hf с type=none означает отсутствие HF-калибровки участия, не нулевую долю ИИ.",
        ]
    return "\n".join(lines) + "\n"


def run(args) -> dict:
    dataset = Path(args.dataset)
    records, splits = load_splits(dataset)
    cache = load_feature_cache(args.features_cache) if args.features_cache else {}
    if args.features_cache:
        validate_cache(cache, records, args.model)
    else:
        # Every A3 pair is extracted only once; training consumes this cache.
        for index, pair in enumerate(records, 1):
            matrix = extract_features(pair["answer"], pair["context"], mode="hf", model_name=args.model)
            matrix.meta.update(model=args.model, mode="hf")
            cache[feature_cache_key(pair["answer"], pair["context"], "hf", args.model)] = matrix
            if index % 10 == 0:
                print(f"features {index}/{len(records)}", flush=True)
    validate_cache(cache, records, args.model)
    previous_cache = get_feature_cache()
    set_feature_cache(cache)
    try:
        adapted = training_adapter(splits["train"])
        internal_train, internal_val = _split_pairs(adapted, test_size=0.3, seed=args.seed)
        overlap = {document_id(p) for p in internal_train} & {document_id(p) for p in internal_val}
        if overlap:
            raise ValueError("internal document overlap")
        verifier = Verifier(
            mode="hf",
            model_name=args.model,
            features_cache=cache,
            weights=WeightsBundle(weights=dict(DEFAULT_WEIGHTS), threshold=0.5, mode="hf"),
        )
        report = train(
            adapted,
            mode="hf",
            seed=args.seed,
            folds=args.folds,
            target_fpr=args.target_fpr,
            dataset_name=str(dataset) + " (official train only)",
            verifier=verifier,
        )
        report.bundle.meta.update(
            hf_model=args.model, dataset_sha256=sha256(dataset), internal_cv="legacy token CV; diagnostic only"
        )
        paths = save_hf_artifacts(report, Path(args.out))
        served = Verifier(mode="hf", weights=report.bundle, features_cache=cache)
        evaluations = {name: served.evaluate(part) for name, part in splits.items()}
        evaluations["whole_corpus_descriptive"] = served.evaluate(records)
        print(report.summary(), flush=True)
        print(json.dumps(safe_json(evaluations), ensure_ascii=False, indent=2), flush=True)
        test_tokens = evaluations["test"]["tokens"]
        return {
            "status": "measured",
            "pairs": len(records),
            "evaluations": evaluations,
            "target_met_test": test_tokens["f1"] >= 0.90 and test_tokens["fpr"] <= args.target_fpr,
            "split_document_ids": {name: sorted({document_id(p) for p in part}) for name, part in splits.items()},
            "internal_shared_documents": len(overlap),
            "validation_diagnostic_only": report.validation,
            "artifacts": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
            "feature_promotion": {"promoted": [], "reason": "pending independent grouped AUC ablation"},
        }
    finally:
        set_feature_cache(previous_cache)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/corpus_a3/pairs.jsonl")
    parser.add_argument("--out", default="config/weights_hf.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--target-fpr", type=float, default=0.1)
    parser.add_argument("--model", default=HF_MODEL_DEFAULT)
    parser.add_argument("--features-cache", action="append", default=[])
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--require-target", action="store_true", help="fail if official test misses quality target")
    args = parser.parse_args(argv)
    if args.folds < 2 or not 0 <= args.target_fpr <= 1:
        parser.error("folds >= 2 and target-fpr in [0, 1] required")
    # Never overwrite existing demo weights even through an alternate spelling.
    reserved = {ROOT / "config" / name for name in ("weights.json", "head.json", "participation.json", "config.json")}
    destinations = [
        Path(args.out),
        Path(args.out).with_name("head_hf.json"),
        Path(args.out).with_name("participation_hf.json"),
    ]
    if any(path.resolve() in {item.resolve() for item in reserved} for path in destinations):
        parser.error("--out must not overwrite legacy config artifacts")
    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    payload = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "seed": args.seed,
        "folds": args.folds,
        "target_fpr": args.target_fpr,
        "command": sys.argv,
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "commit": os.environ.get("GITHUB_SHA"),
        "python": platform.python_version(),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [
                Path(__file__),
                ROOT / "spanverify" / "features.py",
                ROOT / "spanverify" / "train.py",
                ROOT / "spanverify" / "engine.py",
                ROOT / "scripts" / "precompute_features.py",
            ]
        },
        "dataset_sha256": None,
        "evaluations": None,
    }
    with (reports / "hf_a3_train.log").open("a", encoding="utf-8") as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            print(json.dumps(payload, ensure_ascii=False), flush=True)
            try:
                payload["dataset_sha256"] = sha256(Path(args.dataset))
                payload.update(run(args))
                code = 1 if args.require_target and not payload["target_met_test"] else 0
            except Exception as exc:
                traceback.print_exc()
                payload.update(status="blocked", error=f"{type(exc).__name__}: {exc}")
                code = 2
            payload["duration_s"] = time.perf_counter() - started
            write_json(reports / "hf_a3_run.json", payload)
            (reports / "hf_a3_eval.md").write_text(make_report(payload), encoding="utf-8")
            experiment_dir = reports / "experiments"
            experiment_dir.mkdir(parents=True, exist_ok=True)
            (experiment_dir / "hf_a3_experiment.md").write_text(make_report(payload), encoding="utf-8")
            print(f"status={payload['status']}; report={reports / 'hf_a3_run.json'}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
