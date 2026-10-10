"""Grouped train/dev AUC ablation; never reads the official test labels.

These experiment-local heads are not served and cannot change production
weights or thresholds. All feature arms share document-disjoint folds,
training-only scaling and identical optimizer settings. Dev is confirmatory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_hf_a3 import CANDIDATES, document_id, sha256, validate_cache, write_json  # noqa: E402

from spanverify.dataset import read_pairs  # noqa: E402
from spanverify.engine import Verifier, WeightsBundle  # noqa: E402
from spanverify.features import DEFAULT_WEIGHTS, HF_MODEL_DEFAULT, load_feature_cache  # noqa: E402
from spanverify.logreg import probabilities, standardize, train_logreg  # noqa: E402
from spanverify.train import auc_score, collect_samples  # noqa: E402


def grouped_folds(pairs: list[dict], folds: int, seed: int) -> list[dict]:
    groups = sorted({document_id(pair) for pair in pairs})
    if folds < 2 or len(groups) < folds:
        raise ValueError("need at least folds document groups")
    random.Random(seed).shuffle(groups)
    assignments = {group: index % folds for index, group in enumerate(groups)}
    return [
        {
            "train": [str(p["id"]) for p in pairs if assignments[document_id(p)] != fold],
            "validation": [str(p["id"]) for p in pairs if assignments[document_id(p)] == fold],
            "validation_documents": sorted(g for g, f in assignments.items() if f == fold),
        }
        for fold in range(folds)
    ]


def candidate_rows(verifier, pairs: list[dict]) -> tuple[list[list[float]], list[int], list[str]]:
    samples, _ = collect_samples(verifier, pairs)
    by_id = {str(p["id"]): p for p in pairs}
    matrices = {key: verifier.features_for(p["answer"], p["context"]) for key, p in by_id.items()}
    rows = []
    for sample in samples:
        rows.append(
            [
                sample.features["attention_entropy"],
                sample.features["ctx_attention_mass"],
                sample.features["embedding_density"],
                sample.features["risk"],
                sample.index / max(1, sample.total),
                min(1.0, len(sample.text) / 20),
                *(getattr(matrices[sample.pair_id], name)[sample.index] for name in CANDIDATES),
            ]
        )
    return rows, [s.label for s in samples], [s.pair_id for s in samples]


def fit_predict(train_rows, train_labels, eval_rows, columns):
    rows, means, scales = standardize([[row[i] for i in columns] for row in train_rows])
    model = train_logreg(rows, train_labels)
    transformed = [[(row[i] - means[j]) / scales[j] for j, i in enumerate(columns)] for row in eval_rows]
    return probabilities(model, transformed)


def paired_document_bootstrap(labels, baseline, candidate, documents, seed, iterations):
    """Paired bootstrap percentile interval for ΔAUC; resample whole documents."""
    index_by_doc = {}
    for index, document in enumerate(documents):
        index_by_doc.setdefault(document, []).append(index)
    keys = sorted(index_by_doc)
    rng = random.Random(seed)
    deltas = []
    for _ in range(iterations):
        indices = [i for _ in keys for i in index_by_doc[rng.choice(keys)]]
        y = [labels[i] for i in indices]
        base_auc = auc_score(y, [baseline[i] for i in indices])
        candidate_auc = auc_score(y, [candidate[i] for i in indices])
        if base_auc == base_auc and candidate_auc == candidate_auc:
            deltas.append(candidate_auc - base_auc)
    if not deltas:
        return None
    deltas.sort()
    return [deltas[int((len(deltas) - 1) * 0.025)], deltas[int((len(deltas) - 1) * 0.975)]]


def ablate(pairs, dev, verifier, folds, seed, bootstrap):
    if {document_id(p) for p in pairs} & {document_id(p) for p in dev}:
        raise ValueError("train/dev document overlap")
    rows, labels, ids = candidate_rows(verifier, pairs)
    dev_rows, dev_labels, _ = candidate_rows(verifier, dev)
    assignments = grouped_folds(pairs, folds, seed)
    base_columns = list(range(6))
    arms = {
        "baseline": base_columns,
        **{name: base_columns + [6 + i] for i, name in enumerate(CANDIDATES)},
        "all_candidates": list(range(10)),
    }
    out = {}
    by_id = {str(p["id"]): document_id(p) for p in pairs}
    docs = [by_id[pair_id] for pair_id in ids]
    base_oof = None
    for name, columns in arms.items():
        print(f"AUC ablation: {name}", flush=True)
        oof = [None] * len(rows)
        fold_metrics = []
        for assignment in assignments:
            validation_ids = set(assignment["validation"])
            tr = [i for i, pair_id in enumerate(ids) if pair_id not in validation_ids]
            val = [i for i, pair_id in enumerate(ids) if pair_id in validation_ids]
            if len({labels[i] for i in tr}) < 2 or len({labels[i] for i in val}) < 2:
                raise ValueError("both labels required in every grouped fold")
            predicted = fit_predict([rows[i] for i in tr], [labels[i] for i in tr], [rows[i] for i in val], columns)
            for i, score in zip(val, predicted, strict=True):
                oof[i] = score
            fold_metrics.append(auc_score([labels[i] for i in val], predicted))
        if any(score is None for score in oof):
            raise ValueError("incomplete grouped OOF predictions")
        dev_pred = fit_predict(rows, labels, dev_rows, columns)
        oof_auc, dev_auc = auc_score(labels, oof), auc_score(dev_labels, dev_pred)
        if name == "baseline":
            base_oof = oof
        delta = oof_auc - (out["baseline"]["auc_oof"] if out else oof_auc)
        dev_delta = dev_auc - (out["baseline"]["auc_dev"] if out else dev_auc)
        interval = paired_document_bootstrap(labels, base_oof, oof, docs, seed, bootstrap)
        out[name] = {
            "columns": columns,
            "auc_oof": oof_auc,
            "auc_dev": dev_auc,
            "delta_auc_oof": delta,
            "delta_auc_dev": dev_delta,
            "delta_auc_ci_95": interval,
            "fold_auc": fold_metrics,
            "eligible_for_followup": bool(name != "baseline" and interval and interval[0] > 0 and dev_delta > 0),
        }
    return {
        "arms": out,
        "fold_assignments": assignments,
        "seed": seed,
        "bootstrap_documents": bootstrap,
        "promoted_features": [],
        "selection_rule": "OOF paired document-bootstrap lower ΔAUC bound > 0 and dev ΔAUC > 0",
        "thresholds_changed": False,
        "test_used_for_selection": False,
        "note": "AUC experiment only: positive ΔAUC does not establish served token F1; no features promoted.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", default="data/corpus_a3/splits")
    parser.add_argument("--features-cache", action="append", required=True)
    parser.add_argument("--model", default=HF_MODEL_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--out", default="reports/hf_a3_feature_ablation.json")
    args = parser.parse_args(argv)
    if args.bootstrap < 1:
        parser.error("bootstrap must be positive")
    split_dir = Path(args.splits_dir)
    train_pairs = list(read_pairs(split_dir / "train.jsonl"))
    dev = list(read_pairs(split_dir / "dev.jsonl"))
    cache = load_feature_cache(args.features_cache)
    validate_cache(cache, train_pairs + dev, args.model, diagnostics=True)
    verifier = Verifier(
        mode="hf",
        model_name=args.model,
        features_cache=cache,
        weights=WeightsBundle(weights=dict(DEFAULT_WEIGHTS), threshold=0.5, mode="hf"),
    )
    result = ablate(train_pairs, dev, verifier, args.folds, args.seed, args.bootstrap)
    # Hash feature cache inputs and the exact two selection splits.
    files = []
    for path in map(Path, args.features_cache):
        files.extend(sorted(path.rglob("features_*.jsonl")) if path.is_dir() else [path])
    result.update(
        model=args.model,
        input_sha256={str(p): sha256(p) for p in files},
        split_sha256={name: sha256(split_dir / f"{name}.jsonl") for name in ("train", "dev")},
        feature_names=[
            "attention_entropy",
            "ctx_attention_mass",
            "embedding_density",
            "risk",
            "position",
            "length",
            *CANDIDATES,
        ],
        cache_fingerprint=hashlib.sha256(json.dumps(sorted(cache)).encode()).hexdigest(),
    )
    write_json(Path(args.out), result)
    print(json.dumps(result["arms"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
