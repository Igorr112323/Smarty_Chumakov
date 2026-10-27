#!/usr/bin/env python3
"""REALGEN-R1: freeze the full-dev inference bundle in GitHub Actions (dev-only).

This is the *training* step that PR #25 was missing. It fits the three
preregistered systems on the complete 300-answer REALGEN dev split and no other
answers:

* composite  — StandardScaler + LogisticRegression on the 27 fixed features;
* r2_only    — StandardScaler + LogisticRegression on the 16 R2 features only;
* rubert_tiny2_ft — one fine-tune of cointegrated/rubert-tiny2 on all dev
  sentence windows (batch 8, lr 2e-5, 3 epochs, seed 42, max_length 96).

Sealed data boundary: the 100 test_realgen answers/labels/spans, test-only
document rows and the permanently locked real_test 1200 pairs are never parsed.
A Path.open guard makes any accidental read raise instead of silently leaking
test content. Labels come from the already published dev-only
REALGEN_PHASE_B_OOF.jsonl; REALGEN_SPANS.jsonl is not needed at all.

The workflow publishes ONLY small metadata through the GITHUB_STEP_SUMMARY →
check-runs output.text bridge (see realgen_publish.py). The 100+ MB zip is an
Actions artifact, never a git blob, and this workflow never commits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REAL = ROOT / "reports" / "science" / "real"
sys.path.insert(0, str(HERE))
import realgen_common as rc  # noqa: E402
import realgen_phase_b as pb  # noqa: E402
import realgen_publish as rp  # noqa: E402

SEED = 42
TRAINED_ON = "realgen_dev_300_only"
ARTIFACT_NAME = "realgen-test-model-bundle"
BUNDLE_FILENAME = "REALGEN_TEST_MODEL_BUNDLE.zip"
MANIFEST_FILENAME = "REALGEN_TEST_MODEL_BUNDLE_MANIFEST.json"
INNER_MANIFEST = "manifest.json"
PREDICTOR_SOURCE = HERE / "realgen_test_predictor.py"
PREDICTOR_IN_BUNDLE = "realgen_test_predict.py"
TOKEN_THRESHOLD = 0.5
RUBERT_REPO = "cointegrated/rubert-tiny2"
HF_FEATURE_NAMES = pb.OLD5 + (
    "sentence_cosine_answer_doc",
    "sentence_cosine_answer_chunk_max",
    "contrast_doc_chunk",
    "rolling_logprob_next_token",
)
FT_HYPERPARAMS = {
    "batch_size": 8,
    "learning_rate": 2e-5,
    "epochs": 3,
    "max_length": 96,
    "seed": SEED,
    "window": "sentence regex windows (Phase B sentence_windows), fallback = whole answer",
    "token_to_word": "mean of overlapping subtoken class-1 probabilities",
    "optimizer": "AdamW",
    "shuffle": "random.Random(42) permutation per epoch",
}
DOC_ID_PREFIX = re.compile(r'^\{"doc_id":\s*"([^"]+)"')
FORBIDDEN_FILES = (
    REAL / "REALGEN_ANSWERS_TEST_REALGEN.jsonl",
    REAL / "REALGEN_ANSWERS.jsonl",
    REAL / "REALGEN_LABELS.jsonl",
    REAL / "REALGEN_SPANS.jsonl",
    REAL / "real_test_pairs.jsonl",
    REAL / "real_test_spans.jsonl",
)


def install_sealed_read_guard() -> None:
    """Refuse, at runtime, any attempt to open a sealed payload during fitting."""
    forbidden = {path.resolve() for path in FORBIDDEN_FILES}
    original = Path.open

    def guarded(self: Path, *args, **kwargs):
        if self.resolve() in forbidden:
            raise RuntimeError(f"REALGEN freeze refused sealed input: {self.name}")
        return original(self, *args, **kwargs)

    Path.open = guarded  # type: ignore[method-assign]


def load_dev_only() -> tuple[list[dict], dict[str, dict], dict[tuple, int], dict]:
    """Dev answers, dev-only doc rows, dev token labels, Phase B facts."""
    frozen = json.loads((REAL / "REALGEN_FROZEN.json").read_text(encoding="utf-8"))
    phase_b = json.loads((REAL / "REALGEN_PHASE_B.json").read_text(encoding="utf-8"))
    assert frozen["splits"]["dev"]["n_answers"] == 300
    assert frozen["splits"]["test_realgen"]["sealed"] is True
    assert frozen["real_test_viewed"] is False
    assert phase_b["data"]["dev_answers"] == 300
    assert phase_b["provenance"]["test_realgen_file_read"] is False

    answers = rc.read_jsonl(REAL / "REALGEN_ANSWERS_DEV.jsonl")
    assert len(answers) == 300
    assert all(row["split"] == "dev" for row in answers)
    dev_answer_ids = {row["answer_id"] for row in answers}
    dev_doc_ids = {row["doc_id"] for row in answers}

    labels: dict[tuple, int] = {}
    for row in rc.read_jsonl(REAL / "REALGEN_PHASE_B_OOF.jsonl"):
        assert row["answer_id"] in dev_answer_ids, "OOF row outside dev split"
        labels[(row["answer_id"], row["start"], row["end"])] = int(row["label"])
    assert len(labels) == phase_b["data"]["n_tokens"], "dev OOF token count mismatch"

    # Test-specific context rows are skipped before json.loads: their text is
    # never materialized. Only the 300 dev documents are parsed.
    docs: dict[str, dict] = {}
    with (REAL / "REALGEN_DOCS.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            match = DOC_ID_PREFIX.match(line)
            if match is None or match.group(1) not in dev_doc_ids:
                continue
            row = json.loads(line)
            docs[row["doc_id"]] = row
    assert set(docs) == dev_doc_ids, "dev document coverage is incomplete"
    return answers, docs, labels, phase_b


def build_ft_samples(
    answers: list[dict], positive_by_answer: dict[str, list[tuple[int, int]]], tokenizer
) -> list[dict]:
    """Exactly the Phase B window/tokenization recipe, without OOF folds."""
    samples: list[dict] = []
    for rec in answers:
        answer = rec["answer"]
        windows = [(s, e) for s, e in pb.sentence_windows(answer)] or [(0, len(answer))]
        positives = positive_by_answer.get(rec["answer_id"], [])
        for start, end in windows:
            piece = answer[start:end]
            if len(piece.strip()) < 3:
                continue
            encoded = tokenizer(
                piece, return_offsets_mapping=True, truncation=True, max_length=96, add_special_tokens=True
            )
            offsets = encoded.pop("offset_mapping")
            token_labels = []
            for a, b in offsets:
                if b <= a:
                    token_labels.append(-100)
                else:
                    token_labels.append(
                        int(any(ps < start + b and start + a < pe for ps, pe in positives))
                    )
            if not any(label >= 0 for label in token_labels):
                continue
            samples.append(
                {
                    "answer_id": rec["answer_id"],
                    "start": start,
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "offsets": offsets,
                    "labels": token_labels,
                }
            )
    return samples


def train_rubert_ft(
    samples: list[dict], rubert_dir: str, tokenizer, torch, transformers
) -> tuple[object, dict]:
    """One full-dev fine-tune; identical optimization to every Phase B fold."""
    from transformers import AutoModelForTokenClassification

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    model = AutoModelForTokenClassification.from_pretrained(rubert_dir, num_labels=2)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=FT_HYPERPARAMS["learning_rate"])
    order = list(range(len(samples)))
    rng = random.Random(SEED)
    steps = 0
    for _epoch in range(FT_HYPERPARAMS["epochs"]):
        rng.shuffle(order)
        for i in range(0, len(order), FT_HYPERPARAMS["batch_size"]):
            batch = [samples[j] for j in order[i : i + FT_HYPERPARAMS["batch_size"]]]
            max_len = max(len(s["input_ids"]) for s in batch)
            input_ids = torch.tensor(
                [s["input_ids"] + [tokenizer.pad_token_id or 0] * (max_len - len(s["input_ids"])) for s in batch]
            )
            mask = torch.tensor(
                [s["attention_mask"] + [0] * (max_len - len(s["attention_mask"])) for s in batch]
            )
            labels = torch.tensor([s["labels"] + [-100] * (max_len - len(s["labels"])) for s in batch])
            outputs = model(input_ids=input_ids, attention_mask=mask, labels=labels)
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            steps += 1
    model.eval()
    facts = {
        "train_steps": steps,
        "n_windows": len(samples),
        "n_subtokens": sum(sum(1 for a, b in s["offsets"] if b > a) for s in samples),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    return model, facts


def fit_logreg(X: np.ndarray, y: np.ndarray) -> dict:
    """Full-dev StandardScaler + LogisticRegression, exported as plain numbers."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    clf.fit(scaler.transform(X), y)
    assert clf.classes_.tolist() == [0, 1], "unexpected class order"
    return {
        "estimator": "sklearn.linear_model.LogisticRegression",
        "params": {"max_iter": 1000, "class_weight": "balanced", "random_state": SEED},
        "classes": [int(c) for c in clf.classes_],
        "scaler_mean": [float(v) for v in scaler.mean_],
        "scaler_scale": [float(v) for v in scaler.scale_],
        "coefficients": [float(v) for v in clf.coef_[0]],
        "intercept": float(clf.intercept_[0]),
        "token_threshold": TOKEN_THRESHOLD,
        "inference": "p = expit(((X - scaler_mean)/scaler_scale) @ coefficients + intercept)",
    }


def deterministic_zip(bundle_dir: Path, zip_path: Path) -> list[dict]:
    """Byte-stable zip: sorted members, fixed timestamps and attributes."""
    members = sorted((p for p in bundle_dir.rglob("*") if p.is_file()), key=lambda p: p.relative_to(bundle_dir).as_posix())
    files: list[dict] = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in members:
            relative = path.relative_to(bundle_dir).as_posix()
            data = path.read_bytes()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
            files.append(
                {
                    "name": relative,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=".arena/realgen_freeze")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()

    started = time.time()
    assert TRAINED_ON == "realgen_dev_300_only"
    install_sealed_read_guard()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir = out_dir / "bundle_root"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    answers, docs, labels, phase_b = load_dev_only()
    print(f"dev-only inputs: {len(answers)} answers, {len(docs)} dev docs, {len(labels)} labelled tokens", flush=True)

    rows_by_answer: dict[str, list[dict]] = {}
    y_by_answer: dict[str, np.ndarray] = {}
    for rec in answers:
        _windows, rows = pb.deterministic_features(rec["answer"], docs[rec["doc_id"]]["text"])
        label_vector = np.array(
            [labels[(rec["answer_id"], row["start"], row["end"])] for row in rows], dtype=np.int8
        )
        for row in rows:
            for name in HF_FEATURE_NAMES:
                row[name] = 0.0
        rows_by_answer[rec["answer_id"]] = rows
        y_by_answer[rec["answer_id"]] = label_vector

    # HF operations happen only in Actions: the sandbox cannot reach huggingface.co.
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(4)
    gpt_dir = snapshot_download(rc.MODEL)
    rubert_dir = snapshot_download(RUBERT_REPO)
    gpt_revision, rubert_revision = Path(gpt_dir).name, Path(rubert_dir).name
    assert re.fullmatch(r"[0-9a-f]{40}", gpt_revision), "unpinned GPT revision"
    assert re.fullmatch(r"[0-9a-f]{40}", rubert_revision), "unpinned rubert revision"
    gpt_tok = AutoTokenizer.from_pretrained(gpt_dir, use_fast=True)
    try:
        gpt_model = AutoModelForCausalLM.from_pretrained(gpt_dir, attn_implementation="eager")
    except (TypeError, ValueError):
        gpt_model = AutoModelForCausalLM.from_pretrained(gpt_dir)
        try:
            gpt_model.config._attn_implementation = "eager"
        except Exception:  # noqa: BLE001 - конфиг разных версий transformers
            pass
    gpt_model.config.output_attentions = True
    gpt_model.config.output_hidden_states = True
    gpt_model.eval()
    rubert_tok = AutoTokenizer.from_pretrained(rubert_dir, use_fast=True)
    rubert_model = AutoModel.from_pretrained(rubert_dir)
    rubert_model.eval()
    print("computing old5 + E1 features on dev answers...", flush=True)
    pb.hf_old5_and_e1(answers, docs, rows_by_answer, gpt_tok, gpt_model, torch, rubert_tok, rubert_model)
    del gpt_model

    all_rows: list[dict] = []
    ordered_labels: list[int] = []
    for rec in answers:
        rows = rows_by_answer[rec["answer_id"]]
        all_rows.extend(rows)
        ordered_labels.extend(int(v) for v in y_by_answer[rec["answer_id"]])
    y = np.asarray(ordered_labels, dtype=np.int8)
    X = np.array([[float(row.get(name, 0.0)) for name in pb.FEATURES] for row in all_rows], dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    assert len(all_rows) == phase_b["data"]["n_tokens"]
    assert int(y.sum()) == phase_b["data"]["n_positive_tokens"]
    assert X.shape[1] == 27
    print(f"matrix: {X.shape[0]} dev tokens, {int(y.sum())} positives, {X.shape[1]} features", flush=True)

    composite = fit_logreg(X, y)
    composite["features"] = list(pb.FEATURES)
    r2_indices = [pb.FEATURES.index(name) for name in pb.R2]
    r2_only = fit_logreg(X[:, r2_indices], y)
    r2_only["features"] = list(pb.R2)
    assert len(composite["coefficients"]) == 27 and len(r2_only["coefficients"]) == 16
    print("full-dev composite and R2-only estimators fitted", flush=True)

    positive_by_answer = {
        rec["answer_id"]: [
            (row["start"], row["end"])
            for row, label in zip(rows_by_answer[rec["answer_id"]], y_by_answer[rec["answer_id"]], strict=True)
            if label == 1
        ]
        for rec in answers
    }
    ft_samples = build_ft_samples(answers, positive_by_answer, rubert_tok)
    assert len(ft_samples) > 0
    print(f"fine-tuning rubert-tiny2 on {len(ft_samples)} dev windows...", flush=True)
    ft_model, ft_facts = train_rubert_ft(ft_samples, rubert_dir, rubert_tok, torch, transformers)
    ft_out = bundle_dir / "rubert_ft"
    ft_out.mkdir(parents=True, exist_ok=True)
    try:
        ft_model.save_pretrained(ft_out, safe_serialization=True)
    except TypeError:  # более старые версии transformers не знают safe_serialization
        ft_model.save_pretrained(ft_out)
    assert (ft_out / "model.safetensors").is_file(), "fine-tuned weights were not saved"
    rubert_tok.save_pretrained(ft_out)
    del ft_model
    print(f"fine-tune steps: {ft_facts['train_steps']}", flush=True)

    inner: dict = {
        "kind": "REALGEN_TEST_MODEL_BUNDLE_INNER_MANIFEST",
        "version": "v1",
        "campaign": "REALGEN-R1",
        "status": "FROZEN",
        "trained_on": TRAINED_ON,
        "seed": SEED,
        "token_threshold": TOKEN_THRESHOLD,
        "token_threshold_selection": (
            "preregistered fixed 0.5 in REALGEN_TEST_PREDEPLOY.json policies.token_threshold; "
            "no dev/test threshold tuning, calibration or post-hoc fit"
        ),
        "hyperparameters": FT_HYPERPARAMS,
        "models": {"composite": composite, "r2_only": r2_only},
        "rubert_tiny2_ft": {
            "base": RUBERT_REPO,
            "base_revision": rubert_revision,
            "directory": "rubert_ft",
            "weights_file": "rubert_ft/model.safetensors",
            "task": "AutoModelForTokenClassification, num_labels=2",
            **FT_HYPERPARAMS,
            **ft_facts,
        },
        "upstream_revisions": {
            "old5_gpt": {"repo_id": rc.MODEL, "revision": gpt_revision},
            "e1_rubert_base": {"repo_id": RUBERT_REPO, "revision": rubert_revision},
            "download_at_inference": "huggingface_hub.snapshot_download(repo_id, revision=<pinned sha>)",
        },
        "dev": {
            "answers": len(answers),
            "tokens": len(all_rows),
            "positive_tokens": int(y.sum()),
            "documents": len(docs),
            "label_source": "reports/science/real/REALGEN_PHASE_B_OOF.jsonl (already published, dev-only)",
            "phase_b_run_id": phase_b["provenance"]["run_id"],
        },
        "forbidden_at_fit_time": sorted(path.name for path in FORBIDDEN_FILES),
        "no_test_based_fit": True,
    }
    (bundle_dir / INNER_MANIFEST).write_text(
        json.dumps(inner, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    predictor_bytes = PREDICTOR_SOURCE.read_bytes()
    (bundle_dir / PREDICTOR_IN_BUNDLE).write_bytes(predictor_bytes)

    zip_path = out_dir / BUNDLE_FILENAME
    files = deterministic_zip(bundle_dir, zip_path)
    bundle_sha = rc.sha256_file(zip_path)
    bundle_bytes = zip_path.stat().st_size
    weights = next(f for f in files if f["name"] == "rubert_ft/model.safetensors")
    run_id = os.environ.get("GITHUB_RUN_ID", "local-no-ci-evidence")
    head_sha = os.environ.get("HEAD_SHA", "")
    manifest: dict = {
        **inner,
        "kind": "REALGEN_TEST_MODEL_BUNDLE_MANIFEST",
        "bundle": {
            "artifact_name": ARTIFACT_NAME,
            "filename": BUNDLE_FILENAME,
            "sha256": bundle_sha,
            "bytes": bundle_bytes,
            "mb": round(bundle_bytes / 1_000_000, 3),
            "file_count": len(files),
            "uncompressed_bytes": sum(f["bytes"] for f in files),
            "uncompressed_mb": round(sum(f["bytes"] for f in files) / 1_000_000, 3),
            "retention_days": 90,
            "files": files,
        },
        "rubert_tiny2_ft_weights": {
            "name": weights["name"],
            "sha256": weights["sha256"],
            "bytes": weights["bytes"],
            "mb": round(weights["bytes"] / 1_000_000, 3),
        },
        "train_run": {
            "run_id": run_id,
            "head_sha": head_sha,
            "workflow": ".github/workflows/realgen-freeze-bundle.yml",
            "artifact_name": ARTIFACT_NAME,
            "url": f"https://github.com/{os.environ.get('GITHUB_REPOSITORY', 'Igorr112323/Smarty_Chumakov')}/actions/runs/{run_id}",
            "extraction": (
                "manual: GITHUB_STEP_SUMMARY -> check-runs output.text (REALGEN-FREEZE/manifest) "
                "-> gh api extraction in sandbox -> hand commit with extraction field in RUN_ID_AUDIT.json; "
                "no workflow auto-commit"
            ),
        },
        "inputs_sha256": {
            "REALGEN_FROZEN.json": rc.sha256_file(REAL / "REALGEN_FROZEN.json"),
            "REALGEN_ANSWERS_DEV.jsonl": rc.sha256_file(REAL / "REALGEN_ANSWERS_DEV.jsonl"),
            "REALGEN_PHASE_B.json": rc.sha256_file(REAL / "REALGEN_PHASE_B.json"),
            "REALGEN_PHASE_B_OOF.jsonl": rc.sha256_file(REAL / "REALGEN_PHASE_B_OOF.jsonl"),
            "realgen_phase_b.py": rc.sha256_file(REAL / "realgen_phase_b.py"),
            "realgen_test_predictor.py": rc.sha256_file(PREDICTOR_SOURCE),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "runtime_seconds": round(time.time() - started, 3),
        },
        "sealed_access": {
            "test_realgen_answers_opened": False,
            "test_realgen_labels_or_spans_opened": False,
            "test_specific_context_rows_parsed": False,
            "real_test_1200_read": False,
            "official_workflow_dispatched": False,
            "official_test_metrics": None,
            "guard": "Path.open refused all FORBIDDEN_FILES during fitting",
        },
    }
    manifest_path = out_dir / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "trained_on": TRAINED_ON,
        "dev_answers": len(answers),
        "dev_tokens": len(all_rows),
        "train_steps": ft_facts["train_steps"],
        "run_id": run_id,
        "head_sha": head_sha,
        "artifact_name": ARTIFACT_NAME,
        "bundle_sha256": bundle_sha,
        "bundle_mb": manifest["bundle"]["mb"],
        "weights_sha256": weights["sha256"],
        "weights_mb": manifest["rubert_tiny2_ft_weights"]["mb"],
        "official_test_metrics": None,
    }
    rp.step_summary("### REALGEN full-dev frozen bundle\n\n```json\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n```")
    if args.publish:
        repo = os.environ.get("GITHUB_REPOSITORY", "Igorr112323/Smarty_Chumakov")
        rp.publish_file(manifest_path, "REALGEN-FREEZE/manifest", head_sha or "local", repo, summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"bundle={zip_path} manifest={manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
