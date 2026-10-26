#!/usr/bin/env python3
"""REALGEN-R1 фаза B: батарея признаков + fine-tune rubert-tiny2 (запуск в CI).

Читает ТОЛЬКО dev-подмножество (300 ответов). test_realgen и real_test 1200
не открываются (только sha256 в заморозке).

Батарея:
  old5  — 5 исторических семейств E0, пересчитанных на той же модели
          ruGPT3small (энтропия внимания, масса на контекст, norm,
          max similarity, embedding density);
  E1    — 6 признаков (rubert-tiny2 sentence cosine ×3, rolling logprob,
          lexical number attribution, stopword norm);
  R2    — 16 детерминированных фактографии/стилометрии (явные определения ниже);
  FT    — fine-tune cointegrated/rubert-tiny2 токен-классификатор,
          OOF 5-fold по sha256(doc_id), batch 8, lr 2e-5, 3 эпохи.

Ворота: token-AUC ≥0.70, span-AUC ≥0.55, CI95 строго выше 0.5.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import realgen_common as rc  # noqa: E402
import realgen_publish as rp  # noqa: E402

sys.path.insert(0, str(rc.ROOT / "spanverify-module"))
from spanverify.core import tokenize_with_offsets  # noqa: E402
from spanverify.lexicon import STOPWORDS  # noqa: E402

OLD5 = ("attention_entropy", "ctx_attention_mass", "ctx_attention_mass_norm", "ctx_max_similarity", "embedding_density")
E1 = (
    "sentence_cosine_answer_doc",
    "sentence_cosine_answer_chunk_max",
    "contrast_doc_chunk",
    "rolling_logprob_next_token",
    "lexical_number_attribution_window",
    "stopword_norm",
)
R2 = (
    "digit_density_answer",
    "citation_density_answer",
    "token_is_numeric",
    "token_is_citation",
    "token_len_chars",
    "token_letters",
    "position_ratio",
    "sentence_length_chars",
    "local_number_density",
    "local_stopword_share",
    "doc_coverage",
    "distinct_digits",
    "uppercase_ratio",
    "punctuation_ratio",
    "unit_flag",
    "type_token_ratio_answer",
)
FEATURES = OLD5 + E1 + R2
CITATION_WORD_RE = re.compile(r"(?i)стать|пункт|част|глав|раздел|абзац|ст\.|п\.|ч\.")
UNIT_WORD_RE = re.compile(r"(?i)руб|процент|дн[ейя]|месяц|лет|год|час|минут|секунд|шт|коп|млн|млрд|тыс")
NUMBER_RE = re.compile(r"\d+")
SENT_RE = re.compile(r"[^.!?]+[.!?]*", re.S)
WORD_RE = re.compile(r"[а-яёa-z]{3,}", re.I)


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney AUC (устойчив к ties, векторизованные ранги)."""
    from scipy.stats import rankdata

    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    if pos == 0 or neg == 0:
        return 0.5
    ranks = rankdata(score)
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def bootstrap_ci(y: np.ndarray, score: np.ndarray, groups: np.ndarray, n_boot: int = 5000, seed: int = 42) -> list[float]:
    """Кластерный бутстрэп по документам/ответам (5000 итераций)."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    index_by_group = {g: np.flatnonzero(groups == g) for g in unique}
    values = []
    for _ in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_by_group[g] for g in chosen])
        values.append(auc(y[idx], score[idx]))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def sentence_windows(answer: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in SENT_RE.finditer(answer) if m.end() > m.start()]


def deterministic_features(answer: str, doc_text: str) -> tuple[list[tuple[int, int, str]], list[dict]]:
    """16 R2-признаков + часть E1-лексики на каждый(word) токен ответа."""
    tokens = tokenize_with_offsets(answer)
    words = [t.text.strip() for t in tokens]
    answer_len = max(1, len(answer))
    digit_density = sum(ch.isdigit() for ch in answer) / answer_len
    citation_density = len(CITATION_WORD_RE.findall(answer)) / answer_len * 100
    unique_words = len({w.lower() for w in words}) / max(1, len(words))
    doc_word_set = set(WORD_RE.findall(doc_text.lower()))
    sent_windows = sentence_windows(answer)
    out = []
    for i, tok in enumerate(tokens):
        start, end = tok.start, tok.end
        text = answer[start:end]
        lower = text.lower()
        sentence_len = next((e - s for s, e in sent_windows if s <= start < e), answer_len)
        local = answer[max(0, start - 80) : min(len(answer), end + 80)]
        local_numbers = NUMBER_RE.findall(local)
        neighbors = words[max(0, i - 5) : i + 6]
        local_stop = sum(1 for w in neighbors if w.lower() in STOPWORDS) / max(1, len(neighbors))
        context_words = WORD_RE.findall(local.lower())
        coverage = sum(1 for w in context_words if w in doc_word_set) / max(1, len(context_words))
        punct = sum(1 for ch in local if not ch.isalnum() and not ch.isspace()) / max(1, len(local))
        row = {
            "start": start,
            "end": end,
            "text": text,
            "digit_density_answer": digit_density,
            "citation_density_answer": citation_density,
            "token_is_numeric": float(any(ch.isdigit() for ch in text)),
            "token_is_citation": float(bool(CITATION_WORD_RE.search(text))),
            "token_len_chars": float(len(text)),
            "token_letters": float(sum(ch.isalpha() for ch in text)),
            "position_ratio": start / answer_len,
            "sentence_length_chars": float(sentence_len),
            "local_number_density": float(len(local_numbers)),
            "local_stopword_share": local_stop,
            "doc_coverage": coverage,
            "distinct_digits": float(len({ch for ch in text if ch.isdigit()})),
            "uppercase_ratio": sum(ch.isupper() for ch in text) / max(1, len(text)),
            "punctuation_ratio": punct,
            "unit_flag": float(bool(UNIT_WORD_RE.search(text)) or bool(UNIT_WORD_RE.search(answer[end : end + 12]))),
            "type_token_ratio_answer": unique_words,
            "lexical_number_attribution_window": float(
                sum(1 for number in local_numbers if number in doc_text) / max(1, len(local_numbers))
            ),
            "stopword_norm": local_stop / max(1e-6, unique_words),
        }
        out.append(row)
    return sent_windows, out


def char_labels(rows: list[dict], spans: list[dict]) -> np.ndarray:
    labels = np.zeros(len(rows), dtype=np.int8)
    for span in spans:
        for i, row in enumerate(rows):
            if row["start"] < span["end"] and span["start"] < row["end"]:
                labels[i] = 1
    return labels


def load_inputs() -> tuple[list[dict], dict, list[dict]]:
    answers = rc.read_jsonl(rc.REAL / "REALGEN_ANSWERS_DEV.jsonl")
    docs = {r["doc_id"]: r for r in rc.read_jsonl(rc.REAL / "REALGEN_DOCS.jsonl")}
    spans = rc.read_jsonl(rc.REAL / "REALGEN_SPANS.jsonl")
    by_answer: dict[str, list[dict]] = defaultdict(list)
    for span in spans:
        by_answer[span["answer_id"]].append(span)
    assert len(answers) == 300
    return answers, docs, by_answer


def hf_old5_and_e1(answers: list[dict], docs: dict, rows_by_answer: dict, tokenizer, model, torch, rubert_tok, rubert_model) -> None:
    """Один проход GPT-2 на ответ: old5 + rolling logprob; rubert: sentence cosine."""
    device = "cpu"
    for rec in answers:
        answer = rec["answer"]
        doc_text = docs[rec["doc_id"]]["text"][:800]
        prefix = doc_text + "\n" + answer
        encoded = tokenizer(
            prefix,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        input_ids = encoded["input_ids"]
        seq_len = input_ids.shape[1]
        ctx_len = max(1, sum(1 for s, _ in offsets if s < len(doc_text)))
        with torch.no_grad():
            outputs = model(**encoded, output_attentions=True, output_hidden_states=True)
        if not outputs.attentions or not outputs.hidden_states:
            raise RuntimeError(
                "модель не вернула attentions/hidden_states: нужен attn_implementation='eager' "
                "(SDPA не отдаёт веса внимания); старые 5 семейств без этого нечестны"
            )
        attn = outputs.attentions[-1][0].mean(dim=0)  # (seq, seq)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        probs = attn.clamp_min(1e-9)
        entropy = -(probs * probs.log()).sum(dim=-1) / math.log(max(2, seq_len))
        hidden = outputs.hidden_states[-1][0]
        hidden = torch.nn.functional.normalize(hidden, dim=-1)
        ctx_hidden = hidden[:ctx_len]
        logits = outputs.logits[0]
        rows = rows_by_answer[rec["answer_id"]]

        # rubert sentence/doc embeddings (mean pooling).
        sent_windows = sentence_windows(answer)
        sentences = [answer[s:e] for s, e in sent_windows] or [answer]
        chunks = [doc_text[i : i + 400] for i in range(0, min(len(doc_text), 2000), 400)] or [doc_text]
        sent_emb = _mean_embeddings(sentences, rubert_tok, rubert_model, torch, max_len=96)
        chunk_emb = _mean_embeddings(chunks, rubert_tok, rubert_model, torch, max_len=96)
        doc_emb = chunk_emb.mean(axis=0, keepdims=True)
        doc_emb = doc_emb / np.linalg.norm(doc_emb, axis=1, keepdims=True).clip(min=1e-9)
        sent_emb = sent_emb / np.linalg.norm(sent_emb, axis=1, keepdims=True).clip(min=1e-9)
        chunk_emb = chunk_emb / np.linalg.norm(chunk_emb, axis=1, keepdims=True).clip(min=1e-9)
        cos_doc = (sent_emb @ doc_emb.T).ravel()
        cos_chunks = sent_emb @ chunk_emb.T

        answer_offset = len(doc_text) + 1
        for row in rows:
            positions = [
                p
                for p, (s, e) in enumerate(offsets)
                if s >= answer_offset and (s - answer_offset) < row["end"] and (e - answer_offset) > row["start"] and p < seq_len
            ]
            pos = positions[0] if positions else min(seq_len - 1, ctx_len)
            sims = (hidden[pos : pos + 1] @ ctx_hidden.T)[0]
            row["attention_entropy"] = float(entropy[pos].item())
            row["ctx_attention_mass"] = float(attn[pos, :ctx_len].sum().item())
            row["ctx_attention_mass_norm"] = float(attn[pos, :ctx_len].sum().item() / max(1e-9, ctx_len / seq_len))
            row["ctx_max_similarity"] = float(sims.max().item())
            row["embedding_density"] = float(sims.mean().item())
            if pos > 0:
                token_logit = logits[pos - 1]
                logprob = token_logit[input_ids[0, pos]] - torch.logsumexp(token_logit, dim=-1)
                row["rolling_logprob_next_token"] = float(logprob.item())
            else:
                row["rolling_logprob_next_token"] = 0.0
            sent_idx = next((i for i, (s, e) in enumerate(sent_windows) if s <= row["start"] < e), 0)
            row["sentence_cosine_answer_doc"] = float(cos_doc[min(sent_idx, len(cos_doc) - 1)])
            row["sentence_cosine_answer_chunk_max"] = float(cos_chunks[min(sent_idx, cos_chunks.shape[0] - 1)].max())
            row["contrast_doc_chunk"] = row["sentence_cosine_answer_chunk_max"] - row["sentence_cosine_answer_doc"]
    return None


def _mean_embeddings(texts: list[str], tokenizer, model, torch, max_len: int = 96) -> np.ndarray:
    out = []
    for i in range(0, len(texts), 16):
        batch = texts[i : i + 16]
        encoded = tokenizer(batch, return_tensors="pt", truncation=True, max_length=max_len, padding=True)
        with torch.no_grad():
            hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        out.append(pooled.cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 312))


def rubert_finetune_oof(answers: list[dict], rows_by_answer: dict, positive_by_answer: dict, tokenizer, model_dir: str, torch, transformers) -> dict:
    """OOF 5-fold rubert-tiny2 токен-классификатор: batch 8, lr 2e-5, 3 эпохи."""
    from transformers import AutoModelForTokenClassification

    rng = random.Random(42)
    folds = {rec["answer_id"]: int(rc.h(f"42:fold:{rec['doc_id']}"), 16) % 5 for rec in answers}
    samples: list[dict] = []
    for rec in answers:
        answer = rec["answer"]
        spans = [(s, e) for s, e in sentence_windows(answer)]
        windows = spans or [(0, len(answer))]
        for start, end in windows:
            piece = answer[start:end]
            if len(piece.strip()) < 3:
                continue
            encoded = tokenizer(piece, return_offsets_mapping=True, truncation=True, max_length=96, add_special_tokens=True)
            offsets = encoded.pop("offset_mapping")
            positives = positive_by_answer.get(rec["answer_id"], [])
            labels = []
            for s, e in offsets:
                if e <= s:
                    labels.append(-100)
                    continue
                hit = any(ps < start + e and start + s < pe for ps, pe in positives)
                labels.append(1 if hit else 0)
            if not any(label >= 0 for label in labels):
                continue
            samples.append(
                {
                    "answer_id": rec["answer_id"],
                    "fold": folds[rec["answer_id"]],
                    "start": start,
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "offsets": offsets,
                    "labels": labels,
                }
            )

    preds_by_answer: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
    steps = 0
    for fold in range(5):
        train = [s for s in samples if s["fold"] != fold]
        val = [s for s in samples if s["fold"] == fold]
        if not train or not val:
            continue
        model = AutoModelForTokenClassification.from_pretrained(model_dir, num_labels=2)
        model.train()
        optim = torch.optim.AdamW(model.parameters(), lr=2e-5)
        order = list(range(len(train)))
        for epoch in range(3):
            rng.shuffle(order)
            for i in range(0, len(order), 8):
                batch = [train[j] for j in order[i : i + 8]]
                max_len = max(len(s["input_ids"]) for s in batch)
                input_ids = torch.tensor([s["input_ids"] + [tokenizer.pad_token_id or 0] * (max_len - len(s["input_ids"])) for s in batch])
                mask = torch.tensor([s["attention_mask"] + [0] * (max_len - len(s["attention_mask"])) for s in batch])
                labels = torch.tensor([s["labels"] + [-100] * (max_len - len(s["labels"])) for s in batch])
                outputs = model(input_ids=input_ids, attention_mask=mask, labels=labels)
                outputs.loss.backward()
                optim.step()
                optim.zero_grad()
                steps += 1
        model.eval()
        with torch.no_grad():
            for s in val:
                ids = torch.tensor([s["input_ids"]])
                mask = torch.tensor([s["attention_mask"]])
                logits = model(input_ids=ids, attention_mask=mask).logits[0]
                probs = torch.softmax(logits, dim=-1)[:, 1].tolist()
                for (a, b), p in zip(s["offsets"], probs):
                    if b > a:
                        preds_by_answer[s["answer_id"]].append((s["start"] + a, s["start"] + b, float(p)))
        del model
    return {"preds": preds_by_answer, "train_steps": steps, "n_samples": len(samples), "folds": 5}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="без HF: случайные признаки для проверки логики")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--out-dir", default=".arena/realgen_b")
    parser.add_argument("--limit", type=int, default=0, help="smoke-test: только первые N dev-ответов")
    args = parser.parse_args()

    started = time.time()
    answers, docs, spans_by_answer = load_inputs()
    if args.limit:
        answers = answers[: args.limit]
    assert len(answers) == (args.limit or 300)
    print(f"dev answers: {len(answers)}", flush=True)

    rows_by_answer: dict[str, list[dict]] = {}
    labels_by_answer: dict[str, np.ndarray] = {}
    windows_by_answer: dict[str, list[tuple[int, int]]] = {}
    for rec in answers:
        windows, rows = deterministic_features(rec["answer"], docs[rec["doc_id"]]["text"])
        windows_by_answer[rec["answer_id"]] = windows
        rows_by_answer[rec["answer_id"]] = rows
        labels_by_answer[rec["answer_id"]] = char_labels(rows, spans_by_answer.get(rec["answer_id"], []))
        for row in rows:
            for name in OLD5 + ("sentence_cosine_answer_doc", "sentence_cosine_answer_chunk_max", "contrast_doc_chunk", "rolling_logprob_next_token"):
                row[name] = 0.0

    environment = {"python": platform.python_version(), "platform": platform.platform(), "dry_run": args.dry_run}
    ft = {"preds": {}, "train_steps": 0, "n_samples": 0, "folds": 5}
    if not args.dry_run:
        import torch
        import transformers
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(4)
        gpt_dir = snapshot_download(rc.MODEL)
        rubert_name = "cointegrated/rubert-tiny2"
        rubert_dir = snapshot_download(rubert_name)
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
        gpt_files = [{"name": p.name, "bytes": p.stat().st_size} for p in Path(gpt_dir).rglob("*") if p.is_file() and p.name in ("pytorch_model.bin", "model.safetensors")]
        rubert_files = [{"name": p.name, "bytes": p.stat().st_size} for p in Path(rubert_dir).rglob("*") if p.is_file() and p.name in ("pytorch_model.bin", "model.safetensors")]
        environment.update(
            {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "gpt_model": rc.MODEL,
                "gpt_weight_files": gpt_files,
                "rubert_model": rubert_name,
                "rubert_weight_files": rubert_files,
                "rubert_weights_mb": round((next((f for f in rubert_files if f["name"] == "model.safetensors"), rubert_files[0])["bytes"]) / 1_000_000, 3),
                "run_id": os.environ.get("GITHUB_RUN_ID", "local-no-ci-evidence"),
                "head_sha": os.environ.get("HEAD_SHA", ""),
            }
        )
        print("compute old5 + E1 features...", flush=True)
        hf_old5_and_e1(answers, docs, rows_by_answer, gpt_tok, gpt_model, torch, rubert_tok, rubert_model)
        del gpt_model
        print("fine-tune rubert-tiny2 OOF...", flush=True)
        positive_by_answer = {
            rec["answer_id"]: [
                (row["start"], row["end"])
                for row, label in zip(rows_by_answer[rec["answer_id"]], labels_by_answer[rec["answer_id"]])
                if label == 1
            ]
            for rec in answers
        }
        ft = rubert_finetune_oof(answers, rows_by_answer, positive_by_answer, rubert_tok, rubert_dir, torch, transformers)
        print(f"fine-tune steps: {ft['train_steps']}", flush=True)
    else:
        rng = np.random.default_rng(42)
        for rec in answers:
            for row in rows_by_answer[rec["answer_id"]]:
                for name in OLD5 + E1:
                    row[name] = float(rng.normal())

    # Сборка матрицы.
    all_rows: list[dict] = []
    for rec in answers:
        for row in rows_by_answer[rec["answer_id"]]:
            row = dict(row)
            row["answer_id"] = rec["answer_id"]
            row["doc_id"] = rec["doc_id"]
            all_rows.append(row)
    y = np.concatenate([labels_by_answer[rec["answer_id"]] for rec in answers])
    doc_group = np.concatenate([np.full(len(rows_by_answer[rec["answer_id"]]), rec["doc_id"], dtype=object) for rec in answers])
    X = np.array([[float(row.get(name, 0.0)) for name in FEATURES] for row in all_rows], dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"tokens: {len(all_rows)}, positives: {int(y.sum())}, features: {X.shape[1]}", flush=True)

    per_feature = []
    for index, name in enumerate(FEATURES):
        value = X[:, index]
        a = auc(y, value)
        per_feature.append(
            {
                "feature": name,
                "family": "old5" if name in OLD5 else ("E1" if name in E1 else "R2"),
                "auc": round(a, 6),
                "auc_oriented": round(max(a, 1 - a), 6),
                "orientation": "higher=risk" if a >= 0.5 else "lower=risk",
            }
        )

    # Composite: логистическая регрессия OOF 5-fold по sha256(doc_id).
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    fold_of_answer = {rec["answer_id"]: int(rc.h(f"42:fold:{rec['doc_id']}"), 16) % 5 for rec in answers}
    folds = np.array([fold_of_answer[row["answer_id"]] for row in all_rows])
    composite = np.zeros(len(all_rows))
    for fold in range(5):
        train, test = folds != fold, folds == fold
        if train.sum() == 0 or test.sum() == 0 or len(np.unique(y[train])) < 2:
            composite[test] = 0.5
            continue
        scaler = StandardScaler().fit(X[train])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        clf.fit(scaler.transform(X[train]), y[train])
        composite[test] = clf.predict_proba(scaler.transform(X[test]))[:, 1]

    # rubert OOF → probability per word token.
    rubert_score = np.full(len(all_rows), 0.5)
    pos = 0
    for rec in answers:
        rows = rows_by_answer[rec["answer_id"]]
        preds = ft["preds"].get(rec["answer_id"], [])
        for row in rows:
            values = [p for s, e, p in preds if s < row["end"] and row["start"] < e]
            rubert_score[pos] = float(np.mean(values)) if values else 0.5
            pos += 1

    # Span AUC: предложения как кандидаты спана.
    span_y, comp_span, ft_span = [], [], []
    cursor = 0
    for rec in answers:
        rows = rows_by_answer[rec["answer_id"]]
        for s, e in windows_by_answer[rec["answer_id"]]:
            idx = [i for i, row in enumerate(rows) if row["start"] < e and s < row["end"]]
            if not idx:
                continue
            span_y.append(1 if any(y[cursor + i] == 1 for i in idx) else 0)
            comp_span.append(float(np.mean([composite[cursor + i] for i in idx])))
            ft_span.append(float(np.mean([rubert_score[cursor + i] for i in idx])))
        cursor += len(rows)
    span_y = np.array(span_y, dtype=np.int8)
    comp_span = np.array(comp_span)
    ft_span = np.array(ft_span)

    n_boot = args.bootstrap if not args.dry_run else 200
    ci_comp = bootstrap_ci(y, composite, doc_group, n_boot=n_boot)
    ci_ft = bootstrap_ci(y, rubert_score, doc_group, n_boot=n_boot)
    span_groups = np.array([f"span-{i}" for i in range(len(span_y))], dtype=object)
    ci_comp_span = bootstrap_ci(span_y, comp_span, span_groups, n_boot=n_boot)
    ci_ft_span = bootstrap_ci(span_y, ft_span, span_groups, n_boot=n_boot)

    token_auc_comp, token_auc_ft = auc(y, composite), auc(y, rubert_score)
    span_auc_comp, span_auc_ft = auc(span_y, comp_span), auc(span_y, ft_span)
    gates = {
        "token_AUC_min": 0.70,
        "span_AUC_min": 0.55,
        "ci95_strictly_above_0.5": True,
        "composite": {
            "token_AUC": round(token_auc_comp, 6),
            "span_AUC": round(span_auc_comp, 6),
            "token_gate": bool(token_auc_comp >= 0.70),
            "span_gate": bool(span_auc_comp >= 0.55),
            "ci_gate": bool(ci_comp[0] > 0.5 and ci_comp_span[0] > 0.5),
        },
        "rubert_tiny2_finetune": {
            "token_AUC": round(token_auc_ft, 6),
            "span_AUC": round(span_auc_ft, 6),
            "token_gate": bool(token_auc_ft >= 0.70),
            "span_gate": bool(span_auc_ft >= 0.55),
            "ci_gate": bool(ci_ft[0] > 0.5 and ci_ft_span[0] > 0.5),
        },
        "composite_pass": bool(
            token_auc_comp >= 0.70 and span_auc_comp >= 0.55 and ci_comp[0] > 0.5 and ci_comp_span[0] > 0.5
        ),
        "rubert_pass": bool(
            token_auc_ft >= 0.70 and span_auc_ft >= 0.55 and ci_ft[0] > 0.5 and ci_ft_span[0] > 0.5
        ),
        "rule": "verdict=PASS только если ОБЕ модели (composite и rubert-tiny2 FT) проходят все ворота",
        "verdict": "PASS" if (
            token_auc_comp >= 0.70 and span_auc_comp >= 0.55 and ci_comp[0] > 0.5 and ci_comp_span[0] > 0.5
            and token_auc_ft >= 0.70 and span_auc_ft >= 0.55 and ci_ft[0] > 0.5 and ci_ft_span[0] > 0.5
        ) else "FAIL",
    }

    result = {
        "kind": "REALGEN_PHASE_B",
        "version": "v1",
        "campaign": "REALGEN-R1",
        "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "date": "2026-10-26",
        "run_id": os.environ.get("GITHUB_RUN_ID", "local-no-ci-evidence"),
        "evidence_context": {
            "run_id": os.environ.get("GITHUB_RUN_ID", "local-no-ci-evidence"),
            "date": "2026-10-26",
            "status": "authoritative",
            "source": "GitHub Actions phase B measurement (only dev REALGEN)",
            "labeling": "deterministic atom rules from frozen phase A",
        },
        "data": {
            "smoke_test": bool(args.limit),
            "dev_answers": len(answers),
            "n_tokens": int(len(all_rows)),
            "n_positive_tokens": int(y.sum()),
            "n_span_windows": int(len(span_y)),
            "n_positive_span_windows": int(span_y.sum()),
            "folds": 5,
            "fold_rule": "int(sha256('42:fold:'+doc_id),16) % 5",
            "bootstrap": n_boot,
            "bootstrap_unit": "cluster by doc_id for tokens, by window for spans",
        },
        "per_feature_AUC": per_feature,
        "composite_OOF": {
            "model": "LogisticRegression(class_weight=balanced, max_iter=1000) on 27 features, StandardScaler",
            "features": list(FEATURES),
            "token_AUC": round(token_auc_comp, 6),
            "token_ci95": [round(v, 6) for v in ci_comp],
            "span_AUC": round(span_auc_comp, 6),
            "span_ci95": [round(v, 6) for v in ci_comp_span],
        },
        "rubert_tiny2_finetune": {
            "model": "cointegrated/rubert-tiny2 AutoModelForTokenClassification",
            "hyperparams": {"batch": 8, "lr": 2e-5, "epochs": 3, "max_length": 96, "OOF_folds": 5},
            "weights_mb": environment.get("rubert_weights_mb"),
            "train_steps": ft["train_steps"],
            "n_windows": ft["n_samples"],
            "token_AUC": round(token_auc_ft, 6),
            "token_ci95": [round(v, 6) for v in ci_ft],
            "span_AUC": round(span_auc_ft, 6),
            "span_ci95": [round(v, 6) for v in ci_ft_span],
            "note": "OOF вероятности по сабтокенам агрегированы в word-токены средним; train_steps=0 означает dry-run",
        },
        "gates": gates,
        "caveat_label_leakage": (
            "Метки REALGEN получены детерминированными правилами по числам/ссылкам, поэтому "
            "лексические признаки (token_is_numeric, local_number_density, citation_density) "
            "ожидаемо дают высокий AUC. Это НЕ доказательство обнаружения галлюцинаций "
            "в свободной генерации и НЕ перенос на real_test 1200; ворота измеряют только "
            "внутреннюю separability этого корпуса."
        ),
        "old5_definitions": "пересчитаны на ai-forever/rugpt3small_based_on_gpt2 (last-layer attention / hidden states) — не вызов замороженного hf_features_chunked, определения сохранены",
        "E1_R2_note": "E1/R2 признаки — детерминированная реимплементация по опубликованным именам в E0_DIAG/E1_SIGNAL; исходный код E1HUNT_R2/R3 в этом чекауте отсутствует (проверено по trees веток)",
        "environment": environment,
        "runtime_seconds": round(time.time() - started, 3),
        "real_test_viewed": False,
        "provenance": {
            "run_id": os.environ.get("GITHUB_RUN_ID", "local-no-ci-evidence"),
            "head_sha": os.environ.get("HEAD_SHA", ""),
            "inputs": {
                "REALGEN_FROZEN.json": rc.sha256_file(rc.REAL / "REALGEN_FROZEN.json"),
                "REALGEN_ANSWERS_DEV.jsonl": rc.sha256_file(rc.REAL / "REALGEN_ANSWERS_DEV.jsonl"),
                "REALGEN_SPANS.jsonl": rc.sha256_file(rc.REAL / "REALGEN_SPANS.jsonl"),
                "REALGEN_DOCS.jsonl": rc.sha256_file(rc.REAL / "REALGEN_DOCS.jsonl"),
            },
            "test_realgen_file_read": False,
            "real_test_locked_file_read": False,
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "REALGEN_PHASE_B.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    oof_path = out_dir / "REALGEN_PHASE_B_OOF.jsonl"
    with oof_path.open("w", encoding="utf-8") as handle:
        for i, row in enumerate(all_rows):
            handle.write(
                json.dumps(
                    {
                        "answer_id": row["answer_id"],
                        "doc_id": row["doc_id"],
                        "start": row["start"],
                        "end": row["end"],
                        "label": int(y[i]),
                        "composite": round(float(composite[i]), 6),
                        "rubert_ft": round(float(rubert_score[i]), 6),
                        "fold": int(folds[i]),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    print(json.dumps({"gates": gates, "token_auc_comp": token_auc_comp, "token_auc_ft": token_auc_ft}, ensure_ascii=False), flush=True)
    rp.step_summary(f"### REALGEN Phase B\n\n```json\n{json.dumps(gates, ensure_ascii=False, indent=1)}\n```")

    if args.publish:
        head_sha = os.environ.get("HEAD_SHA", "local")
        repo = os.environ.get("GITHUB_REPOSITORY", "Igorr112323/Smarty_Chumakov")
        rp.publish_file(result_path, "REALGEN-B/result", head_sha, repo, {"gates": gates})
        rp.publish_file(oof_path, "REALGEN-B/oof", head_sha, repo, {"n_tokens": len(all_rows)})
    print(f"result={result_path} oof={oof_path}")


if __name__ == "__main__":
    main()
