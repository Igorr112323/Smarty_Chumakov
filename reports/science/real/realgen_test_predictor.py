#!/usr/bin/env python3
"""Frozen REALGEN inference backend shipped INSIDE the model bundle.

The official evaluator imports this file from the verified zip as
``realgen_test_predict.py`` and calls ``predict_all(records, docs, model_dir)``.
It never fits, tunes, calibrates or reads sealed label/span files. It only
recomputes the preregistered 27 features with pinned upstream HF revisions and
applies the frozen coefficients / fine-tuned weights from the same bundle.

This source is committed at reports/science/real/realgen_test_predictor.py so it
can be reviewed; the freeze workflow copies it verbatim into the zip and records
its SHA256 in the bundle manifest.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

# The official workflow sets PYTHONPATH=spanverify-module; the core tokenizer is
# the same one the official evaluator uses to verify returned token offsets.
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
MODEL_NAMES = ("composite", "rubert_tiny2_ft", "r2_only")


def sentence_windows(answer: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in SENT_RE.finditer(answer) if m.end() > m.start()]


def deterministic_features(answer: str, doc_text: str) -> tuple[list[tuple[int, int, str]], list[dict]]:
    """The 16 R2 features + E1 lexical features, word-token aligned (Phase B)."""
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


def hf_old5_and_e1(answers: list[dict], docs: dict, rows_by_answer: dict, tokenizer, model, torch, rubert_tok, rubert_model) -> None:
    """Identical old5/E1 pass to Phase B; pinned revisions are supplied by caller."""
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
            raise RuntimeError("eager attention is required to recompute the frozen old5 features")
        attn = outputs.attentions[-1][0].mean(dim=0)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        probs = attn.clamp_min(1e-9)
        entropy = -(probs * probs.log()).sum(dim=-1) / math.log(max(2, seq_len))
        hidden = outputs.hidden_states[-1][0]
        hidden = torch.nn.functional.normalize(hidden, dim=-1)
        ctx_hidden = hidden[:ctx_len]
        logits = outputs.logits[0]
        rows = rows_by_answer[rec["answer_id"]]

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


def _expit(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def _logreg_probabilities(rows: list[dict], spec: dict) -> list[float]:
    names = spec["features"]
    matrix = np.array([[float(row.get(name, 0.0)) for name in names] for row in rows], dtype=np.float64)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    mean = np.asarray(spec["scaler_mean"], dtype=np.float64)
    scale = np.asarray(spec["scaler_scale"], dtype=np.float64)
    coef = np.asarray(spec["coefficients"], dtype=np.float64)
    decision = ((matrix - mean) / scale) @ coef + float(spec["intercept"])
    return [float(v) for v in _expit(decision)]


def _ft_windows(records: list[dict], tokenizer, max_length: int) -> list[dict]:
    samples: list[dict] = []
    for rec in records:
        answer = rec["answer"]
        windows = [(s, e) for s, e in sentence_windows(answer)] or [(0, len(answer))]
        for start, end in windows:
            piece = answer[start:end]
            if len(piece.strip()) < 3:
                continue
            encoded = tokenizer(
                piece, return_offsets_mapping=True, truncation=True, max_length=max_length, add_special_tokens=True
            )
            offsets = encoded.pop("offset_mapping")
            samples.append(
                {
                    "answer_id": rec["answer_id"],
                    "start": start,
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "offsets": offsets,
                }
            )
    return samples


def _ft_scores(records: list[dict], rows_by_answer: dict, model_dir: Path, manifest: dict, torch) -> dict[str, list[float]]:
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    ft_dir = model_dir / manifest["rubert_tiny2_ft"]["directory"]
    tokenizer = AutoTokenizer.from_pretrained(str(ft_dir), use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(str(ft_dir))
    model.eval()
    samples = _ft_windows(records, tokenizer, int(manifest["rubert_tiny2_ft"]["max_length"]))
    preds: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
    with torch.no_grad():
        for i in range(0, len(samples), int(manifest["rubert_tiny2_ft"]["batch_size"])):
            batch = samples[i : i + int(manifest["rubert_tiny2_ft"]["batch_size"])]
            max_len = max(len(s["input_ids"]) for s in batch)
            ids = torch.tensor(
                [s["input_ids"] + [tokenizer.pad_token_id or 0] * (max_len - len(s["input_ids"])) for s in batch]
            )
            mask = torch.tensor([s["attention_mask"] + [0] * (max_len - len(s["attention_mask"])) for s in batch])
            logits = model(input_ids=ids, attention_mask=mask).logits
            probs = torch.softmax(logits, dim=-1)[:, :, 1]
            for sample, row in zip(batch, probs, strict=True):
                for (a, b), value in zip(sample["offsets"], row.tolist(), strict=True):
                    if b > a:
                        preds[sample["answer_id"]].append((sample["start"] + a, sample["start"] + b, float(value)))
    scores: dict[str, list[float]] = {}
    for rec in records:
        aid = rec["answer_id"]
        answer_preds = preds.get(aid, [])
        scores[aid] = [
            float(np.mean([p for s, e, p in answer_preds if s < row["end"] and row["start"] < e]))
            if any(s < row["end"] and row["start"] < e for s, e, _ in answer_preds)
            else 0.5
            for row in rows_by_answer[aid]
        ]
    return scores


def predict_all(records: list[dict], docs: dict, model_dir: Path) -> tuple[dict, dict]:
    """Return ({model: per-answer token scores}, {model: per-answer token offsets})."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    model_dir = Path(model_dir)
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("trained_on") != "realgen_dev_300_only" or manifest.get("status") != "FROZEN":
        raise RuntimeError("frozen bundle is not a dev-only REALGEN bundle")
    if manifest.get("token_threshold") != 0.5:
        raise RuntimeError("frozen token threshold differs from preregistration")

    upstream = manifest["upstream_revisions"]
    gpt_dir = snapshot_download(upstream["old5_gpt"]["repo_id"], revision=upstream["old5_gpt"]["revision"])
    rubert_dir = snapshot_download(
        upstream["e1_rubert_base"]["repo_id"], revision=upstream["e1_rubert_base"]["revision"]
    )
    torch.set_num_threads(4)
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

    rows_by_answer: dict[str, list[dict]] = {}
    offsets_by_answer: dict[str, list[tuple[int, int]]] = {}
    for rec in records:
        _windows, rows = deterministic_features(rec["answer"], docs[rec["doc_id"]]["text"])
        tokens = tokenize_with_offsets(rec["answer"])
        if len(rows) != len(tokens):
            raise RuntimeError("frozen feature/tokenizer alignment failed")
        for row in rows:
            for name in OLD5 + (
                "sentence_cosine_answer_doc",
                "sentence_cosine_answer_chunk_max",
                "contrast_doc_chunk",
                "rolling_logprob_next_token",
            ):
                row[name] = 0.0
        rows_by_answer[rec["answer_id"]] = rows
        offsets_by_answer[rec["answer_id"]] = [(t.start, t.end) for t in tokens]

    hf_old5_and_e1(records, docs, rows_by_answer, gpt_tok, gpt_model, torch, rubert_tok, rubert_model)
    del gpt_model, rubert_model
    ft_scores = _ft_scores(records, rows_by_answer, model_dir, manifest, torch)

    predictions: dict[str, list[list[float]]] = {
        "composite": [],
        "r2_only": [],
        "rubert_tiny2_ft": [],
    }
    for rec in records:
        rows = rows_by_answer[rec["answer_id"]]
        predictions["composite"].append(_logreg_probabilities(rows, manifest["models"]["composite"]))
        predictions["r2_only"].append(_logreg_probabilities(rows, manifest["models"]["r2_only"]))
        predictions["rubert_tiny2_ft"].append(ft_scores[rec["answer_id"]])
    offsets = {
        name: [list(offsets_by_answer[rec["answer_id"]]) for rec in records] for name in MODEL_NAMES
    }
    for name in MODEL_NAMES:
        if len(predictions[name]) != len(records) or any(
            len(scores) != len(rows_by_answer[rec["answer_id"]])
            for scores, rec in zip(predictions[name], records, strict=True)
        ):
            raise RuntimeError(f"prediction length mismatch for {name}")
    return predictions, offsets
