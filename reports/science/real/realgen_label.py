#!/usr/bin/env python3
"""REALGEN-R1 фаза A, шаг 3 (локально): авторазметка атомов, квоты, заморозка.

Правила (НЕ меняются при нехватке ошибок):
  (a) число/дата/сумма/срок из ответа отсутствует в акте → спан;
  (b) цитата «статья/пункт/часть N», которой нет в акте → спан;
  (c) контрадикция для той же сущности (окно Jaccard ≥0.6) с другим числом → спан;
  (d) «неясные» атомы НЕ размечать; публикуется atom_unclear_rate.
Спаны: char offsets [start:end] в ответе, rule_id.

При нехватке 40% ошибочных ответов используется заранее сгенерированный
раунд R2 с повышенной сложностью промта (разрешённая правка РЕЖИМА
генерации), правила разметки остаются побайтно теми же.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import realgen_common as rc  # noqa: E402

RULE_C_MODE = "any_window"  # буквальное чтение директивы; "best_window" — только sensitivity
SENSITIVITY_RULE_C_MODE = "best_window"

CITATION_RE = re.compile(
    r"(?i)(стать[яиейю]|ст\.|пункт[а-я]*|п\.|част[ьию]|ч\.|подпункт[а-я]*|пп\.|глав[аы]|раздел[а-я]*|абзац[а-я]*)"
    r"\s*(?:№\s*)?(\d+(?:[.\-]\d+)*)"
)
NUMBER_RE = re.compile(r"(?<![\w])\d+(?:[.,]\d+)*")
WORD_RE = re.compile(r"[а-яёa-z]{3,}", re.I)
UNIT_RE = re.compile(
    r"(?i)\s*(процент[а-я]*|%|рубл[а-я]*|руб\.?|копе[йяк]+|тыс\.?|млн|млрд|дн[ейя]+|месяц[ев]*|"
    r"лет|год[ау]?|час[ов]*|минут[ы]*|секунд[ы]*|раз|крат|шт\.?|человек|единиц[ы]*)"
)
UNCLEAR_MARKERS = (
    "около",
    "примерно",
    "приблизительно",
    "не менее",
    "не более",
    "и т.д",
    "и тому подоб",
    "некоторые",
    "несколько",
    "как правило",
    "возможно",
    "в зависимости",
    "соответствующ",
    "установленн",
    "ряд случаев",
    "многие",
    "либо",
    "или более",
    "от ",
)
STOPWORDS = frozenset(
    "для лицо лицо которое который которые которые которые данном данного данного настоящего настоящем "
    "также может быть были было есть при этом случае случае порядке соответствии статья статьей пункт часть "
    "федерального закона российского российской Федерации".split()
)


def norm_value(text: str) -> str:
    value = text.strip().replace(" ", "").replace(" ", "").replace(",", ".")
    return value.strip(".")


def context_tokens(text: str, start: int, end: int, window: int = 90) -> frozenset[str]:
    left = text[max(0, start - window) : start]
    right = text[end : min(len(text), end + window)]
    words = WORD_RE.findall((left + " " + right).lower())
    return frozenset(word for word in words if word not in STOPWORDS and len(word) >= 3)


class ActIndex:
    """Индекс числовых атомов акта: поддержка (a/b) и окна Jaccard (c)."""

    def __init__(self, text: str):
        self.text = text
        self.numbers: set[str] = set()
        self.citations: set[str] = set()
        self.windows: list[tuple[str, frozenset[str]]] = []
        self.inverted: dict[str, set[int]] = defaultdict(set)
        for match in CITATION_RE.finditer(text):
            self.citations.add(norm_value(match.group(2)))
        for match in NUMBER_RE.finditer(text):
            value = norm_value(match.group(0))
            if not value or len(value) > 24:
                continue
            self.numbers.add(value)
            tokens = context_tokens(text, match.start(), match.end())
            idx = len(self.windows)
            self.windows.append((value, tokens))
            for token in tokens:
                self.inverted[token].add(idx)

    def contradiction(self, value: str, tokens: frozenset[str], threshold: float = 0.6, mode: str = RULE_C_MODE) -> str | None:
        """Правило (c): окно акта с Jaccard≥0.6 и другим числом.

        ``mode='any_window'`` (ЗАМОРОЖЕН, буквальное чтение директивы): любое
        окно с Jaccard≥0.6 и другим числом даёт контрадикцию.
        ``mode='best_window'`` (только sensitivity-анализ, НЕ заморозка):
        более строгая трактовка «той же сущности» — берётся окно с максимальным
        Jaccard; противоречие только если у него другое число. Публикуется
        отдельно, чтобы куратор видел чувствительность квоты к трактовке.
        """
        if not tokens:
            return None
        candidates: set[int] = set()
        for token in tokens:
            candidates |= self.inverted.get(token, set())
        best_j, best_value = 0.0, None
        for idx in candidates:
            other_value, other_tokens = self.windows[idx]
            union = len(tokens | other_tokens)
            if not union:
                continue
            j = len(tokens & other_tokens) / union
            if mode == "any_window" and j >= threshold and other_value != value:
                return other_value
            if j > best_j:
                best_j, best_value = j, other_value
        if mode == "best_window" and best_j >= threshold and best_value != value:
            return best_value
        return None


def extract_atoms(answer: str) -> list[dict]:
    atoms: list[dict] = []
    occupied: list[tuple[int, int]] = []

    def overlaps(start: int, end: int) -> bool:
        return any(start < o_end and o_start < end for o_start, o_end in occupied)

    for match in CITATION_RE.finditer(answer):
        start, end = match.span()
        if overlaps(start, end):
            continue
        occupied.append((start, end))
        atoms.append(
            {
                "kind": "citation",
                "start": start,
                "end": end,
                "text": answer[start:end],
                "value": norm_value(match.group(2)),
                "citation_word": match.group(1),
            }
        )
    for match in NUMBER_RE.finditer(answer):
        start, end = match.span()
        if overlaps(start, end):
            continue
        unit = UNIT_RE.match(answer, end)
        if unit:
            end = unit.end()
        # Диапазон «5-10» — неоднозначный атом (d).
        between = answer[end : end + 3]
        is_range = bool(re.match(r"^\s*[-—–]\s*\d", between))
        occupied.append((start, end))
        atoms.append(
            {
                "kind": "number",
                "start": start,
                "end": end,
                "text": answer[start:end],
                "value": norm_value(match.group(0)),
                "unit": unit.group(1) if unit else "",
                "is_range": is_range,
            }
        )
    atoms.sort(key=lambda atom: atom["start"])
    return atoms


def is_unclear(answer: str, atom: dict) -> bool:
    if atom.get("is_range", False):
        return True
    window = answer[max(0, atom["start"] - 90) : min(len(answer), atom["end"] + 90)].lower()
    return any(marker in window for marker in UNCLEAR_MARKERS)


def label_answer(record: dict, act: ActIndex, rule_c_mode: str = RULE_C_MODE) -> dict:
    answer = record["answer"]
    atoms = extract_atoms(answer)
    spans: list[dict] = []
    unclear = 0
    for atom in atoms:
        if is_unclear(answer, atom):
            unclear += 1
            continue  # (d) неясные НЕ размечаем
        tokens = context_tokens(answer, atom["start"], atom["end"])
        if atom["kind"] == "citation":
            if atom["value"] not in act.citations:
                rule_id, evidence = "b", f"citation {atom['value']} отсутствует в акте"
            else:
                conflict = act.contradiction(atom["value"], tokens, mode=rule_c_mode)
                rule_id, evidence = ("c", f"act_number={conflict}") if conflict else (None, "")
        else:
            if atom["value"] not in act.numbers:
                rule_id, evidence = "a", f"number {atom['value']} отсутствует в акте"
            else:
                conflict = act.contradiction(atom["value"], tokens, mode=rule_c_mode)
                rule_id, evidence = ("c", f"act_number={conflict}") if conflict else (None, "")
        if rule_id:
            spans.append(
                {
                    "answer_id": record["answer_id"],
                    "doc_id": record["doc_id"],
                    "start": atom["start"],
                    "end": atom["end"],
                    "text": atom["text"],
                    "rule_id": rule_id,
                    "atom_kind": atom["kind"],
                    "atom_value": atom["value"],
                    "evidence": evidence,
                }
            )
    # Устраняем возможные пересечения спанов (приоритет более ранней позиции).
    spans.sort(key=lambda span: (span["start"], span["end"]))
    deduped = []
    for span in spans:
        if deduped and span["start"] < deduped[-1]["end"]:
            continue
        deduped.append(span)
    category = "erroneous" if deduped else ("unclear" if unclear else "clean")
    return {
        "answer_id": record["answer_id"],
        "doc_id": record["doc_id"],
        "round": record["round"],
        "template_id": record["template_id"],
        "mode": record["mode"],
        "category": category,
        "n_atoms": len(atoms),
        "n_unclear_atoms": unclear,
        "n_spans": len(deduped),
        "rule_ids": sorted({span["rule_id"] for span in deduped}),
        "spans": deduped,
        "answer_len": len(answer),
    }


def median(values: list[int | float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-dir", default=".arena/realgen_extract")
    parser.add_argument("--run-id", default="local-no-ci-evidence")
    parser.add_argument("--ci-run-ids", default="")
    args = parser.parse_args()

    extract_dir = Path(args.extract_dir)
    r1_files = sorted(extract_dir.glob("REALGEN-A__R1__shard-*.jsonl"))
    r2_files = sorted(extract_dir.glob("REALGEN-A__R2__shard-*.jsonl"))
    if not r1_files:
        raise SystemExit("нет извлечённых файлов R1 — сначала realgen_extract.py")

    def load(files: list[Path]) -> list[dict]:
        rows = []
        for path in files:
            rows.extend(rc.read_jsonl(path))
        return rows

    r1, r2 = load(r1_files), load(r2_files)
    selection = json.loads((rc.REAL / "REALGEN_SELECTION.json").read_text(encoding="utf-8"))
    docs = {row["doc_id"]: row for row in rc.read_jsonl(rc.REAL / "REALGEN_DOCS.jsonl")}
    assert {r["doc_id"] for r in r1} == set(selection["doc_ids"]), "несовпадение doc_id выборки"
    assert len(r1) == 400, f"R1 ответов: {len(r1)}"
    assert len({r["answer_id"] for r in r1}) == len(r1)

    print(f"R1={len(r1)} R2={len(r2)}; строю индексы актов по одному документу...", flush=True)

    def label_all(rows: list[dict], rule_c_mode: str = RULE_C_MODE) -> list[dict]:
        """Индекс акта строится на документ и сразу освобождается (память 400 актов не нужна)."""
        grouped: dict[str, list[tuple[int, dict]]] = defaultdict(list)
        for pos, rec in enumerate(rows):
            grouped[rec["doc_id"]].append((pos, rec))
        out: list[dict | None] = [None] * len(rows)
        for doc_id, items in grouped.items():
            act = ActIndex(docs[doc_id]["text"])
            for pos, rec in items:
                out[pos] = label_answer(rec, act, rule_c_mode)
        assert all(row is not None for row in out)
        return out

    labels_r1 = label_all(r1)
    err_r1 = sum(row["category"] == "erroneous" for row in labels_r1)
    share_r1 = err_r1 / len(labels_r1)
    print(f"R1 ошибочных: {err_r1}/{len(labels_r1)} = {share_r1:.4f}", flush=True)

    use_r2 = False
    if share_r1 >= 0.40:
        answers, labels = r1, labels_r1
        remedy = "не потребовался: R1 уже даёт ≥40% ошибочных ответов"
    else:
        if not r2_files:
            raise SystemExit("квота <40% и нет раунда R2 — STOP (правила разметки не меняем)")
        use_r2 = True
        labels_r2 = label_all(r2)
        answers, labels = r1 + r2, labels_r1 + labels_r2
        err_all = sum(row["category"] == "erroneous" for row in labels)
        share_all = err_all / len(labels)
        print(f"R1+R2 ошибочных: {err_all}/{len(labels)} = {share_all:.4f}", flush=True)
        remedy = (
            f"R1 дал {share_r1:.4f} < 0.40; подключён заранее сгенерированный R2 "
            f"(повышенная сложность промта, те же 2 режима), итог {share_all:.4f}. "
            "Правила разметки не менялись."
        )

    # Чувствительность квоты к трактовке правила (c) — публикуется открыто.
    def err_share(rows: list[dict]) -> tuple[int, int, float]:
        n_err_local = sum(row["category"] == "erroneous" for row in rows)
        return n_err_local, len(rows), n_err_local / max(1, len(rows))

    sens_best_r1 = err_share(label_all(r1, SENSITIVITY_RULE_C_MODE))
    sens_best_all = err_share(label_all(answers, SENSITIVITY_RULE_C_MODE)) if use_r2 else None
    ab_only = sum(1 for row in labels if any(s["rule_id"] in ("a", "b") for s in row["spans"]))
    sensitivity = {
        "frozen_rule_c_mode": RULE_C_MODE,
        "sensitivity_rule_c_mode": SENSITIVITY_RULE_C_MODE,
        "frozen_erroneous_share": round(sum(r["category"] == "erroneous" for r in labels) / len(labels), 6),
        "R1_best_window": {"erroneous": sens_best_r1[0], "n": sens_best_r1[1], "share": round(sens_best_r1[2], 6)},
        "all_best_window": (
            {"erroneous": sens_best_all[0], "n": sens_best_all[1], "share": round(sens_best_all[2], 6)}
            if sens_best_all
            else None
        ),
        "answers_with_rule_a_or_b_only": ab_only,
        "share_without_rule_c": round(ab_only / len(labels), 6),
        "note": (
            "Замороженная разметка использует буквальное чтение правила (c): любое окно Jaccard≥0.6 "
            "с другим числом. Более строгая трактовка (только лучшее окно) даёт меньше ошибок и "
            "опубликована здесь как sensitivity. Правила (a)/(b)/(d) не менялись ни в одном варианте."
        ),
    }

    n_answers = len(answers)
    n_err = sum(row["category"] == "erroneous" for row in labels)
    share_err = n_err / n_answers
    assert n_answers >= 400, "квота ≥400 ответов не выполнена"
    assert share_err >= 0.40, f"квота ≥40% ошибочных не выполнена: {share_err:.4f}. {remedy}"

    # Разбиение по doc_id, 300 dev / остальное test_realgen (по ОТВЕТАМ).
    doc_ids = sorted({r["doc_id"] for r in answers})
    per_doc = Counter(r["doc_id"] for r in answers)
    assert set(per_doc.values()) == {len(answers) // len(doc_ids)}
    answers_per_doc = len(answers) // len(doc_ids)
    ordered_docs = sorted(doc_ids, key=lambda d: rc.h(f"{rc.SEED}:realgen_split:{d}"))
    dev_doc_n = 300 // answers_per_doc
    dev_docs = set(ordered_docs[:dev_doc_n])
    test_docs = set(ordered_docs[dev_doc_n:])
    dev_idx = [i for i, r in enumerate(answers) if r["doc_id"] in dev_docs]
    test_idx = [i for i, r in enumerate(answers) if r["doc_id"] in test_docs]
    assert len(dev_idx) == 300 and len(test_idx) == n_answers - 300
    assert not (dev_docs & test_docs)

    # Публикуемые файлы.
    dev_idx_set = set(dev_idx)
    published_rows = [
        {
            "answer_id": rec["answer_id"],
            "doc_id": rec["doc_id"],
            "round": rec["round"],
            "shard": rec["shard"],
            "template_id": rec["template_id"],
            "template_index": rec["template_index"],
            "mode": rec["mode"],
            "mode_index": rec["mode_index"],
            "gen_seed": rec["gen_seed"],
            "prompt_sha256": rec["prompt_sha256"],
            "prompt": rec["prompt"],
            "answer": rec["answer"],
            "answer_sha256": rec["answer_sha256"],
            "prompt_tokens": rec["prompt_tokens"],
            "generated_tokens": rec["generated_tokens"],
            "gen_seconds": rec["gen_seconds"],
            "split": "dev" if i in dev_idx_set else "test_realgen",
        }
        for i, rec in enumerate(answers)
    ]
    dev_rows = [row for i, row in enumerate(published_rows) if i in dev_idx_set]
    test_rows = [row for i, row in enumerate(published_rows) if i not in dev_idx_set]
    rc.write_jsonl(rc.REAL / "REALGEN_ANSWERS.jsonl", published_rows)
    rc.write_jsonl(rc.REAL / "REALGEN_ANSWERS_DEV.jsonl", dev_rows)
    rc.write_jsonl(rc.REAL / "REALGEN_ANSWERS_TEST_REALGEN.jsonl", test_rows)
    rc.write_jsonl(rc.REAL / "REALGEN_LABELS.jsonl", labels)
    all_spans = [span for row in labels for span in row["spans"]]
    rc.write_jsonl(rc.REAL / "REALGEN_SPANS.jsonl", all_spans)
    splits = {
        "kind": "REALGEN_SPLITS",
        "seed": rc.SEED,
        "rule": "sorted(unique doc_id, sha256('42:realgen_split:'+doc_id))[:dev_doc_n]",
        "answers_per_doc": answers_per_doc,
        "dev_answers": 300,
        "test_realgen_answers": len(test_idx),
        "dev_docs": sorted(dev_docs),
        "test_realgen_docs": sorted(test_docs),
        "dev_doc_ids_sha256": rc.h("\n".join(sorted(dev_docs))),
        "test_realgen_doc_ids_sha256": rc.h("\n".join(sorted(test_docs))),
        "overlap": 0,
        "test_realgen_sealed": True,
        "test_realgen_access": "только по шифт-коду куратора; фаза B читает ТОЛЬКО dev",
    }
    rc.write_json(rc.REAL / "REALGEN_SPLITS.json", splits)

    # Статистика.
    categories = Counter(row["category"] for row in labels)
    rule_counts = Counter(span["rule_id"] for span in all_spans)
    answer_lengths = [row["answer_len"] for row in labels]
    span_lengths = [span["end"] - span["start"] for span in all_spans]
    atoms_total = sum(row["n_atoms"] for row in labels)
    unclear_atoms = sum(row["n_unclear_atoms"] for row in labels)
    by_round = {}
    for round_id in ("R1", "R2"):
        subset = [row for row in labels if row["round"] == round_id]
        if not subset:
            continue
        by_round[round_id] = {
            "n_answers": len(subset),
            "erroneous": sum(r["category"] == "erroneous" for r in subset),
            "erroneous_share": round(sum(r["category"] == "erroneous" for r in subset) / len(subset), 6),
            "clean": sum(r["category"] == "clean" for r in subset),
            "unclear": sum(r["category"] == "unclear" for r in subset),
            "spans": sum(r["n_spans"] for r in subset),
        }
    by_template = {}
    for template_id in rc.TEMPLATE_IDS:
        subset = [row for row in labels if row["template_id"] == template_id]
        by_template[template_id] = {
            "n": len(subset),
            "erroneous": sum(r["category"] == "erroneous" for r in subset),
            "spans": sum(r["n_spans"] for r in subset),
        }
    by_mode = {}
    for mode in rc.MODES:
        subset = [row for row in labels if row["mode"] == mode]
        by_mode[mode] = {
            "n": len(subset),
            "erroneous": sum(r["category"] == "erroneous" for r in subset),
            "spans": sum(r["n_spans"] for r in subset),
        }

    gen_seconds = sum(rec["gen_seconds"] for rec in answers)
    gen_tokens = sum(rec["generated_tokens"] for rec in answers)
    ci_runs = [x for x in args.ci_run_ids.split(",") if x]
    generated_rounds = {
        "R1": {
            "n_answers": len(r1),
            "used_in_freeze": True,
            "shard_files_sha256": [rc.sha256_file(p) for p in r1_files],
        },
        "R2": {
            "n_answers": len(r2),
            "used_in_freeze": use_r2,
            "purpose": "заранее сгенерированный contingency-раунд с повышенной сложностью промта",
            "shard_files_sha256": [rc.sha256_file(p) for p in r2_files],
            "note": "не вошёл в заморозку: квота 40% достигнута на R1; файлы остаются в CI check-runs, не коммитятся",
        },
    }

    # Среда/веса из CI-манифестов check-runs (извлечены вручную).
    meta_path = extract_dir / "EXTRACTION_META.json"
    metas = json.loads(meta_path.read_text(encoding="utf-8"))["metas"] if meta_path.exists() else {}
    shard_envs = [value.get("environment", {}) for value in metas.values() if isinstance(value, dict)]
    shard_stats = [value.get("stats", {}) for value in metas.values() if isinstance(value, dict) and "stats" in value]
    weight_files = shard_envs[0].get("weight_files", []) if shard_envs else []
    torch_weights = next((f for f in weight_files if f["name"] in ("pytorch_model.bin", "model.safetensors")), None)
    torch_bytes = torch_weights["bytes"] if torch_weights else None
    model_environment = {
        "model": rc.MODEL,
        "directive_weights_mb": 443.5,
        "weights_budget_mb": 550,
        "measured_pytorch_weights_bytes": torch_bytes,
        "measured_pytorch_weights_mb_decimal": round(torch_bytes / 1_000_000, 3) if torch_bytes else None,
        "measured_pytorch_weights_mib": round(torch_bytes / 1024**2, 3) if torch_bytes else None,
        "pytorch_weights_sha256": torch_weights["sha256"] if torch_weights else None,
        "snapshot_all_frameworks_bytes": shard_envs[0].get("weights_total_mb") if shard_envs else None,
        "budget_deviation_mb_decimal": round(torch_bytes / 1_000_000 - 550, 3) if torch_bytes else None,
        "note": (
            "Директива указывала 443.5 МБ; фактический pytorch_model.bin в snapshot "
            "ai-forever/rugpt3small_based_on_gpt2@a9307e6 больше. Модель используется "
            "только для research-генерации в CI, в запечатанный продуктовый стек P8 "
            "не попадает (пломба не тронута). Превышение бюджета 550 МБ зафиксировано "
            "открыто как deviation, см. поле budget_deviation_mb_decimal."
        ),
        "python": shard_envs[0].get("python") if shard_envs else platform.python_version(),
        "torch": shard_envs[0].get("torch") if shard_envs else None,
        "transformers": shard_envs[0].get("transformers") if shard_envs else None,
        "platform": shard_envs[0].get("platform") if shard_envs else None,
        "threads": shard_envs[0].get("threads") if shard_envs else None,
        "model_load_seconds_mean": (
            round(sum(env.get("model_load_seconds", 0) for env in shard_envs) / len(shard_envs), 3) if shard_envs else None
        ),
        "shard_stats": shard_stats,
        "answers_per_hour_per_shard_mean": (
            round(sum(s.get("answers_per_hour", 0) for s in shard_stats) / len(shard_stats), 2) if shard_stats else None
        ),
        "wall_clock_seconds_est": max((s.get("elapsed_seconds", 0) for s in shard_stats), default=0),
    }
    stats = {
        "kind": "REALGEN_PHASE_A",
        "version": "v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "campaign": "REALGEN-R1",
        "date": "2026-10-26",
        "run_id": (ci_runs[0] if ci_runs else "local-no-ci-evidence"),
        "evidence_context": {
            "run_id": (ci_runs[0] if ci_runs else "local-no-ci-evidence"),
            "date": "2026-10-26",
            "status": "authoritative",
            "source": "GitHub Actions generation + local deterministic markup",
            "labeling": "local-no-ci-evidence",
        },
        "quotas": {
            "min_answers": 400,
            "n_answers": n_answers,
            "quota_answers_pass": n_answers >= 400,
            "min_erroneous_share": 0.40,
            "n_erroneous": n_err,
            "erroneous_share": round(share_err, 6),
            "quota_errors_pass": share_err >= 0.40,
            "remedy": remedy,
            "round_R2_used": use_r2,
            "rules_changed": False,
        },
        "categories": {
            "clean": categories.get("clean", 0),
            "erroneous": categories.get("erroneous", 0),
            "unclear": categories.get("unclear", 0),
            "clean_share": round(categories.get("clean", 0) / n_answers, 6),
            "erroneous_share": round(categories.get("erroneous", 0) / n_answers, 6),
            "unclear_share": round(categories.get("unclear", 0) / n_answers, 6),
        },
        "atoms": {
            "total": atoms_total,
            "unclear_atoms": unclear_atoms,
            "atom_unclear_rate": round(unclear_atoms / max(1, atoms_total), 6),
            "answers_with_unclear_atoms": sum(row["n_unclear_atoms"] > 0 for row in labels),
        },
        "spans": {
            "n_spans": len(all_spans),
            "by_rule": dict(sorted(rule_counts.items())),
            "length_chars": {
                "min": min(span_lengths) if span_lengths else 0,
                "median": median(span_lengths),
                "max": max(span_lengths) if span_lengths else 0,
            },
            "per_erroneous_answer": round(len(all_spans) / max(1, n_err), 4),
        },
        "answers": {
            "length_chars": {
                "min": min(answer_lengths),
                "median": median(answer_lengths),
                "max": max(answer_lengths),
            },
            "generated_tokens_total": gen_tokens,
            "generated_tokens_mean": round(gen_tokens / n_answers, 2),
        },
        "quota_sensitivity": sensitivity,
        "generated_rounds": generated_rounds,
        "by_round": by_round,
        "by_template": by_template,
        "by_mode": by_mode,
        "splits": {
            "dev_answers": 300,
            "test_realgen_answers": len(test_idx),
            "dev_docs": len(dev_docs),
            "test_realgen_docs": len(test_docs),
            "overlap": 0,
            "test_realgen_sealed": True,
        },
        "timing": {
            "generation_seconds_sum_over_answers": round(gen_seconds, 3),
            "answers_per_hour_single_stream": round(n_answers / max(gen_seconds, 1e-9) * 3600, 2),
            "answers_per_hour_per_shard_mean": model_environment["answers_per_hour_per_shard_mean"],
            "wall_clock_seconds_est": model_environment["wall_clock_seconds_est"],
            "note": "сумма времен генерации всех ответов (эквивалент одного потока); 8 шардов CI работают параллельно",
            "platform": {"python": platform.python_version(), "system": platform.system(), "ci": "GitHub Actions ubuntu-24.04"},
        },
        "model_environment": model_environment,
        "provenance": {
            "model": rc.MODEL,
            "weights_mb_directive": 443.5,
            "weights_mb_measured_pytorch": model_environment["measured_pytorch_weights_mb_decimal"],
            "seed": rc.SEED,
            "background_ids_sha256": json.loads((rc.REAL / "REALGEN_BACKGROUND.json").read_text())["background_ids_sha256_canonical"],
            "selection_sha256": selection["doc_ids_sha256"],
            "ci_run_ids": ci_runs,
            "run_id": args.run_id,
            "extraction": "manual: gh api check-runs output.text → .arena/realgen_extract (auto-commit из workflow запрещён)",
            "real_test_viewed": False,
            "real_test_locked_files_read": [],
            "labeling": "local-no-ci-evidence (детерминированные правила, без LLM-судьи)",
        },
    }
    rc.write_json(rc.REAL / "REALGEN_PHASE_A.json", stats)

    # SHA-карта всех замораживаемых файлов.
    code_files = [
        "realgen_common.py",
        "realgen_build_background.py",
        "realgen_generate.py",
        "realgen_publish.py",
        "realgen_extract.py",
        "realgen_label.py",
    ]
    data_files = [
        "REALGEN_BACKGROUND.json",
        "REALGEN_SELECTION.json",
        "REALGEN_TEMPLATES.json",
        "REALGEN_DOCS.jsonl",
        "REALGEN_ANSWERS.jsonl",
        "REALGEN_ANSWERS_DEV.jsonl",
        "REALGEN_ANSWERS_TEST_REALGEN.jsonl",
        "REALGEN_LABELS.jsonl",
        "REALGEN_SPANS.jsonl",
        "REALGEN_SPLITS.json",
    ]
    sha_map = {}
    for name in code_files + data_files:
        path = rc.REAL / name
        sha_map[f"reports/science/real/{name}"] = {"sha256": rc.sha256_file(path), "bytes": rc.file_size(path)}
    answer_sha_map = {row["answer_id"]: row["answer_sha256"] for row in published_rows}
    span_sha_map = {
        rc.h(f"{s['answer_id']}:{s['start']}:{s['end']}:{s['rule_id']}")[:24]: rc.h(s["text"]) for s in all_spans
    }
    frozen = {
        "kind": "REALGEN_FROZEN",
        "version": "v1",
        "campaign": "REALGEN-R1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "author": "deterministic generation (ruGPT3small + rule-based atom markup, no LLM judge)",
        "seed": rc.SEED,
        "n_answers": n_answers,
        "n_documents": len(doc_ids),
        "files": sha_map,
        "answers_sha256_map": answer_sha_map,
        "spans_sha256_map": span_sha_map,
        "shards": {"n": 8, "rounds": ["R1"] + (["R2"] if use_r2 else [])},
        "splits": {
            "dev": {"n_answers": 300, "file": "reports/science/real/REALGEN_ANSWERS_DEV.jsonl"},
            "test_realgen": {
                "n_answers": len(test_idx),
                "file": "reports/science/real/REALGEN_ANSWERS_TEST_REALGEN.jsonl",
                "sha256": sha_map["reports/science/real/REALGEN_ANSWERS_TEST_REALGEN.jsonl"]["sha256"],
                "sealed": True,
                "access": "только по шифт-коду куратора",
            },
        },
        "rules": {"a": "number/date/amount/term unsupported by act", "b": "citation N missing in act", "c": "Jaccard>=0.6 window with different number", "d": "unclear atoms not labelled"},
        "rule_c_mode": RULE_C_MODE,
        "real_test_viewed": False,
        "provenance": {"ci_run_ids": ci_runs, "run_id": args.run_id, "model": rc.MODEL},
    }
    rc.write_json(rc.REAL / "REALGEN_FROZEN.json", frozen)

    # Markdown-отчёт.
    lines = [
        "# REALGEN-R1 — Фаза A: естественные галлюцинации на реальных актах",
        "",
        f"Дата: {stats['created_at']}. Ветка: `arena/233cd01f-smarty-chumakov`. Прогон: `{args.run_id}`.",
        "",
        "## Квоты",
        "",
        f"- Ответов: **{n_answers}** (минимум 400) — {'PASS' if stats['quotas']['quota_answers_pass'] else 'FAIL'}",
        f"- Ошибочных: **{n_err} / {n_answers} = {share_err:.4f}** (минимум 40%) — {'PASS' if stats['quotas']['quota_errors_pass'] else 'FAIL'}",
        f"- Remedy: {remedy}",
        f"- Правила разметки изменялись: **нет**. Замороженная трактовка правила (c): `{RULE_C_MODE}`.",
        f"- Раунды: сгенерировано R1={generated_rounds['R1']['n_answers']}, R2={generated_rounds['R2']['n_answers']}"
        f" (R2 — contingency, в заморозку {'вошёл' if use_r2 else 'не вошёл'}).",
        f"- Sensitivity: stricter `best_window` → R1 {sensitivity['R1_best_window']['share']:.4f}"
        + (f", all {sensitivity['all_best_window']['share']:.4f}" if sensitivity["all_best_window"] else "")
        + f"; только правила (a)+(b) без (c) → {sensitivity['share_without_rule_c']:.4f}.",
        "",
        "## Распределение",
        "",
        f"- чистые: {categories.get('clean', 0)}; ошибочные: {categories.get('erroneous', 0)}; неясные: {categories.get('unclear', 0)}",
        f"- атомов всего: {atoms_total}; неясных атомов: {unclear_atoms}; `atom_unclear_rate = {stats['atoms']['atom_unclear_rate']:.4f}`",
        f"- спанов: {len(all_spans)}; по правилам: {dict(sorted(rule_counts.items()))}",
        f"- длины ответов (символы): min/median/max = {stats['answers']['length_chars']['min']} / {stats['answers']['length_chars']['median']:.0f} / {stats['answers']['length_chars']['max']}",
        f"- длины спанов (символы): min/median/max = {stats['spans']['length_chars']['min']} / {stats['spans']['length_chars']['median']:.0f} / {stats['spans']['length_chars']['max']}",
        f"- темп генерации (один поток, сумма времён): {stats['timing']['answers_per_hour_single_stream']:.2f} ответов/час; всего {gen_seconds:.1f} с",
        "",
        "## Разбиение",
        "",
        f"- dev: 300 ответов / {len(dev_docs)} документов; test_realgen: {len(test_idx)} ответов / {len(test_docs)} документов; overlap=0.",
        "- test_realgen ЗАПЕЧАТАН, доступ только по шифт-коду куратора; фаза B читает только dev.",
        "",
        "## SHA-карта (первые 12 символов)",
        "",
        "| файл | sha256 | байт |",
        "| --- | --- | --- |",
    ]
    for name, info in sha_map.items():
        lines.append(f"| `{name}` | `{info['sha256'][:12]}…` | {info['bytes']} |")
    lines += [
        "",
        "## Отклонение по бюджету весов (честно)",
        "",
        f"- Директива: 443.5 МБ ≤ 550 МБ. Фактический `pytorch_model.bin`: "
        f"{model_environment['measured_pytorch_weights_bytes']} байт = "
        f"{model_environment['measured_pytorch_weights_mb_decimal']} МБ (dec) / "
        f"{model_environment['measured_pytorch_weights_mib']} МиБ, "
        f"sha256 `{model_environment['pytorch_weights_sha256']}`.",
        f"- Отклонение от бюджета 550 МБ: **{model_environment['budget_deviation_mb_decimal']} МБ**. "
        "Модель research-only в CI, продуктовая пломба P8 не изменена.",
        "",
        "## Provenance",
        "",
        f"- Модель: `{rc.MODEL}`, 8 шаблонов, 2 режима, seed=42 от doc_id.",
        f"- Фон: 6510 документов LawCodex/ru@5806f92 минус 1200 test-hosting; overlap=0; `real_test_viewed:false`.",
        f"- CI-прогоны: {', '.join(ci_runs) if ci_runs else 'local-no-ci-evidence'}; экстракция вручную через check-runs output.text.",
        "",
    ]
    (rc.REAL / "REALGEN_PHASE_A.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"FROZEN: {rc.REAL / 'REALGEN_FROZEN.json'}")


if __name__ == "__main__":
    main()
