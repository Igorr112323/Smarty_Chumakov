#!/usr/bin/env python3
"""REALGEN-R1 фаза A, шаг 2: генерация ответов ruGPT3small (запуск в CI).

Реальная модель ai-forever/rugpt3small_based_on_gpt2 (443.5 МБ), 8 шаблонов,
2 режима (greedy / top-p 0.95 temperature 0.8), seed детерминирован от doc_id.
Никаких инъекций ошибок: все числа/ссылки порождены самой моделью.

Результат каждого шарда публикуется через check-runs output.text (см.
realgen_publish.py) — auto-commit из workflow запрещён.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import realgen_common as rc  # noqa: E402
import realgen_publish as rp  # noqa: E402


def snapshot_info(model_dir: Path) -> dict:
    files = []
    total = 0
    for path in sorted(model_dir.rglob("*")):
        if not path.is_file() or "blobs" in path.parts:
            continue
        size = path.stat().st_size
        total += size
        files.append({"name": path.name, "bytes": size, "sha256": rc.sha256_file(path)})
    return {"weights_total_mb": round(total / 1_000_000, 3), "files": files}


def build_prompt(templates: dict, round_id: str, task: str, act_name: str, context: str) -> str:
    key = templates["rounds"][round_id]["prompt"]
    return templates[key].format(task=task, act_name=act_name, context=context)


def fit_prompt(tok, templates: dict, round_id: str, task: str, act_name: str, text: str, doc_id: str) -> tuple[str, list[int], int]:
    """Подгоняет окно контекста под 1024 позиции GPT-2, сохраняя 512 новых токенов."""
    chars = 1100
    for _ in range(8):
        context, start, end = rc.context_window(text, doc_id, chars=chars)
        # полное окно + преамбула уже внутри context_window
        prompt = build_prompt(templates, round_id, task, act_name, context)
        ids = tok.encode(prompt, add_special_tokens=False)
        if len(ids) <= 480:
            return prompt, ids, start
        chars = max(250, chars // 2)
    context, start, _ = rc.context_window(text, doc_id, chars=250)
    prompt = build_prompt(templates, round_id, task, act_name, context)
    ids = tok.encode(prompt, add_special_tokens=False)
    return prompt, ids[-480:], start


class Stopper:
    """Ранняя остановка дегенеративного продолжения (аппаратный критерий).

    Никакой разметки: это generator-level stopping criteria. Ответ остаётся
    дословным декодом сгенерированных токенов.
    """

    def __init__(self, tok, prompt_len: int, torch):
        self.tok = tok
        self.prompt_len = prompt_len
        self.torch = torch
        self.steps = 0

    def __call__(self, input_ids, scores, **kwargs):
        self.steps += 1
        gen = input_ids[0, self.prompt_len :]
        if gen.numel() >= 64 and self.torch.unique(gen[-32:]).numel() <= 3:
            return True
        if self.steps % 16 == 0:
            text = self.tok.decode(gen, skip_special_tokens=True)
            if len(text) >= 400 and (text.count(".") + text.count("!") + text.count("?")) >= 6:
                return True
            if len(text) >= 1800:
                return True
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--rounds", default="R1,R2")
    parser.add_argument("--out-dir", default=".arena/realgen")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--model-dir", default="")
    args = parser.parse_args()

    import numpy as np
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from transformers import StoppingCriteria, StoppingCriteriaList
    except ImportError:  # совместимость с будущими версиями transformers
        StoppingCriteria, StoppingCriteriaList = object, None

    torch.set_num_threads(2)
    selection = json.loads((rc.REAL / "REALGEN_SELECTION.json").read_text(encoding="utf-8"))
    templates = json.loads((rc.REAL / "REALGEN_TEMPLATES.json").read_text(encoding="utf-8"))
    docs = {row["doc_id"]: row for row in rc.read_jsonl(rc.REAL / "REALGEN_DOCS.jsonl")}
    assert len(docs) == 400

    t0 = time.time()
    model_dir = Path(args.model_dir) if args.model_dir else Path(snapshot_download(rc.MODEL))
    weights = snapshot_info(model_dir)
    tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir))
    model.eval()
    if tok.eos_token_id is None:
        tok.eos_token = "<|endoftext|>"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    load_seconds = time.time() - t0

    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "threads": torch.get_num_threads(),
        "model": rc.MODEL,
        "model_dir": str(model_dir),
        "model_load_seconds": round(load_seconds, 3),
        "weights_total_mb": weights["weights_total_mb"],
        "weight_files": weights["files"],
        "shard": args.shard,
        "shards": args.shards,
        "run_id": os.environ.get("GITHUB_RUN_ID", "local-no-ci-evidence"),
        "head_sha": os.environ.get("HEAD_SHA", ""),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    head_sha = os.environ.get("HEAD_SHA", "local")
    repo = os.environ.get("GITHUB_REPOSITORY", "Igorr112323/Smarty_Chumakov")
    published = []

    for round_id in args.rounds.split(","):
        round_id = round_id.strip()
        assignments = [a for a in selection[round_id] if a["index"] % args.shards == args.shard]
        rows = []
        started = time.time()
        for pos, item in enumerate(assignments):
            doc_id = item["doc_id"]
            doc = docs[doc_id]
            task = templates["tasks"][item["template_index"]]
            prompt, prompt_ids, window_start = fit_prompt(tok, templates, round_id, task, doc["act_name"], doc["text"], doc_id)
            max_new = min(templates["model"]["max_new_tokens"], rc.MODEL_MAX_POSITIONS - len(prompt_ids))
            if max_new < 64:
                max_new = 64
            seed = item["gen_seed"]
            random.seed(seed)
            np.random.seed(seed % (2**32 - 1))
            torch.manual_seed(seed)
            try:
                transformers.set_seed(seed)
            except TypeError:
                transformers.set_seed(seed, deterministic=False)
            input_ids = torch.tensor([prompt_ids], dtype=torch.long)
            stopper = Stopper(tok, len(prompt_ids), torch)
            kwargs = {
                "max_new_tokens": max_new,
                "do_sample": item["mode_index"] == 1,
                "pad_token_id": tok.pad_token_id,
                "eos_token_id": tok.eos_token_id,
            }
            if StoppingCriteriaList is not None:
                kwargs["stopping_criteria"] = StoppingCriteriaList([stopper])
            if item["mode_index"] == 1:
                kwargs["temperature"] = 0.8
                kwargs["top_p"] = 0.95
            t1 = time.time()
            with torch.no_grad():
                output = model.generate(input_ids=input_ids, **kwargs)
            gen_seconds = time.time() - t1
            generated_ids = output[0, len(prompt_ids) :].tolist()
            answer = tok.decode(generated_ids, skip_special_tokens=True).strip()
            rows.append(
                {
                    "answer_id": rc.h(f"{round_id}:{doc_id}:{item['template_id']}:{item['mode']}")[:24],
                    "round": round_id,
                    "shard": args.shard,
                    "index": item["index"],
                    "doc_id": doc_id,
                    "template_id": item["template_id"],
                    "template_index": item["template_index"],
                    "mode": item["mode"],
                    "mode_index": item["mode_index"],
                    "gen_seed": seed,
                    "prompt_sha256": rc.h(prompt),
                    "prompt": prompt,
                    "context_window_start": window_start,
                    "prompt_tokens": len(prompt_ids),
                    "max_new_tokens": max_new,
                    "generated_tokens": len(generated_ids),
                    "answer": answer,
                    "answer_sha256": rc.h(answer),
                    "gen_seconds": round(gen_seconds, 3),
                    "environment": environment,
                }
            )
            if (pos + 1) % 10 == 0:
                print(f"[{round_id} shard {args.shard}] {pos + 1}/{len(assignments)} "
                      f"avg_tok={sum(r['generated_tokens'] for r in rows) / len(rows):.1f} "
                      f"elapsed={time.time() - started:.1f}s", flush=True)

        elapsed = time.time() - started
        path = out_dir / f"answers_{round_id}_shard{args.shard:02d}.jsonl"
        rc.write_jsonl(path, rows)
        rate = len(rows) / max(elapsed, 1e-9) * 3600
        stats = {
            "round": round_id,
            "shard": args.shard,
            "n_answers": len(rows),
            "elapsed_seconds": round(elapsed, 3),
            "answers_per_hour": round(rate, 2),
            "mean_generated_tokens": round(sum(r["generated_tokens"] for r in rows) / max(1, len(rows)), 2),
            "mean_answer_chars": round(sum(len(r["answer"]) for r in rows) / max(1, len(rows)), 2),
            "file_sha256": rc.sha256_file(path),
            "file_bytes": rc.file_size(path),
        }
        rp.step_summary(
            f"### REALGEN {round_id} shard {args.shard}\n\n```json\n{json.dumps(stats, ensure_ascii=False, indent=1)}\n```"
        )
        if args.publish:
            prefix = f"REALGEN-A/{round_id}/shard-{args.shard:02d}"
            meta = rp.publish_file(path, prefix, head_sha, repo, {"stats": stats, "environment": environment})
            published.append(meta)
        print(json.dumps(stats, ensure_ascii=False), flush=True)

    (out_dir / f"environment_shard{args.shard:02d}.json").write_text(
        json.dumps({"environment": environment, "published": published}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
