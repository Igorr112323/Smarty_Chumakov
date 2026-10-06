#!/usr/bin/env python3
"""Загрузка внешних размеченных наборов (RAGTruth, RusHallu-RAG и др.) с пинами.

Что делает
----------

1. Качает внешние наборы по **закреплённым ревизиям** (commit SHA для GitHub,
   revision SHA для HuggingFace) в ``data/external/<набор>/``.
2. Сверяет SHA256 каждого файла с ожидаемым; при расхождении — громкая ошибка
   (данные обновились на стороне источника, нужно решение человека).
3. Проверяет **контрольные числа** набора (сколько ответов, источников, спанов,
   совпадают ли смещения) и печатает их.
4. Переводит наборы в наш контракт: ``data/external/adapted/*.jsonl``.
5. Пишет ``data/external/MANIFEST.json``: ревизии, хеши, размеры, лицензия,
   происхождение разметки, дата загрузки.

Внешние данные не коммитятся: ``data/external/`` в ``.gitignore``. В репозитории
остаются скрипт, манифест и агрегаты.

Правило: внешние наборы — **только тест**. Скрипт никогда не пишет в обучающие
файлы и не трогает ``config/``.

Запуск::

    python scripts/fetch_external_tests.py --all --out data/external --verify
    python scripts/fetch_external_tests.py --dataset ragtruth --verify --adapt
    python scripts/fetch_external_tests.py --registry
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spanverify.external_datasets import (  # noqa: E402
    ragtruth_pairs,
    ragtruth_totals,
    summarize_pairs,
    validate_pairs,
)

REPO_ROOT = ROOT.parent

# ---------------------------------------------------------------- реестр наборов

# Файлы RAGTruth: закреплённый коммит main (последняя правка репозитория — 02.10.2026),
# лицензия MIT (файл LICENSE в репозитории). SHA256 посчитан по фактическим байтам.
RAGTRUTH = {
    "name": "ragtruth",
    "title": "RAGTruth",
    "url": "https://github.com/ParticleMedia/RAGTruth",
    "repo": "ParticleMedia/RAGTruth",
    "revision": "c103204b9ce28d6bbad859304bf30de72b8ed8fe",
    "license": "MIT (подтверждено: файл LICENSE в репозитории)",
    "label_origin": "human",
    "language": "en",
    "files": {
        "response.jsonl": {
            "path": "dataset/response.jsonl",
            "sha256": "e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073",
            "size": 21458735,
        },
        "source_info.jsonl": {
            "path": "dataset/source_info.jsonl",
            "sha256": "0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b",
            "size": 15117971,
        },
    },
    # Контрольные числа: сняты внешним проверяющим 03.10.2026 и повторены нами.
    "controls": {
        "responses": 17790,
        "sources": 2965,
        "spans": 14289,
        "verified_spans": 14289,
        "by_task": {"QA": 989, "Summary": 943, "Data2txt": 1033},
        "by_split": {"train": 15090, "test": 2700},
        "by_quality": {"good": 17617, "incorrect_refusal": 144, "truncated": 29},
        "test_qa_good": {"responses": 875, "with_hallucination": 160, "clean": 715, "spans": 235, "sources": 150},
    },
}

# RusHallu-RAG качается существующим загрузчиком (он же сверяет SHA256 CSV).
RUSHALLU = {
    "name": "rushallu",
    "title": "RusHallu-RAG",
    "url": "https://github.com/feudor2/RusHallu-RAG",
    "repo": "feudor2/RusHallu-RAG",
    "revision": "345907f983da91e1b7720b5944d982ea9c9aeb83",
    "license": "не подтверждена (бейдж Apache-2.0, файла LICENSE нет) — сырые данные не публиковать",
    "label_origin": "human",
    "language": "ru",
    "controls": {
        "responses": 1000,
        "clean": 667,
        "with_hallucination": 333,
        "spans": 423,
    },
}

# Наборы, которые берём, но чьи метки — не человеческие: происхождение обязательно
# в отчёте, поэтому оно зафиксировано здесь.
OPTIONAL = {
    "halueval_llm_spans": {
        "title": "HaluEval LLM Spans (vllm-sr/halueval-llm-spans)",
        "url": "https://huggingface.co/datasets/vllm-sr/halueval-llm-spans",
        "license": "Apache-2.0 (подтверждено)",
        "label_origin": "llm",
        "language": "en",
        "note": "разметка LLM (Qwen2.5-72B), не человек; в отчёте помечать обязательно",
        "transport": "hf",
    },
    "lettucedetect": {
        "title": "rag-bioasq-lettucedetect (KRLabsOrg)",
        "url": "https://github.com/KRLabsOrg/rag-bioasq-lettucedetect",
        "license": "MIT (подтверждено)",
        "label_origin": "auto",
        "language": "en",
        "note": "автоматическая разметка, синтетические ответы; вспомогательный набор",
    },
}

# Наборы, которые сознательно НЕ подключаем. Хранится здесь, чтобы решение было
# видно в коде, а не только в переписке.
EXCLUDED = {
    "HaDes": "reference-free набор: контекста нет, есть исходный и искажённый текст — наша задача не измеряется",
    "HDM-Bench": "CC BY-NC-SA 4.0 и закрытый доступ (gated) — некоммерческая лицензия",
    "FAVA": "лицензия не подтверждена",
    "AlsKozlov/legalbench-ru": "не тест-набор: контекст лишь у 239 из 846 строк, все строки needs_expert_review, первая строка — канарейка",
    "Roflmax/Ru-Legal-QA-v1": "нет лицензии",
    "SberQuAD-зеркало (HF)": "лицензия unknown",
}


class FetchError(RuntimeError):
    """Не удалось получить или проверить внешние данные."""


def sha256_file(path: Path) -> str:
    """SHA256 файла по фактическим байтам (по частям, чтобы не держать гигабайты в памяти)."""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_http(url: str, dest: Path) -> None:
    """Скачать файл обычным HTTPS."""
    request = urllib.request.Request(url, headers={"User-Agent": "spanverify-external/1.0"})
    with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310 - фиксированный https
        dest.write_bytes(response.read())


def _download_gh(repo: str, revision: str, path: str, dest: Path) -> None:
    """Скачать файл через GitHub API (работает там, где raw.githubusercontent закрыт)."""
    result = subprocess.run(
        ["gh", "api", "-H", "Accept: application/vnd.github.raw", f"repos/{repo}/contents/{path}?ref={revision}"],
        capture_output=True,
        check=False,
        timeout=1800,
    )
    if result.returncode != 0 or not result.stdout:
        raise FetchError(f"gh api не отдал {repo}/{path}: {result.stderr.decode('utf-8', 'replace').strip()[:200]}")
    dest.write_bytes(result.stdout)


def download_github_file(repo: str, revision: str, path: str, dest: Path, transport: str = "auto") -> str:
    """Скачать файл GitHub и вернуть использованный транспорт (``http`` или ``gh``)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    raw_url = f"https://raw.githubusercontent.com/{repo}/{revision}/{path}"
    attempts: list[str] = []
    if transport in {"auto", "http"}:
        try:
            _download_http(raw_url, dest)
            return "http"
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            attempts.append(f"http: {error}")
            if transport == "http":
                raise FetchError("; ".join(attempts)) from error
    if transport in {"auto", "gh"}:
        try:
            _download_gh(repo, revision, path, dest)
            return "gh"
        except (FetchError, subprocess.SubprocessError) as error:
            attempts.append(f"gh: {error}")
    raise FetchError("не удалось скачать (нужен доступ к GitHub): " + "; ".join(attempts))


def fetch_ragtruth(out_dir: Path, transport: str, verify: bool) -> dict:
    """Скачать RAGTruth, проверить хеши и контрольные числа, вернуть отчёт."""
    target_dir = out_dir / RAGTRUTH["name"]
    report: dict = {
        "title": RAGTRUTH["title"],
        "url": RAGTRUTH["url"],
        "repo": RAGTRUTH["repo"],
        "revision": RAGTRUTH["revision"],
        "license": RAGTRUTH["license"],
        "label_origin": RAGTRUTH["label_origin"],
        "language": RAGTRUTH["language"],
        "files": {},
        "adapters": {},
    }
    for name, spec in RAGTRUTH["files"].items():
        dest = target_dir / name
        used = download_github_file(RAGTRUTH["repo"], RAGTRUTH["revision"], spec["path"], dest, transport)
        digest = sha256_file(dest)
        match = digest == spec["sha256"]
        if verify and not match:
            raise FetchError(
                f"SHA256 {name} не совпал с закреплённым: {digest} != {spec['sha256']}. "
                "Данные изменились на стороне источника — зафиксируйте новую ревизию и обновите реестр."
            )
        report["files"][name] = {
            "sha256": digest,
            "expected_sha256": spec["sha256"],
            "verified": match,
            "size": dest.stat().st_size,
            "expected_size": spec["size"],
            "transport": used,
        }

    responses = [
        json.loads(line)
        for line in (target_dir / "response.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sources = [
        json.loads(line)
        for line in (target_dir / "source_info.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    totals = ragtruth_totals(responses, sources)
    problems = [
        key
        for key, value in RAGTRUTH["controls"].items()
        if key in {"responses", "sources", "spans", "verified_spans"} and totals[key] != value
    ]
    if verify and problems:
        raise FetchError(f"контрольные числа RAGTruth разошлись: {problems}; фактически {totals}")
    report["totals"] = totals
    report["controls_ok"] = not problems

    qa_pairs, qa_stats = ragtruth_pairs(responses, sources, task="QA", split="test")
    expected_qa = RAGTRUTH["controls"]["test_qa_good"]
    qa_ok = all(
        qa_stats[key] == expected_qa[key] for key in ("responses", "with_hallucination", "clean", "spans", "sources")
    )
    if verify and not qa_ok:
        raise FetchError(f"контрольные числа test QA разошлись: {qa_stats} != {expected_qa}")
    report["adapters"]["ragtruth_qa_test"] = {**qa_stats, "controls_ok": qa_ok}
    return report


def fetch_rushallu(out_dir: Path, transport: str, verify: bool) -> dict:
    """Скачать RusHallu-RAG существующим загрузчиком и сверить контрольные числа."""
    from scripts.fetch_rushallu import fetch as fetch_rushallu_data  # noqa: PLC0415

    report = fetch_rushallu_data(out_dir / RUSHALLU["name"], transport=transport, verify=verify)
    summary = report["summary"]
    expected = RUSHALLU["controls"]
    actual = {
        "responses": report["pairs"],
        "clean": summary["clean"],
        "with_hallucination": summary["with_hallucination"],
        "spans": summary["spans"],
    }
    ok = actual == expected
    if verify and not ok:
        raise FetchError(f"контрольные числа RusHallu-RAG разошлись: {actual} != {expected}")
    return {
        "title": RUSHALLU["title"],
        "url": RUSHALLU["url"],
        "repo": RUSHALLU["repo"],
        "revision": RUSHALLU["revision"],
        "license": RUSHALLU["license"],
        "label_origin": RUSHALLU["label_origin"],
        "language": RUSHALLU["language"],
        "files": {
            name: {"sha256": info["sha256"], "verified": info["verified"], "size": None}
            for name, info in report["files"].items()
        },
        "totals": {**actual, "ambiguous_offsets": summary["ambiguous_offsets"], "by_type": summary["types"]},
        "controls_ok": ok,
    }


def write_adapted(out_dir: Path, datasets: list[str], transport: str) -> dict:
    """Собрать адаптированные файлы (наш контракт) для выбранных наборов."""
    adapted_dir = out_dir / "adapted"
    adapted_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, dict] = {}

    if "ragtruth" in datasets:
        target_dir = out_dir / "ragtruth"
        responses = [
            json.loads(line)
            for line in (target_dir / "response.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        sources = [
            json.loads(line)
            for line in (target_dir / "source_info.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for task in ("QA", "Summary", "Data2txt"):
            pairs, stats = ragtruth_pairs(responses, sources, task=task, split="test")
            path = adapted_dir / f"ragtruth_{task.lower()}_test.jsonl"
            path.write_text("".join(json.dumps(pair, ensure_ascii=False) + "\n" for pair in pairs), encoding="utf-8")
            written[path.name] = {
                "dataset": "ragtruth",
                "task": task,
                "split": "test",
                "quality": "good",
                "pairs": len(pairs),
                "stats": stats,
                "summary": summarize_pairs(pairs),
                "sha256": sha256_file(path),
                "problems": validate_pairs(pairs, path.name),
            }
        all_pairs, all_stats = ragtruth_pairs(responses, sources, split="test")
        path = adapted_dir / "ragtruth_test.jsonl"
        path.write_text("".join(json.dumps(pair, ensure_ascii=False) + "\n" for pair in all_pairs), encoding="utf-8")
        written[path.name] = {
            "dataset": "ragtruth",
            "task": "all",
            "split": "test",
            "quality": "good",
            "pairs": len(all_pairs),
            "stats": all_stats,
            "summary": summarize_pairs(all_pairs),
            "sha256": sha256_file(path),
            "problems": validate_pairs(all_pairs, path.name),
        }

    if "rushallu" in datasets:
        from scripts.fetch_rushallu import read_csv_rows  # noqa: PLC0415
        from spanverify.external_datasets import rushallu_pairs  # noqa: PLC0415

        target_dir = out_dir / "rushallu"
        pairs: list[dict] = []
        stats: dict = {}
        for name in ("sberquad-rag.csv", "ruscibench-rag.csv"):
            rows = read_csv_rows(target_dir / name)
            block, block_stats = rushallu_pairs(rows, name.removesuffix("-rag.csv"))
            pairs.extend(block)
            for key, value in block_stats.items():
                if isinstance(value, dict):
                    merged = stats.setdefault(key, {})
                    for sub_key, sub_value in value.items():
                        merged[sub_key] = merged.get(sub_key, 0) + sub_value
                else:
                    stats[key] = stats.get(key, 0) + value
        path = adapted_dir / "rushallu.jsonl"
        path.write_text("".join(json.dumps(pair, ensure_ascii=False) + "\n" for pair in pairs), encoding="utf-8")
        written[path.name] = {
            "dataset": "rushallu",
            "task": "rag",
            "split": "all",
            "pairs": len(pairs),
            "stats": stats,
            "summary": summarize_pairs(pairs),
            "sha256": sha256_file(path),
            "problems": validate_pairs(pairs, path.name),
        }
    return written


def build_manifest(out_dir: Path, reports: dict, adapted: dict, transport: str) -> dict:
    """Собрать манифест: ревизии, хеши, лицензии, происхождение разметки."""
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "transport": transport,
        "rule": "внешние наборы — только тест: ничего не обучается и не калибруется на них",
        "datasets": reports,
        "adapted": adapted,
        "optional_not_fetched": {
            name: {**spec, "status": "не загружен в этой среде"} for name, spec in OPTIONAL.items()
        },
        "excluded": EXCLUDED,
    }


def print_registry() -> None:
    """Напечатать реестр наборов: что берём, что нет и почему."""
    print("Берём:")
    for spec in (RAGTRUTH, RUSHALLU):
        print(f"  {spec['name']:10s} {spec['language']} разметка={spec['label_origin']:5s} {spec['license']}")
    print("Берём как вспомогательные (не человеческая разметка):")
    for name, spec in OPTIONAL.items():
        print(f"  {name:20s} {spec['language']} разметка={spec['label_origin']:5s} {spec['license']}")
    print("Исключены:")
    for name, reason in EXCLUDED.items():
        print(f"  {name:26s} {reason}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Загрузка внешних размеченных наборов")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "external")
    parser.add_argument("--dataset", action="append", choices=["ragtruth", "rushallu"], default=None)
    parser.add_argument("--all", action="store_true", help="все наборы, которые берём")
    parser.add_argument("--verify", action="store_true", help="падать при расхождении хешей и контрольных чисел")
    parser.add_argument("--adapt", action="store_true", help="собрать файлы в нашем контракте")
    parser.add_argument("--transport", choices=["auto", "http", "gh"], default="auto")
    parser.add_argument("--registry", action="store_true", help="показать реестр и выйти")
    args = parser.parse_args()

    if args.registry:
        print_registry()
        return 0

    datasets = args.dataset or (["ragtruth", "rushallu"] if args.all else [])
    if not datasets:
        parser.error("укажите --all или --dataset (или --registry)")

    reports: dict = {}
    try:
        if "ragtruth" in datasets:
            print("[ragtruth] скачиваю два JSONL с закреплённого коммита…")
            reports["ragtruth"] = fetch_ragtruth(args.out, args.transport, args.verify)
            totals = reports["ragtruth"]["totals"]
            print(
                f"[ragtruth] ответов {totals['responses']}, источников {totals['sources']}, "
                f"спанов {totals['spans']} (совпали смещения: {totals['verified_spans']})"
            )
            qa = reports["ragtruth"]["adapters"]["ragtruth_qa_test"]
            print(
                f"[ragtruth] test QA (quality=good): ответов {qa['responses']}, с галлюцинациями "
                f"{qa['with_hallucination']}, чистых {qa['clean']}, спанов {qa['spans']}, источников {qa['sources']}"
            )
        if "rushallu" in datasets:
            print("[rushallu] скачиваю CSV разметки с закреплённого коммита…")
            reports["rushallu"] = fetch_rushallu(args.out, args.transport, args.verify)
            totals = reports["rushallu"]["totals"]
            print(
                f"[rushallu] пар {totals['responses']}, чистых {totals['clean']}, "
                f"с галлюцинациями {totals['with_hallucination']}, спанов {totals['spans']}"
            )
    except FetchError as error:
        print(f"ошибка: {error}", file=sys.stderr)
        return 2

    adapted: dict = {}
    if args.adapt:
        adapted = write_adapted(args.out, datasets, args.transport)
        for name, info in adapted.items():
            mark = "ок" if not info["problems"] else f"ПРОБЛЕМЫ: {info['problems'][:2]}"
            print(f"[adapted] {name}: {info['pairs']} пар, метки {mark}")
            if info["problems"]:
                return 2

    manifest = build_manifest(args.out, reports, adapted, args.transport)
    manifest_path = args.out / "MANIFEST.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Манифест: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
