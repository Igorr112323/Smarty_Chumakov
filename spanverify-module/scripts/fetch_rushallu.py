#!/usr/bin/env python3
"""Загрузка внешнего бенчмарка RusHallu-RAG и перевод его в наш формат.

Что делает
----------

1. Скачивает **только разметку** (два CSV) из репозитория ``feudor2/RusHallu-RAG``
   на закреплённом коммите и сверяет SHA256 каждого файла.
2. Переводит разметку в формат SpanVerify: ``{"id", "context", "answer", "labels",
   "meta"}``, где ``labels`` — смещения в **символах** ответа.
3. Пишет ``pairs.jsonl`` и ``manifest.json`` (хеши, статистика, ссылка на источник).

Почему CSV, а не parquet с HuggingFace
--------------------------------------

В CSV есть всё необходимое: ``query_text``, ``docs`` (5 документов), ``model_output``
и человеческая разметка ``answer``. Проверено на всех 1000 строках: каждый непустой
span находится в ``model_output`` подстрокой (423 из 423), неоднозначных смещений — 1.
В закрытом окружении HuggingFace недоступен, и parquet не нужен: CSV даёт те же
данные (сверка 1:1 по ``query_id`` и ``model_output`` выполнялась внешним проверяющим).

Про лицензию
------------

Лицензия данных **не подтверждена**: в репозитории есть бейдж Apache-2.0, но файла
LICENSE нет, у HF-датасетов лицензия не указана. Поэтому сырые файлы не коммитятся
(``data/external/`` в ``.gitignore``), а в отчётах допускаются только агрегированные
метрики и короткие цитаты со ссылкой на работу. Письмо авторам — в ``docs/КОРПУС_v1.md``.

Запуск::

    python scripts/fetch_rushallu.py --out data/external/rushallu --verify
    python scripts/fetch_rushallu.py --out data/external/rushallu --transport gh --verify
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Закреплённая ревизия: коммит main на 03.10.2026, файлы и их SHA256.
SOURCE_REPO = "feudor2/RusHallu-RAG"
SOURCE_COMMIT = "345907f983da91e1b7720b5944d982ea9c9aeb83"
CITATION = (
    "Sadkovskii F., Nasyrova R., Sorokin A. RusHallu-RAG: бенчмарк обнаружения "
    "галлюцинаций RAG (Диалог-2026, DOI 10.29003/2075-7182-2026-24-516-534)"
)
ANNOTATION_DIR = "results/annotation/human/merged"
FILES: dict[str, str] = {
    "sberquad-rag.csv": "f1e9043bb092eeef2c765e52652d40839aeab84268ed3badb0d417fb987d8f6f",
    "ruscibench-rag.csv": "f44f10e574ce547f832f2215189e40e158d06be3dbeec712926f1c59df60375e",
}
DATASET_VERSION = "rushallu-rag@pinned"

# Разделитель документов контекста: так же, как их видела модель-генератор.
DOCUMENT_SEPARATOR = "\n\n[ДОКУМЕНТ {index}]\n"


class FetchError(RuntimeError):
    """Не удалось получить или проверить внешние данные."""


def sha256_file(path: Path) -> str:
    """SHA256 файла (по частям, чтобы не держать 10 МБ в памяти)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_http(url: str, dest: Path) -> None:
    """Скачать файл обычным HTTPS."""
    request = urllib.request.Request(url, headers={"User-Agent": "spanverify-corpus/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310 - фиксированный https-адрес
        dest.write_bytes(response.read())


def _download_gh(repo: str, commit: str, name: str, dest: Path) -> None:
    """Скачать файл через GitHub API (работает там, где raw.githubusercontent закрыт)."""
    path = f"{ANNOTATION_DIR}/{name}"
    result = subprocess.run(
        [
            "gh",
            "api",
            "-H",
            "Accept: application/vnd.github.raw",
            f"repos/{repo}/contents/{path}?ref={commit}",
        ],
        capture_output=True,
        check=False,
        timeout=600,
    )
    if result.returncode != 0 or not result.stdout:
        raise FetchError(f"gh api не отдал {path}: {result.stderr.decode('utf-8', 'replace').strip()[:200]}")
    dest.write_bytes(result.stdout)


def download_file(name: str, dest: Path, transport: str = "auto") -> str:
    """Скачать файл разметки и вернуть использованный транспорт.

    ``auto`` — сначала обычный HTTPS, при недоступности — GitHub API.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    raw_url = f"https://raw.githubusercontent.com/{SOURCE_REPO}/{SOURCE_COMMIT}/{ANNOTATION_DIR}/{name}"
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
            _download_gh(SOURCE_REPO, SOURCE_COMMIT, name, dest)
            return "gh"
        except (FetchError, subprocess.SubprocessError) as error:
            attempts.append(f"gh: {error}")
    raise FetchError(
        "не удалось скачать данные (нужен доступ к raw.githubusercontent.com или gh): " + "; ".join(attempts)
    )


def parse_documents(raw: str) -> list[dict]:
    """Разобрать колонку ``docs`` (Python-repr списка словарей)."""
    value = ast.literal_eval(raw)
    if not isinstance(value, list):
        raise FetchError(f"колонка docs не список: {type(value).__name__}")
    return [dict(item) for item in value]


def build_context(documents: list[dict]) -> str:
    """Склеить документы в контекст с разделителями (порядок сохраняется)."""
    parts: list[str] = []
    for index, document in enumerate(documents, start=1):
        parts.append(DOCUMENT_SEPARATOR.format(index=index))
        parts.append(str(document.get("doc_text", "")))
    return "".join(parts).strip()


def spans_to_labels(answer: str, annotation: str) -> tuple[list[list[int]], list[str], int]:
    """Перевести разметку (JSON со спанами) в смещения символов ответа.

    Возвращает ``(labels, types, ambiguous)``: ``labels`` — тройки
    ``[start, end, 1]``; ``ambiguous`` — сколько спанов встретилось в ответе
    больше одного раза (берётся первое вхождение, как в приёмке).
    """
    items = json.loads(annotation)
    labels: list[list[int]] = []
    types: list[str] = []
    ambiguous = 0
    for item in items:
        span = (item or {}).get("span")
        if not span:
            continue
        position = answer.find(span)
        if position < 0:
            raise FetchError(f"спан не найден в ответе дословно: {span[:60]!r}")
        if answer.count(span) > 1:
            ambiguous += 1
        labels.append([position, position + len(span), 1])
        types.append(str((item or {}).get("type") or "unknown"))
    return labels, types, ambiguous


def build_pairs(rows: list[dict], source: str) -> tuple[list[dict], dict]:
    """Собрать пары в формате SpanVerify и статистику разметки."""
    pairs: list[dict] = []
    stats: dict = {
        "clean": 0,
        "with_hallucination": 0,
        "spans": 0,
        "ambiguous_offsets": 0,
        "types": {},
    }
    for row in rows:
        documents = parse_documents(row["docs"])
        answer = row["model_output"]
        labels, types, ambiguous = spans_to_labels(answer, row["answer"])
        stats["ambiguous_offsets"] += ambiguous
        if labels:
            stats["with_hallucination"] += 1
            stats["spans"] += len(labels)
            for kind in types:
                stats["types"][kind] = stats["types"].get(kind, 0) + 1
        else:
            stats["clean"] += 1
        pairs.append(
            {
                "id": f"rushallu-{source}-{row['query_id']}",
                "context": build_context(documents),
                "answer": answer,
                "labels": labels,
                "meta": {
                    "kind": "rushallu_rag",
                    "source": source,
                    "query": row["query_text"],
                    "query_id": row["query_id"],
                    "types": types,
                    "clean": not labels,
                    "dataset_version": DATASET_VERSION,
                    "citation": CITATION,
                    "context_reference": row.get("joined_reference", ""),
                },
            }
        )
    return pairs, stats


def read_csv_rows(path: Path) -> list[dict]:
    """Прочитать CSV разметки (пропуская строку-индекс)."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def fetch(out_dir: Path, transport: str = "auto", verify: bool = True) -> dict:
    """Скачать, проверить и собрать внешний набор; вернуть отчёт."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "source_repo": SOURCE_REPO,
        "source_commit": SOURCE_COMMIT,
        "citation": CITATION,
        "dataset_version": DATASET_VERSION,
        "files": {},
        "transport": transport,
    }
    all_pairs: list[dict] = []
    for name, expected in FILES.items():
        target = out_dir / name
        used = download_file(name, target, transport=transport)
        actual = sha256_file(target)
        if verify and actual != expected:
            raise FetchError(
                f"SHA256 файла {name} не совпал с закреплённым: {actual} != {expected}. "
                "Данные обновились на стороне источника — обновите константу и запишите это в отчёт."
            )
        source = name.removesuffix("-rag.csv")
        rows = read_csv_rows(target)
        pairs, stats = build_pairs(rows, source)
        all_pairs.extend(pairs)
        report["files"][name] = {
            "sha256": actual,
            "expected_sha256": expected,
            "verified": actual == expected,
            "rows": len(rows),
            "transport": used,
            "stats": stats,
        }

    pairs_path = out_dir / "pairs.jsonl"
    pairs_path.write_text("".join(json.dumps(pair, ensure_ascii=False) + "\n" for pair in all_pairs), encoding="utf-8")
    report["pairs"] = len(all_pairs)
    report["pairs_sha256"] = sha256_file(pairs_path)
    report["summary"] = {
        "clean": sum(item["stats"]["clean"] for item in report["files"].values()),
        "with_hallucination": sum(item["stats"]["with_hallucination"] for item in report["files"].values()),
        "spans": sum(item["stats"]["spans"] for item in report["files"].values()),
        "ambiguous_offsets": sum(item["stats"]["ambiguous_offsets"] for item in report["files"].values()),
        "types": {
            kind: sum(item["stats"]["types"].get(kind, 0) for item in report["files"].values())
            for kind in sorted({kind for item in report["files"].values() for kind in item["stats"]["types"]})
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Загрузка внешнего бенчмарка RusHallu-RAG")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "external" / "rushallu")
    parser.add_argument("--transport", choices=["auto", "http", "gh"], default="auto")
    parser.add_argument("--verify", action="store_true", help="падать при несовпадении SHA256")
    args = parser.parse_args()

    try:
        report = fetch(args.out, transport=args.transport, verify=args.verify)
    except FetchError as error:
        print(f"ошибка: {error}", file=sys.stderr)
        return 2

    summary = report["summary"]
    print(f"Пар: {report['pairs']} (источник: {SOURCE_REPO}@{SOURCE_COMMIT[:12]})")
    print(
        f"Чистых: {summary['clean']} | с галлюцинациями: {summary['with_hallucination']} | спанов: {summary['spans']}"
    )
    print(f"Неоднозначных смещений: {summary['ambiguous_offsets']}")
    print("Типы:", json.dumps(summary["types"], ensure_ascii=False))
    for name, info in report["files"].items():
        mark = "ок" if info["verified"] else "НЕ СОВПАЛ"
        print(f"  {name}: {info['rows']} строк, SHA256 {mark} ({info['transport']})")
    print(f"Пары: {args.out / 'pairs.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
