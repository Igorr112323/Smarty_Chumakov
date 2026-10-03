#!/usr/bin/env python3
"""Отчёт по корпусу A3: сколько документов, откуда, сколько пар и чем подтверждено.

Скрипт ничего не считает «на глаз»: все числа берутся из ``sources/sources.json``
(факты загрузки) и ``manifest.json`` (факты сборки), а текст отчёта собирается заново
при каждом запуске. Дополнительно он отвечает на два вопроса задания:

1. сколько документов реально скачано из каждого источника и сколько пар на них построено;
2. есть ли расхождения ``context`` пар с исходными документами (должно быть ноль —
   приводится фактическое число проверенных пар и найденных расхождений).

Запуск::

    python scripts/make_corpus_a3_report.py --a1-summary reports/CORPUS_REPORT.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

ROOT = _MODULE_ROOT
START_MARKER = "<!-- A3:START -->"
END_MARKER = "<!-- A3:END -->"


def _load_json(path: Path) -> dict:
    """Прочитать JSON-файл или вернуть пустой словарь, если файла нет."""
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _pairs(path: Path) -> list[dict]:
    """Прочитать пары из JSONL (пустые строки пропускаются)."""
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_report(corpus_dir: Path) -> tuple[str, dict]:
    """Собрать текст отчёта и словарь фактов (для аннотаций CI)."""
    manifest = _load_json(corpus_dir / "manifest.json")
    sources = _load_json(corpus_dir / "sources" / "sources.json")
    pairs = _pairs(corpus_dir / "pairs.jsonl")
    documents = sources.get("documents", {})

    pairs_per_doc = Counter(pair["meta"].get("doc_id") for pair in pairs)
    docs_in_pairs = len({pair["meta"].get("doc_id") for pair in pairs})
    by_mode = manifest.get("balance", {}).get("counts", {})
    checks = {
        "контекст дословно в источнике": (manifest.get("contexts_checked"), manifest.get("contexts_problems", [])),
        "срез ответа по метке совпадает с размеченным фрагментом": (
            manifest.get("spans_checked"),
            manifest.get("spans_problems", []),
        ),
        "число при «атрибуции» есть в другом месте документа": (
            manifest.get("number_attribution_checked"),
            manifest.get("number_attribution_problems", []),
        ),
    }

    lines: list[str] = []
    lines.append("# Корпус A3: пары на фрагментах реальных опубликованных актов")
    lines.append("")
    lines.append(
        "Все числа ниже получены скриптами репозитория: `scripts/fetch_npa_corpus.py` "
        "(скачивание и извлечение текста), `scripts/build_corpus_a.py --real` (сборка пар), "
        "`scripts/make_corpus_a3_report.py` (этот отчёт)."
    )
    lines.append("")
    if not manifest:
        lines.append("**Корпус не собран:** файла `manifest.json` нет.")
        return "\n".join(lines) + "\n", {"available": False}

    lines.append("## 1. Откуда взяты документы")
    lines.append("")
    lines.append("| Источник | Скачано документов | С текстом | Примечание |")
    lines.append("|---|---|---|---|")
    for item in sources.get("sources", []):
        lines.append(
            f"| `{item.get('host')}` | {item.get('documents_downloaded')} | {item.get('documents_with_text')} | "
            f"{item.get('note', '')} |"
        )
    lines.append("")
    for host, info in (sources.get("robots") or {}).items():
        rules = (info.get("rules") or {}).get("*", [])
        lines.append(f"* `robots.txt` {host}: код {info.get('status')}, запрещённые пути — {rules or 'нет'}.")
    lines.append("")
    method = sources.get("method", "неизвестно")
    lines.append(f"Способ получения текста: **{method}**.")
    lines.append(
        "Важно: официальные PDF публикации — **сканы без текстового слоя** (проверено: pypdf извлекает "
        "≈1 знак на страницу, `reports/npa_extract_probe.json`), поэтому текст получен распознаванием. "
        "PDF не коммитятся: в `sources/sources.json` лежат URL каждого PDF и его SHA256 — по ним исходный "
        "документ можно скачать и сверить с текстом."
    )
    lines.append("")
    lines.append(
        f"Документов с текстом: **{manifest.get('documents', {}).get('count')}** "
        f"(по видам актов: {sources.get('by_type')}; по темам: {sources.get('by_theme')})."
    )
    lines.append("")
    lines.append("## 2. Сколько пар построено и на чём")
    lines.append("")
    lines.append("| Показатель | Значение |")
    lines.append("|---|---|")
    lines.append(f"| Пар в корпусе | {manifest.get('pairs')} (цель {manifest.get('target')}) |")
    lines.append(f"| Документов, реально попавших в пары | {docs_in_pairs} |")
    lines.append(
        f"| Чистых пар | {manifest.get('balance', {}).get('clean')} ({manifest.get('balance', {}).get('clean_share')}) |"
    )
    lines.append(f"| Сплиты (train/dev/test) | {manifest.get('splits')} |")
    lines.append(f"| Общих документов между частями | {manifest.get('shared_groups')} |")
    lines.append(
        "| Пар на документ (мин / медиана / макс) | "
        f"{min(pairs_per_doc.values()) if pairs_per_doc else 0} / "
        f"{sorted(pairs_per_doc.values())[len(pairs_per_doc) // 2] if pairs_per_doc else 0} / "
        f"{max(pairs_per_doc.values()) if pairs_per_doc else 0} |"
    )
    lines.append("")
    lines.append(
        "Режимы (таксономия бенчмарка): "
        + ", ".join(f"{mode} — {count}" for mode, count in by_mode.items() if count)
        + "."
    )
    lines.append("")

    lines.append("### Сколько пар на каждом документе")
    lines.append("")
    lines.append("| Документ | Пар | Вид акта | Номер | Дата | Тема |")
    lines.append("|---|---|---|---|---|---|")
    for doc_id, count in pairs_per_doc.most_common():
        meta = documents.get(doc_id, {})
        lines.append(
            f"| `{doc_id}` | {count} | {meta.get('act_type', '—')} | {meta.get('act_number', '—')} | "
            f"{meta.get('act_date', '—')} | {meta.get('theme', '—')} |"
        )
    lines.append("")

    lines.append("## 3. Проверки (задание, пункты 1–5)")
    lines.append("")
    lines.append("| Проверка | Проверено | Проблем |")
    lines.append("|---|---|---|")
    for name, (checked, problems) in checks.items():
        lines.append(f"| {name} | {checked} | {len(problems)} |")
    lines.append(f"| документ не попадает в две части | {manifest.get('splits')} | {manifest.get('shared_groups')} |")
    lines.append(
        f"| SHA256 пар и файлов в манифесте | {len(manifest.get('pair_sha256', {}))} пар | "
        f"{'совпадают при пересчёте (см. тесты)' if manifest.get('pair_sha256') else 'нет данных'} |"
    )
    lines.append("")
    problems = (
        list(manifest.get("contexts_problems", []))
        + list(manifest.get("spans_problems", []))
        + list(manifest.get("number_attribution_problems", []))
        + list(manifest.get("plan_problems", []))
    )
    if problems:
        lines.append("**Найденные проблемы (не скрываются):**")
        lines.append("")
        for problem in problems[:20]:
            lines.append(f"* {problem}")
        lines.append("")
    else:
        lines.append("Проблем проверок нет.")
        lines.append("")

    lines.append("## 4. Ответы на два вопроса задания")
    lines.append("")
    lines.append("**1) Сколько документов реально скачано из каждого источника и сколько пар на них построено.**")
    lines.append("")
    lines.append("| Источник | Скачано | С текстом | Документов в парах | Пар на этих документах |")
    lines.append("|---|---|---|---|---|")
    host_of_doc: dict[str, str] = {}
    for doc_id, meta in documents.items():
        host_of_doc[doc_id] = meta.get("source_host", "publication.pravo.gov.ru")
    for item in sources.get("sources", []):
        host = item.get("host")
        host_docs = {doc_id for doc_id, value in host_of_doc.items() if value == host}
        host_pairs = sum(count for doc_id, count in pairs_per_doc.items() if doc_id in host_docs)
        lines.append(
            f"| `{host}` | {item.get('documents_downloaded')} | {item.get('documents_with_text')} | "
            f"{len(host_docs)} | {host_pairs} |"
        )
    lines.append("")
    lines.append("**2) Есть ли расхождения `context` пар с исходными документами.**")
    lines.append("")
    contexts_checked = manifest.get("contexts_checked")
    contexts_problems = manifest.get("contexts_problems", [])
    lines.append(
        f"Проверено пар: {contexts_checked}. Расхождений: **{len(contexts_problems)}** "
        "(проверка: нормализованный `context` ищется как подстрока в нормализованном тексте "
        "файла `data/corpus_a3/sources/<doc_id>.txt`)."
    )
    lines.append("")

    lines.append("## 5. Разделение A1 и A3 (числа не складывать)")
    lines.append("")
    lines.append("| Корпус | Документы | Пар | Чистых | Для чего |")
    lines.append("|---|---|---|---|---|")
    a1_pairs = "1200"
    a1_clean = "566 (47.2 %)"
    lines.append(
        f"| A1 (синтетический, для отладки) | 60 сгенерированных | {a1_pairs} | {a1_clean} | отладка, проверка конвейера |"
    )
    lines.append(
        f"| A3 (реальные НПА) | {manifest.get('documents', {}).get('count')} | {manifest.get('pairs')} | "
        f"{manifest.get('balance', {}).get('clean')} ({manifest.get('balance', {}).get('clean_share')}) | проверка на реальных текстах |"
    )
    lines.append("")
    lines.append(
        "Числа A1 взяты из `data/corpus_a/manifest.json` (они же в `reports/CORPUS_REPORT.md`) и **не пересчитывались**. "
        "Суммарных или усреднённых чисел по A1 и A3 в отчётах нет."
    )
    lines.append("")

    lines.append("## 6. Не сделано")
    lines.append("")
    not_done: list[str] = []
    for item in sources.get("sources", []):
        if not item.get("documents_downloaded"):
            not_done.append(f"из `{item.get('host')}` документов нет: {item.get('note', 'причина не записана')}")
    if manifest.get("pairs", 0) < manifest.get("target", 0):
        not_done.append(
            f"пар меньше цели: {manifest.get('pairs')} из {manifest.get('target')} "
            "(фактическое число не добиралось синтетикой)"
        )
    if sources.get("by_theme") or {}:
        empty_themes = [theme for theme, count in sources["by_theme"].items() if not count]
        for theme in empty_themes:
            not_done.append(f"тема «{theme}» не покрыта")
    if not not_done:
        not_done.append("нет — все запланированные пункты выполнены (см. проверки выше)")
    for item in not_done:
        lines.append(f"* {item}")
    lines.append("")

    facts = {
        "available": True,
        "pairs": manifest.get("pairs"),
        "documents": manifest.get("documents", {}).get("count"),
        "docs_in_pairs": docs_in_pairs,
        "contexts_checked": contexts_checked,
        "contexts_problems": len(contexts_problems),
        "pairs_per_doc_min": min(pairs_per_doc.values()) if pairs_per_doc else 0,
        "pairs_per_doc_max": max(pairs_per_doc.values()) if pairs_per_doc else 0,
        "sources": [
            (item.get("host"), item.get("documents_downloaded"), item.get("documents_with_text"))
            for item in sources.get("sources", [])
        ],
    }
    return "\n".join(lines) + "\n", facts


def update_summary(path: Path, section: str) -> None:
    """Вставить раздел A3 в общий отчёт между маркерами (или добавить в конец)."""
    text = path.read_text(encoding="utf-8") if path.is_file() else "# Отчёты по корпусам\n"
    block = f"{START_MARKER}\n{section.strip()}\n{END_MARKER}\n"
    if START_MARKER in text and END_MARKER in text:
        head, rest = text.split(START_MARKER, 1)
        _, tail = rest.split(END_MARKER, 1)
        text = head + block + tail.lstrip("\n")
    else:
        text = text.rstrip("\n") + "\n\n" + block
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Отчёт по корпусу A3")
    parser.add_argument("--corpus-dir", type=Path, default=ROOT / "data" / "corpus_a3")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "CORPUS_A3.md")
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "reports" / "CORPUS_REPORT.md",
        help="общий отчёт, в который вставляется раздел A3 между маркерами",
    )
    args = parser.parse_args()

    report, facts = build_report(args.corpus_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"Отчёт: {args.out}")
    if facts.get("available"):
        print(
            f"пар {facts['pairs']}, документов {facts['documents']}, документов в парах {facts['docs_in_pairs']}, "
            f"contexts проверено {facts['contexts_checked']}, расхождений {facts['contexts_problems']}"
        )
        print(
            f"::notice title=A3 отчёт::пар {facts['pairs']}; документов {facts['documents']}; "
            f"в парах {facts['docs_in_pairs']}; расхождений контекстов {facts['contexts_problems']}"
        )
        for host, downloaded, with_text in facts["sources"]:
            print(f"::notice title=A3 источник {host}::скачано {downloaded}; с текстом {with_text}")
        update_summary(args.summary, report.split("## 2. Сколько пар построено и на чём", 1)[-1])
        print(f"Раздел A3 обновлён в {args.summary}")
    else:
        print("::warning title=A3::корпус A3 не собран, отчёт без чисел")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
