"""Обзор аналогов и патентный поиск (пункт 2.6 плана аудита: P2-6).

Скрипт обращается к открытым программным интерфейсам, доступным из сети раннера
GitHub Actions, и складывает результат в JSON **без интерпретации**: что вернул
источник, то и записано, вместе с датой запроса, адресом и статусом.

Источники:

* OpenAlex (``api.openalex.org``) — научные публикации по запросам о детекции
  галлюцинаций и атрибуции;
* Crossref (``api.crossref.org``) — метаданные публикаций;
* GitHub Search API (``api.github.com``) — открытые реализации-аналоги;
* Google Patents через открытую выгрузку патентов (``patents.google.com/xhr``)
  — если отвечает; иначе фиксируется фактический код ответа.

Ничего не «досочиняется»: при недоступности источника в JSON стоит
``status`` и ``error``, а в ``found`` — пустой список.

Запуск::

    python scripts/prior_art_search.py --out reports/prior_art.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "SpanVerify-prior-art/1.0 (research; contact: repository issues)"

#: Запросы: (тема, текст запроса).
QUERIES: tuple[tuple[str, str], ...] = (
    ("детекция галлюцинаций", "hallucination detection retrieval augmented generation"),
    ("атрибуция ответа", "attribution of generated text to source documents"),
    ("неопределённость внимания", "attention entropy uncertainty language model factuality"),
    ("проверка фактов", "fact verification against source document Russian"),
    ("доля участия ИИ", "AI-generated text detection share of machine text"),
)


def get_json(url: str, timeout: int = 40) -> dict:
    """Запрос JSON: вернуть ``{status, data|error, url}`` без исключений."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
        return {"status": response.status, "data": json.loads(raw), "url": url, "error": None}
    except urllib.error.HTTPError as exc:
        return {"status": exc.code, "data": None, "url": url, "error": f"HTTP {exc.code}: {exc.reason}"}
    except Exception as exc:  # noqa: BLE001 - сеть раннера может отказать, это факт для отчёта
        return {"status": None, "data": None, "url": url, "error": f"{type(exc).__name__}: {exc}"}


def openalex(query: str, per_page: int = 5) -> dict:
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(
        {"search": query, "per-page": per_page, "mailto": "spanverify@example.invalid"}
    )
    result = get_json(url)
    works = []
    for item in (result.get("data") or {}).get("results", [])[:per_page]:
        works.append(
            {
                "title": item.get("title"),
                "year": item.get("publication_year"),
                "doi": item.get("doi"),
                "cited_by": item.get("cited_by_count"),
                "url": (item.get("primary_location") or {}).get("landing_page_url"),
            }
        )
    return {"status": result["status"], "error": result["error"], "url": url, "found": works}


def crossref(query: str, rows: int = 5) -> dict:
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode({"query": query, "rows": rows})
    result = get_json(url)
    items = []
    for item in (result.get("data") or {}).get("message", {}).get("items", [])[:rows]:
        items.append(
            {
                "title": (item.get("title") or [None])[0],
                "container": (item.get("container-title") or [None])[0],
                "year": ((item.get("issued") or {}).get("date-parts") or [[None]])[0][0],
                "doi": item.get("DOI"),
                "url": item.get("URL"),
            }
        )
    return {"status": result["status"], "error": result["error"], "url": url, "found": items}


def github_repositories(query: str, per_page: int = 5) -> dict:
    url = "https://api.github.com/search/repositories?" + urllib.parse.urlencode(
        {"q": query, "per_page": per_page, "sort": "stars", "order": "desc"}
    )
    result = get_json(url)
    items = []
    for item in (result.get("data") or {}).get("items", [])[:per_page]:
        items.append(
            {
                "name": item.get("full_name"),
                "stars": item.get("stargazers_count"),
                "language": item.get("language"),
                "description": (item.get("description") or "")[:200],
                "url": item.get("html_url"),
                "updated": item.get("updated_at"),
            }
        )
    return {"status": result["status"], "error": result["error"], "url": url, "found": items}


def patent_search(query: str) -> dict:
    """Патентный поиск через открытый интерфейс Google Patents (XHR, JSON)."""
    url = "https://patents.google.com/xhr/query?" + urllib.parse.urlencode(
        {"url": urllib.parse.quote(f"q={query}&num=5"), "exp": ""}
    )
    result = get_json(url)
    found = []
    data = result.get("data") or {}
    clusters = (((data.get("results") or {}).get("cluster") or [{}])[0]).get("result") or []
    for item in clusters[:5]:
        patent = item.get("patent") or {}
        found.append(
            {
                "id": patent.get("publication_number"),
                "title": patent.get("title"),
                "assignee": patent.get("assignee"),
                "publication_date": patent.get("publication_date"),
                "url": f"https://patents.google.com/patent/{patent.get('publication_number', '')}",
            }
        )
    return {"status": result["status"], "error": result["error"], "url": url, "found": found}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Обзор аналогов и патентный поиск")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "prior_art.json")
    parser.add_argument("--pause", type=float, default=1.0, help="пауза между запросами, секунд")
    parser.add_argument("--limit-queries", type=int, default=0, help="ограничить число запросов (0 — все)")
    args = parser.parse_args(argv)

    started = datetime.now(timezone.utc).isoformat()
    topics = QUERIES[: args.limit_queries] if args.limit_queries else QUERIES
    report: dict = {
        "generated_at": started,
        "user_agent": USER_AGENT,
        "note": (
            "Результаты сохранены как есть, без интерпретации. Недоступность источника "
            "фиксируется кодом ответа и текстом ошибки; выводы делает человек."
        ),
        "topics": {},
    }
    for topic, query in topics:
        entry = {
            "query": query,
            "openalex": openalex(query),
            "crossref": crossref(query),
            "github": github_repositories(query),
            "patents": patent_search(query),
        }
        report["topics"][topic] = entry
        found = sum(len(entry[key]["found"]) for key in ("openalex", "crossref", "github", "patents"))
        print(
            f"{topic}: найдено записей {found} (openalex {len(entry['openalex']['found'])}, "
            f"crossref {len(entry['crossref']['found'])}, github {len(entry['github']['found'])}, "
            f"patents {len(entry['patents']['found'])})"
        )
        time.sleep(max(0.0, args.pause))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    md = render_markdown(report)
    args.out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(f"отчёт: {args.out} и {args.out.with_suffix('.md')}")
    return 0


def render_markdown(report: dict) -> str:
    """Отчёт по аналогам: таблицы из фактически полученных записей."""
    lines = [
        "# Обзор аналогов и патентный поиск",
        "",
        f"Дата запроса: {report['generated_at']}. {report['note']}",
        "",
    ]
    for topic, entry in report["topics"].items():
        lines += [f"## {topic}", "", f"Запрос: `{entry['query']}`", ""]
        for source, title in (
            ("openalex", "OpenAlex"),
            ("crossref", "Crossref"),
            ("github", "GitHub"),
            ("patents", "Google Patents"),
        ):
            payload = entry[source]
            status = payload.get("status")
            error = payload.get("error")
            lines.append(
                f"**{title}** — статус {status if status is not None else 'нет ответа'}"
                + (f" ({error})" if error else "")
                + f", найдено {len(payload['found'])}."
            )
            for item in payload["found"]:
                label = item.get("title") or item.get("name")
                url = item.get("url") or ""
                year = item.get("year") or item.get("publication_date") or ""
                lines.append(f"* {label} {year} — {url}")
            lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
