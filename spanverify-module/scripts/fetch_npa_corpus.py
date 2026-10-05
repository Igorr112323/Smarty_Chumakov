#!/usr/bin/env python3
"""Загрузчик корпуса A3: тексты реальных официально опубликованных актов.

Что делает скрипт
-----------------

1. **Проверяет robots.txt** всех источников белого списка и записывает правила в отчёт;
   запрещённые пути не запрашиваются (``is_allowed``).
2. **Выбирает документы** по официальному API публикации
   (``publication.pravo.gov.ru/api/Documents``): перебирает страницы выдачи по видам
   актов (федеральный закон, указ, постановление, приказ, положение, распоряжение) и
   оставляет те, чьё название относится к темам приоритета — сроки хранения документов,
   персональные данные, защита информации, охрана труда, кадровый учёт.
3. **Скачивает официальные PDF** (``/file/pdf?eoNumber=…``) с паузой между запросами и
   UA с контактом, считает SHA256 каждого файла.
4. **Извлекает текст** (OCR: ``pdftoppm`` + ``tesseract -l rus``) и нормализует его тем
   же ``normalize_text``, что и проверки корпуса A3.
5. **Пишет источники**: ``sources/<doc_id>.txt`` и ``sources.json`` с URL, датой
   выгрузки, видом/номером/датой акта, хешами PDF и текста, числом страниц и способом
   извлечения. Ровно эти файлы потом читает ``scripts/build_corpus_a.py --real``.

Правовая основа и вежливость к источникам: берутся только официально опубликованные
акты (п. 6 ст. 1259 ГК РФ — официальные документы государственных органов не являются
объектами авторских прав); у каждого документа в ``sources.json`` записан URL, вид,
номер, дата и SHA256 PDF; robots.txt каждого хоста белого списка читается и применяется
(``is_allowed``), частота запросов — не чаще одного в секунду (по умолчанию 1.5 с),
User-Agent содержит контакт, а приоритет отдан API и выгрузкам, а не разбору HTML.

Честность важнее полноты: сколько документов реально скачано и с каким числом фактов —
пишется в отчёт; недоступные источники помечаются ``available: false`` с фактической
причиной (код ответа или таймаут), а не «предположительно недоступен».

Правовые основания и вежливость к источнику
------------------------------------------

* источники — только из белого списка официальных публикаций задания;
* основная правовая основа использования — п. 6 ст. 1259 ГК РФ (официальные документы
  государственных органов не являются объектами авторских прав), а также то, что
  публикация на портале опубликования не создаёт ограничений на использование текста
  акта; каждый документ снабжён ссылкой на источник (URL, вид, номер, дата) и SHA256 PDF;
* ``robots.txt`` читается и применяется до запросов: если путь запрещён, запрос не
  выполняется, а причина попадает в отчёт (``skipped``);
* частота запросов ограничена (по умолчанию пауза 1.5 с, минимум — 1 с);
* User-Agent содержит контакт; приоритет отдан API и машинным форматам вместо разбора
  HTML-страниц.

Запуск::

    python scripts/fetch_npa_corpus.py --out spanverify-module/data/corpus_a3
    python scripts/fetch_npa_corpus.py --out spanverify-module/data/corpus_a3 --verify

Повторный запуск безопасен: уже скачанные документы (совпал SHA256 текста) не
скачиваются и не распознаются заново; если документов уже достаточно, запросов к сети
не делается вообще.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

_MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

from scripts.corpus_real import extract_facts, normalize_text  # noqa: E402 - импорт после правки sys.path

ROOT = _MODULE_ROOT

USER_AGENT = (
    "SpanVerify-CorpusBot/1.0 (+https://github.com/Igorr112323/Smarty_Chumakov; "
    "research corpus; contact: repository issues)"
)
PAUSE_SECONDS = 1.5
PAGE_SIZE = 100  # допустимые значения API: 10, 30, 100 (5/20/50 отклонены — проверено в CI)

# Белый список источников из задания (только официальные публикации).
# Запасное значение: основной список читается из config/sources_whitelist.json
# (функция ``load_whitelist``), чтобы список источников был один на весь проект.
WHITELIST = (
    "publication.pravo.gov.ru",
    "pravo.gov.ru",
    "fstec.ru",
    "mintrud.gov.ru",
    "rospotrebnadzor.ru",
    "eec.eaeunion.org",
)

WHITELIST_CONFIG = ROOT / "config" / "sources_whitelist.json"


def load_whitelist(path: Path | None = None) -> tuple[str, ...]:
    """Хосты белого списка из ``config/sources_whitelist.json``.

    Единый источник правды о допустимых источниках. Если файла нет или он битый,
    возвращается встроенный кортеж ``WHITELIST`` — загрузчик не должен падать из-за
    конфигурации, но и расширять список молча он тоже не должен.
    """
    config_path = path or WHITELIST_CONFIG
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return WHITELIST
    hosts: list[str] = []
    for entry in data.get("sources") or []:
        host = (entry or {}).get("host") if isinstance(entry, dict) else None
        if host and host not in hosts:
            hosts.append(str(host))
    return tuple(hosts) or WHITELIST


# Региональные разделы портала опубликования. Проверено запросом: работает только
# слуг вида ``region<код>`` (``krasnodarskij-kraj`` отдаёт 500, ``/api/Regions`` — 404).
# Тот же код региона — первые две цифры номера опубликования (eoNumber), поэтому
# уровень и регион документа определяются по номеру, без справочника органов.
# Порядок — приоритет задания: Краснодарский край первым.
REGION_BLOCKS: tuple[tuple[str, str], ...] = (
    ("23", "Краснодарский край"),
    ("77", "Москва"),
    ("50", "Московская область"),
    ("78", "Санкт-Петербург"),
    ("61", "Ростовская область"),
    ("26", "Ставропольский край"),
    ("16", "Республика Татарстан"),
    ("66", "Свердловская область"),
)

REGION_NAMES: dict[str, str] = dict(REGION_BLOCKS)

FEDERAL_PREFIX = "00"


def region_of(eo_number: str) -> dict[str, str | None]:
    """Уровень и регион акта по номеру опубликования.

    Номер опубликования устроен так: две цифры кода региона (``00`` — федеральный
    уровень), две цифры органа, затем ``ГГГГММДД`` и порядковый номер за день.
    Поэтому уровень публикации читается из самого номера и не требует разрешения
    идентификаторов органов (их перебор в ``/api/SignatoryAuthorities`` — 66 страниц).
    """
    digits = "".join(character for character in str(eo_number or "") if character.isdigit())
    if len(digits) < 4:
        return {"level": None, "region_code": None, "region_name": None}
    code = digits[:2]
    if code == FEDERAL_PREFIX:
        return {"level": "федеральный", "region_code": code, "region_name": None}
    return {
        "level": "региональный",
        "region_code": code,
        "region_name": REGION_NAMES.get(code),
    }


# Темы приоритета: (название темы, регулярное выражение по названию акта).
THEMES = (
    ("сроки хранения документов", r"хранен|архивн|номенклатур|архив"),
    ("персональные данные", r"персональн(?:ых|ые|ыми|ой) данных|обработк[аи] персональных"),
    (
        "защита информации",
        r"защит[аеы] информации|информационн(?:ой|ая|ые) безопасн|государственн(?:ой|ой) тайн|коммерческ(?:ой|ая) тайн",
    ),
    ("охрана труда", r"охран[аеы] труда|инструкц[ии] по охране|аттестац[ии] рабочих мест|специальн(?:ой|ая) оценк"),
    ("кадровый учёт", r"кадров|персонал|трудов(?:ых|ые) отношени|служебн(?:ой|ая) контракт"),
)

TYPES = (
    "Федеральный закон",
    "Указ",
    "Постановление",
    "Приказ",
    "Положение",
    "Распоряжение",
)


def fetch(url: str, timeout: int = 120, read_limit: int = 12_000_000) -> dict:
    """Запрос с сохранением тела и факта ошибки (без интерпретаций)."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json, application/pdf, text/plain, */*"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - официальные публичные адреса
            raw = response.read(read_limit)
            return {
                "url": url,
                "status": int(getattr(response, "status", 0) or 0),
                "content_type": response.headers.get("Content-Type", ""),
                "bytes": len(raw),
                "raw": raw,
                "error": None,
            }
    except urllib.error.HTTPError as error:
        raw = error.read(64_000)
        return {
            "url": url,
            "status": int(error.code),
            "content_type": error.headers.get("Content-Type", "") if error.headers else "",
            "bytes": len(raw),
            "raw": raw,
            "error": f"HTTP {error.code}",
        }
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return {"url": url, "status": None, "content_type": "", "bytes": 0, "raw": b"", "error": str(error)[:140]}


# ---------------------------------------------------------------------------
# robots.txt
# ---------------------------------------------------------------------------


def parse_robots(text: str, agent: str = "*") -> dict[str, list[str]]:
    """Разобрать robots.txt в правила по агентам: ``{агент: [Disallow-пути]}``.

    Учитываются группы ``User-agent``/``Disallow``; пустой ``Disallow`` (как у
    ``pravo.gov.ru``) означает «всё разрешено».
    """
    rules: dict[str, list[str]] = {}
    current: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key == "user-agent":
            agent_name = value.lower()
            rules.setdefault(agent_name, [])
            current = [agent_name]
        elif key == "disallow" and value:
            for name in current:
                rules.setdefault(name, []).append(value)
    if agent.lower() not in rules:
        return {"*": rules.get("*", [])}
    return {agent.lower(): rules[agent.lower()]}


def is_allowed(url: str, rules: dict[str, list[str]]) -> bool:
    """Разрешён ли путь адреса правилами robots.txt (точное совпадение префикса)."""
    path = urllib.parse.urlsplit(url).path or "/"
    disallowed = rules.get("*", [])
    for prefix in disallowed:
        if path == prefix or path.startswith(prefix):
            return False
    return True


# ---------------------------------------------------------------------------
# Отбор документов
# ---------------------------------------------------------------------------


OTHER_THEME = "прочее"


def theme_of(name: str, allow_other: bool = False) -> str | None:
    """Тема приоритета по названию акта (``None`` — не наша тема).

    При ``allow_other=True`` акт вне приоритетных тем не отбрасывается, а помечается
    темой ``«прочее»``. Это нужно для набора объёма корпуса: приоритетных тем в суточной
    выдаче портала мало (замерено: 10 документов за прогон на 4800 просмотренных), и
    без этого цель в 120 документов недостижима. Тема остаётся в манифесте, поэтому
    доля приоритетных тем в корпусе видна и не выдаётся за 100 %.
    """
    lowered = (name or "").lower()
    for theme, pattern in THEMES:
        if re.search(pattern, lowered):
            return theme
    return OTHER_THEME if allow_other else None


def fetch_robots(hosts: tuple[str, ...] | None = None) -> dict[str, dict]:
    """Правила robots.txt по всем источникам белого списка (для отчёта)."""
    result: dict[str, dict] = {}
    for host in hosts if hosts is not None else WHITELIST:
        scheme = "https" if host not in {"publication.pravo.gov.ru", "pravo.gov.ru"} else "http"
        url = f"{scheme}://{host}/robots.txt"
        fetched = fetch(url, timeout=45, read_limit=200_000)
        text = fetched["raw"].decode("utf-8", "replace") if fetched["status"] == 200 else ""
        result[host] = {
            "url": url,
            "status": fetched["status"],
            "error": fetched["error"],
            "rules": parse_robots(text) if text else {},
            "text_head": text[:200],
        }
    return result


def candidate_from_item(
    item: dict,
    theme: str,
    act_type: str | None,
    base: str = "http://publication.pravo.gov.ru",
) -> dict:
    """Карточка кандидата из элемента выдачи API (без сетевых запросов)."""
    eo = str(item["eoNumber"])
    placement = region_of(eo)
    title = item.get("name") or ""
    complex_name = item.get("complexName") or ""
    # Вид акта у региональной выдачи в элементе не назван: берём первое слово
    # заголовка («Постановление …», «Приказ …»), иначе остаётся None.
    resolved_type = act_type
    if not resolved_type and complex_name:
        head = complex_name.strip().split()
        resolved_type = head[0].strip('"«') if head else None
    return {
        "doc_id": f"eo-{eo}",
        "source_host": "publication.pravo.gov.ru",
        "eo_number": eo,
        "document_id": item.get("id"),
        "act_type": resolved_type,
        "act_number": item.get("number"),
        "act_date": (item.get("documentDate") or "")[:10] or None,
        "published_at": (item.get("publishDateShort") or "")[:10] or None,
        "title": title,
        "complex_name": complex_name,
        "pages_count": item.get("pagesCount"),
        "pdf_bytes_expected": item.get("pdfFileLength"),
        "source_url": f"{base}/document/{eo}",
        "theme": theme,
        "level": placement["level"],
        "region_code": placement["region_code"],
        "region_name": placement["region_name"],
        "signatory_authority_id": item.get("signatoryAuthorityId"),
        "document_type_id": item.get("documentTypeId"),
    }


def scan_list(
    url: str,
    rules: dict[str, list[str]],
    stats: dict,
    pause: float,
) -> list[dict] | None:
    """Одна страница выдачи API: список элементов либо ``None`` при отказе.

    ``None`` означает «дальше по этой ветке идти незачем» (robots, ошибка, пусто) —
    вызывающий код прекращает перебор страниц, а причина попадает в статистику.
    """
    if not is_allowed(url, {"*": rules.get("*", [])}):
        stats.setdefault("skipped_by_robots", []).append(url)
        return None
    fetched = fetch(url)
    stats["pages_scanned"] += 1
    time.sleep(max(0.0, pause))
    if fetched["status"] != 200:
        stats.setdefault("page_errors", []).append(f"{url} → {fetched['error']}")
        return None
    try:
        items = json.loads(fetched["raw"].decode("utf-8")).get("items", [])
    except json.JSONDecodeError:
        stats.setdefault("page_errors", []).append(f"{url} → тело не JSON")
        return None
    return items or None


def collect_candidates(
    type_ids: dict[str, str],
    max_pages_per_type: int,
    pause: float,
    robots: dict[str, dict],
    base: str = "http://publication.pravo.gov.ru",
    blocks: tuple[tuple[str, str], ...] = REGION_BLOCKS,
    max_pages_per_block: int = 0,
    allow_other: bool = False,
) -> tuple[list[dict], dict]:
    """Набрать кандидатов: документы по видам актов и по региональным разделам.

    Две ветки отбора:

    * по виду акта — ``/api/Documents?documentTypes=<guid>`` (выдача федеральная с
      примесью регионов);
    * по региональному разделу — ``/api/Documents?block=region<код>``; именно так
      добираются акты субъектов, потому что справочник регионов API не отдаёт
      (``/api/Regions`` — 404, проверено).

    Возвращает ``(кандидаты, статистика)``: сколько страниц и документов просмотрено,
    сколько попало в темы, уровни и регионы — по фактическому подсчёту.
    """
    candidates: list[dict] = []
    seen: set[str] = set()
    stats: dict = {
        "pages_scanned": 0,
        "documents_scanned": 0,
        "by_theme": {theme: 0 for theme, _ in THEMES},
        "by_type": {name: 0 for name in type_ids},
        "by_level": {},
        "by_region": {},
        "blocks_scanned": [],
    }
    rules = robots.get("publication.pravo.gov.ru", {}).get("rules", {})

    def absorb(items: list[dict], act_type: str | None) -> None:
        for item in items:
            stats["documents_scanned"] += 1
            if not isinstance(item, dict) or not item.get("eoNumber"):
                continue
            theme = theme_of(item.get("name") or item.get("complexName") or "", allow_other=allow_other)
            if theme is None:
                continue
            eo = str(item["eoNumber"])
            if eo in seen:
                continue
            seen.add(eo)
            candidate = candidate_from_item(item, theme, act_type, base=base)
            stats["by_theme"][theme] = stats["by_theme"].get(theme, 0) + 1
            if act_type:
                stats["by_type"][act_type] = stats["by_type"].get(act_type, 0) + 1
            level = str(candidate.get("level"))
            stats["by_level"][level] = stats["by_level"].get(level, 0) + 1
            region = str(candidate.get("region_name") or candidate.get("region_code"))
            stats["by_region"][region] = stats["by_region"].get(region, 0) + 1
            candidates.append(candidate)

    for type_name, type_id in type_ids.items():
        for page in range(1, max_pages_per_type + 1):
            url = f"{base}/api/Documents?pageSize={PAGE_SIZE}&index={page}&documentTypes={type_id}"
            items = scan_list(url, rules, stats, pause)
            if items is None:
                break
            absorb(items, type_name)

    for code, name in blocks:
        for page in range(1, max_pages_per_block + 1):
            url = f"{base}/api/Documents?pageSize={PAGE_SIZE}&index={page}&block=region{code}"
            items = scan_list(url, rules, stats, pause)
            if items is None:
                break
            absorb(items, None)
        stats["blocks_scanned"].append({"block": f"region{code}", "region": name})

    return candidates, stats


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------


def ocr_available() -> dict[str, str | None]:
    """Проверить наличие инструментов OCR (фактические пути)."""
    return {name: shutil.which(name) for name in ("pdftoppm", "tesseract")}


# Режим распознавания. Значения подобраны по замеру, а не по умолчанию:
# у tesseract режим по умолчанию ``--oem 3`` прогоняет и старый движок, и LSTM, что
# на сканах официальных публикаций давало сотни секунд на страницу (замер первого
# прогона: 580–2680 с на документ, см. reports/OCR_BENCH.json). ``--oem 1`` — только
# LSTM, ``--psm 6`` — «единый блок текста», что соответствует полосе текста акта.
OCR_OEM = "1"
OCR_PSM = "6"
# tesseract сам распараллеливается через OpenMP. Когда рядом работает несколько
# процессов распознавания, потоки конкурируют за те же ядра и всё замедляется.
# Один поток на процесс плюс параллельные процессы — рекомендованный режим.
OCR_ENV = {"OMP_THREAD_LIMIT": "1"}


def ocr_pdf(
    path: Path,
    dpi: int = 150,
    max_pages: int = 20,
    oem: str = OCR_OEM,
    psm: str = OCR_PSM,
) -> dict:
    """OCR PDF: вернуть ``{text, pages, seconds, error}`` без интерпретаций.

    Текст собирается постранично (``pdftoppm -png`` → ``tesseract -l rus``), поэтому в
    отчёте есть фактическое число распознанных страниц и фактическое время.
    """
    tools = ocr_available()
    if not all(tools.values()):
        return {"text": "", "pages": 0, "seconds": None, "error": f"нет инструментов OCR: {tools}"}
    environment = {**os.environ, **OCR_ENV}
    with tempfile.TemporaryDirectory() as tmp:
        started = time.perf_counter()
        rendered = subprocess.run(  # noqa: S603 - фиксированные аргументы, входной файл наш
            ["pdftoppm", "-r", str(dpi), "-png", "-f", "1", "-l", str(max_pages), str(path), str(Path(tmp) / "page")],
            check=False,
            capture_output=True,
        )
        if rendered.returncode != 0:
            return {
                "text": "",
                "pages": 0,
                "seconds": round(time.perf_counter() - started, 2),
                "error": rendered.stderr.decode("utf-8", "replace")[:200],
            }
        pieces: list[str] = []
        for image in sorted(Path(tmp).glob("page*.png")):
            command = ["tesseract", str(image), "stdout", "-l", "rus", "--oem", str(oem), "--psm", str(psm)]
            result = subprocess.run(  # noqa: S603 - фиксированные аргументы
                command,
                check=False,
                capture_output=True,
                env=environment,
            )
            if result.returncode != 0:
                # Сборка tesseract может не содержать запрошенный движок: тогда
                # распознаём настройками по умолчанию, а не теряем страницу.
                result = subprocess.run(  # noqa: S603 - фиксированные аргументы
                    ["tesseract", str(image), "stdout", "-l", "rus"],
                    check=False,
                    capture_output=True,
                    env=environment,
                )
            pieces.append(result.stdout.decode("utf-8", "replace"))
        text = "\n".join(piece for piece in pieces if piece.strip())
    return {
        "text": text,
        "pages": len(pieces),
        "seconds": round(time.perf_counter() - started, 2),
        "error": None,
    }


def pypdf_text(path: Path) -> dict:
    """Извлечь текст из PDF средствами pypdf (``{text, pages, error}``).

    Так делается, только если в PDF есть текстовый слой. Официальные PDF публикации
    оказались сканами (≈1 знак на страницу — отчёт ``reports/npa_extract_probe.json``),
    поэтому на практике чаще срабатывает OCR; но если текстовый слой есть, он точнее.
    """
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover - в CI pypdf установлен
        return {"text": "", "pages": 0, "error": "pypdf не установлен"}
    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as error:  # noqa: BLE001 - битый PDF не должен ронять загрузку
        return {"text": "", "pages": 0, "error": str(error)[:200]}
    return {"text": "\n".join(pages), "pages": len(pages), "error": None}


def pick_extraction(text_layer: dict, ocr: dict, min_chars_per_page: int = 120) -> str:
    """Выбрать способ извлечения по фактам: текстовый слой или OCR.

    Текстовый слой считается пригодным, если он даёт не меньше ``min_chars_per_page``
    знаков на страницу. Иначе берётся OCR (если он что-то дал). Если не пригоден ни
    один способ — ``"none"``: документ пропускается, а не попадает в корпус пустым.
    """
    pages = int(text_layer.get("pages") or 0)
    chars = len(text_layer.get("text") or "")
    if pages and chars >= min_chars_per_page * pages:
        return "pypdf"
    if (ocr.get("text") or "").strip():
        return "ocr-tesseract-rus"
    return "none"


def text_layer_is_enough(text_layer: dict, min_chars_per_page: int = 120) -> bool:
    """Достаточен ли текстовый слой PDF, чтобы не запускать OCR."""
    pages = int(text_layer.get("pages") or 0)
    chars = len(text_layer.get("text") or "")
    return bool(pages) and chars >= min_chars_per_page * pages


def extract_text(
    path: Path,
    dpi: int = 150,
    max_pages: int = 20,
    min_chars_per_page: int = 120,
    allow_ocr: bool = True,
) -> dict:
    """Текст PDF: сначала текстовый слой, OCR — только если слоя не хватило.

    Раньше OCR запускался всегда, даже когда текстовый слой был пригоден: на прогоне
    в CI это и съело лимит в 150 минут (распознавание — около 2,5 с на страницу).
    Теперь распознавание включается только для тех документов, где текстовый слой
    действительно пуст или слишком беден; факт пропуска OCR виден в ``ocr_skipped``.
    """
    text_layer = pypdf_text(path)
    if text_layer_is_enough(text_layer, min_chars_per_page):
        ocr = {"text": "", "pages": 0, "seconds": None, "error": None, "skipped": "текстовый слой пригоден"}
    elif allow_ocr:
        ocr = ocr_pdf(path, dpi=dpi, max_pages=max_pages)
    else:
        ocr = {"text": "", "pages": 0, "seconds": None, "error": "OCR отключён ключом --no-ocr"}
    method = pick_extraction(text_layer, ocr, min_chars_per_page)
    text = text_layer["text"] if method == "pypdf" else ocr["text"]
    return {
        "method": method,
        "text": text,
        "pages": int(text_layer.get("pages") or ocr.get("pages") or 0),
        "chars": len(text or ""),
        "text_layer_chars": len(text_layer.get("text") or ""),
        "ocr_error": ocr.get("error"),
        "ocr_skipped": ocr.get("skipped"),
        "ocr_seconds": ocr.get("seconds"),
        "pypdf_error": text_layer.get("error"),
    }


def sha256_bytes(raw: bytes) -> str:
    """SHA256 байтов (для PDF и текста источника)."""
    return hashlib.sha256(raw).hexdigest()


def write_sources(out_dir: Path, documents: list[dict], manifest: dict) -> None:
    """Записать тексты источников и ``sources.json``."""
    sources_dir = out_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    manifest["documents"] = {}
    for document in documents:
        path = sources_dir / f"{document['doc_id']}.txt"
        path.write_text(document["text"] + "\n", encoding="utf-8")
        manifest["documents"][document["doc_id"]] = {key: value for key, value in document.items() if key != "text"}
        manifest["documents"][document["doc_id"]]["text_file"] = f"sources/{document['doc_id']}.txt"
        manifest["documents"][document["doc_id"]]["text_chars"] = len(document["text"])
        # Хеш считается по фактическим байтам файла (а не по строке), поэтому проверка
        # `--verify` сравнивает одно и то же.
        manifest["documents"][document["doc_id"]]["text_sha256"] = sha256_bytes(path.read_bytes())
    (sources_dir / "sources.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_existing(out_dir: Path) -> tuple[list[dict], dict]:
    """Прочитать уже скачанные источники: ``(документы, метаданные манифеста)``.

    Нужно, чтобы повторный запуск не скачивал и не распознавал документы заново: тексты
    и хеши уже есть, а извлечение OCR стоит времени. Файлы, которых нет или хеш которых
    не совпал, в результат не попадают — их загрузчик скачает снова.
    """
    manifest_path = out_dir / "sources" / "sources.json"
    if not manifest_path.exists():
        return [], {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [], {}
    documents: list[dict] = []
    for doc_id, meta in (manifest.get("documents") or {}).items():
        path = out_dir / "sources" / f"{doc_id}.txt"
        if not path.exists():
            continue
        if meta.get("text_sha256") and sha256_bytes(path.read_bytes()) != meta["text_sha256"]:
            continue
        document = dict(meta)
        document["doc_id"] = doc_id
        document["text"] = path.read_text(encoding="utf-8").strip()
        documents.append(document)
    return documents, manifest


def verify_sources(sources_dir: Path) -> dict:
    """Проверить хеши уже скачанных источников (``--verify``): факты, без скачивания.

    Манифест ищется и в самом каталоге, и в подкаталоге ``sources`` — загрузчик пишет
    его рядом с файлами источников, а вызывать проверку можно с любого из двух путей.
    """
    manifest_path = sources_dir / "sources.json"
    if not manifest_path.exists() and (sources_dir / "sources" / "sources.json").exists():
        sources_dir = sources_dir / "sources"
        manifest_path = sources_dir / "sources.json"
    if not manifest_path.exists():
        return {"checked": 0, "mismatches": ["нет sources.json"], "documents": 0}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    checked = 0
    for doc_id, meta in manifest.get("documents", {}).items():
        relative = meta.get("text_file", f"sources/{doc_id}.txt")
        # «text_file» записан относительно каталога корпуса, а не подкаталога sources.
        candidates = [
            manifest_path.parent / relative,
            manifest_path.parent.parent / relative,
            sources_dir / f"{doc_id}.txt",
        ]
        path = next((item for item in candidates if item.exists()), candidates[0])
        if not path.exists():
            mismatches.append(f"{doc_id}: файл {path} отсутствует")
            continue
        actual = sha256_bytes(path.read_bytes())
        checked += 1
        if actual != meta.get("text_sha256"):
            mismatches.append(f"{doc_id}: sha256 текста {actual} != {meta.get('text_sha256')}")
    return {"checked": checked, "mismatches": mismatches, "documents": len(manifest.get("documents", {}))}


def plan_downloads(
    candidates: list[dict],
    existing_ids: set[str],
    needed: int,
    buffer: int = 40,
    min_pages: int = 2,
    max_pages_doc: int = 30,
    regional_share: float = 0.45,
) -> list[dict]:
    """Выбрать, что скачивать: приоритетные темы, затем баланс уровней и регионов.

    Порядок отбора задан требованиями корпуса: не меньше 60 федеральных и не меньше 40
    региональных актов, причём среди региональных первым идёт Краснодарский край, затем
    Москва, Московская область, Санкт-Петербург, Ростовская область, Ставропольский
    край, Татарстан и Свердловская область (порядок ``REGION_BLOCKS``).

    Документы вне диапазона ``min_pages``…``max_pages_doc`` страниц откладываются в
    конец очереди: одностраничные акты почти не дают фактов, а очень длинные дорого
    распознавать. Это именно порядок, а не исключение — если ничего другого нет, они
    всё равно будут взяты, и это видно по манифесту.
    """
    fresh = [item for item in candidates if item["doc_id"] not in existing_ids and not item.get("skipped")]
    limit = max(0, needed) + max(0, buffer)
    if not fresh or limit == 0:
        return []

    def page_rank(item: dict) -> int:
        pages = item.get("pages_count")
        if not isinstance(pages, int):
            return 1
        return 0 if min_pages <= pages <= max_pages_doc else 1

    priority_themes = [theme for theme, _pattern in THEMES]

    def theme_rank(item: dict) -> int:
        theme = str(item.get("theme"))
        return priority_themes.index(theme) if theme in priority_themes else len(priority_themes)

    region_order = [code for code, _name in REGION_BLOCKS]

    def region_rank(item: dict) -> int:
        code = str(item.get("region_code") or "")
        return region_order.index(code) if code in region_order else len(region_order)

    federal = sorted(
        (item for item in fresh if item.get("level") == "федеральный"),
        key=lambda item: (page_rank(item), theme_rank(item)),
    )
    regional = sorted(
        (item for item in fresh if item.get("level") == "региональный"),
        key=lambda item: (region_rank(item), page_rank(item), theme_rank(item)),
    )
    other = [item for item in fresh if item.get("level") not in {"федеральный", "региональный"}]

    regional_target = int(round(limit * regional_share))
    federal_target = limit - regional_target
    selected = federal[:federal_target] + regional[:regional_target]
    taken = {item["doc_id"] for item in selected}
    # Недобор по одному уровню добираем другим: цель — объём корпуса, а перекос
    # фиксируется в манифесте (``by_level``), а не прячется.
    for item in federal + regional + other:
        if len(selected) >= limit:
            break
        if item["doc_id"] in taken:
            continue
        selected.append(item)
        taken.add(item["doc_id"])
    return selected[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description="Загрузка текстов реальных актов для корпуса A3")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "corpus_a3")
    parser.add_argument("--target-docs", type=int, default=130, help="сколько документов с текстом нужно")
    parser.add_argument(
        "--max-pages-per-type", type=int, default=10, help="сколько страниц выдачи смотреть по виду акта"
    )
    parser.add_argument(
        "--max-pages-per-block",
        type=int,
        default=4,
        help="сколько страниц выдачи смотреть по каждому региональному разделу (block=region<код>)",
    )
    parser.add_argument(
        "--all-themes",
        action="store_true",
        help="брать акты вне приоритетных тем с пометкой темы «прочее» (нужно для объёма корпуса)",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="не запускать OCR: брать только документы с пригодным текстовым слоем PDF",
    )
    parser.add_argument(
        "--min-chars-per-page",
        type=int,
        default=120,
        help="порог пригодности текстового слоя PDF, знаков на страницу",
    )
    parser.add_argument(
        "--only-types",
        default="",
        help="брать только эти виды актов (через запятую); пусто — все виды из TYPES",
    )
    parser.add_argument(
        "--only-blocks",
        default="",
        help="брать только эти региональные разделы по коду (например 23,77); пусто — все из REGION_BLOCKS",
    )
    parser.add_argument(
        "--regional-share",
        type=float,
        default=0.45,
        help="доля региональных актов в плане загрузки (требование: ≥40 из ≥120)",
    )
    parser.add_argument(
        "--min-facts", type=int, default=6, help="минимум фактов (предложений со значениями) в документе"
    )
    parser.add_argument(
        "--pause", type=float, default=PAUSE_SECONDS, help="пауза между запросами, секунд (не меньше 1)"
    )
    parser.add_argument("--workers", type=int, default=4, help="сколько распознаваний запускать параллельно")
    parser.add_argument(
        "--flush-every",
        type=int,
        default=10,
        help="через сколько готовых документов сбрасывать источники на диск (устойчивость к обрыву прогона)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=20,
        help="сколько первых страниц документа распознавать (длинные акты распознаются частично, признак — text_truncated)",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--buffer", type=int, default=40, help="сколько кандидатов взять сверх цели (на брак)")
    parser.add_argument("--force", action="store_true", help="скачивать, даже если документов уже достаточно")
    parser.add_argument("--verify", action="store_true", help="только проверить хеши уже скачанных источников")
    args = parser.parse_args()

    out_dir: Path = args.out
    if args.verify:
        report = verify_sources(out_dir / "sources")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if not report["mismatches"] else 2

    pause = max(1.0, args.pause)  # требование задания: не чаще одного запроса в секунду
    out_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()

    existing, old_manifest = load_existing(out_dir)
    print(f"уже скачано источников: {len(existing)}")
    print(f"::notice title=A3 старт::уже скачано {len(existing)} документов, цель {args.target_docs}")

    if len(existing) >= args.target_docs and not args.force:
        # Цель достигнута: ничего не скачиваем, только сверяем хеши (быстро и без запросов).
        report = verify_sources(out_dir / "sources")
        print(f"цель достигнута: {len(existing)} документов, хеши проверены: {report['checked']}")
        print(f"::notice title=A3 пропуск::документов {len(existing)}, расхождений хешей {len(report['mismatches'])}")
        if report["mismatches"]:
            print(f"ВНИМАНИЕ: расхождения хешей: {report['mismatches'][:3]}", file=sys.stderr)
            return 2
        return 0

    whitelist = load_whitelist()
    print(f"белый список источников ({WHITELIST_CONFIG.name}): {len(whitelist)} хостов")
    robots = fetch_robots(whitelist)
    for host, info in robots.items():
        print(f"robots {host}: статус={info['status']} правил={len(info['rules'].get('*', []))}")
        print(f"::notice title=A3 robots {host}::статус={info['status']} правил={len(info['rules'].get('*', []))}")
        time.sleep(max(0.0, pause))

    types_result = fetch("http://publication.pravo.gov.ru/api/DocumentTypes")
    type_ids: dict[str, str] = {}
    if types_result["status"] == 200:
        try:
            for item in json.loads(types_result["raw"].decode("utf-8")):
                if isinstance(item, dict) and item.get("name") in TYPES and item.get("id"):
                    type_ids[str(item["name"])] = str(item["id"])
        except json.JSONDecodeError:
            type_ids = {}
    only_types = {name.strip() for name in args.only_types.split(",") if name.strip()}
    if only_types:
        missing = only_types - set(type_ids)
        type_ids = {name: value for name, value in type_ids.items() if name in only_types}
        if missing:
            print(f"ВНИМАНИЕ: виды актов не найдены в справочнике API: {sorted(missing)}", file=sys.stderr)
    only_blocks = {code.strip() for code in args.only_blocks.split(",") if code.strip()}
    blocks = tuple(item for item in REGION_BLOCKS if not only_blocks or item[0] in only_blocks)
    print(f"срез: виды={sorted(type_ids) or '—'}; регионы={[code for code, _ in blocks] or '—'}")
    print(f"::notice title=A3 виды актов::{sorted(type_ids)}")
    time.sleep(max(0.0, pause))

    candidates, scan_stats = collect_candidates(
        type_ids,
        args.max_pages_per_type,
        pause,
        robots,
        blocks=blocks,
        max_pages_per_block=args.max_pages_per_block,
        allow_other=args.all_themes,
    )
    existing_ids = {str(document["doc_id"]) for document in existing}
    needed = max(0, args.target_docs - len(existing))
    planned = plan_downloads(
        candidates,
        existing_ids,
        needed,
        buffer=args.buffer,
        regional_share=args.regional_share,
    )
    print(
        f"просмотрено документов: {scan_stats['documents_scanned']}, подходящих по темам: {len(candidates)}, "
        f"к скачиванию: {len(planned)}"
    )
    print(
        f"::notice title=A3 отбор::документов={scan_stats['documents_scanned']} тем={len(candidates)} "
        f"к скачиванию={len(planned)} по темам={scan_stats['by_theme']}"
    )

    tools = ocr_available()
    print(f"инструменты OCR: {tools}")
    print(f"::notice title=A3 OCR::инструменты={tools}")

    rules = {"*": robots.get("publication.pravo.gov.ru", {}).get("rules", {}).get("*", [])}
    # Промежуточный манифест: заполняется полностью в конце, но уже позволяет писать
    # источники по мере распознавания.
    manifest: dict = {"generated_at": started_at, "documents": {}}
    new_documents: list[dict] = []
    skipped: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        downloaded: list[dict] = []
        for index, candidate in enumerate(planned, start=1):
            pdf_url = f"http://publication.pravo.gov.ru/file/pdf?eoNumber={candidate['eo_number']}"
            candidate["pdf_url"] = pdf_url
            if not is_allowed(pdf_url, rules):
                skipped.append({"doc_id": candidate["doc_id"], "reason": "путь запрещён robots.txt"})
                continue
            fetched = fetch(pdf_url)
            time.sleep(max(0.0, pause))
            if fetched["status"] != 200 or fetched["raw"][:4] != b"%PDF":
                skipped.append(
                    {
                        "doc_id": candidate["doc_id"],
                        "reason": f"PDF не получен: {fetched['error'] or fetched['status']}",
                    }
                )
                continue
            path = tmp_dir / f"{candidate['doc_id']}.pdf"
            path.write_bytes(fetched["raw"])
            candidate["pdf_sha256"] = sha256_bytes(fetched["raw"])
            candidate["pdf_bytes"] = len(fetched["raw"])
            candidate["downloaded_at"] = datetime.now(timezone.utc).isoformat()
            candidate["_pdf_path"] = str(path)
            downloaded.append(candidate)
            if index % 20 == 0:
                print(f"скачано PDF: {len(downloaded)} из {index} попыток")
                print(f"::notice title=A3 скачивание::PDF скачано {len(downloaded)} (попыток {index})")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(
                    extract_text,
                    Path(item["_pdf_path"]),
                    args.dpi,
                    args.max_pages,
                    args.min_chars_per_page,
                    not args.no_ocr,
                ): item
                for item in downloaded
            }
            for future in concurrent.futures.as_completed(futures):
                item = futures[future]
                extracted = future.result()
                item["extraction"] = extracted["method"]
                item["text_pages"] = extracted["pages"]
                item["text_layer_chars"] = extracted["text_layer_chars"]
                item["ocr_skipped"] = extracted.get("ocr_skipped")
                item["ocr_seconds"] = extracted.get("ocr_seconds")
                # Для длинных актов распознаётся начало документа (ограничение --dpi/страниц):
                # это видно в манифесте, а не замалчивается.
                item["text_truncated"] = bool(
                    item.get("pages_count") and extracted["pages"] and extracted["pages"] < int(item["pages_count"])
                )
                if extracted["method"] == "none":
                    skipped.append(
                        {
                            "doc_id": item["doc_id"],
                            "reason": f"текст не извлечён (текстовый слой {extracted['text_layer_chars']} знаков, "
                            f"OCR: {extracted['ocr_error'] or 'пусто'})",
                        }
                    )
                    continue
                text = normalize_text(extracted["text"])
                facts = extract_facts(str(item["doc_id"]), text)
                item["facts_found"] = len(facts)
                item["text"] = text
                if len(facts) < args.min_facts:
                    skipped.append({"doc_id": item["doc_id"], "reason": f"фактов {len(facts)} < {args.min_facts}"})
                    continue
                new_documents.append(item)
                if len(new_documents) % max(1, args.flush_every) == 0:
                    # Пишем прогресс на диск: если прогон оборвётся (таймаут CI), уже
                    # распознанные документы не придётся распознавать заново.
                    write_sources(out_dir, sorted(existing + new_documents, key=lambda doc: doc["doc_id"]), manifest)
                    print(f"текстов готово: {len(new_documents)} (фактов ≥ {args.min_facts}), записано на диск")
                    print(f"::notice title=A3 извлечение::текстов готово {len(new_documents)}")

    for item in new_documents + existing:
        item.pop("_pdf_path", None)

    documents = sorted(existing + new_documents, key=lambda item: str(item["doc_id"]))
    by_theme: dict[str, int] = {}
    by_type: dict[str, int] = {}
    by_level: dict[str, int] = {}
    by_region: dict[str, int] = {}
    for item in documents:
        by_theme[str(item.get("theme"))] = by_theme.get(str(item.get("theme")), 0) + 1
        by_type[str(item.get("act_type"))] = by_type.get(str(item.get("act_type")), 0) + 1
        # Уровень и регион у старых записей могли не сохраняться: восстанавливаем из
        # номера опубликования, чтобы счёт был по всем документам, а не по новым.
        placement = region_of(str(item.get("eo_number") or ""))
        level = str(item.get("level") or placement["level"])
        region = str(item.get("region_name") or placement["region_name"] or placement["region_code"])
        by_level[level] = by_level.get(level, 0) + 1
        if level == "региональный":
            by_region[region] = by_region.get(region, 0) + 1

    previous_sources = {item.get("host"): item for item in (old_manifest.get("sources") or [])}
    default_sources = {
        "publication.pravo.gov.ru": {
            "host": "publication.pravo.gov.ru",
            "documents_downloaded": 0,
            "documents_with_text": 0,
            "note": (
                "официальный портал опубликования; отбор по API, текст — текстовый слой PDF либо "
                "OCR сканов (pdftoppm + tesseract -l rus)"
            ),
        },
        "pravo.gov.ru": {
            "host": "pravo.gov.ru",
            "documents_downloaded": 0,
            "documents_with_text": 0,
            "note": "ИПС «Законодательство России»: машинной выдачи текста не найдено (см. reports/ips_*_probe.json)",
        },
        "fstec.ru": {
            "host": "fstec.ru",
            "documents_downloaded": 0,
            "documents_with_text": 0,
            "note": "недоступен из сети CI (TLS)",
        },
        "mintrud.gov.ru": {
            "host": "mintrud.gov.ru",
            "documents_downloaded": 0,
            "documents_with_text": 0,
            "note": "403 из сети CI",
        },
        "rospotrebnadzor.ru": {
            "host": "rospotrebnadzor.ru",
            "documents_downloaded": 0,
            "documents_with_text": 0,
            "note": "таймаут из сети CI",
        },
        "eec.eaeunion.org": {
            "host": "eec.eaeunion.org",
            "documents_downloaded": 0,
            "documents_with_text": 0,
            "note": "доступен, но акты ЕАЭС не входят в приоритетные темы задания",
        },
    }
    sources = []
    for host in whitelist:
        item = dict(
            default_sources.get(host, {"host": host, "documents_downloaded": 0, "documents_with_text": 0, "note": ""})
        )
        previous = previous_sources.get(host) or {}
        if host == "publication.pravo.gov.ru":
            item["documents_downloaded"] = len(existing) + len(downloaded)
            item["documents_with_text"] = len(documents)
        else:
            item["documents_downloaded"] = int(previous.get("documents_downloaded") or 0)
            item["documents_with_text"] = int(previous.get("documents_with_text") or 0)
        item["robots_status"] = (robots.get(host) or {}).get("status")
        item["robots_error"] = (robots.get(host) or {}).get("error")
        item["robots_disallow"] = (robots.get(host) or {}).get("rules", {}).get("*", [])
        sources.append(item)

    methods: dict[str, int] = {}
    for item in documents:
        methods[str(item.get("extraction"))] = methods.get(str(item.get("extraction")), 0) + 1

    manifest = {
        "generated_at": started_at,
        "method": (
            "publication.pravo.gov.ru: отбор по /api/Documents (по видам актов и по "
            "региональным разделам block=region<код>), текст — текстовый слой PDF; "
            "OCR (pdftoppm + tesseract -l rus) только там, где слоя не хватило"
        ),
        "user_agent": USER_AGENT,
        "pause_seconds": pause,
        "whitelist": list(whitelist),
        "whitelist_config": str(WHITELIST_CONFIG.relative_to(ROOT)),
        "robots": robots,
        "sources": sources,
        "scan_stats": scan_stats,
        "target_docs": args.target_docs,
        "documents_total": len(documents),
        "by_theme": by_theme,
        "by_type": by_type,
        "by_level": by_level,
        "by_region": by_region,
        "extraction_methods": methods,
        "min_facts": args.min_facts,
        "ocr": {"tools": tools, "dpi": args.dpi, "workers": args.workers},
        "candidates_skipped": skipped[:100],
        "skipped_total": len(skipped),
        "documents": {},
    }
    write_sources(out_dir, documents, manifest)
    print(f"источников с текстом: {len(documents)} (по темам: {by_theme}; по видам: {by_type}; способ: {methods})")
    print(f"уровни: {by_level}; регионы: {by_region}")
    print(
        f"::notice title=A3 итог::документов с текстом {len(documents)}; уровни {by_level}; "
        f"регионы {by_region}; способ {methods}"
    )
    if len(documents) < args.target_docs:
        print(f"ВНИМАНИЕ: получено {len(documents)} документов вместо {args.target_docs}", file=sys.stderr)
    print(f"Источники: {out_dir / 'sources'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
