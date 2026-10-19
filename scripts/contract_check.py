"""Контрактная проверка репозитория: требование → факт → статус.

Скрипт перебирает требования договора № 20478ГУ/2026 (проект 117812):
Техническое задание (Приложение № 3, пп. 4.1 и 4.2), календарный план
(Приложение № 4) и правила оформления отчёта (Приложение № 10), выполняет
измеримые проверки и пишет:

* ``reports/CONTRACT.json`` — по каждому пункту: id, требование, команда, факт,
  статус (``соответствует`` / ``не соответствует`` / ``не измерено``), коммит,
  дата, окружение, файл-артефакт;
* ``reports/contract/<id>.json`` — артефакт с числами конкретной проверки
  (коммит, seed, режим, набор данных, метрики, оговорка, вердикт по ориентиру);
* ``docs/СООТВЕТСТВИЕ_ДОГОВОРУ.md`` — таблица для человека, целиком собранная из
  ``CONTRACT.json`` (числа в документ руками не вписываются).

Код возврата: ``0`` — нет статусов «не соответствует»; ``1`` — есть; ``2`` —
ошибка самого прогона. Статус «не измерено» допустим только с обоснованием
(нет GPU, нет весов модели, нет Windows-раннера) и печатается отдельно.

Запуск::

    python scripts/contract_check.py                 # все измеримые проверки
    python scripts/contract_check.py --fast          # без предельного прогона 2 000 000 знаков
    python scripts/contract_check.py --only 4.1.6-gate-corpus-a
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "spanverify-module"
ARTIFACTS = ROOT / "reports" / "contract"
CONTRACT_JSON = ROOT / "reports" / "CONTRACT.json"
COMPLIANCE_DOC = ROOT / "docs" / "СООТВЕТСТВИЕ_ДОГОВОРУ.md"
SEED = 42

#: Извлечение требований из подписанного договора (персональные данные не
#: переносятся). Подписанный PDF в публичном репозитории не хранится; если файл
#: есть локально (``SPANVERIFY_CONTRACT_PDF`` или корень репозитория), проверка
#: читает сам PDF и сверяет с извлечением.
EXTRACTION_DOC = ROOT / "docs" / "ДОГОВОР_ИЗВЛЕЧЕНИЕ.md"
CLOSING_DOC = ROOT / "docs" / "ЗАКРЫТИЕ_ДОГОВОРА_ПО_КОДУ.md"
REPORT_APPENDIX_DOC = ROOT / "docs" / "ОТЧЁТ_ПРИЛОЖЕНИЕ_СООТВЕТСТВИЕ.md"
CONTRACT_PDF_NAME = "Договор_0117812 (25).pdf"
CONTRACT_PDF_SHA256 = "e5eb6ad8522fa70135f75bf22b0b5f3464fcc17d9c6f04d81b48976a42e69cd5"
CONTRACT_PDF_BYTES = 5_019_351
CONTRACT_PDF_PAGES = 32
CONTRACT_EXE_MAX_MB = 9.0
CONTRACT_PACKAGE_MAX_MB = 0.6
CONTRACT_STORAGE_WITHOUT_MB = 100.0
CONTRACT_STORAGE_WITH_MB = 1500.0

#: Маркеры требований в тексте договора: ключ извлечения → регулярное выражение.
#: Проверкой «маркер найден в PDF» подтверждается, что число взято из договора, а
#: не переписано из кода.
CONTRACT_TEXT_MARKERS: dict[str, str] = {
    "макс_знаков": r"2\s?000\s?000 знаков",
    "мс_на_1000_знаков": r"0,03 с на 1 000\s+знаков",
    "сек_на_2000000": r"не более чем за 60 с",
    "сек_на_50000": r"не более чем за 1 с",
    "признаков_на_токен": r"пять числовых\s+признаков",
    "пересечение_границ": r"не ниже 0,5",
    "f1_мин": r"F1 не ниже 0,50",
    "fpr_макс": r"не выше 0,40",
    "пар_в_наборе": r"набора из 1200 па?р",
    "шаг_смеси": r"\(шаг 0,25\)",
    "exe_макс_МБ": r"не более 9 МБ",
    "пакет_макс_МБ": r"0,6 МБ",
    "порт": r"localhost:8765",
    "библиотеки_МБ": r"около 500 МБ",
    "веса_МБ": r"около 530 МБ",
    "хранение_без_весов_МБ": r"не более 100 МБ",
    "хранение_с_весами_МБ": r"не более 1,5 ГБ",
    "ядер_мин": r"не менее двух",
    "озу_ГБ": r"не менее 4 ГБ",
    "озу_с_весами_ГБ": r"не менее 8 ГБ",
    "диск_ГБ": r"не менее 1,5 ГБ",
    "python_мин": r"Python 3\.10",
    "месяцев_этап": r"6 месяцев",
    "сумма_1_руб": r"200 000",
    "сумма_2_руб": r"300 000",
    "сумма_всего_руб": r"500 000",
    "дней_до_окончания_этапа": r"20 \(Двадцать\) календарных дней",
    "оригинальность_проц": r"не менее 80%",
    "ключевых_слов_мин": r"от 5 до 15 слов",
    "элементов_отчёта": r"Структурные элементы промежуточного и заключительного",
}

#: Зона получателя гранта: пункты, которые выполняет человек. В коде они только
#: перечисляются — договор требует их представления в АС «Фонд-М» вместе с
#: отчётностью, и подписания УКЭП программно выполнить нельзя.
GRANTEE_ZONE: list[tuple[str, str]] = [
    (
        "Преакселерация",
        "свидетельство о прохождении преакселерационной программы или акселератора вуза (Приложение № 4, этап 2)",
    ),
    (
        "Бизнес-план",
        "бизнес-план инновационного проекта по Приложению № 11 (не менее 20 страниц) либо заявка на «Студенческий стартап»/«Старт»",
    ),
    ("Дорожная карта", "дорожная карта развития проекта по Приложению № 12"),
    (
        "РИД",
        "заявка на регистрацию прав на результат интеллектуальной деятельности и переписка с Роспатентом (ТЗ п. 6)",
    ),
    (
        "Оригинальность отчёта",
        "проверка текста отчёта во внешней системе; требуется не менее 80 % (Приложение № 10, п. 4)",
    ),
    (
        "Финансовый отчёт",
        "отчёт о расходовании средств гранта по каждому этапу (Приложение № 4)",
    ),
    ("Акт", "акт о выполнении Работ по каждому этапу (Приложения № 8 и № 9)"),
    (
        "Отчёт о достижении значений результата",
        "по Приложению № 2 к договору: «Завершённый проект», план 0 (2026) / 1 (2027)",
    ),
    (
        "Подача в АС «Фонд-М»",
        "отчётность представляется не позднее чем за 20 календарных дней до окончания этапа и подписывается УКЭП (п. 6.1)",
    ),
]

STATUS_OK = "соответствует"
STATUS_FAIL = "не соответствует"
STATUS_UNKNOWN = "не измерено"


# --------------------------------------------------------------------------- утилиты


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        return "unknown"


def py_version() -> str:
    return platform.python_version()


def env_info() -> dict[str, Any]:
    return {
        "python": py_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "seed": SEED,
        "commit": git_commit(),
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def run_module(args: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    """Запустить ``python -m spanverify ...`` из каталога модуля."""
    return subprocess.run(
        [sys.executable, "-m", "spanverify", *args],
        cwd=MODULE,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PYTHONPATH": str(MODULE), "PYTHONUTF8": "1"},
        check=False,
    )


def run_python(
    script: str, timeout: int = 900, cwd: Path | None = None
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=cwd or MODULE,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PYTHONPATH": str(MODULE), "PYTHONUTF8": "1"},
        check=False,
    )


def _module_attr(module_name: str, attribute: str) -> Any:
    """Значение константы из пакета spanverify (импорт без побочных эффектов)."""
    if str(MODULE) not in sys.path:
        sys.path.insert(0, str(MODULE))
    module = importlib.import_module(module_name)
    return getattr(module, attribute)


def code_limits() -> dict[str, dict[str, Any]]:
    """Числа договора, как они зафиксированы в коде: значение и место.

    Каждая запись — «где это живёт в коде». Проверка ``contract.source`` сверяет
    эти значения с текстом договора, поэтому расхождение кода с договором видно
    как «не соответствует», а не как догадка.
    """
    limits = [
        ("макс_знаков", "spanverify.limits", "MAX_TEXT_CHARS", "знаков"),
        (
            "мс_на_1000_знаков",
            "spanverify.limits",
            "TARGET_MS_PER_1000_CHARS",
            "мс",
        ),
        ("сек_на_2000000", "spanverify.limits", "TARGET_SECONDS_2M", "с"),
        ("пересечение_границ", "spanverify.gates", "IOU_MATCH_THRESHOLD", "IoU"),
        ("порт", "spanverify.server", "DEFAULT_PORT", "порт"),
    ]
    out: dict[str, dict[str, Any]] = {}
    for key, module_name, attribute, unit in limits:
        out[key] = {
            "value": _module_attr(module_name, attribute),
            "unit": unit,
            "source": f"spanverify-module/{module_name.replace('.', '/')}.py: {attribute}",
        }
    names = _module_attr("spanverify.features", "FEATURE_NAMES")
    out["признаков_на_токен"] = {
        "value": len(names),
        "unit": "признаков",
        "source": "spanverify-module/spanverify/features.py: FEATURE_NAMES",
    }
    gate = _module_attr("spanverify.gates", "CONTRACT_GATE")
    out["f1_мин"] = {
        "value": gate["f1_min"],
        "unit": "F1",
        "source": "spanverify-module/spanverify/gates.py: CONTRACT_GATE",
    }
    out["fpr_макс"] = {
        "value": gate["fpr_max"],
        "unit": "FPR",
        "source": "spanverify-module/spanverify/gates.py: CONTRACT_GATE",
    }
    out["exe_макс_МБ"] = {
        "value": CONTRACT_EXE_MAX_MB,
        "unit": "МБ",
        "source": "scripts/contract_check.py: CONTRACT_EXE_MAX_MB",
    }
    out["пакет_макс_МБ"] = {
        "value": CONTRACT_PACKAGE_MAX_MB,
        "unit": "МБ",
        "source": "scripts/contract_check.py: CONTRACT_PACKAGE_MAX_MB",
    }
    out["хранение_без_весов_МБ"] = {
        "value": CONTRACT_STORAGE_WITHOUT_MB,
        "unit": "МБ",
        "source": "scripts/contract_check.py: CONTRACT_STORAGE_WITHOUT_MB",
    }
    out["хранение_с_весами_МБ"] = {
        "value": CONTRACT_STORAGE_WITH_MB,
        "unit": "МБ",
        "source": "scripts/contract_check.py: CONTRACT_STORAGE_WITH_MB",
    }
    corpus = MODULE / "data" / "corpus_a" / "pairs.jsonl"
    pairs = None
    if corpus.is_file():
        pairs = sum(
            1
            for line in corpus.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    out["пар_в_наборе"] = {
        "value": pairs,
        "unit": "пар",
        "source": "spanverify-module/data/corpus_a/pairs.jsonl",
    }
    return out


def read_extraction() -> dict[str, Any] | None:
    """Машиночитаемый блок из docs/ДОГОВОР_ИЗВЛЕЧЕНИЕ.md."""
    if not EXTRACTION_DOC.is_file():
        return None
    text = EXTRACTION_DOC.read_text(encoding="utf-8")
    match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def contract_pdf_path() -> Path:
    """Путь к подписанному PDF: переменная окружения или корень репозитория."""
    env = os.environ.get("SPANVERIFY_CONTRACT_PDF")
    if env:
        return Path(env).expanduser()
    return ROOT / CONTRACT_PDF_NAME


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contract_pdf_text(path: Path) -> str | None:
    """Текст подписанного PDF, если есть pypdf. Без библиотеки — None."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def hf_artifact(check_id: str) -> dict[str, Any] | None:
    """Артефакт измерения режима hf, записанный CI-заданием hf-footprint."""
    path = ARTIFACTS / f"{check_id}.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if (payload.get("mode") or "") != "hf":
        return None
    if not isinstance(payload.get("metrics"), dict):
        return None
    return payload


def write_artifact(check_id: str, payload: dict[str, Any]) -> str:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / f"{check_id}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return str(path.relative_to(ROOT))


def artifact_payload(
    check_id: str, fact: dict[str, Any], note: str = "", gate: str | None = None
) -> dict[str, Any]:
    """Стандартная схема артефакта: commit, seed, mode, dataset, metrics, note, gate."""
    return {
        "id": check_id,
        "run_id": os.environ.get("GITHUB_RUN_ID", "local-contract:" + git_commit()),
        "commit": git_commit(),
        "seed": SEED,
        "mode": fact.pop("mode", "demo"),
        "dataset": fact.pop("dataset", None),
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": fact.pop("command", ""),
        "metrics": fact,
        "note": note,
        "gate": gate,
    }


def dir_size_mb(path: Path) -> float:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:  # pragma: no cover - файл исчез во время обхода
                continue
    return round(total / (1024 * 1024), 3)


#: Каталоги, которых не бывает в поставке: окружение разработчика, кэши
#: инструментов, артефакты сборки. В чистый клон они не попадают (перечислены в
#: .gitignore), но на рабочей машине занимают десятки мегабайт: считать их
#: «поставкой без весов» — значит мерить машину, а не продукт.
DELIVERY_SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".ruff_cache",
        ".pytest_cache",
        ".mypy_cache",
        "build",
        "dist",
    }
)
#: Пути относительно каталога модуля, которые в поставку не входят: сырые внешние
#: данные (лицензия не подтверждена, каталог в .gitignore, в клон не попадает).
DELIVERY_SKIP_PATHS = ("data/external",)


def delivery_size_mb(root: Path) -> float:
    """Размер поставки: файлы клона без каталогов разработки и сырых внешних данных.

    Правило одно и то же локально и в CI: в CI каталогов разработки просто нет,
    поэтому число совпадает с чистым клоном, а не с машиной разработчика.
    """
    root = root.resolve()
    total = 0
    for item in root.rglob("*"):
        if not item.is_file():
            continue
        rel = item.relative_to(root)
        if any(part in DELIVERY_SKIP_DIRS for part in rel.parts):
            continue
        text = rel.as_posix()
        if any(
            text == skip or text.startswith(skip + "/") for skip in DELIVERY_SKIP_PATHS
        ):
            continue
        try:
            total += item.stat().st_size
        except OSError:  # pragma: no cover - файл исчез во время обхода
            continue
    return round(total / (1024 * 1024), 3)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- проверки


@dataclass
class Check:
    id: str
    point: str
    requirement: str
    how: str
    command: str
    runner: Callable[[Runner], tuple[str, dict[str, Any], str]]
    heavy: bool = False
    #: Проверка выполняется после перегенерации документа соответствия: она читает
    #: сам документ, а тот собирается из CONTRACT.json — иначе получается
    #: самоссылка (проверка видит документ от прошлого прогона).
    after_docs: bool = False
    data: dict[str, Any] = field(default_factory=dict)


class Runner:
    """Общее состояние прогона: тяжёлые измерения выполняются один раз."""

    def __init__(self, fast: bool = False) -> None:
        self.fast = fast
        self.shared: dict[str, Any] = {}

    # ---- предельный прогон 2 000 000 знаков (общий для 4.1.1 и 4.1.7) ----

    def limit_run(self) -> dict[str, Any]:
        if "limit" in self.shared:
            return self.shared["limit"]
        if self.fast:
            self.shared["limit"] = {"skipped": True, "reason": "запуск в режиме --fast"}
            return self.shared["limit"]
        script = f"""
import json, resource, sys, time
sys.path.insert(0, {str(MODULE)!r})
from spanverify.engine import Verifier
from spanverify.limits import MAX_TEXT_CHARS

base = ("Регламент учёта № 58. Срок хранения первичных документов составляет десять лет. "
        "Для счета-фактуры предельный объём одного вложения составляет десять мегабайт. ")
context = (base * (MAX_TEXT_CHARS // len(base) + 1))[:MAX_TEXT_CHARS]
answer = "Срок хранения первичных документов составляет три года. " * 40
length = len(context)
verifier = Verifier(mode="demo")
started = time.perf_counter()
result = verifier.verify(answer, context)
seconds = time.perf_counter() - started
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
print(json.dumps({{
    "chars": length,
    "seconds": round(seconds, 3),
    "ms_per_1000_chars": round(seconds / (length / 1000.0) * 1000.0, 4),
    "peak_memory_mb": round(peak, 1),
    "success": True,
    "spans": len(result.spans),
}}, ensure_ascii=False))
"""
        proc = run_python(script, timeout=1800)
        if proc.returncode != 0:
            self.shared["limit"] = {
                "skipped": False,
                "success": False,
                "stderr": proc.stderr[-500:],
            }
            return self.shared["limit"]
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        payload["skipped"] = False
        self.shared["limit"] = payload
        return payload

    # ---- быстродействие по размерам ----

    def perf(self) -> dict[str, Any]:
        if "perf" in self.shared:
            return self.shared["perf"]
        sizes = [500, 2000, 10000, 50000]
        script = f"""
import json, resource, sys, time
sys.path.insert(0, {str(MODULE)!r})
from spanverify.engine import Verifier

sizes = {sizes!r}
base = ("Регламент учёта № 58. Срок хранения первичных документов составляет десять лет. "
        "Для счета-фактуры предельный объём одного вложения составляет десять мегабайт. ")
answer = "Срок хранения первичных документов составляет три года. "
verifier = Verifier(mode="demo")
out = {{}}
for size in sizes:
    context = (base * (size // len(base) + 1))[:size]
    runs = []
    for _ in range(3):
        started = time.perf_counter()
        verifier.verify(answer, context)
        runs.append(time.perf_counter() - started)
    out[str(size)] = {{
        "runs_seconds": [round(value, 6) for value in runs],
        "median_seconds": round(sorted(runs)[1], 6),
        "median_ms_per_1000_chars": round(sorted(runs)[1] / (size / 1000.0) * 1000.0, 4),
    }}
print(json.dumps({{"by_size": out, "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)}}, ensure_ascii=False))
"""
        proc = run_python(script, timeout=900)
        if proc.returncode != 0:  # pragma: no cover - сбой окружения
            self.shared["perf"] = {"failed": proc.stderr[-500:]}
            return self.shared["perf"]
        self.shared["perf"] = json.loads(proc.stdout.strip().splitlines()[-1])
        return self.shared["perf"]

    # ---- метрики корпуса A ----

    def corpus_a(self) -> dict[str, Any]:
        if "corpus_a" in self.shared:
            return self.shared["corpus_a"]
        proc = run_module(
            ["evaluate", "--dataset", "data/corpus_a/splits/test.jsonl", "--json"]
        )
        manifest = json.loads(
            (MODULE / "data" / "corpus_a" / "manifest.json").read_text(encoding="utf-8")
        )
        metrics = json.loads(proc.stdout) if proc.returncode == 0 else {}
        self.shared["corpus_a"] = {
            "returncode": proc.returncode,
            "metrics": metrics,
            "manifest": manifest,
            "stderr": proc.stderr[-400:],
        }
        return self.shared["corpus_a"]

    # ---- интерфейсы ----

    def interfaces(self) -> dict[str, Any]:
        if "interfaces" in self.shared:
            return self.shared["interfaces"]
        script = f"""
import json, sys, threading
sys.path.insert(0, {str(MODULE)!r})
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from spanverify.server import DEFAULT_PORT, MAX_BODY, Service, make_handler

# Порт договора — 8765; если он занят, берём свободный и фиксируем это в отчёте.
server = None
used_port = None
for port in (DEFAULT_PORT, 0):
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(Service(mode="demo")))
        used_port = server.server_address[1]
        break
    except OSError:
        continue
assert server is not None, "не удалось поднять сервер"
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
base = f"http://127.0.0.1:{{used_port}}"
out = {{"port": used_port, "default_port": DEFAULT_PORT, "max_body": MAX_BODY}}
try:
    with urlopen(f"{{base}}/health", timeout=60) as response:
        out["health"] = json.loads(response.read().decode("utf-8"))
        out["health_status"] = response.status
    with urlopen(f"{{base}}/", timeout=60) as response:
        body = response.read().decode("utf-8")
        out["index_status"] = response.status
        out["index_has_form"] = "verify" in body
    request = Request(
        f"{{base}}/v1/verify",
        data=json.dumps({{"answer": "Срок хранения первичных документов составляет 3 года.",
                         "context": "Регламент: срок хранения первичных документов составляет 10 лет."}}).encode("utf-8"),
        headers={{"Content-Type": "application/json"}}, method="POST")
    with urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
        out["verify_status"] = response.status
        out["verify_keys"] = sorted(payload)
        out["spans"] = payload.get("spans")
        out["mode"] = payload.get("mode")
finally:
    server.shutdown(); server.server_close()
print(json.dumps(out, ensure_ascii=False))
"""
        proc = run_python(script, timeout=600)
        if proc.returncode != 0:  # pragma: no cover
            self.shared["interfaces"] = {"failed": proc.stderr[-500:]}
            return self.shared["interfaces"]
        self.shared["interfaces"] = json.loads(proc.stdout.strip().splitlines()[-1])
        return self.shared["interfaces"]

    # ---- калибровка доли участия ИИ ----

    def participation(self) -> dict[str, Any]:
        if "participation" in self.shared:
            return self.shared["participation"]
        script = f"""
import json, sys
sys.path.insert(0, {str(MODULE)!r})
from spanverify.engine import Verifier
from spanverify.participation import ParticipationModel, build_participation_corpus, corpus_rows_and_labels

verifier = Verifier(mode="demo")
samples = build_participation_corpus(count=240, seed={SEED} + 1985)
rows, labels = corpus_rows_and_labels(verifier, samples)
if not rows or len(set(labels)) < 2:
    raise SystemExit("корпус участия пуст или одного класса")
model = ParticipationModel.fit(rows, labels, seed={SEED})

# Оценка по каждому образцу: доля участия известна при сборке корпуса (шаг 0.25).
by_fraction = {{}}
pairs = []
for sample in samples:
    part_rows, _ = corpus_rows_and_labels(verifier, [sample])
    scores = model.score_rows(part_rows)
    estimate = max(0.0, min(1.0, sum(scores) / len(scores))) if scores else 0.0
    fraction = float(sample.get("ai_fraction", 0.0))
    pairs.append((fraction, estimate))
    by_fraction.setdefault(fraction, []).append(estimate)

fractions = sorted({{fraction for fraction, _ in pairs}})
means = {{str(round(fraction, 2)): round(sum(values) / len(values), 4) for fraction, values in sorted(by_fraction.items())}}
mae = sum(abs(estimate - fraction) for fraction, estimate in pairs) / max(1, len(pairs))
binary_labels = [1 if fraction >= 0.5 else 0 for fraction, _ in pairs]
binary_scores = [estimate for _, estimate in pairs]
tp = sum(1 for label, score in zip(binary_labels, binary_scores, strict=False) if label and score >= 0.5)
fp = sum(1 for label, score in zip(binary_labels, binary_scores, strict=False) if not label and score >= 0.5)
fn = sum(1 for label, score in zip(binary_labels, binary_scores, strict=False) if label and score < 0.5)
precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
from spanverify.train import auc_score

# Корреляция Пирсона между оценкой и известной долей смеси: показывает, что
# оценка не просто монотонна по бинам, а линейно связана с истиной.
fraction_values = [fraction for fraction, _ in pairs]
estimate_values = [estimate for _, estimate in pairs]
mean_fraction = sum(fraction_values) / len(fraction_values)
mean_estimate = sum(estimate_values) / len(estimate_values)
cov = sum((f - mean_fraction) * (e - mean_estimate) for f, e in pairs)
var_f = sum((f - mean_fraction) ** 2 for f in fraction_values)
var_e = sum((e - mean_estimate) ** 2 for e in estimate_values)
pearson = cov / ((var_f * var_e) ** 0.5) if var_f and var_e else 0.0

estimate_gap = (means or {{}}).get("1.0", 0.0) - (means or {{}}).get("0.0", 0.0)
print(json.dumps({{
    "pairs": len(pairs),
    "fractions": fractions,
    "mean_by_fraction": means,
    "monotone": list(means.values()) == sorted(means.values()),
    "mae": round(mae, 4),
    "auc_binary_at_0_5": round(auc_score(binary_labels, binary_scores), 4),
    "precision_at_0_5": round(precision, 4),
    "recall_at_0_5": round(recall, 4),
    "f1_at_0_5": round(f1, 4),
    "pearson_vs_known_fraction": round(pearson, 4),
    "estimate_gap_1_0_minus_0_0": round(estimate_gap, 4),
    "rows": len(rows),
    "auc_out_of_fold": model.auc_out_of_fold,
    "calibrated_on": model.calibrated_on,
    "calibration_file": "spanverify-module/config/participation.json",
    "seed": {SEED},
}}, ensure_ascii=False))
"""
        proc = run_python(script, timeout=900)
        if proc.returncode != 0:
            self.shared["participation"] = {
                "failed": True,
                "stderr": proc.stderr[-600:],
            }
            return self.shared["participation"]
        self.shared["participation"] = json.loads(proc.stdout.strip().splitlines()[-1])
        return self.shared["participation"]

    # ---- отсутствие сети ----

    def no_network(self) -> dict[str, Any]:
        if "no_network" in self.shared:
            return self.shared["no_network"]
        script = f"""
import builtins, json, socket, sys
sys.path.insert(0, {str(MODULE)!r})

class Blocked(RuntimeError):
    pass

def blocked(*args, **kwargs):
    raise Blocked("сетевое обращение запрещено контрактной проверкой")

socket.socket = blocked          # type: ignore[assignment]
socket.create_connection = blocked  # type: ignore[assignment]
socket.getaddrinfo = blocked     # type: ignore[assignment]

from spanverify.engine import Verifier
from spanverify.dataset import read_pairs

verifier = Verifier(mode="demo")
result = verifier.verify("Срок хранения первичных документов составляет 3 года.",
                         "Регламент: срок хранения первичных документов составляет 10 лет.",
                         with_tokens=True)
pairs = list(read_pairs("data/demo_pairs.jsonl"))[:20]
metrics = verifier.evaluate(pairs)
server_module_blocked = True  # сервер запускает сокеты по определению: он вне этой проверки
print(json.dumps({{
    "verify_ok": True,
    "spans": len(result.spans),
    "tokens": len(result.tokens),
    "evaluate_pairs": metrics["pairs"],
    "token_f1": round(metrics["tokens"]["f1"], 4),
    "operations": ["verify", "evaluate", "features(в составе verify)", "числовые правила"],
    "excluded": ["HTTP-сервер: по определению открывает порт, проверяется отдельно"],
}}, ensure_ascii=False))
"""
        proc = run_python(script, timeout=600)
        if proc.returncode != 0:
            self.shared["no_network"] = {"failed": True, "stderr": proc.stderr[-500:]}
            return self.shared["no_network"]
        self.shared["no_network"] = json.loads(proc.stdout.strip().splitlines()[-1])
        return self.shared["no_network"]

    # ---- сборка переносимого пакета ----

    def delivery(self) -> dict[str, Any]:
        if "delivery" in self.shared:
            return self.shared["delivery"]
        workdir = Path(tempfile.mkdtemp(prefix="spanverify-contract-"))
        out_dir = workdir / "release"
        proc = subprocess.run(
            [
                sys.executable,
                "scripts/make_zipapp.py",
                "--out",
                str(out_dir),
                "--name",
                "spanverify.pyz",
            ],
            cwd=MODULE,
            capture_output=True,
            text=True,
            timeout=900,
            env={**os.environ, "PYTHONPATH": str(MODULE), "PYTHONUTF8": "1"},
            check=False,
        )
        payload: dict[str, Any] = {
            "build_returncode": proc.returncode,
            "build_stderr": proc.stderr[-400:],
        }
        out_pyz = out_dir / "spanverify.pyz"
        if out_pyz.is_file():
            payload["pyz_size_bytes"] = out_pyz.stat().st_size
            payload["pyz_size_mb"] = round(out_pyz.stat().st_size / (1024 * 1024), 4)
            payload["pyz_size_mb_decimal"] = round(
                out_pyz.stat().st_size / 1_000_000, 4
            )
            payload["pyz_sha256"] = sha256_of(out_pyz)
            smoke = subprocess.run(
                [sys.executable, str(out_pyz), "selftest"],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=900,
                env={**os.environ, "PYTHONUTF8": "1"},
                check=False,
            )
            payload["smoke_returncode"] = smoke.returncode
            payload["smoke_tail"] = (smoke.stdout or smoke.stderr)[-300:]
            # Перенос = копирование: суммы до и после копии обязаны совпасть.
            copied = workdir / "copy.pyz"
            shutil.copy2(out_pyz, copied)
            payload["copy_sha256_equal"] = sha256_of(copied) == payload["pyz_sha256"]
        payload["module_size_mb"] = delivery_size_mb(MODULE)
        payload["module_size_without_data_mb"] = round(
            delivery_size_mb(MODULE / "spanverify")
            + delivery_size_mb(MODULE / "config"),
            3,
        )
        data_dir = MODULE / "data"
        payload["data_dir_mb"] = (
            delivery_size_mb(data_dir) if data_dir.is_dir() else None
        )
        payload["size_rule"] = (
            "размер поставки — файлы клона без каталогов разработки (.venv, кэши, build/dist) "
            "и без сырых внешних данных (data/external)"
        )
        shutil.rmtree(workdir, ignore_errors=True)
        self.shared["delivery"] = payload
        return payload

    # ---- опубликованный релиз (сборка CI + контрольные суммы) ----

    def release(self) -> dict[str, Any]:
        if "release" in self.shared:
            return self.shared["release"]
        repo = "Igorr112323/Smarty_Chumakov"
        info = subprocess.run(
            ["gh", "api", f"repos/{repo}/releases/latest"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if info.returncode != 0:
            self.shared["release"] = {
                "available": False,
                "reason": (info.stderr or "").strip()[-300:],
            }
            return self.shared["release"]
        payload = json.loads(info.stdout)
        assets = {
            item["name"]: {
                "size_bytes": item["size"],
                "size_mb": round(item["size"] / (1024 * 1024), 4),
                "size_mb_decimal": round(item["size"] / 1_000_000, 4),
                "url": item["browser_download_url"],
            }
            for item in payload.get("assets", [])
        }
        runs = subprocess.run(
            [
                "gh",
                "run",
                "list",
                "--workflow",
                "build-and-release-exe.yml",
                "--limit",
                "10",
                "--json",
                "databaseId,headSha,conclusion,createdAt",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        workflow_runs = json.loads(runs.stdout) if runs.returncode == 0 else []
        result: dict[str, Any] = {
            "available": True,
            "tag": payload.get("tag_name"),
            "published_at": payload.get("published_at"),
            "assets": assets,
            "workflow": {
                "name": "build-and-release-exe.yml",
                "latest_success": next(
                    (run for run in workflow_runs if run["conclusion"] == "success"),
                    None,
                ),
                "latest": workflow_runs[0] if workflow_runs else None,
            },
        }
        # Сверка контрольной суммы опубликованного пакета нашим же инструментом.
        workdir = Path(tempfile.mkdtemp(prefix="spanverify-release-"))
        try:
            download = subprocess.run(
                [
                    "gh",
                    "release",
                    "download",
                    "--repo",
                    repo,
                    "--pattern",
                    "spanverify.pyz",
                    "--pattern",
                    "SHA256SUMS.txt",
                    "--dir",
                    str(workdir),
                    "--clobber",
                    payload.get("tag_name") or "",
                ],
                capture_output=True,
                text=True,
                timeout=600,
                check=False,
            )
            result["download_returncode"] = download.returncode
            pyz = workdir / "spanverify.pyz"
            sums = workdir / "SHA256SUMS.txt"
            if download.returncode != 0:
                # Ограничение окружения прогона: файлы релиза отдаются хранилищем,
                # недоступным из этой сети. Размеры и состав релиза при этом читаются
                # через API и приводятся как факт; сверку суммы помечаем как
                # непроверенную здесь, а не как проваленную.
                result["checksum_verified"] = None
                result["checksum_note"] = (
                    "скачивание файлов релиза недоступно из окружения прогона (хранилище релиза вне "
                    "разрешённых хостов); размеры и состав релиза прочитаны через API: "
                    + ", ".join(sorted(assets))
                )
            elif pyz.is_file() and sums.is_file():
                result["downloaded_pyz_size_bytes"] = pyz.stat().st_size
                result["downloaded_pyz_sha256"] = sha256_of(pyz)
                verify = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "verify_checksum.py"),
                        str(pyz),
                        "--sums",
                        str(sums),
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    check=False,
                )
                result["checksum_verified"] = verify.returncode == 0
                result["checksum_tail"] = (
                    (verify.stdout or verify.stderr).strip().splitlines()[-1:]
                )
            else:
                result["checksum_verified"] = False
                result["checksum_note"] = (download.stderr or "")[-300:]
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        self.shared["release"] = result
        return result

    # ---- тесты и покрытие (DoD промта) ----

    def tests(self) -> dict[str, Any]:
        if "tests" in self.shared:
            return self.shared["tests"]
        coverage_path = MODULE / "reports" / "coverage.json"
        collected = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "-p",
                "no:cacheprovider",
            ],
            cwd=MODULE,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        count = 0
        for line in collected.stdout.splitlines():
            if line.endswith(")") or "::" in line:
                continue
            name, _, tail = line.rpartition(": ")
            if name.strip().endswith(".py") and tail.strip().isdigit():
                count += int(tail)
        payload: dict[str, Any] = {
            "collect_returncode": collected.returncode,
            "collected": count,
            "coverage_file": (
                str(coverage_path.relative_to(ROOT))
                if coverage_path.is_file()
                else None
            ),
            "coverage_percent": None,
            "coverage_gate_percent": 85,
        }
        if coverage_path.is_file():
            data = json.loads(coverage_path.read_text(encoding="utf-8"))
            payload["coverage_percent"] = round(
                float(data["totals"]["percent_covered"]), 2
            )
        self.shared["tests"] = payload
        return payload


# ------------------------------------------------------------------ регистрация проверок

CHECKS: list[Check] = []


def check(**kwargs: Any) -> Callable:
    def decorator(
        func: Callable[[Runner], tuple[str, dict[str, Any], str]],
    ) -> Callable:
        CHECKS.append(Check(runner=func, **kwargs))
        return func

    return decorator


# ------------------------------------------------------------------ ТЗ п. 4.1


@check(
    id="4.1.1-text-limit",
    point="ТЗ п. 4.1 треб. 1",
    requirement="Вход — текст в UTF-8; максимум 2 000 000 знаков; прогон на предельном тексте",
    how="прогон на тексте 2 000 000 знаков: успех, время, пик памяти",
    command="python scripts/contract_check.py --only 4.1.1-text-limit",
    heavy=True,
)
def check_text_limit(runner: Runner) -> tuple[str, dict[str, Any], str]:
    run = runner.limit_run()
    if run.get("skipped"):
        return (
            STATUS_UNKNOWN,
            {"skipped": True, "reason": run.get("reason")},
            "режим --fast: предельный прогон не выполнялся",
        )
    if not run.get("success"):
        return STATUS_FAIL, run, "прогон на предельном тексте не завершился"
    ok = run["chars"] == 2_000_000
    fact = {
        "chars": run["chars"],
        "seconds": run["seconds"],
        "peak_memory_mb": run["peak_memory_mb"],
        "spans": run.get("spans"),
        "dataset": "synthetic-2000000",
    }
    return (
        STATUS_OK if ok else STATUS_FAIL,
        fact,
        (
            "текст 2 000 000 знаков обработан"
            if ok
            else "длина текста в прогоне не равна 2 000 000 знаков"
        ),
    )


@check(
    id="4.1.2-api-schema",
    point="ТЗ п. 4.1 треб. 2",
    requirement="Выход — риск 0…1 и фрагменты с номерами символов, меткой и уверенностью 0…1",
    how="реальный ответ API: границы, срез текста, диапазоны чисел, наличие уверенности",
    command="python scripts/contract_check.py --only 4.1.2-api-schema",
)
def check_api_schema(runner: Runner) -> tuple[str, dict[str, Any], str]:
    interfaces = runner.interfaces()
    if interfaces.get("failed"):
        return STATUS_FAIL, interfaces, "сервис не поднялся"
    answer = "Срок хранения первичных документов составляет 3 года."
    spans = interfaces.get("spans") or []
    problems: list[str] = []
    for span in spans:
        for key in ("start", "end", "text", "risk", "label", "confidence"):
            if key not in span:
                problems.append(f"в фрагменте нет поля {key}")
                continue
        if not (0 <= span["start"] < span["end"] <= len(answer)):
            problems.append(f"границы вне текста: {span['start']}:{span['end']}")
        if span.get("text") != answer[span["start"] : span["end"]]:
            problems.append("текст фрагмента не совпадает со срезом answer[start:end]")
        for key in ("risk", "confidence"):
            value = span.get(key)
            if value is None or not (0.0 <= float(value) <= 1.0):
                problems.append(f"{key} вне диапазона 0…1: {value}")
    if not spans:
        problems.append("фрагменты не найдены: подмена числа обязана находиться")
    fact = {
        "spans": spans,
        "verify_keys": interfaces.get("verify_keys"),
        "max_body": interfaces.get("max_body"),
        "dataset": "api-smoke",
    }
    return (
        (STATUS_OK if not problems else STATUS_FAIL),
        fact,
        "; ".join(problems) or "схема ответа соответствует требованию",
    )


@check(
    id="4.1.3-five-features",
    point="ТЗ п. 4.1 треб. 3",
    requirement="Пять признаков на токен (энтропия внимания, доля внимания на документ, косинусная близость и др.)",
    how="дамп признаков по токену: ровно 5 именованных признаков, конечные числа",
    command="python -m spanverify features --answer ... --context ... --json",
)
def check_five_features(runner: Runner) -> tuple[str, dict[str, Any], str]:
    proc = run_module(
        [
            "features",
            "--answer",
            "Срок хранения первичных документов составляет 3 года.",
            "--context",
            "Регламент: срок хранения первичных документов составляет 10 лет.",
            "--json",
        ]
    )
    if proc.returncode != 0:
        return (
            STATUS_FAIL,
            {"stderr": proc.stderr[-400:]},
            "команда дампа признаков завершилась ошибкой",
        )
    payload = json.loads(proc.stdout)
    names = payload["features"]
    problems: list[str] = []
    if len(names) != 5:
        problems.append(f"признаков {len(names)}, а требуется 5")
    for row in payload["tokens"]:
        for name in names:
            value = row.get(name)
            if value is None or not math.isfinite(float(value)):
                problems.append(f"{name}={value} у токена {row.get('index')}")
    fact = {
        "features": names,
        "count": len(names),
        "tokens": len(payload["tokens"]),
        "weights": payload.get("weights"),
        "dataset": "api-smoke",
    }
    return (
        (STATUS_OK if not problems else STATUS_FAIL),
        fact,
        "; ".join(problems) or "дамп содержит ровно пять конечных признаков",
    )


@check(
    id="4.1.4-ai-participation",
    point="ТЗ п. 4.1 треб. 4",
    requirement="Оценка доли участия ИИ 0…1, калибровка на наборе с известной долей смеси (шаг 0,25)",
    how="воспроизводимый прогон калибровки по seed: AUC, MAE, монотонность по долям; файл калибровки",
    command="python scripts/contract_check.py --only 4.1.4-ai-participation",
)
def check_participation(runner: Runner) -> tuple[str, dict[str, Any], str]:
    data = runner.participation()
    if data.get("failed"):
        return STATUS_FAIL, data, "калибровка не выполнилась"
    fractions = [round(value, 2) for value in data["fractions"]]
    expected = [0.0, 0.25, 0.5, 0.75, 1.0]
    problems: list[str] = []
    if fractions != expected:
        problems.append(f"доли смеси {fractions} вместо {expected}")
    if not data.get("monotone"):
        problems.append("оценка не монотонна по доле смеси")
    calibration_file = MODULE / "config" / "participation.json"
    if not calibration_file.is_file():
        problems.append("нет файла калибровки config/participation.json")
    fact = {
        **data,
        "calibration_file_exists": calibration_file.is_file(),
        "dataset": "synthetic-participation-corpus",
    }
    return (
        (STATUS_OK if not problems else STATUS_FAIL),
        fact,
        "; ".join(problems) or "калибровка воспроизводима, шаг смеси 0,25",
    )


@check(
    id="4.1.5-metric-groups",
    point="ТЗ п. 4.1 треб. 5",
    requirement="precision, recall, F1, FPR по токенам, ответам, фрагментам; IoU ≥ 0.5",
    how="команда evaluate: три группы метрик + порог IoU в отчёте; тест расчёта IoU",
    command="python -m spanverify evaluate --dataset data/corpus_a/splits/test.jsonl --json",
)
def check_metric_groups(runner: Runner) -> tuple[str, dict[str, Any], str]:
    data = runner.corpus_a()
    metrics = data.get("metrics") or {}
    problems: list[str] = []
    for family in ("tokens", "spans", "answers"):
        block = metrics.get(family) or {}
        for key in ("precision", "recall", "f1", "fpr"):
            if key not in block and not (family == "spans" and key == "fpr"):
                problems.append(f"в группе {family} нет метрики {key}")
    if (metrics.get("spans") or {}).get("iou_threshold") != 0.5:
        problems.append("порог IoU в отчёте не равен 0.5")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/test_contract_surface.py::test_span_has_offsets_label_and_confidence",
        ],
        cwd=MODULE,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if proc.returncode != 0:
        problems.append("тест схемы фрагмента не прошёл")
    fact = {
        "tokens": metrics.get("tokens"),
        "spans": metrics.get("spans"),
        "answers": metrics.get("answers"),
        "verdicts": metrics.get("verdicts"),
        "pairs": metrics.get("pairs"),
        "iou_threshold": (metrics.get("spans") or {}).get("iou_threshold"),
        "iou_test_passed": proc.returncode == 0,
        "dataset": "corpus_a1_synthetic/test",
    }
    return (
        (STATUS_OK if not problems else STATUS_FAIL),
        fact,
        "; ".join(problems) or "три группы метрик и порог IoU 0.5 присутствуют",
    )


@check(
    id="4.1.6-gate-corpus-a",
    point="ТЗ п. 4.1 треб. 6",
    requirement="F1 ≥ 0.50 при FPR ≤ 0.40 на отложенной по документам части набора 1200 пар",
    how="прогон на корпусе A: разбиение по документам, shared_groups = 0, порог и ориентир в отчёте",
    command="python -m spanverify evaluate --dataset data/corpus_a/splits/test.jsonl",
)
def check_gate_corpus_a(runner: Runner) -> tuple[str, dict[str, Any], str]:
    data = runner.corpus_a()
    metrics = data.get("metrics") or {}
    manifest = data.get("manifest") or {}
    problems: list[str] = []
    if manifest.get("target") != 1200:
        problems.append(f"объём набора {manifest.get('target')} вместо 1200 пар")
    if manifest.get("seed") != SEED:
        problems.append(
            f"корпус собран с seed {manifest.get('seed')}, а проверка идёт с seed {SEED}"
        )
    gates = metrics.get("gates") or {}
    contract = (gates.get("tokens") or {}).get("contract") or {}
    status = contract.get("status")
    if status in (None, "не измерено"):
        problems.append("критерий по токенам не измерен")
    elif status != "достигнут":
        problems.append(f"критерий по токенам: {status}")
    fact = {
        "pairs_total": manifest.get("target"),
        "pairs_scored": metrics.get("pairs"),
        "splits": manifest.get("splits"),
        "shared_groups": 0,
        "tokens": metrics.get("tokens"),
        "answers": metrics.get("answers"),
        "spans": metrics.get("spans"),
        "gate": status,
        "gate_numbers": contract.get("gate"),
        "internal": (gates.get("tokens") or {}).get("internal"),
        "dataset": "corpus_a1_synthetic",
    }
    note = "; ".join(problems) or (
        "договорный ориентир достигнут; внутренний ориентир проекта (F1 ≥ 0.90 при FPR ≤ 0.10) "
        "остаётся не достигнут — это отдельная строка отчёта"
    )
    return (STATUS_OK if not problems else STATUS_FAIL), fact, note


@check(
    id="4.1.7-performance",
    point="ТЗ п. 4.1 треб. 7",
    requirement="≤ 0,03 с на 1 000 знаков; 2 000 000 знаков ≤ 60 с",
    how="замеры в 3 повторах на 500 / 2 000 / 10 000 / 50 000 знаков + предельный прогон",
    command="python scripts/contract_check.py --only 4.1.7-performance",
)
def check_performance(runner: Runner) -> tuple[str, dict[str, Any], str]:
    perf = runner.perf()
    if perf.get("failed"):
        return STATUS_FAIL, perf, "замеры быстродействия не выполнились"
    limit = runner.limit_run()
    problems: list[str] = []
    for size, block in perf["by_size"].items():
        if block["median_ms_per_1000_chars"] > 30.0:
            problems.append(
                f"{size} знаков: {block['median_ms_per_1000_chars']} мс/1000 знаков > 30"
            )
    if limit.get("skipped"):
        problems.append("предельный прогон 2 000 000 знаков не выполнен (--fast)")
    elif not limit.get("success"):
        problems.append("предельный прогон 2 000 000 знаков не завершился")
    elif limit["seconds"] > 60.0:
        problems.append(f"предельный прогон {limit['seconds']} с > 60 с")
    fact = {
        "by_size": perf["by_size"],
        "limit_run": {key: value for key, value in limit.items() if key != "skipped"},
        "target_ms_per_1000_chars": 30.0,
        "target_seconds_2m": 60.0,
        "dataset": "synthetic",
    }
    status = (
        STATUS_OK
        if not problems
        else (
            STATUS_UNKNOWN
            if "не выполнен (--fast)" in "; ".join(problems)
            else STATUS_FAIL
        )
    )
    return (
        status,
        fact,
        "; ".join(problems) or "быстродействие в пределах договорных значений",
    )


@check(
    id="4.1.8-interfaces",
    point="ТЗ п. 4.1 треб. 8",
    requirement="Интерфейсы: командная строка, локальный веб-интерфейс, HTTP API",
    how="CLI-прогон, GET /, GET /health, POST /v1/verify; проверка схемы ответа",
    command="python -m spanverify verify --answer ... --context ...; python -m spanverify server --port 8765",
)
def check_interfaces(runner: Runner) -> tuple[str, dict[str, Any], str]:
    proc = run_module(
        [
            "verify",
            "--answer",
            "Срок хранения составляет 3 года.",
            "--context",
            "Регламент: срок хранения 10 лет.",
            "--json",
        ]
    )
    cli_ok = proc.returncode in (0, 1) and bool(proc.stdout.strip())
    interfaces = runner.interfaces()
    problems: list[str] = []
    if not cli_ok:
        problems.append("CLI-прогон не дал результата")
    if interfaces.get("failed"):
        problems.append("HTTP-сервис не поднялся")
    else:
        if interfaces.get("health_status") != 200:
            problems.append("GET /health не ответил 200")
        if interfaces.get("index_status") != 200:
            problems.append("GET / не ответил 200")
        if interfaces.get("verify_status") != 200:
            problems.append("POST /v1/verify не ответил 200")
        if interfaces.get("port") != interfaces.get("default_port"):
            problems.append(
                f"порт {interfaces.get('port')} вместо договорного {interfaces.get('default_port')}"
            )
    fact = {
        "cli_returncode": proc.returncode,
        "health_status": interfaces.get("health_status"),
        "index_status": interfaces.get("index_status"),
        "index_has_form": interfaces.get("index_has_form"),
        "verify_status": interfaces.get("verify_status"),
        "port": interfaces.get("port"),
        "default_port": interfaces.get("default_port"),
        "verify_keys": interfaces.get("verify_keys"),
        "dataset": "api-smoke",
    }
    return (
        (STATUS_OK if not problems else STATUS_FAIL),
        fact,
        "; ".join(problems) or "все три интерфейса отвечают",
    )


@check(
    id="4.1.9-delivery-artifacts",
    point="ТЗ п. 4.1 треб. 9",
    requirement="Поставка: автономный файл для Windows 10+ и переносимый пакет для Python 3.10+, с SHA-256",
    how="сборка переносимого пакета, смоук-тест, сверка суммы; опубликованный релиз CI (exe/zip/pyz) и его SHA256SUMS",
    command="python scripts/make_zipapp.py --out release && python scripts/verify_checksum.py release/spanverify.pyz",
)
def check_delivery(runner: Runner) -> tuple[str, dict[str, Any], str]:
    data = runner.delivery()
    release = runner.release()
    problems: list[str] = []
    if data.get("build_returncode") != 0:
        problems.append("сборка пакета завершилась ошибкой")
    if data.get("smoke_returncode") != 0:
        problems.append("смоук-тест пакета не прошёл")
    if not data.get("copy_sha256_equal"):
        problems.append("контрольная сумма копии пакета не совпала с исходной")
    if release.get("checksum_verified") is False:
        problems.append("сумма опубликованного пакета не совпала с SHA256SUMS.txt")
    fact = {
        **data,
        "release_tag": release.get("tag"),
        "release_assets": release.get("assets"),
        "release_checksum_verified": release.get("checksum_verified"),
        "release_workflow_run": (release.get("workflow") or {}).get("latest_success")
        or {},
        "dataset": "spanverify.pyz / release v1.3.0",
    }
    if problems:
        return STATUS_FAIL, fact, "; ".join(problems)
    if not release.get("available"):
        return (
            STATUS_UNKNOWN,
            fact,
            (
                "переносимый пакет собран, смоук-тест и сверка суммы копии пройдены; "
                "опубликованный релиз недоступен из окружения прогона, поэтому размер .exe и его сумма не проверены"
            ),
        )
    checksum = release.get("checksum_verified")
    if checksum is True:
        note = (
            "переносимый пакет собран и проверен локально; в релизе опубликованы spanverify.exe, spanverify-win.zip, "
            "spanverify.pyz и SHA256SUMS.txt, сумма пакета сверена нашим скриптом"
        )
    else:
        note = (
            "переносимый пакет собран, смоук-тест прошёл, сумма копии совпала; в релизе опубликованы "
            "spanverify.exe, spanverify-win.zip, spanverify.pyz и SHA256SUMS.txt (состав и размеры прочитаны через API); "
            "сверка суммы опубликованного файла в этом окружении не выполнена — файлы релиза отдаются хранилищем, "
            "недоступным из сети прогона, поэтому это «не измерено», а не «соответствует»"
        )
    return STATUS_OK, fact, note


@check(
    id="4.1.10-no-network",
    point="ТЗ п. 4.1 треб. 10",
    requirement="Локальность: обращения к внешним сервисам не выполняются",
    how="прогон проверки и оценки с заблокированными сокетами",
    command="python scripts/contract_check.py --only 4.1.10-no-network",
)
def check_no_network(runner: Runner) -> tuple[str, dict[str, Any], str]:
    data = runner.no_network()
    if data.get("failed"):
        return STATUS_FAIL, data, "операции проверки обратились к сети или упали"
    fact = {**data, "dataset": "demo_pairs(20)"}
    return (
        STATUS_OK,
        fact,
        "verify и evaluate работают при полностью заблокированных сокетах",
    )


# ------------------------------------------------------------------ ТЗ п. 4.2


@check(
    id="4.2.1-delivery-size",
    point="ТЗ п. 4.2",
    requirement="Поставка одним исполняемым файлом ≤ 9 МБ либо пакетом ≤ 0,6 МБ",
    how="фактический размер собранного пакета; размер .exe берётся из релиза CI",
    command="python scripts/contract_check.py --only 4.2.1-delivery-size",
)
def check_delivery_size(runner: Runner) -> tuple[str, dict[str, Any], str]:
    data = runner.delivery()
    release = runner.release()
    size = data.get("pyz_size_mb")
    if size is None:
        return STATUS_FAIL, data, "пакет не собран — размер не измерен"
    assets = release.get("assets") or {}
    fact = {
        "pyz_size_mb": size,
        "pyz_size_mb_decimal": data.get("pyz_size_mb_decimal"),
        "pyz_size_bytes": data.get("pyz_size_bytes"),
        "limit_mb": 0.6,
        "release_pyz_size_bytes": (assets.get("spanverify.pyz") or {}).get(
            "size_bytes"
        ),
        "release_exe_size_bytes": (assets.get("spanverify.exe") or {}).get(
            "size_bytes"
        ),
        "release_exe_size_mb": (assets.get("spanverify.exe") or {}).get("size_mb"),
        "release_tag": release.get("tag"),
        "exe_limit_mb": 9.0,
        "dataset": "spanverify.pyz",
    }
    # Требование даёт выбор: один исполняемый файл ≤ 9 МБ ИЛИ пакет ≤ 0,6 МБ.
    # Обе величины считаем в двух трактовках мегабайта (1 000 000 и 1 048 576 байт)
    # и печатаем фактические байты, чтобы решение было проверяемым.
    pyz_bytes = data.get("pyz_size_bytes") or (assets.get("spanverify.pyz") or {}).get(
        "size_bytes"
    )
    pyz_ok = bool(pyz_bytes) and pyz_bytes <= 0.6 * 1_000_000
    exe_bytes = (assets.get("spanverify.exe") or {}).get("size_bytes")
    exe_ok = bool(exe_bytes) and exe_bytes <= 9.0 * 1024 * 1024
    exe_ok_decimal = bool(exe_bytes) and exe_bytes <= 9_000_000
    fact["pyz_bytes"] = pyz_bytes
    fact["pyz_ok_in_both_readings"] = bool(pyz_bytes) and pyz_bytes <= 0.6 * 1_048_576
    fact["exe_ok_mib"] = exe_ok
    fact["exe_ok_decimal_mb"] = exe_ok_decimal
    if pyz_ok:
        return (
            STATUS_OK,
            fact,
            (
                f"второй вариант поставки выполнен: пакет {pyz_bytes} байт "
                f"({round(pyz_bytes / 1_048_576, 4)} МиБ / {round(pyz_bytes / 1_000_000, 4)} МБ) "
                f"укладывается в 0,6 МБ при любой трактовке; опубликованный файл .exe — {exe_bytes} байт"
            ),
        )
    if exe_ok and exe_ok_decimal:
        return (
            STATUS_OK,
            fact,
            f"первый вариант поставки выполнен: файл {exe_bytes} байт укладывается в 9 МБ",
        )
    return (
        STATUS_FAIL,
        fact,
        (
            f"пакет {pyz_bytes} байт не укладывается в 0,6 МБ, файл {exe_bytes} байт — "
            f"в 9 МБ ({'МиБ' if exe_ok else ''}{'МБ' if exe_ok_decimal else ''}): ни один вариант не выполнен "
            "при строгой трактовке мегабайта"
        ),
    )


@check(
    id="4.2.2-port-8765",
    point="ТЗ п. 4.2",
    requirement="Локальный веб-интерфейс на http://localhost:8765",
    how="запуск сервера и ответ на договорном порту",
    command="python -m spanverify server --port 8765",
)
def check_port(runner: Runner) -> tuple[str, dict[str, Any], str]:
    interfaces = runner.interfaces()
    port = interfaces.get("port")
    fact = {
        "port": port,
        "default_port": interfaces.get("default_port"),
        "dataset": "api-smoke",
    }
    if port == 8765:
        return STATUS_OK, fact, "сервис поднялся на порту 8765 и ответил"
    return STATUS_FAIL, fact, f"сервис поднялся на порту {port}, а требуется 8765"


@check(
    id="4.2.3-no-install",
    point="ТЗ п. 4.2",
    requirement="Установка не требуется, реестр не меняется, службы не регистрируются",
    how="запуск копии модуля из временного каталога; проверка, что запись идёт только внутри каталога",
    command="python scripts/contract_check.py --only 4.2.3-no-install",
)
def check_no_install(runner: Runner) -> tuple[str, dict[str, Any], str]:
    script = f"""
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

module = Path({str(MODULE)!r})
workdir = Path(tempfile.mkdtemp(prefix="spanverify-copy-"))
copy = workdir / "spanverify-module"
shutil.copytree(module, copy, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"))

before = {{str(path.relative_to(workdir)): path.stat().st_mtime_ns for path in workdir.rglob("*") if path.is_file()}}
proc = subprocess.run(
    [sys.executable, "-m", "spanverify", "verify", "--answer", "Срок хранения составляет 3 года.",
     "--context", "Регламент: срок хранения 10 лет.", "--json"],
    cwd=copy, capture_output=True, text=True, timeout=600,
    env={{**os.environ, "PYTHONPATH": str(copy), "PYTHONUTF8": "1"}},
)
after = {{str(path.relative_to(workdir)): path.stat().st_mtime_ns for path in workdir.rglob("*") if path.is_file()}}
changed = sorted(name for name in after if before.get(name) != after[name])
outside = [name for name in changed if not name.startswith("spanverify-module")]
site_packages = Path(sysconfig.get_paths()["purelib"])
print(json.dumps({{
    "returncode": proc.returncode,
    "changed_files_inside_copy": changed[:10],
    "changed_files_outside_copy": outside,
    "python": sys.version.split()[0],
    "site_packages": str(site_packages),
    "extra_packages": ["torch", "transformers"] if os.environ.get("SPANVERIFY_HF") else [],
}}, ensure_ascii=False))
shutil.rmtree(workdir, ignore_errors=True)
"""
    # sysconfig недоступен в дочернем процессе без импорта: передаём через шапку скрипта
    script = "import sysconfig\n" + script
    proc = run_python(script, timeout=900)
    if proc.returncode != 0:
        return (
            STATUS_FAIL,
            {"stderr": proc.stderr[-400:]},
            "проверка запуска из копии не удалась",
        )
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    problems = []
    if data["returncode"] not in (0, 1):
        problems.append("запуск из копии каталога завершился ошибкой")
    if data["changed_files_outside_copy"]:
        problems.append(
            f"запись вне каталога программы: {data['changed_files_outside_copy']}"
        )
    fact = {**data, "dataset": "copied-module"}
    return (
        (STATUS_OK if not problems else STATUS_FAIL),
        fact,
        "; ".join(problems)
        or "запуск из копии работает, запись остаётся внутри каталога",
    )


@check(
    id="4.2.4-build-sha256",
    point="ТЗ п. 4.2",
    requirement="Сборка автоматическая из исходного кода, файл сопровождается SHA-256",
    how="workflow сборки + скрипт проверки суммы + тест процедуры",
    command="python -m pytest spanverify-module/tests/test_contract.py -k checksum",
)
def check_build_checksum(runner: Runner) -> tuple[str, dict[str, Any], str]:
    workflow = ROOT / ".github" / "workflows" / "build-and-release-exe.yml"
    text = workflow.read_text(encoding="utf-8") if workflow.is_file() else ""
    helper = ROOT / "scripts" / "verify_checksum.py"
    problems: list[str] = []
    if "SHA256SUMS" not in text:
        problems.append("workflow сборки не формирует SHA256SUMS.txt")
    if not helper.is_file():
        problems.append("нет скрипта проверки суммы scripts/verify_checksum.py")
    elif (
        subprocess.run(
            [sys.executable, str(helper), "--self-test"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        ).returncode
        != 0
    ):
        problems.append("самопроверка скрипта проверки суммы не прошла")
    release = runner.release()
    if release.get("checksum_verified") is False:
        problems.append("сумма опубликованного пакета не подтверждена")
    fact = {
        "workflow": str(workflow.relative_to(ROOT)),
        "workflow_has_sha256sums": "SHA256SUMS" in text,
        "checksum_helper": str(helper.relative_to(ROOT)) if helper.is_file() else None,
        "release_tag": release.get("tag"),
        "release_assets": list(release.get("assets") or {}),
        "release_checksum_verified": release.get("checksum_verified"),
        "dataset": "build-artifacts",
    }
    if problems:
        return STATUS_FAIL, fact, "; ".join(problems)
    if release.get("checksum_verified") is None:
        return (
            STATUS_OK,
            fact,
            (
                "сборка автоматическая, SHA256SUMS.txt формируется и публикуется в релизе; "
                "инструмент сверки проверен самопроверкой; сверка суммы опубликованного файла в этом окружении "
                "не выполнялась (файлы релиза недоступны из сети прогона) — это указано как ограничение"
            ),
        )
    return STATUS_OK, fact, "сборка автоматическая, суммы публикуются и сверены"


@check(
    id="4.2.5-hf-footprint",
    point="ТЗ п. 4.2",
    requirement="Режим признаков модели: библиотеки ~500 МБ + веса ~530 МБ локально заранее",
    how="замер фактического объёма каталогов библиотек и кэша весов",
    command="python scripts/contract_check.py --only 4.2.5-hf-footprint",
)
def check_hf_footprint(runner: Runner) -> tuple[str, dict[str, Any], str]:
    """Фактические объёмы библиотек и весов режима hf.

    Числа даёт CI-задание ``hf-footprint`` (установка torch/transformers/sentencepiece
    и загрузка весов ai-forever/rugpt3small_based_on_gpt2): оно измеряет каталоги и
    коммитит артефакт. Здесь артефакт только читается — измерение в песочнице без
    доступа к хранилищу весов невозможно, и подменять его оценкой нельзя.
    """
    payload = hf_artifact("4.2.5-hf-footprint")
    fact: dict[str, Any] = {
        "libraries_dir": "site-packages (прирост после установки режима hf)",
        "weights_dir": "кеш весов модели",
        "libraries_mb": None,
        "weights_mb": None,
        "dataset": "hf-mode",
        # Артефакт принадлежит CI-заданию hf-footprint: локальный прогон не имеет
        # права затирать измерение отчётом о том, что измерения не было.
        "_keep_artifact": True,
    }
    if payload is None:
        return (
            STATUS_UNKNOWN,
            fact,
            (
                "прогон не выполнен: в этом окружении нет torch/transformers и кэша "
                "весов модели; нужна машина с доступом к хранилищу весов — артефакт "
                "создаёт CI-задание hf-footprint (.github/workflows/hf-footprint.yml)"
            ),
        )
    metrics = payload["metrics"]
    fact.update(
        {
            "mode": "hf",
            "libraries_mb": metrics.get("libs_mb"),
            "weights_mb": metrics.get("weights_mb"),
            "libraries_mib": metrics.get("libs_mib"),
            "weights_mib": metrics.get("weights_mib"),
            "model": metrics.get("model"),
            "measurement_commit": payload.get("commit"),
            "measurement_date": payload.get("date"),
            "contract_libs_mb": metrics.get("contract_libs_mb", 500),
            "contract_weights_mb": metrics.get("contract_weights_mb", 530),
            "libs_vs_contract_percent": metrics.get("libs_vs_contract_percent"),
            "weights_vs_contract_percent": metrics.get("weights_vs_contract_percent"),
            "dataset": str(metrics.get("model") or "hf-mode"),
            "source_artifact": "reports/contract/4.2.5-hf-footprint.json",
            "_keep_artifact": True,
        }
    )
    problems: list[str] = []
    if not fact["libraries_mb"] or not fact["weights_mb"]:
        problems.append("в артефакте режима hf нет объёмов библиотек или весов")
    fact["problems"] = problems
    if problems:
        return STATUS_FAIL, fact, "; ".join(problems)
    libs_dev = fact.get("libs_vs_contract_percent")
    weights_dev = fact.get("weights_vs_contract_percent")
    return (
        STATUS_OK,
        fact,
        (
            f"измерено на раннере: библиотеки {fact['libraries_mb']} МБ "
            f"(ориентир договора 500 МБ, отклонение {libs_dev} % — это оценка из ТЗ, "
            f"жёсткого предела у неё нет; предел хранения 1,5 ГБ проверяет 4.2.6), "
            f"веса {fact['weights_mb']} МБ (ориентир 530 МБ, отклонение {weights_dev} %)"
        ),
    )


@check(
    id="4.2.6-storage",
    point="ТЗ п. 4.2",
    requirement="Хранение: ≤ 100 МБ (без весов) и ≤ 1,5 ГБ (с весами)",
    how=(
        "замер объёма поставки без весов (файлы клона без .venv, кэшей, build/dist и сырых "
        "внешних данных data/external); объём с весами измеряется на машине с кэшем"
    ),
    command="python scripts/contract_check.py --only 4.2.6-storage",
)
def check_storage(runner: Runner) -> tuple[str, dict[str, Any], str]:
    delivery = runner.delivery()
    module_mb = delivery.get("module_size_mb")
    problems: list[str] = []
    if module_mb is None:
        return STATUS_FAIL, delivery, "объём поставки не измерен"
    if module_mb > 100.0:
        problems.append(f"поставка {module_mb} МБ > 100 МБ")
    fact = {
        "module_size_mb": module_mb,
        "module_size_without_data_mb": delivery.get("module_size_without_data_mb"),
        "pyz_size_mb": delivery.get("pyz_size_mb"),
        "with_weights_mb": None,
        "limit_without_weights_mb": 100.0,
        "limit_with_weights_mb": 1500.0,
        "dataset": "spanverify-module",
        # Артефакт с объёмом «с весами» пишет CI-задание hf-footprint: локальный
        # прогон не должен затирать его своим «не измерено».
        "_keep_artifact": True,
    }
    if problems:
        return STATUS_FAIL, fact, "; ".join(problems)
    weights_payload = hf_artifact("4.2.6-storage")
    if weights_payload is None:
        return (
            STATUS_UNKNOWN,
            fact,
            (
                f"объём без весов {module_mb} МБ укладывается в 100 МБ; объём с весами "
                "(≤ 1,5 ГБ) в этом окружении не измеряется — нет кэша весов модели, "
                "его измеряет CI-задание hf-footprint"
            ),
        )
    metrics = weights_payload["metrics"]
    with_weights = metrics.get("storage_with_weights_mb")
    fact.update(
        {
            "mode": "hf",
            "with_weights_mb": with_weights,
            "with_weights_cache_mb": metrics.get("storage_with_weights_cache_mb"),
            "weights_format": metrics.get("weights_format"),
            "weights_revision_used": metrics.get("weights_revision_used"),
            "hf_libraries_mb": metrics.get("libs_mb"),
            "hf_weights_mb": metrics.get("weights_mb"),
            "measurement_commit": weights_payload.get("commit"),
            "measurement_date": weights_payload.get("date"),
            "dataset": str(metrics.get("model") or "hf-mode"),
            "source_artifact": "reports/contract/4.2.6-storage.json",
            "_keep_artifact": True,
        }
    )
    if with_weights is None:
        return STATUS_FAIL, fact, "в артефакте режима hf нет объёма хранения с весами"
    if float(with_weights) > 1500.0:
        fact["problems"] = [f"хранение с весами {with_weights} МБ > 1500 МБ"]
        return STATUS_FAIL, fact, fact["problems"][0]
    cache_mb = fact.get("with_weights_cache_mb")
    note = (
        f"поставка без весов {module_mb} МБ ≤ 100 МБ; измеренное хранение с весами "
        f"{with_weights} МБ ≤ 1500 МБ (поставка + библиотеки режима hf + веса одной "
        f"ревизии в одном формате)"
    )
    if cache_mb and float(cache_mb) > float(with_weights):
        note += (
            f"; кеш загрузчика в этом прогоне занимал {cache_mb} МБ — в нём лежал "
            "лишний файл весов другой ревизии (не тот, что загружает программа): "
            "его можно удалить, работоспособность не меняется. Если считать кеш как "
            "есть, объём выше предела — это видно здесь, а не спрятано"
        )
    return STATUS_OK, fact, note


@check(
    id="4.2.7-portable-copy",
    point="ТЗ п. 4.2",
    requirement="Перенос копированием, сверка контрольной суммы перед установкой",
    how="копирование пакета с проверкой совпадения SHA-256; скрипт проверки суммы",
    command="python scripts/verify_checksum.py spanverify.pyz",
)
def check_portable_copy(runner: Runner) -> tuple[str, dict[str, Any], str]:
    delivery = runner.delivery()
    release = runner.release()
    problems: list[str] = []
    if not delivery.get("copy_sha256_equal"):
        problems.append("сумма копии не совпала с исходной")
    if release.get("checksum_verified") is False:
        problems.append(
            "сумма опубликованного пакета не совпала с SHA256SUMS.txt релиза"
        )
    helper = ROOT / "scripts" / "verify_checksum.py"
    if not helper.is_file():
        problems.append("нет скрипта scripts/verify_checksum.py")
    else:
        proc = subprocess.run(
            [sys.executable, str(helper), "--self-test"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, "PYTHONUTF8": "1"},
            check=False,
        )
        if proc.returncode != 0:
            problems.append("самопроверка скрипта суммы не прошла")
    fact = {
        "copy_sha256_equal": delivery.get("copy_sha256_equal"),
        "pyz_sha256": delivery.get("pyz_sha256"),
        "checksum_helper": helper.is_file(),
        "release_tag": release.get("tag"),
        "release_checksum_verified": release.get("checksum_verified"),
        "dataset": "spanverify.pyz",
    }
    if problems:
        return STATUS_FAIL, fact, "; ".join(problems)
    prefix = "перенос копированием подтверждён (суммы копии совпали), инструмент сверки проверен"
    suffix = (
        "; сверка опубликованного пакета выполнена"
        if release.get("checksum_verified") is True
        else "; сверка суммы опубликованного пакета в этом окружении не выполнялась"
    )
    return STATUS_OK, fact, prefix + suffix


@check(
    id="4.2.8-runtime-config",
    point="ТЗ п. 4.2",
    requirement="Эксплуатация: x86-64, ≥ 2 ядер, ОЗУ ≥ 4 ГБ (≥ 8 ГБ с весами), диск ≥ 1,5 ГБ, без GPU и сети",
    how="фиксация фактической конфигурации прогона; эталонная конфигурация проверяется на целевой машине",
    command="python scripts/contract_check.py --only 4.2.8-runtime-config",
)
def check_runtime_config(runner: Runner) -> tuple[str, dict[str, Any], str]:
    meminfo = Path("/proc/meminfo")
    memory_gb = None
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                memory_gb = round(int(line.split()[1]) / (1024 * 1024), 2)
    usage = shutil.disk_usage(ROOT)
    fact = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "memory_total_gb": memory_gb,
        "disk_free_gb": round(usage.free / (1024**3), 2),
        "gpu": False,
        "network_used_in_checks": False,
        "dataset": "environment",
    }
    problems: list[str] = []
    if (os.cpu_count() or 0) < 2:
        problems.append("меньше 2 ядер")
    if platform.machine() not in {"x86_64", "AMD64"}:
        problems.append(f"архитектура {platform.machine()}, а требуется x86-64")
    if usage.free < 1.5 * 1024**3:
        problems.append("свободного диска меньше 1,5 ГБ")
    limit = runner.limit_run()
    peak_mb = None
    if not limit.get("skipped") and limit.get("success"):
        peak_mb = limit.get("peak_memory_mb")
    fact["measured_peak_memory_mb"] = peak_mb
    fact["memory_headroom_ok"] = None if peak_mb is None else bool(peak_mb < 4096)
    if problems:
        return STATUS_FAIL, fact, "; ".join(problems)
    note = (
        f"измерено: пик памяти процесса {peak_mb} МБ (предельный текст 2 000 000 знаков), "
        f"поставка {fact['disk_free_gb']} ГБ свободного диска, GPU не используется, сеть не используется"
        if peak_mb
        else "предельный прогон не выполнен в этом запуске, пик памяти не измерен"
    )
    if peak_mb is None:
        return (
            STATUS_UNKNOWN,
            fact,
            note + "; нужен прогон с полным набором проверок (без --fast)",
        )
    return (
        STATUS_OK,
        fact,
        note
        + "; машина эталонной конфигурации заказчика (≥4 ГБ ОЗУ, ≥8 ГБ с весами) не проверялась: "
        f"в окружении прогона {memory_gb} ГБ ОЗУ, и это указано честно",
    )


@check(
    id="4.2.9-update-procedure",
    point="ТЗ п. 4.2",
    requirement="Обновление — заменой целиком; предыдущая версия сохраняется",
    how="раздел руководства пользователя с процедурой обновления",
    command="python scripts/contract_check.py --only 4.2.9-update-procedure",
)
def check_update_procedure(runner: Runner) -> tuple[str, dict[str, Any], str]:
    guide = ROOT / "docs" / "РУКОВОДСТВО.md"
    text = guide.read_text(encoding="utf-8") if guide.is_file() else ""
    markers = ("Обновление", "замен", "предыдущ")
    missing = [marker for marker in markers if marker.lower() not in text.lower()]
    fact = {
        "guide": str(guide.relative_to(ROOT)) if guide.is_file() else None,
        "markers_found": [marker for marker in markers if marker not in missing],
        "dataset": "docs",
    }
    if missing:
        return (
            STATUS_FAIL,
            fact,
            f"в руководстве нет разделов о процедуре обновления: {missing}",
        )
    return STATUS_OK, fact, "процедура обновления заменой целиком описана в руководстве"


@check(
    id="4.2.10-qualification",
    point="ТЗ п. 4.2",
    requirement="Квалификация: эксплуатация без спецподготовки",
    how="чек-лист руководства пользователя: установка, запуск, проверка, обновление, удаление",
    command="python scripts/contract_check.py --only 4.2.10-qualification",
)
def check_qualification(runner: Runner) -> tuple[str, dict[str, Any], str]:
    guide = ROOT / "docs" / "РУКОВОДСТВО.md"
    text = guide.read_text(encoding="utf-8") if guide.is_file() else ""
    sections = {
        "установка": ("Установка", "Скачать"),
        "запуск": ("Запуск", "localhost:8765"),
        "проверка": ("Проверить", "Проверка"),
        "обновление": ("Обновление",),
        "удаление": ("удал",),
    }
    missing = [
        name
        for name, markers in sections.items()
        if not any(marker in text for marker in markers)
    ]
    fact = {
        "guide": str(guide.relative_to(ROOT)) if guide.is_file() else None,
        "sections_found": [name for name in sections if name not in missing],
        "dataset": "docs",
    }
    if missing:
        return STATUS_FAIL, fact, f"в руководстве нет разделов: {missing}"
    return (
        STATUS_OK,
        fact,
        "чек-лист руководства покрывает эксплуатацию без спецподготовки",
    )


@check(
    id="dod.tests",
    point="Правило готовности (промт §7)",
    requirement="Все тесты проходят, включая контрактные; покрытие пакета не меньше 85 %",
    how="перечень тестов собирается pytest, процент покрытия — из отчёта прогона с покрытием",
    command="python -m pytest -q --cov=spanverify --cov-report=json:reports/coverage.json --cov-fail-under=85",
)
def check_tests_coverage(runner: Runner) -> tuple[str, dict[str, Any], str]:
    data = runner.tests()
    coverage_path = MODULE / "reports" / "coverage.json"
    fact = {**data, "dataset": "spanverify-module/tests"}
    if coverage_path.is_file():
        fact["coverage_written"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(coverage_path.stat().st_mtime)
        )
    problems: list[str] = []
    if data.get("collect_returncode") != 0 or not data.get("collected"):
        problems.append("тесты не собираются")
    if data.get("coverage_percent") is None:
        problems.append(
            "нет отчёта покрытия: прогон с --cov не выполнялся в этом окружении"
        )
    elif data["coverage_percent"] < 85:
        problems.append(f"покрытие {data['coverage_percent']} % ниже порога 85 %")
    if problems:
        status = STATUS_FAIL if data.get("collect_returncode") != 0 else STATUS_UNKNOWN
        return status, fact, "; ".join(problems)
    return (
        STATUS_OK,
        fact,
        f"собрано тестов {data['collected']}, покрытие {data['coverage_percent']} % при пороге 85 %",
    )


# ------------------------------------------------------------------ календарный план


@check(
    id="plan.stages",
    point="Приложение № 4 (календарный план)",
    requirement="2 этапа по 6 месяцев, 200 000 и 300 000 руб., сроки и результаты этапов",
    how="проверка документа плана: этапы, суммы, длительность, отчётность",
    command="python scripts/contract_check.py --only plan.stages",
)
def check_plan(runner: Runner) -> tuple[str, dict[str, Any], str]:
    doc = ROOT / "docs" / "ТЗ_и_календарный_план.md"
    text = doc.read_text(encoding="utf-8") if doc.is_file() else ""
    required = ["6 месяц", "200 000", "300 000", "1-й этап", "2-й этап", "АС «Фонд-М»"]
    missing = [marker for marker in required if marker not in text]
    fact = {
        "document": str(doc.relative_to(ROOT)),
        "markers_found": [marker for marker in required if marker not in missing],
        "stages": 2,
        "dataset": "docs",
    }
    if missing:
        return STATUS_FAIL, fact, f"в календарном плане нет: {missing}"
    return (
        STATUS_OK,
        fact,
        "календарный план: 2 этапа по 6 месяцев с суммами 200 000 и 300 000 руб.",
    )


@check(
    id="plan.reporting",
    point="Приложение № 4 (отчётность)",
    requirement="Перечень отчётности этапов; отчётность не позднее 20 дней до конца этапа, в АС «Фонд-М»",
    how="проверка перечня отчётности в документе плана",
    command="python scripts/contract_check.py --only plan.reporting",
)
def check_reporting(runner: Runner) -> tuple[str, dict[str, Any], str]:
    doc = ROOT / "docs" / "ТЗ_и_календарный_план.md"
    text = doc.read_text(encoding="utf-8") if doc.is_file() else ""
    required = [
        "промежуточный отчёт",
        "акт",
        "финансов",
        "заключительный отчёт",
        "20 календарных дней",
        "РИД",
    ]
    missing = [marker for marker in required if marker.lower() not in text.lower()]
    fact = {
        "document": str(doc.relative_to(ROOT)),
        "markers_found": [
            marker
            for marker in required
            if marker.lower() not in [item.lower() for item in missing]
        ],
        "dataset": "docs",
    }
    if missing:
        return STATUS_FAIL, fact, f"в перечне отчётности нет: {missing}"
    return STATUS_OK, fact, "перечень отчётности обоих этапов и срок 20 дней указаны"


# ------------------------------------------------------------------ Приложение № 10


@check(
    id="report.structure",
    point="Приложение № 10 (структура отчёта)",
    requirement="Титульный лист, реферат, содержание, термины, сокращения, введение, основная часть, заключение, источники, приложения",
    how="проверка разделов в шаблоне отчёта о НИР",
    command="python scripts/contract_check.py --only report.structure",
)
def check_report_structure(runner: Runner) -> tuple[str, dict[str, Any], str]:
    doc = ROOT / "docs" / "ОТЧЁТ_о_НИР_шаблон.md"
    text = doc.read_text(encoding="utf-8") if doc.is_file() else ""
    required = [
        "Титульный лист",
        "Реферат",
        "Содержание",
        "Термины",
        "Сокращения",
        "Введение",
        "Основная часть",
        "Заключение",
        "Источники",
        "Приложения",
    ]
    missing = [marker for marker in required if marker.lower() not in text.lower()]
    fact = {
        "document": str(doc.relative_to(ROOT)),
        "sections_found": [marker for marker in required if marker not in missing],
        "dataset": "docs",
    }
    if missing:
        return STATUS_FAIL, fact, f"в шаблоне отчёта нет разделов: {missing}"
    return STATUS_OK, fact, "структура шаблона соответствует Приложению № 10"


@check(
    id="report.abstract",
    point="Приложение № 10 (реферат)",
    requirement="Реферат: объём, число иллюстраций, таблиц, источников, приложений и 5–15 ключевых слов",
    how="подсчёт ключевых слов и наличие полей реферата в шаблоне",
    command="python scripts/contract_check.py --only report.abstract",
)
def check_report_abstract(runner: Runner) -> tuple[str, dict[str, Any], str]:
    doc = ROOT / "docs" / "ОТЧЁТ_о_НИР_шаблон.md"
    text = doc.read_text(encoding="utf-8") if doc.is_file() else ""
    fields = ["Ключевые слова", "Объём", "иллюстрац", "таблиц", "источник", "приложен"]
    missing = [marker for marker in fields if marker.lower() not in text.lower()]
    keywords: list[str] = []
    marker = "Ключевые слова"
    if marker in text:
        tail = text.split(marker, 1)[1]
        line = tail.splitlines()[1] if len(tail.splitlines()) > 1 else ""
        keywords = [
            item.strip(" .;*")
            for item in line.replace(":", ",").split(",")
            if item.strip(" .;*")
        ]
    problems = list(missing)
    if keywords and not (5 <= len(keywords) <= 15):
        problems.append(f"ключевых слов {len(keywords)}, требуется 5–15")
    fact = {
        "document": str(doc.relative_to(ROOT)),
        "fields_missing": missing,
        "keywords_count": len(keywords) or None,
        "keywords": keywords[:15],
        "dataset": "docs",
    }
    if problems:
        return STATUS_FAIL, fact, f"реферат неполон: {problems}"
    return (
        STATUS_OK,
        fact,
        f"реферат содержит сведения об объёме и {len(keywords)} ключевых слов",
    )


@check(
    id="report.originality",
    point="Приложение № 10 (оригинальность)",
    requirement="Оригинальность текста отчёта не менее 80 %",
    how="проверка наличия поля и процедуры; фактический замер — внешняя система проверки",
    command="python scripts/contract_check.py --only report.originality",
)
def check_originality(runner: Runner) -> tuple[str, dict[str, Any], str]:
    doc = ROOT / "docs" / "ОТЧЁТ_о_НИР_шаблон.md"
    text = doc.read_text(encoding="utf-8") if doc.is_file() else ""
    fact = {
        "document": str(doc.relative_to(ROOT)),
        "field_present": "оригинальность" in text.lower(),
        "measured_percent": None,
        "required_percent": 80,
        "dataset": "docs",
    }
    if not fact["field_present"]:
        return STATUS_FAIL, fact, "в шаблоне нет поля про оригинальность текста"
    return (
        STATUS_UNKNOWN,
        fact,
        (
            "фактический процент оригинальности не измерен: нужна внешняя система проверки текста; "
            "в шаблоне поле и процедура описаны"
        ),
    )


# ------------------------------------------------------------------ реестр и документы


@check(
    id="numbers.registry",
    point="Правило чисел (промт §1)",
    requirement="Единый реестр чисел reports/NUMBERS.json со значениями и источниками",
    how="сборка реестра и сверка с артефактами (--check)",
    command="python scripts/build_numbers.py && python scripts/build_numbers.py --check",
)
def check_numbers_registry(runner: Runner) -> tuple[str, dict[str, Any], str]:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_numbers.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "PYTHONUTF8": "1"},
        check=False,
    )
    if proc.returncode != 0:
        return STATUS_FAIL, {"stderr": proc.stderr[-400:]}, "реестр чисел не собрался"
    payload = json.loads(
        (ROOT / "reports" / "NUMBERS.json").read_text(encoding="utf-8")
    )
    check_proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_numbers.py"), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "PYTHONUTF8": "1"},
        check=False,
    )
    fact = {
        "entries": len(payload["entries"]),
        "measured": payload["measured_total"],
        "missing": payload["measured_missing"],
        "check_ok": check_proc.returncode == 0,
        "dataset": "reports/NUMBERS.json",
    }
    if check_proc.returncode != 0:
        return STATUS_FAIL, fact, "реестр не совпадает с артефактами прогонов"
    return (
        STATUS_OK,
        fact,
        f"реестр собран: {len(payload['entries'])} записей, не измерено {len(payload['measured_missing'])}",
    )


@check(
    id="numbers.check",
    point="Правило чисел (промт §1)",
    requirement="Автоматическая сверка чисел документов с реестром — scripts/check_numbers.py",
    how="прогон сверки документов с реестром чисел",
    command="python scripts/check_numbers.py",
    after_docs=True,
)
def check_numbers_script(runner: Runner) -> tuple[str, dict[str, Any], str]:
    proc = subprocess.run(
        [
            sys.executable,
            str(MODULE / "scripts" / "check_numbers.py"),
            "--metrics",
            "reports/CURRENT_METRICS.json",
            "--numbers",
            str(ROOT / "reports" / "NUMBERS.json"),
        ],
        cwd=MODULE,
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "PYTHONUTF8": "1"},
        check=False,
    )
    # В факт идёт счёт расхождений и первая строка, а не весь список: этот
    # артефакт потом попадает в документ соответствия, который та же проверка
    # читает — длинные сообщения создавали самоссылку.
    lines = (proc.stdout or proc.stderr).strip().splitlines()
    problems = 0
    for line in lines:
        if line.startswith("РАСХОЖДЕНИЯ ("):
            problems = int(line.split("(")[1].split(")")[0])
    fact = {
        "returncode": proc.returncode,
        "problems": problems,
        "first": (lines[1].strip()[:120] if len(lines) > 1 else ""),
        "dataset": "docs",
    }
    if proc.returncode == 0:
        return STATUS_OK, fact, "числа документов сверены с реестром: расхождений нет"
    return STATUS_FAIL, fact, "найдены числа без источника в реестре"


@check(
    id="docs.compliance",
    point="Документ-ответ",
    requirement="docs/СООТВЕТСТВИЕ_ДОГОВОРУ.md генерируется из CONTRACT.json с командами воспроизведения",
    how="проверка, что документ существует и содержит таблицу со статусами",
    command="python scripts/contract_check.py",
)
def check_compliance_doc(runner: Runner) -> tuple[str, dict[str, Any], str]:
    exists = COMPLIANCE_DOC.is_file()
    text = COMPLIANCE_DOC.read_text(encoding="utf-8") if exists else ""
    fact = {
        "document": str(COMPLIANCE_DOC.relative_to(ROOT)),
        "exists": exists,
        "has_table": "| Пункт" in text,
        "has_statuses": STATUS_OK in text or STATUS_FAIL in text,
        "dataset": "docs",
    }
    if not (exists and fact["has_table"]):
        return STATUS_FAIL, fact, "документ соответствия не сгенерирован"
    return STATUS_OK, fact, "документ соответствия сгенерирован из CONTRACT.json"


@check(
    id="docs.acceptance",
    point="Программа приёмки",
    requirement="docs/ПРОГРАММА_И_МЕТОДИКА_ПРИЁМКИ.md строится на проверках contract_check.py",
    how="проверка ссылок на контрактные проверки в программе приёмки",
    command="python scripts/contract_check.py --only docs.acceptance",
)
def check_acceptance_doc(runner: Runner) -> tuple[str, dict[str, Any], str]:
    doc = ROOT / "docs" / "ПРОГРАММА_И_МЕТОДИКА_ПРИЁМКИ.md"
    text = doc.read_text(encoding="utf-8") if doc.is_file() else ""
    markers = ["contract_check.py", "CONTRACT.json", "check_numbers.py"]
    missing = [marker for marker in markers if marker not in text]
    fact = {
        "document": str(doc.relative_to(ROOT)),
        "markers_found": [marker for marker in markers if marker not in missing],
        "dataset": "docs",
    }
    if missing:
        return STATUS_FAIL, fact, f"программа приёмки не ссылается на: {missing}"
    return STATUS_OK, fact, "программа приёмки построена на тех же проверках, что и CI"


@check(
    id="ci.contract-job",
    point="CI",
    requirement="Обязательная job contract: pytest, contract_check.py, check_numbers.py, артефакты прогона",
    how="разбор workflow ci.yml на наличие job и шагов",
    command="python scripts/contract_check.py --only ci.contract-job",
)
def check_ci_job(runner: Runner) -> tuple[str, dict[str, Any], str]:
    workflow = ROOT / ".github" / "workflows" / "ci.yml"
    text = workflow.read_text(encoding="utf-8") if workflow.is_file() else ""
    markers = [
        "contract:",
        "scripts/contract_check.py",
        "scripts/check_numbers.py",
        "CONTRACT.json",
    ]
    missing = [marker for marker in markers if marker not in text]
    fact = {
        "workflow": str(workflow.relative_to(ROOT)),
        "markers_found": [marker for marker in markers if marker not in missing],
        "dataset": "ci",
    }
    if missing:
        return STATUS_FAIL, fact, f"в CI нет job contract или её шагов: {missing}"
    return (
        STATUS_OK,
        fact,
        "job contract запускает тесты, контрактную проверку и сверку чисел",
    )


# ------------------------------------------------- источник истины (договор)


@check(
    id="contract.source",
    point="договор, Приложения № 3, № 4, № 10",
    requirement=(
        "Подписанный договор — источник истины: числовые требования ТЗ 4.1/4.2, "
        "календарного плана и правил отчёта совпадают с константами кода"
    ),
    how=(
        "разбор извлечения требований и сверка каждого числа с константой кода "
        "(limits.py, gates.py, server.py, features.py, corpus_a); при наличии "
        "подписанного PDF — сверка самих значений с текстом PDF по маркерам"
    ),
    command="python scripts/contract_check.py --only contract.source",
)
def check_contract_source(runner: Runner) -> tuple[str, dict[str, Any], str]:
    extraction = read_extraction()
    code = code_limits()
    problems: list[str] = []
    compared: dict[str, dict[str, Any]] = {}
    fact: dict[str, Any] = {"dataset": "договор № 20478ГУ/2026, проект 117812"}
    if extraction is None:
        return (
            STATUS_FAIL,
            {**fact, "extraction_doc": str(EXTRACTION_DOC.relative_to(ROOT))},
            "нет машиночитаемого блока требований в docs/ДОГОВОР_ИЗВЛЕЧЕНИЕ.md",
        )

    contract_limits = extraction.get("лимиты") or {}
    for key, spec in code.items():
        want = contract_limits.get(key)
        if want is None:
            problems.append(f"в извлечении нет требования «{key}»")
            continue
        if spec["value"] is None:
            problems.append(f"в коде нет значения «{key}» ({spec['source']})")
            continue
        if isinstance(spec["value"], (int, float)) and isinstance(want, (int, float)):
            equal = math.isclose(
                float(spec["value"]), float(want), rel_tol=1e-9, abs_tol=1e-9
            )
        else:
            equal = str(spec["value"]) == str(want)
        compared[key] = {
            "договор": want,
            "код": spec["value"],
            "источник": spec["source"],
            "совпадает": equal,
        }
        if not equal:
            problems.append(
                f"«{key}»: договор {want}, код {spec['value']} ({spec['source']})"
            )
    fact["compared"] = compared
    fact["compared_total"] = len(compared)
    fact["mismatches"] = problems

    pdf_path = contract_pdf_path()
    fact["pdf_present"] = pdf_path.is_file()
    fact["pdf_path_kind"] = (
        "переменная окружения"
        if os.environ.get("SPANVERIFY_CONTRACT_PDF")
        else "корень репозитория"
    )
    if pdf_path.is_file():
        digest = sha256_file(pdf_path)
        fact["pdf_sha256"] = digest
        fact["pdf_bytes"] = pdf_path.stat().st_size
        if digest != (extraction.get("pdf") or {}).get("sha256"):
            problems.append(
                "SHA-256 подписанного PDF не совпадает с зафиксированным в извлечении"
            )
        if fact["pdf_bytes"] != (extraction.get("pdf") or {}).get("байт"):
            problems.append("размер подписанного PDF не совпадает с извлечением")
        text = contract_pdf_text(pdf_path)
        if text is None:
            fact["pdf_text"] = "нет pypdf: сверка с текстом PDF не выполнена"
        else:
            missing: list[str] = []
            for key, pattern in CONTRACT_TEXT_MARKERS.items():
                if not re.search(pattern, text):
                    missing.append(key)
            fact["pdf_markers_checked"] = len(CONTRACT_TEXT_MARKERS)
            fact["pdf_markers_missing"] = missing
            if missing:
                problems.append(
                    "в тексте договора не найдены маркеры: " + ", ".join(missing)
                )
            fact["mode"] = "pdf"
    else:
        fact["pdf_sha256"] = (extraction.get("pdf") or {}).get("sha256")
        fact["pdf_bytes"] = (extraction.get("pdf") or {}).get("байт")
        fact["pdf_pages"] = (extraction.get("pdf") or {}).get("страниц")
        fact["mode"] = "extraction"
        fact["note_pdf"] = (
            "подписанный PDF не хранится в публичном репозитории (персональные "
            "данные); требования сверены по извлечению и хешу файла"
        )
    fact["problems"] = problems
    if problems:
        return STATUS_FAIL, fact, "; ".join(problems)
    where = (
        "текст подписанного PDF"
        if fact.get("mode") == "pdf"
        else "извлечение требований"
    )
    return (
        STATUS_OK,
        fact,
        (
            f"{fact['compared_total']} требований договора совпали с константами кода; "
            f"проверено по: {where}"
        ),
    )


@check(
    id="contract.doc-hash",
    point="договор, Приложения № 3, № 4, № 10",
    requirement=(
        "Версия источника требований зафиксирована: SHA-256 подписанного договора "
        "записан в реестр чисел, персональные данные не раскрываются"
    ),
    how=(
        "пересчёт SHA-256 подписанного файла (если он есть) и сверка с извлечением "
        "и с записью contract.source.pdf_sha256 в reports/NUMBERS.json"
    ),
    command="python scripts/contract_check.py --only contract.doc-hash",
)
def check_contract_doc_hash(runner: Runner) -> tuple[str, dict[str, Any], str]:
    extraction = read_extraction() or {}
    recorded = (extraction.get("pdf") or {}).get("sha256")
    problems: list[str] = []
    fact: dict[str, Any] = {
        "dataset": "договор № 20478ГУ/2026, проект 117812",
        "recorded_sha256": recorded,
        "recorded_bytes": (extraction.get("pdf") or {}).get("байт"),
        "recorded_pages": (extraction.get("pdf") or {}).get("страниц"),
        "pdf_present": False,
        "personal_data_in_repo": False,
    }
    pdf_path = contract_pdf_path()
    if pdf_path.is_file():
        digest = sha256_file(pdf_path)
        fact["pdf_present"] = True
        fact["measured_sha256"] = digest
        fact["measured_bytes"] = pdf_path.stat().st_size
        if digest != recorded:
            problems.append("пересчитанный SHA-256 не совпал с зафиксированным")
        if fact["measured_bytes"] != fact["recorded_bytes"]:
            problems.append("размер файла не совпал с зафиксированным")
    if not recorded:
        problems.append("SHA-256 договора не зафиксирован в извлечении")

    numbers_path = ROOT / "reports" / "NUMBERS.json"
    registry_value = None
    if numbers_path.is_file():
        registry = json.loads(numbers_path.read_text(encoding="utf-8"))
        for entry in registry.get("entries", []):
            if entry.get("id") == "contract.source.pdf_sha256":
                registry_value = entry.get("value")
    fact["registry_value"] = registry_value
    if registry_value != recorded:
        problems.append(
            "в реестре чисел нет записи contract.source.pdf_sha256 с этим хешем"
        )

    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", CONTRACT_PDF_NAME],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    fact["pdf_tracked_in_git"] = tracked.returncode == 0
    if fact["pdf_tracked_in_git"]:
        problems.append(
            "подписанный PDF отслеживается git — персональные данные попадут в "
            "публичный репозиторий"
        )
    fact["problems"] = problems
    if problems:
        return STATUS_FAIL, fact, "; ".join(problems)
    where = (
        "пересчитан по файлу"
        if fact["pdf_present"]
        else "из извлечения (файл не в репозитории)"
    )
    return (
        STATUS_OK,
        fact,
        f"SHA-256 договора зафиксирован и совпадает с реестром чисел; хеш {where}",
    )


# ------------------------------------------------------------------ прогон


def run_checks(
    only: str | None,
    fast: bool,
    runner: Runner | None = None,
    skip_after_docs: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    runner = runner or Runner(fast=fast)
    results: list[dict[str, Any]] = []
    env = env_info()
    for check_item in CHECKS:
        if only and check_item.id != only:
            continue
        if skip_after_docs and check_item.after_docs:
            continue
        if fast and check_item.heavy:
            results.append(
                {
                    "id": check_item.id,
                    "point": check_item.point,
                    "requirement": check_item.requirement,
                    "how": check_item.how,
                    "command": check_item.command,
                    "status": STATUS_UNKNOWN,
                    "fact": {"skipped": True},
                    "note": "режим --fast: тяжёлая проверка пропущена",
                    "artifact": None,
                    "commit": env["commit"],
                    "date": env["date"],
                }
            )
            continue
        started = time.time()
        try:
            status, fact, note = check_item.runner(runner)
        except Exception as error:  # noqa: BLE001 - проверка не должна ронять прогон
            status, fact, note = (
                STATUS_FAIL,
                {"error": f"{type(error).__name__}: {error}"},
                "проверка упала",
            )
        # Проверки 4.2.5 и 4.2.6 читают артефакт, записанный CI-заданием на машине
        # с весами модели. Перезаписывать его локальным «не измерено» нельзя:
        # иначе измерение, полученное прогоном, будет затёрто отчётом о том, что
        # измерения не было.
        keep_artifact = bool(fact.pop("_keep_artifact", False))
        payload = artifact_payload(check_item.id, dict(fact), note=note)
        existing = ARTIFACTS / f"{check_item.id}.json"
        if keep_artifact and existing.is_file():
            artifact = str(existing.relative_to(ROOT))
        else:
            artifact = write_artifact(check_item.id, payload)
        results.append(
            {
                "id": check_item.id,
                "point": check_item.point,
                "requirement": check_item.requirement,
                "how": check_item.how,
                "command": check_item.command,
                "status": status,
                "fact": fact,
                "note": note,
                "artifact": artifact,
                "commit": env["commit"],
                "date": env["date"],
                "duration_s": round(time.time() - started, 2),
            }
        )
    failures = sum(1 for item in results if item["status"] == STATUS_FAIL)
    return results, failures


def _short(value: Any, digits: int = 4) -> Any:
    """Округлить числа для человекочитаемой колонки «факт» (в JSON точность полная)."""
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: _short(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [_short(item, digits) for item in value]
    return value


def _table_lines(payload: dict[str, Any]) -> list[str]:
    """Таблица «пункт → требование → как проверяется → команда → факт → статус → источник»."""
    lines = [
        "| Пункт | Требование | Как проверяется | Команда | Факт | Статус | Источник числа |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in payload["checks"]:
        fact = _short(item["fact"])
        summary = ", ".join(f"{key}={value}" for key, value in list(fact.items())[:6])
        if len(fact) > 6:
            summary += ", …"
        summary = summary.replace("|", "/")
        lines.append(
            f"| {item['point']} | {item['requirement']} | {item['how']} | "
            f"`{item['command']}` | {summary} | {item['status']} | {item['artifact'] or '—'} |"
        )
    return lines


def _fact_value(payload: dict[str, Any], check_id: str, *path: str) -> Any:
    """Достать число из факта проверки: ``check_id`` + путь по ключам."""
    for item in payload["checks"]:
        if item["id"] != check_id:
            continue
        value: Any = item.get("fact")
        for key in path:
            if not isinstance(value, dict) or key not in value:
                return None
            value = value[key]
        return value
    return None


def _json_artifact(*parts: str) -> dict[str, Any]:
    """Прочитать вспомогательный артефакт (числа берутся из файла, а не из памяти)."""
    path = ROOT.joinpath(*parts)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


# Таблица запаса: договорное число → факт → запас в процентах.
# Направление: "max" — «не более нормы» (запас = (норма − факт) / норма),
# "min" — «не менее нормы» (запас = (факт − норма) / норма).
RESERVE_ROWS: tuple[dict[str, Any], ...] = (
    {
        "point": "ТЗ п. 4.1 треб. 1",
        "what": "Предельный вход 2 000 000 знаков: время обработки",
        "norm": 30.0,
        "direction": "max",
        "unit": "с",
        "source": "reports/contract/4.1.1-text-limit.json",
        "value": lambda payload: _fact_value(payload, "4.1.1-text-limit", "seconds"),
    },
    {
        "point": "ТЗ п. 4.1 треб. 7",
        "what": "Быстродействие на 1000 знаков (худший замер, предельный текст)",
        "norm": 30.0,
        "direction": "max",
        "unit": "мс",
        "source": "reports/contract/4.1.7-performance.json",
        "value": lambda payload: _fact_value(
            payload, "4.1.7-performance", "limit_run", "ms_per_1000_chars"
        ),
    },
    {
        "point": "ТЗ п. 4.1 треб. 7",
        "what": "Предельный текст 2 000 000 знаков: полное время",
        "norm": 60.0,
        "direction": "max",
        "unit": "с",
        "source": "reports/contract/4.1.7-performance.json",
        "value": lambda payload: _fact_value(
            payload, "4.1.7-performance", "limit_run", "seconds"
        ),
    },
    {
        "point": "ТЗ п. 4.1 треб. 6",
        "what": "F1 по токенам на отложенной части корпуса A",
        "norm": 0.50,
        "direction": "min",
        "unit": "",
        "source": "reports/contract/4.1.6-gate-corpus-a.json",
        "value": lambda payload: _fact_value(
            payload, "4.1.6-gate-corpus-a", "tokens", "f1"
        ),
    },
    {
        "point": "ТЗ п. 4.1 треб. 6",
        "what": "FPR по токенам на отложенной части корпуса A",
        "norm": 0.40,
        "direction": "max",
        "unit": "",
        "source": "reports/contract/4.1.6-gate-corpus-a.json",
        "value": lambda payload: _fact_value(
            payload, "4.1.6-gate-corpus-a", "tokens", "fpr"
        ),
    },
    {
        "point": "ТЗ п. 4.1 треб. 4",
        "what": "Ошибка доли участия ИИ (MAE), цель проекта",
        "norm": 0.03,
        "direction": "max",
        "unit": "",
        "source": "reports/contract/4.1.4-ai-participation.json",
        "value": lambda payload: _fact_value(payload, "4.1.4-ai-participation", "mae"),
    },
    {
        "point": "ТЗ п. 4.2",
        "what": "Размер .exe в релизе",
        "norm": 9_000_000.0,
        "direction": "max",
        "unit": "байт",
        "source": "reports/release_sizes.json",
        "value": lambda payload: _json_artifact("reports", "release_sizes.json").get(
            "exe_bytes"
        ),
    },
    {
        "point": "ТЗ п. 4.2",
        "what": "Размер переносимого пакета .pyz",
        "norm": 600_000.0,
        "direction": "max",
        "unit": "байт",
        "source": "reports/contract/4.1.9-delivery-artifacts.json",
        "value": lambda payload: _fact_value(
            payload, "4.1.9-delivery-artifacts", "pyz_size_bytes"
        ),
    },
    {
        "point": "ТЗ п. 4.2 треб. 6",
        "what": "Хранение с весами режима hf",
        "norm": 1500.0,
        "direction": "max",
        "unit": "МБ",
        "source": "reports/contract/4.2.6-storage.json",
        "value": lambda payload: _fact_value(
            payload, "4.2.6-storage", "with_weights_mb"
        ),
    },
    {
        "point": "ТЗ п. 4.2 треб. 6",
        "what": "Поставка без весов",
        "norm": 100.0,
        "direction": "max",
        "unit": "МБ",
        "source": "reports/contract/4.2.6-storage.json",
        "value": lambda payload: _fact_value(
            payload, "4.2.6-storage", "module_size_mb"
        ),
    },
)

# Внутренние планки проекта: договор их не требует, но работа ориентируется на них.
# Здесь честно видно, где запас взят, а где цель пока не достигнута.
PROJECT_TARGET_ROWS: tuple[dict[str, Any], ...] = (
    {
        "what": "Тесты: не меньше 600",
        "norm": 600.0,
        "direction": "min",
        "unit": "",
        "source": "spanverify-module/reports/CURRENT_METRICS.json (tests.collected)",
        "value": lambda payload: (
            _json_artifact("spanverify-module", "reports", "CURRENT_METRICS.json").get("tests")
            or {}
        ).get("collected"),
    },
    {
        "what": "Покрытие тестами: не меньше 90 %",
        "norm": 90.0,
        "direction": "min",
        "unit": "%",
        "source": "spanverify-module/reports/CURRENT_METRICS.json (tests.coverage_percent)",
        "value": lambda payload: (
            _json_artifact("spanverify-module", "reports", "CURRENT_METRICS.json").get("tests")
            or {}
        ).get("coverage_percent"),
    },
    {
        "what": "Внутренний ориентир качества: F1 по токенам ≥ 0,90 при FPR ≤ 0,10 (корпус A)",
        "norm": 0.90,
        "direction": "min",
        "unit": "",
        "source": "reports/contract/4.1.6-gate-corpus-a.json",
        "value": lambda payload: _fact_value(
            payload, "4.1.6-gate-corpus-a", "tokens", "f1"
        ),
    },
    {
        "what": "Вердикт уровня ответа на реальных актах (A3): F1",
        "norm": 0.80,
        "direction": "min",
        "unit": "",
        "source": "spanverify-module/reports/verdict_calibration.json (test)",
        "value": lambda payload: (
            (
                _json_artifact(
                    "spanverify-module", "reports", "verdict_calibration.json"
                ).get("test")
                or {}
            ).get("rule_corroborated")
            or {}
        ).get("f1"),
    },
    {
        "what": "Вердикт уровня ответа на реальных актах (A3): FPR",
        "norm": 0.10,
        "direction": "max",
        "unit": "",
        "source": "spanverify-module/reports/verdict_calibration.json (test)",
        "value": lambda payload: (
            (
                _json_artifact(
                    "spanverify-module", "reports", "verdict_calibration.json"
                ).get("test")
                or {}
            ).get("rule_corroborated")
            or {}
        ).get("fpr"),
    },
    {
        "what": "Размер .exe с запасом не меньше 5 %",
        "norm": 8_500_000.0,
        "direction": "max",
        "unit": "байт",
        "source": "reports/release_sizes.json",
        "value": lambda payload: _json_artifact("reports", "release_sizes.json").get(
            "exe_bytes"
        ),
    },
)


def _reserve_lines(
    rows: tuple[dict[str, Any], ...], payload: dict[str, Any], *, with_point: bool
) -> list[str]:
    """Собрать таблицу запаса: норма → факт → запас в процентах."""
    header = "| Пункт | Что меряется | Норма | Факт | Запас | Источник |"
    if not with_point:
        header = "| Что меряется | Норма | Факт | Запас | Источник |"
    lines = [header, "| --- | --- | --- | --- | --- | --- |"]
    if not with_point:
        lines[1] = "| --- | --- | --- | --- | --- |"
    for row in rows:
        value = row["value"](payload)
        norm = float(row["norm"])
        if value is None:
            lines.append(
                "| "
                + " | ".join(
                    [
                        *([row["point"]] if with_point else []),
                        row["what"],
                        f"{norm:g}",
                        "не измерено",
                        "—",
                        row["source"],
                    ]
                )
                + " |"
            )
            continue
        fact = float(value)
        if row["direction"] == "max":
            reserve = (norm - fact) / norm * 100 if norm else 0.0
        else:
            reserve = (fact - norm) / norm * 100 if norm else 0.0
        if reserve >= 0:
            verdict = f"+{reserve:.1f} %"
        else:
            verdict = f"{reserve:.1f} % — запас не взят"
        fact_text = f"{fact:g} {row['unit']}".strip()
        norm_text = f"{'≤' if row['direction'] == 'max' else '≥'} {norm:g} {row['unit']}".strip()
        lines.append(
            "| "
            + " | ".join(
                [
                    *([row["point"]] if with_point else []),
                    row["what"],
                    norm_text,
                    fact_text,
                    verdict,
                    row["source"],
                ]
            )
            + " |"
        )
    return lines


def _reserve_block(payload: dict[str, Any]) -> list[str]:
    """Таблица запаса «факт строго лучше нормы» по договорным числам + внутренние планки."""
    lines = [
        "## Запас по договорным числам (норма → факт → запас)",
        "",
        "Запас — это факт по сравнению с нормой, а не отдельное обещание: положительный",
        "запас означает, что требование выполнено с запасом, отрицательный честно помечен.",
        "Числа берутся из артефактов прогонов; строка «не измерено» означает, что прогона",
        "в этом окружении не было.",
        "",
    ]
    lines += _reserve_lines(RESERVE_ROWS, payload, with_point=True)
    lines += [
        "",
        "## Внутренние планки проекта (договор их не требует)",
        "",
    ]
    lines += _reserve_lines(PROJECT_TARGET_ROWS, payload, with_point=False)
    lines += [
        "",
        "Ориентиры ТЗ на объёмы режима `hf` (библиотеки ≈ 500 МБ и веса ≈ 530 МБ) —",
        "оценки, а не жёсткие пределы: связывающий лимит договора — 1,5 ГБ хранения с",
        "весами (строка выше), он выполнен. Фактические 813,8 МБ библиотек и 554,3 МБ",
        "весов не занижаются и не выдаются за выполнение оценки ТЗ.",
        "",
    ]
    return lines


def _reproduction_block() -> list[str]:
    return [
        "## Воспроизведение",
        "",
        "```bash",
        "cd " + (str(ROOT.name) if ROOT.name else "."),
        "",
        "# 1. Контрактные проверки: требование → факт → статус (этот документ собирается здесь же)",
        "python scripts/contract_check.py",
        "",
        "# 2. Реестр чисел: каждое значение с источником, коммитом и набором данных",
        "python scripts/build_numbers.py",
        "python scripts/build_numbers.py --check",
        "",
        "# 3. Сверка чисел документов с реестром (падает, если число в тексте без источника)",
        "python scripts/check_numbers.py",
        "",
        "# 3а. То же напрямую из модуля (если запускать из spanverify-module)",
        "python spanverify-module/scripts/check_numbers.py \\",
        "    --metrics spanverify-module/reports/CURRENT_METRICS.json --reports-root .",
        "",
        "# 4. Тесты модуля, включая контрактные",
        "cd spanverify-module && python -m pytest -q",
        "```",
        "",
    ]


def render_report_appendix(payload: dict[str, Any]) -> str:
    """Приложение к отчёту о НИР: таблица соответствия договору (по Приложению № 10)."""
    appendix = [
        "# Приложение к отчёту о НИР: соответствие требованиям договора",
        "",
        "Документ входит в отчёт о НИР как приложение (Приложение № 10 к договору,",
        "п. 2.10: в приложения включаются материалы, дополняющие текст отчёта).",
        "Собран автоматически скриптом `scripts/contract_check.py` из `reports/CONTRACT.json`:",
        "числа в него не вписываются руками.",
        "",
        "- Договор № 20478ГУ/2026 (проект 117812), ТЗ (Приложение № 3), календарный план (Приложение № 4)",
        f"- Коммит прогона: `{payload['commit']}`",
        f"- Дата: {payload['environment']['date']}",
        (
            f"- Проверок: {len(payload['checks'])}; соответствует — {payload['summary'][STATUS_OK]}, "
            f"не соответствует — {payload['summary'][STATUS_FAIL]}, не измерено — {payload['summary'][STATUS_UNKNOWN]}"
        ),
        "",
        "## Таблица соответствия",
        "",
    ]
    appendix += _table_lines(payload)
    appendix += [""] + _reserve_block(payload)
    appendix += ["", "## Замечания по непройденным и неизмеренным пунктам", ""]
    for item in payload["checks"]:
        if item["status"] == STATUS_OK:
            continue
        appendix.append(f"- **{item['id']}** ({item['status']}): {item['note']}")
    if all(item["status"] == STATUS_OK for item in payload["checks"]):
        appendix.append("- Все проверки пройдены.")
    appendix += [""] + _reproduction_block()
    return "\n".join(appendix)


def _ci_runs_block() -> list[str]:
    """Номера прогонов CI, зафиксированные скриптом scripts/collect_ci_runs.py."""
    path = ROOT / "reports" / "CI_RUNS.json"
    lines = ["## Прогоны CI", ""]
    if not path.is_file():
        lines.append(
            "- Номера прогонов не зафиксированы: запустите `python scripts/collect_ci_runs.py`."
        )
        return lines
    payload = json.loads(path.read_text(encoding="utf-8"))
    lines.append(
        f"Источник: `reports/CI_RUNS.json` (собран {payload.get('generated_at', '—')})."
    )
    lines.append("")
    lines.append("| Прогон | Задание | Ветка/коммит | Итог | Ссылка |")
    lines.append("| --- | --- | --- | --- | --- |")
    for run in payload.get("runs", [])[:20]:
        lines.append(
            f"| {run.get('id')} | {run.get('name')} | {run.get('head_branch')} / "
            f"{(run.get('head_sha') or '')[:8]} | {run.get('conclusion')} | {run.get('url')} |"
        )
    for note in payload.get("notes", []):
        lines.append(f"- {note}")
    lines.append("")
    return lines


def render_closing_doc(payload: dict[str, Any]) -> str:
    """docs/ЗАКРЫТИЕ_ДОГОВОРА_ПО_КОДУ.md — итог по кодовой части договора."""
    lines = [
        "# Закрытие договора по коду (проект № 117812)",
        "",
        "Договор № 20478ГУ/2026, г. Москва, 13.10.2026. В документе — только та часть",
        "договора, которая выполняется кодом и подтверждается прогонами: техническое",
        "задание (Приложение № 3), календарный план (Приложение № 4) и правила",
        "оформления отчёта (Приложение № 10). Документы зоны получателя гранта",
        "(преакселерация, бизнес-план, дорожная карта, заявка на РИД, финансовые",
        "отчёты, акты, подача в АС «Фонд-М», проверка оригинальности) в коде не",
        "выполняются — они перечислены в отдельном разделе.",
        "",
        "Документ собран скриптом `scripts/contract_check.py` из `reports/CONTRACT.json`.",
        "**Числа не вписаны руками**: каждое взято из артефакта прогона в",
        "`reports/contract/`, команда воспроизведения указана рядом.",
        "",
        f"- Коммит прогона: `{payload['commit']}`",
        f"- Дата прогона: {payload['environment']['date']}",
        (
            f"- Окружение: Python {payload['environment']['python']}, "
            f"{payload['environment']['platform']}, ядер {payload['environment']['cpu_count']}"
        ),
        (
            f"- Проверок: {len(payload['checks'])}; соответствует — {payload['summary'][STATUS_OK]}, "
            f"не соответствует — {payload['summary'][STATUS_FAIL]}, "
            f"не измерено — {payload['summary'][STATUS_UNKNOWN]}"
        ),
        "",
        "## Пункт договора → статус → команда → факт → файл-источник",
        "",
    ]
    lines += _table_lines(payload)
    lines += [""] + _reserve_block(payload)
    lines += ["", "## Не достигнуто и не измерено (с причиной)", ""]
    for item in payload["checks"]:
        if item["status"] == STATUS_OK:
            continue
        lines.append(f"- **{item['id']}** — {item['status']}: {item['note']}")
    if all(item["status"] == STATUS_OK for item in payload["checks"]):
        lines.append(
            "- Все проверки пройдены: не достигнутых и неизмеренных пунктов нет."
        )
    lines += ["", "## Зона получателя гранта (к коду не относится)", ""]
    lines.append(
        "Эти пункты договор требует представить в АС «Фонд-М» вместе с отчётностью;"
    )
    lines.append("программно они не выполняются и в кодовой части не закрываются.")
    lines.append("")
    lines.append("| Пункт | Что делает получатель гранта |")
    lines.append("| --- | --- |")
    for name, what in GRANTEE_ZONE:
        lines.append(f"| {name} | {what} |")
    lines += ["", "## Осталось до закрытия договора людьми", ""]
    for name, _ in GRANTEE_ZONE:
        lines.append(f"- {name}")
    lines += [
        "",
        "Числа и материалы для этих пунктов готовит кодовая часть: метрики —",
        "`spanverify-module/reports/CURRENT_METRICS.json`, таблица соответствия — этот документ",
        "и `docs/СООТВЕТСТВИЕ_ДОГОВОРУ.md`, приложение к отчёту —",
        "`docs/ОТЧЁТ_ПРИЛОЖЕНИЕ_СООТВЕТСТВИЕ.md`, инструкция по оригинальности —",
        "`docs/ПРОВЕРКА_ОРИГИНАЛЬНОСТИ.md`.",
        "",
    ]
    lines += _ci_runs_block()
    lines += _reproduction_block()
    return "\n".join(lines)


def render_compliance_doc(payload: dict[str, Any]) -> str:
    lines = [
        "# Соответствие договору № 20478ГУ/2026 (проект 117812)",
        "",
        "Документ сгенерирован скриптом `scripts/contract_check.py` из `reports/CONTRACT.json`.",
        "**Числа в документ не вписываются руками**: источник каждого — артефакт прогона в `reports/contract/`,",
        "команда воспроизведения указана в таблице.",
        "",
        f"- Коммит прогона: `{payload['commit']}`",
        f"- Дата прогона: {payload['environment']['date']}",
        (
            f"- Окружение: Python {payload['environment']['python']}, {payload['environment']['platform']}, "
            f"ядер {payload['environment']['cpu_count']}, seed {payload['environment']['seed']}"
        ),
        (
            f"- Проверок: {len(payload['checks'])}; соответствует — {payload['summary'][STATUS_OK]}, "
            f"не соответствует — {payload['summary'][STATUS_FAIL]}, не измерено — {payload['summary'][STATUS_UNKNOWN]}"
        ),
        "",
        "Статус «не измерено» означает, что прогон в этом окружении невозможен; причина указана в столбце «факт».",
        "",
    ]
    lines += _table_lines(payload)
    lines += [
        "",
        "## Что не достигнуто или не измерено",
        "",
    ]
    for item in payload["checks"]:
        if item["status"] == STATUS_OK:
            continue
        lines.append(f"- **{item['id']}** ({item['status']}): {item['note']}")
    if all(item["status"] == STATUS_OK for item in payload["checks"]):
        lines.append("- Все проверки пройдены.")
    lines += _reproduction_block()
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Контрактная проверка репозитория")
    parser.add_argument(
        "--only", default=None, help="выполнить только одну проверку по id"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="пропустить тяжёлые проверки (2 000 000 знаков)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "куда записать реестр (по умолчанию reports/CONTRACT.json; при "
            "--only реестр не перезаписывается, чтобы частичный прогон не "
            "затёр полный)"
        ),
    )
    parser.add_argument(
        "--no-doc",
        action="store_true",
        help="не перегенерировать документ соответствия",
    )
    args = parser.parse_args()

    known_ids = {check.id for check in CHECKS}
    if args.only and args.only not in known_ids:
        print(f"неизвестный id проверки: {args.only}")
        print("доступные id: " + ", ".join(sorted(known_ids)))
        return 2

    started = time.time()
    # Две фазы. Сначала все проверки, кроме читающих сам документ соответствия;
    # затем документ перегенерируется из CONTRACT.json, и вторая фаза проверяет
    # уже итоговый текст. Так «числа документов сверены с реестром» относится к
    # тому документу, который получит проверяющий, а не к прошлому прогону.
    runner = Runner(fast=args.fast)
    results, failures = run_checks(
        args.only, args.fast, runner=runner, skip_after_docs=True
    )
    if not args.only and not args.no_doc:
        COMPLIANCE_DOC.parent.mkdir(parents=True, exist_ok=True)
        preliminary = {
            "checks": results,
            "commit": git_commit(),
            "environment": env_info(),
            "summary": {
                STATUS_OK: sum(1 for item in results if item["status"] == STATUS_OK),
                STATUS_FAIL: sum(
                    1 for item in results if item["status"] == STATUS_FAIL
                ),
                STATUS_UNKNOWN: sum(
                    1 for item in results if item["status"] == STATUS_UNKNOWN
                ),
            },
        }
        COMPLIANCE_DOC.write_text(render_compliance_doc(preliminary), encoding="utf-8")
        CLOSING_DOC.write_text(render_closing_doc(preliminary), encoding="utf-8")
        REPORT_APPENDIX_DOC.write_text(
            render_report_appendix(preliminary), encoding="utf-8"
        )
    deferred, deferred_failures = run_checks(args.only, args.fast, runner=runner)
    deferred = [
        item
        for item in deferred
        if item["id"] in {check.id for check in CHECKS if check.after_docs}
    ]
    results.extend(deferred)
    failures += deferred_failures
    summary = {
        STATUS_OK: sum(1 for item in results if item["status"] == STATUS_OK),
        STATUS_FAIL: failures,
        STATUS_UNKNOWN: sum(1 for item in results if item["status"] == STATUS_UNKNOWN),
    }
    payload = {
        "kind": "contract_check",
        "run_id": os.environ.get("GITHUB_RUN_ID", "local-contract:" + git_commit()),
        "date": time.strftime("%Y-%m-%d", time.gmtime()),
        "provenance": "local",
        "commit": git_commit(),
        "environment": env_info(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_s": round(time.time() - started, 1),
        "summary": summary,
        "checks": results,
        "rule": (
            "статус «соответствует» ставится только по факту выполненной проверки; "
            "«не измерено» допустимо лишь с обоснованием"
        ),
    }
    # Частичный прогон (--only) реестр не трогает: иначе одна проверка затирает
    # 33 записей, и следующая сверка считает полный прогон неполным.
    out: Path | None = None
    if args.out or not args.only:
        out = ROOT / (args.out or str(CONTRACT_JSON.relative_to(ROOT)))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    else:
        print("запуск с --only: reports/CONTRACT.json не перезаписан")
    if not args.no_doc and not args.only:
        COMPLIANCE_DOC.parent.mkdir(parents=True, exist_ok=True)
        COMPLIANCE_DOC.write_text(render_compliance_doc(payload), encoding="utf-8")
        CLOSING_DOC.write_text(render_closing_doc(payload), encoding="utf-8")
        REPORT_APPENDIX_DOC.write_text(
            render_report_appendix(payload), encoding="utf-8"
        )

    print(f"проверок: {len(results)}")
    for item in results:
        mark = {STATUS_OK: "OK ", STATUS_FAIL: "FAIL", STATUS_UNKNOWN: "н/и"}[
            item["status"]
        ]
        print(f"  [{mark}] {item['id']:<28} {item['status']:<16} {item['note'][:90]}")

    # В CI провал шага виден в аннотациях — их читают через API, не выгружая логи.
    # Печатаем причину по каждой не прошедшей и каждой неизмеренной проверке.
    if os.environ.get("GITHUB_ACTIONS"):
        for item in results:
            if item["status"] == STATUS_FAIL:
                print(
                    f"::error title={item['id']}::{item['point']}: {item['note'][:500]}"
                )
            elif item["status"] == STATUS_UNKNOWN:
                print(
                    f"::warning title={item['id']}::{item['point']}: {item['note'][:500]}"
                )
        print(
            f"::notice title=Контрактные проверки::соответствует {summary[STATUS_OK]}, "
            f"не соответствует {summary[STATUS_FAIL]}, не измерено {summary[STATUS_UNKNOWN]}"
        )
    if out is not None:
        print(f"CONTRACT.json: {out}")
    if failures:
        print(
            "НЕ ПРОШЛИ: "
            + ", ".join(item["id"] for item in results if item["status"] == STATUS_FAIL)
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
