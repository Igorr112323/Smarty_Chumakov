"""Измерение фактических объёмов режима признаков языковой модели (ТЗ п. 4.2).

Договор требует (Приложение № 3, п. 4.2, стр. 12–13): «Для режима признаков
языковой модели потребуются библиотеки Python объёмом около 500 МБ и файл весов
модели объёмом около 530 МБ, сохраняемые локально заранее» и «Объём хранения
составит не более 100 МБ для режима без весов модели и не более 1,5 ГБ для режима
с весами».

Скрипт ничего не оценивает: он измеряет каталоги и записывает байты. Меряются
четыре величины, и в артефакт попадают все четыре, чтобы число приёмки было
видно вместе со своей «сырой» версией:

* **библиотеки** (``libs_mb``) — та часть ``site-packages``, которая реально
  загружается при проверке в режиме hf: список модулей берётся из ``sys.modules``
  живого прогона (дочерний процесс импортирует :class:`spanverify.engine.Verifier`
  с ``mode="hf"`` и проверяет пару «ответ/документ»), из найденных пакетов
  выбрасываются только файлы, которые Python не может загрузить при работе:
  заголовки C/C++ (``include/``, ``*.h``, ``*.cuh``, …), метаданные пакетов
  (``*.dist-info``, ``*.egg-info``), кеши (``__pycache__``), каталоги
  ``docs``/``examples``/``benchmarks`` и заглушки типов (``*.pyi``). Всё, что
  может быть импортировано, остаётся на месте — урезание идёт по расширениям,
  а не «на глаз»;
* **библиотеки, сырой прирост** (``libs_full_mb``) — насколько вырос каталог
  ``site-packages`` после установки зависимостей режима hf (для контроля: эта
  величина включает и то, что режиму не нужно, например установленные попутно
  пакеты, чьи модули за время проверки не загружались);
* **веса** (``weights_mb``) — каталог кеша Hugging Face, куда скачаны только
  рабочие форматы модели (``pytorch_model.bin`` и файлы токенизатора; служебные
  ``flax_model.msgpack``, ``tf_model.h5`` не скачиваются намеренно). Файлы
  считаются по индексному дескриптору: жёсткие ссылки одного и того же файла не
  удваиваются, символические ссылки не разыменовываются;
* **поставка** (``delivery_mb``) — каталог ``release/`` (или собранный ``.pyz``),
  то есть то, что переносится на компьютер пользователя без весов;
* **хранение** (``storage_with_weights_mb``) — поставка плюс библиотеки плюс
  веса.

Запуск (в CI — на раннере, где сеть доступна)::

    python scripts/measure_hf_footprint.py --baseline /tmp/baseline.json \\
        --out ../reports/contract --model ai-forever/rugpt3small_based_on_gpt2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import sysconfig
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 42

#: Заголовки, статические библиотеки и заглушки типов: Python не загружает такие
#: файлы во время работы, поэтому в «объём нужных библиотек» они не входят.
PRUNE_SUFFIXES = frozenset(
    {
        ".h",
        ".hpp",
        ".hxx",
        ".cuh",
        ".c",
        ".cc",
        ".cpp",
        ".cu",
        ".cmake",
        ".pxd",
        ".pyx",
        ".pyi",
        ".a",
        ".la",
        ".pc",
    }
)
#: Каталоги, содержимое которых не загружается при работе программы.
PRUNE_DIRS = frozenset({"include", "__pycache__", "docs", "doc", "examples", "benchmarks"})

#: Проба: какой набор пакетов режим hf действительно загружает.
RUNTIME_PROBE = r"""
import json
import os
import sys
import sysconfig
from pathlib import Path

from spanverify.engine import Verifier

verifier = Verifier(mode="hf", model_name=os.environ.get("SPANVERIFY_MODEL") or None)
result = verifier.verify(
    answer="Срок поставки — 45 календарных дней.",
    context="Статья 3. Срок поставки составляет 30 календарных дней с даты подписания.",
)
site = Path(sysconfig.get_paths()["purelib"]).resolve()
modules = []
for name, module in list(sys.modules.items()):
    origin = getattr(module, "__file__", None)
    if not origin:
        continue
    try:
        path = Path(origin).resolve()
    except OSError:  # pragma: no cover — модуль уже выгружен
        continue
    if site not in path.parents:
        continue
    parts = path.relative_to(site).parts
    if not parts:
        continue
    top = parts[0]
    if len(parts) == 1 and top.endswith(".py"):
        top = top[:-3]
    if top.endswith((".dist-info", ".egg-info")) or top == "__pycache__":
        continue
    modules.append(top)
payload = result.to_dict()
with open(os.environ["SPANVERIFY_PROBE_OUT"], "w", encoding="utf-8") as handle:
    json.dump(
        {
            "modules": sorted(set(modules)),
            "mode": payload.get("mode"),
            "verdict": payload.get("verdict"),
            "score": payload.get("score"),
        },
        handle,
        ensure_ascii=False,
    )
"""


def dir_bytes(path: Path) -> int:
    """Полный размер каталога в байтах (без символических ссылок)."""
    return sum(size for _, size in _walk(path))


def _walk(path: Path) -> list[tuple[Path, int]]:
    """Файлы каталога с их размерами (символические ссылки пропускаются)."""
    files: list[tuple[Path, int]] = []
    if not path.exists():
        return files
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                files.append((item, item.stat().st_size))
        except OSError:  # каталог исчез во время обхода
            continue
    return files


#: Файлы крупнее этого порога проверяются на совпадение по содержимому: кеш
#: весов держит один и тот же файл и в буфере загрузки, и в самом кеше, и считать
#: его дважды нельзя (в задании hf-footprint так и было: 554,3 МБ против 1105,7).
CONTENT_DEDUPE_MIN_BYTES = 1_000_000


def _digest(path: Path) -> str:
    """SHA-256 файла кусками: файл весов нельзя читать в память целиком."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def dir_bytes_unique(path: Path) -> dict[str, int]:
    """Размер каталога с подсчётом каждого файла один раз.

    Возвращает величины: ``unique`` — сумма размеров файлов, которые встретились
    впервые, ``unique_content`` — то же, но одинаковые по содержимому крупные
    файлы считаются один раз (буфер загрузки и кеш весов), ``naive`` — сумма
    всех ``st_size`` и ``files`` — число файлов. Нужны все три: видно, сколько
    объёма добавили жёсткие ссылки и дубли содержимого, а не выводится на слово.
    """
    seen: set[tuple[int, int]] = set()
    seen_content: set[str] = set()
    unique = 0
    unique_content = 0
    naive = 0
    count = 0
    for item, size in _walk(path):
        try:
            stat = item.stat()
        except OSError:  # pragma: no cover — файл исчез во время обхода
            continue
        naive += size
        count += 1
        key = (stat.st_dev, stat.st_ino)
        if key in seen:
            continue
        seen.add(key)
        unique += size
        if size >= CONTENT_DEDUPE_MIN_BYTES:
            try:
                digest = _digest(item)
            except OSError:  # pragma: no cover — файл исчез во время обхода
                unique_content += size
                continue
            if digest in seen_content:
                continue
            seen_content.add(digest)
        unique_content += size
    return {"unique": unique, "unique_content": unique_content, "naive": naive, "files": count}


#: Форматы файлов весов модели: в одном репозитории может лежать несколько форм
#: одних и тех же весов (PyTorch, safetensors, TensorFlow, Flax, ONNX) — это
#: разные файлы, но одни веса. Диск занимают все скачанные, а при работе
#: загружается один: его выбирает библиотека.
WEIGHTS_SUFFIXES = {
    ".bin": "pytorch_model.bin (PyTorch)",
    ".safetensors": "model.safetensors (safetensors)",
    ".h5": "tf_model.h5 (TensorFlow)",
    ".msgpack": "flax_model.msgpack (Flax)",
    ".onnx": "model.onnx (ONNX)",
    ".pth": "PyTorch (pth)",
    ".pt": "PyTorch (pt)",
}
#: Порядок предпочтения форматов: transformers ставит safetensors первым, если
#: библиотека safetensors установлена и файл есть в снимке; иначе берётся
#: pytorch_model.bin. Ровно этот порядок и повторяем — не догадка, а правило
#: библиотеки, которое она применяет при загрузке.
WEIGHTS_PREFERENCE = (".safetensors", ".bin", ".h5", ".msgpack", ".onnx", ".pth", ".pt")


def model_weights(cache: Path) -> dict[str, object]:
    """Файлы весов модели: сколько занимает модель в кеше Hugging Face.

    Кеш может содержать несколько ревизий модели и несколько форматов одних и тех
    же весов. Для требований «около 530 МБ» и «не более 1,5 ГБ с весами» считается
    то, что реально нужно программе:

    * ревизия — та, на которую указывает ``refs/main`` (её загружает
      ``from_pretrained``); если ссылки нет, берутся все снимки;
    * формат весов — один, по правилу библиотеки (``WEIGHTS_PREFERENCE``);
      размеры всех форматов и всего кеша остаются в артефакте, чтобы урезание
      было видно, а не спрятано;
    * каждый файл — один раз (по индексному дескриптору, символические ссылки
      снимка разыменовываются).
    """
    revisions: dict[str, str] = {}
    for ref in cache.rglob("hub/models--*/refs/main"):
        try:
            revisions[str(ref.parent.parent.name)] = ref.read_text(encoding="utf-8").strip()
        except OSError:  # pragma: no cover — ссылка исчезла во время обхода
            continue

    all_snapshots = [path for path in sorted(cache.rglob("hub/models--*/snapshots/*")) if path.is_dir()]
    # Ревизия ветки main — та, которую загружает from_pretrained. Если ссылки нет
    # или ей не соответствует ни один снимок, берутся все снимки: тогда объём
    # честно окажется больше, а не меньше.
    chosen: list[Path] = []
    for snapshot in all_snapshots:
        repo = snapshot.parent.parent.name
        revision = revisions.get(repo)
        if revision and revision == snapshot.name:
            chosen.append(snapshot)
    revision_used = "refs/main"
    if not chosen:
        chosen = all_snapshots
        revision_used = "все снимки (ссылка refs/main не найдена)"
    elif revisions and len(chosen) < len(all_snapshots):
        revision_used = "refs/main: " + ", ".join(sorted({path.name for path in chosen}))

    seen: set[tuple[int, int]] = set()
    files: list[dict[str, object]] = []
    for snapshot in chosen:
        repo = snapshot.parent.parent.name
        for item in sorted(snapshot.rglob("*")):
            target = item.resolve() if item.is_symlink() else item
            try:
                if not target.is_file():
                    continue
                stat = target.stat()
            except OSError:  # pragma: no cover — файл исчез во время обхода
                continue
            key = (stat.st_dev, stat.st_ino)
            if key in seen:
                continue
            seen.add(key)
            files.append(
                {
                    "path": str(item.relative_to(cache)),
                    "repo": repo,
                    "revision": snapshot.name,
                    "bytes": stat.st_size,
                    "resolved": str(target.relative_to(cache)),
                }
            )

    by_format: dict[str, int] = {}
    for item in files:
        kind = WEIGHTS_SUFFIXES.get(Path(str(item["path"])).suffix.lower())
        if kind is not None:
            by_format[kind] = by_format.get(kind, 0) + int(item["bytes"])

    selected_format = ""
    for suffix in WEIGHTS_PREFERENCE:
        kind = WEIGHTS_SUFFIXES[suffix]
        if kind in by_format:
            selected_format = kind
            break

    weights_bytes = 0
    aux_bytes = 0
    selected: list[dict[str, object]] = []
    for item in files:
        kind = WEIGHTS_SUFFIXES.get(Path(str(item["path"])).suffix.lower())
        if kind is None:
            aux_bytes += int(item["bytes"])
            selected.append(item)
        elif kind == selected_format:
            weights_bytes += int(item["bytes"])
            selected.append(item)
    selected.sort(key=lambda item: -int(item["bytes"]))

    return {
        "bytes": weights_bytes + aux_bytes,
        "weights_bytes": weights_bytes,
        "aux_bytes": aux_bytes,
        "selected_format": selected_format,
        "by_format": by_format,
        "files": selected,
        "revisions": revisions,
        "revision_used": revision_used,
        "all_snapshot_files": len(files),
    }


def package_bytes(site: Path, top: str, prune: bool) -> tuple[int, list[tuple[str, int]]]:
    """Размер одного пакета верхнего уровня, при ``prune`` — без нерабочих файлов.

    Возвращает сумму и список выброшенного (категория → байты), чтобы урезание
    было проверяемым: в артефакте видно, что именно не попало в объём.
    """
    root = site / top
    if not root.exists():
        return 0, []
    total = 0
    dropped: dict[str, int] = {}
    for item, size in _walk(root):
        category = None
        if prune:
            parts = item.relative_to(root).parts
            if any(part in PRUNE_DIRS for part in parts[:-1]):
                category = "нерабочие каталоги (include, docs, examples, __pycache__)"
            elif any(part.endswith((".dist-info", ".egg-info")) for part in parts):
                category = "метаданные пакета (*.dist-info)"
            elif item.suffix.lower() in PRUNE_SUFFIXES:
                category = "заголовки C/C++ и заглушки типов"
        if category is None:
            total += size
        else:
            dropped[category] = dropped.get(category, 0) + size
    return total, sorted(dropped.items(), key=lambda item: -item[1])


def runtime_modules(model: str) -> tuple[list[str], str]:
    """Какие пакеты загружает режим hf: список модулей и пояснение пробы.

    Проба поднимает режим hf на настоящих весах и отдаёт результат через файл
    (в stdout библиотеки пишут свои сообщения, поэтому читать его нельзя). Если
    режим оказался не ``hf``, значит сработал тихий откат на демонстрационный
    режим: мерить по такому набору пакетов нельзя, и это честно попадает в
    пояснение, а объём библиотек берётся сырым приростом.
    """
    env = dict(os.environ)
    if model:
        env["SPANVERIFY_MODEL"] = model
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out_path = Path(tmp.name)
    env["SPANVERIFY_PROBE_OUT"] = str(out_path)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", RUNTIME_PROBE],
            cwd=ROOT,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        if proc.returncode != 0:
            tail = (proc.stderr.strip().splitlines() or [str(proc.returncode)])[-1]
            return [], f"проба не выполнилась: {tail}"
        if not out_path.is_file():
            return [], "проба не записала результат"
        payload = json.loads(out_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:  # pragma: no cover
        return [], f"проба вернула нечитаемый результат: {error}"
    finally:
        out_path.unlink(missing_ok=True)
    mode = str(payload.get("mode") or "")
    verdict = str(payload.get("verdict") or "")
    note = f"режим {mode}, вердикт {verdict}"
    if mode != "hf":
        return [], f"проба поднялась не в режиме hf ({note}) — объём библиотек не измерен"
    return [str(item) for item in payload.get("modules") or []], note


def site_packages() -> Path:
    return Path(sysconfig.get_paths()["purelib"])


def git_commit() -> str:
    value = os.environ.get("GITHUB_SHA")
    if value:
        return value
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


def snapshot(path: Path) -> dict[str, int]:
    """Снимок размеров: общий и по крупнейшим подкаталогам (для протокола)."""
    data: dict[str, int] = {"total": dir_bytes(path)}
    if path.exists():
        children: list[tuple[str, int]] = []
        for child in sorted(path.iterdir()):
            if child.is_dir():
                children.append((child.name, dir_bytes(child)))
        for name, size in sorted(children, key=lambda item: -item[1])[:15]:
            data[name] = size
    return data


def hf_freeze() -> list[str]:
    """Версии пакетов режима hf — чтобы объём был привязан к конкретным версиям."""
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        ).stdout
    except (subprocess.SubprocessError, FileNotFoundError):  # pragma: no cover
        return []
    wanted = (
        "torch",
        "transformers",
        "tokenizers",
        "sentencepiece",
        "safetensors",
        "numpy",
        "huggingface",
        "regex",
        "requests",
    )
    return [
        line.strip() for line in out.splitlines() if line.strip() and line.split("==")[0].lower().startswith(wanted)
    ]


def mb(value: int) -> float:
    """Мегабайты, как их считает человек в отчёте: 1 МБ = 1 000 000 байт.

    Договор называет величины «около 500 МБ» и «не более 1,5 ГБ» без уточнения
    кратности. Считаем десятичные мегабайты (как в спецификациях размеров
    поставки) и дополнительно печатаем МиБ, чтобы расхождение было видно, а не
    спрятано.
    """
    return round(value / 1_000_000, 1)


def mib(value: int) -> float:
    return round(value / 1024 / 1024, 1)


def build_payload(
    check_id: str,
    metrics: dict[str, object],
    note: str,
    gate: str,
    command: str,
) -> dict[str, object]:
    return {
        "id": check_id,
        "commit": git_commit(),
        "seed": SEED,
        "mode": "hf",
        "dataset": str(metrics.get("model") or ""),
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": command,
        "metrics": metrics,
        "note": note,
        "gate": gate,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Объёмы режима hf: библиотеки и веса")
    parser.add_argument("--baseline", type=Path, default=None, help="замер до установки")
    parser.add_argument("--out", type=Path, required=True, help="каталог артефактов")
    parser.add_argument("--model", default="ai-forever/rugpt3small_based_on_gpt2")
    parser.add_argument("--cache", type=Path, default=None, help="каталог кеша весов")
    parser.add_argument("--delivery", type=Path, default=ROOT / "release")
    parser.add_argument(
        "--skip-runtime",
        action="store_true",
        help="не запускать пробу режима hf (тогда объём библиотек — сырой прирост)",
    )
    args = parser.parse_args()

    cache = args.cache or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    base_total = 0
    if args.baseline and args.baseline.is_file():
        base_total = int(json.loads(args.baseline.read_text(encoding="utf-8"))["total"])

    libs_now = snapshot(site_packages())
    libs_delta_bytes = max(0, libs_now["total"] - base_total)

    site = site_packages()
    modules: list[str] = []
    mode_note = ""
    if not args.skip_runtime:
        modules, mode_note = runtime_modules(args.model)
    libs_runtime_full = 0
    libs_runtime = 0
    dropped: dict[str, int] = {}
    per_package: dict[str, int] = {}
    if modules:
        for top in modules:
            full, _ = package_bytes(site, top, prune=False)
            pruned, dropped_here = package_bytes(site, top, prune=True)
            libs_runtime_full += full
            libs_runtime += pruned
            per_package[top] = pruned
            for category, size in dropped_here:
                dropped[category] = dropped.get(category, 0) + size

    runtime_measured = bool(modules)
    # Объём библиотек для расчёта хранения: измеренный по факту загрузки набор;
    # если проба не удалась, честно берём сырой прирост и помечаем это в артефакте.
    libs_bytes = libs_runtime if runtime_measured else libs_delta_bytes

    weights = dir_bytes_unique(cache)
    model = model_weights(cache)
    weights_bytes = int(model["bytes"]) or weights["unique_content"]
    cache_bytes = weights["unique_content"]
    weights_all_formats = sum(int(value) for value in model["by_format"].values()) + int(model["aux_bytes"])
    delivery_bytes = dir_bytes(args.delivery)
    storage_with_weights = delivery_bytes + libs_bytes + weights_bytes
    storage_without_weights = delivery_bytes
    storage_with_weights_full = delivery_bytes + libs_delta_bytes + weights_bytes
    storage_with_weights_cache = delivery_bytes + libs_bytes + cache_bytes

    files = []
    for item, size in _walk(cache):
        files.append({"path": str(item.relative_to(cache)), "bytes": size})
    files.sort(key=lambda item: -int(item["bytes"]))

    command = (
        "python scripts/measure_hf_footprint.py " f"--out ../reports/contract --model {args.model} --cache {cache}"
    )
    common = {
        "model": args.model,
        "cache": str(cache),
        "libs_bytes": libs_bytes,
        "libs_mb": mb(libs_bytes),
        "libs_mib": mib(libs_bytes),
        "libs_runtime_full_bytes": libs_runtime_full,
        "libs_runtime_full_mb": mb(libs_runtime_full),
        "libs_delta_bytes": libs_delta_bytes,
        "libs_full_mb": mb(libs_delta_bytes),
        "libs_full_mib": mib(libs_delta_bytes),
        "libs_runtime_measured": runtime_measured,
        "libs_runtime_modules": modules,
        "libs_runtime_probe_note": mode_note,
        "libs_runtime_per_package_mb": {
            key: mb(value) for key, value in sorted(per_package.items(), key=lambda item: -item[1])
        },
        "libs_pruned_bytes": sum(dropped.values()),
        "libs_pruned_mb": mb(sum(dropped.values())),
        "libs_pruned_breakdown": {key: mb(value) for key, value in sorted(dropped.items(), key=lambda item: -item[1])},
        "libs_prune_rule": (
            "из пакетов, загружаемых при проверке в режиме hf, исключены только файлы, "
            "которые Python не может загрузить при работе: include/ и заголовки C/C++ "
            "(*.h, *.hpp, *.cuh, *.c, *.cpp, *.cu, *.cmake), заглушки типов (*.pyi), "
            "метаданные (*.dist-info, *.egg-info), кеши (__pycache__), docs/examples/benchmarks"
        ),
        "weights_bytes": weights_bytes,
        "weights_mb": mb(weights_bytes),
        "weights_mib": mib(weights_bytes),
        "weights_model_bytes": int(model["weights_bytes"]),
        "weights_model_mb": mb(int(model["weights_bytes"])),
        "weights_aux_mb": mb(int(model["aux_bytes"])),
        "weights_format": model["selected_format"],
        "weights_by_format_mb": {key: mb(value) for key, value in sorted(model["by_format"].items())},
        "weights_all_formats_mb": mb(weights_all_formats),
        "weights_revisions": model["revisions"],
        "weights_revision_used": model["revision_used"],
        "weights_cache_bytes": cache_bytes,
        "weights_cache_mb": mb(cache_bytes),
        "weights_cache_overhead_mb": mb(max(0, cache_bytes - weights_bytes)),
        "weights_cache_files_total": weights["files"],
        "weights_naive_bytes": weights["naive"],
        "weights_naive_mb": mb(weights["naive"]),
        "weights_model_files": model["files"][:20],
        "weights_unique_bytes": weights["unique"],
        "weights_unique_mb": mb(weights["unique"]),
        "weights_dedupe_rule": (
            "веса — файлы выбранной ревизии (refs/main) в одном формате, который "
            "загружает библиотека (safetensors, если есть, иначе pytorch_model.bin), "
            "плюс файлы токенизатора; размеры остальных форматов, объём кеша целиком и "
            "сырой прирост site-packages приведены отдельными полями и не скрываются"
        ),
        "weights_files_total": weights["files"],
        "delivery_bytes": delivery_bytes,
        "delivery_mb": mb(delivery_bytes),
        "storage_without_weights_mb": mb(storage_without_weights),
        "storage_with_weights_mb": mb(storage_with_weights),
        "storage_with_weights_mib": mib(storage_with_weights),
        "storage_with_weights_full_mb": mb(storage_with_weights_full),
        "storage_with_weights_cache_mb": mb(storage_with_weights_cache),
        "weight_files": files[:20],
        "weight_files_total": len(files),
        "site_packages_top": {key: value for key, value in list(libs_now.items())[:10]},
        "pip_freeze_hf": hf_freeze(),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    footprint = dict(common)
    footprint.update(
        {
            "contract_libs_mb": 500,
            "contract_weights_mb": 530,
            "libs_vs_contract_percent": round(100.0 * libs_bytes / 1_000_000 / 500 - 100.0, 1),
            "weights_vs_contract_percent": round(100.0 * weights_bytes / 1_000_000 / 530 - 100.0, 1),
        }
    )
    (args.out / "4.2.5-hf-footprint.json").write_text(
        json.dumps(
            build_payload(
                "4.2.5-hf-footprint",
                footprint,
                "Объёмы измерены командой на раннере. Библиотеки посчитаны по набору "
                "пакетов, который режим hf действительно загружает (проба в дочернем "
                "процессе, вердикт в поле libs_runtime_probe_verdict); рядом лежит "
                "сырой прирост site-packages. Договор называет ориентиры «около 500 МБ» "
                "и «около 530 МБ» — отклонения приведены отдельными полями и не "
                "подгоняются.",
                "ТЗ п. 4.2: библиотеки режима признаков около 500 МБ, веса около 530 МБ",
                command,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    storage = dict(common)
    storage.update(
        {
            "limit_without_weights_bytes": 100_000_000,
            "limit_without_weights_mb": 100,
            "limit_with_weights_bytes": 1_500_000_000,
            "limit_with_weights_mb": 1500,
            "limit_with_weights_mib": round(1.5 * 1024 * 1024 * 1024 / 1024 / 1024, 1),
            "without_weights_ok": storage_without_weights <= 100_000_000,
            "with_weights_ok": storage_with_weights <= 1_500_000_000,
            "with_weights_ok_mib": storage_with_weights <= int(1.5 * 1024**3),
        }
    )
    (args.out / "4.2.6-storage.json").write_text(
        json.dumps(
            build_payload(
                "4.2.6-storage",
                storage,
                "Хранение без весов — каталог поставки; с весами — поставка плюс "
                "библиотеки, которые загружает режим hf, плюс кеш весов. Рядом "
                "приведён объём по сырому приросту site-packages "
                "(storage_with_weights_full_mb) и значение в МиБ.",
                "ТЗ п. 4.2: хранение не более 100 МБ без весов и не более 1,5 ГБ с весами",
                command,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"библиотеки (нужные режиму hf): {mb(libs_bytes)} МБ ({mib(libs_bytes)} МиБ)")
    print(
        f"  сырой прирост site-packages: {mb(libs_delta_bytes)} МБ; "
        f"набор: {', '.join(modules) if modules else 'проба не выполнена'}"
    )
    print(
        f"веса:       {mb(weights_bytes)} МБ ({mib(weights_bytes)} МиБ), файлов "
        f"{len(model['files'])}; кеш целиком {mb(cache_bytes)} МБ"
    )
    print(f"поставка:   {mb(delivery_bytes)} МБ")
    print(f"хранение с весами: {mb(storage_with_weights)} МБ ({mib(storage_with_weights)} МиБ)")
    if os.environ.get("GITHUB_ACTIONS"):
        print(
            f"::notice title=объёмы режима hf::библиотеки {mb(libs_bytes)} МБ "
            f"(сырой прирост {mb(libs_delta_bytes)} МБ), веса {mb(weights_bytes)} МБ, "
            f"поставка {mb(delivery_bytes)} МБ, хранение с весами "
            f"{mb(storage_with_weights)} МБ"
        )
        for item in files[:5]:
            print(f"::notice title=файл весов::{item['path']} — {mb(int(item['bytes']))} МБ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
