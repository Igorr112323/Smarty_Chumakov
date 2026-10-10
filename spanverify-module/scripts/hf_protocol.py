"""Протокол измерения режима hf: заморозка, базовая линия, experiments, финал.

Назначение файла одно: чтобы каждое число режима `hf` на отложенной по
документам выборке было получено по протоколу, а не «посмотрели и подправили».
Определения метрик — `docs/METRIC_SPEC.md`; метрики считаются **функциями
продукта** (`spanverify.engine._token_metrics`, `_answer_metrics`, `_span_f1`),
а не копией формул: расхождение определений исследования и продукта было бы
самым дорогим из возможных ошибок.

Подкоманды:

* ``freeze``   — аудит разбиения (sha256 файлов, списки документов, отсутствие
  пересечений) и фиксация конфигурации: ``reports/hf_protocol/freeze.json``.
  Запускается ДО любых чисел на val/test.
* ``baseline`` — прогон текущего кода в режиме hf на val (A1 и A3) без изменений:
  ``reports/hf_baseline/manifest.json`` + ``predictions_val.jsonl``.
* ``sweep``    — не более 10 конфигураций на val по сетке из JSON-файла; отбор по
  ``token_f1`` при ``token_fpr ≤ 0,40``; порог — только из CV по train.
* ``final``    — ровно один запуск на test с ``--seeds`` (обучение на train[+val],
  test в подборе не участвует): ``reports/hf_final/manifest.json`` и
  ``predictions_test.jsonl.gz`` со строками на каждый токен каждого seed'а.
* ``verify-manifest`` — пересчитать метрики из ``predictions_*`` и сверить с
  манифестом (то же делает ``tests/test_reported_metrics.py``, допуск 1e-6).

Работает на кеше предпосчитанных признаков (``scripts/precompute_features.py
--grid``): прямой проход по модели стоит секунды на пару, а перебор
конфигураций — миллисекунды, поэтому весь sweep прогоняется на кеше и в CI, и
локально, и результаты совпадают побайтово.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spanverify.core import tokenize_with_offsets  # noqa: E402
from spanverify.dataset import read_pairs  # noqa: E402
from spanverify.engine import _answer_metrics, _span_f1, _token_metrics  # noqa: E402
from spanverify.features import is_scored_token  # noqa: E402
from spanverify.hf_grid import GRID_CACHE_FORMAT, GRID_FEATURE_NAMES  # noqa: E402
from spanverify.logreg import standardize, train_logreg  # noqa: E402

MAX_EXPERIMENTS = 10
CRITERION_F1 = 0.60
CRITERION_FPR = 0.40
TOLERANCE = 1e-6
PREDICTIONS_FIELDS = ("id", "level", "seed", "token", "start", "end", "gold", "score", "pred")
#: Списки документов пишутся в файл заморозки целиком, пока они помещаются:
#: «разбиение по документам» должно проверяемым, а не выводимым из доверия.
DOCUMENTS_LIST_LIMIT = 4000
#: Признаки базовой линии — те же три, что у продукта (энтропия, масса,
#: плотность), чтобы шаг 1 измерял текущий код, а не новый признак.
BASELINE_FEATURES = ("entropy_last", "mass_last", "density_k1_last")


# ------------------------------------------------------------------- утилиты


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    total = 0
    with opener(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            total += 1
    return total


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def document_of(record: dict[str, Any]) -> str:
    """Идентификатор документа (группа разбиения): из meta, иначе из контекста.

    Группа нужна двухместная: и для групповой CV на train, и для теста на утечку.
    Пар без идентификатора документа не бывает — для демо-корпуса, где ``meta``
    пуст, роль идентификатора играет хеш контекста: два ответа к одному документу
    не могут попасть в разные части.
    """
    meta = record.get("meta") or {}
    for key in ("group", "doc_id", "document", "source_id"):
        value = meta.get(key)
        if value:
            return str(value)
    for key in ("doc_id", "group"):
        value = record.get(key)
        if value:
            return str(value)
    context = record.get("context")
    text = context if isinstance(context, str) else json.dumps(context, ensure_ascii=False, sort_keys=True)
    return "context-sha:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_split(path: Path) -> list[dict[str, Any]]:
    return list(read_pairs(path))


def val_path_of(directory: Path) -> Path:
    dev = directory / "dev.jsonl"
    return dev if dev.is_file() else directory / "val.jsonl"


def _round(value: Any) -> Any:
    return round(float(value), 6) if isinstance(value, (int, float)) and not isinstance(value, bool) else value


def _plain(numbers: dict[str, Any] | None) -> dict[str, Any]:
    return {key: _round(value) for key, value in (numbers or {}).items()}


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True, timeout=10
        ).stdout.strip()[:40]
    except Exception:  # noqa: BLE001 - вне git-дерева коммит просто неизвестен
        return ""


def hardware_manifest() -> dict[str, Any]:
    """Параметры машины: модель CPU, число процессоров, RAM, версии — по протоколу."""
    import platform

    info: dict[str, Any] = {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    try:
        lines = Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
        info["cpu_model"] = next((line.split(":", 1)[1].strip() for line in lines if line.startswith("model name")), "")
        info["cpu_processors"] = sum(1 for line in lines if line.startswith("processor"))
        info["cpu_mhz"] = next((line.split(":", 1)[1].strip() for line in lines if line.startswith("cpu MHz")), "")
    except OSError:
        info["cpu_model"] = "недоступно"
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
        total = next(line for line in meminfo if line.startswith("MemTotal"))
        kb = int(total.split()[1])
        info["ram_total_kb"] = kb
        info["ram_total_gb"] = round(kb / 1024 / 1024, 2)
    except (OSError, ValueError, StopIteration):
        info["ram_total_gb"] = None
    for module in ("torch", "transformers", "numpy"):
        try:
            package = __import__(module)
            info[f"{module}_version"] = str(getattr(package, "__version__", "без версии"))
        except ImportError:
            info[f"{module}_version"] = "не установлен"
    # Сырые вывода `lscpu` и `free -h`: по требованию протокола ограничения
    # «CPU ≤ 16 GB, без GPU» должны проверяться по тем же командам, что их
    # формулирует заявка, а не по производной от /proc/meminfo оценке.
    for key, command in (("lscpu", ("lscpu",)), ("free_h", ("free", "-h"))):
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
            text = (completed.stdout or "").strip()
            info[key] = text[:2000] if text else "недоступно"
        except (OSError, subprocess.SubprocessError):
            info[key] = "недоступно"
    return info


# --------------------------------------------------------- разметка на токены


def token_rows(record: dict[str, Any], arrays: dict[str, Sequence[float]] | None) -> list[dict[str, Any]]:
    """Строки «по одному на оцениваемый токен ответа» с gold-меткой.

    Gold берётся из ``labels`` пары ровно так же, как в ``Verifier.evaluate``:
    токен положительный, если пересекается с gold-фрагментом. Оцениваются только
    содержательные токены (``is_scored_token``) — иная выборка токенов была бы
    другой метрикой.
    """
    answer = record.get("answer", "")
    tokens = list(tokenize_with_offsets(answer))
    truth = spans_of(record)
    rows: list[dict[str, Any]] = []
    for index, token in enumerate(tokens):
        if not is_scored_token(token.text):
            continue
        features: dict[str, float] = {}
        if arrays:
            for name, values in arrays.items():
                if index < len(values):
                    features[name] = float(values[index])
        rows.append(
            {
                "id": str(record.get("id") or ""),
                "index": index,
                "token": token.text,
                "start": token.start,
                "end": token.end,
                "gold": 1 if any(token.start < end and token.end > start for start, end in truth) else 0,
                "spans": truth,
                "features": features,
            }
        )
    return rows


def spans_of(record: dict[str, Any]) -> list[tuple[int, int]]:
    return [(int(start), int(end)) for start, end, label in record.get("labels", []) if int(label) == 1]


def merge_spans(items: Sequence[dict[str, Any]], threshold: float, gap: int) -> list[tuple[int, int]]:
    """Метка токенов → интервалы: соседние помеченные токены склеиваются.

    Склейка — рычаг «агрегация» протокола: значение «10 (десяти) лет» модель
    часто видит как два кусочка, и без склейки фрагментный уровень штрафует за
    разбиение, а не за ошибку.
    """
    spans = [(int(item["start"]), int(item["end"])) for item in items if float(item["score"]) >= threshold]
    if not spans:
        return []
    spans.sort()
    merged = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


# ------------------------------------------------------------------- кеш сетки


def load_grid_cache(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    """Кеш v3: ``id пары → {'pair_id', 'arrays', 'tokens', 'meta'}``.

    Несовпадение формата — ошибка, а не тихая деградация: молча обучиться на
    признаках другого формата значит получить числа, которые невозможно
    воспроизвести.
    """
    cache: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            identifier = str(row.get("pair_id") or row.get("key") or "")
            if not identifier:
                continue
            format_value = int(row.get("grid_format", row.get("cache_format", 0)) or 0)
            if format_value != GRID_CACHE_FORMAT:
                raise SystemExit(
                    f"{path.name}: строка {identifier} в формате v{format_value}, нужен v{GRID_CACHE_FORMAT}. "
                    "Пересоберите кеш: python scripts/precompute_features.py --grid …"
                )
            cache[identifier] = row
    if not cache:
        raise SystemExit("кеш сетки пуст: укажите --grid-cache (результат precompute_features.py --grid)")
    return cache


def _cache_paths(directory: Path) -> list[Path]:
    if directory.is_file():
        return [directory]
    paths = sorted(
        [*directory.glob("grid_*.jsonl"), *directory.glob("grid_*.jsonl.gz"), *directory.glob("features_grid_*.jsonl")]
    )
    if not paths:
        raise SystemExit(f"в {directory} нет файлов кеша сетки (grid_*.jsonl[.gz])")
    return paths


def prepare(
    records: Sequence[dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    names: Sequence[str],
    *,
    require: bool = True,
) -> list[dict[str, Any]]:
    """Строки «токен → gold + признаки» для списка пар, только из кеша."""
    out: list[dict[str, Any]] = []
    missing = 0
    for record in records:
        row = cache.get(str(record.get("id") or ""))
        if row is None:
            missing += 1
            continue
        arrays = {name: list((row.get("arrays") or {}).get(name, [])) for name in names}
        for item in token_rows(record, arrays):
            item["doc"] = document_of(record)
            out.append(item)
    if missing and require:
        raise SystemExit(
            f"в кеше нет {missing} из {len(records)} пар: кеш собран по другой выборке "
            "(нужен precompute_features.py --grid на тех же файлах разбиения)"
        )
    return out


def feature_matrix(rows: Sequence[dict[str, Any]], names: Sequence[str]) -> list[list[float]]:
    return [[float(row["features"].get(name, 0.0)) for name in names] for row in rows]


def group_by_pair(rows: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["id"], []).append(row)
    return list(groups.values())


def smooth(groups: Sequence[Sequence[dict[str, Any]]], radius: int) -> None:
    """Сглаживание оценок скользящим окном по соседним токенам того же ответа."""
    if radius <= 0:
        return
    for items in groups:
        values = [float(item["score"]) for item in items]
        smoothed = [
            statistics.fmean(values[max(0, index - radius) : min(len(values), index + radius + 1)])
            for index in range(len(values))
        ]
        for item, value in zip(items, smoothed, strict=True):
            item["score"] = float(value)


# ------------------------------------------------- модели, пороги, групповая CV


@dataclass
class Fitted:
    """Обученная модель: логистическая регрессия или бустинг пней (свой, без sklearn)."""

    kind: str
    means: list[float]
    scales: list[float]
    intercept: float = 0.0
    weights: list[float] = field(default_factory=list)
    stumps: list[dict[str, Any]] = field(default_factory=list)

    def logits(self, rows: Sequence[Sequence[float]]) -> list[float]:
        scaled = [
            [(value - mean) / scale for value, mean, scale in zip(row, self.means, self.scales, strict=False)]
            for row in rows
        ]
        if self.kind == "logreg":
            return [
                self.intercept + sum(weight * value for weight, value in zip(self.weights, row, strict=False))
                for row in scaled
            ]
        out: list[float] = []
        for row in scaled:
            total = self.intercept
            for stump in self.stumps:
                feature = int(stump["feature"])
                value = row[feature] if feature < len(row) else 0.0
                total += float(stump["left"]) if value < float(stump["threshold"]) else float(stump["right"])
            out.append(total)
        return out

    def scores(self, rows: Sequence[Sequence[float]]) -> list[float]:
        return [1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit)))) for logit in self.logits(rows)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "means": [round(value, 8) for value in self.means],
            "scales": [round(value, 8) for value in self.scales],
            "weights": [round(value, 8) for value in self.weights],
            "intercept": round(self.intercept, 8),
            "stumps": self.stumps,
        }


def fit_model(rows: list[list[float]], labels: list[int], config: dict[str, Any], seed: int) -> Fitted:
    """Логистическая регрессия продукта или свой бустинг пней (рычаг «классификатор»).

    Линейный случай — ``spanverify.logreg.train_logreg`` без изменений: харнесс
    обязан измерять тот же классификатор, что работает в продукте, иначе «лучшая
    конфигурация на val» относилась бы к другой модели, чем та, что попадёт в
    поставку. Нелинейный случай — свой бустинг пней ниже (sklearn в поставку не
    входит), с одним приёмом: квантильные бины считаются один раз на обучение, а
    не на каждый раунд.
    """
    standardized, means, scales = standardize(rows)
    if str(config.get("classifier", "logreg")) == "boost":
        return _fit_boost(standardized, means, scales, labels, config)
    del seed
    model = train_logreg(
        standardized,
        labels,
        epochs=int(config.get("epochs", 200)),
        learning_rate=float(config.get("learning_rate", 0.5)),
        l2=float(config.get("l2", 1e-3)),
    )
    return Fitted("logreg", means, scales, float(model["bias"]), [float(value) for value in model["weights"]])


def _fit_boost(
    standardized: list[list[float]], means: list[float], scales: list[float], labels: list[int], config: dict[str, Any]
) -> Fitted:
    """Градиентный бустинг пней на логистическом лоссе.

    sklearn в поставку не входит (тяжёлая зависимость ради одного рычага), а
    нелинейность нужна: пень выбирает один признак и один порог, бустинг
    собирает из них аддитивную модель. Пороги кандидатов — квантили признака,
    значения листьев — сумма градиентов группы; приращение лосса считается
    стандартной формулой ``Σg²/n`` по группам, поэтому поиск лучшего пня — один
    проход по отсортированным значениям, а не перебор всех разбиений.
    """
    total = len(standardized)
    prior = sum(labels) / total if total else 0.5
    base = math.log(max(1e-6, prior) / max(1e-6, 1.0 - prior))
    if total < 20 or len(set(labels)) < 2:
        return Fitted("boost", means, scales, base)
    rounds = int(config.get("rounds", 40))
    rate = float(config.get("learning_rate", 0.3))
    max_thresholds = int(config.get("leaf_steps", 16))
    min_leaf = int(config.get("min_leaf", 20))
    # Бины по квантилям считаются ОДИН раз: сортировка на каждый признак и
    # каждый раунд на 46 607 строках стоила бы больше, чем весь остальной прогон.
    bins: list[list[int]] = []
    thresholds: list[list[float]] = []
    for feature in range(len(means)):
        column = sorted(row[feature] for row in standardized)
        if column[0] == column[-1]:
            bins.append([0] * total)
            thresholds.append([])
            continue
        steps = max(2, min(max_thresholds, total - 1))
        edges = [column[min(total - 1, int(total * step / steps))] for step in range(1, steps)]
        edges = sorted(set(edges))
        thresholds.append(edges)
        bins.append([_bin_of(row[feature], edges) for row in standardized])
    widths = [len(edge) + 1 for edge in thresholds]
    predictions = [base] * total
    stumps: list[dict[str, Any]] = []
    for _round in range(max(1, rounds)):
        probabilities = [1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value)))) for value in predictions]
        gradients = [labels[index] - probabilities[index] for index in range(total)]
        best_gain = 0.0
        best: dict[str, Any] | None = None
        for feature, edges in enumerate(thresholds):
            if not edges:
                continue
            size = widths[feature]
            sums = [0.0] * size
            counts = [0] * size
            for index, slot in enumerate(bins[feature]):
                sums[slot] += gradients[index]
                counts[slot] += 1
            total_gradient = sum(gradients)
            left_gradient = 0.0
            left_size = 0
            for slot in range(size - 1):
                left_gradient += sums[slot]
                left_size += counts[slot]
                right_size = total - left_size
                if left_size < min_leaf or right_size < min_leaf:
                    continue
                right_gradient = total_gradient - left_gradient
                gain = left_gradient * left_gradient / left_size + right_gradient * right_gradient / right_size
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best = {
                        "feature": feature,
                        "threshold": round(float(edges[slot]), 6),
                        "left": round(rate * left_gradient / left_size, 6),
                        "right": round(rate * right_gradient / right_size, 6),
                    }
        if best is None:
            break
        stumps.append(best)
        feature, edge, left, right = best["feature"], best["threshold"], best["left"], best["right"]
        for index, row in enumerate(standardized):
            predictions[index] += left if row[feature] < edge else right
    return Fitted("boost", means, scales, base, [], stumps)


def _bin_of(value: float, edges: Sequence[float]) -> int:
    """Номер бина по границам (бинарный поиск: leftmost edge > value)."""
    low, high = 0, len(edges)
    while low < high:
        middle = (low + high) // 2
        if edges[middle] <= value:
            low = middle + 1
        else:
            high = middle
    return low


def choose_threshold(
    scores: Sequence[float], labels: Sequence[int], target_fpr: float = CRITERION_FPR
) -> dict[str, Any]:
    """Порог по максимуму F1 при FPR ≤ target — на out-of-fold оценках train.

    Кандидаты — сами значения оценок: оптимум F1 всегда достигается там, где
    забор пересекает оценку какого-то токена. Делать это по val или test
    запрещено протоколом (см. docs/METRIC_SPEC.md).
    """
    order = sorted(range(len(scores)), key=lambda index: -scores[index])
    positives = sum(labels)
    negatives = len(labels) - positives
    best: tuple[float, float, float, float] | None = None
    tp = fp = 0
    for index in order:
        if labels[index]:
            tp += 1
        else:
            fp += 1
        fpr = fp / negatives if negatives else 0.0
        if fpr > target_fpr:
            break
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / positives if positives else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if best is None or f1 > best[0] + 1e-12 or (abs(f1 - best[0]) <= 1e-12 and fpr < best[2]):
            best = (f1, fpr, scores[index], precision)
    if best is None:
        return {"threshold": 0.5, "reason": "ни один порог не уложился в FPR", "f1": 0.0, "fpr": 1.0}
    return {
        "threshold": float(best[2]),
        "f1": round(best[0], 6),
        "fpr": round(best[1], 6),
        "precision": round(best[3], 6),
    }


def group_folds(keys: Sequence[str], folds: int, seed: int) -> list[list[int]]:
    """Групповая CV: все пары одного документа попадают в один фолд."""
    groups: dict[str, list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    folds = max(2, min(folds, len(names) or 2))
    buckets: list[list[int]] = [[] for _ in range(folds)]
    for position, name in enumerate(names):
        buckets[position % folds].extend(groups[name])
    return [sorted(item) for item in buckets]


def _smooth_within(values: list[float | None], pairs: Sequence[str], radius: int) -> list[float | None]:
    """Сглаживание оценок внутри одного ответа — тот же приём, что на val/test.

    Порог обязан искаться по тем же оценкам, по которым он потом применяется:
    иначе рычаг «агрегация» меняет шкалу между подбором и прогоном, и
    зафиксированный порог перестаёт быть тем, что действительно выбирали.
    """
    if radius <= 0:
        return values
    groups: dict[str, list[int]] = {}
    for index, pair in enumerate(pairs):
        groups.setdefault(pair, []).append(index)
    out: list[float | None] = list(values)
    for indices in groups.values():
        present = [index for index in indices if values[index] is not None]
        if not present:
            continue
        numbers = [float(values[index]) for index in present]
        for position, index in enumerate(present):
            window = numbers[max(0, position - radius) : min(len(numbers), position + radius + 1)]
            out[index] = statistics.fmean(window)
    return out


def fit_with_cv(
    rows: list[list[float]],
    labels: list[int],
    keys: list[str],
    pairs: Sequence[str],
    config: dict[str, Any],
    seed: int,
    window: int = 0,
) -> tuple[Fitted, float, dict[str, Any]]:
    """Обучение на обучающей части + порог по out-of-fold оценкам её же.

    Groups — документы: фолд строится так, чтобы все пары одного документа
    оказались в удержанной части. Это и есть «group CV по документам» из
    протокола: порог не видит тех же документов, на которых учился.
    """
    folds = group_folds(keys, int(config.get("folds", 5)), seed)
    oof: list[float | None] = [None] * len(rows)
    holdouts: list[list[int]] = []
    for fold in folds:
        hold = set(fold)
        train_index = [index for index in range(len(rows)) if index not in hold]
        if len(train_index) < 40 or not fold:
            continue
        model = fit_model(
            [rows[index] for index in train_index], [labels[index] for index in train_index], config, seed
        )
        for index in fold:
            oof[index] = float(model.scores([rows[index]])[0])
        holdouts.append(fold)
    filled = [index for index, value in enumerate(oof) if value is not None]
    if len(filled) < 40:
        holdouts = []
        oof = [float(value) for value in fit_model(rows, labels, config, seed).scores(rows)]
        filled = list(range(len(rows)))
    smoothed = _smooth_within(oof, pairs, window)
    cut = choose_threshold([float(smoothed[index]) for index in filled], [labels[index] for index in filled])
    final = fit_model(rows, labels, config, seed)
    details = {
        "folds": len(holdouts),
        "oof_rows": len(filled),
        "threshold": cut["threshold"],
        "smoothing_radius": int(window),
        "selection": {key: value for key, value in cut.items() if key != "threshold"},
    }
    return final, float(cut["threshold"]), details


# ------------------------------------------------------------------- метрики


def score_metrics(rows: Sequence[dict[str, Any]], threshold: float, merge_gap: int = 2) -> dict[str, Any]:
    """Метрики всех трёх уровней по строкам токенов (определения — из продукта)."""
    labels = [int(row["gold"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    flags = [score >= threshold for score in scores]
    payload: dict[str, Any] = {
        "tokens": _plain(_token_metrics(labels, flags, scores)),
        "rows": len(rows),
        "positive_rows": sum(labels),
    }
    groups = group_by_pair(rows)
    answers = [
        {
            "id": items[0]["id"],
            "gold": 1 if any(int(item["gold"]) for item in items) else 0,
            "score": max((float(item["score"]) for item in items), default=0.0),
            "predicted": merge_spans(items, threshold, merge_gap),
            "truth": list(items[0].get("spans") or []),
        }
        for items in groups
    ]
    payload["answers"] = _plain(
        _answer_metrics(
            [int(item["gold"]) for item in answers],
            [float(item["score"]) for item in answers],
            threshold,
        )
    )
    payload["spans"] = _plain(
        _span_f1([item["predicted"] for item in answers], [item["truth"] for item in answers], iou_threshold=0.5)
    )
    payload["answers_flagged"] = sum(1 for item in answers if item["score"] >= threshold)
    payload["answers_gold"] = sum(int(item["gold"]) for item in answers)
    return payload


def criterion_status(tokens: dict[str, Any] | None) -> dict[str, Any]:
    """Статус ориентира ТЗ по первичной метрике (токены). Не «скрывать» и не «недомерить»."""
    numbers = tokens or {}
    f1 = float(numbers.get("f1", 0.0) or 0.0)
    fpr = float(numbers.get("fpr", 1.0) if numbers.get("fpr") is not None else 1.0)
    passed = f1 >= CRITERION_F1 and fpr <= CRITERION_FPR
    return {
        "target": f"F1 ≥ {CRITERION_F1} при FPR ≤ {CRITERION_FPR} (токены, hf, отложенный test)",
        "status": "PASS" if passed else "NOT MET",
        "token_f1": round(f1, 6),
        "token_fpr": round(fpr, 6),
        "f1_gap": round(CRITERION_F1 - f1, 6),
        "fpr_gap": round(fpr - CRITERION_FPR, 6),
    }


# ----------------------------------------------------------------- команды


def parse_corpora(spec: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in str(spec).split(";"):
        if not item.strip():
            continue
        name, _, directory = item.partition("=")
        if not directory:
            raise SystemExit(f"--corpus ждёт формат имя=каталог, получено «{item}»")
        out[name.strip()] = Path(directory.strip())
    if not out:
        raise SystemExit("--corpus обязателен (например a3=data/corpus_a3/splits)")
    return out


def command_freeze(args: argparse.Namespace) -> int:
    """Шаг 0: аудит разбиения + фиксация конфигурации и правил отбора."""
    started = time.time()
    corpora: dict[str, Any] = {}
    problems: list[str] = []
    for name, split_dir in sorted(parse_corpora(args.corpus).items()):
        parts: dict[str, Any] = {}
        documents: dict[str, set[str]] = {}
        for part, filename in (("train", "train.jsonl"), ("val", None), ("test", "test.jsonl")):
            path = val_path_of(split_dir) if filename is None else split_dir / filename
            if not path.is_file():
                problems.append(f"{name}: нет файла разбиения {path}")
                continue
            records = load_split(path)
            names = sorted({document_of(record) for record in records})
            documents[part] = set(names)
            parts[part] = {
                "document_ids": names if len(names) <= DOCUMENTS_LIST_LIMIT else [],
                "document_ids_listed": len(names) <= DOCUMENTS_LIST_LIMIT,
                "file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                "sha256": sha256_file(path),
                "pairs": len(records),
                "documents": len(names),
                "document_list_sha256": hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest(),
                "mode_counts": _mode_counts(records),
                "gold_spans": sum(len(spans_of(record)) for record in records),
            }
        for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = documents.get(first, set()) & documents.get(second, set())
            if overlap:
                problems.append(f"{name}: документы пересекаются между {first} и {second}: {sorted(overlap)[:5]}")
        corpora[name] = {"splits_dir": str(split_dir), "parts": parts, "document_overlap": {}}
    config_path = Path(args.config) if args.config else None
    config_payload: dict[str, Any] = {}
    if config_path and config_path.is_file():
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload = {
        "protocol": "hf-criterion-v1",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "metric_spec": "docs/METRIC_SPEC.md",
        "metric_spec_sha256": sha256_file(ROOT / "docs" / "METRIC_SPEC.md"),
        "criterion": {"token_f1_min": CRITERION_F1, "token_fpr_max": CRITERION_FPR, "primary_level": "tokens"},
        "selection_rule": "максимум token_f1 на val при token_fpr ≤ 0,40; порог и гиперпараметры — только из CV на train",
        "answer_rule": "оценка ответа = максимум токенных оценок, решение — тем же порогом (span-уровень — склейка маски)",
        "experiments_limit": MAX_EXPERIMENTS,
        "feature_names": list(GRID_FEATURE_NAMES),
        "grid_cache_format": GRID_CACHE_FORMAT,
        "corpora": corpora,
        "config": config_payload,
        "config_sha256": sha256_file(config_path) if config_path and config_path.is_file() else "",
        "hardware": hardware_manifest(),
        "code_commit": _git_commit(),
        "problems": problems,
        "duration_s": round(time.time() - started, 2),
    }
    out = Path(args.out)
    write_json(out / "freeze.json", payload)
    print(f"заморозка записана: {out / 'freeze.json'} (sha256 спецификации {payload['metric_spec_sha256'][:12]})")
    for name, block in corpora.items():
        for part, numbers in block["parts"].items():
            print(
                f"  {name}/{part}: пар {numbers['pairs']}, документов {numbers['documents']}, sha256 {numbers['sha256'][:12]}"
            )
    if problems:
        print("ПРОБЛЕМЫ РАЗБИЕНИЯ:")
        for item in problems:
            print(f"  - {item}")
        return 1
    return 0


def _mode_counts(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        mode = str((record.get("meta") or {}).get("mode") or "без mode")
        counts[mode] = counts.get(mode, 0) + 1
    return counts


def command_baseline(args: argparse.Namespace) -> int:
    """Шаг 1: текущий код в hf на val, без изменений; test не трогаем."""
    started = time.time()
    from spanverify.engine import Verifier

    cache_stats: dict[str, Any] = {"source": "", "enabled": False}
    counting_cache = None
    if args.features_cache and args.mode != "hf":
        raise SystemExit(
            "--features-cache имеет смысл только в режиме hf: демо-признаки лексические и в кеш v2 не пишутся"
        )
    if args.features_cache:
        # Продуктовые признаки можно взять из кеша v2 (его считает
        # precompute_features.py без --grid): шаг 1 от этого быстрее, а
        # «молчаливый» промах по ключу виден в счётчике hits/misses, который
        # попадает в манифест.
        from spanverify.features import (  # noqa: PLC0415
            load_feature_cache,
            set_feature_cache,
            validate_feature_cache,
        )

        counting_cache = load_feature_cache(args.features_cache, counting=True)
        report = validate_feature_cache(counting_cache, "hf", args.model)
        set_feature_cache(counting_cache)
        cache_stats = {
            "source": str(args.features_cache),
            "enabled": True,
            "rows": len(counting_cache),
            "validation": report if isinstance(report, dict) else {"ok": True},
        }

    manifest: dict[str, Any] = {
        "step": "baseline",
        "protocol": "hf-criterion-v1",
        "code_commit": _git_commit(),
        "mode": args.mode,
        "model": args.model or "по умолчанию конфигурации продукта",
        "hardware": hardware_manifest(),
        "corpora": {},
        "note": "числа уровня токенов считаются по pred продукта (result.tokens[i].flagged), "
        "уровень ответа — по product-вердикту; test в этом шаге не используется",
        # Соответствие продукдовых признаков сетке: шаг 2 начинает с того же
        # состава, поэтому разрыв шага 1 и шага 2 — это измерение, а не разные
        # признаки по умолчанию.
        "product_features_in_grid_terms": list(BASELINE_FEATURES),
        "feature_cache": cache_stats,
    }
    for name, split_dir in sorted(parse_corpora(args.corpus).items()):
        path = val_path_of(split_dir)
        records = load_split(path)
        verifier = Verifier(mode=args.mode, weights_path=args.weights, model_name=args.model or None)
        rows: list[dict[str, Any]] = []
        verdicts: list[tuple[int, float]] = []
        for record in records:
            answer = record.get("answer", "")
            truth = spans_of(record)
            result = verifier.verify(answer, record.get("context", ""), with_tokens=True)
            tokens = list(tokenize_with_offsets(answer))
            for index, token in enumerate(tokens):
                if not is_scored_token(token.text):
                    continue
                info = result.tokens[index] if index < len(result.tokens) else {}
                rows.append(
                    {
                        "id": str(record.get("id") or ""),
                        "level": "tokens",
                        "seed": int(args.seed),
                        "token": token.text,
                        "start": token.start,
                        "end": token.end,
                        "gold": 1 if any(token.start < end and token.end > start for start, end in truth) else 0,
                        "score": round(float(info.get("risk", 0.0)), 8),
                        "pred": int(bool(info.get("flagged"))),
                    }
                )
            verdicts.append(
                (
                    1 if truth else 0,
                    1.0 if str(result.verdict) not in {"grounded", "empty"} else 0.0,
                )
            )
        labels = [int(row["gold"]) for row in rows]
        flags = [bool(row["pred"]) for row in rows]
        scores = [float(row["score"]) for row in rows]
        tokens_metrics = _plain(_token_metrics(labels, flags, scores))
        answers_metrics = _plain(
            _answer_metrics([label for label, _ in verdicts], [float(score) for _, score in verdicts], None)
        )
        block = {
            "splits_dir": str(split_dir),
            "val_file": path.name,
            "val_sha256": sha256_file(path),
            "pairs": len(records),
            "documents": len({document_of(record) for record in records}),
            "token_rows": len(rows),
            "metrics_tokens": tokens_metrics,
            "metrics_answers_verdict": {
                key: value for key, value in answers_metrics.items() if key in {"precision", "recall", "f1", "fpr"}
            },
            "criterion": criterion_status(tokens_metrics),
        }
        if args.predictions_out:
            target = Path(str(args.predictions_out).format(corpus=name))
            block["predictions_file"] = str(target)
            block["prediction_rows"] = write_jsonl(target, rows)
        manifest["corpora"][name] = block
        print(
            f"{name}: токены F1 {tokens_metrics.get('f1')} FPR {tokens_metrics.get('fpr')} "
            f"AUC {tokens_metrics.get('auc')} · вердикты F1 {answers_metrics.get('f1')} → {block['criterion']['status']}"
        )
    if counting_cache is not None:
        manifest["feature_cache"].update({"hits": counting_cache.hits, "misses": counting_cache.misses})
        from spanverify.features import set_feature_cache  # noqa: PLC0415

        set_feature_cache(None)
    manifest["duration_s"] = round(time.time() - started, 2)
    out = Path(args.out)
    write_json(out / "manifest.json", manifest)
    print(f"базовая линия: {out / 'manifest.json'}")
    if manifest["feature_cache"].get("enabled"):
        cache = manifest["feature_cache"]
        print(f"кеш признаков: строк {cache['rows']}, попаданий {cache['hits']}, промахов {cache['misses']}")
    return 0


def _experiment_entry(number: int, experiment: dict[str, Any]) -> dict[str, Any]:
    names = [str(name) for name in experiment.get("features", [])]
    unknown = sorted(set(names) - set(GRID_FEATURE_NAMES))
    if unknown:
        raise SystemExit(f"эксперимент {number} ({experiment.get('name')}): неизвестные признаки {unknown}")
    reserved = {"name", "hypothesis", "change", "features", "window", "answer_share", "merge_gap"}
    return {
        "number": number,
        "name": str(experiment.get("name") or f"exp-{number}"),
        "hypothesis": str(experiment.get("hypothesis", "")),
        "change": str(experiment.get("change", "")),
        "features": names,
        "window": int(experiment.get("window", 0)),
        "merge_gap": int(experiment.get("merge_gap", 2)),
        "classifier": str(experiment.get("classifier", "logreg")),
        "params": {key: value for key, value in experiment.items() if key not in reserved},
    }


def run_experiment(
    cache: dict[str, dict[str, Any]],
    fit_records: Sequence[dict[str, Any]],
    eval_records: Sequence[dict[str, Any]],
    entry: dict[str, Any],
    seed: int,
    *,
    keep_rows: bool = False,
) -> dict[str, Any]:
    """Один эксперимент: обучение на fit-части, оценка на eval-части, порог с CV по fit."""
    names = entry["features"]
    fit_rows = prepare(fit_records, cache, names)
    eval_rows = prepare(eval_records, cache, names)
    if not fit_rows or not eval_rows:
        return {"error": "нет строк: кеш не покрывает выборку"}
    config = dict(entry["params"])
    config["classifier"] = entry["classifier"]
    model, threshold, details = fit_with_cv(
        feature_matrix(fit_rows, names),
        [int(row["gold"]) for row in fit_rows],
        [str(row["doc"]) for row in fit_rows],
        [str(row["id"]) for row in fit_rows],
        config,
        seed,
        int(entry["window"]),
    )
    for row, value in zip(eval_rows, model.scores(feature_matrix(eval_rows, names)), strict=True):
        row["score"] = float(value)
    groups = group_by_pair(eval_rows)
    if entry["window"]:
        smooth(groups, int(entry["window"]))
    fit_scores = model.scores(feature_matrix(fit_rows, names))
    block = {
        "threshold": round(float(threshold), 6),
        "threshold_selection": details,
        "threshold_scale": "вероятность модели; порог подобран по out-of-fold оценкам того же масштаба и с тем же сглаживанием",
        "fit": _plain(
            _token_metrics(
                [int(row["gold"]) for row in fit_rows],
                [float(score) >= threshold for score in fit_scores],
                fit_scores,
            )
        ),
        "eval": score_metrics(eval_rows, threshold, int(entry.get("merge_gap", 2))),
        "model": model.to_dict(),
        "model_sha256": hashlib.sha256(json.dumps(model.to_dict(), sort_keys=True).encode("utf-8")).hexdigest(),
    }
    if keep_rows:
        block["_rows"] = [dict(row) for row in eval_rows]
    return block


def command_sweep(args: argparse.Namespace) -> int:
    """Шаг 2: не больше 10 конфигураций на val, отбор по правилу, а не по ощущению."""
    started = time.time()
    cache = load_grid_cache(_cache_paths(Path(args.grid_cache)))
    grid = json.loads(Path(args.grid).read_text(encoding="utf-8"))
    experiments = grid.get("experiments") or []
    if len(experiments) > MAX_EXPERIMENTS:
        raise SystemExit(f"экспериментов {len(experiments)} > {MAX_EXPERIMENTS}: стоп-правило протокола")
    seed = int(grid.get("seed", 42))
    corpora: dict[str, dict[str, Any]] = {}
    for name, split_dir in sorted(parse_corpora(args.corpus).items()):
        corpora[name] = {
            "dir": split_dir,
            "train": load_split(split_dir / "train.jsonl"),
            "val": load_split(val_path_of(split_dir)),
        }
    results: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for number, experiment in enumerate(experiments, start=1):
        entry = _experiment_entry(number, experiment)
        if entry["name"] in seen_names:
            raise SystemExit(f"имя эксперимента «{entry['name']}» повторяется: имена — ключи разбора")
        seen_names.add(entry["name"])
        for name, parts in corpora.items():
            block = run_experiment(cache, parts["train"], parts["val"], entry, seed)
            parts.setdefault("results", {})[entry["name"]] = block
            numbers = (block.get("eval") or {}).get("tokens") or {}
            print(
                f"[{number}/{len(experiments)}] {entry['name']} · {name}: "
                f"F1(val) {numbers.get('f1')} FPR(val) {numbers.get('fpr')} AUC {numbers.get('auc')}"
            )
        if args.primary not in corpora:
            raise SystemExit(f"--primary {args.primary}: корпуса нет в --corpus")
        primary = corpora[args.primary]["results"][entry["name"]]
        tokens = (primary.get("eval") or {}).get("tokens", {})
        entry["selection"] = {
            "corpus": args.primary,
            "token_f1_val": tokens.get("f1"),
            "token_fpr_val": tokens.get("fpr"),
            "auc_val": tokens.get("auc"),
        }
        entry["per_corpus"] = {name: _summary(parts["results"][entry["name"]]) for name, parts in corpora.items()}
        results.append(entry)
    eligible = [
        item
        for item in results
        if (item["selection"]["token_f1_val"] or 0.0) > 0.0
        and (item["selection"]["token_fpr_val"] if item["selection"]["token_fpr_val"] is not None else 1.0)
        <= CRITERION_FPR
    ]
    best = (
        max(
            eligible, key=lambda item: (item["selection"]["token_f1_val"], -(item["selection"]["token_fpr_val"] or 0.0))
        )
        if eligible
        else None
    )
    best_name = best["name"] if best else None
    payload = {
        "step": "sweep",
        "protocol": "hf-criterion-v1",
        "rule": "выбирается конфигурация с максимальным token_f1 на val при token_fpr ≤ 0,40; ties → меньший FPR",
        "criterion": {"token_f1_min": CRITERION_F1, "token_fpr_max": CRITERION_FPR},
        "limit": MAX_EXPERIMENTS,
        "run_count": len(results),
        "seed": seed,
        "grid_file": str(args.grid),
        "grid_sha256": sha256_file(Path(args.grid)),
        "primary_corpus": args.primary,
        "best": best_name,
        "best_selection": best["selection"] if best else None,
        "best_reaches_criterion_on_val": bool(best and (best["selection"]["token_f1_val"] or 0.0) >= CRITERION_F1),
        "experiments": results,
        "hardware": hardware_manifest(),
        "code_commit": _git_commit(),
        "duration_s": round(time.time() - started, 2),
    }
    out = Path(args.out)
    write_json(out / "sweep.json", payload)
    if best is not None and args.write_predictions:
        rows, run = _best_predictions(cache, corpora, best, seed)
        target = out / "predictions_val.jsonl.gz"
        payload["predictions_val_rows"] = write_jsonl(target, rows)
        write_json(out / "sweep.json", payload)
        write_json(out / "manifest_val.json", val_manifest(best, run, rows, corpora, seed, payload))
        print(f"предсказания лучшей конфигурации на val: {payload['predictions_val_rows']} строк → {target}")
        print(f"манифест val для пересчёта: {out / 'manifest_val.json'}")
    print(
        f"sweep: прогонов {len(results)} (лимит {MAX_EXPERIMENTS}), лучшая — {best_name or 'нет: ни одна не прошла ограничение по FPR'}"
    )
    return 0


def _summary(block: dict[str, Any]) -> dict[str, Any]:
    eval_numbers = (block.get("eval") or {}) if isinstance(block, dict) else {}
    return {
        "threshold": block.get("threshold"),
        "model_sha256": block.get("model_sha256"),
        "tokens": eval_numbers.get("tokens"),
        "answers": eval_numbers.get("answers"),
        "spans": eval_numbers.get("spans"),
        "rows": eval_numbers.get("rows"),
        "positive_rows": eval_numbers.get("positive_rows"),
        "criterion": criterion_status(eval_numbers.get("tokens")),
        "threshold_selection": block.get("threshold_selection"),
        "error": block.get("error"),
    }


def _best_predictions(
    cache: dict[str, dict[str, Any]],
    corpora: dict[str, dict[str, Any]],
    best: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    """Строки предсказаний выбранной конфигурации: любую цифру обязано можно пересчитать.

    Прогон повторяется детерминированно на том же seed'е — это не новый
    эксперимент, а сохранение результата уже выбранного.
    """
    corpus_name = str(best["selection"]["corpus"])
    parts = corpora[corpus_name]
    run = run_experiment(cache, parts["train"], parts["val"], dict(best), seed, keep_rows=True)
    threshold = float(run["threshold"])
    out: list[dict[str, Any]] = []
    run["_threshold"] = threshold
    for row in run.get("_rows") or []:
        out.append(
            {
                "id": f"{corpus_name}:{row['id']}",
                "level": "tokens",
                "seed": seed,
                "token": row["token"],
                "start": row["start"],
                "end": row["end"],
                "gold": int(row["gold"]),
                "score": round(float(row["score"]), 8),
                "pred": int(float(row["score"]) >= threshold),
            }
        )
    return out, run


def val_manifest(
    best: dict[str, Any],
    run: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    corpora: dict[str, dict[str, Any]],
    seed: int,
    sweep_payload: dict[str, Any],
) -> dict[str, Any]:
    """Манифест вида «как финальный», но на val: пересчёт чисел работает одинаково.

    Нужен, чтобы расхождение «манифест ≠ предсказания» ловилось ещё до шага 3:
    тот же ``verify-manifest`` и тот же тест, что проверяют финал.
    """
    corpus = str(best["selection"]["corpus"])
    split_dir = Path(str(corpora[corpus]["dir"]))
    numbers = run["eval"]
    per_seed = {
        "seed": seed,
        "threshold": float(run["threshold"]),
        "model_sha256": run["model_sha256"],
        "tokens": numbers["tokens"],
        "answers": numbers["answers"],
        "spans": numbers["spans"],
        "rows": numbers["rows"],
        "positive_rows": numbers["positive_rows"],
    }
    aggregate = aggregate_seeds([per_seed])
    return {
        "step": "sweep-best-on-val",
        "protocol": "hf-criterion-v1",
        "selection_corpus": corpus,
        "splits_dir": str(split_dir),
        "splits_sha256": {
            part: (sha256_file(path) if path.is_file() else "")
            for part, path in (
                ("train", split_dir / "train.jsonl"),
                ("val", val_path_of(split_dir)),
                ("test", split_dir / "test.jsonl"),
            )
        },
        "selected_by": best["name"],
        "hypothesis": best.get("hypothesis", ""),
        "features": list(best["features"]),
        "classifier": best.get("classifier"),
        "window": int(best.get("window", 0)),
        "merge_gap": int(best.get("merge_gap", 2)),
        "params": best.get("params", {}),
        "grid_sha256": sweep_payload.get("grid_sha256"),
        "code_commit": sweep_payload.get("code_commit"),
        "per_seed": [per_seed],
        "metrics_mean_std": aggregate,
        "threshold": float(run["threshold"]),
        "criterion": criterion_status(numbers["tokens"]),
        "criterion_note": "это val, а не test: статус показывает, достигнут ли ориентир на отобранной части, и не является ответом по договору",
        "rows": len(rows),
    }


def command_final(args: argparse.Namespace) -> int:
    """Шаг 3: ровно один запуск на test; обучение на train(+val), N seed'ов, отбора по test нет."""
    started = time.time()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("step") != "final-config":
        raise SystemExit(
            "конфигурация не размечена как final-config: шаг 3 выполняется один раз по "
            "зафиксированному до запуска описанию (см. docs/EXPERIMENTS.md)"
        )
    names = [str(name) for name in config["features"]]
    entry = {
        "name": str(config.get("name", "final")),
        "features": names,
        "window": int(config.get("window", 0)),
        "merge_gap": int(config.get("merge_gap", 2)),
        "classifier": str(config.get("classifier", "logreg")),
        "params": dict(config.get("params") or {}),
    }
    cache = load_grid_cache(_cache_paths(Path(args.grid_cache)))
    split_dir = Path(args.splits)
    train = load_split(split_dir / "train.jsonl")
    val_file = val_path_of(split_dir)
    val = load_split(val_file) if (args.include_val and val_file.is_file()) else []
    test = load_split(split_dir / "test.jsonl")
    seeds = [int(value) for value in str(args.seeds).split(",") if value.strip()]
    per_seed: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for seed in seeds:
        block = run_experiment(cache, list(train) + list(val), test, entry, seed, keep_rows=True)
        if "error" in block:
            raise SystemExit(f"seed {seed}: {block['error']}")
        threshold = float(block["threshold"])
        rows = block.pop("_rows")
        metrics = block["eval"]
        per_seed.append(
            {
                "seed": seed,
                "threshold": threshold,
                "model_sha256": block["model_sha256"],
                "tokens": metrics["tokens"],
                "answers": metrics["answers"],
                "spans": metrics["spans"],
                "rows": metrics["rows"],
                "positive_rows": metrics["positive_rows"],
            }
        )
        for row in rows:
            predictions.append(
                {
                    "id": f"{args.corpus_name}:{row['id']}",
                    "level": "tokens",
                    "seed": seed,
                    "token": row["token"],
                    "start": row["start"],
                    "end": row["end"],
                    "gold": int(row["gold"]),
                    "score": round(float(row["score"]), 8),
                    "pred": int(float(row["score"]) >= threshold),
                }
            )
        numbers = metrics["tokens"]
        print(f"seed {seed}: F1(test) {numbers.get('f1')} FPR(test) {numbers.get('fpr')} порог {threshold}")
    aggregate = aggregate_seeds(per_seed)
    status = criterion_status(aggregate)
    freeze_path = Path(args.freeze) if args.freeze else Path(args.out) / "freeze.json"
    splits_sha = {}
    for part, filename in (("train", "train.jsonl"), ("val", val_file.name), ("test", "test.jsonl")):
        path = split_dir / filename
        splits_sha[part] = sha256_file(path) if path.is_file() else ""
    payload = {
        "step": "final",
        "protocol": "hf-criterion-v1",
        "code_commit": _git_commit(),
        "freeze_file": str(freeze_path),
        "freeze_sha256": sha256_file(freeze_path) if freeze_path.is_file() else "",
        "config_file": str(config_path),
        "config_sha256": sha256_file(config_path),
        "config": config,
        "metric_spec": "docs/METRIC_SPEC.md",
        "metric_spec_sha256": sha256_file(ROOT / "docs" / "METRIC_SPEC.md"),
        "corpus": args.corpus_name,
        "splits_dir": str(split_dir),
        "splits_sha256": splits_sha,
        "pairs": {"train": len(train), "val": len(val), "test": len(test)},
        "documents": {
            "train": len({document_of(row) for row in train}),
            "test": len({document_of(row) for row in test}),
        },
        "fit_on": "train+val" if val else "train",
        "seeds": seeds,
        "seed_rule": "seed'ы зафиксированы до запуска; лучший не выбирается — среднее по всем",
        "threshold": aggregate.get("threshold"),
        "threshold_source": "CV по обучающей части (docs/METRIC_SPEC.md, раздел «отбор»)",
        "per_seed": per_seed,
        "metrics_mean_std": aggregate,
        "criterion": status,
        "hardware": hardware_manifest(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "duration_s": round(time.time() - started, 2),
        "predictions": "predictions_test.jsonl.gz",
        "gold_file": "gold_test.jsonl.gz",
        "gold_rows": 0,
        "gold_sha256": "",
        "predictions_format": list(PREDICTIONS_FIELDS),
        "deviations": list(config.get("deviations") or []),
    }
    out = Path(args.out)
    gold_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for record in test:
        identifier = f"{args.corpus_name}:{record.get('id')}"
        if identifier in seen_ids:
            continue
        seen_ids.add(identifier)
        gold_rows.append({"id": identifier, "spans": [list(span) for span in spans_of(record)]})
    rows_written = write_jsonl(out / "predictions_test.jsonl.gz", predictions)
    payload["prediction_rows"] = rows_written
    payload["gold_rows"] = len(gold_rows)
    payload["gold_sha256"] = sha256_file(out / "gold_test.jsonl.gz")
    write_json(out / "manifest.json", payload)
    print(
        f"итог: токены F1 {status['token_f1']} ± {aggregate.get('f1_std')} FPR {status['token_fpr']} → {status['status']}"
    )
    print(f"строк предсказаний {rows_written}; манифест {out / 'manifest.json'}; записей {len(per_seed)} seed'ов")
    return 0


def aggregate_seeds(per_seed: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """mean ± std по seed'ам; худший и лучший не выбираются (протокол)."""
    out: dict[str, Any] = {}
    if not per_seed:
        return out
    for key in ("f1", "precision", "recall", "fpr", "auc", "tp", "fp", "fn", "tn"):
        values = [float((item.get("tokens") or {}).get(key, 0.0) or 0.0) for item in per_seed]
        out[key] = round(statistics.fmean(values), 6)
        out[f"{key}_std"] = round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0
        out[f"{key}_per_seed"] = [round(value, 6) for value in values]
    prefixes = {"answers": "answer", "spans": "span"}
    # Контейнмент и ширина фрагмента — у product-оценки (engine.evaluate);
    # харнесс их не воспроизводит, и «придумать» их формулой нельзя: в манифест
    # попадают только пересчитанные из предсказаний числа.
    for level, keys in (
        ("answers", ("f1", "precision", "recall", "fpr", "auc")),
        ("spans", ("f1", "precision", "recall")),
    ):
        for key in keys:
            values = [float((item.get(level) or {}).get(key, 0.0) or 0.0) for item in per_seed]
            out[f"{prefixes[level]}_{key}"] = round(statistics.fmean(values), 6)
            if len(values) > 1:
                out[f"{prefixes[level]}_{key}_std"] = round(statistics.pstdev(values), 6)
    out["threshold"] = float(per_seed[0].get("threshold", 0.0))
    out["thresholds_per_seed"] = [float(item.get("threshold", 0.0)) for item in per_seed]
    out["model_sha256_per_seed"] = [str(item.get("model_sha256", "")) for item in per_seed]
    out["rows"] = int(per_seed[0].get("rows", 0))
    out["positive_rows"] = int(per_seed[0].get("positive_rows", 0))
    return out


def command_verify(args: argparse.Namespace) -> int:
    """Пересчёт метрик из предсказаний и сверка с манифестом (то же, что тест)."""
    gold_argument = Path(args.gold) if args.gold else Path(args.manifest).parent / "gold_test.jsonl.gz"
    problems, numbers = compare_manifest(Path(args.manifest), Path(args.predictions), gold_argument)
    if problems:
        print("РАСХОЖДЕНИЯ:")
        for item in problems:
            print(f"  - {item}")
        return 1
    print(
        f"числа воспроизведены: строк {numbers['rows']}, seed'ов {numbers['seeds']}, "
        f"F1 {numbers['f1']} ± {numbers['f1_std']}, FPR {numbers['fpr']} → {numbers['criterion']}"
    )
    if args.check_splits:
        extra, more = check_splits(Path(args.manifest))
        problems.extend(more)
        if problems:
            print("РАСХОЖДЕНИЯ РАЗБИЕНИЯ:")
            for item in problems:
                print(f"  - {item}")
            return 1
        print(f"разбиение: sha256 теста {extra['test_sha256'][:12]}, пересечений документов нет")
    return 0


def compare_manifest(
    manifest_path: Path, predictions_path: Path, gold_path: Path | None = None
) -> tuple[list[str], dict[str, Any]]:
    """Сверка каждого числа манифеста с пересчётом из строк предсказаний.

    Уровень токенов восстанавливается из одних только строк ``gold``/``score``/
    ``pred``. Уровни ответов и фрагментов нуждаются ещё и в gold-границах пар:
    их финальный прогон пишет в ``gold_test.jsonl.gz`` рядом с предсказаниями.
    Без этого файла фрагментный уровень не пересчитывается, и это фиксируется
    отдельной проблемой — «нельзя пересчитать» не равно «совпало».
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = read_jsonl(predictions_path)
    gold: dict[str, list[tuple[int, int]]] = {}
    if gold_path and gold_path.is_file():
        for record in read_jsonl(gold_path):
            gold[str(record["id"])] = [(int(item[0]), int(item[1])) for item in record.get("spans", [])]
    problems: list[str] = []
    per_seed: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        missing = [key for key in PREDICTIONS_FIELDS if key not in row]
        if missing:
            problems.append(f"строка предсказания без полей {missing}")
            break
        per_seed.setdefault(int(row.get("seed", 0)), []).append(row)
    recorded = {int(item["seed"]): item for item in manifest.get("per_seed", [])}
    threshold_default = float(manifest.get("threshold") or 0.5)
    gap = int(((manifest.get("config") or {}).get("merge_gap")) or ((manifest.get("params") or {}).get("merge_gap", 2)))
    recomputed: list[dict[str, Any]] = []
    for seed, items in sorted(per_seed.items()):
        threshold = float(recorded.get(seed, {}).get("threshold", threshold_default))
        prepared = [
            {
                **row,
                "score": float(row["score"]),
                "gold": int(row["gold"]),
                "spans": gold.get(str(row["id"]), []),
            }
            for row in items
        ]
        metrics = score_metrics(prepared, threshold, gap)
        tokens = metrics["tokens"]
        recomputed.append({"seed": seed, "tokens": tokens, "answers": metrics["answers"], "spans": metrics["spans"]})
        if seed not in recorded:
            problems.append(f"seed {seed}: в манифесте нет записи per_seed")
            continue
        for key in ("f1", "precision", "recall", "fpr", "auc", "tp", "fp", "fn", "tn"):
            left = float(recorded[seed]["tokens"].get(key, 0.0) or 0.0)
            right = float(tokens.get(key, 0.0) or 0.0)
            if abs(left - right) > TOLERANCE:
                problems.append(f"seed {seed}: {key} — манифест {left}, пересчёт из предсказаний {right}")
        if gold:
            for level, keys in (
                ("answers", ("f1", "precision", "recall", "fpr")),
                ("spans", ("f1", "precision", "recall")),
            ):
                for key in keys:
                    left = float((recorded[seed].get(level) or {}).get(key, 0.0) or 0.0)
                    right = float((metrics[level] or {}).get(key, 0.0) or 0.0)
                    if abs(left - right) > TOLERANCE:
                        problems.append(f"seed {seed}: {level}/{key} — манифест {left}, пересчёт {right}")
    if not gold and str(manifest.get("step", "")).startswith("final"):
        problems.append("gold_test.jsonl.gz не найден: уровни ответов и фрагментов не пересчитаны")
    aggregate = manifest.get("metrics_mean_std") or {}
    if recomputed:
        fresh = aggregate_seeds(recomputed)
        checks = ["f1", "precision", "recall", "fpr", "auc"]
        if gold:
            checks += ["answer_f1", "answer_fpr", "span_f1"]
        for key in checks:
            left = float(aggregate.get(key, 0.0) or 0.0)
            right = float(fresh.get(key, 0.0) or 0.0)
            if abs(left - right) > TOLERANCE:
                problems.append(f"среднее по seed'ам {key}: манифест {left}, пересчёт {right}")
    status = criterion_status(aggregate)
    recorded_status = str((manifest.get("criterion") or {}).get("status") or "")
    if recorded_status != status["status"]:
        problems.append(
            f"статус критерия: манифест «{recorded_status}», пересчёт «{status['status']}» (F1 {status['token_f1']}, FPR {status['token_fpr']})"
        )
    numbers = {
        "rows": len(rows),
        "seeds": len(per_seed),
        "f1": aggregate.get("f1"),
        "f1_std": aggregate.get("f1_std"),
        "fpr": aggregate.get("fpr"),
        "criterion": status["status"],
    }
    return problems, numbers


def check_splits(manifest_path: Path) -> tuple[dict[str, Any], list[str]]:
    """Документы частей не пересекаются, sha256 теста совпадает с зафиксированным."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    split_dir = Path(str(manifest.get("splits_dir") or ""))
    recorded = manifest.get("splits_sha256") or {}
    documents: dict[str, set[str]] = {}
    for part, filename in (("train", "train.jsonl"), ("val", "dev.jsonl"), ("test", "test.jsonl")):
        path = split_dir / filename
        if not path.is_file():
            path = split_dir / ("val.jsonl" if filename == "dev.jsonl" else filename)
        if not path.is_file():
            continue
        records = load_split(path)
        documents[part] = {document_of(record) for record in records}
        expected = str(recorded.get(part) or "")
        actual = sha256_file(path)
        if expected and expected != actual:
            problems.append(f"{part}: sha256 файла {actual[:12]} != зафиксированный в манифесте {expected[:12]}")
    for first, second in (("train", "test"), ("val", "test"), ("train", "val")):
        overlap = documents.get(first, set()) & documents.get(second, set())
        if overlap:
            problems.append(f"документы пересекаются между {first} и {second}: {sorted(overlap)[:5]}")
    if not documents.get("test"):
        problems.append(f"тестовая выборка не найдена в {split_dir}")
    return {
        "test_sha256": str(recorded.get("test") or ""),
        "documents": {key: len(value) for key, value in documents.items()},
    }, problems


# ------------------------------------------------------------------- парсер


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Протокол измерения критерия в режиме hf")
    sub = parser.add_subparsers(dest="command", required=True)

    freeze = sub.add_parser("freeze", help="шаг 0: аудит разбиения и фиксация правил")
    freeze.add_argument(
        "--corpus", default="a1=data/corpus_a/splits;a3=data/corpus_a3/splits", help="имя=каталог через ';'"
    )
    freeze.add_argument("--config", default=None, help="файл конфигурации, который замораживается вместе с правилами")
    freeze.add_argument("--out", default="reports/hf_protocol")
    freeze.set_defaults(func=command_freeze)

    baseline = sub.add_parser("baseline", help="шаг 1: текущий код на val без изменений")
    baseline.add_argument("--corpus", default="a3=data/corpus_a3/splits")
    baseline.add_argument("--mode", choices=["demo", "hf"], default="hf")
    baseline.add_argument("--model", default="")
    baseline.add_argument("--weights", default=None)
    baseline.add_argument("--seed", type=int, default=42)
    baseline.add_argument("--predictions-out", default="reports/hf_baseline/predictions_val_{corpus}.jsonl.gz")
    baseline.add_argument("--out", default="reports/hf_baseline")
    baseline.add_argument(
        "--features-cache",
        default=None,
        help="каталог или файл кеша product-признаков v2 (precompute_features.py без --grid): ускоряет шаг 1",
    )
    baseline.set_defaults(func=command_baseline)

    sweep = sub.add_parser("sweep", help="шаг 2: до 10 конфигураций на val")
    sweep.add_argument("--grid", required=True, help="JSON с полем experiments")
    sweep.add_argument("--grid-cache", required=True, help="каталог или файл кеша v3")
    sweep.add_argument("--corpus", default="a3=data/corpus_a3/splits")
    sweep.add_argument("--primary", default="a3", help="корпус, по которому идёт отбор")
    sweep.add_argument("--out", default="reports/hf_protocol")
    sweep.add_argument(
        "--write-predictions", action="store_true", help="сохранить строки предсказаний лучшей конфигурации"
    )
    sweep.set_defaults(func=command_sweep)

    final = sub.add_parser("final", help="шаг 3: один запуск на test, N seed'ов")
    final.add_argument("--config", required=True, help="зафиксированная конфигурация (step=final-config)")
    final.add_argument("--grid-cache", required=True)
    final.add_argument("--splits", default="data/corpus_a3/splits")
    final.add_argument("--corpus-name", default="a3")
    final.add_argument("--seeds", default="42,43,44,45,46")
    final.add_argument(
        "--include-val", action="store_true", help="обучение на train+val (val уже использован в отборе)"
    )
    final.add_argument("--freeze", default=None)
    final.add_argument("--out", default="reports/hf_final")
    final.set_defaults(func=command_final)

    verify = sub.add_parser("verify-manifest", help="пересчитать метрики из предсказаний и сверить с манифестом")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--predictions", required=True)
    verify.add_argument("--check-splits", action="store_true", help="плюс сверка sha256 и непересечения частей")
    verify.add_argument(
        "--gold",
        default=None,
        help="gold_test.jsonl.gz (по умолчанию — рядом с манифестом): нужен для пересчёта уровней ответов и фрагментов",
    )
    verify.set_defaults(func=command_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
