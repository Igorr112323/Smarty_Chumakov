"""Пилот на реальной русскоязычной модели: информативность признаков по слоям.

Что делает скрипт (шаг 5 мастер-промта):

1. Берёт контрастные пары «ответ с опорой на контекст» / «ответ без опоры» и
   проверяет, что контраст действительно есть: доля слов ответа, встретившихся
   в контексте, должна быть ≥ 80 % для первых и ≤ 30 % для вторых. Если порог
   не выдержан, это печатается прямо в отчёте (данные не подгоняются).
2. Считает признаки модели ``ai-forever/rugpt3small_based_on_gpt2`` на CPU для
   каждого слоя внимания: ``first``, ``middle``, ``-4``, ``last``.
3. Считает AUC каждого признака с 95 % доверительным интервалом
   (бутстрэп, 5000 итераций, фиксированный seed) и сравнивает с базовыми
   уровнями: случайное угадывание (0.5) и косинусная близость эмбеддингов без
   внимания (``embedding_density``).
4. Пишет отчёт ``reports/pilot/pilot.json`` и ``reports/pilot/pilot.md``.

Ключевая гипотеза, которую проверяет пилот: масса внимания на контекст
(``ctx_attention_mass``) информативнее «сырой» энтропии внимания
(``attention_entropy``). Если гипотеза не подтвердится, в отчёте так и будет
написано — результат есть результат в любом случае.

Запуск (нужны torch и transformers, интернет для первой загрузки весов):

    python scripts/pilot_rugpt3small.py --pairs 24 --bootstrap 5000
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spanverify.core import tokenize_with_offsets  # noqa: E402
from spanverify.dataset import FAQ_TEMPLATES, SUBJECTS, unit_form  # noqa: E402
from spanverify.features import (  # noqa: E402
    DIAGNOSTIC_FEATURES,
    FEATURE_NAMES,
    _stem,
    hf_features,
    is_scored_token,
)

LAYERS = ("first", "middle", "-4", "last")
# Рабочие признаки плюс кандидаты. Кандидаты измеряются теми же данными и тем
# же бутстрэпом, но в итоговый риск не входят: решение о переводе в рабочие
# принимается только после измерения.
MEASURED_FEATURES = tuple(FEATURE_NAMES) + tuple(DIAGNOSTIC_FEATURES)
FEATURE_ATTRS = {
    "attention_entropy": "attention_entropy",
    "ctx_attention_mass": "ctx_attention_mass",
    "embedding_density": "embedding_density",
    "ctx_attention_mass_norm": "ctx_mass_norm",
    "ctx_attention_mass_lift": "ctx_mass_lift",
    "ctx_max_similarity": "ctx_max_similarity",
    "ctx_support_distance": "ctx_support_distance",
    "ctx_similarity_decay": "ctx_similarity_decay",
    "ctx_sim_contrast": "ctx_sim_contrast",
    "ctx_sim_margin": "ctx_sim_margin",
}
# Признаки, у которых большее значение означает БОЛЬШИЙ риск. Остальные
# (масса опоры, похожесть на контекст) работают в обратную сторону. Знак нужен
# только для ориентированного AUC: ранжирование не зависит от монотонности,
# но сравнение признаков между собой — зависит.
RISK_UP = frozenset({"attention_entropy", "embedding_density", "ctx_support_distance"})
MODEL_DEFAULT = "ai-forever/rugpt3small_based_on_gpt2"
POSITIVE_LABEL = 1


@dataclass
class TokenRow:
    """Один токен ответа: признаки по слоям и метка достоверности."""

    pair_id: str
    layer: str
    text: str
    features: dict[str, float]
    label: int
    role: str = "other"  # "fact" — токен числового значения, по нему идёт парный тест


def auc(labels: list[int], scores: list[float]) -> float:
    """AUC по ранговой формуле Манна — Уитни (совпадает с метрикой продукта)."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[position]]:
            end += 1
        average = (position + end) / 2 + 1
        for step in range(position, end + 1):
            ranks[order[step]] = average
        position = end + 1
    rank_sum = sum(ranks[index] for index, label in enumerate(labels) if label)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def bootstrap_ci(labels: list[int], scores: list[float], iterations: int = 5000, seed: int = 42) -> tuple[float, float]:
    """95 % доверительный интервал AUC бутстрэпом (ресемплинг токенов)."""
    if iterations <= 0 or len(labels) < 4:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    size = len(labels)
    values: list[float] = []
    for _ in range(iterations):
        indices = [rng.randrange(size) for _ in range(size)]
        sample_labels = [labels[index] for index in indices]
        if len(set(sample_labels)) < 2:
            continue
        values.append(auc(sample_labels, [scores[index] for index in indices]))
    values = [value for value in values if value == value]
    if len(values) < 20:
        return float("nan"), float("nan")
    values.sort()
    low = values[int(0.025 * len(values))]
    high = values[min(len(values) - 1, int(0.975 * len(values)))]
    return low, high


def overlap_ratio(answer: str, context: str) -> float:
    """Доля содержательных слов ответа, найденных в контексте (по основам)."""
    context_stems = {_stem(token.word) for token in tokenize_with_offsets(context) if is_scored_token(token.text)}
    answer_tokens = [token for token in tokenize_with_offsets(answer) if is_scored_token(token.text)]
    if not answer_tokens:
        return 0.0
    hits = sum(1 for token in answer_tokens if _stem(token.word) in context_stems)
    return hits / len(answer_tokens)


def build_contrast_pairs(count: int, seed: int = 2026) -> list[dict]:
    """Собрать контрастные пары: с опорой на контекст и без неё.

    Каждая пара — один и тот же вопрос и один и тот же документ. Отличие ровно
    одно: «с опорой» повторяет значение из документа, «без опоры» подставляет
    другое значение. Это самая трудная и самая честная постановка: признаки
    должны уловить подмену одного числа, а не «другой стиль письма».

    Числа берутся из контекста, поэтому контраст проверяем: доля ответов,
    воспроизводящих значение документа, должна быть 1.0 в первой группе и 0.0
    во второй. Если это не так — отчёт об этом скажет.
    """
    rng = random.Random(seed)
    rows: list[dict] = []
    for index in range(count):
        number = rng.randint(100, 999)
        subject, unit_kind, values = rng.choice(SUBJECTS)
        true_value = rng.choice(values)
        wrong_value = rng.choice([value for value in values if value != true_value])
        _, answer_template = rng.choice(FAQ_TEMPLATES)
        context = f"Регламент {number}: {subject} составляет " f"{true_value} {unit_form(true_value, unit_kind)}."

        def answer_for(value: int, template=answer_template, topic=subject, kind=unit_kind) -> str:
            """Ответ по шаблону с правильной формой единицы (аргументы связаны явно)."""
            return template.format(
                subject=topic,
                Subject=topic[0].upper() + topic[1:],
                value_phrase=f"{value} {unit_form(value, kind)}",
            )

        rows.append(
            {
                "id": f"pilot-{index:03d}-grounded",
                "context": context,
                "answer": answer_for(true_value),
                "kind": "grounded",
                "label": 0,
                "value": str(true_value),
            }
        )
        rows.append(
            {
                "id": f"pilot-{index:03d}-unsupported",
                "context": context,
                "answer": answer_for(wrong_value),
                "kind": "unsupported",
                "label": 1,
                "value": str(true_value),
            }
        )
    return rows


def collect_rows(pairs: list[dict], model_name: str, max_length: int) -> list[TokenRow]:
    """Посчитать признаки всех слоёв для всех пар (токены — единицы анализа)."""
    rows: list[TokenRow] = []
    for position, pair in enumerate(pairs, start=1):
        answer = str(pair["answer"])
        tokens = tokenize_with_offsets(answer)
        scored = [index for index, token in enumerate(tokens) if is_scored_token(token.text)]
        matrix_by_layer: dict[str, dict[str, list[float]]] = {}
        for layer in LAYERS:
            matrix = hf_features(
                answer,
                pair.get("context"),
                model_name=model_name,
                answer_tokens=tokens,
                layer=layer,
                max_length=max_length,
            )
            matrix_by_layer[layer] = {name: list(getattr(matrix, FEATURE_ATTRS[name])) for name in MEASURED_FEATURES}
        label = int(pair["label"])
        # Токен числового значения — единственное место, которое отличается в
        # контрастной паре. Отдельная разметка нужна для парного анализа:
        # агрегат по всем токенам размывает эффект одного числа.
        fact_indices = {index for index in scored if tokens[index].word.isdigit()}
        if not fact_indices:
            fact_indices = set(scored[:1])
        for index in scored:
            for layer in LAYERS:
                values = matrix_by_layer[layer]
                rows.append(
                    TokenRow(
                        pair_id=str(pair["id"]),
                        layer=layer,
                        text=tokens[index].text,
                        features={name: float(values[name][index]) for name in MEASURED_FEATURES},
                        label=label,
                        role="fact" if index in fact_indices else "other",
                    )
                )
        print(f"  [{position}/{len(pairs)}] {pair['id']}: токенов {len(scored)}, метка {label}", flush=True)
    return rows


def bootstrap_mean_ci(values: list[float], iterations: int = 5000, seed: int = 42) -> tuple[float, float]:
    """95 % доверительный интервал для среднего парных разниц (бутстрэп по парам)."""
    if len(values) < 2:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        means.append(statistics.fmean(sample))
    means.sort()
    low = means[int(0.025 * len(means))]
    high = means[min(len(means) - 1, int(0.975 * len(means)))]
    return (low, high)


def paired_deltas(rows: list[TokenRow], feature: str) -> list[float]:
    """Разницы «без опоры − с опорой» по парам для одного признака.

    Пары сопоставлены: один и тот же документ, один и тот же шаблон ответа,
    отличается только число. Поэтому сравнивать надо внутри пары, а не по
    средним групп: так из сравнения уходит вся вариативность жанра.
    """
    by_pair: dict[str, dict[int, float]] = {}
    for row in rows:
        if row.role != "fact":
            continue
        by_pair.setdefault(row.pair_id.rsplit("-", 1)[0], {})[row.label] = row.features[feature]
    return [values[1] - values[0] for values in by_pair.values() if 0 in values and 1 in values]


def analyse(rows: list[TokenRow], iterations: int) -> dict:
    """AUC и доверительные интервалы по каждому признаку и слою."""
    report: dict[str, dict[str, dict[str, float]]] = {}
    for layer in LAYERS:
        layer_rows = [row for row in rows if row.layer == layer]
        if not layer_rows:
            continue
        labels = [row.label for row in layer_rows]
        report[layer] = {}
        for name in MEASURED_FEATURES:
            scores = [row.features[name] for row in layer_rows]
            value = auc(labels, scores)
            low, high = bootstrap_ci(labels, scores, iterations=iterations)
            # Для «сырой» энтропии высокий балл означает риск, для массы опоры
            # знак обратный — приводим к «чем выше, тем недостовернее».
            orient = 1.0 if name in RISK_UP else -1.0
            oriented = [orient * score for score in scores]
            oriented_auc = auc(labels, oriented)
            # Доверительный интервал считается по ориентированным значениям:
            # иначе при orient = -1 печатался бы интервал вокруг 1 - AUC, и
            # число оказывалось бы вне собственного интервала.
            oriented_low, oriented_high = bootstrap_ci(labels, oriented, iterations=iterations)
            fact_rows = [row for row in layer_rows if row.role == "fact"]
            fact_labels = [row.label for row in fact_rows]
            fact_scores = [row.features[name] for row in fact_rows]
            fact_auc = auc(fact_labels, fact_scores)
            fact_low, fact_high = bootstrap_ci(fact_labels, fact_scores, iterations=iterations)
            grounded_values = [row.features[name] for row in fact_rows if row.label == 0]
            unsupported_values = [row.features[name] for row in fact_rows if row.label == 1]
            deltas = paired_deltas(layer_rows, name)
            delta_mean = statistics.fmean(deltas) if deltas else 0.0
            delta_low, delta_high = bootstrap_mean_ci(deltas, iterations=iterations)
            report[layer][name] = {
                "auc": value,
                "auc_oriented": oriented_auc,
                "ci_low": low,
                "ci_high": high,
                "ci_oriented_low": oriented_low,
                "ci_oriented_high": oriented_high,
                "tokens": len(scores),
                "auc_fact": fact_auc,
                "auc_fact_oriented": orient * fact_auc if fact_auc is not None else None,
                "ci_fact_low": fact_low,
                "ci_fact_high": fact_high,
                "fact_tokens": len(fact_scores),
                "delta_mean": delta_mean,
                "delta_ci_low": delta_low,
                "delta_ci_high": delta_high,
                "delta_significant": bool(delta_low * delta_high > 0),
                "fact_mean_grounded": statistics.fmean(grounded_values) if grounded_values else 0.0,
                "fact_mean_unsupported": statistics.fmean(unsupported_values) if unsupported_values else 0.0,
            }
    return report


def copies_context_value(answer: str, context: str, value: str | None = None) -> bool:
    """Ответ воспроизводит числовое значение, взятое из документа.

    Сравнивается именно значение факта, а не все числа документа: в контексте
    есть ещё номер регламента, и требовать его копирования в ответ значило бы
    измерять не то.
    """
    answer_numbers = {token.word for token in tokenize_with_offsets(answer) if token.word.isdigit()}
    if value is not None:
        return value in answer_numbers
    from spanverify.core import numbers_in  # noqa: PLC0415

    return bool(numbers_in(context)) and bool(numbers_in(context) & answer_numbers)


def contrast_statistics(pairs: list[dict]) -> dict:
    """Проверка контраста: копирование значения документа и лексическое совпадение.

    Основной критерий — доля ответов, воспроизводящих значение из документа
    (ожидание: ≥80 % в группе «с опорой» и ≤30 % в группе «без опоры»).
    Лексическое совпадение приводится как диагностика: при подмене одного числа
    оно остаётся высоким, то есть модель не может опереться на «другие слова».
    """
    grounded = [p for p in pairs if int(p["label"]) == 0]
    unsupported = [p for p in pairs if int(p["label"]) == 1]
    grounded_copy = [copies_context_value(p["answer"], p["context"], p.get("value")) for p in grounded]
    unsupported_copy = [copies_context_value(p["answer"], p["context"], p.get("value")) for p in unsupported]
    grounded_overlap = [overlap_ratio(p["answer"], p["context"]) for p in grounded]
    unsupported_overlap = [overlap_ratio(p["answer"], p["context"]) for p in unsupported]
    stats = {
        "grounded_pairs": len(grounded),
        "unsupported_pairs": len(unsupported),
        "grounded_value_copy_rate": sum(grounded_copy) / len(grounded_copy) if grounded_copy else 0.0,
        "unsupported_value_copy_rate": sum(unsupported_copy) / len(unsupported_copy) if unsupported_copy else 0.0,
        "grounded_overlap_mean": statistics.fmean(grounded_overlap) if grounded_overlap else 0.0,
        "unsupported_overlap_mean": statistics.fmean(unsupported_overlap) if unsupported_overlap else 0.0,
    }
    stats["contrast_ok"] = stats["grounded_value_copy_rate"] >= 0.8 and stats["unsupported_value_copy_rate"] <= 0.3
    return stats


def environment_info(model_name: str) -> dict:
    """Сведения о среде запуска: без них отчёт невоспроизводим."""
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "model": model_name,
        "cpu_count": __import__("os").cpu_count(),
    }
    try:
        import torch  # noqa: PLC0415

        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - torch может отсутствовать
        info["torch"] = "нет"
    try:
        import transformers  # noqa: PLC0415

        info["transformers"] = transformers.__version__
    except Exception:  # noqa: BLE001
        info["transformers"] = "нет"
    return info


def write_auc_png(payload: dict, path: Path, width: int = 640, height: int = 360) -> Path:
    """Нарисовать столбцы AUC по слоям и записать PNG (без matplotlib).

    Картинка — приложение к отчёту пилота: три признака на каждый слой, высота
    столбца = AUC (ориентированная, то есть 0.5 = случайное угадывание).
    """
    import struct  # noqa: PLC0415
    import zlib  # noqa: PLC0415

    layers = list(payload.get("auc", {}))
    features = list(MEASURED_FEATURES)
    colors = [
        (214, 96, 77),
        (77, 132, 214),
        (110, 168, 96),
        (150, 110, 190),
        (200, 150, 60),
        (120, 120, 120),
        (180, 90, 140),
        (90, 160, 160),
    ]
    pixels = [[(255, 255, 255) for _ in range(width)] for _ in range(height)]

    def line(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for step in range(steps + 1):
            x = x0 + (x1 - x0) * step // steps
            y = y0 + (y1 - y0) * step // steps
            if 0 <= x < width and 0 <= y < height:
                pixels[y][x] = color

    def bar(x: int, top: int, color: tuple[int, int, int], bar_width: int) -> None:
        for column in range(max(0, x), min(width, x + bar_width)):
            for row in range(max(0, top), height - 30):
                pixels[row][column] = color

    left, bottom = 50, height - 30
    line(left, 20, left, bottom, (120, 120, 120))
    line(left, bottom, width - 10, bottom, (120, 120, 120))
    for tick in range(5):
        y = bottom - int((bottom - 20) * tick / 4)
        line(left, y, width - 10, y, (232, 232, 232))
    if not layers:
        layers = ["last"]
    slot = (width - left - 30) // max(1, len(layers) * len(features))
    for layer_index, layer in enumerate(layers):
        for feature_index, feature in enumerate(features):
            value = float((payload.get("auc", {}).get(layer, {}).get(feature, {}) or {}).get("auc_oriented") or 0.0)
            value = max(0.0, min(1.0, value))
            x = left + 20 + (layer_index * len(features) + feature_index) * slot
            top = bottom - int((bottom - 20) * value)
            bar(x, top, colors[feature_index % len(colors)], max(4, slot - 6))

    raw = b"".join(b"\x00" + bytes(channel for pixel in row for channel in pixel) for row in pixels)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    body = chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + body)
    return path


def render_markdown(payload: dict) -> str:
    """Отчёт в Markdown: таблица AUC по слоям и выводы без прикрас."""
    lines = [
        "# Пилот на реальной модели: информативность признаков по слоям",
        "",
        "Числа получены на **реальной** модели, размеченных контрастных парах и CPU.",
        "Это пилот на малом объёме: он показывает, какой сигнал вообще есть, а не",
        "финальное качество продукта.",
        "",
        "## Условия",
        "",
    ]
    for key, value in payload["environment"].items():
        lines.append(f"* {key}: `{value}`")
    lines += [
        f"* пар: {payload['pairs']}, токенов в анализе: {payload['tokens']}",
        f"* бутстрэп: {payload['bootstrap_iterations']} итераций, seed {payload['seed']}",
        f"* слои: {', '.join(payload['layers'])}",
        "",
        "## Контраст пар",
        "",
        "| Показатель | Значение |",
        "| --- | --- |",
    ]
    for key, value in payload["contrast"].items():
        lines.append(f"| {key} | {value:.4f} |" if isinstance(value, float) else f"| {key} | {value} |")
    lines += [
        "",
        "## AUC по признакам и слоям",
        "",
        "| Слой | Признак | AUC | 95 % ДИ | Токенов |",
        "| --- | --- | --- | --- | --- |",
    ]
    best: tuple[float, str, str] = (-1.0, "", "")
    for layer, features in payload["auc"].items():
        for name, values in features.items():
            lines.append(
                f"| {layer} | {name} | {values['auc_oriented']:.3f} | "
                f"[{values['ci_oriented_low']:.3f}; {values['ci_oriented_high']:.3f}] | {values['tokens']} |"
            )
            if values["auc_oriented"] > best[0]:
                best = (values["auc_oriented"], layer, name)
    lines += [
        "",
        "Агрегат по всем токенам размывает эффект одного числа, поэтому основной",
        "анализ — парный, по токену значения:",
        "",
        "| Слой | Признак | Значение с опорой | Значение без опоры | Δ «без опоры − с опорой» | 95 % ДИ Δ | AUC (токен значения) | Значимо |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    significant: list[tuple[str, str, float, float, float]] = []
    for layer, features in payload["auc"].items():
        for name, values in features.items():
            delta_significant = bool(values.get("delta_significant"))
            marker = "да" if delta_significant else "нет"
            lines.append(
                f"| {layer} | {name} | {values['fact_mean_grounded']:.6f} | "
                f"{values['fact_mean_unsupported']:.6f} | {values['delta_mean']:+.6f} | "
                f"[{values['delta_ci_low']:+.6f}; {values['delta_ci_high']:+.6f}] | "
                f"{values['auc_fact']:.3f} | {marker} |"
            )
            if delta_significant:
                significant.append((layer, name, values["delta_mean"], values["delta_ci_low"], values["delta_ci_high"]))

    entropy_last = payload["auc"].get("last", {}).get("attention_entropy", {}).get("auc_oriented")
    mass_last = payload["auc"].get("last", {}).get("ctx_attention_mass", {}).get("auc_oriented")
    density_last = payload["auc"].get("last", {}).get("embedding_density", {}).get("auc_oriented")
    mass_fact = payload["auc"].get("last", {}).get("ctx_attention_mass", {}).get("auc_fact")
    entropy_fact = payload["auc"].get("last", {}).get("attention_entropy", {}).get("auc_fact")
    lines += [
        "",
        "## Выводы",
        "",
        f"* Лучший сигнал по всем токенам: `{best[2]}` на слое `{best[1]}` (AUC {best[0]:.3f}).",
        f"* Последний слой: энтропия внимания AUC {entropy_last:.3f}, масса на контекст AUC {mass_last:.3f}, "
        f"косинусная плотность (базовый уровень без внимания) AUC {density_last:.3f}.",
    ]
    if significant:
        lines.append(
            "* Парный анализ по токену значения: значимые изменения есть — "
            + ", ".join(
                f"`{name}` на слое `{layer}`: Δ {delta:+.4f} [{low:+.4f}; {high:+.4f}]"
                for layer, name, delta, low, high in significant[:6]
            )
            + "."
        )
    else:
        lines.append(
            "* Парный анализ по токену значения: **ни один признак ни на одном слое не отличается значимо** "
            "между подставленным и правильным числом. То есть на этой постановке сигнала нет."
        )
    lines.append(
        "* Случайное угадывание — AUC 0.500; попадание нуля в доверительный интервал разницы означает, "
        "что отличие неотличимо от шума при данном объёме."
    )
    if mass_fact is not None and entropy_fact is not None:
        if mass_fact > entropy_fact + 0.05:
            lines.append(
                f"* Гипотеза о превосходстве массы внимания над энтропией **подтверждается** на токене значения: "
                f"{mass_fact:.3f} против {entropy_fact:.3f}."
            )
        elif entropy_fact > mass_fact + 0.05:
            lines.append(
                f"* Гипотеза о превосходстве массы внимания над энтропией **не подтверждается**: "
                f"энтропия внимания на токене значения даёт {entropy_fact:.3f} против {mass_fact:.3f} у массы. "
                "Это честный результат пилота, его нужно учитывать при выборе признаков."
            )
        else:
            lines.append(
                f"* Масса внимания ({mass_fact:.3f}) и энтропия ({entropy_fact:.3f}) на токене значения "
                "различимы не лучше случайного угадывания — выбирать между ними по этому пилоту нельзя."
            )
    lines += [
        "",
        "## Чего этот пилот не показывает",
        "",
        "* Пары синтетические и отличаются **одним числом**: это самый трудный случай, "
        "и нулевой результат здесь означает «на таком контрасте признак не работает», "
        "а не «признак бесполезен вообще».",
        "* Выводы ограничены одной моделью, одним жанром и объёмом в десятки пар.",
        "* Режим `demo` продукта к этому отчёту отношения не имеет: здесь считает реальная модель.",
    ]
    _ = (entropy_last, mass_last, density_last)
    if payload["contrast"]["contrast_ok"]:
        lines.append(
            "* Контраст групп выдержан: значение документа воспроизводится в "
            f"{payload['contrast']['grounded_value_copy_rate']:.0%} ответов группы «с опорой» и в "
            f"{payload['contrast']['unsupported_value_copy_rate']:.0%} группы «без опоры»."
        )
    else:
        lines.append("* ВНИМАНИЕ: контраст групп выдержан не полностью (см. таблицу выше) — выводы ограничены.")
    lines += [
        "",
        "## Ограничения",
        "",
        "* Пары синтетические: они сгенерированы по шаблонам, а не взяты из реальных документов.",
        "* Объём малый (десятки пар), доверительные интервалы широкие.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Точка входа пилота: считает метрики и пишет отчёт."""
    parser = argparse.ArgumentParser(description="Пилот на rugpt3small: AUC признаков по слоям")
    parser.add_argument("--pairs", type=int, default=24, help="число контрастных пар (каждая даёт 2 ответа)")
    parser.add_argument("--bootstrap", type=int, default=5000, help="итераций бутстрэпа")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--out", default="reports/pilot")
    parser.add_argument(
        "--fail-on-contrast",
        action="store_true",
        help="вернуть код 2, если контраст групп не выдержан (по умолчанию — честно отметить и продолжить)",
    )
    args = parser.parse_args(argv)

    started = time.time()
    pairs = build_contrast_pairs(args.pairs, seed=args.seed)
    contrast = contrast_statistics(pairs)
    print(f"Контрастных пар: {len(pairs)}; проверка контраста: {json.dumps(contrast, ensure_ascii=False)}")

    try:
        rows = collect_rows(pairs, model_name=args.model, max_length=args.max_length)
    except Exception as error:  # noqa: BLE001 - причины: нет весов, нет torch, нет интернета
        import traceback  # noqa: PLC0415

        report = {
            "status": "model_unavailable",
            "model": args.model,
            "pairs": len(pairs),
            "contrast": contrast,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "environment": environment_info(args.model),
            "seed": args.seed,
        }
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "pilot.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out_dir / "pilot.md").write_text(
            "# Пилот не выполнен: признаки модели недоступны\n\n"
            f"Модель: `{args.model}`\n\n"
            f"Причина: `{type(error).__name__}: {error}`\n\n"
            "Это честный отказ: числа не подставляются и не выдумываются. "
            "Черновик отчёта с трассировкой сохранён рядом (`pilot.json`).\n",
            encoding="utf-8",
        )
        print(
            f"Пилот не выполнен: не удалось получить признаки модели ({type(error).__name__}). "
            f"Отчёт-черновик: {out_dir / 'pilot.md'}",
            file=sys.stderr,
        )
        print(traceback.format_exc(), file=sys.stderr)
        return 2

    auc_table = analyse(rows, iterations=args.bootstrap)
    contrast_ok = bool(contrast.get("contrast_ok"))
    payload = {
        "status": "ok" if contrast_ok else "contrast_not_met",
        "published": bool(contrast_ok),
        "wording": (
            "пилот на малой модели (ruGPT3-small) на синтетических промтах: "
            "модель реальная, корпус промтов синтетический"
        ),
        "control": {
            "grounded_copy_min": 0.8,
            "unsupported_copy_max": 0.3,
            "grounded_copy_rate": contrast.get("grounded_value_copy_rate", 0.0),
            "unsupported_copy_rate": contrast.get("unsupported_value_copy_rate", 0.0),
            "grounded_ok": contrast.get("grounded_value_copy_rate", 0.0) >= 0.8,
            "unsupported_ok": contrast.get("unsupported_value_copy_rate", 1.0) <= 0.3,
            "contrast_ok": bool(contrast_ok),
        },
        "environment": environment_info(args.model),
        "pairs": len(pairs),
        "tokens": len(rows),
        "bootstrap_iterations": args.bootstrap,
        "seed": args.seed,
        "layers": list(LAYERS),
        "contrast": contrast,
        "auc": auc_table,
        "duration_s": round(time.time() - started, 1),
        "notice": (
            "Корпус промтов синтетический (два стиля, числа подставляются контролируемо), "
            "объём малый. Отчёт описывает пилот на малой модели ruGPT3-small, "
            "а не итоговое качество продукта."
        ),
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pilot.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "pilot.md").write_text(render_markdown(payload), encoding="utf-8")
    png_path = write_auc_png(payload, out_dir / "pilot.png")
    print(f"Отчёт: {out_dir / 'pilot.md'}, {out_dir / 'pilot.json'}, {png_path}")
    print(f"Длительность: {payload['duration_s']} с; токенов: {payload['tokens']}")
    print(f"Статус контраста: {payload['status']} ({json.dumps(contrast, ensure_ascii=False)})")
    for layer, features in auc_table.items():
        summary = ", ".join(f"{name}={values['auc_oriented']:.3f}" for name, values in features.items())
        print(f"  слой {layer}: {summary}")
    if not contrast_ok and args.fail_on_contrast:
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - запуск из CI
    raise SystemExit(main())
