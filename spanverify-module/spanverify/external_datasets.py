"""Адаптеры внешних размеченных наборов к контракту SpanVerify.

Зачем отдельный модуль
----------------------

Внешние наборы (RAGTruth, RusHallu-RAG) устроены каждый по-своему, а наш конвейер
принимает один контракт: ``{"id", "context", "answer", "labels": [[start, end, 1]],
"meta"}`` со **символьными** смещениями. Здесь собрано всё, что превращает чужой
формат в наш, — и всё, что при этом может пойти не так:

* у RAGTruth поле ``source_info`` бывает и строкой (Summary), и структурой
  (QA — вопрос и пассажи, Data2txt — карточка организации), поэтому текст контекста
  извлекается явно, а не «как получится»;
* метка, которая не совпадает со срезом ответа, **не чинится молча**: она
  откладывается в ``meta["unverified"]`` и попадает в счётчик;
* происхождение разметки фиксируется в ``meta["label_origin"]``
  (``human`` / ``llm`` / ``auto``): LLM-разметку нельзя выдавать за человеческую.

Модуль ничего не скачивает и не оценивает — только переводит формат.
"""

from __future__ import annotations

import ast
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

# Типы меток RAGTruth (человеческая разметка, MIT). Порядок фиксирован для отчётов.
RAGTRUTH_LABEL_TYPES = (
    "Evident Baseless Info",
    "Evident Conflict",
    "Subtle Baseless Info",
    "Subtle Conflict",
)

# Происхождение разметки: колонка обязательна во всех отчётах по внешним наборам.
LABEL_ORIGINS = ("human", "llm", "auto")


class ExternalFormatError(ValueError):
    """Формат внешнего набора не совпал с ожидаемым."""


def text_offsets(answer: str, fragment: str, occurrence: int = 0) -> tuple[int, int] | None:
    """Найти смещения фрагмента в ответе; ``None`` — если фрагмента нет.

    ``occurrence`` — какой по счёту повтор брать (0 — первый). Внешние наборы
    иногда размечают фразу, которая встречается в ответе дважды; выбор первого
    вхождения совпадает с поведением авторов RusHallu-RAG, а сам факт
    неоднозначности считает :func:`count_ambiguous_spans`.
    """
    position = -1
    for _ in range(occurrence + 1):
        position = answer.find(fragment, position + 1)
        if position < 0:
            return None
    return position, position + len(fragment)


def count_ambiguous_spans(answer: str, fragment: str) -> int:
    """Сколько раз фрагмент встречается в ответе (1 — норма, больше — неоднозначность)."""
    return answer.count(fragment) if fragment else 0


# --------------------------------------------------------------------- RAGTruth


def ragtruth_context(source_info: Any) -> str:
    """Извлечь текст документа-контекста из поля ``source_info`` RAGTruth.

    Поле неоднородно (проверено на всех 2965 источниках): у Summary это строка,
    у QA — словарь ``{"question", "passages"}``, у Data2txt — карточка из девяти
    полей. Для проверки «ответ против документа» нужен именно документ, поэтому
    у QA берутся пассажи, а карточка Data2txt разворачивается в строки «ключ: значение».
    """
    if isinstance(source_info, str):
        return source_info
    if isinstance(source_info, dict):
        passages = source_info.get("passages")
        if isinstance(passages, str) and passages.strip():
            return passages
        lines: list[str] = []
        for key, value in source_info.items():
            if isinstance(value, (dict, list)):
                rendered = json.dumps(value, ensure_ascii=False)
            else:
                rendered = str(value)
            lines.append(f"{key}: {rendered}")
        return "\n".join(lines)
    raise ExternalFormatError(f"source_info неожиданного типа: {type(source_info).__name__}")


def adapt_ragtruth(response_row: dict, source_row: dict) -> dict:
    """Преобразовать пару «ответ — источник» RAGTruth в нашу запись.

    Метки, у которых ``response[start:end] != text``, не попадают в ``labels``:
    они сохраняются в ``meta["unverified"]`` вместе с позициями и текстом, чтобы
    их можно было пересчитать, а не потерять. Для RAGTruth таких не было ни одной
    (14289 из 14289 совпали), но правило действует всегда.
    """
    answer = str(response_row.get("response", ""))
    context = ragtruth_context(source_row.get("source_info", ""))
    labels: list[list[int]] = []
    verified_types: list[str] = []
    unverified: list[dict] = []
    ambiguous = 0
    for label in response_row.get("labels", []) or []:
        fragment = str(label.get("text", ""))
        start, end = int(label.get("start", -1)), int(label.get("end", -1))
        matches = 0 <= start < end <= len(answer) and fragment and answer[start:end] == fragment
        if not matches:
            unverified.append({"start": start, "end": end, "text": fragment, "label_type": label.get("label_type")})
            continue
        ambiguous += 1 if count_ambiguous_spans(answer, fragment) > 1 else 0
        labels.append([start, end, 1])
        verified_types.append(str(label.get("label_type", "")))
    return {
        "id": f"ragtruth-{response_row.get('id')}",
        "context": context,
        "answer": answer,
        "labels": labels,
        "meta": {
            "dataset": "ragtruth",
            "task": source_row.get("task_type"),
            "source": source_row.get("source"),
            "source_id": response_row.get("source_id"),
            "model": response_row.get("model"),
            "temperature": response_row.get("temperature"),
            "split": response_row.get("split"),
            "quality": response_row.get("quality"),
            "label_origin": "human",
            "label_types": verified_types,
            "unverified": unverified,
            "ambiguous_offsets": ambiguous,
        },
    }


def ragtruth_pairs(
    responses: Iterable[dict],
    sources: Iterable[dict],
    *,
    task: str | None = None,
    split: str | None = None,
    quality: str | None = "good",
) -> tuple[list[dict], dict]:
    """Собрать пары RAGTruth по фильтрам и посчитать статистику.

    ``quality="good"`` отбрасывает ответы с обрывами и «корректными отказами»:
    у них разметка не сопоставима с обычными ответами (29 и 144 строки).
    """
    source_index = {str(row.get("source_id")): row for row in sources}
    pairs: list[dict] = []
    stats: dict = {
        "responses": 0,
        "sources": 0,
        "spans": 0,
        "clean": 0,
        "with_hallucination": 0,
        "unverified_spans": 0,
        "ambiguous_offsets": 0,
        "by_task": {},
        "by_label_type": {},
    }
    seen_sources: set[str] = set()
    for row in responses:
        if split is not None and row.get("split") != split:
            continue
        if quality is not None and row.get("quality") != quality:
            continue
        source = source_index.get(str(row.get("source_id")))
        if source is None:
            raise ExternalFormatError(f"нет источника для source_id={row.get('source_id')}")
        if task is not None and source.get("task_type") != task:
            continue
        pair = adapt_ragtruth(row, source)
        pairs.append(pair)
        seen_sources.add(str(row.get("source_id")))
        stats["responses"] += 1
        stats["unverified_spans"] += len(pair["meta"]["unverified"])
        stats["ambiguous_offsets"] += int(pair["meta"]["ambiguous_offsets"] or 0)
        kinds = pair["meta"]["label_types"]
        if pair["labels"]:
            stats["with_hallucination"] += 1
            stats["spans"] += len(pair["labels"])
            for kind in kinds:
                stats["by_label_type"][kind] = stats["by_label_type"].get(kind, 0) + 1
        else:
            stats["clean"] += 1
        task_name = str(source.get("task_type"))
        block = stats["by_task"].setdefault(
            task_name, {"responses": 0, "clean": 0, "with_hallucination": 0, "spans": 0, "_sources": set()}
        )
        block["responses"] += 1
        block["_sources"].add(str(row.get("source_id")))
        if pair["labels"]:
            block["with_hallucination"] += 1
            block["spans"] += len(pair["labels"])
        else:
            block["clean"] += 1
    stats["sources"] = len(seen_sources)
    for block in stats["by_task"].values():
        block["sources"] = len(block.pop("_sources"))
    return pairs, stats


def ragtruth_totals(responses: Iterable[dict], sources: Iterable[dict]) -> dict:
    """Контрольные числа всего набора (без фильтров) — для самопроверки загрузки."""
    responses = list(responses)
    sources = list(sources)
    spans = 0
    by_type: Counter[str] = Counter()
    verified = 0
    for row in responses:
        answer = str(row.get("response", ""))
        for label in row.get("labels", []) or []:
            spans += 1
            by_type[str(label.get("label_type"))] += 1
            if answer[int(label.get("start", -1)) : int(label.get("end", -1))] == label.get("text"):
                verified += 1
    return {
        "responses": len(responses),
        "sources": len(sources),
        "spans": spans,
        "verified_spans": verified,
        "by_label_type": dict(by_type),
        "by_task": dict(Counter(str(row.get("task_type")) for row in sources)),
        "by_split": dict(Counter(str(row.get("split")) for row in responses)),
        "by_quality": dict(Counter(str(row.get("quality")) for row in responses)),
        "by_model": dict(Counter(str(row.get("model")) for row in responses)),
    }


# ----------------------------------------------------------------- RusHallu-RAG


def rushallu_context(documents: Sequence[dict]) -> str:
    """Склеить документы RusHallu-RAG в контекст, сохранив порядок и разделители."""
    parts: list[str] = []
    for index, document in enumerate(documents, start=1):
        parts.append(f"\n\n[ДОКУМЕНТ {index}]\n")
        parts.append(str(document.get("doc_text", "")))
    return "".join(parts).strip()


def rushallu_spans(answer: str, annotation: str) -> tuple[list[list[int]], list[str], int, list[dict]]:
    """Разметка RusHallu-RAG (JSON со спанами) → смещения символов.

    Возвращает ``(labels, types, ambiguous, unverified)``. Спан, не найденный в
    ответе дословно, попадает в ``unverified``, а не «исправляется» поиском похожего.
    """
    items = json.loads(annotation)
    labels: list[list[int]] = []
    types: list[str] = []
    unverified: list[dict] = []
    ambiguous = 0
    for item in items or []:
        fragment = str((item or {}).get("span") or "")
        if not fragment:
            continue
        offsets = text_offsets(answer, fragment)
        if offsets is None:
            unverified.append({"text": fragment, "type": (item or {}).get("type")})
            continue
        start, end = offsets
        if count_ambiguous_spans(answer, fragment) > 1:
            ambiguous += 1
        labels.append([start, end, 1])
        types.append(str((item or {}).get("type") or "unknown"))
    return labels, types, ambiguous, unverified


def adapt_rushallu(row: dict, source: str) -> dict:
    """Преобразовать строку CSV RusHallu-RAG в нашу запись.

    Разметка человеческая, ответы сгенерированы ``yandex/YandexGPT-5-Lite-8B-instruct`` —
    оба факта важны для отчёта, поэтому пишутся в ``meta``.
    """
    answer = str(row.get("model_output", ""))
    labels, types, ambiguous, unverified = rushallu_spans(answer, row.get("answer", "[]"))
    return {
        "id": f"rushallu-{source}-{row.get('query_id')}",
        "context": rushallu_context(
            # Поле ``docs`` — это Python-repr списка словарей (одинарные кавычки),
            # а не JSON: авторы складывали его через ``str()``. Проверено на всех
            # 1000 строках; json.loads на нём падает.
            ast.literal_eval(row["docs"])
            if isinstance(row.get("docs"), str)
            else list(row.get("docs") or [])
        ),
        "answer": answer,
        "labels": labels,
        "meta": {
            "dataset": "rushallu_rag",
            "task": "rag",
            "source": source,
            "query": row.get("query_text"),
            "query_id": row.get("query_id"),
            "label_origin": "human",
            "answer_origin": "yandexgpt-5-lite-8b",
            "types": types,
            "unverified": unverified,
            "ambiguous_offsets": ambiguous,
            "clean": not labels,
            "context_reference": row.get("joined_reference", ""),
        },
    }


def rushallu_pairs(rows: Iterable[dict], source: str) -> tuple[list[dict], dict]:
    """Собрать пары RusHallu-RAG и посчитать статистику (для сверки с внешним числом)."""
    pairs: list[dict] = []
    stats: dict = {
        "responses": 0,
        "spans": 0,
        "clean": 0,
        "with_hallucination": 0,
        "unverified_spans": 0,
        "ambiguous_offsets": 0,
        "by_type": {},
    }
    for row in rows:
        pair = adapt_rushallu(row, source)
        pairs.append(pair)
        stats["responses"] += 1
        stats["unverified_spans"] += len(pair["meta"]["unverified"])
        stats["ambiguous_offsets"] += int(pair["meta"]["ambiguous_offsets"] or 0)
        if pair["labels"]:
            stats["with_hallucination"] += 1
            stats["spans"] += len(pair["labels"])
            for kind in pair["meta"]["types"]:
                stats["by_type"][kind] = stats["by_type"].get(kind, 0) + 1
        else:
            stats["clean"] += 1
    return pairs, stats


# ------------------------------------------------------------------------- утилиты


def summarize_pairs(pairs: Sequence[dict]) -> dict:
    """Сводка по адаптированному набору: объём, разметка, происхождение."""
    origins = Counter(str((pair.get("meta") or {}).get("label_origin", "не указано")) for pair in pairs)
    labelled = [pair for pair in pairs if pair.get("labels")]
    return {
        "pairs": len(pairs),
        "clean": len(pairs) - len(labelled),
        "with_hallucination": len(labelled),
        "spans": sum(len(pair["labels"]) for pair in pairs),
        "unverified_spans": sum(len((pair.get("meta") or {}).get("unverified") or []) for pair in pairs),
        "label_origin": dict(origins),
        "answers_with_span": len(labelled),
    }


def validate_pairs(pairs: Sequence[dict], name: str = "внешний набор") -> list[str]:
    """Проверить, что каждая метка совпадает со срезом ответа (независимая проверка).

    Возвращает список проблем; пустой список — все метки корректны. Это та самая
    проверка, которую требует приёмка: «для каждого спана ``answer[start:end] == text``».
    """
    problems: list[str] = []
    for pair in pairs:
        answer = pair.get("answer", "")
        for start, end, label in pair.get("labels", []):
            if label != 1:
                problems.append(f"{pair.get('id')}: метка {label} вместо 1")
            if not (0 <= int(start) < int(end) <= len(answer)):
                problems.append(f"{pair.get('id')}: границы [{start}, {end}] вне ответа длиной {len(answer)}")
            elif (pair.get("meta") or {}).get("unverified"):
                problems.append(f"{pair.get('id')}: у пары есть непроверенные метки")
    if not pairs:
        problems.append(f"{name}: пустой набор")
    return problems
