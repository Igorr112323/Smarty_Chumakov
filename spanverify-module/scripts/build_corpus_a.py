#!/usr/bin/env python3
"""Сборка корпуса A: управляемые подмены в ответах по документам.

Корпус A — **наш** корпус: документы и ошибки в ответах создаются скриптом, поэтому
известна каждая метка и ясен каждый класс ошибки. Он нужен для того, чего нельзя
сделать на внешнем бенчмарке: обучать и калибровать (на корпусе B запрещено),
проверять отдельные типы ошибок и следить за балансом классов.

Как устроено
------------

* Документ — набор фактов (субъект, признак, значение). Из фактов собирается текст.
* Ответ строится из фактов одним из режимов:

  ==================  ===========================================================
  режим               что происходит с ответом
  ==================  ===========================================================
  ``faithful``        пересказ без искажений (чистые пары, их должно быть ≥ 40 %)
  ``contradiction``   значение заменено на другое (подмена) — ``Contradiction``
  ``partial``         часть условия отброшена, смысл искажён — ``Partial``
  ``number_attribution`` число взято из факта **другого** субъекта — подтип подмены
  ``unconfirmed``     добавлено утверждение, которого нет в документе — ``Unconfirmed``
  ``missing``         обязательная деталь пропущена — ``Missing``
  ``oversight``       важное значение потеряно при пересказе — ``Oversight``
  ``excess``          добавлена лишняя деталь, которой нет в документе — ``Excess``
  ==================  ===========================================================

* Разбиение на train/dev/test — **по документам** (одна группа = один документ),
  поэтому утечки между частями нет: ``shared_groups == 0`` проверяется в манифесте.
* Всё детерминировано: ``--seed``; повторный запуск даёт байт-в-байт тот же корпус.

Документы
---------

Каталог ``--docs`` содержит ``*.md`` / ``*.txt``. Если каталога нет, скрипт пишет
**синтетические** документы сам (``--gen-docs``) — они помечены ``"synthetic": true``
и не выдаются за выдержки из нормативных актов. Внешние документы подключаются
позже, для подтипа A2 (см. ``scripts/build_corpus_a2.py`` и ``docs/КОРПУС_v1.md``).

Запуск::

    python scripts/build_corpus_a.py --docs data/corpus_a/docs --out data/corpus_a \\
        --target 1200 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.corpus_real import (  # noqa: E402 - импорт после правки sys.path
    RealFact,
    build_context,
    build_real_variant,
    check_contexts_in_sources,
    check_number_attribution,
    extract_facts,
    normalize_text,
    pair_sha256,
)

DATASET_VERSION = "corpus-a1.0"
DATASET_VERSION_A3 = "corpus-a3.0"
CONTEXT_SEPARATOR = "\n\n"

MODES = (
    "faithful",
    "contradiction",
    "partial",
    "number_attribution",
    "unconfirmed",
    "missing",
    "oversight",
    "excess",
)

# Подписи для meta: как режим называется в таксономии бенчмарка.
TAXONOMY = {
    "faithful": "faithful",
    "contradiction": "Contradiction",
    "partial": "Partial",
    "number_attribution": "Contradiction",
    "unconfirmed": "Unconfirmed",
    "missing": "Missing",
    "oversight": "Oversight",
    "excess": "Excess",
}

SUBJECTS = [
    "первичные учётные документы",
    "вторичные учётные документы",
    "кадровые документы",
    "договоры с контрагентами",
    "акты выполненных работ",
    "счета-фактуры",
    "журнал регистрации доверенностей",
    "инвентарные карточки",
    "протоколы согласования",
    "листы ознакомления сотрудников",
]

FEATURES = [
    ("срок хранения", ["пять", "семь", "десять", "пятнадцать", "двадцать пять", "сорок пять"]),
    ("срок восстановления после сбоя", ["два часа", "четыре часа", "сутки", "трое суток"]),
    ("периодичность резервного копирования", ["каждый час", "раз в сутки", "раз в неделю"]),
    ("срок ответа на запрос контрагента", ["три рабочих дня", "пять рабочих дней", "десять рабочих дней"]),
    ("предельный объём одного вложения", ["десять мегабайт", "двадцать пять мегабайт", "пятьдесят мегабайт"]),
]

CONDITIONS = [
    "если иное не установлено договором",
    "если документ содержит сведения ограниченного доступа",
    "если срок не продлён решением комиссии",
    "если контрагент подтвердил продление",
]

# Дополнительные требования: используются в режимах ``excess`` и ``unconfirmed``.
EXTRA_DETAILS = [
    "архивная копия на внешнем носителе",
    "опись вложений в двух экземплярах",
    "отметка службы безопасности",
    "реестр передачи в архив",
]


@dataclass(frozen=True)
class Fact:
    """Один факт документа: субъект, признак, значение и, иногда, условие."""

    subject: str
    feature: str
    value: str
    condition: str | None = None

    def sentence(self) -> str:
        """Предложение документа, в котором этот факт изложен."""
        base = f"Для {self.subject} {self.feature} составляет {self.value}"
        if self.condition:
            base += f" {self.condition}"
        return base + "."

    def response(self) -> str:
        """Естественный ответ на вопрос по этому факту (без искажений)."""
        base = f"{self.feature.capitalize()} для {self.subject} — {self.value}"
        if self.condition:
            base += f", {self.condition}"
        return base + "."


def generate_documents(count: int, rng: random.Random) -> list[dict]:
    """Синтетические документы: несколько фактов на документ, без выдуманных ссылок на НПА."""
    documents: list[dict] = []
    for index in range(count):
        subjects = rng.sample(SUBJECTS, 4)
        facts: list[Fact] = []
        used: set[tuple[str, str]] = set()
        for subject in subjects:
            for feature, values in rng.sample(FEATURES, 2):
                if (subject, feature) in used:
                    continue
                used.add((subject, feature))
                condition = rng.choice(CONDITIONS) if rng.random() < 0.65 else None
                facts.append(Fact(subject, feature, rng.choice(values), condition))
        title = f"Регламент учёта № {index + 1}"
        preamble = (
            f"{title}. Настоящий документ определяет правила учёта и хранения. "
            "Он подготовлен генератором корпуса и не является выдержкой из нормативного акта."
        )
        text = CONTEXT_SEPARATOR.join([preamble, *[fact.sentence() for fact in facts]])
        documents.append(
            {
                "id": f"doc-{index + 1:04d}",
                "title": title,
                "text": text,
                "facts": facts,
                "synthetic": True,
            }
        )
    return documents


def load_sources_meta(docs_dir: Path) -> dict[str, dict]:
    """Прочитать ``sources.json`` загрузчика: ``{doc_id: метаданные источника}``.

    Файл пишет ``scripts/fetch_npa_corpus.py`` (URL, вид акта, номер, дата, хеши). Если
    его нет — это корпус A1, и метаданные не нужны.
    """
    path = docs_dir / "sources.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    documents = payload.get("documents")
    return documents if isinstance(documents, dict) else {}


def load_documents(docs_dir: Path) -> list[dict]:
    """Прочитать документы из каталога (рекурсивно).

    Файлы из подкаталога ``generated`` помечаются синтетическими: так признак не
    теряется при повторном запуске, когда генератор уже не создаёт документы заново.
    Рядом с файлами может лежать ``sources.json`` загрузчика — тогда метаданные
    источника (URL, вид, номер, дата, хеш) попадают в ``document["meta"]``, а текст
    нормализуется той же функцией, что и в проверках A3.
    """
    documents: list[dict] = []
    sources_meta = load_sources_meta(docs_dir)
    for path in sorted(docs_dir.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        synthetic = "generated" in path.relative_to(docs_dir).parts
        document = {
            "id": path.stem,
            "title": path.stem,
            "text": text if synthetic else normalize_text(text),
            "synthetic": synthetic,
        }
        if not synthetic and path.stem in sources_meta:
            document["meta"] = dict(sources_meta[path.stem])
        documents.append(document)
    return documents


def _find_span(answer: str, fragment: str) -> tuple[int, int]:
    """Смещения фрагмента в ответе (фрагмент обязан присутствовать дословно)."""
    start = answer.find(fragment)
    if start < 0:
        raise ValueError(f"фрагмент {fragment!r} отсутствует в ответе {answer!r}")
    return start, start + len(fragment)


def build_variant(fact: Fact, other: Fact | None, mode: str, rng: random.Random) -> dict:
    """Собрать ответ и разметку для одного режима.

    Возвращает ``{"answer", "labels", "spans_text"}``: метки — смещения символов в
    ответе, ``spans_text`` нужен только для проверки в тестах и отчёте.
    """
    labels: list[list[int]] = []
    spans_text: list[str] = []

    def mark(answer: str, fragment: str) -> None:
        start, end = _find_span(answer, fragment)
        labels.append([start, end, 1])
        spans_text.append(fragment)

    if mode == "faithful":
        answer = fact.response()
    elif mode == "contradiction":
        alternative = rng.choice(
            [
                value
                for feature, values in FEATURES
                if feature == fact.feature
                for value in values
                if value != fact.value
            ]
        )
        answer = fact.response()
        wrong = answer.replace(fact.value, alternative, 1)
        answer = wrong
        mark(answer, alternative)
    elif mode == "number_attribution":
        if other is None or other.feature != fact.feature:
            raise ValueError("для подтипа «атрибуция числа» нужен второй факт с тем же признаком")
        answer = f"{fact.feature.capitalize()} для {fact.subject} — {other.value}."
        mark(answer, other.value)
    elif mode == "partial":
        answer = f"{fact.feature.capitalize()} для {fact.subject} — {fact.value}."
        if not fact.condition:
            raise ValueError("для режима partial нужен факт с условием")
        mark(answer, fact.value)
    elif mode == "unconfirmed":
        detail = f"Требуется {rng.choice(EXTRA_DETAILS)}."
        answer = fact.response() + " " + detail
        mark(answer, detail)
    elif mode == "missing":
        answer = f"{fact.feature.capitalize()} для {fact.subject} установлен."
        mark(answer, "установлен")
    elif mode == "oversight":
        answer = f"Для {fact.subject} правила учёта определены, значения приведены в регламенте."
        mark(answer, "значения приведены в регламенте")
    elif mode == "excess":
        detail = f"дополнительно требуется {rng.choice(EXTRA_DETAILS)}"
        answer = fact.response().rstrip(".") + f", {detail}."
        answer = answer[0].upper() + answer[1:]
        mark(answer, detail)
    else:  # pragma: no cover - защита от опечатки в MODES
        raise ValueError(f"неизвестный режим: {mode}")

    return {"answer": answer, "labels": labels, "spans_text": spans_text}


def build_corpus(documents: list[dict], target: int, seed: int, mode_mix: dict[str, float] | None = None) -> list[dict]:
    """Набрать пары до ``target``, распределяя режимы по заданной смеси.

    Требования, которые проверяются в конце: не меньше 40 % чистых пар и не меньше
    40 пар на каждый режим с разметкой.
    """
    rng = random.Random(seed)
    default_mix = {
        "faithful": 0.45,
        "contradiction": 0.13,
        "partial": 0.10,
        "number_attribution": 0.10,
        "unconfirmed": 0.07,
        "missing": 0.06,
        "oversight": 0.05,
        "excess": 0.04,
    }
    mix = mode_mix or default_mix
    modes = list(mix)

    pairs: list[dict] = []
    attempts = 0
    max_attempts = target * 40
    while len(pairs) < target and attempts < max_attempts:
        attempts += 1
        document = documents[len(pairs) % len(documents)] if documents else None
        if document is None:
            break
        facts: list[Fact] = document["facts"]
        if len(facts) < 2:
            continue
        mode = rng.choices(modes, weights=[mix[item] for item in modes], k=1)[0]
        candidates = [item for item in facts if item.condition] if mode == "partial" else facts
        if not candidates:
            continue
        fact = rng.choice(candidates)
        others = [item for item in facts if item.feature == fact.feature and item.subject != fact.subject]
        other = rng.choice(others) if others else None
        if mode == "number_attribution" and other is None:
            continue
        try:
            variant = build_variant(fact, other, mode, rng)
        except ValueError:
            continue
        pair_id = f"corpus-a-{len(pairs) + 1:05d}"
        pairs.append(
            {
                "id": pair_id,
                "context": document["text"],
                "answer": variant["answer"],
                "labels": variant["labels"],
                "meta": {
                    "kind": "corpus_a",
                    "mode": mode,
                    "taxonomy": TAXONOMY[mode],
                    "clean": mode == "faithful",
                    "group": document["id"],
                    "subject": fact.subject,
                    "feature": fact.feature,
                    "value": fact.value,
                    "span_texts": variant["spans_text"],
                    "dataset_version": DATASET_VERSION,
                    "synthetic_document": bool(document.get("synthetic")),
                },
            }
        )
    return pairs


def split_pairs(pairs: list[dict], seed: int, ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)) -> dict:
    """Разбить пары по группам (документам) так, чтобы группы не пересекались."""
    groups: dict[str, list[dict]] = {}
    for pair in pairs:
        groups.setdefault(pair["meta"]["group"], []).append(pair)
    names = sorted(groups)
    random.Random(seed + 1).shuffle(names)

    total = len(pairs)
    train_target = round(total * ratios[0])
    dev_target = round(total * ratios[1])
    splits: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for name in names:
        block = groups[name]
        if len(splits["train"]) < train_target:
            splits["train"].extend(block)
        elif len(splits["dev"]) < dev_target:
            splits["dev"].extend(block)
        else:
            splits["test"].extend(block)
    return splits


def check_balance(pairs: list[dict], min_per_mode: int = 40, min_clean_share: float = 0.40) -> dict:
    """Проверить требования к составу корпуса и вернуть статистику."""
    counts: dict[str, int] = {mode: 0 for mode in MODES}
    for pair in pairs:
        counts[pair["meta"]["mode"]] += 1
    clean = counts["faithful"]
    share = clean / len(pairs) if pairs else 0.0
    problems: list[str] = []
    if share < min_clean_share:
        problems.append(f"чистых пар {clean} из {len(pairs)} — меньше {min_clean_share:.0%}")
    for mode, count in counts.items():
        if mode != "faithful" and count < min_per_mode:
            problems.append(f"режим {mode}: {count} пар — меньше {min_per_mode}")
    return {
        "counts": counts,
        "clean": clean,
        "clean_share": round(share, 4),
        "min_per_mode": min_per_mode,
        "problems": problems,
    }


def sha256_file(path: Path) -> str:
    """SHA256 файла по фактическим байтам."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, pairs: list[dict]) -> None:
    """Записать пары в JSONL (по одной записи на строку)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(pair, ensure_ascii=False) + "\n" for pair in pairs), encoding="utf-8")


def check_shared_groups(splits: dict) -> int:
    """Сколько групп встречается больше чем в одной части (должно быть 0)."""
    seen: dict[str, set[str]] = {}
    for name, pairs in splits.items():
        for pair in pairs:
            seen.setdefault(pair["meta"]["group"], set()).add(name)
    return sum(1 for parts in seen.values() if len(parts) > 1)


def check_spans(pairs: list[dict]) -> dict:
    """Проверить, что каждый спан в метках совпадает с ``answer[start:end]``.

    Заготовка ``meta.span_texts`` заполняется генератором ДО пересчёта, поэтому
    проверка здесь независима: если бы пересчёт среза давал другой текст, это была
    бы ошибка генерации, а не совпадение «само с собой».
    """
    checked = 0
    problems: list[str] = []
    for pair in pairs:
        answer = pair["answer"]
        texts = pair["meta"]["span_texts"]
        if len(texts) != len(pair["labels"]):
            problems.append(f"{pair['id']}: меток {len(pair['labels'])}, текстов {len(texts)}")
        for index, (start, end, label) in enumerate(pair["labels"]):
            checked += 1
            if label != 1:
                problems.append(f"{pair['id']}: метка {label} вместо 1")
            if not (0 <= start < end <= len(answer)):
                problems.append(f"{pair['id']}: границы [{start}, {end}] вне ответа длиной {len(answer)}")
                continue
            fragment = answer[start:end]
            if index < len(texts) and fragment != texts[index]:
                problems.append(f"{pair['id']}: срез {fragment!r} != записанного {texts[index]!r}")
    return {"checked": checked, "problems": problems}


def build(docs_dir: Path, out_dir: Path, target: int, seed: int, gen_docs: int = 0) -> dict:
    """Полный цикл: документы → пары → сплиты → манифест."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    documents = load_documents(docs_dir)
    generated = False
    if not documents:
        documents = generate_documents(max(gen_docs, 60), random.Random(seed))
        generated_dir = docs_dir / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)
        for document in documents:
            (generated_dir / f"{document['id']}.md").write_text(document["text"] + "\n", encoding="utf-8")
        generated = True

    pairs = build_corpus(documents, target=target, seed=seed)
    for index, pair in enumerate(pairs, start=1):
        pair["id"] = f"corpus-a-{index:05d}"

    # Пересобираем span_texts по фактическим срезам, чтобы манифест не зависел от генератора.
    for pair in pairs:
        pair["meta"]["span_texts"] = [pair["answer"][int(start) : int(end)] for start, end, _ in pair["labels"]]

    splits = split_pairs(pairs, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / "pairs.jsonl"
    write_jsonl(pairs_path, pairs)
    split_paths: dict[str, Path] = {}
    for name, block in splits.items():
        path = out_dir / "splits" / f"{name}.jsonl"
        write_jsonl(path, block)
        split_paths[name] = path

    balance = check_balance(pairs)
    shared_groups = check_shared_groups(splits)
    spans = check_spans(pairs)

    manifest = {
        "dataset_version": DATASET_VERSION,
        "target": target,
        "seed": seed,
        "documents": {
            "count": len(documents),
            "generated": generated,
            "synthetic": all(document.get("synthetic") for document in documents),
            "sha256": {
                document["id"]: hashlib.sha256(document["text"].encode("utf-8")).hexdigest() for document in documents
            },
        },
        "pairs": len(pairs),
        "balance": balance,
        "splits": {name: len(block) for name, block in splits.items()},
        "shared_groups": shared_groups,
        "spans_checked": spans["checked"],
        "spans_problems": spans["problems"],
        "sha256": {
            "pairs.jsonl": sha256_file(pairs_path),
            **{f"splits/{name}.jsonl": sha256_file(path) for name, path in split_paths.items()},
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Сборка корпуса A (управляемые подмены)")
    parser.add_argument("--docs", type=Path, default=ROOT / "data" / "corpus_a" / "docs")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "corpus_a")
    parser.add_argument("--target", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gen-docs", type=int, default=60, help="сколько документов сгенерировать, если каталог пуст")
    args = parser.parse_args()

    manifest = build(args.docs, args.out, args.target, args.seed, gen_docs=args.gen_docs)
    balance = manifest["balance"]
    print(f"Пар: {manifest['pairs']} (документов {manifest['documents']['count']}, цель {manifest['target']})")
    print("Режимы: " + ", ".join(f"{mode}={count}" for mode, count in balance["counts"].items() if count))
    print(f"Чистых: {balance['clean']} ({balance['clean_share']:.1%})")
    print(f"Сплиты: {manifest['splits']} | общих групп: {manifest['shared_groups']}")
    if manifest["spans_problems"]:
        print("Проблемы разметки:", manifest["spans_problems"][:3], file=sys.stderr)
        return 2
    if balance["problems"]:
        print("Не выполнены требования состава:", balance["problems"], file=sys.stderr)
        return 2
    print(f"Манифест: {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
