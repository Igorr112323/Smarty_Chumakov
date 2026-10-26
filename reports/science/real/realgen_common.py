"""REALGEN-R1: общие детерминированные константы и правила (без LLM-судьи).

Этот файл — часть замораживаемого кода фазы A. Любое изменение правил
разметки/выборки после заморозки запрещено; sha256 файла лежит в
REALGEN_FROZEN.json.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # /home/user/Smarty_Chumakov
REAL = ROOT / "reports" / "science" / "real"
SEED = 42
MODEL = "ai-forever/rugpt3small_based_on_gpt2"
MODEL_MAX_POSITIONS = 1024

# 8 шаблонов запросов (дословно из директивы REALGEN-R1).
TASKS = (
    "Резюмируй главу нормативного акта.",
    "Перечисли штрафы, установленные актом.",
    "Дай точные сроки, установленные актом.",
    "Какие документы нужны по этому акту?",
    "Опиши условия применения акта.",
    "Назови ключевые требования акта.",
    "Кто является субъектами регулирования?",
    "Опиши порядок обжалования по акту.",
)

TEMPLATE_IDS = (
    "summarize_chapter",
    "list_penalties",
    "give_deadlines",
    "required_documents",
    "application_conditions",
    "key_requirements",
    "regulated_subjects",
    "appeal_procedure",
)

MODES = ("greedy", "top-p 0.95 temperature 0.8")


PROMPT_V1 = (
    "Ты — помощник юриста. Ответь, опираясь ТОЛЬКО на приведённый фрагмент "
    "нормативного правового акта. Не выдумывай числа, даты, суммы, сроки и "
    "номера статей.\n"
    "Акт: {act_name}\n"
    "Фрагмент акта:\n"
    "---\n{context}\n---\n"
    "Задание: {task}\n"
    "Ответ (по тексту акта, с конкретными числами и ссылками на статьи):\n"
)

PROMPT_V2_HARD = (
    "Ты — помощник юриста. Дай развёрнутый ответ, опираясь ТОЛЬКО на "
    "приведённый фрагмент нормативного правового акта.\n"
    "Акт: {act_name}\n"
    "Фрагмент акта:\n"
    "---\n{context}\n---\n"
    "Задание: {task}\n"
    "В ответе обязательно укажи не менее пяти конкретных числовых значений "
    "(суммы, сроки, проценты, даты), не менее трёх точных номеров статей, "
    "пунктов или частей, а также перечисли все подходящие случаи и органы. "
    "Приведи точные цитаты с номерами статей и пунктов.\n"
    "Ответ:\n"
)


def prompt_v1(task: str, act_name: str, context: str) -> str:
    """Базовый промт раунда R1. Один и тот же для обоих режимов генерации."""
    return PROMPT_V1.format(task=task, act_name=act_name, context=context)


def prompt_v2_hard(task: str, act_name: str, context: str) -> str:
    """Раунд R2 — ТОЛЬКО remedy квоты ошибок: сложность промта повышена.

    Никаких инъекций: модель сама порождает все числа и ссылки. Повышенная
    сложность запроса — единственное отличие от R1 (разрешённая директивой
    правка «промт-сложность», правила разметки не меняются).
    """
    return PROMPT_V2_HARD.format(task=task, act_name=act_name, context=context)


def h(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def h_int(text: str) -> int:
    return int(h(text), 16)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_size(path: Path) -> int:
    return Path(path).stat().st_size


def write_json(path: Path, payload: object) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_rule_ids(entries: list[dict]) -> tuple[list[str], list[str], list[str]]:
    """Восстановление 300 dev / 1200 test из CORPUS_FROZEN.json и generate_phase_b_v2.py.

    Правило (дословно из кода generate_phase_b_v2.py, SEED=42):
      selected = sorted(entries, key=sha256(f"42:{id}"))[:1500]
      ordered  = sorted(selected_ids, key=sha256(f"42:split:{id}"))
      dev = ordered[:300], test = ordered[300:1500]
    Возвращает (dev_ids, test_ids, background_ids) без чтения запертого файла
    real_test_pairs.jsonl.
    """
    ids = [entry["id"] for entry in entries]
    selected = sorted(ids, key=lambda doc_id: h(f"{SEED}:{doc_id}"))[:1500]
    ordered = sorted(selected, key=lambda doc_id: h(f"{SEED}:split:{doc_id}"))
    dev = ordered[:300]
    test = ordered[300:1500]
    background = sorted(set(ids) - set(test))
    return dev, test, background


def select_sample(background_ids: list[str], n: int = 400) -> list[str]:
    """≥400 документов детерминированно sha256(seed:doc_id, seed 42)."""
    ordered = sorted(background_ids, key=lambda doc_id: h(f"{SEED}:{doc_id}"))
    assert len(ordered) >= n, "background слишком мал"
    return ordered[:n]


def assignment(doc_ids: list[str], round_id: str) -> list[dict]:
    """Ровно 25 документов на каждую из 16 комбинаций (8 шаблонов × 2 режима).

    combo = index % 16; template_id = combo // 2; mode = combo % 2.
    """
    out = []
    for index, doc_id in enumerate(doc_ids):
        combo = index % (len(TEMPLATE_IDS) * len(MODES))
        out.append(
            {
                "index": index,
                "doc_id": doc_id,
                "round": round_id,
                "template_index": combo // len(MODES),
                "template_id": TEMPLATE_IDS[combo // len(MODES)],
                "mode_index": combo % len(MODES),
                "mode": MODES[combo % len(MODES)],
                "gen_seed": h_int(f"{SEED}:gen:{round_id}:{doc_id}") % (2**31 - 1),
            }
        )
    return out


def context_window(text: str, doc_id: str, chars: int = 1100) -> tuple[str, int, int]:
    """Детерминированное окно акта, помещающееся в 1024 позиции GPT-2.

    Модель ruGPT3small имеет n_positions=1024; промт + 512 новых токенов
    должны укладываться. Берём окно из ~1100 символов (≈350-450 русских
    субтокенов), выбранное по sha256(doc_id), и добавляем первые 200 символов
    акта (название/преамбула).
    """
    clean = " ".join(text.split())
    if len(clean) <= chars:
        return clean, 0, len(clean)
    start = h_int(f"{SEED}:window:{doc_id}") % max(1, len(clean) - chars)
    window = clean[start : start + chars]
    prefix = clean[:200]
    return prefix + " … " + window, start, start + len(window)
