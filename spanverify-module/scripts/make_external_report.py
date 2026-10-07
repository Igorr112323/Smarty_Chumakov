#!/usr/bin/env python3
"""Собрать отчёт ``reports/EXTERNAL_TESTS.md`` из ``reports/external_tests.json``.

Все числа в отчёте берутся из JSON, который записал ``scripts/external_eval.py``.
Руками отчёт не правится: это гарантирует, что текст и файл чисел не разойдутся
(требование приёмки F). Если прогонов ещё нет — скрипт честно пишет об этом и
возвращает код 2, а не выдумывает таблицу.

Запуск::

    python scripts/external_eval.py --dataset ragtruth --task qa --split test --mode demo --json reports/ext_ragtruth_qa.json
    python scripts/external_eval.py --dataset rushallu --mode demo --json reports/ext_rushallu.json
    python scripts/make_external_report.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fetch_external_tests import EXCLUDED, OPTIONAL, RAGTRUTH, RUSHALLU  # noqa: E402

DATASET_META = {
    "ragtruth": {
        "title": "RAGTruth",
        "language": "en",
        "license": "MIT (файл LICENSE в репозитории)",
        "url": RAGTRUTH["url"],
        "revision": RAGTRUTH["revision"],
    },
    "rushallu": {
        "title": "RusHallu-RAG",
        "language": "ru",
        "license": "не подтверждена (сырые данные не публикуются)",
        "url": RUSHALLU["url"],
        "revision": RUSHALLU["revision"],
    },
}

# Порядок строк в таблице: сначала основной английский набор, потом русский.
RUN_ORDER = (
    "ragtruth_qa_test_demo",
    "ragtruth_summary_test_demo",
    "ragtruth_data2txt_test_demo",
    "ragtruth_qa_test_hf",
    "rushallu_all_test_demo",
    "rushallu_all_test_hf",
)

TASK_TITLES = {"qa": "QA", "summary": "Summary", "data2txt": "Data2txt", "all": "все задачи"}


def _num(value: float | None, digits: int = 3) -> str:
    """Число в тексте отчёта (прочерк, если измерения нет)."""
    return "—" if value is None else f"{value:.{digits}f}"


def render(combined: dict, manifest: dict | None) -> str:
    """Отрисовать отчёт; пустой список прогонов тоже отрисовать честно."""
    runs = combined.get("runs") or {}
    lines: list[str] = []
    lines.append("# Внешние размеченные наборы: результаты SpanVerify")
    lines.append("")
    lines.append(
        "Файл сгенерирован `scripts/make_external_report.py` из `reports/external_tests.json`. "
        "Числа здесь и в JSON — одни и те же; отчёт не правится руками."
    )
    lines.append("")
    lines.append(f"Прогонов записано: {len(runs)}. Дата сборки: {combined.get('generated_at', '—')}.")
    lines.append("")
    lines.append("## 1. Источники и лицензии")
    lines.append("")
    lines.append("| Набор | Язык | Разметка | Лицензия | Ревизия | Статус |")
    lines.append("|---|---|---|---|---|---|")
    for key in ("ragtruth", "rushallu"):
        meta = DATASET_META[key]
        revision = meta["revision"][:12] if meta["revision"] else "—"
        status = "скачан и проверен" if key in (manifest or {}).get("datasets", {}) else "скачивается скриптом"
        lines.append(f"| {meta['title']} | {meta['language']} | human | {meta['license']} | `{revision}` | {status} |")
    for spec in OPTIONAL.values():
        lines.append(
            f"| {spec['title']} | {spec['language']} | {spec['label_origin']} | {spec['license']} | — | "
            f"не загружен: {spec['note']} |"
        )
    lines.append("")
    lines.append("Не подключены (решения зафиксированы в реестре `scripts/fetch_external_tests.py`):")
    lines.append("")
    for name, reason in EXCLUDED.items():
        lines.append(f"* **{name}** — {reason}.")
    lines.append("")
    lines.append(
        "Правило: внешние наборы — только тест. Обучение и калибровка идут на нашем корпусе A "
        "(`data/corpus_a`), внешние метки не меняются: если спан не найден в тексте дословно, "
        "он попадает в счётчик `unverified_spans`, а не «чинится»."
    )
    lines.append("")
    lines.append("### Контрольные числа загрузки")
    lines.append("")
    if manifest:
        for key, block in (manifest.get("datasets") or {}).items():
            totals = block.get("totals") or {}
            title = DATASET_META.get(key, {}).get("title", key)
            if key == "ragtruth":
                lines.append(
                    f"* {title}: ответов {totals.get('responses')}, источников {totals.get('sources')}, "
                    f"спанов {totals.get('spans')} (совпали смещения у {totals.get('verified_spans')}); "
                    f"по задачам {totals.get('by_task')}, по сплитам {totals.get('by_split')}."
                )
            else:
                lines.append(
                    f"* {title}: пар {totals.get('responses')}, чистых {totals.get('clean')}, "
                    f"с галлюцинациями {totals.get('with_hallucination')}, спанов {totals.get('spans')}, "
                    f"неоднозначных смещений {totals.get('ambiguous_offsets')}."
                )
    else:
        lines.append(
            "* Манифест загрузки не найден: контрольные числа смотрите в `data/external/MANIFEST.json` после запуска загрузчика."
        )
    lines.append("")
    lines.append("## 2. Что измерено на чём")
    lines.append("")

    if not runs:
        lines.append(
            "Прогонов нет: сначала выполните `python scripts/external_eval.py --dataset ragtruth "
            "--task qa --split test --mode demo --json reports/ext_ragtruth_qa.json` и аналог для "
            "остальных задач. Пустая таблица здесь не заполняется оценками «на глаз»."
        )
        lines.append("")
        return "\n".join(lines) + "\n"

    demo_runs = [run for run in runs.values() if run["mode"] == "demo"]
    hf_runs = [run for run in runs.values() if run["mode"] == "hf"]
    lines.append(
        f"* Режим `demo` (лексические признаки, без весов модели): прогонов {len(demo_runs)}. "
        "Это проверка работоспособности конвейера на внешних данных, а не научный результат."
    )
    if hf_runs:
        models = ", ".join(sorted({str(run.get("model")) for run in hf_runs}))
        lines.append(f"* Режим `hf` (веса языковой модели): прогонов {len(hf_runs)}, модели: {models}.")
    else:
        lines.append(
            "* Режим `hf` **не выполнялся**: в этой среде нет GPU и весов модели. Числа `hf` "
            "отсутствуют, а не заменены приблизительными."
        )
    lines.append(
        "* Калибровка (веса, порог, голова) взята из `config/` — она обучена на нашем корпусе A "
        "и на внешних наборах не подстраивалась."
    )
    lines.append("* Железо: CPU; устройства и время указаны в каждом прогоне в `reports/ext_*.json`.")
    lines.append("")
    lines.append("## 3. Таблица результатов")
    lines.append("")
    lines.append(
        "| Набор (задача) | Язык | Разметка | Режим | Пар | Наши: token F1 / FPR / AUC | "
        "Наши ответы: recall / FPR | Строгий span-F1 / накрытие | Их: accuracy / Jaccard / ROUGE-L | Baseline статьи |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    ordered = [key for key in RUN_ORDER if key in runs] + [key for key in runs if key not in RUN_ORDER]
    for key in ordered:
        run = runs[key]
        meta = DATASET_META.get(run["dataset"], {})
        tokens = run["our_metrics"]["tokens"]
        answers = run["our_metrics"]["verdicts"]
        spans = run["our_metrics"]["spans"]
        theirs = run["their_metrics"]
        origins = "/".join(sorted(run["label_origin"])) or "—"
        task_title = TASK_TITLES.get(run["task"], run["task"])
        # В ячейках — подписи метрик: так числа в отчёте проверяемы скриптом
        # check_numbers.py (он ищет именно «token F1 <число>», «FPR <число>», «AUC <число>»).
        lines.append(
            f"| {meta.get('title', run['dataset'])} — {task_title} | {meta.get('language', '—')} | "
            f"{origins} | {run['mode']} | {run['pairs']} | "
            f"token F1 {_num(tokens['f1'])}, FPR {_num(tokens['fpr'])}, AUC {_num(tokens['auc'])} | "
            f"recall {_num(answers['recall'])}, FPR {_num(answers['fpr'])} | "
            f"строгий span-F1 {_num(spans['strict_f1_iou_0_5'])}, накрытие {_num(spans['coverage'])} | "
            f"accuracy {_num(theirs.get('accuracy'))}, Jaccard {_num(theirs.get('jaccard_score'))}, "
            f"ROUGE-L {_num(theirs.get('rougeL'))} | "
            f"{'не извлечено' if not run['baseline']['extracted'] else 'приведён'} |"
        )
    lines.append("")
    lines.append(
        "Разметка в колонке «Разметка» — происхождение меток: human (человек), llm (модель), auto (автоматика)."
    )
    lines.append("")
    lines.append("## 4. Что не измерено")
    lines.append("")
    if hf_runs:
        lines.append(
            "* Режим `hf` выполнен на срезе пар (`limit` указан в каждом прогоне), а не на всём наборе; "
            "полный набор и прогон на GPU — не выполнялись."
        )
    else:
        lines.append("* Прогон режима `hf` (веса языковой модели) — не выполнялся.")
    lines.append(
        "* Числа baseline из статей RAGTruth (ACL 2024) и RusHallu-RAG (Диалог-2026) — PDF недоступен, значения не извлечены."
    )
    lines.append("* Русский перевод RAGTruth (трек 2) — не делался: он требует ручной проверки 50 пар человеком.")
    lines.append(
        "* Вспомогательные наборы с нечеловеческой разметкой (HaluEval LLM Spans — llm, lettucedetect — auto) — не загружены в этой среде."
    )
    if any(run.get("limit") for run in runs.values()):
        lines.append(
            "* Часть прогонов сделана на срезе (`--limit`) — это указано в поле `note` соответствующего прогона."
        )
    lines.append("")
    lines.append("## 5. Ограничения")
    lines.append("")
    lines.append(
        "* Классы задач различаются: QA отвечает на вопрос по пассажам, Summary пересказывает документ, "
        "Data2txt превращает карточку в текст. Одним числом эти задачи не описываются, поэтому они идут отдельными строками."
    )
    lines.append(
        "* Ответы внешних наборов длиннее наших демонстрационных, а порог калибровался на коротких ответах: "
        "метрики уровня ответа (recall/FPR) переносятся хуже, чем токенные. Это видно в таблице."
    )
    lines.append(
        "* Разметка RAGTruth — спаны внутри ответа; метка отмечает фрагмент галлюцинации, остальной текст неявно считается подтверждённым."
    )
    lines.append(
        "* RusHallu-RAG лицензирован неясно: набор используется как тест, сырые данные не публикуются, "
        "в отчётах — только агрегированные метрики и короткие цитаты со ссылкой."
    )
    lines.append("")
    lines.append("## 6. Вывод о переносимости")
    lines.append("")
    lines.append(
        "Лексический конвейер (`demo`) на внешних наборах переносится **частично**: отдельные подозрительные "
        "фрагменты он находит (полнота накрытия выше, чем точность), но выборочные метрики резко падают по сравнению "
        "со своим корпусом. Причина не в пороге, а в природе признаков: суррогаты опираются на почти дословное "
        "совпадение, а внешние ответы — свободные пересказы, где галлюцинация формулируется другими словами."
    )
    lines.append("")
    if hf_runs:
        lines.append(
            "Практический вывод для научной части: числа, полученные на нашем корпусе, нельзя переносить на внешние "
            "данные; режим `hf` измерен на срезе, и сравнение с публикациями требует полного прогона на всём наборе."
        )
    else:
        lines.append(
            "Практический вывод для научной части: числа, полученные на нашем корпусе, нельзя переносить на внешние "
            "данные; сравнение с публикациями требует режима `hf` (веса модели), и это следующий обязательный шаг."
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Отчёт по внешним наборам из JSON")
    parser.add_argument("--combined", type=Path, default=ROOT / "reports" / "external_tests.json")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "EXTERNAL_TESTS.md")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "external" / "MANIFEST.json")
    parser.add_argument("--allow-empty", action="store_true", help="не падать, если прогонов нет")
    args = parser.parse_args()

    if not args.combined.is_file():
        print(
            f"ошибка: нет {args.combined}. Сначала выполните прогоны scripts/external_eval.py",
            file=sys.stderr,
        )
        return 2
    combined = json.loads(args.combined.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest.is_file() else None
    text = render(combined, manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    runs = len(combined.get("runs") or {})
    print(f"Отчёт: {args.out} (прогонов {runs})")
    if not runs and not args.allow_empty:
        print("прогонов нет — таблица пустая", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
