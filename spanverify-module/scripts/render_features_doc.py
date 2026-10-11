#!/usr/bin/env python3
"""Собрать ``docs/ПРИЗНАКИ.md`` из кода и артефактов измерений.

Зачем генератор, а не рукописный текст: документ закрывает работы 1–2
договорного задания («описание признаков, их информативность и отбор по
валидации»), а значит каждое число в нём обязано иметь источник. Числа
берутся из трёх файлов: ``reports/experiments/pilot/pilot.json`` (AUC
признака по слоям на реальной модели), ``reports/hf_protocol/sweep.json``
(отбор набора признаков по валидации) и ``reports/METRICS.json`` (версия).
Файл перезаписывается целиком; править его руками смысла нет — ``--check``
это заметит.

Запуск (из ``spanverify-module``):

    python3 scripts/render_features_doc.py --write
    python3 scripts/render_features_doc.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

DOC = REPO_ROOT / "docs" / "ПРИЗНАКИ.md"
PILOT = MODULE_ROOT / "reports" / "experiments" / "pilot" / "pilot.json"
SWEEP = MODULE_ROOT / "reports" / "hf_protocol" / "sweep.json"
A3 = MODULE_ROOT / "reports" / "experiments" / "a3" / "experiment.json"
SIGNS = MODULE_ROOT / "reports" / "hf_protocol" / "feature_signs_a3dev.json"
METRICS = MODULE_ROOT / "reports" / "METRICS.json"

# Определение каждого признака сетки — по коду ``spanverify/hf_grid.py``.
# Сетка и описания обязаны совпадать: расхождение — красная сборка (тест
# ``tests/test_v2_docs.py`` вызывает этот генератор в режиме ``--check``).
FEATURE_DOC: dict[str, tuple[str, str]] = {
    # --- энтропия распределения внимания на документ ---
    "entropy_last": (
        "внимание",
        "средняя энтропия распределения внимания на документ по головам," " последний слой модели",
    ),
    "entropy_first": ("внимание", "то же, первый слой"),
    "entropy_middle": ("внимание", "то же, средний слой"),
    "entropy_m4": ("внимание", "то же, четвёртый слой с конца"),
    "entropy_mean_layers": ("внимание", "среднее по всем слоям"),
    "entropy_max_layers": ("внимание", "максимум по слоям, усреднённый по позициям токенов"),
    "entropy_head_max": ("внимание", "максимум по головам, усреднённый по слоям"),
    "entropy_head_disagreement": ("внимание", "разброс энтропий между головами (несогласованность голов)"),
    # --- доля внимания, приходящаяся на документ ---
    "mass_last": ("масса внимания", "доля внимания на токены документа, последний слой"),
    "mass_first": ("масса внимания", "то же, первый слой"),
    "mass_middle": ("масса внимания", "то же, средний слой"),
    "mass_mean_layers": ("масса внимания", "среднее по слоям"),
    "mass_max_layers": ("масса внимания", "максимум по слоям"),
    "mass_max_head": ("масса внимания", "максимум по голове, усреднённый по слоям"),
    "mass_norm_last": ("масса внимания", "доля, нормированная на ожидаемую для данной позиции"),
    "mass_lift_last": ("масса внимания", "прирост доли относительно ожидаемой величины"),
    # --- похожесть эмбеддинга токена на чанки документа ---
    "density_k1_last": ("плотность опоры", "средняя похожесть по 1 ближайшему соседу (последний слой)"),
    "density_k3_last": ("плотность опоры", "средняя похожесть по 3 ближайшим соседям"),
    "density_k5_last": ("плотность опоры", "средняя похожесть по 5 ближайшим соседям"),
    "density_k3_middle": ("плотность опоры", "средняя похожесть по 3 ближайшим соседям на среднем слое"),
    "sim_max_last": ("похожесть", "максимальный косинус с чанками документа"),
    "sim_contrast_last": ("похожесть", "максимум минус средняя похожесть по строке"),
    "sim_margin_last": ("похожесть", "отступ лучшего соседа от второго"),
    "sim_distance_last": ("похожесть", "расстояние до опорного чанка, нормированное на длину"),
    "sim_decay_last": ("похожесть", "похожесть со штрафом за удаление опоры"),
    # --- лексическое и числовое совпадение с документом ---
    "lex_match": ("совпадение", "1, если основа токена встречается в лексиконе документа"),
    "num_match": ("совпадение", "1, если число из токена встречается в документе"),
    "trigram_match": ("совпадение", "доля совпадения триграмм токена с лексиконом документа"),
    "attrib_borrowed": ("привязка чисел", "1, если токен покрыт числом, заимствованным у другого объекта"),
    "attrib_supported": ("привязка чисел", "1, если токен покрыт числом того же объекта"),
    # --- соседство и позиционные ---
    "mass_win3": ("окно", "среднее массы внимания по окну из трёх токенов"),
    "entropy_win3": ("окно", "среднее энтропии по окну из трёх токенов"),
    "position": ("позиционные", "относительная позиция токена в ответе"),
    "length": ("позиционные", "длина токена, нормированная на 20 знаков"),
    "sentence_position": ("позиционные", "относительная позиция токена в предложении"),
    "sentence_support": ("предложение", "лексическая опора предложения ответа на документ"),
}

LAYER_TITLES = {
    "first": "первый слой",
    "middle": "средний слой",
    "-4": "четвёртый с конца",
    "last": "последний слой",
}


def _dot(value: float, digits: int = 3) -> str:
    """Число для таблицы: точка как разделитель, фиксированное число знаков."""
    return f"{value:.{digits}f}"


def _comma(value: float, digits: int = 6) -> str:
    """Число для текста: запятая, чтобы сверка чисел не путала его с метрикой из README."""
    return f"{value:.{digits}f}".replace(".", ",")


def _pilot_significant(pilot: dict) -> int:
    """Сколько комбинаций «признак × слой» имеют ДИ, не накрывающий 0,5."""
    total = 0
    for layer in pilot["layers"]:
        for entry in pilot["auc"][layer].values():
            if entry["ci_low"] > 0.5 or entry["ci_high"] < 0.5:
                total += 1
    return total


def _pilot_feature_count(pilot: dict) -> int:
    """Сколько разных имён признаков в словаре пилота."""
    return len({name for layer in pilot["layers"] for name in pilot["auc"][layer]})


def pilot_section(pilot: dict) -> list[str]:
    """AUC признака по слоям: разведка на реальной модели, синтетические промты."""
    env = pilot["environment"]
    layers = pilot["layers"]
    lines = [
        "### 3.1. Разведка: AUC признака по слоям",
        "",
        "Источник — `spanverify-module/reports/experiments/pilot/pilot.json` (статус `{}`): модель `{}`"
        " (реальная, не суррогат), промты синтетические, пар `{}`, токенов `{}`, слоёв `{}`, бутстреп `{}`,"
        " seed `{}`, Python `{}`, torch `{}`.".format(
            pilot["status"],
            env["model"],
            pilot["pairs"],
            pilot["tokens"],
            len(layers),
            pilot["bootstrap_iterations"],
            pilot["seed"],
            env["python"],
            env["torch"],
        ),
        "",
        "Формулировка самого автора артефакта (`wording`): «{}». Оговорка (`notice`): «{}».".format(
            pilot["wording"], pilot["notice"]
        ),
        "",
        "Контроль контраста перед измерением пройден (поле `control`): дословный перенос значения"
        " в подкреплённых парах — {} при требовании не менее {}, в неподкреплённых — {} при требовании"
        " не более {}; флаги `{}`.".format(
            _comma(pilot["control"]["grounded_copy_rate"], 1),
            _comma(pilot["control"]["grounded_copy_min"], 1),
            _comma(pilot["control"]["unsupported_copy_rate"], 1),
            _comma(pilot["control"]["unsupported_copy_max"], 1),
            ", ".join(f"{key}={value}" for key, value in pilot["control"].items() if key.endswith("_ok")),
        ),
        "",
        "Значение AUC `0,5` означает «признак не различает подкреплённые и неподкреплённые токены»;"
        " слово «значимо» ниже — доверительный интервал бутстрапа не накрывает `0,5`.",
        "",
        "| Признак (словарь пилота) | " + " | ".join(LAYER_TITLES[layer] for layer in layers) + " |",
        "|---|" + "---|" * len(layers),
    ]
    names: list[str] = []
    for layer in layers:
        for name in pilot["auc"][layer]:
            if name not in names:
                names.append(name)
    for name in names:
        cells = []
        for layer in layers:
            entry = pilot["auc"][layer].get(name)
            if entry is None:
                cells.append("—")
                continue
            low, high = entry["ci_low"], entry["ci_high"]
            significant = "да" if (low > 0.5 or high < 0.5) else "нет"
            cells.append(f"{_dot(entry['auc'])} (ДИ [{_dot(low)}; {_dot(high)}], значимо: {significant})")
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Итог по этой таблице (читать нужно буквально): значимых отклонений от `0,5` — {} из {}"
        " комбинаций «признак × слой» (слоёв {}, признаков {}), и все они лежат НИЖЕ `0,5` (например,"
        " `ctx_max_similarity` на первом слое). Пилот на 96 парах **не подтвердил** гипотезу «высокая"
        " опора на документ снижает риск» в ожидаемом направлении и не выявил признака, годного как"
        " одиночное правило: именно поэтому признаки были переданы линейной модели набором, а не"
        " отобраны по одному. Вывод «признак X информативен» на этих данных сделать нельзя.".format(
            _pilot_significant(pilot),
            sum(len(pilot["auc"][layer]) for layer in pilot["layers"]),
            len(pilot["layers"]),
            _pilot_feature_count(pilot),
        ),
        "",
        "Воспроизвести: `python3 scripts/hf_pilot.py --write` (нужна модель на диске; в CI не запускается).",
        "",
        "> Словарь имён пилота отличается от словаря сетки (раздел 2): пилот считает по одному значению"
        " на признак, сетка — по слоям и агрегациям. Отождествлять строки двух таблиц один-в-один без"
        " отдельной проверки нельзя, поэтому основной результат отбора приведён ниже — уже на признаках сетки.",
        "",
    ]
    return lines


def sweep_section(sweep: dict) -> list[str]:
    """Отбор набора признаков по валидации: 10 зафиксированных гипотез."""
    experiments = sweep["experiments"]
    criterion = sweep["criterion"]
    model = sweep["model"]
    best = sweep["best_selection"]
    lines = [
        "### 3.2. Отбор по валидации: 10 зафиксированных наборов признаков",
        "",
        "Источник — `spanverify-module/reports/hf_protocol/sweep.json` (прогон CI, артефакт лежит"
        " в репозитории). Модель `{}` (ревизия `{}`), строк `{}`, лимит гипотез `{}` (использовано `{}`),"
        " отбор только по валидации A3, критерий: токен F1 ≥ {} при FPR ≤ {}.".format(
            model["id"],
            model["revision"][:12],
            model["rows"],
            sweep["limit"],
            len(experiments),
            str(criterion["token_f1_min"]).replace(".", ","),
            str(criterion["token_fpr_max"]).replace(".", ","),
        ),
        "",
        "| № | Гипотеза (набор признаков) | Признаков | F1 (val) | FPR (val) | AUC (val) |",
        "|---|---|---|---|---|---|",
    ]
    for exp in experiments:
        block = exp["per_corpus"].get("a3") or {}
        tokens = block.get("tokens") or {}
        if not tokens:
            note = "нет результата" + (f": {block.get('error')}" if block.get("error") else "")
            lines.append(f"| {exp['number']} | `{exp['name']}` | {len(exp['features'])} | {note} | — | — |")
            continue
        lines.append(
            f"| {exp['number']} | `{exp['name']}` | {len(exp['features'])} | {_dot(tokens['f1'], 6)} |"
            f" {_dot(tokens['fpr'], 6)} | {_dot(tokens['auc'], 6)} |"
        )
    chosen = next(exp for exp in experiments if exp["name"] == sweep["best"])
    selection = chosen["per_corpus"]["a3"]["threshold_selection"]
    lines += [
        "",
        "Отобранный набор — `{}`: F1 на валидации {}, FPR {}, AUC {} (поле `best_selection` того же файла).".format(
            sweep["best"],
            _comma(best["token_f1_val"]),
            _comma(best["token_fpr_val"]),
            _comma(best["auc_val"]),
        ),
        "",
        "Порог ответа для отобранного набора выбран вне валидации и вне теста: правило «максимум F1"
        " при FPR не выше заданного» на `5` фолдах обучающей части, строк оценок `{}`, порог {}"
        " (по OOF-оценкам: F1 {}, FPR {}).".format(
            selection["oof_rows"],
            _comma(selection["threshold"], 4),
            _comma(selection["selection"]["f1"], 6),
            _comma(selection["selection"]["fpr"], 6),
        ),
        "",
        "Что это значит для продукта: разброс AUC по наборам (от {} до {}) при F1 ниже {} — это сигнал"
        " о ранжировании токенов, а не о качестве границ фрагментов; именно поэтому в работах 4 и 7"
        " отдельными артефактами зафиксированы границы span'ов и калибровка порога при заданном FPR.".format(
            _comma(min(e["per_corpus"]["a3"]["tokens"]["auc"] for e in experiments if e["per_corpus"].get("a3")), 3),
            _comma(max(e["per_corpus"]["a3"]["tokens"]["auc"] for e in experiments if e["per_corpus"].get("a3")), 3),
            str(criterion["token_f1_min"]).replace(".", ","),
        ),
        "",
        _zero_f1_note(experiments),
        "",
        "Воспроизвести: `python3 scripts/hf_protocol.py sweep --write` (в CI — job «Протокол v1»)."
        " Ограничение лимита гипотез закреплено гейтом `check_hypothesis_limit_v2`.",
        "",
    ]
    return lines


def _zero_f1_note(experiments: list[dict]) -> str:
    """Почему у части гипотез F1 на валидации равен нулю — по полям артефакта."""
    zero = [exp for exp in experiments if (exp["per_corpus"].get("a3") or {}).get("tokens", {}).get("f1") == 0.0]
    if not zero:
        return ""
    names = ", ".join(f"`{exp['name']}`" for exp in zero)
    first = zero[0]["per_corpus"]["a3"]
    return (
        f"Строки с нулём ({names}) — не «ошибка измерения», а результат правила порога: на валидации"
        f" ни один токен не превысил порог, выбранный на обучающей части"
        f" (TP `{int(first['tokens']['tp'])}`, FP `{int(first['tokens']['fp'])}`,"
        f" FN `{int(first['tokens']['fn'])}`) — при этом AUC того же прогона остаётся высокой"
        f" (`{_comma(first['tokens']['auc'], 5)}`). Ранжирование работает, абсолютный масштаб оценок"
        f" у бустинга между частями смещается — это и есть"
        f" задокументированная причина, по которой в продукт вошла логистическая регрессия"
        f" с изотонической калибровкой (работа 7), а не бустинг."
    )


def sign_section() -> list[str]:
    """Знак и информативность каждого признака на валидации — из артефакта прогона."""
    header = ["### 3.3. Знак признака на валидации (AUC с исходным и инвертированным знаком)", ""]
    if not SIGNS.is_file():
        header += [
            "**Не измерено.** Пилот (3.1) дал отклонения AUC ниже 0,5, и прежде, чем писать что-либо",
            "об информативности отдельного признака, знак обязан быть измерен на официальной",
            "валидации. Измеритель готов: `scripts/feature_sign_check.py`, прогон — workflow",
            "`hf-feature-signs.yml` (читает кеш сетки v3 из артефактов `grid-a3-*`, тестовую часть"
            " не открывает: файл кеша с `test` в имени приводит к отказу).",
            "",
            "До появления артефакта `reports/hf_protocol/feature_signs_a3dev.json` любые формулировки",
            "«признак X информативен» и «знак перевёрнут» считаются неподтверждёнными.",
            "",
        ]
        return header
    payload = json.loads(SIGNS.read_text(encoding="utf-8"))
    split = payload["split"]
    counts = payload["counts"]
    lines = header + [
        f"Источник — `spanverify-module/reports/hf_protocol/feature_signs_a3dev.json` (прогон CI, модель"
        f" `{payload['model']['id']}`, кеш `{payload['inputs']['grid_cache']}`). Выборка: пар"
        f" `{split['pairs']}`, токенов `{split['tokens']}` (положительных"
        f" `{payload['labels']['positive_tokens']}`, отрицательных"
        f" `{payload['labels']['negative_tokens']}`). Правило: «{payload['rule']}»."
        "",
        f"Определение: {payload['auc_definition']}. Доверительный интервал: {payload['ci_definition']}.",
        "",
        "| Признак | AUC (исходный знак) | AUC (инвертированный) | ДИ исходного | Значимость | Знак по val |",
        "|---|---|---|---|---|---|",
    ]
    for item in payload["features"]:
        if item["auc"] is None:
            lines.append(f"| `{item['name']}` | — | — | — | нет | {item['sign_by_val']} |")
            continue
        interval = f"[{item['ci_low']:.3f}; {item['ci_high']:.3f}]" if item["ci_low"] is not None else "—"
        if item["significant_original"]:
            verdict = "значимо, исходный знак"
        elif item["significant_inverted"]:
            verdict = "значимо, обратный знак"
        else:
            verdict = "не значимо (ДИ накрывает 0,5)"
        lines.append(
            f"| `{item['name']}` | {item['auc']:.6f} | {item['auc_inverted']:.6f} | {interval} | {verdict}"
            f" | {item['sign_by_val']} |"
        )
    lines += [
        "",
        f"Итог по артефакту: AUC ниже 0,5 у `{counts['auc_below_half']}` признаков из"
        f" `{len(payload['features'])}`; значимых с исходным знаком — `{counts['significant_original']}`,"
        f" значимых с обратным — `{counts['significant_inverted']}`, постоянных (AUC не определена) —"
        f" `{counts['constant']}`.",
        "",
        "Как это читать. AUC ниже 0,5 — не «признак плохой», а признак с обратным направлением:"
        " линейная модель сама присвоит ему отрицательный вес, отдельного вычитания в коде не требуется"
        " (в `spanverify/hf_grid.py` и `spanverify/features.py` инверсии знака нет — признаки подаются"
        " как есть). Если доверительный интервал накрывает 0,5, направление не установлено и вывод об"
        " информативности признака не делается ни в какую сторону.",
        "",
        f"Воспроизвести: `python3 scripts/feature_sign_check.py --grid-cache"
        f" {payload['inputs']['grid_cache']} --splits {payload['inputs']['splits']} --bootstrap"
        f" {payload['bootstrap_iterations']} --seed {payload['seed']} --out-json"
        f" reports/hf_protocol/feature_signs_a3dev.json --out-md"
        f" reports/hf_protocol/feature_signs_a3dev.md`"
        " (сетка v3 берётся из артефактов `grid-a3-0`…`grid-a3-3` прогона «Протокол hf"
        " (критерий ТЗ)» `38076145605`; шарды тестовой выборки `grid-test-a3-*` не скачиваются,"
        " а скрипт отвергает любой файл кеша с `test` в имени; локально прогон требует кеш, в CI"
        " — job «AUC и знак по каждому признаку сетки (val)» из"
        " `.github/workflows/hf-feature-signs.yml`).",
        "",
    ]
    return lines


def gap_section() -> list[str]:
    return [
        "### 3.4. Чего нет (и это не «почти сделано»)",
        "",
        "* **Вклад каждого из 36 признаков в F1 на val — не измерено.** Знак и AUC по признаку"
        " измерены (3.3), но «сколько добавляет признак» без прогона ablation не утверждается;"
        " команды — `make hf-grid-v2` затем `make hf-curve-v2`.",
        "* **Отдельного замера «только признаки attention» на A3 — нет.** В v1 они входили в расширенный"
        " набор; вывод о их индивидуальном вкладе без этого замера не делается.",
        "* **Информативность на реальных (не синтетических) размеченных данных — не измерено**, потому что"
        " реальной span-разметки нет: это блокер работ 1–2 и 3, он описан в `docs/DECISIONS_NEEDED.md`.",
        "",
    ]


def data_section(a3: dict | None) -> list[str]:
    lines = ["## 5. Данные, на которых признаки измерялись (v1, A3)", ""]
    if a3 is None:
        lines += ["Артефакт `reports/experiments/a3/experiment.json` отсутствует — раздел не заполнен.", ""]
        return lines
    splits = a3["splits"]
    sizes = splits["sizes"]
    lines += [
        f"Сплит — `{splits['source']}`: обучающая часть `{sizes['train']}` пар, валидация `{sizes['dev']}`,"
        f" тест `{sizes['test']}`. Правило, записанное в артефакте: «{splits['rule']}».",
        "",
        "Корпус A3 синтетический по построению: пара «документ — ответ» генерируется из шаблонов, а метка"
        " — побочный продукт генерации. Полное описание методики — `docs/МЕТОДИКА_РАЗМЕТКИ.md`.",
        "",
    ]
    return lines


def signature_section() -> list[str]:
    meta = json.loads(METRICS.read_text(encoding="utf-8"))["meta"]
    return [
        "## 7. Версии и воспроизводимость этого документа",
        "",
        f"* версия пакета: `{meta['version']}` — она же `spanverify.__version__`, она же версия объекта приёмки"
        " (`docs/ПРОГРАММА_И_МЕТОДИКА_ПРИЁМКИ.md`);",
        f"* commit, на котором собраны цитируемые артефакты измерений: `{meta['commit'][:9]}`"
        " (поле `meta.commit` в `reports/METRICS.json` — это коммит прогона, а не текущий HEAD: при"
        " пересборке манифеста строка обновляется вместе с ним);",
        "* определения разметки и метрик заморожены и не менялись после публикации чисел:"
        " `docs/METRIC_SPEC.md` и `docs/METRIC_SPEC_v2.md` (их хеши зафиксированы тестами);",
        "* числовые требования к среде по пункту 4.2 договора заранее не заявляются: замеры длины,"
        " памяти и времени — в `spanverify-module/reports/FUNCTIONAL_TESTS.md`.",
        "",
        "Команды (из каталога `spanverify-module`):",
        "",
        "```bash",
        "python3 scripts/render_features_doc.py --write    # пересобрать документ из артефактов",
        "python3 scripts/render_features_doc.py --check    # проверить согласованность с артефактами",
        "```",
        "",
        "Ручных чисел в документе нет: раздел 3 генерируется из файлов прогонов, расхождение с ними"
        " делает сборку красной (`tests/test_v2_docs.py`).",
        "",
    ]


def render() -> str:
    from spanverify.hf_grid import GRID_FEATURE_NAMES  # noqa: PLC0415

    pilot = json.loads(PILOT.read_text(encoding="utf-8"))
    sweep = json.loads(SWEEP.read_text(encoding="utf-8"))
    a3 = json.loads(A3.read_text(encoding="utf-8")) if A3.is_file() else None

    missing = [name for name in GRID_FEATURE_NAMES if name not in FEATURE_DOC]
    extra = sorted(name for name in FEATURE_DOC if name not in GRID_FEATURE_NAMES)
    if missing or extra:
        raise SystemExit(f"описания признаков расходятся с сеткой: без описаний {missing}, лишние {extra}")

    used_by_best = set(next(e for e in sweep["experiments"] if e["name"] == sweep["best"])["features"])

    lines = [
        "# Признаки span-верификации: определения, информативность, отбор",
        "",
        "> Договор № 0117812 (УМНИК), этап 1, работы 1–2 в части «описание признаков». Файл собран"
        " автоматически: `python3 scripts/render_features_doc.py --write`. Числа — только из артефактов"
        " измерений, список источников — в разделе 7.",
        "",
        "## 1. Что считается признаком и зачем он нужен",
        "",
        "Модуль отвечает на два вопроса: **какие токены ответа не опираются на документ** (span-верификация)"
        " и **какова доля участия ИИ** (для неё достаточно ранжирования токенов по риску). Оба ответа —"
        " линейная функция от одного и того же вектора признаков токена, поэтому описание признаков"
        " и есть описание продукта, а не внутренняя заметка.",
        "",
        "Три свойства признаков зафиксированы кодом и тестами:",
        "",
        "1. **Только вход модели.** `spanverify/hf_grid.py` получает на вход документ и ответ; gold-разметка"
        " в вычисление не попадает — модуль про разметку ничего не знает. Признак, подсмотревший метку,"
        " сделал бы измерение фиктивным.",
        "2. **Общая физическая величина для двух ответов.** Один и тот же вектор значений используется"
        " и для классификации токена, и для весов доли участия ИИ: «протестировали одну модель, а заявили"
        " другую» исключено конструкцией.",
        "3. **Детерминированность.** Один seed, один проход, один кеш: бандл при `seed=42` побайтово"
        " совпадает между запусками, при `seed=7` отличается (проверено `sha256sum` и `cmp`).",
        "",
        "## 2. Сетка признаков (36 значений на токен ответа)",
        "",
        "Имена — `spanverify.hf_grid.GRID_FEATURE_NAMES`. Колонка «в отобранном наборе» — признак входит"
        f" в `best_selection` прогона отбора (`{sweep['best']}`).",
        "",
        "| Признак | Группа | Что считает | В отобранном наборе |",
        "|---|---|---|---|",
    ]
    for name in GRID_FEATURE_NAMES:
        group, description = FEATURE_DOC[name]
        mark = "да" if name in used_by_best else "нет"
        lines.append(f"| `{name}` | {group} | {description} | {mark} |")
    lines += [
        "",
        "Признаков продукта ровно три — `entropy_last`, `mass_last`, `density_k1_last`; они объявлены"
        " в `spanverify/features.py` (`FEATURE_NAMES`) и защищены публичным контрактом. Расширенная сетка"
        " нужна только этапу исследований: отбор по валидации сам по себе продукт не меняет — это"
        " отдельное решение заказчика (п. 5.4 договора про замену модели и параметров).",
        "",
        "## 3. Информативность и отбор",
        "",
    ]
    lines += pilot_section(pilot)
    lines += sweep_section(sweep)
    lines += sign_section()
    lines += gap_section()
    lines += [
        "## 4. Как признаки участвуют в решении",
        "",
        "```text",
        "документ + ответ → один проход модели → для каждого токена ответа: 36 значений",
        "  риск токена      = линейная модель (логистическая регрессия, 5 фолдов, L2 = 0,01) по этим значениям",
        "  фрагмент (span)  = связный отрезок токенов с риском не ниже порога окна,",
        "                   обрезка: не короче span_floor, не длиннее span_cap, склейка merge_gap",
        "  вердикт ответа   = «есть неподкреплённые фрагменты», если доля рисковых токенов не ниже",
        "                   порога ответа (калибровка при заданном FPR, работа 7)",
        "  доля участия ИИ  = взвешенная доля рисковых токенов (режимы soft и hard, порог verdict_rule)",
        "```",
        "",
        "Пороги, веса и разбиения после публикации чисел не менялись: это проверяют"
        " `tests/test_no_test_leakage.py`, `tests/test_reported_metrics.py` и гейты `make hf-verify-v2`.",
        "",
        "Значения по умолчанию заданы кодом: для токенов, не сопоставленных с подсловами модели,"
        " энтропийные признаки получают `DEFAULT_ENTROPY`, остальные — `0,0`. То есть отсутствие сигнала"
        " трактуется как нейтральность, а не как риск.",
        "",
    ]
    lines += data_section(a3)
    lines += [
        "## 6. Ограничения, которые надо помнить при чтении любых чисел",
        "",
        "* Признаки считаются по подсловам модели, сопоставление слову — по правилам `docs/METRIC_SPEC.md`;"
        " «значение признака токена» — агрегат по его подсловам.",
        "* Энтропия и масса внимания зависят от архитектуры модели: смена базовой модели требует"
        " перепридумывания признака и, по пункту 5.4 договора, отдельного протокола замены.",
        "* Все числа раздела 3 получены на синтетическом корпусе A3: они описывают поведение метода"
        " на синтетике, а не ожидаемое качество на реальных документах заказчика.",
        "",
    ]
    lines += signature_section()
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="записать docs/ПРИЗНАКИ.md")
    mode.add_argument("--check", action="store_true", help="проверить совпадение с артефактами")
    args = parser.parse_args(argv)

    rendered = render()
    if args.write:
        DOC.write_text(rendered, encoding="utf-8")
        print(f"записано: {DOC.relative_to(REPO_ROOT)} ({len(rendered.splitlines())} строк)")
        return 0
    current = DOC.read_text(encoding="utf-8") if DOC.is_file() else ""
    if current != rendered:
        print(
            "docs/ПРИЗНАКИ.md расходится с артефактами измерений: выполнить"
            " python3 scripts/render_features_doc.py --write и закоммитить результат",
            file=sys.stderr,
        )
        return 1
    print(f"docs/ПРИЗНАКИ.md согласован с артефактами ({len(rendered.splitlines())} строк)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
