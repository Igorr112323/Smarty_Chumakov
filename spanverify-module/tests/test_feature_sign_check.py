"""Тесты измерителя знака признаков (``scripts/feature_sign_check.py``).

Ценность проверок не в том, что скрипт «запускается», а в трёх свойствах, без
которых числа нельзя использовать:

1. знак признака с обратным направлением определяется как обратный, а не как
   «признак мусор»;
2. постоянный признак не даёт AUC (и не «дотягивается» до 0,5);
3. измеритель физически не читает тестовую часть: выборка — только val, а
   подстановка test-файла приводит к падению.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "feature_sign_check.py"
SIGNS_ARTIFACT = ROOT / "reports" / "hf_protocol" / "feature_signs_a3dev.json"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from spanverify.core import tokenize_with_offsets  # noqa: E402
from spanverify.hf_grid import GRID_FEATURE_NAMES  # noqa: E402

# Признаки, которыми тест различает направления: один «как надо», второй перевёрнут,
# третий постоянен. Остальные сетки заполняются нулями — они не участвуют в проверках.
POSITIVE = "entropy_last"
INVERTED = "mass_last"
CONSTANT = "density_k1_last"


def _arrays(gold: list[int], answer: str) -> dict[str, list[float]]:
    """Значения по токенам ответа: риск растёт там, где метка = 1 (и наоборот для mass)."""
    count = len(list(tokenize_with_offsets(answer)))
    padded = (gold + [0] * count)[:count]
    return {
        name: [
            (1.0 if flag else 0.0) if name == POSITIVE else ((0.0 if flag else 1.0) if name == INVERTED else 0.5)
            for flag in padded
        ]
        for name in GRID_FEATURE_NAMES
    }


def _write_dataset(tmp_path: Path, *, with_test: bool = True) -> Path:
    """Две пары: в первой размечен один токен, во второй — нет (метрика не вырождена)."""
    splits = tmp_path / "splits"
    splits.mkdir(parents=True, exist_ok=True)

    answer_a = "двадцать пять лет"
    tokens_a = [token.text for token in tokenize_with_offsets(answer_a)]
    gold = [1 if text == tokens_a[0] else 0 for text in tokens_a]
    # Проверка сама на себе: gold из `labels` обязан совпадать с тем, чем заполнены
    # признаки, иначе тест измеряет не то, что заявлено (и молча «позеленет» при
    # изменении токенизатора).
    assert gold == [1, 0, 0], "ожидается ровно три содержательных токена в первом ответе"
    pairs = [
        {
            "id": "pair-a",
            "context": "Срок хранения договоров — десять лет.",
            "answer": answer_a,
            "labels": [[0, len(tokens_a[0]), 1]],
            "meta": {"kind": "test"},
        },
        {
            "id": "pair-b",
            "context": "Срок хранения договоров — десять лет.",
            "answer": "десять лет",
            "labels": [],
            "meta": {"kind": "test"},
        },
    ]
    rows = []
    for pair in pairs:
        gold_flags = [1, 0, 0] if pair["id"] == "pair-a" else [0, 0, 0]
        arrays = _arrays(gold_flags, pair["answer"])
        rows.append((pair, arrays))

    def dump(path: Path, items: list[tuple[dict, dict]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for pair, _ in items:
                handle.write(json.dumps(pair, ensure_ascii=False) + "\n")

    dump(splits / "train.jsonl", rows)
    dump(splits / "dev.jsonl", rows)
    if with_test:
        # Тестовая часть намеренно «другая»: если её открыть, числа не совпадут с ожидаемыми.
        other = [
            (
                {
                    "id": "pair-t",
                    "context": "Срок хранения договоров — десять лет.",
                    "answer": "тридцать лет",
                    "labels": [[0, 7, 1]],
                    "meta": {"kind": "test"},
                },
                None,
            )
        ]
        with (splits / "test.jsonl").open("w", encoding="utf-8") as handle:
            for pair, _ in other:
                handle.write(json.dumps(pair, ensure_ascii=False) + "\n")

    cache = tmp_path / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "grid_all_00.jsonl").open("w", encoding="utf-8") as handle:
        for pair, arrays in rows:
            handle.write(
                json.dumps(
                    {
                        "pair_id": pair["id"],
                        "grid_format": 3,
                        "model": "test/model",
                        "revision": "0" * 40,
                        "arrays": arrays,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return splits


def run(tmp_path: Path, splits: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--grid-cache",
            str(tmp_path / "cache"),
            "--splits",
            str(splits),
            "--bootstrap",
            "200",
            "--out-json",
            str(tmp_path / "signs.json"),
            "--out-md",
            str(tmp_path / "signs.md"),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )


def test_signs_are_measured_in_both_directions(tmp_path: Path) -> None:
    splits = _write_dataset(tmp_path)
    proc = run(tmp_path, splits)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads((tmp_path / "signs.json").read_text(encoding="utf-8"))
    by_name = {item["name"]: item for item in payload["features"]}
    assert by_name[POSITIVE]["auc"] > 0.9, "признак, совпадающий с меткой, обязан дать AUC выше 0,9"
    assert by_name[POSITIVE]["sign_by_val"] == "исходный"
    assert by_name[INVERTED]["auc"] < 0.1, "перевёрнутый признак обязан дать AUC ниже 0,1"
    assert by_name[INVERTED]["auc_inverted"] > 0.9
    assert by_name[INVERTED]["sign_by_val"] == "инвертированный"
    assert by_name[CONSTANT]["auc"] is None
    assert "постоянен" in by_name[CONSTANT]["sign_by_val"]
    assert by_name[POSITIVE]["ci_low"] is not None and by_name[POSITIVE]["ci_low"] > 0.5


def test_only_val_split_is_read(tmp_path: Path) -> None:
    splits = _write_dataset(tmp_path)
    proc = run(tmp_path, splits)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads((tmp_path / "signs.json").read_text(encoding="utf-8"))
    assert payload["split"]["file"].endswith("dev.jsonl")
    assert payload["split"]["pairs"] == 2, "тестовая пара не должна попасть в измерение"


def test_test_split_cannot_be_substituted(tmp_path: Path) -> None:
    """Если в каталоге нет val, а есть только test — скрипт обязан отказаться, а не «посчитать по тесту»."""
    splits = tmp_path / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    _write_dataset(tmp_path, with_test=True)
    (splits / "dev.jsonl").unlink()
    proc = run(tmp_path, splits)
    assert proc.returncode != 0
    assert "тестов" in (proc.stdout + proc.stderr) or "val.jsonl" in (proc.stdout + proc.stderr)


def test_markdown_table_lists_every_grid_feature(tmp_path: Path) -> None:
    splits = _write_dataset(tmp_path)
    proc = run(tmp_path, splits)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    text = (tmp_path / "signs.md").read_text(encoding="utf-8")
    for name in GRID_FEATURE_NAMES:
        assert f"`{name}`" in text, f"в таблице нет строки признака {name}"
    assert "Знак по val" in text


def test_grid_cache_with_test_shard_is_rejected(tmp_path: Path) -> None:
    """Кеш, собранный по тестовой выборке, отвергается по имени файла.

    Проверяет не математику, а запрет: артефакты вида `grid-test-a3-0` существуют в CI, и их
    легко скачать «по соседству». Посчитать на них AUC — значит открыть тестовую выборку
    второй раз, поэтому отказ происходит до любых вычислений и без записи файлов отчёта.
    """
    splits = _write_dataset(tmp_path)
    cache = tmp_path / "cache"
    (cache / "grid_test_a3_00.jsonl").write_text(
        (cache / "grid_all_00.jsonl").read_text(encoding="utf-8"), encoding="utf-8"
    )
    proc = run(tmp_path, splits)
    assert proc.returncode != 0, "кеш с тестовой выборкой принят"
    combined = proc.stdout + proc.stderr
    assert "только на валидации" in combined, combined[-400:]
    assert not (tmp_path / "signs.json").exists(), "отчёт записан до отказа"


def test_known_grid_aliases_and_degenerate_feature_are_recorded() -> None:
    """Два совпадения в сетке объяснимы определением, третий признак на A3 dev вырожден.

    Тест держит согласованность кода и документа (`docs/ПРИЗНАКИ.md`, раздел 3.3, абзац
    «Совпадения…»): роль «m4» в `resolve_layer` не распознаётся и откатывается на последний
    слой, поэтому `entropy_m4` — дубль `entropy_last`; `density_k1_last` — среднее по одному
    лучшему совпадению, то есть в точности `sim_max_last`; `attrib_borrowed` на валидации
    постоянен, и AUC для него не определена. Если кто-то починит `resolve_layer` или определение
    `k`, тест обязан упасть: вместе с кодом тогда меняют документ и состав сетки, а состав сетки —
    решение заказчика (лист «г»), а не агента.
    """
    from spanverify.features import resolve_layer

    assert resolve_layer("m4", 6) == resolve_layer("last", 6) == 5, "роль m4 перестала быть дублем last"
    assert resolve_layer(-4, 6) == 2, "числовая роль -4 обязана считать четвёртый с конца"

    payload = json.loads(SIGNS_ARTIFACT.read_text(encoding="utf-8"))
    by_name = {item["name"]: item for item in payload["features"]}
    for left, right in (("entropy_last", "entropy_m4"), ("density_k1_last", "sim_max_last")):
        for key in ("auc", "mean_positive", "mean_negative", "distinct_values", "sign_by_val"):
            assert by_name[left][key] == by_name[right][key], (
                f"{left} и {right}: значение «{key}» разошлось — определение изменилось, и абзац"
                " «Совпадения…» в docs/ПРИЗНАКИ.md больше не соответствует коду"
            )
    assert by_name["attrib_borrowed"]["auc"] is None, "признак перестал быть постоянным — 3.3 надо пересобрать"
    assert by_name["attrib_borrowed"]["distinct_values"] == 1
