"""Тесты харнесса цикла v2: чистые функции, независимость выборки, пол объёма.

Отдельный файл (а не дополнения в `test_hf_protocol*.py`) нужен затем, чтобы права
v1-гейтов не «поехали»: тесты v1 замережают замороженный протокол, эти — новый.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import hf_protocol as v1  # noqa: E402
import hf_protocol_v2 as v2  # noqa: E402


def make_rows() -> list[dict]:
    """Две пары по три токена: в первой — два gold-токена (0,9 и 0,15), во второй — ложные оценки."""
    rows: list[dict] = []
    plan = {
        "p1": {
            "doc": "doc-1",
            "spans": [(1, 6)],
            "tokens": [(0, 1, "a", 0, 0.2), (1, 4, "b", 1, 0.9), (4, 6, "c", 1, 0.15)],
        },
        "p2": {"doc": "doc-2", "spans": [], "tokens": [(0, 2, "d", 0, 0.6), (2, 4, "e", 0, 0.3), (4, 5, "f", 0, 0.1)]},
    }
    for pair, block in plan.items():
        for start, end, token, gold, score in block["tokens"]:
            rows.append(
                {
                    "id": pair,
                    "doc": block["doc"],
                    "token": token,
                    "start": start,
                    "end": end,
                    "gold": gold,
                    "score": score,
                    "features": {},
                    "spans": list(block["spans"]),
                }
            )
    return rows


def write_pairs(directory: Path, per_part: dict[str, int]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for part, count in per_part.items():
        filename = "dev.jsonl" if part == "val" else f"{part}.jsonl"
        rows = []
        for index in range(count):
            rows.append(
                {
                    "id": f"{part}-{index}",
                    "context": f"контекст {part} {index}",
                    "answer": f"ответ {part} {index} содержит значение 5,5 м3",
                    "labels": [[7, 10, 1]] if index % 2 == 0 else [],
                    "meta": {"group": f"doc-{part}-{index}"},
                }
            )
        with (directory / filename).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ определения


def test_goal_status_arithmetic_matches_the_frozen_rule() -> None:
    """0,70 при FPR ≤ 0,40; «почти» не превращается в PASS, FPR сверх потолка — тоже."""
    assert v2.goal_status({"f1": 0.70, "fpr": 0.40}, 0.70)["status"] == "PASS"
    assert v2.goal_status({"f1": 0.699999}, 0.70)["status"] == "NOT MET"
    assert v2.goal_status({"f1": 0.90, "fpr": 0.4001}, 0.70)["status"] == "NOT MET"
    contract = v2.goal_status({"f1": 0.61, "fpr": 0.10}, v2.GOAL_CONTRACT_F1)
    assert contract["status"] == "PASS" and contract["f1_gap"] <= 0
    assert v2.GOAL_INTERNAL_F1 > v2.GOAL_CONTRACT_F1, "внутренняя цель обязана быть строже контрактного минимума"


def test_goal_status_handles_missing_and_none_metrics() -> None:
    """FPR = 1,0 при отсутствии чисел: «не измерено» не читается как «пройдено»."""
    for payload in (None, {}, {"f1": None, "fpr": None}):
        status = v2.goal_status(payload, v2.GOAL_INTERNAL_F1)
        assert status["status"] == "NOT MET"
        assert status["token_fpr"] == 1.0


def test_apply_rule_counts_micro_confusion_by_hand() -> None:
    rows = make_rows()
    block = v2.apply_rule({"merge_gap": 2}, rows, 0.55)
    # Порог 0,55: помечены 0,9 (gold → TP=1) и 0,6 из второй пары (не gold → FP=1);
    # второй gold-токен (0,15) ниже порога → FN=1; оставшиеся незаченные → TN=3.
    assert block["threshold"] == 0.55 and block["rows"] == 6 and block["positive_rows"] == 2
    precision = 1 / 2
    recall = 1 / 2
    assert block["f1"] == pytest.approx(2 * precision * recall / (precision + recall), abs=1e-6)
    assert block["fpr"] == pytest.approx(1 / (1 + 3), abs=1e-6), "FPR = FP / (FP + TN) из спецификации"


def test_curve_fpr_is_non_increasing_in_threshold() -> None:
    rows = make_rows()
    curve = v2.curve_on(rows, 2, v2.threshold_candidates([row["score"] for row in rows]))
    assert len(curve) > 3
    fpr_values = [float(point["fpr"]) for point in curve]
    assert fpr_values == sorted(fpr_values, reverse=True), "FPR обязан не расти при росте порога"
    for previous, current in zip(curve, curve[1:], strict=False):
        assert float(current["tp"]) <= float(previous["tp"])


def test_threshold_candidates_stay_inside_open_interval() -> None:
    candidates = v2.threshold_candidates([0.0, 0.5, 1.0] * 50 + [index / 1000 for index in range(1000)])
    assert candidates == sorted(candidates)
    assert all(0.0 < value < 1.0 for value in candidates)
    assert len(candidates) <= v2.CURVE_MAX_POINTS


# ------------------------------------------------------------------- независимость


def test_collect_document_ids_covers_every_v1_corpus() -> None:
    """Проверка на реальных данных репозитория: счётчики должны быть осмысленными."""
    a3 = v2.collect_document_ids(ROOT, "data/corpus_a3")
    assert len(a3) == 409, "список источников A3 — ровно 409 актов (сверка с `ls`)"
    assert not any(str(item).endswith(".txt") for item in a3), "расширения не должны попадать в doc_id"
    assert "0001202212080001" in a3
    assert len(v2.collect_document_ids(ROOT, "data/corpus_a")) == 60
    assert len(v2.collect_document_ids(ROOT, "data/corpus_a3_local70")) == 159
    assert len(v2.collect_document_ids(ROOT, "data/demo_pairs.jsonl")) == 240


def test_independence_detector_fires_on_a_measured_corpus(tmp_path: Path) -> None:
    """Если «независимость» всегда 0 — это не проверка, а декорация.

    Поэтому тест сначала убеждается, что детектор срабатывает на самом A3 (тот
    корпус v1 измерял), и только потом — что на синтетических id он молчит.
    """
    report = v2.independence_report(ROOT, ROOT / "data" / "corpus_a3" / "splits")
    assert report["overlap_total"] > 0, "A3 целиком состоит из виденных документов — пересечение обязано быть"
    assert report["v1_universes"]["data/corpus_a3"]["overlap"] == report["v2_documents"]
    assert report["v1_universes"]["data/corpus_a"]["overlap"] == 0, "синтетические документы A1 — другая вселенная id"
    split_dir = tmp_path / "splits"
    write_pairs(split_dir, {"train": 2, "val": 2, "test": 2})
    clean = v2.independence_report(tmp_path, split_dir)
    assert clean["overlap_total"] == 0 and clean["v2_documents"] == 6


# ------------------------------------------------------------------ пол объёма


def test_audit_sizes_rejects_splits_below_the_frozen_floor(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits"
    write_pairs(split_dir, {"train": 3, "val": 3, "test": 3})
    info, problems = v2.audit_sizes(v2.parts_of(split_dir))
    assert set(info) == {"train", "val", "test"}
    text = "\n".join(problems)
    for part, (pairs, documents) in v2.FLOORS.items():
        assert f"{part}: пар 3 — ниже замороженного пола {pairs}" in text
        assert f"{part}: документов 3 — ниже замороженного пола {documents}" in text


def test_audit_sizes_detects_document_leakage_between_parts(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits"
    write_pairs(split_dir, {"train": 1, "val": 1, "test": 1})
    # Тот же документ в train и в test — ровно та ошибка, из-за которой метрика
    # перестаёт быть оценкой обобщения.
    train = split_dir / "train.jsonl"
    test = split_dir / "test.jsonl"
    row = json.loads(test.read_text(encoding="utf-8").splitlines()[0])
    row["id"] = "test-copy"
    with train.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    _info, problems = v2.audit_sizes(v2.parts_of(split_dir))
    assert any("документы пересекаются между train и test" in item for item in problems), problems


# ------------------------------------------------------------------ seed и отбор


def test_bootstrap_resamples_documents_and_stays_deterministic(tmp_path: Path) -> None:
    records = [
        {"id": f"p{i}", "meta": {"group": f"doc-{i // 3}"}, "context": f"c{i}", "answer": f"a{i}", "labels": []}
        for i in range(9)
    ]
    first = v2.bootstrap_records(records, 42)
    second = v2.bootstrap_records(records, 42)
    other = v2.bootstrap_records(records, 43)
    assert len(first) == len(records), "бутстрап обязан оставлять объём обучающей выборки"
    assert [row["id"] for row in first] == [row["id"] for row in second], "seed обязан давать тот же состав"
    # Пары одного документа не разрываются: они идут блоками одинаковой длины.
    sizes = Counter(v1.document_of(row) for row in first)
    blocks = [len(list(group)) for _key, group in _group_by(first, v1.document_of)]
    assert set(blocks) <= set(sizes.values())
    assert blocks[0] == 3, "в исходной выборке по 3 пары на документ — блоки обязаны сохраниться"
    assert [row["id"] for row in first] != [row["id"] for row in other], "разные seed'ы дают разный состав"


def _group_by(rows, key):
    from itertools import groupby

    return groupby(rows, key=key)


def test_entries_limit_is_ten_and_feature_names_are_validated() -> None:
    good = {"name": "x", "hypothesis": "h", "change": "c", "features": ["entropy_last", "mass_last"]}
    with pytest.raises(SystemExit, match="больше разрешённых"):
        v2.entries_of({"experiments": [good] * (v2.MAX_EXPERIMENTS + 1)})
    with pytest.raises(SystemExit, match="неизвестные признаки"):
        v2.entries_of({"experiments": [{**good, "features": ["entropy_of_doom"]}]})
    assert len(v2.entries_of({"experiments": [good] * v2.MAX_EXPERIMENTS})) == v2.MAX_EXPERIMENTS


def test_curve_and_sweep_cannot_read_the_test_split() -> None:
    """Доступ к test — только у `final`: проверкам на уровне CLI и исходного кода."""
    parser = v2.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["curve", "--config", "c.json", "--grid-cache", "g", "--split", "test"])
    for command in ("curve", "sweep", "seedcheck"):
        source = Path(ROOT / "scripts" / "hf_protocol_v2.py").read_text(encoding="utf-8")
        body = source.split(f"def command_{command}(")[1].split("\ndef command_")[0]
        assert 'parts["test"]' not in body and "test.jsonl" not in body, f"{command} читает тестовую выборку"


def test_frozen_final_command_is_the_makefile_command() -> None:
    """`make hf-final-v2` и `final_command` из замороженного конфига — одна команда.

    Иначе «единственный запуск» описан в двух местах, и расхождение между ними
    обнаружилось бы только после того, как числа получили.
    """
    import freeze_final_config_v2 as freeze

    text = (ROOT / "Makefile").read_text(encoding="utf-8")
    recipe = text.split("hf-final-v2:")[1].split("\n\n")[0]
    line = [item.strip().lstrip("\t") for item in recipe.splitlines() if item.strip().startswith("$")][0]
    for variable, value in (
        ("$(PYTHON)", "python"),
        ("$(GRID)", "reports/v2-grid"),
        ("$(SPLITS)", "data/corpus_v2/splits"),
        ("$(SEEDS)", "42,43,44,45,46"),
        ("$(OUT)", "reports/v2"),
    ):
        line = line.replace(variable, value)
    assert line == freeze.FINAL_COMMAND


def test_makefile_targets_are_present() -> None:
    text = (ROOT / "Makefile").read_text(encoding="utf-8")
    for target in ("hf-freeze-v2", "hf-curve-v2", "hf-sweep-v2", "hf-seedcheck-v2", "hf-final-v2", "hf-verify-v2"):
        assert f"{target}:" in text, target


def test_experiments_config_is_pre_registered() -> None:
    config = json.loads((ROOT / "config" / "hf_experiments_v2.json").read_text(encoding="utf-8"))
    assert config["protocol"] == v2.PROTOCOL
    assert config["goal_internal"]["token_f1_min"] == v2.GOAL_INTERNAL_F1
    assert config["goal_contract_min"]["token_f1_min"] == v2.GOAL_CONTRACT_F1
    names = [str(item["name"]) for item in config["experiments"]]
    assert len(names) == v2.MAX_EXPERIMENTS and len(set(names)) == len(names)
    for item in config["experiments"]:
        assert item["hypothesis"] and item["change"], f"у гипотезы {item['name']} нет заранее записанной формулировки"
        assert item["features"], "гипотеза без признаков — не гипотеза"
    assert set(config["e0_experiments"]) <= set(names)


def test_harness_runs_on_repo_without_a_corpus(tmp_path: Path) -> None:
    """`freeze` обязан запускаться и честно говорить, что корпуса v2 ещё нет."""
    code = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "hf_protocol_v2.py"),
            "freeze",
            "--splits",
            str(tmp_path / "нет-такого-каталога"),
            "--out",
            str(tmp_path / "отчёт"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert code.returncode == 1
    assert "нет файла разбиения" in code.stdout
