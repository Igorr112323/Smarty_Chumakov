"""Тест изоляции официального теста при --no-test (P0 шаг 1).

Проверяет, что при --no-test:
1. Без splits → ошибка (код 3)
2. С splits → test split не передаётся в verifier.evaluate()
3. metrics считаются только по train+dev
4. test_official = null в JSON отчёте
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from spanverify.engine import Verifier


def _write_jsonl(path: Path, records: list[dict]):
    """Записать список словарей в JSONL файл."""
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _make_corpus(corpus_dir: Path, n_train=3, n_dev=2, n_test=2):
    """Создать минимальный корпус с разбиениями."""
    all_pairs = []
    total = n_train + n_dev + n_test
    for i in range(total):
        # Создаём пары с labels в формате [[start, end, label]]
        answer = f"В документе {i} содержится противоречивая информация о событиях"
        labels = []
        if i % 3 != 2:  # Не все пары имеют labels
            # labels: [[start, end, is_contradiction]]
            labels = [[10, 30, 1]]
        
        pair = {
            "id": f"pair_{i:03d}",
            "context": f"Документ {i}: некоторые данные для анализа.",
            "answer": answer,
            "labels": labels,
            "meta": {"source": "test"},
        }
        all_pairs.append(pair)

    pairs_file = corpus_dir / "pairs.jsonl"
    _write_jsonl(pairs_file, all_pairs)

    splits_dir = corpus_dir / "splits"
    splits_dir.mkdir()

    train = all_pairs[:n_train]
    dev = all_pairs[n_train:n_train + n_dev]
    test = all_pairs[n_train + n_dev:]

    _write_jsonl(splits_dir / "train.jsonl", train)
    _write_jsonl(splits_dir / "dev.jsonl", dev)
    _write_jsonl(splits_dir / "test.jsonl", test)

    return pairs_file, {
        "train": [p["id"] for p in train],
        "dev": [p["id"] for p in dev],
        "test": [p["id"] for p in test],
    }


def test_no_test_without_splits_errors():
    """При --no-test без splits должен быть код ошибки 3."""
    from scripts.run_experiments import main as run_experiments_main

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        corpus_dir = tmppath / "corpus"
        corpus_dir.mkdir()

        # Создаём пары без splits
        pairs = []
        for i in range(6):
            answer = f"Ответ {i} содержит информацию о событиях"
            labels = [[10, 20, 1]] if i % 3 != 2 else []
            pairs.append({
                "id": f"pair_{i:03d}",
                "context": f"Документ {i}: данные.",
                "answer": answer,
                "labels": labels,
                "meta": {},
            })

        pairs_file = corpus_dir / "pairs.jsonl"
        _write_jsonl(pairs_file, pairs)

        out_dir = tmppath / "output"
        out_dir.mkdir()

        exit_code = run_experiments_main([
            "--dataset", str(pairs_file),
            "--mode", "demo",
            "--no-test",
            "--out", str(out_dir),
        ])

        assert exit_code == 3, f"Ожидался код 3 (--no-test без splits), получен {exit_code}"


def test_no_test_with_splits_excludes_test():
    """При --no-test с splits test не оценивается, только train+dev."""
    from scripts.run_experiments import main as run_experiments_main

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        corpus_dir = tmppath / "corpus"
        corpus_dir.mkdir()

        pairs_file, split_ids = _make_corpus(corpus_dir)

        out_dir = tmppath / "output"
        out_dir.mkdir()

        # Перехватываем evaluate, чтобы запомнить, какие пары передавались
        original_evaluate = Verifier.evaluate
        evaluated_pair_sets = []

        def tracking_evaluate(self, pairs):
            ids = sorted(p.get("id") for p in pairs)
            evaluated_pair_sets.append(ids)
            return original_evaluate(self, pairs)

        with patch.object(Verifier, "evaluate", tracking_evaluate):
            exit_code = run_experiments_main([
                "--dataset", str(pairs_file),
                "--mode", "demo",
                "--no-test",
                "--out", str(out_dir),
            ])

        assert exit_code == 0, f"Ожидался код 0, получен {exit_code}"

        # Проверяем, что test IDs не встречались ни в одном вызове evaluate
        test_id_set = set(split_ids["test"])
        for ids in evaluated_pair_sets:
            overlap = test_id_set.intersection(ids)
            assert not overlap, (
                f"Test IDs {overlap} найдены в evaluate(), "
                f"но при --no-test тест должен быть изолирован. "
                f"Все вызовы: {evaluated_pair_sets}"
            )

        # Проверяем JSON: test_official должен быть null
        experiment_json = out_dir / "experiment.json"
        assert experiment_json.is_file(), "experiment.json не создан"
        data = json.loads(experiment_json.read_text(encoding="utf-8"))
        assert data["test_official"] is None, (
            f"test_official должен быть null при --no-test, "
            f"получено: {data.get('test_official')}"
        )

        # dev_official должен быть заполнен
        assert data["dev_official"] is not None, (
            "dev_official должен быть заполнен при --no-test с splits"
        )
        assert data["dev_official"]["pairs"] == len(split_ids["dev"]), (
            f"dev_official.pairs = {data['dev_official']['pairs']}, "
            f"ожидается {len(split_ids['dev'])}"
        )
