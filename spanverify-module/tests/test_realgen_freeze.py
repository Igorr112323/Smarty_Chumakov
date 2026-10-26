"""REALGEN-R1: заморозка фазы A не должна расшатываться последующими правками.

Тесты читают только разрешённые файлы: REALGEN_FROZEN.json, REALGEN_PHASE_A.json,
REALGEN_BACKGROUND.json, REALGEN_SPLITS.json и реальные данные, перечисленные в
заморозке. Запертый real_test_pairs.jsonl не открывается НИ ОДНИМ тестом.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parent
REAL = REPO_ROOT / "reports" / "science" / "real"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(name: str) -> dict:
    return json.loads((REAL / name).read_text(encoding="utf-8"))


def test_frozen_quotas_and_sealed_split_are_preserved() -> None:
    """400+ ответов, ≥40% ошибочных, 300 dev, test_realgen запечатан."""
    phase_a = _load("REALGEN_PHASE_A.json")
    frozen = _load("REALGEN_FROZEN.json")
    assert phase_a["quotas"]["n_answers"] >= 400
    assert phase_a["quotas"]["erroneous_share"] >= 0.40
    assert phase_a["quotas"]["rules_changed"] is False
    assert frozen["n_answers"] == phase_a["quotas"]["n_answers"]
    assert frozen["splits"]["dev"]["n_answers"] == 300
    assert frozen["splits"]["test_realgen"]["sealed"] is True
    assert frozen["real_test_viewed"] is False
    assert set(frozen["rules"]) == {"a", "b", "c", "d"}
    assert frozen["rule_c_mode"] in ("any_window", "best_window")


def test_frozen_file_hashes_match_the_committed_bytes() -> None:
    """SHA-карта заморозки побайтно соответствует выложенным файлам."""
    frozen = _load("REALGEN_FROZEN.json")
    for relative, info in frozen["files"].items():
        path = REPO_ROOT / relative
        assert path.is_file(), f"нет файла из заморозки: {relative}"
        assert _sha256(path) == info["sha256"], f"изменился файл заморозки: {relative}"
    assert _sha256(REAL / "REALGEN_ANSWERS_TEST_REALGEN.jsonl") == frozen["splits"]["test_realgen"]["sha256"]


def test_background_has_zero_overlap_with_test_hosting_docs() -> None:
    """Фон ровно 6510 документов и не содержит 1200 test-hosting doc_id."""
    background = _load("REALGEN_BACKGROUND.json")
    manifest = _load("CORPUS_MANIFEST.json")
    ids = [entry["id"] for entry in manifest["entries"]]
    assert background["n_documents"] == 6510 == len(background["doc_ids"])
    assert background["asserts"]["overlap_with_test_hosting_docs"] == 0
    assert background["asserts"]["real_test_locked_file_read"] is False
    assert set(background["doc_ids"]) <= set(ids), "фон содержит id вне манифеста"
    selected = sorted(ids, key=lambda x: hashlib.sha256(f"42:{x}".encode()).hexdigest())[:1500]
    ordered = sorted(selected, key=lambda x: hashlib.sha256(f"42:split:{x}".encode()).hexdigest())
    test_ids = set(ordered[300:1500])
    assert not test_ids & set(background["doc_ids"]), "фон содержит test-hosting документы"


def test_dev_and_test_realgen_splits_do_not_intersect() -> None:
    """Сплит по документам: 300 dev / остальное test_realgen, пересечений нет."""
    splits = _load("REALGEN_SPLITS.json")
    assert splits["dev_answers"] == 300
    assert splits["overlap"] == 0
    assert splits["test_realgen_sealed"] is True
    assert not set(splits["dev_docs"]) & set(splits["test_realgen_docs"])
    rows = [(json.loads(line))["split"] for line in (REAL / "REALGEN_ANSWERS.jsonl").open(encoding="utf-8")]
    assert rows.count("dev") == 300
    assert rows.count("test_realgen") == splits["test_realgen_answers"]


def test_phase_b_reports_only_dev_and_honest_gates() -> None:
    """Фаза B: 300 dev, real_test не читался, ворота и CI на месте."""
    phase_b = _load("REALGEN_PHASE_B.json")
    assert phase_b["data"]["dev_answers"] == 300
    assert phase_b["data"]["smoke_test"] is False
    assert phase_b["real_test_viewed"] is False
    assert phase_b["provenance"]["test_realgen_file_read"] is False
    assert phase_b["provenance"]["real_test_locked_file_read"] is False
    assert len(phase_b["per_feature_AUC"]) == 27
    assert phase_b["gates"]["token_AUC_min"] == 0.70
    assert phase_b["gates"]["span_AUC_min"] == 0.55
    assert phase_b["gates"]["verdict"] in ("PASS", "FAIL")
    for key in ("composite_OOF", "rubert_tiny2_finetune"):
        block = phase_b[key]
        assert 0.0 <= block["token_AUC"] <= 1.0
        assert 0.0 <= block["span_AUC"] <= 1.0
        assert len(block["token_ci95"]) == len(block["span_ci95"]) == 2


def test_phase_b_oof_rows_never_touch_test_realgen_docs() -> None:
    """OOF-строки фазы B не содержат ни одного test_realgen документа."""
    splits = _load("REALGEN_SPLITS.json")
    test_docs = set(splits["test_realgen_docs"])
    seen, folds, labels = 0, set(), set()
    with (REAL / "REALGEN_PHASE_B_OOF.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            assert row["doc_id"] not in test_docs
            folds.add(row["fold"])
            labels.add(row["label"])
            seen += 1
    assert seen == _load("REALGEN_PHASE_B.json")["data"]["n_tokens"]
    assert folds == {0, 1, 2, 3, 4}
    assert labels <= {0, 1}


def test_phase_a_reports_run_id_provenance() -> None:
    """Честный provenance: ref-прогон генерации + local-no-ci-evidence разметки."""
    phase_a = _load("REALGEN_PHASE_A.json")
    audit = _load("RUN_ID_AUDIT.json")
    assert phase_a["provenance"]["real_test_viewed"] is False
    assert phase_a["provenance"]["ci_run_ids"], "нет проверяемого CI run-id"
    assert phase_a["provenance"]["run_id"] in ("local-no-ci-evidence", *phase_a["provenance"]["ci_run_ids"])
    assert any(entry.get("kind") == "verified-actions-run" for entry in audit["entries"])
