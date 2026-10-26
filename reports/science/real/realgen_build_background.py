#!/usr/bin/env python3
"""REALGEN-R1 фаза A, шаг 1 (локально, без HF): фон, выборка, шаблоны, документы.

НИ ОДИН шаг не читает real_test_pairs.jsonl / real_all_pairs.jsonl /
real_test_spans.jsonl. 1200 test-hosting doc_id восстанавливаются из правила
разбиения CORPUS_FROZEN.json и проверяются по разрешённому real_dev_pairs.jsonl.

Выход:
  REALGEN_BACKGROUND.json   — 6510 doc_id, ассерты, sha-карты
  REALGEN_SELECTION.json    — 400 doc_id + назначения R1/R2, sha256
  REALGEN_TEMPLATES.json    — 8 шаблонов + 2 режима + 2 текста промта
  REALGEN_DOCS.jsonl        — тексты 400 документов (источник проверок)
"""

from __future__ import annotations

import gzip
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import realgen_common as rc  # noqa: E402


def main() -> None:
    manifest_path = rc.REAL / "CORPUS_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest["entries"]
    assert manifest["n_documents"] == 7710 == len(entries)
    assert manifest["pin"] == "5806f92"

    frozen = json.loads((rc.REAL / "CORPUS_FROZEN.json").read_text(encoding="utf-8"))
    dev_ids, test_ids, background_ids = rc.split_rule_ids(entries)
    assert len(dev_ids) == 300 and len(test_ids) == 1200
    assert len(background_ids) == 6510, len(background_ids)

    # Независимая проверка по РАЗРЕШЁННОМУ файлу dev (meta.group = исходный doc_id).
    dev_groups = {
        json.loads(line)["meta"]["group"]
        for line in (rc.REAL / "real_dev_pairs.jsonl").open(encoding="utf-8")
    }
    assert dev_groups == set(dev_ids), "реконструкция dev-сплита не совпала с real_dev_pairs.jsonl"
    assert all(doc_id in set(test_ids) for doc_id in frozen["split"]["test_ids"])
    assert not (set(dev_ids) & set(test_ids))
    assert not (set(background_ids) & set(test_ids)), "OVERLAP с test-hosting docs!"

    # Каноническая sha-карта фона: отсортированные id, \n-разделитель.
    canonical = "\n".join(background_ids).encode("utf-8")
    background_sha = rc.h(canonical.decode("utf-8"))
    test_ids_sha = rc.h("\n".join(sorted(test_ids)))

    sample = rc.select_sample(background_ids, n=400)
    assert len(sample) == 400 and len(set(sample)) == 400
    assert not (set(sample) & set(test_ids))
    assign_r1 = rc.assignment(sample, "R1")
    assign_r2 = rc.assignment(sample, "R2")
    combos_r1 = Counter((a["template_id"], a["mode"]) for a in assign_r1)
    assert len(combos_r1) == 16 and set(combos_r1.values()) == {25}

    # Тексты документов из corpus_raw.jsonl.gz (внешний источник из репозитория).
    sample_set = set(sample)
    texts: dict[str, str] = {}
    with gzip.open(rc.REAL / "corpus_raw.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            if rec["id"] in sample_set:
                texts[rec["id"]] = rec["text"]
    assert len(texts) == 400, f"нет текстов для {400 - len(texts)} документов"
    by_id = {entry["id"]: entry for entry in entries}

    docs_rows = [
        {
            "doc_id": doc_id,
            "act_name": by_id[doc_id]["act_name"],
            "type": by_id[doc_id]["type"],
            "manifest_sha256": by_id[doc_id]["sha256"],
            "size_chars": by_id[doc_id]["size_chars"],
            "text_sha256": rc.h(texts[doc_id]),
            "text": texts[doc_id],
        }
        for doc_id in sample
    ]
    rc.write_jsonl(rc.REAL / "REALGEN_DOCS.jsonl", docs_rows)

    templates = {
        "kind": "REALGEN_TEMPLATES",
        "version": "v1",
        "n_templates": 8,
        "template_ids": list(rc.TEMPLATE_IDS),
        "tasks": list(rc.TASKS),
        "modes": list(rc.MODES),
        "prompt_v1": rc.PROMPT_V1,
        "prompt_v2_hard": rc.PROMPT_V2_HARD,
        "rounds": {
            "R1": {"prompt": "prompt_v1", "purpose": "основной набор (8 шаблонов × 2 режима)"},
            "R2": {
                "prompt": "prompt_v2_hard",
                "purpose": "только remedy квоты ошибок при <40%: повышена сложность промта",
                "rules_changed": False,
            },
        },
        "context_window": {
            "chars": 1100,
            "prefix_chars": 200,
            "selection": "sha256(42:window:doc_id) по тексту без переносов",
            "reason": "n_positions=1024 модели; полный акт (2-50k) не помещается",
        },
        "model": {"name": rc.MODEL, "weights_mb": 443.5, "max_new_tokens": 512},
    }
    rc.write_json(rc.REAL / "REALGEN_TEMPLATES.json", templates)

    background = {
        "kind": "REALGEN_BACKGROUND",
        "version": "v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "LawCodex/ru pin 5806f92 (CORPUS_MANIFEST.json, 7710 docs) minus 1200 test-hosting docs",
        "manifest_path": "reports/science/real/CORPUS_MANIFEST.json",
        "manifest_sha256": rc.sha256_file(manifest_path),
        "manifest_sha256_of_entries": manifest["manifest_sha256_of_entries"],
        "n_documents": len(background_ids),
        "split_reconstruction": {
            "rule": "generate_phase_b_v2.py: selected=sort(sha256('42:'+id))[:1500]; split=sort(sha256('42:split:'+id))",
            "dev_n": 300,
            "test_n": 1200,
            "verified_against": "real_dev_pairs.jsonl meta.group (300/300 совпало) + CORPUS_FROZEN.json 5 test_id",
        },
        "test_hosting_ids_sha256": test_ids_sha,
        "background_ids_sha256_canonical": background_sha,
        "canonical_serialization": "sorted(doc_id).join(chr(10)), utf-8",
        "directive_sha_prefix_f8198329": {
            "reproduced": False,
            "reason": "ни один файл background 6510 и ни одна каноническая сериализация в этом чекауте/ветках не даёт префикс f8198329; готового артефакта нет (проверено trees всех веток через GitHub API)",
            "replacement": "полный sha256 канонической карты 6510 id выше; ассерты overlap=0 сохранены",
        },
        "asserts": {
            "n_documents_6510": True,
            "overlap_with_test_hosting_docs": 0,
            "dev_in_background": 300,
            "real_test_locked_file_read": False,
            "real_all_pairs_file_read": False,
        },
        "test_hosting_ids_sha256_prefix": test_ids_sha[:12],
        "doc_ids": background_ids,
        "platform": {"python": platform.python_version(), "system": platform.system()},
    }
    rc.write_json(rc.REAL / "REALGEN_BACKGROUND.json", background)

    selection = {
        "kind": "REALGEN_SELECTION",
        "version": "v1",
        "seed": rc.SEED,
        "rule": "sorted(background_ids, key=sha256('42:'+doc_id))[:400]",
        "n_documents": 400,
        "doc_ids": sample,
        "doc_ids_sha256": rc.h("\n".join(sample)),
        "assignments_per_combo": 25,
        "R1": assign_r1,
        "R2": assign_r2,
        "background_ids_sha256": background_sha,
        "test_hosting_ids_sha256": test_ids_sha,
        "asserts": {"no_test_hosting_docs": True, "unique": True, "deterministic": True},
    }
    rc.write_json(rc.REAL / "REALGEN_SELECTION.json", selection)

    for name in (
        "REALGEN_BACKGROUND.json",
        "REALGEN_SELECTION.json",
        "REALGEN_TEMPLATES.json",
        "REALGEN_DOCS.jsonl",
    ):
        path = rc.REAL / name
        print(f"{name}: {rc.file_size(path)} bytes sha256={rc.sha256_file(path)}")
    print(f"background=6510 test_hosting_sha={test_ids_sha} dev_match=300/300 sample=400 combos={dict(combos_r1)}")


if __name__ == "__main__":
    main()
