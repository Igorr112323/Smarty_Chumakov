"""Тесты документации цикла v2: спецификация заморожена, числа — только из манифеста.

Отдельная ценность этих проверок в том, что они запрещают «тихую» правку правил
после того, как числа получены: спецификация, хеш которой попадает в заморозку,
сравнимо с контрактом на измерения.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import hf_protocol as v1  # noqa: E402
import render_readme_v2 as renderer  # noqa: E402

SPEC_V2 = ROOT / "docs" / "METRIC_SPEC_v2.md"
SPEC_V1 = ROOT / "docs" / "METRIC_SPEC.md"
DOC = ROOT / "docs" / "EXPERIMENTS_v2.md"
CORPUS_DOC = ROOT / "docs" / "CORPUS_v2.md"
MANIFEST = ROOT / "reports" / "v2" / "manifest.json"
FREEZE = ROOT / "reports" / "v2" / "freeze.json"
FINAL_CONFIG = ROOT / "config" / "hf_final_config_v2.json"


def flat(path: Path) -> str:
    """Текст без переносов: markdown перечитывает абзац целиком, а `in` — нет."""
    return " ".join(path.read_text(encoding="utf-8").split())


def test_metric_spec_v2_freezes_every_definition_that_matters() -> None:
    text = flat(SPEC_V2)
    required = [
        "FPR = FP / (FP + TN)",
        "is_scored_token",
        "label == 1",
        "IoU ≥ 0,5",
        "grounded",
        "AUC",
        "не вычислено",
        "FPR = FP / (FP + TN)",
        "0,70",
        "0,60",
        "стоп-правил",
        "std",
        "«почти»",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"в спецификации v2 не хватает определений: {missing}"
    assert "не допускается" in text or "Запрещено" in text
    assert "не открыва" in text, "стоп-правило обязано явно запечатывать тестовую выборку"


def test_metric_spec_v1_untouched_by_the_new_cycle() -> None:
    """Спецификацию v1 нельзя «поправить» под новый цикл: её хеш зафиксирован в манифестах v1."""
    digest = v1.sha256_file(SPEC_V1)
    for name in ("hf_final", "hf_final_a1"):
        manifest = ROOT / "reports" / name / "manifest.json"
        if not manifest.is_file():
            continue
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        assert payload.get("metric_spec_sha256") == digest, f"{name}: хеш спецификации v1 разошёлся с файлом"


def test_metric_spec_v2_hash_matches_the_freeze_record() -> None:
    if not FREEZE.is_file():
        # Шага 0 ещё не было: сверять нечего, но и чисел v2 быть не должно —
        # это проверяется здесь же, чтобы отсутствие гейта не выглядело «зелёным».
        assert "не измерено" in DOC.read_text(encoding="utf-8")
        return
    payload = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert payload["metric_spec_v2_sha256"] == v1.sha256_file(
        SPEC_V2
    ), "спецификацию v2 изменили после заморозки: это новое измерение, а не правка формулировки"


def test_corpus_procedure_and_floor_are_written_down() -> None:
    text = flat(CORPUS_DOC)
    for item in ("build_corpus_a.py --real", "--text-only", "--exclude-dirs", "review_server.py", "4280", "cp -n"):
        assert item in text, f"в описании процедуры сбора корпуса v2 нет строки «{item}»"
    assert (
        "экспертная верификация разметки не выполнена" in text
    ), "нужна явная формулировка на случай отказа от проверки"


def test_journal_lists_the_pre_registered_hypotheses_and_both_threshold_rules() -> None:
    doc = DOC.read_text(encoding="utf-8")
    assert "0,70" in doc and "0,60" in doc, "цели цикла обязаны быть в журнале до измерений"
    config = json.loads((ROOT / "config" / "hf_experiments_v2.json").read_text(encoding="utf-8"))
    for item in config["experiments"]:
        assert f"`{item['name']}`" in doc, f"гипотеза {item['name']} не описана в журнале до прогона"
    for marker in ("E0", "E1", "СТОП-ПРАВИЛО", "не достигнуто на val"):
        assert marker in doc or marker.lower() in doc.lower(), marker
    assert "не измерено" in doc, "журнал до артефактов обязан содержать «не измерено»"


def test_frozen_threshold_line_exists_only_with_a_frozen_config() -> None:
    doc = DOC.read_text(encoding="utf-8")
    marker = "Порог, зафиксированный до шага 3 v2:"
    if not FINAL_CONFIG.is_file():
        assert marker not in doc, "число порога появилось в журнале раньше замороженной конфигурации"
        return
    config = json.loads(FINAL_CONFIG.read_text(encoding="utf-8"))
    line = next(item for item in doc.splitlines() if marker in item)
    assert f"{config['threshold']}" in line, f"порог в журнале ({line}) != в конфиге ({config['threshold']})"
    assert v1.sha256_file(FINAL_CONFIG) in doc, "в журнале нет sha256 замороженной конфигурации"


def test_readme_block_is_generated_from_the_manifest_only(tmp_path: Path) -> None:
    """Круговой тест рендера: блок = манифест, а `--check` ловит подправленную цифру."""
    manifest = {
        "metrics_mean_std": {
            "f1": 0.712345,
            "f1_std": 0.011222,
            "fpr": 0.2,
            "auc": 0.95,
            "answer_f1": 0.7,
            "answer_fpr": 0.3,
            "span_f1": 0.4,
        },
        "goal_internal": {"status": "PASS", "token_f1": 0.712345, "token_fpr": 0.2},
        "goal_contract_min": {"status": "PASS", "token_f1": 0.712345, "token_fpr": 0.2},
        "pairs": {"train": 3000, "val": 640, "test": 640},
        "documents": {"train": 1050, "val": 220, "test": 220},
        "prediction_rows": 51200,
        "code_commit": "deadbeef",
        "model": {"id": "ai-forever/rugpt3small_based_on_gpt2"},
        "duration_s": 1234.5,
        "seeds": [42, 43, 44, 45, 46],
        "threshold": 0.1234,
        "threshold_rule": "E1",
        "config": {"goals": {"internal": {"token_f1_min": 0.70}, "contract_min": {"token_f1_min": 0.60}}},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text("# Отчёт\n\nи абзац про ограничения\n", encoding="utf-8")
    script = ROOT / "scripts" / "render_readme_v2.py"
    subprocess.run(
        [sys.executable, str(script), "--manifest", str(manifest_path), "--readme", str(readme), "--write"],
        check=True,
        capture_output=True,
    )
    text = readme.read_text(encoding="utf-8")
    assert "0,712345" in text and "0,200000" in text and renderer.BEGIN in text
    assert (
        subprocess.run(
            [sys.executable, str(script), "--manifest", str(manifest_path), "--readme", str(readme), "--check"],
            check=False,
            capture_output=True,
            text=True,
        ).returncode
        == 0
    )
    readme.write_text(text.replace("0,712345", "0,720000"), encoding="utf-8")
    check = subprocess.run(
        [sys.executable, str(script), "--manifest", str(manifest_path), "--readme", str(readme), "--check"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 1 and "не совпадает" in check.stdout, "подправленная цифра обязана быть найдена"


def test_readme_explains_the_second_red_gate() -> None:
    """Читатель CI обязан понимать, почему он красный: v1 — измерено и NOT MET,
    v2 — цикл не доведён до измерения. Оба случая описаны словами, без чисел."""
    text = (ROOT.parent / "README.md").read_text(encoding="utf-8")
    assert "Почему CI бывает красным" in text
    assert "test_v2_criterion.py" in text, "в README нет объяснения второго красного гейта (цикл v2)"
    assert "не выполнено" in text or "не выполнялось" in text, "объяснение обязано называть состояние словами"


def test_metrics_json_knows_the_cycle_state_but_not_invented_numbers() -> None:
    """`reports/METRICS.json` обязан различать «не измерено» и числа из манифеста."""
    metrics = json.loads((ROOT / "reports" / "METRICS.json").read_text(encoding="utf-8"))
    section = metrics.get("cycle_v2")
    assert isinstance(section, dict), "в METRICS.json нет секции cycle_v2"
    if MANIFEST.is_file():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        aggregate = manifest.get("metrics_mean_std") or {}
        assert abs(float(section["f1"]) - float(aggregate["f1"])) < 1e-9, "число в METRICS.json != из манифеста"
        assert abs(float(section["fpr"]) - float(aggregate["fpr"])) < 1e-9
    else:
        assert section.get("status") == "не измерено", "до шага 3 в METRICS.json не может быть чисел v2"
        assert section.get("step3") is False
        for key in ("f1", "fpr", "threshold"):
            assert key not in section, f"поле {key} появилось раньше измерения"


def test_corpus_doc_marks_the_mode_as_awaiting_customer_approval() -> None:
    """Режим загрузки — решение заказчика: документ не вправе писать о нём «решено»."""
    text = flat(CORPUS_DOC)
    for item in (".github/corpus-v2.mode", "ожидает подтверждения заказчика", "на паузе", "ocr_pending_approval"):
        assert item in text, f"в `docs/CORPUS_v2.md` нет обязательной строки «{item}»"
    for banned in ("измерен и отвергнут", "решено заменить", "text_only: false"):
        assert banned not in text, f"документ заявляет отменённое заказчиком решение как факт: «{banned}»"


def test_corpus_doc_distinguishes_the_volume_target_from_the_download_stop_threshold() -> None:
    """1500 — цель по объёму выборки, 1800 — порог остановки загрузки; путать их нельзя."""
    text = flat(CORPUS_DOC)
    assert "1500" in text and "1800" in text, "в документе обязаны стоять оба числа"
    assert "порог остановки" in text, "1800 обязан быть подписан как порог остановки, а не как цель"


def test_every_share_of_the_text_layer_is_backed_by_the_measuring_script() -> None:
    """Доля годовых слоёв упоминается только вместе с командой, которая её считает."""
    for doc in (CORPUS_DOC, DOC):
        if not doc.is_file():
            continue
        for line in doc.read_text(encoding="utf-8").splitlines():
            if "текстовым слоем" in line and "%" in line:
                assert (
                    "text_layer_yield" in line or "ДИ" in line
                ), f"{doc.name}: доля текстового слоя без команды и интервала: {line}"


def test_numbers_doc_binds_every_reported_number_to_a_command() -> None:
    """`docs/ЧИСЛА_v2.md`: таблица «число → команда → файл → статус», и ничего без статуса."""
    path = ROOT / "docs" / "ЧИСЛА_v2.md"
    assert path.is_file(), "числа цикла обязаны быть сверены с командами воспроизведения"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 4 and cells[0] not in ("число", "что ранее звучало", "величина"):
            rows.append(cells)
    assert rows, "в таблице чисел нет ни одной строки"
    for number, command, source, status in rows:
        assert "`" in command or "git " in command or "gh " in command, f"число «{number}» без команды воспроизведения"
        assert source, f"число «{number}» без файла-источника"
        assert (
            ("проверено" in status) or ("не" in status) or ("заморожен" in status)
        ), f"число «{number}» без статуса проверки"


def test_reports_never_call_agent_review_expert() -> None:
    """Пока эксперт заказчика не назначен, «экспертная верификация» возможна только с отрицанием."""
    for doc in (CORPUS_DOC, DOC):
        text = doc.read_text(encoding="utf-8")
        for line in text.splitlines():
            if "экспертн" in line.lower() and "верификац" in line.lower():
                lowered = line.lower()
                assert any(
                    marker in lowered
                    for marker in ("не выполнена", "не выполнялась", "подписывает", "экспертной проверки не будет", "|")
                ), f"{doc.name}: утверждение об экспертной проверке без отрицания: {line}"
