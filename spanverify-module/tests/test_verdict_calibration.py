"""Калибровка агрегации вердикта: замороженный конфиг, отчёт, таблицы свидетельств.

Причина проверки: вердикт уровня ответа считался той же маской, что и токены, и
на длинных честных ответах реальных актов почти всегда находилась пара ложных
токенов. Отдельная калибровка агрегации (dev → заморозка → один тест) закрывает
это, но её результат живёт в трёх артефактах — ``config/verdict.json``,
``reports/verdict_calibration.json`` и ``reports/verdict/evidence_*.json``.
Эти тесты следят, чтобы артефакты не разошлись между собой и с кодом: правило в
конфиге обязано быть известно движку, параметры — совпадать с отчётом, а
объявленное правило выбора — не подменяться подгонкой по тесту.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from spanverify.core import SpanResult
from spanverify.engine import (
    DEFAULT_VERDICT_PARAMS,
    VERDICT_ANY_SPAN,
    VERDICT_CORROBORATED,
    VERDICT_MIN_TWO_TOKENS,
    VERDICT_RULES,
    Verifier,
    _verdict,
    verdict_config_from_params,
)

MODULE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = MODULE_ROOT / "config" / "verdict.json"
REPORT_PATH = MODULE_ROOT / "reports" / "verdict_calibration.json"
EVIDENCE_DIR = MODULE_ROOT / "reports" / "verdict"


def _load_json(path: Path) -> dict:
    assert path.is_file(), f"нет артефакта {path.name}: он создаётся scripts/calibrate_verdict.py"
    return json.loads(path.read_text(encoding="utf-8"))


def _span(n_tokens: int, source: str, chars: int = 10) -> SpanResult:
    return SpanResult(
        start=0,
        end=chars,
        text="ф" * chars,
        risk=0.5,
        label="doubtful",
        n_tokens=n_tokens,
        source=source,
    )


def _row(**over: object) -> dict:
    """Строка таблицы свидетельств — ровно те поля, что пишет scripts/calibrate_verdict.py."""
    row: dict = {
        "id": "p",
        "y": 0,
        "mode": "faithful",
        "score": 0.0,
        "threshold": 0.5,
        "spans_total": 0,
        "mask_spans": 0,
        "mask_tokens": 0,
        "max_fragment_chars": 0,
        "max_fragment_tokens": 0,
        "text_spans": 0,
        "text_sources": [],
        "scored_tokens": 100,
        "flagged_share": 0.0,
        "mean_risk": 0.0,
        "p90_risk": 0.0,
        "raw_score": 0.0,
    }
    row.update(over)
    return row


# --- замороженный конфиг и отчёт -------------------------------------------------


def test_frozen_config_is_valid_and_complete() -> None:
    """Конфиг калибровки полон: известное правило, все параметры, выбор на dev."""
    config = _load_json(CONFIG_PATH)
    assert config["rule"] in VERDICT_RULES, config["rule"]
    assert config["rule"] == VERDICT_CORROBORATED, "замораживается только калибруемое правило"
    params = config["params"]
    assert set(params) == set(DEFAULT_VERDICT_PARAMS), "набор параметров не совпадает с движком"
    assert config.get("chosen_on") == "dev", "параметры выбираются на dev, а не на тесте"
    assert 0.0 < float(config["target_fpr"]) <= 0.4, "цель FPR вне договорного диапазона"


def test_frozen_config_matches_calibration_report() -> None:
    """Отчёт калибровки — источник конфига: параметры совпадают, происхождение есть."""
    config = _load_json(CONFIG_PATH)
    report = _load_json(REPORT_PATH)
    assert report["rule"] == config["rule"]
    assert report["params"] == config["params"], "конфиг и отчёт разошлись"
    assert report["target_fpr"] == config["target_fpr"]
    assert report["chosen_on"] == config["chosen_on"]
    for key in ("dataset", "dataset_sha256", "seed", "mode", "commit", "generated_at", "command"):
        assert report.get(key), f"в отчёте нет поля {key}: число без происхождения не число"
    assert len(report["dataset_sha256"]) == 64, "хеш корпуса должен быть SHA-256"


def test_calibration_used_declared_selection_rule() -> None:
    """Правило выбора объявлено в отчёте: максимум F1 при FPR ≤ цели, тест не участвует."""
    report = _load_json(REPORT_PATH)
    selection = report["selection"]
    text = selection["rule"].lower()
    assert "dev" in text and "тест в выборе не участвует" in text
    assert selection["grid_size"] > 0
    assert selection["points_within_target"] >= 0
    assert report["dev"][f"rule_{report['rule']}"]["pairs"] > 0
    assert report["test"][f"rule_{report['rule']}"]["pairs"] > 0


def test_dev_reaches_declared_target() -> None:
    """На dev цель по FPR достигнута — иначе отчёт обязан сказать это честно."""
    report = _load_json(REPORT_PATH)
    stage = report["selection"]["stage"]
    target = float(report["target_fpr"])
    dev_fpr = float(report["dev"][f"rule_{report['rule']}"]["fpr"])
    if "выбор внутри цели" in stage:
        assert dev_fpr <= target + 1e-9, (stage, dev_fpr)
    else:
        assert dev_fpr > target, "иначе выбор был бы «внутри цели»"


def test_test_split_is_reported_once_with_frozen_rule() -> None:
    """Тест оценён тем же конфигом: числа теста относятся к официальному тесту."""
    report = _load_json(REPORT_PATH)
    config = _load_json(CONFIG_PATH)
    test_block = report["test"][f"rule_{config['rule']}"]
    assert test_block["pairs"] == report["split_sizes"]["test"]
    assert 0.0 <= test_block["fpr"] <= 1.0 and 0.0 <= test_block["f1"] <= 1.0


def test_evidence_tables_match_report_splits() -> None:
    """Таблицы свидетельств существуют и покрывают те же пары, что и отчёт."""
    report = _load_json(REPORT_PATH)
    for split in ("dev", "test"):
        path = EVIDENCE_DIR / f"evidence_corpus_a3_{split}.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        assert len(rows) == report["split_sizes"][split], (split, len(rows))
        required = {
            "id",
            "y",
            "score",
            "threshold",
            "mask_tokens",
            "max_fragment_chars",
            "text_spans",
            "flagged_share",
            "scored_tokens",
        }
        assert required <= set(rows[0]), sorted(set(rows[0]))


def test_dev_f1_matches_recount_from_evidence_table() -> None:
    """Числа dev воспроизводятся по таблице свидетельств — правило, а не магия прогона."""
    module = importlib.import_module("scripts.calibrate_verdict")
    config = _load_json(CONFIG_PATH)
    report = _load_json(REPORT_PATH)
    rows = json.loads((EVIDENCE_DIR / "evidence_corpus_a3_dev.json").read_text(encoding="utf-8"))
    recount = module.metrics(rows, module.apply_rule(rows, config["params"]))
    reported = report["dev"][f"rule_{config['rule']}"]
    for key in ("tp", "fp", "fn", "tn", "f1", "fpr"):
        assert recount[key] == reported[key], (key, recount[key], reported[key])


# --- поведение правила -----------------------------------------------------------


def test_corroborated_requires_token_count_and_fragment_length() -> None:
    """Маска делает ответ спорным только при ≥2 токенах и фрагменте нужной длины."""
    params = {"min_mask_tokens": 2, "min_fragment_chars": 20, "min_flagged_share": 0.0}
    assert _verdict(0.1, 0.5, [_span(1, "mask", chars=100)], VERDICT_CORROBORATED, params) == "grounded"
    assert _verdict(0.1, 0.5, [_span(2, "mask", chars=5)], VERDICT_CORROBORATED, params) == "grounded"
    assert _verdict(0.1, 0.5, [_span(2, "mask", chars=25)], VERDICT_CORROBORATED, params) == "doubtful"


def test_corroborated_respects_flagged_share() -> None:
    """Доля помеченных токенов — третий параметр; при большом пороге ответ подтверждён."""
    params = {"min_mask_tokens": 2, "min_fragment_chars": 20, "min_flagged_share": 0.5}
    stats = {"scored_tokens": 100}
    assert _verdict(0.1, 0.5, [_span(2, "mask", chars=25)], VERDICT_CORROBORATED, params, stats) == "grounded"


def test_text_rule_decisive_switch_controls_attribution() -> None:
    """Решающая роль текстовых правил — параметр, а не жёстко зашитое поведение."""
    spans = [_span(1, "attribution", chars=10)]
    strict = {"min_mask_tokens": 2, "min_fragment_chars": 20, "min_flagged_share": 0.0, "text_rule_decisive": True}
    soft = dict(strict, text_rule_decisive=False)
    assert _verdict(0.1, 0.5, spans, VERDICT_CORROBORATED, strict) == "doubtful"
    assert _verdict(0.1, 0.5, spans, VERDICT_CORROBORATED, soft) == "grounded"


def test_frozen_rule_replaces_default_and_can_be_overridden() -> None:
    """Движок берёт правило из конфига, но явный аргумент сильнее; чужое — ошибка."""
    config = _load_json(CONFIG_PATH)
    frozen = Verifier(mode="demo")
    assert frozen.verdict_rule == config["rule"]
    assert frozen.verdict_params == config["params"], "конфиг не доехал до движка"
    assert frozen.verdict_config_file, "источник конфига должен быть виден"
    explicit = Verifier(mode="demo", verdict_rule=VERDICT_ANY_SPAN)
    assert explicit.verdict_rule == VERDICT_ANY_SPAN
    with pytest.raises(ValueError):
        Verifier(mode="demo", verdict_rule="выдуманное-правило")


def test_verdict_config_from_params_round_trip() -> None:
    """Сборка конфига из параметров не теряет ключей и приводится к типам."""
    config = verdict_config_from_params(
        {
            "min_mask_tokens": "3",
            "min_fragment_chars": 40,
            "min_flagged_share": 0,
            "text_rule_decisive": 1,
        }
    )
    assert config == {
        "rule": VERDICT_CORROBORATED,
        "params": {
            "min_mask_tokens": 3,
            "min_fragment_chars": 40,
            "min_flagged_share": 0.0,
            "text_rule_decisive": True,
        },
    }


def test_legacy_rules_still_available_for_comparison() -> None:
    """Исторические правила остаются рабочими: сравнение «до/после» обязано быть."""
    assert _verdict(0.1, 0.5, [_span(1, "mask")], VERDICT_ANY_SPAN) == "doubtful"
    assert _verdict(0.1, 0.5, [_span(1, "mask")], VERDICT_MIN_TWO_TOKENS) == "grounded"


# --- пересчёт по таблице свидетельств --------------------------------------------


def test_apply_rule_and_metrics_over_evidence_table() -> None:
    """Правило считается по таблице свидетельств и воспроизводится вне прогона."""
    module = importlib.import_module("scripts.calibrate_verdict")
    rows = [
        _row(id="tp", y=1, mask_tokens=3, max_fragment_chars=30, flagged_share=0.03, text_sources=["attribution"]),
        _row(id="fp", y=0, mask_tokens=3, max_fragment_chars=30, flagged_share=0.03),
        _row(id="fn", y=1, mask_tokens=1, max_fragment_chars=30, flagged_share=0.01),
        _row(id="tn", y=0),
    ]
    params = {"min_mask_tokens": 2, "min_fragment_chars": 20, "min_flagged_share": 0.0, "text_rule_decisive": True}
    flags = module.apply_rule(rows, params)
    assert flags == [True, True, False, False]
    report = module.metrics(rows, flags)
    assert (report["tp"], report["fp"], report["fn"], report["tn"]) == (1, 1, 1, 1)
    assert report["f1"] == 0.5 and report["fpr"] == 0.5


def test_choose_respects_target_and_never_takes_test() -> None:
    """Выбор идёт по dev, цель объявлена, и сетка не пустеет при недостижимой цели."""
    module = importlib.import_module("scripts.calibrate_verdict")
    rows = [
        _row(id="a", y=1, mask_tokens=6, max_fragment_chars=60, flagged_share=0.06),
        _row(id="b", y=0, mask_tokens=6, max_fragment_chars=60, flagged_share=0.06),
        _row(id="c", y=1, mask_tokens=2, max_fragment_chars=30, flagged_share=0.02),
        _row(id="d", y=0, mask_tokens=2, max_fragment_chars=30, flagged_share=0.02),
    ]
    params, frontier, selection = module.choose(rows, target_fpr=0.0)
    assert frontier, "фронт решений не должен быть пустым"
    assert set(params) == {"min_mask_tokens", "min_fragment_chars", "min_flagged_share", "text_rule_decisive"}
    assert selection["target_fpr"] == 0.0
    assert "тест в выборе не участвует" in selection["rule"].lower()
    flagged = module.metrics(rows, module.apply_rule(rows, params))
    assert flagged["pairs"] == 4
