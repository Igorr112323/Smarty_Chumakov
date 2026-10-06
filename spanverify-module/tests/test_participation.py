"""Тесты оценки доли участия ИИ (``ai_participation``) — требование заявки.

Заявка (У-640148) требует «определение доли участия ИИ в контенте»; в коде это
отдельное поле, не совпадающее с ``ai_share`` (долей недостоверного текста).
Здесь проверяется: поле есть в API/CLI/UI, лежит в [0..1], растёт с долей
машинных оборотов и отличается по смыслу от доли флагов маски.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from spanverify import webui
from spanverify.api import Service
from spanverify.engine import Verifier
from spanverify.participation import ParticipationModel, build_participation_corpus

MODEL_PATH = Path(__file__).resolve().parent.parent / "config" / "participation.json"


def load_model() -> ParticipationModel:
    """Загрузить обученную оценку участия ИИ из ``config/participation.json``."""
    assert MODEL_PATH.is_file(), (
        f"нет {MODEL_PATH}: обучите модель командой "
        "python -m spanverify train --dataset data/demo_pairs.jsonl --out config/weights.json"
    )
    return ParticipationModel.load(MODEL_PATH)


def test_participation_present_and_bounded_in_service() -> None:
    """Ответ сервиса содержит поле ai_participation в диапазоне [0..1]."""
    service = Service(mode="demo")
    body = service.verify(
        {
            "answer": "Необходимо отметить, что срок хранения составляет 5 лет.",
            "context": "Регламент: срок хранения составляет 10 лет.",
        }
    )
    assert "ai_participation" in body, "в ответе /v1/verify нет поля ai_participation"
    value = float(body["ai_participation"])
    assert 0.0 <= value <= 1.0, f"ai_participation вне [0..1]: {value}"
    model_info = service.model()
    assert "measures" in model_info and "ai_participation" in model_info["measures"]
    assert model_info["participation"]["loaded"] is True


def test_participation_grows_with_ai_fraction() -> None:
    """Оценка монотонно растёт с долей машинных предложений в тексте."""
    verifier = Verifier(mode="demo")
    verifier.participation = load_model()
    samples = build_participation_corpus(count=30, seed=777)
    means: dict[float, list[float]] = {}
    for sample in samples:
        matrix = verifier.features_for(sample["text"], sample["context"])
        means.setdefault(sample["ai_fraction"], []).append(verifier.participation.estimate(sample["text"], matrix))
    low = sum(means[0.0]) / len(means[0.0])
    middle = sum(means[0.5]) / len(means[0.5])
    high = sum(means[1.0]) / len(means[1.0])
    assert low <= middle <= high, f"нет монотонности: {low:.3f} / {middle:.3f} / {high:.3f}"
    assert high - low > 0.5, f"слишком слабое различие стилей: {low:.3f} против {high:.3f}"


def test_participation_differs_from_flag_share() -> None:
    """Подтверждённый документом текст в канцелярском стиле: флагов нет, участие есть."""
    verifier = Verifier(mode="demo")
    verifier.participation = load_model()
    text = (
        "Необходимо отметить, что данный регламент обеспечивает комплексное решение задачи. "
        "Следует отметить, что процедура представляет собой последовательность этапов."
    )
    context = f"{text} Инвентаризация проводится раз в год."
    result = verifier.verify(text, context)
    assert result.ai_share_hard == 0.0, "текст подтверждён документом — флагов быть не должно"
    assert result.ai_participation > 0.5, (
        "доля участия ИИ не увидела канцелярский стиль: "
        f"{result.ai_participation:.3f} (это не доля флагов, а отдельная оценка)"
    )


def test_participation_model_roundtrip(tmp_path: Path) -> None:
    """Модель участия ИИ сохраняется и читается без потерь."""
    model = load_model()
    path = tmp_path / "participation.json"
    model.save(path)
    restored = ParticipationModel.load(path)
    assert restored.to_dict() == model.to_dict()


def test_participation_available_without_torch() -> None:
    """Demo-режим отдаёт ai_participation, не импортируя torch/transformers."""
    script = """
import sys

class Blocker:
    def find_module(self, name, path=None):
        return self if name.split(".")[0] in {"torch", "transformers"} else None

    def load_module(self, name):
        raise ImportError("запрещено: " + name)

sys.meta_path.insert(0, Blocker())
from spanverify.engine import Verifier

verifier = Verifier(mode="demo")
result = verifier.verify("Срок хранения составляет 5 лет.", "Регламент: 10 лет.")
print("OK", round(result.ai_participation, 4))
heavy = [name for name in sys.modules if name.split(".")[0] in {"torch", "transformers"}]
assert not heavy, heavy
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(MODEL_PATH.parent.parent),
    )
    assert completed.returncode == 0, completed.stderr
    assert "OK" in completed.stdout, completed.stdout


def test_cli_prints_participation() -> None:
    """CLI печатает оценку участия ИИ отдельной строкой (и в --json есть поле)."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "spanverify",
            "verify",
            "--answer",
            "Срок хранения составляет 5 лет.",
            "--context",
            "Регламент: срок хранения составляет 10 лет.",
        ],
        capture_output=True,
        text=True,
        cwd=str(MODEL_PATH.parent.parent),
    )
    # код 1 — «найдены сомнительные фрагменты», это штатный исход проверки
    assert completed.returncode in (0, 1), completed.stderr
    assert "участия ИИ" in completed.stdout, completed.stdout
    as_json = subprocess.run(
        [
            sys.executable,
            "-m",
            "spanverify",
            "verify",
            "--answer",
            "Срок хранения составляет 5 лет.",
            "--context",
            "Регламент: срок хранения составляет 10 лет.",
            "--json",
        ],
        capture_output=True,
        text=True,
        cwd=str(MODEL_PATH.parent.parent),
    )
    payload = json.loads(as_json.stdout)
    assert "ai_participation" in payload


def test_ui_shows_participation_metric() -> None:
    """В веб-интерфейсе есть отдельная метрика участия ИИ (не подпись про долю флагов)."""
    assert 'id="participation"' in webui.INDEX_HTML
    assert "участия ИИ" in webui.INDEX_HTML
    assert "data.ai_participation" in webui.INDEX_HTML
