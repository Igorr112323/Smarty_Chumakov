"""Запрет тихой подмены режима hf на demo (дефект A3 реестра).

Правило продукта: если запрошен реальный режим (``hf``), ответ должен быть либо
реальным, либо явной ошибкой. Молчаливый переход на лексические суррогаты
недопустим — иначе таблица метрик с колонкой «режим» теряет смысл, и demo
выдаётся за научный результат.

Тесты написаны так, чтобы работать в обоих окружениях: там, где torch и
transformers установлены, проверяется, что признаки действительно считаны
моделью; там, где их нет — что ошибка явная и понятная.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

from spanverify.engine import Verifier
from spanverify.features import hf_features

TORCH_INSTALLED = importlib.util.find_spec("torch") is not None
TRANSFORMERS_INSTALLED = importlib.util.find_spec("transformers") is not None
HF_AVAILABLE = TORCH_INSTALLED and TRANSFORMERS_INSTALLED


def test_hf_missing_dependencies_raise_clear_error() -> None:
    """Без torch и transformers — явная ошибка, а не голый ModuleNotFoundError."""
    if HF_AVAILABLE:
        pytest.skip("torch и transformers установлены: проверяется другой путь")
    from spanverify.backends.base import BackendUnavailable

    with pytest.raises(BackendUnavailable) as caught:
        hf_features("Срок хранения составляет 5 лет.", "Регламент: срок хранения составляет 10 лет.")
    message = str(caught.value)
    assert "hf" in message
    assert "requirements-hf.txt" in message, "сообщение обязано указывать, что установить"


def test_verifier_in_hf_mode_does_not_fall_back_to_demo() -> None:
    """Verifier в режиме hf не возвращает результат demo-режима."""
    if HF_AVAILABLE:
        pytest.skip("torch и transformers установлены: проверяется другой путь")
    from spanverify.backends.base import BackendUnavailable

    verifier = Verifier(mode="hf")
    assert verifier.mode == "hf"
    with pytest.raises(BackendUnavailable):
        verifier.verify("Срок хранения составляет 5 лет.", "Регламент: срок хранения составляет 10 лет.")


def test_demo_mode_is_reported_honestly() -> None:
    """Режим demo обязан честно помечать себя предупреждением, а hf — нет."""
    demo = Verifier(mode="demo")
    assert demo.mode == "demo"
    assert demo.warning, "demo-режим обязан предупреждать о суррогатах"

    hf = Verifier(mode="hf")
    assert hf.mode == "hf"
    assert hf.warning == "", "hf-режим не должен нести предупреждение демо-режима"


def test_hf_features_return_model_backend_when_available() -> None:
    """Если зависимости есть, признаки считает модель, а не лексический суррогат."""
    if not HF_AVAILABLE:
        pytest.skip("torch и transformers не установлены")
    matrix = hf_features(
        "Срок хранения составляет 5 лет.",
        "Регламент: срок хранения составляет 10 лет.",
        model_name="ai-forever/rugpt3small_based_on_gpt2",
    )
    assert matrix.meta.get("backend") == "hf", matrix.meta


def test_cli_returns_error_code_when_hf_unavailable() -> None:
    """CLI в режиме hf без зависимостей возвращает код 2, а не трассировку."""
    if HF_AVAILABLE:
        pytest.skip("torch и transformers установлены: CLI отработает штатно")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "spanverify",
            "verify",
            "--mode",
            "hf",
            "--answer",
            "Срок хранения составляет 5 лет.",
            "--context",
            "Регламент: срок хранения составляет 10 лет.",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "Traceback" not in result.stderr, "пользователь не должен видеть трассировку"
    assert "requirements-hf.txt" in result.stderr, result.stderr
