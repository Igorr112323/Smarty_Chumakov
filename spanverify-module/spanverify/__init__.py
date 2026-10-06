"""SpanVerify — проверка достоверности ответа относительно контекста.

Продукт решает одну задачу: дан ответ и документ-контекст — найти фрагменты
ответа, которые контекстом не подтверждаются, и вернуть оценку риска.

    from spanverify import Verifier
    verifier = Verifier(mode="demo")          # или mode="hf" (нужны torch+transformers)
    result = verifier.verify(answer, context)
    print(result.verdict, result.score, [s.text for s in result.spans])

Историческая ветка (определение «текст написан ИИ» без контекста) остаётся
доступной через :class:`spanverify.detector.Detector` и команду
``spanverify analyze``; продукт её не использует.

Точки входа: ``python -m spanverify <команда>`` (см. ``spanverify.cli``),
HTTP-сервис — ``spanverify server`` (порт по умолчанию 8765).
"""

from ._version import __version__
from .config import Config
from .detector import Detector, Span, VerifyResult
from .engine import WEIGHTS_FILENAME, VerificationResult, Verifier, WeightsBundle

__all__ = [
    "Config",
    "Detector",
    "Span",
    "VerifyResult",
    "Verifier",
    "VerificationResult",
    "WeightsBundle",
    "WEIGHTS_FILENAME",
    "__version__",
]
