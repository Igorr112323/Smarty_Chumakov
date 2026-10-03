"""SpanVerify — локализация фрагментов текста, сгенерированных ИИ.

Модуль определяет, какие участки документа написаны языковой моделью,
и оценивает долю участия ИИ. Ядро работает на стандартной библиотеке
Python; режим ``hf`` подключает HuggingFace-модель как опциональную
зависимость.

Публичный API:

    from spanverify import Config, Detector
    det = Detector(Config())
    result = det.analyze(text)
    print(result.ai_fraction, result.spans)
"""

from .config import Config
from .detector import Detector, VerifyResult, Span

__version__ = "1.0.0"
__all__ = ["Config", "Detector", "VerifyResult", "Span", "__version__"]
