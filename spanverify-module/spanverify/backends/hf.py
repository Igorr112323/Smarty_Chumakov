"""Бэкенд на реальной языковой модели (HuggingFace transformers).

Что считается:

* **энтропия внимания последнего слоя.** Для каждого токена-запроса берётся
  распределение внимания по всем ключам, усредняется по головам, считается
  энтропия Шеннона и нормируется на ``log(seq_len)``. Предсказуемость =
  ``1 - H_norm``: модель, «уверенная» в токене, даёт низкую энтропию.
* **контекстные эмбеддинги.** Вектор токена = скрытое состояние последнего
  слоя, усреднённое по сабтокенам, попавшим в этот токен (по offsets
  быстрого токенизатора). Используются для kNN-плотности.

Это приближение к «перплексии на уровне модели-судьи», которое работает
локально и не требует API-ключей. Для научных выводов нужен
размеченный корпус: см. ``scripts/run_experiments.py``.
"""

from __future__ import annotations

from typing import Any, Sequence

from .base import Backend, BackendResult, BackendUnavailable


class HFBackend(Backend):
    name = "hf"
    description = "энтропия внимания последнего слоя + контекстные эмбеддинги"

    def __init__(
        self,
        model: str = "cointegrated/rubert-tiny2",
        max_tokens: int = 512,
        device: str | None = None,
        **_: Any,
    ) -> None:
        self.model_name = model
        self.max_tokens = max_tokens
        self.device = device
        self._cache: dict[str, Any] = {}

    # ---------- доступность ----------

    @staticmethod
    def dependencies() -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception as exc:  # pragma: no cover - зависит от окружения
            return False, (
                "режим 'hf' требует torch и transformers: "
                "pip install -r requirements-hf.txt "
                f"(причина: {type(exc).__name__}: {exc})"
            )
        return True, ""

    def available(self) -> bool:
        return self.dependencies()[0]

    # ---------- ленивая загрузка ----------

    def _load(self) -> tuple[Any, Any, Any]:
        ok, reason = self.dependencies()
        if not ok:
            raise BackendUnavailable(reason)
        if "model" not in self._cache:
            import torch
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            model = AutoModel.from_pretrained(
                self.model_name, output_attentions=True, output_hidden_states=True
            )
            model.eval()
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            self._cache.update(model=model, tokenizer=tokenizer, torch=torch, device=device)
        return self._cache["model"], self._cache["tokenizer"], self._cache["torch"]

    # ---------- признаки ----------

    def process(self, words: Sequence[str], text: str, dim: int = 4096) -> BackendResult:
        model, tokenizer, torch = self._load()
        device = self._cache["device"]

        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_tokens,
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)

        attentions = outputs.attentions[-1][0]  # (heads, seq, seq)
        hidden = outputs.hidden_states[-1][0]    # (seq, hidden_size)

        eps = 1e-9
        probs = attentions.clamp_min(eps)
        entropy = -(probs * probs.log()).sum(dim=-1)          # (heads, seq)
        entropy = entropy.mean(dim=0)                          # (seq,)
        seq_len = entropy.shape[0]
        import math

        entropy = entropy / max(eps, math.log(max(2, seq_len)))  # -> [0, 1]
        predictability = (1.0 - entropy).tolist()

        # Привязка сабтокенов к словам через offsets.
        word_spans = _word_spans(words, text, tokenizer, offsets)
        word_pred: list[float] = []
        vectors: list[dict[int, float]] = []
        hidden_list = hidden.tolist()

        for start, end in word_spans:
            if start is None or start >= end:
                word_pred.append(0.5)
                vectors.append({})
                continue
            window = range(start, min(end, seq_len))
            values = [predictability[i] for i in window]
            word_pred.append(sum(values) / len(values) if values else 0.5)
            vec_hidden = [0.0] * len(hidden_list[0])
            for i in window:
                for j, value in enumerate(hidden_list[i]):
                    vec_hidden[j] += value
            count = max(1, len(list(window)))
            vec_hidden = [v / count for v in vec_hidden]
            vectors.append(_project(vec_hidden, dim))

        return BackendResult(
            predictability=word_pred,
            vectors=vectors,
            meta={
                "backend": self.name,
                "model": self.model_name,
                "seq_len": seq_len,
                "device": str(device),
                "mean_attention_entropy": round(sum(predictability) / len(predictability), 4)
                if predictability
                else 0.0,
            },
        )


def _word_spans(words: Sequence[str], text: str, tokenizer: Any, offsets: list[list[int]]):
    """Диапазоны сабтокенов для каждого слова (по символьным смещениям)."""
    from ..text import tokenize

    tokens = [t for t in tokenize(text) if t.is_word]
    spans: list[tuple[int | None, int | None]] = []
    for tok in tokens:
        first = last = None
        for i, (start, end) in enumerate(offsets):
            if end <= start:  # спецтокены
                continue
            if start < tok.end and end > tok.start:
                if first is None:
                    first = i
                last = i
        spans.append((first, None if first is None else last + 1))
    if len(spans) < len(words):
        spans.extend([(None, None)] * (len(words) - len(spans)))
    return spans[: len(words)]


def _project(vector: list[float], dim: int, gram_n: int = 2) -> dict[int, float]:
    """Проекция плотного эмбеддинга в разреженное хешированное пространство.

    Знаки случайных проекций фиксированы детерминированно, поэтому векторы
    воспроизводимы между запусками и совместимы с kNN-плотностью.
    """
    from hashlib import blake2b

    vector = vector[:dim]
    norm = sum(v * v for v in vector) ** 0.5
    if norm <= 0:
        return {}
    out: dict[int, float] = {}
    for i in range(0, len(vector), gram_n):
        chunk = vector[i : i + gram_n]
        value = sum(chunk)
        digest = blake2b(f"{i}:{len(chunk)}".encode(), digest_size=8).digest()
        sign = 1.0 if digest[0] % 2 == 0 else -1.0
        idx = int.from_bytes(digest, "big") % dim
        out[idx] = out.get(idx, 0.0) + sign * value / norm
    return out
