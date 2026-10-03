"""Ядро SpanVerify: оценка текста и локализация машинных фрагментов."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .backends import Backend, get_backend
from .calibration import IsotonicCalibrator
from .config import Config
from .text import Token, sentences, tokenize
from .vectors import knn_density

PUNCT_ONLY = set(".,;:!?…)»\"'")


@dataclass
class Span:
    """Непрерывный фрагмент, помеченный как написанный ИИ."""

    index: int
    start_char: int
    end_char: int
    start_token: int
    end_token: int
    n_tokens: int
    mean_prob: float
    peak_prob: float
    text: str

    @property
    def share_of_text(self) -> float:
        return 0.0

    def to_dict(self, text_length: int | None = None) -> dict[str, Any]:
        data = {
            "index": self.index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "start_token": self.start_token,
            "end_token": self.end_token,
            "n_tokens": self.n_tokens,
            "mean_prob": round(self.mean_prob, 4),
            "peak_prob": round(self.peak_prob, 4),
            "text": self.text,
        }
        if text_length:
            data["share_of_text"] = round((self.end_char - self.start_char) / text_length, 4)
        return data


@dataclass
class VerifyResult:
    """Результат проверки документа."""

    text_length: int
    n_tokens: int
    n_word_tokens: int
    threshold: float
    calibrated: bool
    backend: str
    ai_fraction: float
    ai_fraction_tokens: float
    ai_fraction_soft: float = 0.0
    spans: list[Span] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def share(self) -> float:
        """Доля участия ИИ: пороговая оценка по найденным фрагментам.

        Осознанно консервативна: человеческий текст не должен получать
        ненулевое «обвинение». Мягкая оценка (``ai_fraction_soft``) отдаётся
        отдельным полем как диагностическая и смещена вверх.
        """
        return self.ai_fraction

    @property
    def verdict(self) -> str:
        """Вердикт по доле найденных машинных фрагментов."""
        if self.share >= 0.5:
            return "likely_ai"
        if self.share >= 0.15:
            return "mixed"
        return "likely_human"

    def to_dict(self, include_spans: bool = True, include_text: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "verdict": self.verdict,
            "ai_fraction": round(self.ai_fraction, 4),
            "ai_fraction_soft": round(self.ai_fraction_soft, 4),
            "share": round(self.share, 4),
            "ai_fraction_tokens": round(self.ai_fraction_tokens, 4),
            "threshold": round(self.threshold, 4),
            "calibrated": self.calibrated,
            "backend": self.backend,
            "text_length": self.text_length,
            "n_tokens": self.n_tokens,
            "n_word_tokens": self.n_word_tokens,
            "warnings": self.warnings,
            "meta": self.meta,
            "spans_count": len(self.spans),
        }
        if include_spans:
            data["spans"] = [
                span.to_dict(self.text_length if include_text else None) for span in self.spans
            ]
        return data


class Detector:
    """Детектор «машинных» фрагментов.

    Пример:
        >>> det = Detector(Config(backend="surrogate"))
        >>> result = det.analyze("Текст для проверки...")
        >>> result.ai_fraction >= 0.0
        True
    """

    def __init__(
        self,
        config: Config | None = None,
        backend: Backend | None = None,
        calibrator: IsotonicCalibrator | None | bool = None,
    ) -> None:
        self.config = config or Config.load()
        self.backend = backend or get_backend(
            self.config.backend,
            model=self.config.hf_model,
            max_tokens=self.config.hf_max_tokens,
        )
        if calibrator is None:
            self.calibrator = IsotonicCalibrator.load(self.config.calibration_path)
        elif calibrator is False:
            self.calibrator = None
        else:
            self.calibrator = calibrator  # type: ignore[assignment]

    # ---------- публичный API ----------

    @property
    def backend_name(self) -> str:
        return getattr(self.backend, "name", "unknown")

    def analyze(self, text: str, threshold: float | None = None) -> VerifyResult:
        """Проверить текст и вернуть результат с локализованными фрагментами."""
        cfg = self.config
        thr = float(threshold) if threshold is not None else cfg.threshold

        warnings: list[str] = []
        if self.calibrator is None:
            warnings.append(
                "калибратор не найден: используется сырая оценка без калибровки "
                "(обучите его на своих данных: python -m spanverify calibrate)"
            )
        if self.backend_name == "surrogate":
            warnings.append(
                "ДЕМО-РЕЖИМ (surrogate): проверяется работоспособность конвейера, "
                "а не достоверность текста. Метрики и сравнения допустимы только "
                "в режиме 'hf' на размеченных данных."
            )
        if not text or not text.strip():
            return VerifyResult(
                text_length=0, n_tokens=0, n_word_tokens=0, threshold=thr,
                calibrated=self.calibrator is not None, backend=self.backend_name,
                ai_fraction=0.0, ai_fraction_tokens=0.0, spans=[], warnings=warnings,
                meta={"empty": True},
            )

        tokens, word_tokens, smoothed_raw, probs, features = self.score_text(text)

        selected = [i for i, p in enumerate(probs) if p >= thr]
        spans = self._build_spans(probs, selected, word_tokens, text)
        spans = _expand_to_sentences(spans, probs, word_tokens, text, cfg)

        ai_chars = sum(span.end_char - span.start_char for span in spans)
        ai_words = sum(span.n_tokens for span in spans)

        # Мягкая оценка: вероятность каждого слова распространяется на символы
        # слова и следующий за ним разделитель. Не зависит от выбора порога,
        # поэтому используется как оценка доли участия ИИ в документе.
        soft_chars = 0.0
        for i, token in enumerate(word_tokens):
            next_start = word_tokens[i + 1].start if i + 1 < len(word_tokens) else len(text)
            soft_chars += probs[i] * (next_start - token.start)
        ai_fraction_soft = soft_chars / len(text) if text else 0.0

        meta = dict(features.meta)
        meta.update(
            {
                "mean_raw": round(sum(smoothed_raw) / len(smoothed_raw), 4) if smoothed_raw else 0.0,
                "mean_prob": round(sum(probs) / len(probs), 4) if probs else 0.0,
                "config": cfg.describe(),
                "calibration": (self.calibrator.meta if self.calibrator else None),
            }
        )

        return VerifyResult(
            text_length=len(text),
            n_tokens=len(tokens),
            n_word_tokens=len(word_tokens),
            threshold=thr,
            calibrated=self.calibrator is not None,
            backend=self.backend_name,
            ai_fraction=ai_chars / len(text) if text else 0.0,
            ai_fraction_tokens=ai_words / len(word_tokens) if word_tokens else 0.0,
            ai_fraction_soft=ai_fraction_soft,
            spans=spans,
            warnings=warnings,
            meta=meta,
        )

    def score_text(self, text: str):
        """Единый проход по документу.

        Возвращает ``(tokens, word_tokens, smoothed_raw, probs, features)``:
        сырые сглаженные оценки и калиброванные вероятности по словам. Все
        потребители (analyze, explain, обучение калибратора, эксперименты)
        используют этот метод, поэтому оценка всегда в одной шкале.
        """
        cfg = self.config
        tokens = tokenize(text)
        word_tokens = [t for t in tokens if t.is_word]
        words = [t.text for t in word_tokens]
        features = self.backend.process(words, text, dim=cfg.vector_dim)
        raw = self.raw_scores(
            features.predictability, features.vectors, informative=features.informative
        )
        smoothed_raw = _moving_average(raw, cfg.smoothing_window)
        probs = self._calibrate(smoothed_raw)
        return tokens, word_tokens, smoothed_raw, probs, features

    def token_probabilities(self, text: str) -> list[float]:
        """Калиброванные вероятности «токен написан ИИ» по словам документа."""
        return list(self.score_text(text)[3])

    def explain(self, text: str, threshold: float | None = None) -> dict[str, Any]:
        """Результат + покадровая таблица для отладки и визуализации."""
        thr = float(threshold) if threshold is not None else self.config.threshold
        result = self.analyze(text, threshold=thr)
        _, word_tokens, smoothed_raw, probs, _ = self.score_text(text)
        if not word_tokens:
            return {"result": result.to_dict(), "tokens": []}
        table = [
            {
                "token": word_tokens[i].text,
                "start": word_tokens[i].start,
                "end": word_tokens[i].end,
                "raw": round(smoothed_raw[i], 4),
                "prob": round(probs[i], 4),
                "flag": probs[i] >= thr,
            }
            for i in range(len(word_tokens))
        ]
        return {"result": result.to_dict(), "tokens": table}

    # ---------- внутреннее ----------

    def raw_scores(
        self,
        predictability: Sequence[float],
        vectors: Sequence[dict[int, float]] | None,
        informative: Sequence[bool] | None = None,
    ) -> list[float]:
        """Взвешенная комбинация предсказуемости и контекстной плотности.

        Оценки неинформативных токенов (служебных слов) восстанавливаются по
        ближайшему окружению: собственного стилевого сигнала у них нет, но
        разрывать ими фрагмент нельзя.
        """
        cfg = self.config
        if not predictability:
            return []
        mask = list(informative) if informative else [True] * len(predictability)
        density = (
            knn_density(vectors, k=cfg.k_neighbors, mask=mask)
            if vectors and len(vectors) == len(predictability)
            else [0.0] * len(predictability)
        )
        ref = max(1e-6, cfg.density_ref)
        out: list[float] = []
        for pred, dens in zip(predictability, density):
            density_scaled = min(1.0, max(0.0, dens / ref))
            score = cfg.w_predictability * float(pred) + cfg.w_density * density_scaled
            out.append(min(1.0, max(0.0, score)))
        return _impute_uninformative(out, mask, cfg.smoothing_window)

    def _calibrate(self, raw: Sequence[float]) -> list[float]:
        if self.calibrator is None or not raw:
            return [float(x) for x in raw]
        return self.calibrator.transform(raw)

    def _build_spans(
        self,
        probs: Sequence[float],
        selected: Sequence[int],
        word_tokens: Sequence[Token],
        text: str,
    ) -> list[Span]:
        cfg = self.config
        if not selected:
            return []

        groups: list[list[int]] = [[selected[0]]]
        for idx in selected[1:]:
            if idx - groups[-1][-1] <= cfg.merge_gap_tokens + 1:
                groups[-1].append(idx)
            else:
                groups.append([idx])

        spans: list[Span] = []
        for group in groups:
            if len(group) < cfg.min_span_tokens:
                continue
            first, last = group[0], group[-1]
            start_char = word_tokens[first].start
            end_char = word_tokens[last].end
            end_char = _extend_end(text, end_char)
            values = [probs[i] for i in group]
            spans.append(
                Span(
                    index=len(spans),
                    start_char=start_char,
                    end_char=end_char,
                    start_token=first,
                    end_token=last + 1,
                    n_tokens=len(group),
                    mean_prob=sum(values) / len(values),
                    peak_prob=max(values),
                    text=text[start_char:end_char],
                )
            )
        return spans


def _expand_to_sentences(
    spans: Sequence[Span],
    probs: Sequence[float],
    word_tokens: Sequence[Token],
    text: str,
    config: Config,
) -> list[Span]:
    """Расширить найденные фрагменты до границ предложений и склеить соседние.

    Фрагмент машинного текста — это, как правило, целые предложения, а не
    отдельные слова. Расширение убирает «рваную» разметку и делает границы
    интерпретируемыми; после склейки повторно применяется ограничение на
    минимальную длину фрагмента.
    """
    if not spans or not word_tokens:
        return list(spans)

    bounds = sentences(text)
    if not bounds:
        return list(spans)

    ranges: list[tuple[int, int]] = []
    for span in spans:
        start = _sentence_index(bounds, span.start_char)
        end = _sentence_index(bounds, max(span.start_char, span.end_char - 1))
        ranges.append((bounds[start][0], bounds[end][1]))

    ranges.sort()
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    out: list[Span] = []
    for start, end in merged:
        # обрезаем внешние пробелы, сохраняя смещения
        fragment = text[start:end]
        lead = len(fragment) - len(fragment.lstrip())
        trail = len(fragment) - len(fragment.rstrip())
        start += lead
        end -= trail

        inside = [
            i
            for i, token in enumerate(word_tokens)
            if token.start >= start and token.end <= end
        ]
        if len(inside) < config.min_span_tokens:
            continue
        values = [probs[i] for i in inside]
        out.append(
            Span(
                index=len(out),
                start_char=start,
                end_char=end,
                start_token=inside[0],
                end_token=inside[-1] + 1,
                n_tokens=len(inside),
                mean_prob=sum(values) / len(values),
                peak_prob=max(values),
                text=text[start:end],
            )
        )
    return out


def _sentence_index(bounds: Sequence[tuple[int, int]], position: int) -> int:
    """Индекс предложения, содержащего символьную позицию."""
    for i, (start, end) in enumerate(bounds):
        if start <= position < end:
            return i
    return len(bounds) - 1 if position >= bounds[-1][1] else 0


def _impute_uninformative(
    scores: Sequence[float], mask: Sequence[bool], window: int
) -> list[float]:
    """Заменить оценки неинформативных токенов локальным средним по окружению."""
    if not scores or all(mask):
        return [float(s) for s in scores]
    span = max(1, window)
    out = [float(s) for s in scores]
    for i, informative in enumerate(mask):
        if informative:
            continue
        lo, hi = max(0, i - span), min(len(scores), i + span + 1)
        local = [scores[j] for j in range(lo, hi) if mask[j]]
        if local:
            out[i] = sum(local) / len(local)
    return out


def _moving_average(values: Sequence[float], window: int) -> list[float]:
    if window <= 1 or len(values) < 3:
        return [float(v) for v in values]
    half = max(1, window // 2)
    out: list[float] = []
    n = len(values)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = values[lo:hi]
        out.append(sum(chunk) / len(chunk))
    return out


def _extend_end(text: str, end: int, limit: int = 4) -> int:
    """Приклеить к фрагменту замыкающую пунктуацию (для читаемой разметки)."""
    i = end
    added = 0
    while i < len(text) and added < limit:
        ch = text[i]
        if ch.isspace():
            # пробел приклеиваем только если дальше идёт знак препинания
            if i + 1 < len(text) and text[i + 1] in PUNCT_ONLY:
                i += 1
                continue
            break
        if ch in PUNCT_ONLY:
            i += 1
            added += 1
            continue
        break
    return i


def load_detector(config_path: str | Path | None = None, **overrides: Any) -> Detector:
    """Удобный конструктор: конфиг из файла + переопределения."""
    config = Config.load(config_path).with_overrides(**overrides)
    return Detector(config)
