"""Движок верификации: контекст + ответ → риск, фрагменты, доля участия ИИ.

Конвейер (соответствует спецификации):

    токены ответа
      → три признака (attention_entropy, ctx_attention_mass, embedding_density)
      → нормировка и взвешенная комбинация r = w1·Ĥ + w2·(1 − m̂) + w3·d̂
      → сглаживание окном 3
      → порог по span-маске T = μ + z·σ
      → склейка соседних токенов и расширение до границ предложений
      → метки likely_hallucination (risk ≥ 0.75) / doubtful (risk ≥ T)
      → оценка ответа s = 0.5·mean(r) + 0.5·mean(верхние 20 % r)
      → изотоническая калибровка (PAVA) → порог по целевому FPR

Доля участия ИИ в ответе считается отдельным, независимым от риска способом
(стилометрические признаки, см. ``detector.py``): две оценки —
``ai_share_hard`` (доля токенов выше порога) и ``ai_share_soft`` (средняя
калиброванная вероятность «машинный токен»).
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .calibration import IsotonicCalibrator
from .config import Config, read_runtime_text, runtime_roots
from .core import (
    ContextChunks,
    SpanResult,
    Token,
    VerificationResult,
    mean,
    split_chunks,
    split_sentences,
    tokenize_with_offsets,
)
from .detector import Detector
from .features import (
    DEFAULT_WEIGHTS,
    DEMO_WARNING,
    FEATURE_NAMES,
    FeatureMatrix,
    combine,
    extract_features,
    is_scored_token,
    number_attribution,
)
from .participation import PARTICIPATION_FILENAME, ParticipationModel

WEIGHTS_FILENAME = "config/weights.json"
HALLUCINATION_LABEL_RISK = 0.75
TOP_SHARE = 0.2
SMOOTH_WINDOW = 3


@dataclass
class WeightsBundle:
    """Обученные параметры: веса признаков, пороги, калибровка, голова."""

    weights: dict[str, float]
    threshold: float
    target_fpr: float = 0.1
    span_z: float = 1.0
    span_floor: float = 0.35
    span_cap: float = 0.6
    isotonic: IsotonicCalibrator | None = None
    head: dict[str, Any] | None = None
    folds: list[dict[str, Any]] | None = None
    seed: int = 42
    version: str = "1.1.0"
    mode: str = "demo"
    meta: dict[str, Any] | None = None
    source: str = "defaults"  # откуда взяты параметры: disk | embedded | defaults

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": {name: round(self.weights.get(name, 0.0), 4) for name in FEATURE_NAMES},
            "threshold": round(self.threshold, 6),
            "target_fpr": self.target_fpr,
            "span_z": self.span_z,
            "span_floor": self.span_floor,
            "span_cap": self.span_cap,
            "isotonic": {
                "x": [round(v, 6) for v in (self.isotonic.thresholds if self.isotonic else [])],
                "y": [round(v, 6) for v in (self.isotonic.values if self.isotonic else [])],
            },
            "head": self.head or {"type": "none", "file": None},
            "folds": self.folds or [],
            "seed": self.seed,
            "version": self.version,
            "mode": self.mode,
            "meta": self.meta or {},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WeightsBundle:
        isotonic_data = data.get("isotonic") or {}
        calibrator = (
            IsotonicCalibrator(
                thresholds=[float(v) for v in isotonic_data.get("x", [])],
                values=[float(v) for v in isotonic_data.get("y", [])],
            )
            if isotonic_data.get("x")
            else None
        )
        return cls(
            weights={name: float(data.get("weights", {}).get(name, DEFAULT_WEIGHTS[name])) for name in FEATURE_NAMES},
            threshold=float(data.get("threshold", 0.5)),
            target_fpr=float(data.get("target_fpr", 0.1)),
            span_z=float(data.get("span_z", 1.0)),
            span_floor=float(data.get("span_floor", 0.35)),
            span_cap=float(data.get("span_cap", 0.6)),
            isotonic=calibrator,
            head=data.get("head"),
            folds=list(data.get("folds", [])),
            seed=int(data.get("seed", 42)),
            version=str(data.get("version", "1.1.0")),
            mode=str(data.get("mode", "demo")),
            meta=dict(data.get("meta", {})),
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return target

    @classmethod
    def load(cls, path: str | Path | None = WEIGHTS_FILENAME) -> WeightsBundle:
        """Загрузить обученные параметры; при отсутствии — значения по умолчанию.

        Порядок поиска для относительного пути (он же путь по умолчанию):
        содержимое бандла (``sys._MEIPASS`` у собранного ``.exe``) → папка рядом
        с исполняемым файлом → текущая папка. Так ``.exe`` работает автономно
        (веса внутри файла), но обученные параметры, положенные рядом с ним или
        указанные явно, имеют приоритет. Абсолютный путь используется как есть.
        """
        if path is None:
            return cls(weights=dict(DEFAULT_WEIGHTS), threshold=0.5)
        target = Path(path)
        if target.is_absolute():
            if target.is_file():
                with target.open("r", encoding="utf-8") as handle:
                    bundle = cls.from_dict(json.load(handle))
                bundle.source = "disk"
                return bundle
            return cls(weights=dict(DEFAULT_WEIGHTS), threshold=0.5)
        meipass = getattr(sys, "_MEIPASS", None)
        if getattr(sys, "frozen", False):
            # Собранное приложение: сначала параметры рядом с .exe (осознанное
            # переобучение пользователем), затем встроенные в файл, затем
            # текущая папка — она не должна перебивать содержимое бандла.
            roots = [Path(sys.executable).resolve().parent]
            if meipass:
                roots.append(Path(meipass))
            roots.append(Path.cwd())
        else:
            roots = runtime_roots()
        for root in roots:
            candidate = Path(root) / target
            if candidate.is_file():
                with candidate.open("r", encoding="utf-8") as handle:
                    bundle = cls.from_dict(json.load(handle))
                bundle.source = "embedded" if meipass and Path(root) == Path(meipass) else "disk"
                return bundle
        embedded = read_runtime_text(str(target))
        if embedded:
            bundle = cls.from_dict(json.loads(embedded))
            bundle.source = "embedded"
            return bundle
        return cls(weights=dict(DEFAULT_WEIGHTS), threshold=0.5)

    @property
    def loaded(self) -> bool:
        """Обучены ли параметры (а не взяты значения по умолчанию)."""
        return self.source in {"disk", "embedded"}


class Verifier:
    """Проверка ответа относительно контекста и оценка доли участия ИИ."""

    def __init__(
        self,
        config: Config | None = None,
        weights: WeightsBundle | None = None,
        mode: str | None = None,
        detector: Detector | None = None,
        model_name: str | None = None,
        weights_path: str | Path | None = WEIGHTS_FILENAME,
    ) -> None:
        self.config = config or Config.load()
        # Явно переданные веса имеют приоритет: иначе обучение, которое считает
        # сквозные метрики «в памяти», случайно перечитало бы файл с диска.
        self.bundle = weights if weights is not None else WeightsBundle.load(weights_path)
        self.weights_loaded = weights is not None or self.bundle.loaded
        self.mode = (mode or self.bundle.mode or self.config.backend or "demo").lower()
        self.model_name = model_name or getattr(self.config, "hf_model", "ai-forever/rugpt3small_based_on_gpt2")
        self._detector = detector
        # Оценка доли участия ИИ (требование заявки) — отдельная голова; если
        # файла нет, поле остаётся нулевым и это видно в stats.
        self.participation = self._load_participation()

    # ---------------------------------------------------------------- доступ

    def _load_participation(self) -> ParticipationModel | None:
        """Прочитать оценку доли участия ИИ из ``config/participation.json``."""
        for root in runtime_roots():
            path = root / PARTICIPATION_FILENAME
            if path.is_file():
                try:
                    return ParticipationModel.load(path)
                except (OSError, ValueError, KeyError):
                    return None
        return None

    @property
    def detector(self) -> Detector:
        """Детектор доли участия ИИ (ленивая инициализация)."""
        if self._detector is None:
            config = self.config
            if self.mode == "hf" and config.backend != "hf":
                config = config.with_overrides(backend="hf", hf_model=self.model_name)
            self._detector = Detector(config)
        return self._detector

    @property
    def warning(self) -> str:
        return DEMO_WARNING if self.mode == "demo" else ""

    # ---------------------------------------------------------------- основной

    def verify(
        self,
        answer: str,
        context: str | Sequence[str] | None = None,
        prompt: str | None = None,
        with_tokens: bool = False,
        max_new_tokens: int | None = None,
    ) -> VerificationResult:
        """Проверить ответ относительно контекста.

        ``prompt`` и ``max_new_tokens`` принимаются для совместимости с
        контрактом API: в текущей реализации они учитываются только в режиме
        ``hf`` (добавляются к входной последовательности).
        """
        started = time.perf_counter()
        answer = answer or ""
        context_text = prompt if prompt else context

        if not answer.strip():
            return VerificationResult(
                score=0.0,
                is_hallucination=False,
                ai_share=0.0,
                ai_share_hard=0.0,
                threshold=self.bundle.threshold,
                spans=[],
                stats={"token_count": 0, "empty": True, "warning": self.warning},
                latency_ms=(time.perf_counter() - started) * 1000,
                mode=self.mode,
                warning=self.warning,
                verdict="empty",
            )

        tokens = tokenize_with_offsets(answer)
        features = self._features(answer, context_text, tokens)
        risk = combine(
            features.attention_entropy,
            features.ctx_attention_mass,
            features.embedding_density,
            self.bundle.weights,
        )
        if self.bundle.head and self.bundle.head.get("type") == "logreg":
            head_risk = _head_risk(self.bundle.head, features, risk, tokens)
            if head_risk is not None:
                risk = head_risk
        # Пунктуация и короткие служебные слова не могут быть выдумкой: их риск
        # обнуляется до сглаживания, иначе соседние пики «размываются» их шумом.
        scored_indices = [i for i, token in enumerate(tokens) if is_scored_token(token.text)]
        masked = [risk[i] if i in set(scored_indices) else 0.0 for i in range(len(risk))]
        smoothed = _smooth(masked, SMOOTH_WINDOW)
        scored = [smoothed[i] for i in scored_indices] or [0.0]
        span_threshold = span_threshold_for(scored, self.bundle.span_z, self.bundle.span_floor, self.bundle.span_cap)
        spans = self._build_spans(answer, tokens, smoothed, span_threshold)
        # Правило привязки числа к объекту (дефект D): текстовое, поверх маски.
        spans = _merge_spans([*spans, *self._attribution_spans(answer, context_text, tokens, smoothed, span_threshold)])

        raw_score = _answer_score(scored)
        score = self._calibrate(raw_score)
        threshold = self.bundle.threshold
        ai_share_soft, ai_share_hard = self._shares(tokens, smoothed, scored_indices, spans)
        ai_participation = self.participation.estimate(answer, features) if self.participation is not None else 0.0

        tokens_payload = _token_payload(tokens, features, smoothed, span_threshold)
        stats = {
            "token_count": len(tokens),
            "scored_tokens": len(scored_indices),
            "number_attribution": number_attribution(answer, context_text) if context_text else [],
            "mean_risk": round(mean(scored), 4),
            "p90_risk": round(_percentile(scored, 90), 4),
            "raw_score": round(raw_score, 4),
            "span_threshold": round(span_threshold, 4),
            "span_z": self.bundle.span_z,
            "span_floor": self.bundle.span_floor,
            "weights": {name: round(self.bundle.weights.get(name, 0.0), 3) for name in FEATURE_NAMES},
            "features": features.meta,
            "ai_participation": round(ai_participation, 4),
            "ai_participation_calibrated_on": (
                self.participation.calibrated_on if self.participation is not None else None
            ),
            "warning": self.warning,
        }

        return VerificationResult(
            score=score,
            is_hallucination=score >= threshold,
            ai_share=ai_share_soft,
            ai_share_hard=ai_share_hard,
            ai_participation=ai_participation,
            threshold=threshold,
            spans=spans,
            tokens=tokens_payload if with_tokens else [],
            stats=stats,
            latency_ms=(time.perf_counter() - started) * 1000,
            mode=self.mode,
            warning=self.warning,
            verdict=_verdict(score, threshold, spans),
        )

    # ---------------------------------------------------------------- оценка

    def evaluate(
        self,
        pairs: Sequence[dict],
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Сквозная оценка конвейера на размеченных парах «контекст — ответ».

        Оценивается ровно то, что отдаёт :meth:`verify` (не отдельная ветка
        кода): метрики по токенам, F1 по фрагментам (совпадение с IoU ≥ 0.5) и
        разделение ответов. Метки берутся из ``labels`` корпуса по символам.
        """
        from .dataset import Pair  # noqa: PLC0415 - локально, чтобы не плодить импортный цикл

        labels: list[int] = []
        flags: list[bool] = []
        risks: list[float] = []
        answer_labels: list[int] = []
        answer_scores: list[float] = []
        predicted: list[list[tuple[int, int]]] = []
        truth: list[list[tuple[int, int]]] = []
        truth_expanded: list[list[tuple[int, int]]] = []
        verdict_labels: list[int] = []
        verdict_flags: list[bool] = []

        for pair in pairs:
            data = pair.to_dict() if isinstance(pair, Pair) else pair
            result = self.verify(data.get("answer", ""), data.get("context", ""), with_tokens=True)
            tokens = tokenize_with_offsets(data.get("answer", ""))
            pair_truth = [(int(start), int(end)) for start, end, label in data.get("labels", []) if int(label) == 1]
            pair_predicted = [(span.start, span.end) for span in result.spans]
            pair_truth_expanded = [_expand_to_sentence(data.get("answer", ""), start, end) for start, end in pair_truth]

            for index, token in enumerate(tokens):
                if not is_scored_token(token.text):
                    continue
                token_rows = result.tokens[index] if index < len(result.tokens) else {}
                labels.append(1 if any(token.start < end and token.end > start for start, end in pair_truth) else 0)
                flags.append(bool(token_rows.get("flagged")))
                risks.append(float(token_rows.get("risk", 0.0)))

            answer_labels.append(1 if pair_truth else 0)
            answer_scores.append(result.score)
            # Вердикт — уровень ответа: ``doubtful`` приходит и от текстового правила
            # привязки числа к объекту, которое иначе не видно в метриках по токенам.
            verdict_labels.append(1 if pair_truth else 0)
            verdict_flags.append(result.verdict not in {"grounded", "empty"})
            predicted.append(pair_predicted)
            truth.append(pair_truth)
            truth_expanded.append(pair_truth_expanded)

        token_metrics = _token_metrics(labels, flags, risks)
        span_metrics = _span_f1(predicted, truth, iou_threshold=0.5)
        # Конвейер намеренно расширяет найденные токены до границ предложения,
        # поэтому строгий IoU с узкой разметкой («5» против целого предложения)
        # мало информативен. Поэтому дополнительно считаем: (а) полноту по
        # покрытию — размеченный фрагмент целиком попал в найденный; (б) F1 при
        # том же расширении разметки, то есть качество склейки и расширения.
        span_metrics["recall_containment"] = _containment_recall(predicted, truth)
        span_metrics["f1_expanded_labels"] = _span_f1(predicted, truth_expanded, iou_threshold=0.5)["f1"]
        span_metrics["mean_width_ratio"] = _mean_width_ratio(predicted, truth)
        answer_metrics = _answer_metrics(
            answer_labels, answer_scores, threshold if threshold is not None else self.bundle.threshold
        )
        verdict_metrics = _binary_metrics(verdict_labels, verdict_flags)
        return {
            "tokens": token_metrics,
            "spans": span_metrics,
            "answers": answer_metrics,
            "verdicts": verdict_metrics,
            "pairs": len(truth),
            "mode": self.mode,
            "warning": self.warning if self.mode != "hf" else "",
        }

    # ---------------------------------------------------------------- шаги

    def features_for(
        self, answer: str, context: str | Sequence[str] | None, tokens: Sequence[Token] | None = None
    ) -> FeatureMatrix:
        """Публичный доступ к признакам (используется обучением и отчётами)."""
        return self._features(answer, context, tokens or tokenize_with_offsets(answer))

    def _features(self, answer: str, context: str | Sequence[str] | None, tokens: Sequence[Token]) -> FeatureMatrix:
        if self.mode == "hf":
            return extract_features(answer, context, mode="hf", model_name=self.model_name, answer_tokens=tokens)
        return extract_features(answer, context, mode="demo", answer_tokens=tokens)

    def _span_threshold(self, risk: Sequence[float]) -> float:
        """Порог маски для текущего набора весов (см. :func:`span_threshold_for`)."""
        return span_threshold_for(risk, self.bundle.span_z, self.bundle.span_floor, self.bundle.span_cap)

    def _calibrate(self, raw_score: float) -> float:
        if self.bundle.isotonic is None:
            return raw_score
        return self.bundle.isotonic.transform_one(raw_score)

    def _shares(
        self,
        tokens: Sequence[Token],
        smoothed: Sequence[float],
        scored_indices: Sequence[int],
        spans: Sequence[SpanResult] = (),
    ) -> tuple[float, float]:
        """Доли «недостоверного» текста в ответе (поля ai_share/ai_share_hard).

        * ``ai_share`` (мягкая) — средняя калиброванная вероятность
          недостоверности по содержательным токенам: не зависит от порога и
          показывает, «насколько тревожный» ответ в целом;
        * ``ai_share_hard`` (жёсткая) — доля содержательных токенов, попавших
          выше порога маски, то есть прямо помеченных конвейером.

        Названия полей исторические (контракт API), смысл — доля недостоверного
        текста, а не «доля ИИ»: это разные оси, и здесь измеряется вторая.
        """
        if not scored_indices:
            return 0.0, 0.0
        mask_threshold = span_threshold_for(
            [smoothed[index] for index in scored_indices],
            self.bundle.span_z,
            self.bundle.span_floor,
            self.bundle.span_cap,
        )
        # Токены, попавшие в отчётные фрагменты (в том числе по правилу привязки
        # числа к объекту), считаются помеченными: иначе ответ с найденным
        # фрагментом сообщал бы «доля спорного текста 0.000».
        in_span = {
            index for index in scored_indices if any(span.start <= tokens[index].start < span.end for span in spans)
        }
        probabilities = [1.0 if index in in_span else self._calibrate(smoothed[index]) for index in scored_indices]
        soft = sum(probabilities) / len(probabilities)
        hard_numerator = sum(1 for index in scored_indices if index in in_span or smoothed[index] >= mask_threshold)
        hard = hard_numerator / len(scored_indices)
        return min(1.0, max(0.0, soft)), min(1.0, max(0.0, hard))

    def _attribution_spans(
        self,
        answer: str,
        context: str | None,
        tokens: Sequence[Token],
        risk: Sequence[float],
        threshold: float,
    ) -> list[SpanResult]:
        """Фрагменты по правилу привязки числа к объекту (дефект D).

        Правило текстовое, а не статистическое: если число ответа совпадает с
        числом **другого** объекта документа, оно не подтверждено — независимо от
        того, как его оценили признаки и обученная голова. Поэтому фрагмент
        добавляется поверх обычной маски.
        """
        if not context:
            return []
        allowed = {id(token) for token in tokens}
        found: list[SpanResult] = []
        for item in number_attribution(answer, context):
            start, end = int(item["start"]), int(item["end"])
            token_risk = [
                risk[index]
                for index, token in enumerate(tokens)
                if id(token) in allowed and token.start >= start and token.end <= end
            ]
            risk_value = max([*token_risk, threshold])
            span_start, span_end = _expand_to_sentence(answer, start, end)
            fragment = answer[span_start:span_end]
            if not fragment.strip():
                continue
            found.append(
                SpanResult(
                    start=span_start,
                    end=span_end,
                    text=fragment,
                    risk=risk_value,
                    label="likely_hallucination" if risk_value >= HALLUCINATION_LABEL_RISK else "doubtful",
                    n_tokens=len(token_risk),
                )
            )
        return found

    def _build_spans(
        self,
        answer: str,
        tokens: Sequence[Token],
        risk: Sequence[float],
        threshold: float,
    ) -> list[SpanResult]:
        """Маска → склейка → расширение до границ предложений → метки."""
        # Маска строится только по содержательным токенам: пунктуация и
        # служебные слова не могут быть «недостоверными» сами по себе, они
        # лишь попадают внутрь найденного фрагмента при расширении.
        flagged = [
            index
            for index, token in enumerate(tokens)
            if index < len(risk) and is_scored_token(token.text) and risk[index] >= threshold
        ]
        if not flagged:
            return []

        groups: list[list[int]] = [[flagged[0]]]
        for index in flagged[1:]:
            if index - groups[-1][-1] <= 1:
                groups[-1].append(index)
            else:
                groups.append([index])

        spans: list[SpanResult] = []
        for group in groups:
            start = tokens[group[0]].start
            end = tokens[group[-1]].end
            start, end = _expand_to_sentence(answer, start, end)
            fragment = answer[start:end]
            if not fragment.strip():
                continue
            values = [risk[index] for index in group]
            risk_value = max(values)
            label = "likely_hallucination" if risk_value >= HALLUCINATION_LABEL_RISK else "doubtful"
            spans.append(
                SpanResult(
                    start=start,
                    end=end,
                    text=fragment,
                    risk=risk_value,
                    label=label,
                    n_tokens=len(group),
                )
            )
        return _merge_spans(spans)


def _token_metrics(labels: Sequence[int], flags: Sequence[bool], risks: Sequence[float]) -> dict[str, float]:
    """Precision/recall/F1 по токенам при заданном пороге маски."""
    tp = sum(1 for label, flag in zip(labels, flags, strict=False) if label and flag)
    fp = sum(1 for label, flag in zip(labels, flags, strict=False) if not label and flag)
    fn = sum(1 for label, flag in zip(labels, flags, strict=False) if label and not flag)
    tn = sum(1 for label, flag in zip(labels, flags, strict=False) if not label and not flag)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "auc": _auc(labels, risks),
    }


def _binary_metrics(labels: Sequence[int], flags: Sequence[bool]) -> dict[str, float]:
    """precision / recall / F1 / FPR по булевым решениям (здесь — вердиктам)."""
    tp = sum(1 for label, flag in zip(labels, flags, strict=False) if label and flag)
    fp = sum(1 for label, flag in zip(labels, flags, strict=False) if not label and flag)
    fn = sum(1 for label, flag in zip(labels, flags, strict=False) if label and not flag)
    tn = sum(1 for label, flag in zip(labels, flags, strict=False) if not label and not flag)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fp / (fp + tn), 4) if fp + tn else 0.0,
        "flagged_share": round((tp + fp) / len(labels), 4) if labels else 0.0,
    }


def _answer_metrics(labels: Sequence[int], scores: Sequence[float], threshold: float | None) -> dict[str, float]:
    """Метрики уровня ответа: AUC и качество при пороге решения."""
    cut = threshold if threshold is not None else 0.5
    tp = sum(1 for label, score in zip(labels, scores, strict=False) if label and score >= cut)
    fp = sum(1 for label, score in zip(labels, scores, strict=False) if not label and score >= cut)
    fn = sum(1 for label, score in zip(labels, scores, strict=False) if label and score < cut)
    tn = sum(1 for label, score in zip(labels, scores, strict=False) if not label and score < cut)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": cut,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "auc": _auc(labels, scores),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """AUC по ранговой формуле Манна — Уитни (NaN, если класс один)."""
    positives = sum(1 for label in labels if label)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[position]]:
            end += 1
        average = (position + end) / 2 + 1
        for step in range(position, end + 1):
            ranks[order[step]] = average
        position = end + 1
    rank_sum = sum(ranks[index] for index, label in enumerate(labels) if label)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _iou(left: tuple[int, int], right: tuple[int, int]) -> float:
    """Пересечение поверх объединения для двух символьных диапазонов."""
    inter = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    if not inter:
        return 0.0
    union = (left[1] - left[0]) + (right[1] - right[0]) - inter
    return inter / union if union else 0.0


def _span_f1(
    predicted: Sequence[Sequence[tuple[int, int]]],
    truth: Sequence[Sequence[tuple[int, int]]],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """F1 по фрагментам: жадное сопоставление с порогом IoU (как в разметке)."""
    tp = fp = fn = 0
    for predicted_pair, truth_pair in zip(predicted, truth, strict=False):
        used: set[int] = set()
        for guess in predicted_pair:
            best_index, best_iou = -1, 0.0
            for index, gold in enumerate(truth_pair):
                if index in used:
                    continue
                overlap = _iou(guess, gold)
                if overlap > best_iou:
                    best_index, best_iou = index, overlap
            if best_index >= 0 and best_iou >= iou_threshold:
                used.add(best_index)
                tp += 1
            else:
                fp += 1
        fn += len(truth_pair) - len(used)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "iou_threshold": iou_threshold,
    }


def _containment_recall(
    predicted: Sequence[Sequence[tuple[int, int]]],
    truth: Sequence[Sequence[tuple[int, int]]],
) -> float:
    """Доля размеченных фрагментов, целиком накрытых найденным фрагментом."""
    total = matched = 0
    for predicted_pair, truth_pair in zip(predicted, truth, strict=False):
        for gold in truth_pair:
            total += 1
            if any(guess[0] <= gold[0] and guess[1] >= gold[1] for guess in predicted_pair):
                matched += 1
    return matched / total if total else 0.0


def _mean_width_ratio(
    predicted: Sequence[Sequence[tuple[int, int]]],
    truth: Sequence[Sequence[tuple[int, int]]],
) -> float:
    """Во сколько раз найденные фрагменты шире разметки (в среднем, по накрытым)."""
    ratios: list[float] = []
    for predicted_pair, truth_pair in zip(predicted, truth, strict=False):
        for gold in truth_pair:
            for guess in predicted_pair:
                if guess[0] <= gold[0] and guess[1] >= gold[1]:
                    width = max(1, gold[1] - gold[0])
                    ratios.append((guess[1] - guess[0]) / width)
                    break
    return sum(ratios) / len(ratios) if ratios else 0.0


def span_threshold_for(risk: Sequence[float], span_z: float, span_floor: float, span_cap: float) -> float:
    """Порог маски: median + z·1.4826·MAD, зажатый в [floor, cap].

    Отклонение от буквального T = μ + z·σ и его причина: искомый сигнал —
    единичная подмена факта в остальном подтверждённом предложении. Сама такая
    подмена увеличивает σ и «прячет» себя за порогом. Медиана и MAD к выбросам
    устойчивы, поэтому порог остаётся на уровне разброса подтверждённых токенов.
    Коэффициент 1.4826 приводит MAD к масштабу σ нормального распределения, так
    что z имеет привычный смысл. Формула одна на движок и обучение — иначе
    метрики обучения не описывали бы то, что отдаёт API.
    """
    if not risk:
        return 1.0
    center = _median(risk)
    deviation = _median([abs(value - center) for value in risk])
    adaptive = center + span_z * 1.4826 * deviation
    return max(0.0, min(min(0.999, float(span_cap)), max(adaptive, float(span_floor))))


def _smooth(values: Sequence[float], window: int) -> list[float]:
    """Сглаживание по окну ``window`` с весом центра 0.6.

    Обычное среднее по окну 3 «размывает» узкие подмены: один высокий токен
    между двумя спокойными даёт всего треть своего значения и уходит под порог.
    Поэтому центр окна весит 0.6, а соседи — по 0.2: шум соседей подавляется,
    но одиночный пик сохраняется. На границах веса перенормируются.
    """
    if len(values) < 3 or window <= 1:
        return [float(value) for value in values]
    half = max(1, window // 2)
    centre_weight = 0.6
    side_weight = (1.0 - centre_weight) / (2 * half)
    out: list[float] = []
    for index in range(len(values)):
        low, high = max(0, index - half), min(len(values), index + half + 1)
        total_weight = centre_weight + side_weight * (index - low) + side_weight * (high - 1 - index)
        weighted = centre_weight * values[index] + sum(
            side_weight * values[position] for position in range(low, high) if position != index
        )
        out.append(weighted / total_weight)
    return out


def _answer_score(values: Sequence[float]) -> float:
    """Оценка ответа: 0.5·среднее + 0.5·среднее верхних 20 %."""
    if not values:
        return 0.0
    ordered = sorted(values, reverse=True)
    top_count = max(1, int(len(ordered) * TOP_SHARE))
    return 0.5 * mean(values) + 0.5 * mean(ordered[:top_count])


def _head_risk(
    head: dict[str, Any],
    features: FeatureMatrix,
    risk: Sequence[float],
    tokens: Sequence[Token],
) -> list[float] | None:
    """Применить обученную голову (логистическую регрессию) к признакам токенов.

    Возвращает вероятности недостоверности по токенам или ``None``, если
    артефакт головы повреждён — тогда используется пороговое правило.
    """
    import json as _json
    import math as _math

    from .config import read_runtime_text

    model_payload = head
    if head.get("file") and not head.get("model"):
        raw = read_runtime_text(str(head["file"]))
        if not raw:
            return None
        try:
            model_payload = _json.loads(raw)
        except ValueError:
            return None

    model = model_payload.get("model") or {}
    weights = model.get("weights")
    if not weights:
        return None
    means = model_payload.get("scaler", {}).get("means", [])
    scales = model_payload.get("scaler", {}).get("scales", [])
    if len(means) != len(weights) or len(scales) != len(weights):
        return None

    probabilities: list[float] = []
    total = max(1, len(tokens))
    for index in range(len(features)):
        row = [
            features.attention_entropy[index],
            features.ctx_attention_mass[index],
            features.embedding_density[index],
            risk[index],
            index / total,
            min(1.0, len(tokens[index].text.strip()) / 20),
        ]
        normalized = [(value - means[j]) / (scales[j] or 1.0) for j, value in enumerate(row)]
        score = model.get("bias", 0.0) + sum(w * x for w, x in zip(weights, normalized, strict=False))
        probabilities.append(1.0 / (1.0 + _math.exp(-max(-30.0, min(30.0, score)))))
    return probabilities


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _is_content_token(token: Token) -> bool:
    """Содержательный токен: несёт смысл (слово длины ≥ 3 или число)."""
    text = token.word.lower()
    if not text:
        return False
    if text.isdigit():
        return True
    return len(text) >= 3 and not text.isascii() or len(text) >= 4


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(percentile / 100 * len(ordered))) - 1))
    return ordered[index]


def _expand_to_sentence(answer: str, start: int, end: int) -> tuple[int, int]:
    """Расширить диапазон токенов до границ предложения."""
    for sentence_start, sentence_end in split_sentences(answer):
        if sentence_start <= start < sentence_end:
            return sentence_start, max(sentence_end, end)
    return start, end


def _merge_spans(spans: Sequence[SpanResult]) -> list[SpanResult]:
    """Склеить пересекающиеся фрагменты и перенумеровать."""
    ordered = sorted(spans, key=lambda span: span.start)
    merged: list[SpanResult] = []
    for span in ordered:
        if merged and span.start <= merged[-1].end:
            previous = merged[-1]
            merged[-1] = SpanResult(
                start=previous.start,
                end=max(previous.end, span.end),
                text=previous.text,
                risk=max(previous.risk, span.risk),
                label=(
                    "likely_hallucination" if "likely_hallucination" in (previous.label, span.label) else "doubtful"
                ),
                n_tokens=previous.n_tokens + span.n_tokens,
            )
        else:
            merged.append(span)
    return merged


def _token_payload(
    tokens: Sequence[Token],
    features: FeatureMatrix,
    risk: Sequence[float],
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, token in enumerate(tokens):
        if index >= len(risk):
            break
        rows.append(
            {
                "index": index,
                "text": token.text.strip(),
                "start": token.start,
                "end": token.end,
                "attention_entropy": round(features.attention_entropy[index], 4),
                "ctx_attention_mass": round(features.ctx_attention_mass[index], 4),
                "embedding_density": round(features.embedding_density[index], 4),
                "risk": round(risk[index], 4),
                "flagged": risk[index] >= threshold,
                "label": (
                    "likely_hallucination"
                    if risk[index] >= HALLUCINATION_LABEL_RISK
                    else ("doubtful" if risk[index] >= threshold else "ok")
                ),
            }
        )
    return rows


def _verdict(score: float, threshold: float, spans: Sequence[SpanResult]) -> str:
    """Вердикт: недостоверно → спорно → подтверждено контекстом."""
    if score >= threshold:
        return "likely_hallucination"
    if any(span.label == "likely_hallucination" for span in spans):
        return "doubtful"
    if spans:
        return "doubtful"
    return "grounded"


def verify_text(
    answer: str,
    context: str | Sequence[str] | None = None,
    mode: str = "demo",
    weights_path: str | Path | None = WEIGHTS_FILENAME,
    with_tokens: bool = False,
) -> VerificationResult:
    """Удобная функция для быстрой проверки без создания объекта."""
    verifier = Verifier(mode=mode, weights_path=weights_path)
    return verifier.verify(answer, context, with_tokens=with_tokens)


def context_chunks(context: str | Sequence[str] | None) -> ContextChunks:
    """Публичный доступ к разбору контекста (для интерфейсов и тестов)."""
    return split_chunks(context)
