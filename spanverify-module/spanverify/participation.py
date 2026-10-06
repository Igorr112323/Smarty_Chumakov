"""Оценка доли участия ИИ в тексте — поле ``ai_participation``.

Заявка (У-640148) и ТЗ формулируют результат как «определение доли участия ИИ в
контенте». Это **отдельная** величина от ``ai_share``:

* ``ai_share`` — доля недостоверного (не подтверждённого документом) текста,
  то есть «сколько ответа вызывает вопросы к достоверности»;
* ``ai_participation`` — оценка того, какая часть ответа выглядит
  машинно-порождённой, независимо от того, подтверждена она документом или нет.

Как считается
-------------
Логистическая регрессия по признакам токена:

* ``attention_entropy``, ``ctx_attention_mass``, ``embedding_density`` — из того
  же конвейера, что и основная проверка (в demo-режиме это лексические
  суррогаты, в hf — реальные сигналы модели);
* ``cliche_rate`` — доля «машинных» оборотов в предложении токена;
* ``len_norm`` — нормированная длина предложения: машинный текст ровнее.

Доля = среднее ``P(машинный токен)`` по содержательным токенам ответа. Величина
монотонна по построению: чем больше в тексте машинных оборотов, тем выше оценка.

Про честность калибровки
------------------------
Корпус для обучения — **синтетический**: два пула предложений (шаблонные
«канцелярские» клише против разговорных фраз), смешиваемые с известной долей
``ai_fraction``. До появления размеченного набора «человек / ИИ / смешанный» на
реальных документах оценка калибрована на синтетике; это видно в поле
``calibrated_on``, в ответе ``/v1/model`` и в предупреждении demo-режима.
Менять заявку нельзя — поэтому код отдаёт именно ту величину, которая в ней
названа, но не выдаёт синтетическую калибровку за измерение на реальных данных.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .core import Token, tokenize_with_offsets
from .features import FeatureMatrix, is_scored_token
from .logreg import probabilities, standardize, train_logreg
from .text import sentences

__all__ = [
    "AI_CLICHES",
    "PARTICIPATION_FILENAME",
    "ParticipationModel",
    "build_participation_corpus",
    "corpus_rows_and_labels",
]

PARTICIPATION_FILENAME = "config/participation.json"
CALIBRATED_ON = "synthetic-participation-corpus 1.0 (два стиля, доля смеси известна)"

FEATURES = (
    "attention_entropy",
    "ctx_attention_mass",
    "embedding_density",
    "cliche_rate",
    "len_norm",
)

AI_CLICHES = (
    "данный",
    "осуществляется",
    "необходимо отметить",
    "комплексное решение",
    "эффективное решение",
    "в рамках",
    "в целях",
    "обеспечивает",
    "позволяет достичь",
    "высоких показателей",
    "представляет собой",
    "является",
    "следует отметить",
    "в соответствии с",
)

AI_SENTENCES = (
    "Необходимо отметить, что данный регламент обеспечивает комплексное решение задачи.",
    "Осуществляется контроль сроков хранения в рамках установленных требований.",
    "В целях повышения эффективности является целесообразным проводить инвентаризацию ежегодно.",
    "Данный подход позволяет достичь высоких показателей качества обработки документов.",
    "Следует отметить, что процедура представляет собой последовательность этапов.",
    "В соответствии с регламентом обеспечивается хранение документов двадцать пять лет.",
    "Осуществляется формирование отчётности в рамках установленного регламента 669.",
    "Необходимо отметить, что срок хранения составляет пятнадцать лет.",
    "Данная процедура является обязательной для всех подразделений организации.",
    "В целях обеспечения сохранности осуществляется архивное хранение документов.",
    "Следует отметить, что инвентаризация проводится два раза в год.",
    "Комплексное решение обеспечивает эффективное сопровождение процессов.",
)

HUMAN_SENTENCES = (
    "Папки сдаём в архив до конца марта, раньше не получается.",
    "Храним договоры пять лет, потом можно списывать.",
    "Инвентаризацию делаем раз в год, обычно в феврале.",
    "Акты подшиваем в отдельную папку, чтобы не потерялись.",
    "Срок хранения у нас пятнадцать лет, так в регламенте написано.",
    "Отчёт готовим к десятому числу, потом отправляем в бухгалтерию.",
    "Старые накладные лежат в подвале, туда никто не ходит.",
    "Проверяем наличие документов каждый квартал, иногда чаще.",
    "Договоры подряда храним двадцать пять лет, как положено.",
    "Журнал регистрации ведёт секретарь, она же следит за сроками.",
    "Если документ утерян, составляем акт и докладываем руководству.",
    "Приказы подшиваем отдельно от служебных записок.",
)

DOC_CONTEXTS = (
    "Согласно регламенту 669, срок хранения договоров подряда установлен в размере 25 лет, "
    "инвентаризация проводится один раз в год, отчётность сдаётся до 10 числа.",
    "Регламент 343: срок хранения первичных документов составляет 10 лет, "
    "инвентаризация — ежегодно до 1 марта, акты подшиваются в отдельную папку.",
    "Положение об архиве: документы хранятся 15 лет, проверка наличия — раз в квартал, "
    "журнал регистрации ведёт секретарь.",
)


def build_participation_corpus(count: int = 240, seed: int = 2027) -> list[dict[str, Any]]:
    """Собрать синтетический корпус «ИИ + человек» с известной долей смеси.

    Каждый элемент: ``{"text", "context", "spans", "ai_fraction"}``, где
    ``spans`` — интервалы символов машинных предложений (метка 1) и
    человеческих (метка 0).
    """
    rng = random.Random(seed)
    samples: list[dict[str, Any]] = []
    for _ in range(count):
        fraction = rng.choice([0.0, 0.25, 0.5, 0.75, 1.0])
        total = rng.randint(4, 6)
        ai_count = int(round(fraction * total))
        parts: list[tuple[int, str]] = [(1, rng.choice(AI_SENTENCES)) for _ in range(ai_count)]
        parts += [(0, rng.choice(HUMAN_SENTENCES)) for _ in range(total - ai_count)]
        rng.shuffle(parts)
        text_parts: list[str] = []
        spans: list[tuple[int, int, int]] = []
        offset = 0
        for label, sentence in parts:
            spans.append((offset, offset + len(sentence), label))
            text_parts.append(sentence)
            offset += len(sentence) + 1
        samples.append(
            {
                "text": " ".join(text_parts),
                "context": rng.choice(DOC_CONTEXTS),
                "spans": spans,
                "ai_fraction": fraction,
            }
        )
    return samples


def _sentence_bounds(text: str) -> list[tuple[int, int]]:
    """Границы предложений (через общий модуль текста, без дублирования логики)."""
    return sentences(text) or [(0, len(text))]


def _sentence_of(index: int, bounds: Sequence[tuple[int, int]]) -> tuple[int, int]:
    for start, end in bounds:
        if start <= index < end:
            return start, end
    return bounds[0]


def _cliche_rate(fragment: str) -> float:
    """Доля «машинных» оборотов в предложении, приведённая к [0..1]."""
    lowered = fragment.lower()
    hits = sum(1 for cliche in AI_CLICHES if cliche in lowered)
    return min(1.0, hits / 2.0)


def _len_norm(fragment: str) -> float:
    """Нормированная длина предложения: машинные фразы обычно длиннее."""
    return min(1.0, len(fragment) / 120.0)


def token_rows(text: str, matrix: FeatureMatrix) -> tuple[list[list[float]], list[Token]]:
    """Строки признаков по содержательным токенам текста.

    Возвращает ``(строки, токены)`` в одном порядке; пунктуация и служебные
    односложные слова пропускаются так же, как в основном конвейере.
    """
    tokens = tokenize_with_offsets(text)
    bounds = _sentence_bounds(text)
    rows: list[list[float]] = []
    kept: list[Token] = []
    for index, token in enumerate(tokens):
        if not is_scored_token(token.text):
            continue
        if index >= len(matrix):
            break
        start, end = _sentence_of(token.start, bounds)
        fragment = text[start:end]
        rows.append(
            [
                matrix.attention_entropy[index],
                matrix.ctx_attention_mass[index],
                matrix.embedding_density[index],
                _cliche_rate(fragment),
                _len_norm(fragment),
            ]
        )
        kept.append(token)
    return rows, kept


def corpus_rows_and_labels(verifier: Any, samples: Sequence[dict[str, Any]]) -> tuple[list[list[float]], list[int]]:
    """Признаки и метки токенов для корпуса участия ИИ.

    Метка токена — 1, если он попал в машинное предложение (по разметке корпуса),
    иначе 0. Признаки считает тот же ``Verifier``, что и в рабочем режиме: если
    режим ``hf``, признаки будут модельными, если ``demo`` — суррогатными.
    """
    rows: list[list[float]] = []
    labels: list[int] = []
    for sample in samples:
        matrix = verifier.features_for(sample["text"], sample["context"])
        sample_rows, tokens = token_rows(sample["text"], matrix)
        for row, token in zip(sample_rows, tokens, strict=False):
            label = 0
            for start, end, span_label in sample["spans"]:
                if start <= token.start < end:
                    label = int(span_label)
                    break
            rows.append(row)
            labels.append(label)
    return rows, labels


@dataclass
class ParticipationModel:
    """Обученная оценка доли участия ИИ: масштабы, веса и калибровка."""

    model: dict[str, Any] = field(default_factory=dict)
    means: list[float] = field(default_factory=list)
    scales: list[float] = field(default_factory=list)
    auc_out_of_fold: float = 0.0
    calibrated_on: str = CALIBRATED_ON
    seed: int = 42
    version: str = ""
    features: list[str] = field(default_factory=lambda: list(FEATURES))

    @classmethod
    def fit(
        cls,
        rows: Sequence[Sequence[float]],
        labels: Sequence[int],
        seed: int = 42,
        folds: int = 5,
        version: str = "",
    ) -> ParticipationModel:
        """Обучить модель и посчитать AUC кросс-валидацией (без внешних данных)."""
        normalized, means, scales = standardize(rows)
        model = train_logreg(normalized, labels)
        auc = _out_of_fold_auc(rows, labels, folds=folds, seed=seed)
        return cls(
            model=model,
            means=means,
            scales=scales,
            auc_out_of_fold=round(auc, 4),
            seed=seed,
            version=version,
        )

    def score_rows(self, rows: Sequence[Sequence[float]]) -> list[float]:
        """Вероятность «машинного» стиля для каждой строки признаков."""
        if not rows:
            return []
        scaled = [
            [(row[j] - self.means[j]) / (self.scales[j] or 1.0) for j in range(min(len(row), len(self.means)))]
            for row in rows
        ]
        return probabilities(self.model, scaled)

    def estimate(self, text: str, matrix: FeatureMatrix) -> float:
        """Оценка доли участия ИИ в тексте: среднее по содержательным токенам."""
        rows, _ = token_rows(text, matrix)
        scores = self.score_rows(rows)
        if not scores:
            return 0.0
        return max(0.0, min(1.0, sum(scores) / len(scores)))

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "type": "participation-logreg",
            "features": list(self.features),
            "scaler": {"means": self.means, "scales": self.scales},
            "model": self.model,
            "auc_out_of_fold": self.auc_out_of_fold,
            "calibrated_on": self.calibrated_on,
            "seed": self.seed,
        }
        if self.version:
            payload["version"] = self.version
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParticipationModel:
        scaler = data.get("scaler") or {}
        return cls(
            model=dict(data.get("model") or {}),
            means=list(scaler.get("means") or []),
            scales=list(scaler.get("scales") or []),
            auc_out_of_fold=float(data.get("auc_out_of_fold") or 0.0),
            calibrated_on=str(data.get("calibrated_on") or CALIBRATED_ON),
            seed=int(data.get("seed") or 42),
            version=str(data.get("version") or ""),
            features=list(data.get("features") or FEATURES),
        )

    def save(self, path: str | Path) -> Path:
        """Сохранить модель в JSON (ключ ``config/participation.json``)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> ParticipationModel:
        """Прочитать модель из JSON."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _out_of_fold_auc(rows: Sequence[Sequence[float]], labels: Sequence[int], folds: int = 5, seed: int = 42) -> float:
    """Честная AUC: кросс-валидация по строкам, без обучения на проверяемых."""
    if not rows or len(set(labels)) < 2:
        return 0.0
    normalized, _, _ = standardize(rows)
    rng = random.Random(seed)
    order = list(range(len(normalized)))
    rng.shuffle(order)
    fold_count = max(2, min(folds, len(order)))
    fold_size = max(1, len(order) // fold_count)
    scores = [0.0] * len(order)
    for fold in range(fold_count):
        start = fold * fold_size
        stop = len(order) if fold == fold_count - 1 else start + fold_size
        test_idx = order[start:stop]
        test_set = set(test_idx)
        train_idx = [index for index in order if index not in test_set]
        if not train_idx or not test_idx:
            continue
        model = train_logreg([normalized[i] for i in train_idx], [labels[i] for i in train_idx])
        for index, probability in zip(test_idx, probabilities(model, [normalized[i] for i in test_idx]), strict=False):
            scores[index] = probability
    return _auc(labels, scores)


def _auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """AUC по определению (ранги), без внешних зависимостей."""
    pairs = sorted(zip(scores, labels, strict=False))
    positives = sum(1 for label in labels if label == 1)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.0
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        stop = index
        while stop + 1 < len(pairs) and pairs[stop + 1][0] == pairs[index][0]:
            stop += 1
        average_rank = (index + stop) / 2 + 1
        for position in range(index, stop + 1):
            if pairs[position][1] == 1:
                rank_sum += average_rank
        index = stop + 1
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
