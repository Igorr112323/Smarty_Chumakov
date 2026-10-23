"""R2 — изотонный агрегатор доли по фрагмент-статистике (дорожка Д, H1).

Цель: заменить смещённое произведение p_ai (product) на монотонное
изотонное картирование по mean-of-fragment-p_ai.

Обучение строго на dev-корпусе, seed-free, детерминировано, монотонность
гарантирована формой (PAVA), а не везением. Веса/модель/пороги заморожены,
меняется только агрегация доли.

Использование:
    from spanverify.r2_share import IsotonicFragmentAggregator
    agg = IsotonicFragmentAggregator.load("reports/science/r2/r2_calibrator.json")
    share = agg.estimate(text, spans, token_scores)  # 0..1, monotone

Тренировка (только dev):
    python -m spanverify.r2_share --mode demo --out reports/science/r2/

Формально: raw = mean_{fragment} mean_{token in fragment} p_ai
           calibrated = PAVA(raw -> true_fraction)
"""

from __future__ import annotations

import json
import math
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True)
class IsotonicFragmentAggregator:
    """Монотонный калибратор доли: PAVA по fragment-агрегату."""

    upper: tuple[float, ...]
    values: tuple[float, ...]
    mode: str
    bundle_sha256: str
    source: str = "dev181-fragment-mean"
    schema: ClassVar[str] = "r2_fragment_aggregator_v1"

    @classmethod
    def fit(cls, scores, targets, *, mode, bundle_sha256):
        """PAVA, детерминированно, без seed."""
        rows = sorted(zip(scores, targets, strict=True))
        if not rows or any(
            not math.isfinite(x) or not math.isfinite(y) or not 0 <= x <= 1 or not 0 <= y <= 1
            for x, y in rows
        ):
            raise ValueError("calibration requires finite unit-interval observations")
        # групповая агрегация по одинаковым x
        groups = []
        for x, y in rows:
            if groups and groups[-1][0] == x:
                groups[-1][1] += y
                groups[-1][2] += 1
            else:
                groups.append([x, y, 1])
        blocks = []
        for end, total, weight in groups:
            blocks.append([end, total, weight])
            while len(blocks) > 1 and blocks[-2][1] / blocks[-2][2] > blocks[-1][1] / blocks[-1][2]:
                b, a = blocks.pop(), blocks.pop()
                blocks.append([b[0], a[1] + b[1], a[2] + b[2]])
        return cls(
            tuple(b[0] for b in blocks),
            tuple(b[1] / b[2] for b in blocks),
            mode,
            bundle_sha256,
        )

    def transform(self, score: float) -> float:
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("invalid raw fragment share")
        idx = bisect_left(self.upper, score)
        if idx >= len(self.upper):
            idx = len(self.upper) - 1
        return self.values[idx]

    def estimate(self, text: str, spans, token_scores: list[float], tokens) -> float:
        """Оценка доли: mean-of-fragment-p_ai -> isotonic."""
        if not spans:
            # нет фрагментов — считаем среднее по всем scored токенам
            if not token_scores:
                return 0.0
            raw = sum(token_scores) / len(token_scores)
        else:
            # mean по фрагментам: для каждого спана среднее p_ai внутри
            frag_means = []
            for sp in spans:
                # sp: SpanResult with start/end
                inside = [
                    token_scores[i]
                    for i, tok in enumerate(tokens)
                    if sp.start <= tok.start < sp.end
                ]
                if inside:
                    frag_means.append(sum(inside) / len(inside))
            if not frag_means:
                raw = sum(token_scores) / len(token_scores) if token_scores else 0.0
            else:
                raw = sum(frag_means) / len(frag_means)
        raw = max(0.0, min(1.0, raw))
        return max(0.0, min(1.0, self.transform(raw)))

    def to_dict(self):
        return {
            self.schema: dict(
                upper=list(self.upper),
                values=list(self.values),
                mode=self.mode,
                bundle_sha256=self.bundle_sha256,
                source=self.source,
            )
        }

    @classmethod
    def from_dict(cls, value):
        if set(value) != {cls.schema} or not isinstance(value[cls.schema], dict):
            raise ValueError("unknown r2 schema")
        d = value[cls.schema]
        if set(d) != {"upper", "values", "mode", "bundle_sha256", "source"}:
            # backward compat: allow missing source
            if set(d) == {"upper", "values", "mode", "bundle_sha256"}:
                d["source"] = "dev181-fragment-mean"
            else:
                raise ValueError("unknown calibration fields")
        upper = tuple(map(float, d["upper"]))
        values = tuple(map(float, d["values"]))
        if (
            not upper
            or len(upper) != len(values)
            or any(not math.isfinite(x) or not 0 <= x <= 1 for x in upper + values)
        ):
            raise ValueError("invalid calibration numbers")
        if any(a >= b for a, b in zip(upper, upper[1:], strict=False)) or any(
            a > b for a, b in zip(values, values[1:], strict=False)
        ):
            raise ValueError("calibration must be strictly monotone")
        if d["mode"] not in ("demo", "hf") or len(d["bundle_sha256"]) != 64:
            raise ValueError("invalid calibrator provenance")
        return cls(upper, values, d["mode"], d["bundle_sha256"], d["source"])

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)
