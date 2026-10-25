"""Train-only monotone calibration of frozen ai_share, not a quality model."""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class AiShareCalibrator:
    upper: tuple[float, ...]
    values: tuple[float, ...]
    mode: str
    bundle_sha256: str
    schema: ClassVar[str] = "ai_share_calibrator_v1"

    @classmethod
    def fit(cls, scores, targets, *, mode, bundle_sha256):
        rows = sorted(zip(scores, targets, strict=True))
        if not rows or any(
            not math.isfinite(x) or not math.isfinite(y) or not 0 <= x <= 1 or not 0 <= y <= 1 for x, y in rows
        ):
            raise ValueError("calibration requires finite unit-interval train observations")
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

    def transform(self, score):
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("invalid raw ai_share")
        return self.values[min(bisect_left(self.upper, score), len(self.upper) - 1)]

    def to_dict(self):
        return {
            self.schema: dict(
                upper=list(self.upper),
                values=list(self.values),
                mode=self.mode,
                bundle_sha256=self.bundle_sha256,
            )
        }

    @classmethod
    def from_dict(cls, value):
        if set(value) != {cls.schema} or not isinstance(value[cls.schema], dict):
            raise ValueError("unknown ai-share calibration schema")
        d = value[cls.schema]
        if set(d) != {"upper", "values", "mode", "bundle_sha256"}:
            raise ValueError("unknown calibration fields")
        upper, values = tuple(map(float, d["upper"])), tuple(map(float, d["values"]))
        if (
            not upper
            or len(upper) != len(values)
            or any(not math.isfinite(x) or not 0 <= x <= 1 for x in upper + values)
        ):
            raise ValueError("invalid calibration numbers")
        if any(a >= b for a, b in zip(upper, upper[1:], strict=False)) or any(
            a > b for a, b in zip(values, values[1:], strict=False)
        ):
            raise ValueError("calibration must be monotone")
        if d["mode"] not in ("demo", "hf") or len(d["bundle_sha256"]) != 64:
            raise ValueError("invalid calibrator provenance")
        return cls(upper, values, d["mode"], d["bundle_sha256"])
