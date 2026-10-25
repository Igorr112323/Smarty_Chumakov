"""P4: deterministic scored-neighbour derivatives of the frozen eight inputs.

The five contractual HF columns remain unchanged. These are derived inputs,
not additional HF measurements. Runtime has no numpy/sklearn dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .features import FEATURE_NAMES
from .token_head import DecisionTree, TokenGBDTModel

BASE_NAMES = (*FEATURE_NAMES, "position", "length", "has_digit")
OPS = ("previous", "next", "left_max", "right_max", "neighbour_mean", "self_minus_mean")


def feature_spec(k):
    return validate_spec(
        {
            "schema": "context_features_schema_v1",
            "k": k,
            "edge": "clamp",
            "sequence": "scored_within_pair",
            "unscored": "self_only_then_zero_mask",
            "base_order": list(BASE_NAMES),
            "feature_order": [
                *BASE_NAMES,
                *[f"{f}:{op}" for f in BASE_NAMES for op in OPS],
            ],
            "n_features": 56,
        }
    )


def validate_spec(spec):
    if not isinstance(spec, dict) or type(spec.get("k")) is not int or spec["k"] not in (1, 2, 3):
        raise ValueError("Invalid context feature window")
    expected = {
        "schema": "context_features_schema_v1",
        "k": spec["k"],
        "edge": "clamp",
        "sequence": "scored_within_pair",
        "unscored": "self_only_then_zero_mask",
        "base_order": list(BASE_NAMES),
        "feature_order": [
            *BASE_NAMES,
            *[f"{f}:{op}" for f in BASE_NAMES for op in OPS],
        ],
        "n_features": 56,
    }
    if spec != expected:
        raise ValueError("Malformed context_features_schema_v1")
    return spec


def build_context_features(base, scored_mask, spec):
    """Accept exactly ONE full answer. Never concatenate pairs before calling."""
    validate_spec(spec)
    if len(base) != len(scored_mask) or any(type(m) is not bool for m in scored_mask):
        raise ValueError("Invalid scored mask")
    if any(len(row) != 8 or any(not math.isfinite(x) for x in row) for row in base):
        raise ValueError("Expected finite eight-column base matrix")
    indices = [i for i, scored in enumerate(scored_mask) if scored]
    positions = {i: j for j, i in enumerate(indices)}
    result = []
    for i, row in enumerate(base):
        output = list(row)
        for f in range(8):
            if i not in positions:
                left = right = [row[f]] * spec["k"]
            else:
                j = positions[i]
                left = [base[indices[max(0, j - d)]][f] for d in range(1, spec["k"] + 1)]
                right = [base[indices[min(len(indices) - 1, j + d)]][f] for d in range(1, spec["k"] + 1)]
            mean = sum(left + right) / (2 * spec["k"])
            output.extend([left[0], right[0], max(left), max(right), mean, row[f] - mean])
        result.append(output)
    return result


def _tree(data, n_features):
    if "value" in data:
        return DecisionTree.from_dict(data)
    if not all(k in data for k in ("feature", "threshold", "left", "right")):
        raise ValueError("Missing context tree children/fields")
    if (
        type(data["feature"]) is not int
        or not 0 <= data["feature"] < n_features
        or not math.isfinite(data["threshold"])
    ):
        raise ValueError("Invalid context tree feature/threshold")
    return DecisionTree(
        threshold=float(data["threshold"]),
        feature_index=data["feature"],
        left=_tree(data["left"], n_features),
        right=_tree(data["right"], n_features),
    )


@dataclass
class ContextGBDTModel(TokenGBDTModel):
    context_features: dict | None = None

    def to_dict(self):
        validate_spec(self.context_features)
        data = super().to_dict()
        data.update(
            type="context_token_gbdt",
            schema_version="context_token_gbdt_v1",
            context_features=self.context_features,
        )
        return data

    @classmethod
    def from_dict(cls, data):
        if (
            data.get("type") != "context_token_gbdt"
            or data.get("schema_version") != "context_token_gbdt_v1"
            or data.get("n_features") != 56
        ):
            raise ValueError("Unknown context token model schema/width")
        spec = validate_spec(data.get("context_features"))
        rate, initial = float(data["learning_rate"]), float(data["initial_prediction"])
        if not math.isfinite(rate) or rate <= 0 or initial != 0.0 or not data.get("trees"):
            raise ValueError("Invalid context model parameters")
        return cls(
            trees=[_tree(t, 56) for t in data["trees"]],
            learning_rate=rate,
            initial_prediction=initial,
            n_features=56,
            schema_version="context_token_gbdt_v1",
            context_features=spec,
        )


def export_context_gbdt(gbdt, spec):
    """Training-only adapter; export the SAME sklearn object for parity."""
    validate_spec(spec)
    if gbdt.init != "zero" or gbdt.n_features_in_ != 56:
        raise ValueError("Expected zero-init 56-input GBDT")

    def convert(tree, node=0):
        if tree.children_left[node] == tree.children_right[node]:
            return {"value": float(tree.value[node, 0, 0])}
        return {
            "feature": int(tree.feature[node]),
            "threshold": float(tree.threshold[node]),
            "left": convert(tree, int(tree.children_left[node])),
            "right": convert(tree, int(tree.children_right[node])),
        }

    model = ContextGBDTModel(
        trees=[_tree(convert(t.tree_), 56) for t in gbdt.estimators_[:, 0]],
        learning_rate=gbdt.learning_rate,
        initial_prediction=0.0,
        n_features=56,
        schema_version="context_token_gbdt_v1",
        context_features=spec,
    )
    return ContextGBDTModel.from_dict(model.to_dict())
