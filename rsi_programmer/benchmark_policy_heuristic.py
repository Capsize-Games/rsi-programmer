"""Fixed operand-similarity heuristic for frozen search comparisons."""
from __future__ import annotations

from typing import Any

from rsi_programmer.benchmark_policy import (
    Group,
    rank_scores,
    state_fingerprint,
    validate_features,
)


class ClassicalHeuristicPolicy:
    """Rank candidates by operand agreement and simplicity.

    The formula is 2 * (left_hit + right_hit) + 0.5 *
    (left_inside + right_inside) - 0.05 * (left_size + right_size).
    It uses feature indices 3 through 9: sizes and operand overlap,
    with no operation or layer-ordinal features.
    """

    def __init__(self) -> None:
        """Create a fixed, stateless score rule."""
        self._steps = 1

    def scores(self, feats: list[list[float]]) -> list[float]:
        """Compute the documented operand-only heuristic score."""
        validate_features(feats)
        return [
            2.0 * (row[6] + row[8])
            + 0.5 * (row[7] + row[9])
            - 0.05 * (row[3] + row[4])
            for row in feats
        ]

    def rank(self, feats: list[list[float]], triples: Any = None,
             gap: float | None = None) -> list[int]:
        """Return stable descending heuristic score order."""
        del triples, gap
        import torch

        return rank_scores(torch.tensor(self.scores(feats)))

    def fit(self, groups: list[Group], epochs: int) -> float:
        """Keep this non-trainable baseline fixed during benchmark setup."""
        del groups, epochs
        return 0.0

    def weights(self) -> dict[str, float]:
        """Return the fixed score coefficients as the baseline state."""
        return {"left_hit": 2.0, "right_hit": 2.0,
                "left_inside": 0.5, "right_inside": 0.5,
                "left_size": -0.05, "right_size": -0.05}

    def metadata(self) -> dict[str, Any]:
        """Describe the fixed rule and its lack of simulation state."""
        return {
            "name": "classical_operand_heuristic",
            "architecture": "fixed_linear_rule",
            "trainable_parameters": 0,
            "simulation_steps": 1,
            "state_semantics": "stateless_per_candidate",
            "fit_mode": "fixed_noop",
            "formula": (
                "2*(left_hit+right_hit) + 0.5*(left_inside+right_inside)"
                " - 0.05*(left_size+right_size)"
            ),
            "feature_encoding": "layer-ordinal-v1",
            "feature_dimension": 12,
        }

    def fingerprint(self) -> str:
        """Hash the immutable rule and its declared feature contract."""
        return state_fingerprint(self.weights(), self.metadata())
