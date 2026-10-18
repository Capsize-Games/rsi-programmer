"""Seeded stateless random-order control for frozen benchmarks."""
from __future__ import annotations

import hashlib
from typing import Any

from rsi_programmer.benchmark_policy import (
    Group,
    state_fingerprint,
    validate_features,
)


class SeededRandomPolicy:
    """Rank each candidate with a reproducible seeded hash score."""

    def __init__(self, seed: int = 0) -> None:
        """Store a local seed without touching Python or torch RNG state."""
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")
        self._seed = seed
        self._steps = 1

    def scores(self, feats: list[list[float]]) -> list[int]:
        """Return deterministic pseudo-random keys for candidate rows."""
        validate_features(feats)
        keys: list[int] = []
        for index, row in enumerate(feats):
            payload = f"{self._seed}:{index}:" + ",".join(
                float(value).hex() for value in row
            )
            digest = hashlib.blake2b(payload.encode("ascii"),
                                     digest_size=8).digest()
            keys.append(int.from_bytes(digest, "big"))
        return keys

    def rank(self, feats: list[list[float]], triples: Any = None,
             gap: float | None = None) -> list[int]:
        """Return descending seeded order, stable on hash collisions."""
        del triples, gap
        scores = self.scores(feats)
        return sorted(range(len(scores)), key=lambda i: scores[i],
                      reverse=True)

    def fit(self, groups: list[Group], epochs: int) -> float:
        """Keep this non-trainable baseline fixed during setup."""
        del groups, epochs
        return 0.0

    def weights(self) -> dict[str, int]:
        """Return the only configuration value affecting random order."""
        return {"seed": self._seed}

    def metadata(self) -> dict[str, Any]:
        """Describe the seeded, stateless control."""
        return {
            "name": "seeded_random_order",
            "architecture": "blake2b_seeded_candidate_order",
            "trainable_parameters": 0,
            "simulation_steps": 1,
            "state_semantics": "stateless_per_candidate_and_index",
            "fit_mode": "fixed_noop",
            "seed": self._seed,
            "feature_encoding": "layer-ordinal-v1",
            "feature_dimension": 12,
        }

    def fingerprint(self) -> str:
        """Hash the seed and declared behavior."""
        return state_fingerprint(self.weights(), self.metadata())
