"""Cached feature statistics for policy-ranked pre-evaluation layers."""
from __future__ import annotations

from rsi_programmer.policy import DIM, OP_ORDER, Triple, Vals


class FeatureStatistics:
    """Reuse operand hit/range counts across features and layer gap."""

    def __init__(self, target: Vals) -> None:
        if not target:
            raise ValueError("non-empty target required")
        self.target = target
        self.low = min(target)
        self.high = max(target)
        self.target_range_elements_scanned = 2 * len(target)
        self.operand_feature_elements_scanned = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_lookups = 0
        self._stats: dict[int, tuple[int, int]] = {}

    def operand(self, key: int, signature: Vals) -> tuple[int, int]:
        """Return target-hit and range counts for one retained operand."""
        self.cache_lookups += 1
        cached = self._stats.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        hit = sum(1 for a, b in zip(signature, self.target)
                  if a == b)
        inside = sum(1 for value in signature
                     if self.low <= value <= self.high)
        self.operand_feature_elements_scanned += (
            min(len(signature), len(self.target)) + len(signature))
        self.cache_misses += 1
        result = hit, inside
        self._stats[key] = result
        return result

    def gap(self, previous: list[tuple[int, Vals]]) -> float:
        """Return the existing best-hit gap, reusing operand hit counts."""
        best = max((self.operand(key, signature)[0]
                    for key, signature in previous), default=0)
        return 1.0 - best / (len(self.target) or 1)

    def encode(self, op: str, left_key: int, left_sig: Vals,
               right_key: int, right_sig: Vals,
               left_size: int, right_size: int, size: int,
               bound: int, ordinal: int, width: int) -> list[float]:
        """Build the exact layer-ordinal-v1 feature vector."""
        feats = [1.0 if op == name else 0.0 for name in OP_ORDER]
        norm = float(bound) or 1.0
        feats += [left_size / norm, right_size / norm, size / norm]
        total = len(self.target)
        for key, signature in ((left_key, left_sig),
                               (right_key, right_sig)):
            hit, inside = self.operand(key, signature)
            feats += [hit / total, inside / total]
        fraction = ordinal / (width or 1)
        result = feats + [fraction, 1.0 - fraction]
        if len(result) != DIM:
            raise AssertionError("feature dimension changed")
        return result

