#!/usr/bin/env python3
"""Paired promotion gate with bootstrap bound and curriculum (RSI W3).

Implements issue #17's decision rule in miniature: promotion needs zero
safety failures, a paired seed-cluster bootstrap lower bound above zero,
a minimum mean gain, and spend inside the predeclared budgets. Failure
clusters map to proposed next tasks; nothing is auto-promoted.
"""
from __future__ import annotations

import random
from typing import Any


class PromotionGate:
    """Decide promotion and propose curriculum from failure clusters."""

    def __init__(self, *, min_gain: float = 0.05, alpha: float = 0.05,
                 samples: int = 2000, seed: int = 0,
                 max_tokens: int = 100000,
                 max_wall_s: float = 1800.0) -> None:
        if min_gain < 0 or not 0 < alpha < 1 or samples < 100:
            raise ValueError("gate parameters are invalid")
        if max_tokens < 1 or max_wall_s <= 0:
            raise ValueError("gate budgets must be positive")
        self._min_gain = min_gain
        self._alpha = alpha
        self._samples = samples
        self._seed = seed
        self._max_tokens = max_tokens
        self._max_wall_s = max_wall_s

    def _lower_bound(self, diffs: list[float]) -> float:
        rng = random.Random(self._seed)
        means = []
        for _ in range(self._samples):
            pick = [rng.choice(diffs) for _ in range(len(diffs))]
            means.append(sum(pick) / len(pick))
        means.sort()
        return means[int(self._alpha * self._samples)]

    def _paired_stats(self, base: list[float],
                      cand: list[float]) -> tuple[float, float]:
        if not base or len(base) != len(cand):
            raise ValueError("paired non-empty scores required")
        diffs = [c - b for b, c in zip(base, cand)]
        return sum(diffs) / len(diffs), self._lower_bound(diffs)

    def _budget_reasons(self, tokens_used: int,
                        wall_s: float) -> list[str]:
        reasons = []
        if tokens_used > self._max_tokens:
            reasons.append("token budget exceeded")
        if wall_s > self._max_wall_s:
            reasons.append("wall budget exceeded")
        return reasons

    def decide(self, base: list[float], cand: list[float],
               safety: int, tokens_used: int,
               wall_s: float) -> dict[str, Any]:
        """Verdict with bootstrap bound, gain, and budget usage."""
        gain, lower = self._paired_stats(base, cand)
        reasons = []
        if safety:
            reasons.append("safety failures present")
        if lower <= 0:
            reasons.append("bootstrap lower bound not above zero")
        if gain < self._min_gain:
            reasons.append("mean gain below minimum")
        reasons += self._budget_reasons(tokens_used, wall_s)
        return {"promote": not reasons, "reasons": reasons,
                "mean_gain": gain, "bootstrap_lower": lower,
                "safety_failures": safety, "tokens_used": tokens_used,
                "tokens_cap": self._max_tokens, "wall_s": wall_s,
                "wall_cap_s": self._max_wall_s}

    def curriculum(self, failures: list[dict[str, Any]]
                   ) -> list[dict[str, Any]]:
        """Cluster failures by kind into prioritized next tasks."""
        if not failures:
            return [{"task": "confirm on a fresh seed cluster",
                     "kind": "none", "count": 0, "priority": 0}]
        counts: dict[str, int] = {}
        for item in failures:
            kind = str(item.get("kind", "unknown"))
            counts[kind] = counts.get(kind, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return [{"task": "add cases covering " + kind, "kind": kind,
                 "count": n, "priority": i}
                for i, (kind, n) in enumerate(ranked)]
