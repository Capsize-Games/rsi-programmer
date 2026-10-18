"""Causal residual alarms and separate evaluator-only event metrics."""
from __future__ import annotations

import math
from typing import Any


class SensingAlarm:
    """Apply a fixed residual threshold without access to event labels."""

    def __init__(self, threshold: float) -> None:
        if (type(threshold) not in (int, float)
                or not math.isfinite(threshold) or threshold <= 0):
            raise ValueError("threshold must be finite and positive")
        self.threshold = float(threshold)
        self._last_index = -1
        self._last_alarm = -5

    def observe(self, index: int, score: float | None) -> bool:
        """Decide before a predictor consumes this target for learning."""
        if type(index) is not int or index != self._last_index + 1:
            raise ValueError("window indices must start at zero and advance")
        if score is not None and (
                type(score) not in (int, float)
                or not math.isfinite(score) or score < 0):
            raise ValueError("score must be finite and nonnegative")
        if index > 0 and score is None:
            raise ValueError("a previous forecast must be scored")
        alarm = (index >= 5 and score is not None
                 and score > self.threshold and index - self._last_alarm >= 5)
        self._last_index = index
        if alarm:
            self._last_alarm = index
        return alarm


def calibrated_threshold(scores: list[float]) -> float:
    """Use the frozen nearest-rank quantile, without test labels."""
    if not scores or any(not math.isfinite(x) or x < 0 for x in scores):
        raise ValueError("calibration scores must be finite and nonempty")
    return max(1e-9, sorted(scores)[math.ceil(0.995 * len(scores)) - 1])


def event_metrics(windows: list[dict[str, Any]], events: list[int]
                  ) -> dict[str, Any]:
    """Join evaluator labels after the causal policy and alarm run ends."""
    alarms = [row["i"] for row in windows if row["alarm"]]
    matched: set[int] = set()
    delays = []
    for event in events:
        candidates = [i for i in alarms if event <= i < event + 5
                      and i not in matched]
        if candidates:
            index = min(candidates)
            matched.add(index)
            delays.append((index + 1) * 0.2 - event * 0.2)
    stable = [row for row in windows if row["scored"]
              and not any(event <= row["i"] < event + 5 for event in events)]
    stable_seconds = len(stable) * 0.2
    if stable_seconds <= 0:
        raise ValueError("evaluation requires scored stable windows")
    scores = [row["score"] for row in windows if row["scored"]]
    false = len(alarms) - len(matched)
    return {
        "events": len(events), "detected": len(matched),
        "false_alarms": false, "stable_seconds": stable_seconds,
        "false_alarms_per_minute": false * 60.0 / stable_seconds,
        "delay_seconds": delays, "scored_windows": len(scores),
        "squared_error_sum": sum(score * score for score in scores),
    }
