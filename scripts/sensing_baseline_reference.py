"""Independent persistence, EMA, and NLMS reference implementation."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

FIRST_DT = 0.2
UPDATE_LIMIT = 0.01
LIMIT = 2.0


def _finite_number(value: object) -> bool:
    """Return whether a value is finite numeric data, excluding booleans."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_observation(
    timestamp: float,
    value: float,
) -> tuple[float, float]:
    """Validate and normalize a timestamp and bounded sensor feature."""
    if not _finite_number(timestamp):
        raise ValueError("timestamp must be finite")
    if not _finite_number(value) or not 0.0 <= float(value) <= 1.0:
        raise ValueError("value must be finite and in [0, 1]")
    return float(timestamp), float(value)


def _clamp(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


class ReferenceBaseline:
    """Independent persistence, EMA, or normalized LMS forecast state."""

    def __init__(self, arm: str) -> None:
        if arm not in ("persistence", "ema", "nlms"):
            raise ValueError("unknown baseline")
        self.arm = arm
        self.metrics = self._empty_metrics()
        self.reset_state()
        self.weights = [1.0, 0.0, 0.0]

    @staticmethod
    def _empty_metrics() -> dict[str, int]:
        return {"sessions": 0, "predictions": 0, "scored_targets": 0,
                "updates": 0, "parameter_visits": 0,
                "changed_parameters": 0, "filter_updates": 0}

    def reset_state(self) -> None:
        self.active = False
        self.last_t: float | None = None
        self.pending: float | None = None
        self.pending_f: list[float] | None = None
        self.previous: float | None = None
        self.ema: float | None = None
        self.last: float | None = None

    def weight_hash(self) -> str:
        return _digest({"mode": self.arm, "weights": self.weights})

    def snapshot(self) -> dict[str, Any]:
        if self.active:
            raise ValueError("cannot snapshot active baseline")
        payload = {"version": 1, "mode": self.arm, "learning": True,
                   "config": self._config(), "weights": self.weights[:],
                   "metrics": dict(self.metrics)}
        payload["weight_hash"] = self.weight_hash()
        return payload

    def _config(self) -> dict[str, Any]:
        return {"mode": self.arm, "learning": True,
                "ema_tau_seconds": 5.0, "nlms_learning_rate": 0.05,
                "nlms_epsilon": 0.001, "update_limit": 0.01,
                "parameter_limit": 2.0,
                "first_window_dt_seconds": FIRST_DT,
                "initial_nlms_weights": [1.0, 0.0, 0.0]}

    def start(self, timestamp: float, value: float) -> float:
        if self.active:
            raise ValueError("invalid baseline start")
        timestamp, value = _validate_observation(timestamp, value)
        self.reset_state()
        self.active = True
        self.last_t = timestamp
        self.last = value
        self.metrics["sessions"] += 1
        return self._forecast(value)

    def _forecast(self, value: float) -> float:
        if self.arm == "persistence":
            result = value
        elif self.arm == "ema":
            if self.ema is None:
                self.ema = value
            result = self.ema
        else:
            if self.last is None:
                raise ValueError("NLMS has no current feature")
            prev = self.last if self.previous is None else self.previous
            features = [self.last, prev, 1.0]
            result = max(0.0, min(1.0, sum(
                weight * feature for weight, feature in zip(
                    self.weights, features))))
            self.pending_f = features
        self.pending = result
        self.metrics["predictions"] += 1
        return result

    def advance(self, timestamp: float, value: float) -> float:
        if (not self.active or self.last_t is None or self.pending is None):
            raise ValueError("invalid baseline observation")
        timestamp, value = _validate_observation(timestamp, value)
        if timestamp <= self.last_t:
            raise ValueError("timestamps must increase within a session")
        dt = timestamp - self.last_t
        if not math.isfinite(dt):
            raise ValueError("elapsed time must be finite")
        self.metrics["scored_targets"] += 1
        if self.arm == "nlms":
            if self.pending_f is None:
                raise ValueError("NLMS forecast has no saved features")
            err = value - self.pending
            norm = 0.001 + sum(x * x for x in self.pending_f)
            for i, feature in enumerate(self.pending_f):
                step = _clamp(0.05 * err * feature / norm, UPDATE_LIMIT)
                new_weight = _clamp(self.weights[i] + step, LIMIT)
                self.metrics["changed_parameters"] += (
                    self.weights[i] != new_weight)
                self.weights[i] = new_weight
            self.metrics["parameter_visits"] += 3
            self.previous = self.last
            self.last = value
            self.metrics["updates"] += 1
        elif self.arm == "ema":
            if self.ema is None:
                raise ValueError("EMA was not initialized")
            leak = math.exp(-dt / 5.0)
            self.ema = leak * self.ema + (1.0 - leak) * value
            self.last = value
            self.metrics["filter_updates"] += 1
        else:
            self.previous, self.last = self.last, value
        self.last_t = timestamp
        prediction = self._forecast(value)
        return prediction

    def end(self) -> None:
        if not self.active:
            raise ValueError("baseline session not active")
        self.active = False
        self.last_t = None
        self.pending = None
        self.pending_f = None
        self.last = None
        self.previous = None
        self.ema = None
