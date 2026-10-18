"""Causal persistence, EMA, and NLMS baselines for sensing prediction."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any


MODES = ("persistence", "ema", "nlms")
EMA_TAU_SECONDS = 5.0
NLMS_LEARNING_RATE = 0.05
NLMS_EPSILON = 0.001
UPDATE_LIMIT = 0.01
PARAMETER_LIMIT = 2.0
FIRST_WINDOW_DT = 0.2
SNAPSHOT_VERSION = 1
INITIAL_NLMS_WEIGHTS = (1.0, 0.0, 0.0)


def _finite_number(value: object) -> bool:
    """Return whether value is a finite number, excluding booleans."""
    if (not isinstance(value, (int, float))
            or isinstance(value, bool)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


class SensingBaseline:
    """Predict the next normalized RMS value with a simple causal baseline.

    The NLMS mode uses AR(2) features and updates from a saved forecast only
    after its target arrives. Session boundaries clear history and retain
    NLMS weights. EMA state updates continue when learning is disabled.
    """

    def __init__(self, mode: str, learning: bool = True) -> None:
        """Initialize a baseline mode and its fixed or learned parameters."""
        if mode not in MODES:
            raise ValueError("mode must be persistence, ema, or nlms")
        if type(learning) is not bool:
            raise ValueError("learning must be bool")
        self._mode = mode
        self._learning = learning
        self._weights = list(INITIAL_NLMS_WEIGHTS)
        self._active = False
        self._last_timestamp: float | None = None
        self._pending_prediction: float | None = None
        self._pending_features: list[float] | None = None
        self._metrics = self._empty_metrics()
        self._reset_transient()

    @staticmethod
    def _empty_metrics() -> dict[str, int]:
        """Return counters with a stable schema for snapshots and reports."""
        return {
            "sessions": 0,
            "predictions": 0,
            "scored_targets": 0,
            "updates": 0,
            "parameter_visits": 0,
            "changed_parameters": 0,
            "filter_updates": 0,
        }

    def start_session(self, timestamp: float, value: float) -> float:
        """Consume the first value and return its one-step-ahead forecast."""
        if self._active:
            raise RuntimeError("a session is already active")
        ts, measurement = self._validate_observation(timestamp, value)
        self._reset_transient()
        self._active = True
        self._last_timestamp = ts
        self._metrics["sessions"] += 1
        if self._mode == "ema":
            self._ema = measurement
        elif self._mode == "nlms":
            self._previous_value = measurement
            self._last_value = measurement
        return self._predict_from(measurement)

    def advance(self, timestamp: float, value: float) -> float:
        """Score the saved forecast, update, and forecast the next value."""
        if not self._active or self._last_timestamp is None:
            raise RuntimeError("start_session must be called before advance")
        ts, measurement = self._validate_observation(timestamp, value)
        if ts <= self._last_timestamp:
            raise ValueError("timestamps must increase within a session")
        if self._pending_prediction is None:
            raise RuntimeError("there is no forecast awaiting its target")
        dt = ts - self._last_timestamp
        if not math.isfinite(dt):
            raise ValueError("elapsed time must be finite")
        self._score_pending(measurement)
        self._last_timestamp = ts
        if self._mode == "ema":
            if self._ema is None:
                raise RuntimeError("EMA state is missing")
            alpha = math.exp(-dt / EMA_TAU_SECONDS)
            self._ema = alpha * self._ema + (1.0 - alpha) * measurement
            self._metrics["filter_updates"] += 1
        elif self._mode == "nlms":
            if self._last_value is None:
                raise RuntimeError("NLMS history is missing")
            self._previous_value = self._last_value
            self._last_value = measurement
        return self._predict_from(measurement)

    def end_session(self) -> None:
        """Drop the final unscored forecast and clear transient state."""
        if not self._active:
            raise RuntimeError("no active session")
        self._active = False
        self._last_timestamp = None
        self._reset_transient()

    def evaluation_copy(self) -> SensingBaseline:
        """Return a no-learning copy at a session boundary."""
        if self._active:
            raise RuntimeError("evaluation copy requires a session boundary")
        result = copy.deepcopy(self)
        result._learning = False
        result._metrics = self._empty_metrics()
        result._reset_transient()
        return result

    def metrics(self) -> dict[str, int | str | bool]:
        """Return work counters, mode flags, and parameter fingerprint."""
        result: dict[str, int | str | bool] = dict(self._metrics)
        result["mode"] = self._mode
        result["learning"] = self._learning
        result["weight_hash"] = self.weight_hash()
        return result

    def weight_hash(self) -> str:
        """Hash mode and learned NLMS weights in canonical JSON form."""
        payload = {"mode": self._mode, "weights": self._weights}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def snapshot(self) -> dict[str, Any]:
        """Serialize mode, weights, and counters at a session boundary."""
        if self._active:
            raise RuntimeError("snapshot requires a session boundary")
        return {
            "version": SNAPSHOT_VERSION,
            "mode": self._mode,
            "learning": self._learning,
            "config": self._config(),
            "weights": self._weights[:],
            "metrics": dict(self._metrics),
            "weight_hash": self.weight_hash(),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Atomically restore a compatible session-boundary snapshot."""
        if self._active:
            raise RuntimeError("restore requires a session boundary")
        if not isinstance(snapshot, dict):
            raise ValueError("snapshot must be an object")
        if snapshot.get("version") != SNAPSHOT_VERSION:
            raise ValueError("unsupported snapshot version")
        if (snapshot.get("mode") != self._mode
                or snapshot.get("learning") is not self._learning
                or snapshot.get("config") != self._config()):
            raise ValueError("snapshot configuration mismatch")
        weights = self._validated_weights(snapshot.get("weights"))
        metrics = snapshot.get("metrics")
        if (not isinstance(metrics, dict)
                or set(metrics) != set(self._metrics)
                or any(type(item) is not int or item < 0
                       for item in metrics.values())):
            raise ValueError("invalid metrics snapshot")
        expected_hash = self._hash_for(self._mode, weights)
        if snapshot.get("weight_hash") != expected_hash:
            raise ValueError("snapshot weight hash mismatch")
        self._weights = weights
        self._metrics = dict(metrics)
        self._last_timestamp = None
        self._reset_transient()

    def _predict_from(self, value: float) -> float:
        """Save a causal forecast and its pre-update feature vector."""
        features: list[float] | None = None
        if self._mode == "persistence":
            prediction = value
        elif self._mode == "ema":
            if self._ema is None:
                raise RuntimeError("EMA state is missing")
            prediction = self._ema
        else:
            if self._last_value is None or self._previous_value is None:
                raise RuntimeError("NLMS history is missing")
            features = [self._last_value, self._previous_value, 1.0]
            prediction = sum(weight * feature for weight, feature
                             in zip(self._weights, features))
            prediction = max(0.0, min(1.0, prediction))
        self._pending_prediction = prediction
        self._pending_features = features
        self._metrics["predictions"] += 1
        return prediction

    def _score_pending(self, target: float) -> None:
        """Score the saved forecast and update NLMS from saved features."""
        prediction = self._pending_prediction
        if prediction is None:
            raise RuntimeError("there is no forecast awaiting its target")
        self._metrics["scored_targets"] += 1
        if self._mode == "nlms" and self._learning:
            features = self._pending_features
            if features is None:
                raise RuntimeError("NLMS forecast features are missing")
            old_weights = self._weights[:]
            denominator = NLMS_EPSILON + sum(
                feature * feature for feature in features)
            error = target - prediction
            updates = [self._clip_update(
                NLMS_LEARNING_RATE * error * feature / denominator)
                for feature in features]
            new_weights = [self._clip_parameter(weight + update)
                           for weight, update in zip(old_weights, updates)]
            self._metrics["parameter_visits"] += len(old_weights)
            self._metrics["changed_parameters"] += sum(
                old != new for old, new in zip(old_weights, new_weights))
            self._weights = new_weights
            self._metrics["updates"] += 1
        self._pending_prediction = None
        self._pending_features = None

    def _config(self) -> dict[str, Any]:
        """Return the fixed mode-specific baseline configuration."""
        return {
            "mode": self._mode,
            "learning": self._learning,
            "ema_tau_seconds": EMA_TAU_SECONDS,
            "nlms_learning_rate": NLMS_LEARNING_RATE,
            "nlms_epsilon": NLMS_EPSILON,
            "update_limit": UPDATE_LIMIT,
            "parameter_limit": PARAMETER_LIMIT,
            "first_window_dt_seconds": FIRST_WINDOW_DT,
            "initial_nlms_weights": list(INITIAL_NLMS_WEIGHTS),
        }

    @staticmethod
    def _validate_observation(timestamp: float,
                              value: float) -> tuple[float, float]:
        """Validate and normalize a timestamp and bounded feature."""
        if not _finite_number(timestamp):
            raise ValueError("timestamp must be finite")
        if not _finite_number(value) or not 0.0 <= float(value) <= 1.0:
            raise ValueError("value must be finite and in [0, 1]")
        return float(timestamp), float(value)

    @staticmethod
    def _validated_weights(value: object) -> list[float]:
        """Validate and copy a bounded three-element NLMS vector."""
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError("invalid NLMS weight vector shape")
        if any(not _finite_number(item) for item in value):
            raise ValueError("invalid NLMS weight value")
        weights = [float(item) for item in value]
        if any(abs(weight) > PARAMETER_LIMIT for weight in weights):
            raise ValueError("NLMS weight exceeds configured bound")
        return weights

    @staticmethod
    def _clip_update(value: float) -> float:
        """Clip one NLMS update to the frozen per-update limit."""
        return max(-UPDATE_LIMIT, min(UPDATE_LIMIT, value))

    @staticmethod
    def _clip_parameter(value: float) -> float:
        """Clip one learned NLMS parameter to its frozen bound."""
        return max(-PARAMETER_LIMIT, min(PARAMETER_LIMIT, value))

    @staticmethod
    def _hash_for(mode: str, weights: list[float]) -> str:
        """Hash mode and weights using the public fingerprint format."""
        payload = {"mode": mode, "weights": weights}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _reset_transient(self) -> None:
        """Clear temporal state and all pending one-step prediction data."""
        self._ema: float | None = None
        self._last_value: float | None = None
        self._previous_value: float | None = None
        self._pending_prediction = None
        self._pending_features = None
