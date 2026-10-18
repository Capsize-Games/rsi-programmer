"""Small time-aware online predictor for normalized vibration features."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from typing import Any

UNIT_COUNT = 12
TAU_SECONDS = (0.2, 0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0,
               5.0, 5.0, 5.0, 5.0)
FILTER_TAU_SECONDS = 0.2
THRESHOLD = 1.0
BASE_INPUT_WEIGHTS = (-1.5, -0.5, 0.5, 1.5)
BASE_INPUT_BIASES = (0.75, 0.85, 0.95, 1.05)
READOUT_LR = 0.01
INPUT_LR = 0.001
PARAMETER_LIMIT = 2.0
UPDATE_LIMIT = 0.01
FIRST_WINDOW_DT = 0.2
SNAPSHOT_VERSION = 1


def _finite_number(value: object) -> bool:
    """Return whether value is a finite int or float, but not bool."""
    return (isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _sigmoid(value: float) -> float:
    """Return a numerically stable logistic activation."""
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


class SensingPredictor:
    """Predict the next normalized RMS value from causal sensor history.

    ``start_session`` consumes the first measurement and emits a forecast.
    Each ``advance`` scores that saved forecast against the new actual value,
    updates from the saved pre-update readout and eligibility, then emits the
    next forecast. Session resets clear temporal traces while retaining
    learned parameters.
    """

    def __init__(self, seed: int = 0, spiking: bool = True,
                 learning: bool = True,
                 encoder_learning: bool = True) -> None:
        """Initialize deterministic neural parameters and empty state."""
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if type(spiking) is not bool:
            raise ValueError("spiking must be bool")
        if type(learning) is not bool:
            raise ValueError("learning must be bool")
        if type(encoder_learning) is not bool:
            raise ValueError("encoder_learning must be bool")
        self._seed = seed
        self._spiking = spiking
        self._learning = learning
        self._encoder_learning = encoder_learning
        rng = random.Random(seed)
        self._input_weights = []
        self._input_biases = []
        for index in range(UNIT_COUNT):
            base_index = index % 4
            self._input_weights.append(
                BASE_INPUT_WEIGHTS[base_index] + rng.uniform(-0.05, 0.05)
            )
            self._input_biases.append(
                BASE_INPUT_BIASES[base_index] + rng.uniform(-0.02, 0.02)
            )
        self._readout_weights = [
            -0.05 if index % 2 == 0 else 0.05
            for index in range(UNIT_COUNT)
        ]
        self._output_bias = 0.0
        self._initial_input_weights = self._input_weights[:]
        self._initial_input_biases = self._input_biases[:]
        self._initial_readout_weights = self._readout_weights[:]
        self._initial_output_bias = self._output_bias
        self._active = False
        self._last_timestamp: float | None = None
        self._pending_prediction: float | None = None
        self._pending_rates = [0.0] * UNIT_COUNT
        self._pending_weight_eligibility = [0.0] * UNIT_COUNT
        self._pending_bias_eligibility = [0.0] * UNIT_COUNT
        self._metrics = self._empty_metrics()
        self._reset_transient()

    @staticmethod
    def _empty_metrics() -> dict[str, int]:
        """Return all integer work counters in a stable order."""
        return {
            "sessions": 0,
            "predictions": 0,
            "scored_targets": 0,
            "updates": 0,
            "neuron_ticks": 0,
            "spikes": 0,
            "readout_synapse_visits": 0,
            "input_synapse_visits": 0,
            "changed_readout_parameters": 0,
            "changed_input_parameters": 0,
        }

    def start_session(self, timestamp: float, value: float) -> float:
        """Start a session from one real value and predict the next value."""
        if self._active:
            raise RuntimeError("a session is already active")
        ts, measurement = self._validate_observation(timestamp, value)
        self._reset_transient()
        self._active = True
        self._last_timestamp = ts
        self._metrics["sessions"] += 1
        return self._predict_from(measurement, FIRST_WINDOW_DT)

    def advance(self, timestamp: float, value: float) -> float:
        """Score the pending forecast, learn, then forecast from new input."""
        if not self._active or self._last_timestamp is None:
            raise RuntimeError("start_session must be called before advance")
        ts, measurement = self._validate_observation(timestamp, value)
        if ts <= self._last_timestamp:
            raise ValueError("timestamps must increase within a session")
        if self._pending_prediction is None:
            raise RuntimeError("there is no forecast awaiting its target")
        dt = ts - self._last_timestamp
        if not math.isfinite(dt):
            raise ValueError("time delta must be finite")
        self._score_pending(measurement)
        self._last_timestamp = ts
        return self._predict_from(measurement, dt)

    def end_session(self) -> None:
        """End the session and discard its final unscored forecast."""
        if not self._active:
            raise RuntimeError("no active session")
        self._active = False
        self._last_timestamp = None
        self._reset_transient()

    def reset_learning(self) -> None:
        """Restore deterministic initial parameters and clear all counters."""
        if self._active:
            raise RuntimeError("cannot reset learning during a session")
        self._input_weights = self._initial_input_weights[:]
        self._input_biases = self._initial_input_biases[:]
        self._readout_weights = self._initial_readout_weights[:]
        self._output_bias = self._initial_output_bias
        self._metrics = self._empty_metrics()
        self._reset_transient()

    def evaluation_copy(self) -> SensingPredictor:
        """Return a boundary copy with parameter learning disabled."""
        if self._active:
            raise RuntimeError("evaluation copy requires a session boundary")
        result = copy.deepcopy(self)
        result._learning = False
        result._encoder_learning = False
        result._metrics = self._empty_metrics()
        result._reset_transient()
        return result

    def metrics(self) -> dict[str, int | str | bool]:
        """Return work counters, mode flags, and parameter fingerprints."""
        result: dict[str, int | str | bool] = dict(self._metrics)
        result["spiking"] = self._spiking
        result["learning"] = self._learning
        result["encoder_learning"] = self._encoder_learning
        result["weight_hash"] = self.weight_hash()
        result["encoder_hash"] = self.encoder_hash()
        return result

    def weight_hash(self) -> str:
        """Hash all trainable weights and biases in canonical JSON form."""
        return self._hash_payload({
            "input_weights": self._input_weights,
            "input_biases": self._input_biases,
            "readout_weights": self._readout_weights,
            "output_bias": self._output_bias,
        })

    def encoder_hash(self) -> str:
        """Hash the input projection weights and biases."""
        return self._hash_payload({
            "input_weights": self._input_weights,
            "input_biases": self._input_biases,
        })

    def snapshot(self) -> dict[str, Any]:
        """Serialize learned parameters and counters at a session boundary."""
        if self._active:
            raise RuntimeError("snapshot requires a session boundary")
        payload: dict[str, Any] = {
            "version": SNAPSHOT_VERSION,
            "config": self._config(),
            "input_weights": self._input_weights[:],
            "input_biases": self._input_biases[:],
            "readout_weights": self._readout_weights[:],
            "output_bias": self._output_bias,
            "metrics": dict(self._metrics),
            "weight_hash": self.weight_hash(),
        }
        return payload

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Atomically restore a compatible session-boundary snapshot."""
        if self._active:
            raise RuntimeError("restore requires a session boundary")
        if not isinstance(snapshot, dict):
            raise ValueError("snapshot must be an object")
        if snapshot.get("version") != SNAPSHOT_VERSION:
            raise ValueError("unsupported snapshot version")
        if snapshot.get("config") != self._config():
            raise ValueError("snapshot configuration mismatch")
        input_weights = self._vector(snapshot.get("input_weights"),
                                     UNIT_COUNT, "input weights")
        input_biases = self._vector(snapshot.get("input_biases"),
                                    UNIT_COUNT, "input biases")
        readout_weights = self._vector(snapshot.get("readout_weights"),
                                       UNIT_COUNT, "readout weights")
        output_bias = snapshot.get("output_bias")
        if not _finite_number(output_bias):
            raise ValueError("invalid output bias")
        params = [*input_weights, *input_biases, *readout_weights,
                  float(output_bias)]
        if any(abs(value) > PARAMETER_LIMIT for value in params):
            raise ValueError("snapshot parameter exceeds configured bound")
        metrics = snapshot.get("metrics")
        if (not isinstance(metrics, dict)
                or set(metrics) != set(self._metrics)
                or any(type(value) is not int or value < 0
                       for value in metrics.values())):
            raise ValueError("invalid metrics snapshot")
        expected_hash = self._hash_payload({
            "input_weights": input_weights,
            "input_biases": input_biases,
            "readout_weights": readout_weights,
            "output_bias": float(output_bias),
        })
        if snapshot.get("weight_hash") != expected_hash:
            raise ValueError("snapshot weight hash mismatch")
        self._input_weights = input_weights
        self._input_biases = input_biases
        self._readout_weights = readout_weights
        self._output_bias = float(output_bias)
        self._metrics = dict(metrics)
        self._last_timestamp = None
        self._reset_transient()

    def _config(self) -> dict[str, Any]:
        """Return immutable model and learning configuration."""
        return {
            "seed": self._seed,
            "spiking": self._spiking,
            "learning": self._learning,
            "encoder_learning": self._encoder_learning,
            "unit_count": UNIT_COUNT,
            "tau_seconds": list(TAU_SECONDS),
            "filter_tau_seconds": FILTER_TAU_SECONDS,
            "threshold": THRESHOLD,
            "base_input_weights": list(BASE_INPUT_WEIGHTS),
            "base_input_biases": list(BASE_INPUT_BIASES),
            "first_window_dt_seconds": FIRST_WINDOW_DT,
            "readout_learning_rate": READOUT_LR,
            "input_learning_rate": INPUT_LR,
            "parameter_limit": PARAMETER_LIMIT,
            "update_limit": UPDATE_LIMIT,
            "surrogate": "max(0,1-abs(u-threshold))",
        }

    def _predict_from(self, value: float, dt: float) -> float:
        """Process a known measurement and save its next-step forecast."""
        rates, weight_elig, bias_elig, spikes = self._process(value, dt)
        logit = sum(weight * rate for weight, rate in zip(
            self._readout_weights, rates)) + self._output_bias
        prediction = _sigmoid(logit)
        self._pending_prediction = prediction
        self._pending_rates = rates[:]
        self._pending_weight_eligibility = weight_elig[:]
        self._pending_bias_eligibility = bias_elig[:]
        self._metrics["predictions"] += 1
        self._metrics["neuron_ticks"] += UNIT_COUNT
        self._metrics["spikes"] += spikes
        return prediction

    def _score_pending(self, target: float) -> None:
        """Use the saved forecast and pre-update readout for one target."""
        prediction = self._pending_prediction
        if prediction is None:
            raise RuntimeError("there is no forecast awaiting its target")
        self._metrics["scored_targets"] += 1
        if self._learning:
            delta = (prediction - target) * prediction * (1.0 - prediction)
            old_readout = self._readout_weights[:]
            readout_deltas = [self._clip_update(
                -READOUT_LR * delta * rate)
                for rate in self._pending_rates]
            bias_delta = self._clip_update(-READOUT_LR * delta)
            input_deltas = [0.0] * UNIT_COUNT
            input_bias_deltas = [0.0] * UNIT_COUNT
            if self._encoder_learning:
                input_deltas = [self._clip_update(
                    -INPUT_LR * delta * old_readout[index]
                    * self._pending_weight_eligibility[index])
                    for index in range(UNIT_COUNT)]
                input_bias_deltas = [self._clip_update(
                    -INPUT_LR * delta * old_readout[index]
                    * self._pending_bias_eligibility[index])
                    for index in range(UNIT_COUNT)]
            changed_readout = 0
            for index, update in enumerate(readout_deltas):
                if update:
                    old_value = self._readout_weights[index]
                    new_value = self._clip_parameter(old_value + update)
                    self._readout_weights[index] = new_value
                    changed_readout += int(new_value != old_value)
            if bias_delta:
                old_bias = self._output_bias
                self._output_bias = self._clip_parameter(old_bias + bias_delta)
                changed_readout += int(self._output_bias != old_bias)
            changed_input = 0
            if self._encoder_learning:
                for index, update in enumerate(input_deltas):
                    if update:
                        old_value = self._input_weights[index]
                        new_value = self._clip_parameter(old_value + update)
                        self._input_weights[index] = new_value
                        changed_input += int(new_value != old_value)
                for index, update in enumerate(input_bias_deltas):
                    if update:
                        old_value = self._input_biases[index]
                        new_value = self._clip_parameter(old_value + update)
                        self._input_biases[index] = new_value
                        changed_input += int(new_value != old_value)
                self._metrics["input_synapse_visits"] += 2 * UNIT_COUNT
            self._metrics["readout_synapse_visits"] += UNIT_COUNT + 1
            self._metrics["changed_readout_parameters"] += changed_readout
            self._metrics["changed_input_parameters"] += changed_input
            self._metrics["updates"] += 1
        self._pending_prediction = None

    def _process(self, value: float, dt: float
                 ) -> tuple[list[float], list[float], list[float], int]:
        """Advance LIF/ANN states and local surrogate eligibility traces."""
        rates = [0.0] * UNIT_COUNT
        weight_eligibility = [0.0] * UNIT_COUNT
        bias_eligibility = [0.0] * UNIT_COUNT
        spikes = 0
        for index, tau in enumerate(TAU_SECONDS):
            leak = math.exp(-dt / tau)
            filter_leak = math.exp(-dt / FILTER_TAU_SECONDS)
            drive = (self._input_weights[index] * value
                     + self._input_biases[index])
            voltage = leak * self._membrane[index] + (1.0 - leak) * drive
            weight_eu = (leak * self._membrane_weight_eligibility[index]
                         + (1.0 - leak) * value)
            bias_eu = (leak * self._membrane_bias_eligibility[index]
                       + (1.0 - leak))
            if self._spiking:
                spike = float(voltage >= THRESHOLD)
                surrogate = max(0.0, 1.0 - abs(voltage - THRESHOLD))
                spike_weight_eligibility = surrogate * weight_eu
                spike_bias_eligibility = surrogate * bias_eu
                self._membrane[index] = voltage - spike
                self._membrane_weight_eligibility[index] = (
                    weight_eu - spike_weight_eligibility)
                self._membrane_bias_eligibility[index] = (
                    bias_eu - spike_bias_eligibility)
                spikes += int(spike)
            else:
                spike = math.tanh(voltage)
                derivative = 1.0 - spike * spike
                spike_weight_eligibility = derivative * weight_eu
                spike_bias_eligibility = derivative * bias_eu
                self._membrane[index] = voltage
                self._membrane_weight_eligibility[index] = weight_eu
                self._membrane_bias_eligibility[index] = bias_eu
            self._rates[index] = (
                filter_leak * self._rates[index]
                + (1.0 - filter_leak) * spike)
            self._rate_weight_eligibility[index] = (
                filter_leak * self._rate_weight_eligibility[index]
                + (1.0 - filter_leak) * spike_weight_eligibility)
            self._rate_bias_eligibility[index] = (
                filter_leak * self._rate_bias_eligibility[index]
                + (1.0 - filter_leak) * spike_bias_eligibility)
            rates[index] = self._rates[index]
            weight_eligibility[index] = self._rate_weight_eligibility[index]
            bias_eligibility[index] = self._rate_bias_eligibility[index]
        return rates, weight_eligibility, bias_eligibility, spikes

    @staticmethod
    def _validate_observation(timestamp: float,
                              value: float) -> tuple[float, float]:
        """Validate and normalize a timestamp and bounded RMS feature."""
        if not _finite_number(timestamp):
            raise ValueError("timestamp must be finite")
        if not _finite_number(value) or not 0.0 <= float(value) <= 1.0:
            raise ValueError("value must be finite and in [0, 1]")
        return float(timestamp), float(value)

    @staticmethod
    def _vector(value: object, length: int, label: str) -> list[float]:
        """Validate and copy a finite vector."""
        if not isinstance(value, list) or len(value) != length:
            raise ValueError(f"invalid {label} vector shape")
        if any(not _finite_number(item) for item in value):
            raise ValueError(f"invalid {label} vector value")
        return [float(item) for item in value]

    @staticmethod
    def _clip_update(value: float) -> float:
        """Bound a single parameter update by the frozen contract limit."""
        return max(-UPDATE_LIMIT, min(UPDATE_LIMIT, value))

    @staticmethod
    def _clip_parameter(value: float) -> float:
        """Bound a learned parameter by the frozen contract limit."""
        return max(-PARAMETER_LIMIT, min(PARAMETER_LIMIT, value))

    @staticmethod
    def _hash_payload(payload: dict[str, Any]) -> str:
        """Hash JSON-compatible state deterministically."""
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _reset_transient(self) -> None:
        """Clear membrane, filter, eligibility, and pending forecast state."""
        self._membrane = [0.0] * UNIT_COUNT
        self._rates = [0.0] * UNIT_COUNT
        self._membrane_weight_eligibility = [0.0] * UNIT_COUNT
        self._membrane_bias_eligibility = [0.0] * UNIT_COUNT
        self._rate_weight_eligibility = [0.0] * UNIT_COUNT
        self._rate_bias_eligibility = [0.0] * UNIT_COUNT
        self._pending_prediction = None
        self._pending_rates = [0.0] * UNIT_COUNT
        self._pending_weight_eligibility = [0.0] * UNIT_COUNT
        self._pending_bias_eligibility = [0.0] * UNIT_COUNT
