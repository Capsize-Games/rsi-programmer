"""Independent SNN predictor reference used by the replay auditor."""
from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any

TAUS = (0.2, 0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0,
        5.0, 5.0, 5.0, 5.0)
FILTER_TAU = 0.2
LIMIT = 2.0
UPDATE_LIMIT = 0.01
READOUT_LR = 0.01
INPUT_LR = 0.001
FIRST_DT = 0.2
NEURAL_ARMS = ("plastic_snn", "readout_snn", "frozen_snn", "plastic_ann")


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _clamp(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _model_mode(arm: str) -> tuple[bool, bool, bool]:
    """Return spiking, readout-learning, and encoder-learning flags."""
    if arm not in NEURAL_ARMS:
        raise ValueError(f"not a neural arm: {arm}")
    return (arm != "plastic_ann", arm != "frozen_snn",
            arm in ("plastic_snn", "plastic_ann"))


class ReferencePredictor:
    """Plain reference implementation; never imports production predictor."""

    def __init__(self, seed: int, arm: str) -> None:
        self.seed = seed
        self.arm = arm
        self.spiking, self.learning, self.encoder_learning = _model_mode(arm)
        base_w = (-1.5, -0.5, 0.5, 1.5)
        base_b = (0.75, 0.85, 0.95, 1.05)
        rng = random.Random(seed)
        self.w: list[float] = []
        self.b: list[float] = []
        for i in range(12):
            self.w.append(base_w[i % 4] + rng.uniform(-0.05, 0.05))
            self.b.append(base_b[i % 4] + rng.uniform(-0.02, 0.02))
        self.rw: list[float] = [
            -0.05 if i % 2 == 0 else 0.05 for i in range(12)]
        self.ob = 0.0
        self.metrics = self._zero_metrics()
        self._reset_state()
        self.initial = self._parameters()

    @staticmethod
    def _zero_metrics() -> dict[str, int]:
        return {
            "sessions": 0, "predictions": 0, "scored_targets": 0,
            "updates": 0, "neuron_ticks": 0, "spikes": 0,
            "readout_synapse_visits": 0, "input_synapse_visits": 0,
            "changed_readout_parameters": 0,
            "changed_input_parameters": 0,
        }

    def _reset_state(self) -> None:
        self.active = False
        self.last_t: float | None = None
        self.pending: float | None = None
        self.pending_rates = [0.0] * 12
        self.pending_we = [0.0] * 12
        self.pending_be = [0.0] * 12
        self.m = [0.0] * 12
        self.r = [0.0] * 12
        self.emw = [0.0] * 12
        self.emb = [0.0] * 12
        self.erw = [0.0] * 12
        self.erb = [0.0] * 12

    def _parameters(self) -> dict[str, Any]:
        return {"input_weights": self.w[:], "input_biases": self.b[:],
                "readout_weights": self.rw[:], "output_bias": self.ob}

    def weight_hash(self) -> str:
        return _digest(self._parameters())

    def snapshot(self) -> dict[str, Any]:
        if self.active:
            raise ValueError("cannot snapshot active reference predictor")
        return {
            "version": 1,
            "config": {
                "seed": self.seed, "spiking": self.spiking,
                "learning": self.learning,
                "encoder_learning": self.encoder_learning,
                "unit_count": 12, "tau_seconds": list(TAUS),
                "filter_tau_seconds": FILTER_TAU, "threshold": 1.0,
                "base_input_weights": [-1.5, -0.5, 0.5, 1.5],
                "base_input_biases": [0.75, 0.85, 0.95, 1.05],
                "first_window_dt_seconds": FIRST_DT,
                "readout_learning_rate": READOUT_LR,
                "input_learning_rate": INPUT_LR,
                "parameter_limit": LIMIT, "update_limit": UPDATE_LIMIT,
                "surrogate": "max(0,1-abs(u-threshold))",
            },
            **self._parameters(), "metrics": dict(self.metrics),
            "weight_hash": self.weight_hash(),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        if self.active or snapshot.get("config") != self.snapshot()["config"]:
            raise ValueError("incompatible reference snapshot")
        params = {key: snapshot[key] for key in (
            "input_weights", "input_biases", "readout_weights", "output_bias")}
        if _digest(params) != snapshot.get("weight_hash"):
            raise ValueError("reference snapshot hash mismatch")
        self.w = [float(x) for x in snapshot["input_weights"]]
        self.b = [float(x) for x in snapshot["input_biases"]]
        self.rw = [float(x) for x in snapshot["readout_weights"]]
        self.ob = float(snapshot["output_bias"])
        self.metrics = dict(snapshot["metrics"])
        self._reset_state()

    def _process(self, x: float, dt: float) -> float:
        rates = [0.0] * 12
        we = [0.0] * 12
        be = [0.0] * 12
        spike_count = 0
        for i, tau in enumerate(TAUS):
            leak = math.exp(-dt / tau)
            q = math.exp(-dt / FILTER_TAU)
            u = leak * self.m[i] + (1.0 - leak) * (self.w[i] * x + self.b[i])
            we_pre = leak * self.emw[i] + (1.0 - leak) * x
            be_pre = leak * self.emb[i] + (1.0 - leak)
            if self.spiking:
                spike = float(u >= 1.0)
                psi = max(0.0, 1.0 - abs(u - 1.0))
                es_w, es_b = psi * we_pre, psi * be_pre
                self.m[i] = u - spike
                self.emw[i] = we_pre - es_w
                self.emb[i] = be_pre - es_b
                spike_count += int(spike)
            else:
                spike = math.tanh(u)
                deriv = 1.0 - spike * spike
                es_w, es_b = deriv * we_pre, deriv * be_pre
                self.m[i] = u
                self.emw[i], self.emb[i] = we_pre, be_pre
            self.r[i] = q * self.r[i] + (1.0 - q) * spike
            self.erw[i] = q * self.erw[i] + (1.0 - q) * es_w
            self.erb[i] = q * self.erb[i] + (1.0 - q) * es_b
            rates[i], we[i], be[i] = self.r[i], self.erw[i], self.erb[i]
        logit = sum(a * b for a, b in zip(self.rw, rates)) + self.ob
        prediction = _sigmoid(logit)
        self.pending = prediction
        self.pending_rates, self.pending_we, self.pending_be = rates, we, be
        self.metrics["predictions"] += 1
        self.metrics["neuron_ticks"] += 12
        self.metrics["spikes"] += spike_count
        return prediction

    def _learn(self, target: float) -> None:
        if self.pending is None:
            raise ValueError("no pending reference forecast")
        self.metrics["scored_targets"] += 1
        if self.learning:
            p = self.pending
            delta = (p - target) * p * (1.0 - p)
            old_rw = self.rw[:]
            dr = [_clamp(-READOUT_LR * delta * rate, UPDATE_LIMIT)
                  for rate in self.pending_rates]
            dob = _clamp(-READOUT_LR * delta, UPDATE_LIMIT)
            dw = [0.0] * 12
            db = [0.0] * 12
            if self.encoder_learning:
                dw = [_clamp(-INPUT_LR * delta * old_rw[i]
                             * self.pending_we[i], UPDATE_LIMIT)
                      for i in range(12)]
                db = [_clamp(-INPUT_LR * delta * old_rw[i]
                             * self.pending_be[i], UPDATE_LIMIT)
                      for i in range(12)]
            changed_r = 0
            for i, d in enumerate(dr):
                if d:
                    old_value = self.rw[i]
                    new_value = _clamp(old_value + d, LIMIT)
                    self.rw[i] = new_value
                    changed_r += int(new_value != old_value)
            if dob:
                old_bias = self.ob
                self.ob = _clamp(old_bias + dob, LIMIT)
                changed_r += int(self.ob != old_bias)
            changed_i = 0
            if self.encoder_learning:
                for i, d in enumerate(dw):
                    if d:
                        old_value = self.w[i]
                        new_value = _clamp(old_value + d, LIMIT)
                        self.w[i] = new_value
                        changed_i += int(new_value != old_value)
                for i, d in enumerate(db):
                    if d:
                        old_value = self.b[i]
                        new_value = _clamp(old_value + d, LIMIT)
                        self.b[i] = new_value
                        changed_i += int(new_value != old_value)
                self.metrics["input_synapse_visits"] += 24
            self.metrics["readout_synapse_visits"] += 13
            self.metrics["changed_readout_parameters"] += changed_r
            self.metrics["changed_input_parameters"] += changed_i
            self.metrics["updates"] += 1
        self.pending = None

    def start(self, timestamp: float, value: float) -> float:
        if self.active or not math.isfinite(timestamp) \
                or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("invalid reference session start")
        self._reset_state()
        self.active = True
        self.last_t = timestamp
        self.metrics["sessions"] += 1
        return self._process(value, FIRST_DT)

    def advance(self, timestamp: float, value: float) -> float:
        if (not self.active or self.last_t is None or self.pending is None
                or not math.isfinite(timestamp) or timestamp <= self.last_t
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("invalid reference observation")
        dt = timestamp - self.last_t
        self._learn(value)
        self.last_t = timestamp
        return self._process(value, dt)

    def end(self) -> None:
        if not self.active:
            raise ValueError("reference session not active")
        self.active = False
        self.last_t = None
        self._reset_state()
