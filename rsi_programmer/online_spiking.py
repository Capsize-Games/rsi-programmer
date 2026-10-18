"""Reward-modulated online policy with a fixed binary LIF encoder.

Only the small softmax readout learns. The encoder emits binary spikes and
uses reward-modulated eligibility traces; this is a software prototype, not
evidence of hardware energy efficiency or full-network plasticity.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any

ACTION_ORDER = ("add", "sub", "mul")
FEATURE_ENCODING = "episode-context-v1"
SNAPSHOT_VERSION = 1
FEATURE_COUNT = 6
HIDDEN_COUNT = 16
STEPS = 6
BETA = 0.8
THRESHOLD = 1.0
BASELINE_RATE = 0.05
MAX_UPDATE = 0.05
MAX_WEIGHT = 5.0
MIN_TEMPERATURE = 0.1
TEMPERATURE = 1.0


def _finite(value: object) -> bool:
    """Return whether ``value`` is a finite real number, excluding bools."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _nested_lists(value: Any) -> Any:
    """Convert tuple-valued random state into JSON-compatible lists."""
    if isinstance(value, tuple):
        return [_nested_lists(item) for item in value]
    if isinstance(value, list):
        return [_nested_lists(item) for item in value]
    return value


def _nested_tuples(value: Any) -> Any:
    """Convert JSON-compatible random state lists back into tuples."""
    if isinstance(value, list):
        return tuple(_nested_tuples(item) for item in value)
    return value


class OnlineSpikingPolicy:
    """Select root-operation lanes and adapt from observed rewards.

    The six input features are a bias, previous-operation one-hot (or all
    zeros initially), previous train error, and search progress. ``observe``
    must be called exactly once after each selected lane's real outcome.
    Version 1 assigns each reward only to its pending decision; it does not
    implement delayed credit across multiple actions.
    """

    def __init__(
        self,
        seed: int = 0,
        learning: bool = True,
        spiking: bool = True,
        lr: float = 0.05,
        eligibility_decay: float = 0.0,
    ) -> None:
        """Create a seeded fixed LIF encoder and zeroed plastic readout."""
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if type(learning) is not bool:
            raise ValueError("learning must be bool")
        if type(spiking) is not bool:
            raise ValueError("spiking must be bool")
        if not _finite(lr) or float(lr) <= 0.0:
            raise ValueError("lr must be finite and positive")
        if not _finite(eligibility_decay) or float(eligibility_decay) != 0.0:
            raise ValueError("v1 supports current-decision credit only")
        self._seed = seed
        self._learning = learning
        self._spiking = spiking
        self._lr = float(lr)
        self._eligibility_decay = float(eligibility_decay)
        self._rng = random.Random(seed)
        self._encoder = [
            [float(self._rng.choice((-1, 0, 1)))
             for _ in range(FEATURE_COUNT)]
            for _ in range(HIDDEN_COUNT)
        ]
        for row in self._encoder:
            if not any(row):
                row[self._rng.randrange(FEATURE_COUNT)] = 1.0
        self._initial_encoder = [row[:] for row in self._encoder]
        self._weights = [
            [0.0 for _ in range(HIDDEN_COUNT)]
            for _ in ACTION_ORDER
        ]
        self._bias = [0.0 for _ in ACTION_ORDER]
        self._baseline = 0.0
        self._membrane = [0.0 for _ in range(HIDDEN_COUNT)]
        self._eligibility_weights = [
            [0.0 for _ in range(HIDDEN_COUNT)]
            for _ in ACTION_ORDER
        ]
        self._eligibility_bias = [0.0 for _ in ACTION_ORDER]
        self._episode_id: str | None = None
        self._pending_action: int | None = None
        self._metrics = {
            "decisions": 0,
            "neuron_ticks": 0,
            "encoder_mac": 0,
            "readout_mac": 0,
            "updates": 0,
            "update_synapses": 0,
            "update_biases": 0,
            "update_synapse_visits": 0,
            "spikes": 0,
            "reward_observations": 0,
            "episodes": 0,
        }

    def start_episode(self, episode_id: str) -> None:
        """Start an episode and clear all transient state and credit traces."""
        if self._episode_id is not None:
            raise RuntimeError("an episode is already active")
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("episode_id must be a non-empty string")
        self._episode_id = episode_id
        self._pending_action = None
        self._reset_transient()
        self._clear_eligibility()
        self._metrics["episodes"] += 1

    def choose(
        self,
        features: tuple[float, ...],
        available: tuple[str, ...],
    ) -> str:
        """Sample one legal action from masked SNN readout probabilities."""
        if self._episode_id is None:
            raise RuntimeError("start_episode must be called before choose")
        if self._pending_action is not None:
            raise RuntimeError("observe the pending action before choose")
        values = self._validate_features(features)
        actions = self._validate_available(available)
        rates, spikes = self._encode(values)
        logits = [
            sum(w * h for w, h in zip(self._weights[i], rates))
            + self._bias[i]
            for i in range(len(ACTION_ORDER))
        ]
        probs = self._masked_softmax(logits, actions)
        picked = self._sample(actions, probs)
        if self._learning:
            self._accumulate_eligibility(picked, rates, probs, actions)
        self._pending_action = picked
        self._metrics["decisions"] += 1
        self._metrics["spikes"] += spikes
        self._metrics["neuron_ticks"] += STEPS * HIDDEN_COUNT
        self._metrics["encoder_mac"] += (
            STEPS * FEATURE_COUNT * HIDDEN_COUNT
        )
        self._metrics["readout_mac"] += (
            len(ACTION_ORDER) * HIDDEN_COUNT
        )
        return ACTION_ORDER[picked]

    def observe(self, reward: float) -> float:
        """Consume the pending action's real outcome and update the readout.

        Reward is clipped to [-1, 1]. Returns the baseline-centered reward
        used for this update. Calling without a pending choice is an error.
        """
        if self._episode_id is None:
            raise RuntimeError("no active episode")
        if self._pending_action is None:
            raise RuntimeError("there is no action awaiting observation")
        if not _finite(reward):
            raise ValueError("reward must be a finite real number")
        clipped = max(-1.0, min(1.0, float(reward)))
        advantage = clipped - self._baseline
        if self._learning:
            changed = 0
            changed_biases = 0
            for action in range(len(ACTION_ORDER)):
                for hidden in range(HIDDEN_COUNT):
                    eligibility = self._eligibility_weights[action][hidden]
                    delta = self._clip_update(self._lr * advantage
                                              * eligibility)
                    if delta:
                        self._weights[action][hidden] = max(
                            -MAX_WEIGHT,
                            min(MAX_WEIGHT,
                                self._weights[action][hidden] + delta),
                        )
                        changed += 1
                delta = self._clip_update(
                    self._lr * advantage
                    * self._eligibility_bias[action]
                )
                if delta:
                    self._bias[action] = max(
                        -MAX_WEIGHT,
                        min(MAX_WEIGHT, self._bias[action] + delta),
                    )
                    changed_biases += 1
            self._baseline += BASELINE_RATE * advantage
            self._metrics["updates"] += 1
            self._metrics["update_synapses"] += changed
            self._metrics["update_biases"] += changed_biases
            self._metrics["update_synapse_visits"] += (
                len(ACTION_ORDER) * (HIDDEN_COUNT + 1)
            )
        self._metrics["reward_observations"] += 1
        self._pending_action = None
        self._clear_eligibility()
        return advantage

    def end_episode(self) -> None:
        """End an episode, clearing traces while retaining learned weights."""
        if self._episode_id is None:
            raise RuntimeError("no active episode")
        if self._pending_action is not None:
            raise RuntimeError("observe the pending action before ending")
        self._episode_id = None
        self._reset_transient()
        self._clear_eligibility()

    def reset_learning(self) -> None:
        """Restore initial readout without rewinding exploration RNG."""
        if self._episode_id is not None:
            raise RuntimeError("cannot reset learning during an episode")
        self._weights = [
            [0.0 for _ in range(HIDDEN_COUNT)]
            for _ in ACTION_ORDER
        ]
        self._bias = [0.0 for _ in ACTION_ORDER]
        self._baseline = 0.0
        self._reset_transient()
        self._clear_eligibility()
        for key in self._metrics:
            self._metrics[key] = 0

    def metrics(self) -> dict[str, int | str]:
        """Return counters and the current adaptive readout hash."""
        result: dict[str, int | str] = dict(self._metrics)
        result["encoder_mode"] = (
            "binary_lif" if self._spiking else "continuous_leaky"
        )
        result["weight_hash"] = self.weight_hash()
        return result

    def weight_hash(self) -> str:
        """Hash only plastic readout weights and bias."""
        return self._hash_readout(self._weights, self._bias)

    @staticmethod
    def _hash_readout(
        weights: list[list[float]], bias: list[float]
    ) -> str:
        """Hash a candidate readout without mutating policy state."""
        payload = json.dumps(
            {"weights": weights, "bias": bias},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def snapshot(self) -> dict[str, Any]:
        """Return JSON-safe persistent model state at an episode boundary."""
        if self._episode_id is not None or self._pending_action is not None:
            raise RuntimeError("snapshot only at an episode boundary")
        return {
            "version": SNAPSHOT_VERSION,
            "feature_encoding": FEATURE_ENCODING,
            "actions": list(ACTION_ORDER),
            "config": self._config(),
            "encoder": [row[:] for row in self._encoder],
            "weights": [row[:] for row in self._weights],
            "bias": self._bias[:],
            "baseline": self._baseline,
            "rng_state": _nested_lists(self._rng.getstate()),
            "metrics": dict(self._metrics),
            "weight_hash": self.weight_hash(),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore a compatible boundary snapshot after validating its fields.

        The supplied snapshot must match this policy's fixed architecture,
        seed, and learning mode.
        """
        if self._episode_id is not None or self._pending_action is not None:
            raise RuntimeError("restore only at an episode boundary")
        if not isinstance(snapshot, dict):
            raise ValueError("snapshot must be an object")
        if snapshot.get("version") != SNAPSHOT_VERSION:
            raise ValueError("unsupported snapshot version")
        if snapshot.get("feature_encoding") != FEATURE_ENCODING:
            raise ValueError("incompatible feature encoding")
        if snapshot.get("actions") != list(ACTION_ORDER):
            raise ValueError("incompatible action alphabet")
        if snapshot.get("config") != self._config():
            raise ValueError("snapshot configuration mismatch")
        encoder = self._matrix(snapshot.get("encoder"),
                               HIDDEN_COUNT, FEATURE_COUNT, "encoder")
        if any(value not in (-1.0, 0.0, 1.0)
               for row in encoder for value in row):
            raise ValueError("invalid fixed encoder weights")
        if encoder != self._initial_encoder:
            raise ValueError("snapshot encoder differs from seeded encoder")
        weights = self._matrix(snapshot.get("weights"),
                               len(ACTION_ORDER), HIDDEN_COUNT, "weights")
        bias = self._vector(snapshot.get("bias"), len(ACTION_ORDER), "bias")
        if any(abs(value) > MAX_WEIGHT for row in weights for value in row):
            raise ValueError("readout weight exceeds configured bound")
        if any(abs(value) > MAX_WEIGHT for value in bias):
            raise ValueError("readout bias exceeds configured bound")
        baseline = snapshot.get("baseline")
        if not _finite(baseline) or not -1.0 <= float(baseline) <= 1.0:
            raise ValueError("invalid reward baseline")
        rng_state = _nested_tuples(snapshot.get("rng_state"))
        try:
            probe = random.Random()
            probe.setstate(rng_state)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid RNG state") from exc
        metrics = snapshot.get("metrics")
        if not isinstance(metrics, dict) or set(metrics) != set(self._metrics):
            raise ValueError("invalid metrics snapshot")
        if any(type(value) is not int or value < 0
               for value in metrics.values()):
            raise ValueError("invalid metric counter")
        if snapshot.get("weight_hash") != self._hash_readout(weights, bias):
            raise ValueError("snapshot readout hash mismatch")
        self._encoder = encoder
        self._weights = weights
        self._bias = bias
        self._baseline = float(baseline)
        self._rng.setstate(rng_state)
        self._metrics = dict(metrics)
        self._reset_transient()
        self._clear_eligibility()

    def _config(self) -> dict[str, Any]:
        """Return the immutable encoder and learning configuration."""
        return {
            "seed": self._seed,
            "learning": self._learning,
            "spiking": self._spiking,
            "lr": self._lr,
            "eligibility_decay": self._eligibility_decay,
            "feature_count": FEATURE_COUNT,
            "hidden_count": HIDDEN_COUNT,
            "steps": STEPS,
            "beta": BETA,
            "threshold": THRESHOLD,
            "baseline_rate": BASELINE_RATE,
            "max_update": MAX_UPDATE,
            "max_weight": MAX_WEIGHT,
            "temperature": TEMPERATURE,
        }

    @staticmethod
    def _validate_features(features: tuple[float, ...]) -> list[float]:
        """Validate the fixed context vector and return normalized floats."""
        if not isinstance(features, tuple) or len(features) != FEATURE_COUNT:
            raise ValueError("features must be a six-value tuple")
        if any(not _finite(value) for value in features):
            raise ValueError("features must be finite real numbers")
        values = [float(value) for value in features]
        if values[0] != 1.0:
            raise ValueError("feature bias must be 1")
        if any(value not in (0.0, 1.0) for value in values[1:4]):
            raise ValueError("previous-action features must be binary")
        if sum(values[1:4]) not in (0.0, 1.0):
            raise ValueError("previous action must be one-hot or all-zero")
        if not 0.0 <= values[4] <= 1.0:
            raise ValueError("previous train error must be in [0, 1]")
        if not 0.0 <= values[5] <= 1.0:
            raise ValueError("progress must be in [0, 1]")
        return values

    @staticmethod
    def _validate_available(available: tuple[str, ...]) -> tuple[int, ...]:
        """Validate a nonempty, duplicate-free subset of legal actions."""
        if not isinstance(available, tuple) or not available:
            raise ValueError("available must be a non-empty action tuple")
        if any(
            not isinstance(action, str) or action not in ACTION_ORDER
            for action in available
        ):
            raise ValueError("unknown available action")
        if len(set(available)) != len(available):
            raise ValueError("available actions must be unique")
        allowed = set(available)
        return tuple(i for i, action in enumerate(ACTION_ORDER)
                     if action in allowed)

    def _encode(self, features: list[float]) -> tuple[list[float], int]:
        """Run the fixed binary LIF encoder for one static context input."""
        spike_counts = [0 for _ in range(HIDDEN_COUNT)]
        total_spikes = 0
        for _ in range(STEPS):
            for neuron in range(HIDDEN_COUNT):
                current = sum(
                    weight * value
                    for weight, value in zip(self._encoder[neuron], features)
                )
                membrane = BETA * self._membrane[neuron] + current
                if self._spiking and membrane >= THRESHOLD:
                    spike = 1
                    membrane = 0.0
                else:
                    spike = 0
                self._membrane[neuron] = membrane
                if self._spiking:
                    spike_counts[neuron] += spike
                else:
                    spike_counts[neuron] += math.tanh(membrane)
                total_spikes += spike
        return ([count / STEPS for count in spike_counts], total_spikes)

    @staticmethod
    def _masked_softmax(
        logits: list[float], actions: tuple[int, ...]
    ) -> dict[int, float]:
        """Compute stable softmax probabilities over only legal actions."""
        scaled = {i: logits[i] / TEMPERATURE for i in actions}
        highest = max(scaled.values())
        exp_values = {i: math.exp(value - highest)
                      for i, value in scaled.items()}
        total = sum(exp_values.values())
        return {i: value / total for i, value in exp_values.items()}

    def _sample(
        self, actions: tuple[int, ...], probabilities: dict[int, float]
    ) -> int:
        """Sample from the legal masked policy with the seeded RNG."""
        draw = self._rng.random()
        cumulative = 0.0
        for action in actions:
            cumulative += probabilities[action]
            if draw < cumulative:
                return action
        return actions[-1]

    def _accumulate_eligibility(
        self,
        picked: int,
        rates: list[float],
        probabilities: dict[int, float],
        actions: tuple[int, ...],
    ) -> None:
        """Store the current masked policy-gradient eligibility only."""
        self._clear_eligibility()
        for action in actions:
            advantage = (1.0 if action == picked else 0.0)
            advantage -= probabilities[action]
            for hidden, rate in enumerate(rates):
                self._eligibility_weights[action][hidden] += (
                    advantage * rate
                )
            self._eligibility_bias[action] += advantage

    @staticmethod
    def _clip_update(value: float) -> float:
        """Limit each plastic synapse update to a fixed safe interval."""
        return max(-MAX_UPDATE, min(MAX_UPDATE, value))

    @staticmethod
    def _matrix(
        value: object, rows: int, columns: int, label: str
    ) -> list[list[float]]:
        """Validate and copy a finite rectangular numeric matrix."""
        if not isinstance(value, list) or len(value) != rows:
            raise ValueError(f"invalid {label} matrix shape")
        result = []
        for row in value:
            if not isinstance(row, list) or len(row) != columns:
                raise ValueError(f"invalid {label} matrix shape")
            if any(not _finite(item) for item in row):
                raise ValueError(f"non-finite {label} matrix value")
            result.append([float(item) for item in row])
        return result

    @staticmethod
    def _vector(value: object, length: int, label: str) -> list[float]:
        """Validate and copy a finite numeric vector."""
        if not isinstance(value, list) or len(value) != length:
            raise ValueError(f"invalid {label} vector shape")
        if any(not _finite(item) for item in value):
            raise ValueError(f"non-finite {label} value")
        return [float(item) for item in value]

    def _clear_eligibility(self) -> None:
        """Clear the three-factor policy-gradient traces."""
        for row in self._eligibility_weights:
            for index in range(len(row)):
                row[index] = 0.0
        for index in range(len(self._eligibility_bias)):
            self._eligibility_bias[index] = 0.0

    def _reset_transient(self) -> None:
        """Reset episode-local membrane state."""
        self._membrane = [0.0 for _ in range(HIDDEN_COUNT)]
