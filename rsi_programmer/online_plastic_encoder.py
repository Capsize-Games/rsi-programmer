"""Reward-modulated plastic encoders for the observed-context policy."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from typing import Any

from rsi_programmer.online_observed_context import (
    CONTEXT_COUNT,
    CONTEXT_SCALE,
    ObservedContextPolicy,
)
from rsi_programmer.online_spiking import (
    ACTION_ORDER,
    BETA,
    FEATURE_COUNT,
    HIDDEN_COUNT,
    MAX_UPDATE,
    MAX_WEIGHT,
    STEPS,
    THRESHOLD,
    _finite,
    _nested_lists,
    _nested_tuples,
)

ENCODER_VERSION = 1
ENCODER_FEATURE_ENCODING = "online-plastic-encoder-context-v1"
ENCODER_LR = 0.01
SURROGATE_SCALE = 0.3


class OnlinePlasticEncoderPolicy(ObservedContextPolicy):
    """Learn both input projections from each action's observed outcome.

    The hidden encoder is a small experiment using a local surrogate for
    threshold crossings. Eligibility is reset at every decision boundary;
    it is not a full-network gradient or a delayed-credit mechanism.
    """

    def __init__(
        self,
        seed: int = 0,
        learning: bool = True,
        spiking: bool = True,
        lr: float = 0.05,
        eligibility_decay: float = 0.0,
        encoder_learning: bool = True,
        encoder_lr: float = ENCODER_LR,
    ) -> None:
        """Initialize inherited encoders without consuming extra RNG draws."""
        if type(encoder_learning) is not bool:
            raise ValueError("encoder_learning must be bool")
        if not _finite(encoder_lr) or float(encoder_lr) <= 0.0:
            raise ValueError("encoder_lr must be finite and positive")
        super().__init__(seed=seed, learning=learning, spiking=spiking,
                         lr=lr, eligibility_decay=eligibility_decay)
        self._encoder_learning = encoder_learning
        self._encoder_lr = float(encoder_lr)
        self._initial_context_projection = [
            row[:] for row in self._context_projection
        ]
        self._initial_encoder_hash = self._matrix_hash(self._encoder)
        self._initial_context_hash = self._matrix_hash(
            self._context_projection)
        self._last_rate_derivative = [0.0] * HIDDEN_COUNT
        self._last_rates = [0.0] * HIDDEN_COUNT
        self._pending_hidden_signal: list[float] | None = None
        self._last_features = [0.0] * FEATURE_COUNT
        self._metrics.update({
            "plastic_context_projection_macs": 0,
            "plastic_context_projection_recomputes": 0,
            "plastic_feedback_readout_macs": 0,
            "plastic_feedback_signal_terms": 0,
            "plastic_encoder_updates": 0,
            "plastic_eligibility_neuron_ticks": 0,
            "plastic_history_update_synapse_visits": 0,
            "plastic_context_update_synapse_visits": 0,
            "plastic_encoder_changed_synapses": 0,
        })

    def choose(
        self,
        features: tuple[float, ...],
        available: tuple[str, ...],
    ) -> str:
        """Choose normally and retain the pre-update hidden action signal."""
        if not self._encoder_learning or not self._learning:
            self._pending_hidden_signal = None
            return super().choose(features, available)
        weights = [row[:] for row in self._weights]
        action_name = super().choose(features, available)
        picked = ACTION_ORDER.index(action_name)
        actions = self._validate_available(available)
        logits = [
            sum(weight * rate for weight, rate in zip(row, self._last_rates))
            + self._bias[index]
            for index, row in enumerate(weights)
        ]
        probabilities = self._masked_softmax(logits, actions)
        self._pending_hidden_signal = [
            sum(
                weights[action][hidden]
                * ((1.0 if action == picked else 0.0)
                   - probabilities.get(action, 0.0))
                for action in actions
            )
            for hidden in range(HIDDEN_COUNT)
        ]
        self._metrics["plastic_feedback_readout_macs"] += (
            len(ACTION_ORDER) * HIDDEN_COUNT
        )
        self._metrics["plastic_feedback_signal_terms"] += (
            len(actions) * HIDDEN_COUNT
        )
        return action_name

    def observe(self, reward: float) -> float:
        """Apply inherited readout learning, then local encoder updates."""
        if self._encoder_learning and self._learning \
                and self._pending_hidden_signal is None:
            raise RuntimeError("encoder has no pending decision signal")
        advantage = super().observe(reward)
        signal = self._pending_hidden_signal
        self._pending_hidden_signal = None
        if self._encoder_learning and self._learning \
                and signal is not None:
            changed = self._update_projection(
                self._encoder, signal, self._last_rate_derivative,
                self._last_features, 1.0, advantage, transposed=False,
            )
            context = self._active_context
            if context is None:
                raise RuntimeError("observed context ended before feedback")
            changed += self._update_projection(
                self._context_projection, signal,
                self._last_rate_derivative, context, CONTEXT_SCALE,
                advantage, transposed=True,
            )
            self._metrics["plastic_encoder_updates"] += 1
            self._metrics["plastic_history_update_synapse_visits"] += (
                HIDDEN_COUNT * FEATURE_COUNT
            )
            self._metrics["plastic_context_update_synapse_visits"] += (
                HIDDEN_COUNT * CONTEXT_COUNT
            )
            self._metrics["plastic_encoder_changed_synapses"] += changed
            self._context_projection_hash = self._matrix_hash(
                self._context_projection)
        self._last_rate_derivative = [0.0] * HIDDEN_COUNT
        return advantage

    def end_episode(self) -> None:
        """End the episode and clear decision-local eligibility state."""
        super().end_episode()
        self._clear_plastic_pending()

    def start_episode(self, episode_id: str) -> None:
        """Start a fresh episode with no eligibility carried across tasks."""
        super().start_episode(episode_id)
        self._clear_plastic_pending()

    def reset_learning(self) -> None:
        """Restore both seeded projections and inherited readout state."""
        super().reset_learning()
        self._encoder = [row[:] for row in self._initial_encoder]
        self._context_projection = [
            row[:] for row in self._initial_context_projection
        ]
        self._context_projection_hash = self._matrix_hash(
            self._context_projection)
        self._clear_plastic_pending()

    def evaluation_copy(self) -> OnlinePlasticEncoderPolicy:
        """Return an isolated, read-only copy of learned parameters."""
        if (self._episode_id is not None
                or self._pending_action is not None
                or self._pending_context is not None):
            raise RuntimeError("evaluation copy requires an episode boundary")
        result = copy.deepcopy(self)
        result._learning = False
        result._encoder_learning = False
        result._clear_plastic_pending()
        result._reset_transient()
        result._clear_eligibility()
        return result

    def encoder_hash(self) -> str:
        """Hash both learned input projection matrices."""
        return self._hash_payload({
            "encoder": self._encoder,
            "context_projection": self._context_projection,
        })

    def weight_hash(self) -> str:
        """Hash both projections and the inherited action readout."""
        return self._hash_payload({
            "encoder": self._encoder,
            "context_projection": self._context_projection,
            "weights": self._weights,
            "bias": self._bias,
        })

    def metrics(self) -> dict[str, int | str]:
        """Return inherited counters plus hashes for both learned layers."""
        result = super().metrics()
        result["encoder_hash"] = self.encoder_hash()
        return result

    def snapshot(self) -> dict[str, Any]:
        """Serialize learned projections only at an episode boundary."""
        if (self._episode_id is not None
                or self._pending_action is not None
                or self._pending_context is not None):
            raise RuntimeError("snapshot requires an episode boundary")
        payload: dict[str, Any] = {
            "version": ENCODER_VERSION + 1,
            "feature_encoding": ENCODER_FEATURE_ENCODING,
            "actions": list(ACTION_ORDER),
            "config": self._config(),
            "encoder": [row[:] for row in self._encoder],
            "context_projection": [
                row[:] for row in self._context_projection
            ],
            "weights": [row[:] for row in self._weights],
            "bias": self._bias[:],
            "baseline": self._baseline,
            "rng_state": _nested_lists(self._rng.getstate()),
            "metrics": dict(self._metrics),
            "readout_hash": self._hash_readout(self._weights, self._bias),
            "encoder_hash": self.encoder_hash(),
            "weight_hash": self.weight_hash(),
        }
        return payload

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Atomically restore a compatible boundary snapshot."""
        if (self._episode_id is not None
                or self._pending_action is not None
                or self._pending_context is not None):
            raise RuntimeError("restore requires an episode boundary")
        if not isinstance(snapshot, dict):
            raise ValueError("snapshot must be an object")
        if snapshot.get("version") != ENCODER_VERSION + 1:
            raise ValueError("unsupported snapshot version")
        if snapshot.get("feature_encoding") != ENCODER_FEATURE_ENCODING:
            raise ValueError("incompatible feature encoding")
        if snapshot.get("actions") != list(ACTION_ORDER):
            raise ValueError("incompatible actions")
        encoder = self._matrix(snapshot.get("encoder"), HIDDEN_COUNT,
                               FEATURE_COUNT, "encoder")
        projection = self._matrix(snapshot.get("context_projection"),
                                  CONTEXT_COUNT, HIDDEN_COUNT,
                                  "context projection")
        if any(abs(value) > MAX_WEIGHT
               for matrix in (encoder, projection)
               for row in matrix for value in row):
            raise ValueError("encoder weight exceeds configured bound")
        weights = self._matrix(snapshot.get("weights"), len(ACTION_ORDER),
                               HIDDEN_COUNT, "weights")
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
            random.Random().setstate(rng_state)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid RNG state") from exc
        metrics = snapshot.get("metrics")
        if not isinstance(metrics, dict) or set(metrics) != set(self._metrics):
            raise ValueError("invalid metrics snapshot")
        if any(type(value) is not int or value < 0
               for value in metrics.values()):
            raise ValueError("invalid metric counter")
        if snapshot.get("readout_hash") != self._hash_readout(weights, bias):
            raise ValueError("snapshot readout hash mismatch")
        if snapshot.get("encoder_hash") != self._hash_payload({
                "encoder": encoder,
                "context_projection": projection,
        }):
            raise ValueError("snapshot encoder hash mismatch")
        full_hash = self._hash_payload({
            "encoder": encoder,
            "context_projection": projection,
            "weights": weights,
            "bias": bias,
        })
        if snapshot.get("weight_hash") != full_hash:
            raise ValueError("snapshot weight hash mismatch")
        candidate = copy.deepcopy(self)
        candidate._encoder = encoder
        candidate._context_projection = projection
        candidate._context_projection_hash = candidate._matrix_hash(
            projection)
        candidate._weights = weights
        candidate._bias = bias
        candidate._baseline = float(baseline)
        candidate._rng.setstate(rng_state)
        candidate._metrics = dict(metrics)
        candidate._reset_transient()
        candidate._clear_eligibility()
        candidate._clear_plastic_pending()
        if snapshot.get("config") != candidate._config():
            raise ValueError("snapshot configuration mismatch")
        self.__dict__ = candidate.__dict__

    def _encode(self, features: list[float]) -> tuple[list[float], int]:
        """Encode one decision and retain local current-decision gradients."""
        if not self._encoder_learning or not self._learning:
            rates, spikes = super()._encode(features)
            self._last_rates = rates[:]
            self._last_rate_derivative = [0.0] * HIDDEN_COUNT
            self._last_features = features[:]
            return rates, spikes
        context = (self._active_context if self._active_context is not None
                   else self._pending_context)
        if context is None:
            raise RuntimeError("observed context is required for encoding")
        self._context_current = [
            sum(self._context_projection[index][neuron] * value
                for index, value in enumerate(context)) / CONTEXT_SCALE
            for neuron in range(HIDDEN_COUNT)
        ]
        self._metrics["plastic_context_projection_macs"] += (
            CONTEXT_COUNT * HIDDEN_COUNT
        )
        self._metrics["plastic_context_projection_recomputes"] += 1
        rate_sums = [0.0] * HIDDEN_COUNT
        derivative_sums = [0.0] * HIDDEN_COUNT
        total_spikes = 0
        for neuron in range(HIDDEN_COUNT):
            derivative_membrane = 0.0
            for _ in range(STEPS):
                base_current = sum(
                    weight * value for weight, value in zip(
                        self._encoder[neuron], features
                    )
                )
                current = base_current + self._context_current[neuron]
                membrane = BETA * self._membrane[neuron] + current
                derivative_current = BETA * derivative_membrane + 1.0
                if self._spiking:
                    spike = int(membrane >= THRESHOLD)
                    psi = SURROGATE_SCALE * max(
                        0.0, 1.0 - abs(membrane - THRESHOLD)
                    )
                    derivative_spike = psi * derivative_current
                    derivative_membrane = (
                        (1 - spike) * derivative_current
                        - membrane * derivative_spike
                    )
                    self._membrane[neuron] = (
                        0.0 if spike else membrane
                    )
                    rate_sums[neuron] += spike
                    derivative_sums[neuron] += derivative_spike
                    total_spikes += spike
                else:
                    rate = math.tanh(membrane)
                    derivative_spike = (1.0 - rate * rate) \
                        * derivative_current
                    derivative_membrane = derivative_current
                    self._membrane[neuron] = membrane
                    rate_sums[neuron] += rate
                    derivative_sums[neuron] += derivative_spike
        rates = [value / STEPS for value in rate_sums]
        rate_derivative = [value / STEPS for value in derivative_sums]
        self._metrics["plastic_eligibility_neuron_ticks"] += (
            STEPS * HIDDEN_COUNT
        )
        self._last_features = features[:]
        self._last_rates = rates[:]
        self._last_rate_derivative = rate_derivative
        return rates, total_spikes

    def _config(self) -> dict[str, Any]:
        """Bind snapshots to plasticity flags, limits and projections."""
        result = super()._config()
        result.update({
            "plastic_encoder_version": ENCODER_VERSION,
            "encoder_learning": self._encoder_learning,
            "encoder_lr": self._encoder_lr,
            "encoder_max_update": MAX_UPDATE,
            "encoder_max_weight": MAX_WEIGHT,
            "encoder_surrogate_scale": SURROGATE_SCALE,
            "encoder_surrogate": "triangular threshold surrogate",
            "encoder_projection_dimensions": [HIDDEN_COUNT, FEATURE_COUNT],
            "context_projection_dimensions": [CONTEXT_COUNT, HIDDEN_COUNT],
            "encoder_initial_sha256": self._initial_encoder_hash,
            "context_initial_sha256": self._initial_context_hash,
        })
        return result

    def _clear_plastic_pending(self) -> None:
        """Clear eligibility and hidden credit at task/decision boundaries."""
        self._last_rate_derivative = [0.0] * HIDDEN_COUNT
        self._last_rates = [0.0] * HIDDEN_COUNT
        self._pending_hidden_signal = None
        self._last_features = [0.0] * FEATURE_COUNT

    def _update_projection(
        self,
        projection: list[list[float]],
        signal: list[float],
        derivative: list[float],
        inputs: list[float],
        scale: float,
        advantage: float,
        transposed: bool,
    ) -> int:
        """Apply bounded reward-modulated updates to one input projection."""
        changed = 0
        for neuron in range(HIDDEN_COUNT):
            hidden_signal = signal[neuron] * derivative[neuron]
            for index, value in enumerate(inputs):
                delta = self._clip_update(
                    self._encoder_lr * advantage * hidden_signal
                    * value / scale
                )
                if delta:
                    row, column = ((index, neuron) if transposed
                                   else (neuron, index))
                    projection[row][column] = max(
                        -MAX_WEIGHT,
                        min(MAX_WEIGHT, projection[row][column] + delta),
                    )
                    changed += 1
        return changed

    @staticmethod
    def _matrix_hash(matrix: list[list[float]]) -> str:
        """Hash a projection matrix in canonical JSON form."""
        payload = json.dumps(
            OnlinePlasticEncoderPolicy._canonical_json_value(matrix),
            separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _hash_payload(payload: dict[str, Any]) -> str:
        """Hash canonical JSON policy state for persistent snapshots."""
        encoded = json.dumps(
            OnlinePlasticEncoderPolicy._canonical_json_value(payload),
            sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _canonical_json_value(value: Any) -> Any:
        """Normalize integral floats for stable matrix snapshot hashes."""
        if isinstance(value, dict):
            return {
                key: OnlinePlasticEncoderPolicy._canonical_json_value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [OnlinePlasticEncoderPolicy._canonical_json_value(item)
                    for item in value]
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value
