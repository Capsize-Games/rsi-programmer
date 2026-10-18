"""Observed-row context extension for the online spiking policy."""
from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any

from rsi_programmer.online_spiking import (
    BETA,
    FEATURE_COUNT,
    HIDDEN_COUNT,
    STEPS,
    THRESHOLD,
    OnlineSpikingPolicy,
)

OBSERVED_ROWS = 6
CONTEXT_COUNT = OBSERVED_ROWS * 3
CONTEXT_SCALE = 18.0
CONTEXT_VERSION = 1
PROJECTION_SEED_PREFIX = "online-observation-context-v1:"


class ObservedContextPolicy(OnlineSpikingPolicy):
    """Condition the existing plastic readout on observed rows only.

    A caller must provide exactly six current ``(x, y, target)`` rows before
    each episode. The context projection is fixed and separate from the
    inherited action RNG and six-feature projection.
    """

    def __init__(
        self,
        seed: int = 0,
        learning: bool = True,
        spiking: bool = True,
        lr: float = 0.05,
        eligibility_decay: float = 0.0,
    ) -> None:
        """Initialize base policy and fixed context projection."""
        super().__init__(seed=seed, learning=learning, spiking=spiking,
                         lr=lr, eligibility_decay=eligibility_decay)
        projection_rng = random.Random(f"{PROJECTION_SEED_PREFIX}{seed}")
        self._context_projection = [
            [projection_rng.choice((-1, 0, 1))
             for _ in range(HIDDEN_COUNT)]
            for _ in range(CONTEXT_COUNT)
        ]
        self._context_projection_hash = self._projection_hash(
            self._context_projection)
        self._pending_context: list[float] | None = None
        self._active_context: list[float] | None = None
        self._context_current: list[float] | None = None
        self._metrics.update({
            "observation_scalars": 0,
            "projection_mac": 0,
            "context_adds": 0,
        })

    def set_observed_rows(
        self, rows: tuple[tuple[int, int, int], ...]
    ) -> None:
        """Set the six observed triples for the next episode only."""
        if self._episode_id is not None:
            raise RuntimeError("cannot replace context during an episode")
        if self._pending_context is not None:
            raise RuntimeError("context is already waiting for an episode")
        if not isinstance(rows, tuple) or len(rows) != OBSERVED_ROWS:
            raise ValueError("observed context requires exactly six rows")
        if any(not isinstance(row, tuple) or len(row) != 3
               or any(type(value) is not int for value in row)
               for row in rows):
            raise ValueError("observed rows must be integer triples")
        context = [value / (1 + abs(value))
                   for row in rows for value in row]
        current = [sum(self._context_projection[index][neuron] * value
                       for index, value in enumerate(context))
                   / CONTEXT_SCALE for neuron in range(HIDDEN_COUNT)]
        self._pending_context = context
        self._context_current = current
        self._metrics["observation_scalars"] += len(context)
        self._metrics["projection_mac"] += CONTEXT_COUNT * HIDDEN_COUNT

    def start_episode(self, episode_id: str) -> None:
        """Require fresh observed context and activate it for one episode."""
        if self._pending_context is None or self._context_current is None:
            raise RuntimeError("set observed rows before each episode")
        context, current = self._pending_context, self._context_current
        self._pending_context = None
        try:
            super().start_episode(episode_id)
        except Exception:
            self._pending_context = context
            raise
        self._active_context = context
        self._context_current = current

    def choose(
        self,
        features: tuple[float, ...],
        available: tuple[str, ...],
    ) -> str:
        """Choose through the base policy and count dense context adds."""
        action = super().choose(features, available)
        self._metrics["context_adds"] += STEPS * HIDDEN_COUNT
        return action

    def end_episode(self) -> None:
        """End the base episode and erase its observed-row context."""
        super().end_episode()
        self._active_context = None
        self._context_current = None

    def reset_learning(self) -> None:
        """Reset learned state and any not-yet-consumed observed context."""
        super().reset_learning()
        self._pending_context = None
        self._active_context = None
        self._context_current = None

    def snapshot(self) -> dict[str, Any]:
        """Keep context payloads out of boundary model snapshots."""
        if self._pending_context is not None:
            raise RuntimeError("snapshot with unconsumed context is invalid")
        return super().snapshot()

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore only at a boundary with no pending observed rows."""
        if self._pending_context is not None:
            raise RuntimeError("restore with unconsumed context is invalid")
        super().restore(snapshot)

    def _encode(self, features: list[float]) -> tuple[list[float], int]:
        """Use the base LIF/continuous dynamics plus fixed context current."""
        context = (self._active_context if self._active_context is not None
                   else self._pending_context)
        if context is None or self._context_current is None:
            raise RuntimeError("observed context is required for encoding")
        if len(features) != FEATURE_COUNT:
            raise ValueError("base feature vector has invalid width")
        spike_counts = [0.0 for _ in range(HIDDEN_COUNT)]
        total_spikes = 0
        for _ in range(STEPS):
            for neuron in range(HIDDEN_COUNT):
                base_current = sum(
                    weight * value
                    for weight, value in zip(
                        self._encoder[neuron], features)
                )
                current = base_current + self._context_current[neuron]
                membrane = self._membrane[neuron] * BETA + current
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

    def _config(self) -> dict[str, Any]:
        """Bind snapshots to this fixed observed-context representation."""
        result = super()._config()
        result.update({
            "observed_context_version": CONTEXT_VERSION,
            "observed_rows": OBSERVED_ROWS,
            "observed_context_dimensions": CONTEXT_COUNT,
            "observed_context_scale": CONTEXT_SCALE,
            "observed_context_transform": "v/(1+abs(v)); row-major x,y,target",
            "observed_context_projection_seed": PROJECTION_SEED_PREFIX,
            "observed_context_projection_sha256": (
                self._context_projection_hash),
        })
        return result

    @staticmethod
    def _projection_hash(projection: list[list[int]]) -> str:
        """Hash the fixed context projection without exporting its matrix."""
        payload = json.dumps(
            projection, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
