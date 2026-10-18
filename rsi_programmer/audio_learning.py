"""Six-band online learning for normalized fan audio features."""
from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from typing import Any

from rsi_programmer.sensing_predictor import SensingPredictor

BAND_EDGES_HZ = (80, 160, 320, 640, 1280, 2560, 6000)
BAND_COUNT = len(BAND_EDGES_HZ) - 1
FRAME_SECONDS = 0.2


class AudioLearner:
    """A matched bank of causal scalar predictors, one per frequency band."""

    def __init__(self, seed: int = 0, spiking: bool = True,
                 learning: bool = True,
                 encoder_learning: bool = True) -> None:
        """Create six predictors with deterministic, band-specific seeds."""
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self._seed = seed
        self._spiking = spiking
        self._learning = learning
        self._encoder_learning = encoder_learning
        self._bands = [SensingPredictor(
            seed=seed + band, spiking=spiking, learning=learning,
            encoder_learning=encoder_learning) for band in range(BAND_COUNT)]
        self._active = False
        self._last_timestamp: float | None = None

    @property
    def active(self) -> bool:
        """Whether a feature sequence is currently being processed."""
        return self._active

    def start_session(self, timestamp: float,
                      features: Sequence[float]) -> list[float]:
        """Consume the first six-band frame and emit one forecast per band."""
        if self._active:
            raise RuntimeError("a session is already active")
        ts = self._validate_timestamp(timestamp)
        values = self._validate_features(features)
        forecasts = [model.start_session(ts, value)
                     for model, value in zip(self._bands, values)]
        self._active = True
        self._last_timestamp = ts
        return forecasts

    def advance(self, timestamp: float,
                features: Sequence[float]) -> list[float]:
        """Score prior forecasts, then emit forecasts from this frame."""
        if not self._active or self._last_timestamp is None:
            raise RuntimeError("start_session must be called before advance")
        ts = self._validate_timestamp(timestamp)
        values = self._validate_features(features)
        if ts <= self._last_timestamp:
            raise ValueError("timestamps must increase within a session")
        if not math.isfinite(ts - self._last_timestamp):
            raise ValueError("time delta must be finite")
        forecasts = [model.advance(ts, value)
                     for model, value in zip(self._bands, values)]
        self._last_timestamp = ts
        return forecasts

    def end_session(self) -> None:
        """End all six sessions together, discarding final forecasts."""
        if not self._active:
            raise RuntimeError("no active session")
        for model in self._bands:
            model.end_session()
        self._active = False
        self._last_timestamp = None

    def snapshot(self) -> dict[str, Any]:
        """Return session-boundary snapshots for all six bands."""
        if self._active:
            raise RuntimeError("snapshot requires a session boundary")
        return {
            "version": 1,
            "band_edges_hz": list(BAND_EDGES_HZ),
            "seed": self._seed,
            "spiking": self._spiking,
            "learning": self._learning,
            "encoder_learning": self._encoder_learning,
            "bands": [model.snapshot() for model in self._bands],
        }

    def evaluation_copy(self) -> AudioLearner:
        """Copy a boundary learner with both learning modes disabled."""
        if self._active:
            raise RuntimeError("evaluation copy requires a session boundary")
        result = copy.deepcopy(self)
        result._learning = False
        result._encoder_learning = False
        result._bands = [model.evaluation_copy() for model in self._bands]
        return result

    def clone_for_evaluation(self) -> AudioLearner:
        """Create a frozen boundary evaluator without ending the live session.

        The copy discards its pending forecasts and transient traces. The
        active learner, including its pending credit, is left untouched.
        """
        result = copy.deepcopy(self)
        if result._active:
            result.end_session()
        return result.evaluation_copy()

    def frozen_reference(self) -> AudioLearner:
        """Create a matched, untrained bank with parameter learning disabled."""
        if self._active:
            raise RuntimeError("reference creation requires a session boundary")
        return AudioLearner(seed=self._seed, spiking=self._spiking,
                            learning=False, encoder_learning=False)

    def weight_hash(self) -> list[str]:
        """Return each band's current parameter fingerprint."""
        return [model.weight_hash() for model in self._bands]

    def metrics(self) -> dict[str, Any]:
        """Aggregate counters and expose per-band mode and hash evidence."""
        per_band = [model.metrics() for model in self._bands]
        counter_names = (
            "sessions", "predictions", "scored_targets", "updates",
            "neuron_ticks", "spikes", "readout_synapse_visits",
            "input_synapse_visits", "changed_readout_parameters",
            "changed_input_parameters",
        )
        totals = {name: sum(int(item[name]) for item in per_band)
                  for name in counter_names}
        return {
            **totals,
            "band_edges_hz": list(BAND_EDGES_HZ),
            "spiking": self._spiking,
            "learning": self._learning,
            "encoder_learning": self._encoder_learning,
            "weight_hashes": [item["weight_hash"] for item in per_band],
            "encoder_hashes": [item["encoder_hash"] for item in per_band],
            "bands": per_band,
        }

    @staticmethod
    def _validate_timestamp(timestamp: float) -> float:
        if (isinstance(timestamp, bool)
                or not isinstance(timestamp, (int, float))
                or not math.isfinite(float(timestamp))):
            raise ValueError("timestamp must be finite")
        return float(timestamp)

    @staticmethod
    def _validate_features(features: Sequence[float]) -> list[float]:
        if (not isinstance(features, Sequence)
                or isinstance(features, (str, bytes))
                or len(features) != BAND_COUNT):
            raise ValueError("features must contain six normalized values")
        values = list(features)
        if any(isinstance(value, bool)
               or not isinstance(value, (int, float))
               or not math.isfinite(float(value))
               or not 0.0 <= float(value) <= 1.0 for value in values):
            raise ValueError("features must be finite values in [0, 1]")
        return [float(value) for value in values]
