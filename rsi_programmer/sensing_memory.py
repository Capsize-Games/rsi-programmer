"""Sensing predictor with isolated causal replay memory."""
from __future__ import annotations

import copy
import time
from typing import Any

from rsi_programmer.sensing_predictor import SensingPredictor


class MemorySensingPredictor(SensingPredictor):
    """Add periodic replay while preserving the live temporal state."""

    MEMORY_CAPACITY = 150
    REPLAY_INTERVAL = 300

    def __init__(self, seed: int = 0, spiking: bool = True,
                 learning: bool = True,
                 encoder_learning: bool = True) -> None:
        super().__init__(seed=seed, spiking=spiking, learning=learning,
                         encoder_learning=encoder_learning)
        self._memory_mode = "disabled"
        self._memory_rows: list[float] = []
        self._replay_enabled = False
        self._online_targets = 0
        self._replay_passes = 0
        self._replay_targets = 0
        self._replay_wall_seconds = 0.0

    def configure_memory(self, mode: str,
                         rows: list[float] | None = None) -> None:
        """Set fit-only or recent-only memory after the bootstrap fit."""
        if self._active:
            raise RuntimeError("memory configuration requires a boundary")
        if mode not in ("fit", "recent", "disabled"):
            raise ValueError("memory mode must be fit, recent, or disabled")
        if mode == "fit":
            if rows is None or len(rows) != self.MEMORY_CAPACITY:
                raise ValueError("fit memory requires exactly 150 rows")
            self._memory_rows = [self._validate_value(item) for item in rows]
        else:
            if rows is not None:
                raise ValueError("rows are only accepted for fit memory")
            self._memory_rows = []
        self._memory_mode = mode
        self._replay_enabled = mode != "disabled"
        self._online_targets = 0

    def disable_memory(self) -> None:
        """Disable replay and writes on evaluation/calibration copies."""
        self._memory_mode = "disabled"
        self._memory_rows = []
        self._replay_enabled = False

    def start_session(self, timestamp: float, value: float) -> float:
        result = super().start_session(timestamp, value)
        if (self._learning and self._memory_mode == "recent"
                and self._replay_enabled):
            self._memory_rows.append(float(value))
        return result

    def _score_pending(self, target: float) -> None:
        super()._score_pending(target)
        if (not self._learning or not self._replay_enabled
                or self._memory_mode == "disabled"):
            return
        self._online_targets += 1
        if self._memory_mode == "recent":
            self._memory_rows.append(float(target))
            self._memory_rows = self._memory_rows[-self.MEMORY_CAPACITY:]
        if (self._online_targets % self.REPLAY_INTERVAL == 0
                and len(self._memory_rows) == self.MEMORY_CAPACITY):
            self._replay()

    def _replay(self) -> None:
        """Train on memory in a transient copy and copy parameters only."""
        started = time.perf_counter()
        source_metrics = dict(self._metrics)
        trainer = copy.deepcopy(self)
        trainer.end_session()
        trainer._replay_enabled = False
        trainer._memory_mode = "disabled"
        trainer._memory_rows = []
        trainer._learning = self._learning
        trainer._encoder_learning = self._encoder_learning
        values = [self._validate_value(item) for item in self._memory_rows]
        trainer.start_session(0.2, values[0])
        for position, value in enumerate(values[1:], 1):
            trainer.advance((position + 1) * 0.2, value)
        trainer.end_session()
        replay_metrics = dict(trainer._metrics)
        for key, before in source_metrics.items():
            self._metrics[key] += replay_metrics[key] - before
        self._input_weights = trainer._input_weights[:]
        self._input_biases = trainer._input_biases[:]
        self._readout_weights = trainer._readout_weights[:]
        self._output_bias = trainer._output_bias
        self._replay_passes += 1
        self._replay_targets += len(values) - 1
        self._replay_wall_seconds += time.perf_counter() - started

    def evaluation_copy(self) -> MemorySensingPredictor:
        result = super().evaluation_copy()
        result.disable_memory()
        result._online_targets = 0
        result._replay_passes = 0
        result._replay_targets = 0
        result._replay_wall_seconds = 0.0
        return result

    def metrics(self) -> dict[str, Any]:
        return {
            **super().metrics(),
            "replay_passes": self._replay_passes,
            "replay_targets": self._replay_targets,
            "replay_wall_seconds": self._replay_wall_seconds,
            "memory_capacity_frames": self.MEMORY_CAPACITY,
            "memory_mode": self._memory_mode,
            "online_training_targets": self._online_targets,
        }

    def snapshot(self) -> dict[str, Any]:
        result = super().snapshot()
        result["memory_metrics"] = {
            "replay_passes": self._replay_passes,
            "replay_targets": self._replay_targets,
            "replay_wall_seconds": self._replay_wall_seconds,
            "capacity_frames": self.MEMORY_CAPACITY,
            "mode": self._memory_mode,
            "online_training_targets": self._online_targets,
        }
        return result

    @staticmethod
    def _validate_value(value: float) -> float:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not 0.0 <= float(value) <= 1.0):
            raise ValueError("memory values must be in [0, 1]")
        return float(value)
