"""Controlled audio comparison with readout and replay-memory arms."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

from rsi_programmer.audio_learning import AudioLearner
from rsi_programmer.audio_memory import AudioMemoryLearner
from rsi_programmer.audio_session import AudioSession

CONTINUAL_ARMS = (
    "adaptive_snn", "readout_snn", "memory_snn", "recent_memory_snn",
    "fixed_snn", "adaptive_ann", "memory_ann", "ema", "persistence",
)


class AudioContinualSession(AudioSession):
    """Compare matched online learners with fit and recent replay memory."""

    ARM_NAMES = CONTINUAL_ARMS

    def __init__(self, protocol: dict[str, Any] | None = None) -> None:
        if protocol is None:
            fixture = (Path(__file__).resolve().parents[1]
                       / "fixtures" / "audio_continual_v1.json")
            protocol = json.loads(fixture.read_text(encoding="utf-8"))
        self._processing_seconds = 0.0
        self._initial_fit_hashes: dict[str, str] = {}
        super().__init__(protocol)
        if self.protocol.get("arms") != list(self.ARM_NAMES):
            raise ValueError("continual protocol arms do not match fixture")
        if (self.protocol.get("memory_capacity_frames") != 150
                or self.protocol.get("memory_interval_targets") != 300):
            raise ValueError("continual memory settings do not match fixture")

    def observe(self, features: dict[str, Any],
                received_seconds: float) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            return super().observe(features, received_seconds)
        finally:
            self._processing_seconds += time.perf_counter() - started

    def _build_models(self, rows: list[list[float]],
                      seed: int) -> dict[str, AudioLearner]:
        """Fit matched encoders, then configure independent memory arms."""
        adaptive_snn = self._train(
            AudioLearner(seed=seed, spiking=True), rows)
        readout_snn = self._train(
            AudioLearner(seed=seed, spiking=True), rows)
        readout_snn._encoder_learning = False
        for predictor in readout_snn._bands:
            predictor._encoder_learning = False

        memory_snn = self._train(
            AudioMemoryLearner(seed=seed, spiking=True), rows)
        memory_snn.configure_memory("fit", rows)
        recent_memory_snn = self._train(
            AudioMemoryLearner(seed=seed, spiking=True), rows)
        recent_memory_snn.configure_memory("recent")

        adaptive_ann = self._train(
            AudioLearner(seed=seed, spiking=False), rows)
        memory_ann = self._train(
            AudioMemoryLearner(seed=seed, spiking=False), rows)
        memory_ann.configure_memory("fit", rows)
        models = {
            "adaptive_snn": adaptive_snn,
            "readout_snn": readout_snn,
            "memory_snn": memory_snn,
            "recent_memory_snn": recent_memory_snn,
            "fixed_snn": adaptive_snn.evaluation_copy(),
            "adaptive_ann": adaptive_ann,
            "memory_ann": memory_ann,
        }
        self._initial_fit_hashes = {
            name: self._bank_hash(model) for name, model in models.items()
        }
        return models

    def _clone_calibration_models(self) -> dict[str, AudioLearner]:
        clones = {name: copy.deepcopy(model)
                  for name, model in self._models.items()}
        for model in clones.values():
            if isinstance(model, AudioMemoryLearner):
                model.disable_memory()
        return clones

    def _model_metrics(self) -> dict[str, Any]:
        result = super()._model_metrics()
        result["processing_wall_seconds"] = self._processing_seconds
        result["initial_fit_weight_hashes"] = dict(
            self._initial_fit_hashes)
        result["memory_capacity_frames"] = 150
        result["memory_interval_targets"] = 300
        result["memory_values_per_bank"] = 150 * 6
        result["trainable_parameters_per_model"] = 6 * 37
        result["memory_samples_per_arm"] = {
            "memory_snn": 150,
            "recent_memory_snn": 150,
            "memory_ann": 150,
        }
        return result

    def summary(self) -> dict[str, Any]:
        result = super().summary()
        result["model_metrics"] = self._model_metrics()
        result["experiment"] = "audio_continual_v1"
        return result
