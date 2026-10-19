"""Six-band learner with per-band bounded replay memory."""
from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

from rsi_programmer.audio_learning import BAND_COUNT, AudioLearner
from rsi_programmer.sensing_memory import MemorySensingPredictor


class AudioMemoryLearner(AudioLearner):
    """Matched six-band bank with replay over bounded training examples."""

    def __init__(self, seed: int = 0, spiking: bool = True,
                 learning: bool = True,
                 encoder_learning: bool = True) -> None:
        super().__init__(seed=seed, spiking=spiking, learning=learning,
                         encoder_learning=encoder_learning)
        self._bands = [MemorySensingPredictor(
            seed=seed + band, spiking=spiking, learning=learning,
            encoder_learning=encoder_learning) for band in range(BAND_COUNT)]
        self._memory_mode = "disabled"

    def configure_memory(self, mode: str,
                         rows: Sequence[Sequence[float]] | None = None
                         ) -> None:
        """Configure each band's independent 150-sample memory bank."""
        if self._active:
            raise RuntimeError("memory configuration requires a boundary")
        if mode not in ("fit", "recent", "disabled"):
            raise ValueError("memory mode must be fit, recent, or disabled")
        if mode == "fit":
            if (not isinstance(rows, Sequence)
                    or isinstance(rows, (str, bytes))
                    or len(rows) != 150):
                raise ValueError("fit memory requires 150 six-band rows")
            validated: list[list[float]] = []
            for row in rows:
                if (not isinstance(row, Sequence)
                        or isinstance(row, (str, bytes))):
                    raise ValueError("fit memory rows must contain six bands")
                validated.append(AudioLearner._validate_features(list(row)))
            band_rows = [[row[band] for row in validated]
                         for band in range(BAND_COUNT)]
        else:
            if rows is not None:
                raise ValueError("rows are only valid for fit memory")
            band_rows = [None] * BAND_COUNT
        for predictor, values in zip(self._bands, band_rows):
            predictor.configure_memory(mode, values)
        self._memory_mode = mode

    def disable_memory(self) -> None:
        """Disable replay in all bands, for independent calibration copies."""
        for predictor in self._bands:
            predictor.disable_memory()
        self._memory_mode = "disabled"

    def evaluation_copy(self) -> AudioMemoryLearner:
        """Return a frozen bank with replay and all memory writes disabled."""
        result = copy.deepcopy(self)
        if result._active:
            result.end_session()
        result._learning = False
        result._encoder_learning = False
        result._bands = [predictor.evaluation_copy()
                         for predictor in result._bands]
        result._memory_mode = "disabled"
        return result

    def metrics(self) -> dict[str, Any]:
        result = super().metrics()
        bands = result["bands"]
        result["replay_passes"] = sum(
            int(item["replay_passes"]) for item in bands)
        result["replay_targets"] = sum(
            int(item["replay_targets"]) for item in bands)
        result["replay_wall_seconds"] = sum(
            float(item["replay_wall_seconds"]) for item in bands)
        result["memory_capacity_frames"] = 150
        result["memory_mode"] = self._memory_mode
        result["online_training_targets"] = max(
            (int(item["online_training_targets"]) for item in bands),
            default=0,
        )
        return result
