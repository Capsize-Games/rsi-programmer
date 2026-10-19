"""Deterministic fit, calibration, probe, and monitor audio protocol."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any

from rsi_programmer.audio_features import fit_band_normalizer, normalize_bands
from rsi_programmer.audio_learning import AudioLearner

ARMS = ("adaptive_snn", "fixed_snn", "adaptive_ann", "ema", "persistence")
DEFAULT_PROTOCOL = {
    "frame_seconds": 0.2,
    "fit_frames": 150,
    "calibration_frames": 150,
    "probe_frames": 100,
    "warmup_frames": 25,
    "cooldown_frames": 25,
    "block_frames": 300,
    "ema_alpha": 1.0 - math.exp(-0.2 / 5.0),
    "seed": 0,
}


class AudioSession:
    """Run a fixed five-arm online comparison over six normalized bands."""

    ARM_NAMES = ARMS

    def __init__(self, protocol: dict[str, Any] | None = None) -> None:
        """Create an empty session; frame counts are overridable for checks."""
        self.protocol = dict(DEFAULT_PROTOCOL)
        if protocol is not None:
            if not isinstance(protocol, dict):
                raise ValueError("protocol must be an object")
            self.protocol.update(protocol)
        self._validate_protocol()
        self._phase = "fit"
        self._index = 0
        self._fit_rows: list[dict[str, Any]] = []
        self._cal_rows: list[list[float]] = []
        self._probe_rows: list[list[float]] = []
        self._normalizer: dict[str, list[float]] | None = None
        self._models: dict[str, AudioLearner] = {}
        self._cal_models: dict[str, AudioLearner] = {}
        self._cal_work_start: dict[str, dict[str, Any]] = {}
        self._calibration_work: dict[str, dict[str, int]] = {}
        self._probe_work: dict[str, dict[str, int]] = {}
        self._cal_predictions: dict[str, list[float]] = {}
        self._thresholds: dict[str, float] = {}
        self._cal_errors: dict[str, list[float]] = {
            arm: [] for arm in self.ARM_NAMES
        }
        self._probe_result: dict[str, Any] | None = None
        self._reference_hash: str | None = None
        self._fixed_hash: str | None = None
        self._monitor_frames = 0
        self._monitor_predictions: dict[str, list[float]] = {}
        self._monitor_cooldown = {arm: 0 for arm in self.ARM_NAMES}
        self._monitor_alarms = {arm: 0 for arm in self.ARM_NAMES}
        self._block_alarms = {arm: 0 for arm in self.ARM_NAMES}
        self._monitor_sse = {arm: 0.0 for arm in self.ARM_NAMES}
        self._monitor_scored_frames = {arm: 0 for arm in self.ARM_NAMES}
        self._block_errors: dict[str, list[float]] = {
            arm: [] for arm in self.ARM_NAMES
        }
        self._blocks: list[dict[str, Any]] = []
        self._checkpoints: list[dict[str, Any]] = []
        self._last_received_seconds: float | None = None

    @property
    def ready(self) -> bool:
        """Whether the fixed probe is sealed and monitoring has begun."""
        return self._phase == "monitor"

    @property
    def phase(self) -> str:
        """Current protocol phase."""
        return self._phase

    def observe(self, features: dict[str, Any],
                received_seconds: float) -> dict[str, Any]:
        """Consume one delivered frame and return an auditable private row."""
        if self._phase == "finished":
            raise RuntimeError("session is finished")
        received = self._finite(received_seconds, "received_seconds")
        if (self._last_received_seconds is not None
                and received < self._last_received_seconds):
            raise ValueError("received_seconds must not move backwards")
        if not isinstance(features, dict):
            raise ValueError("features must be an object")
        band_dbfs = features.get("band_dbfs")
        values_db = self._validate_band_values(band_dbfs)
        index = self._index
        timestamp = (index + 1) * float(self.protocol["frame_seconds"])
        row: dict[str, Any] = {
            "i": index, "phase": self._phase, "j": self._phase_index(),
            "timestamp": timestamp, "received_seconds": received,
            "features": copy.deepcopy(features), "normalized": None,
            "arms": {},
        }
        self._last_received_seconds = received
        if self._phase == "fit":
            self._fit_rows.append({"band_dbfs": values_db})
            self._after_fit_row()
        elif self._phase == "calibration":
            normalized = self._normalize(values_db)
            row["normalized"] = normalized
            row["arms"] = self._calibration_step(timestamp, normalized)
            self._cal_rows.append(normalized)
            self._after_calibration_row()
        elif self._phase == "probe":
            normalized = self._normalize(values_db)
            row["normalized"] = normalized
            self._probe_rows.append(normalized)
            self._after_probe_row()
        else:
            normalized = self._normalize(values_db)
            row["normalized"] = normalized
            row["arms"] = self._monitor_step(timestamp, normalized)
        self._index += 1
        return row

    def finish(self) -> None:
        """End active learners while preserving their learned parameters."""
        for model in (*self._models.values(), *self._cal_models.values()):
            if model.active:
                model.end_session()
        self._phase = "finished"

    def snapshot(self) -> dict[str, Any]:
        """Copy current model parameters without disturbing active learners."""
        models: dict[str, Any] = {}
        for name, model in self._models.items():
            evaluator = model.clone_for_evaluation() if model.active else model
            models[name] = evaluator.snapshot()
        return {
            "version": 1, "phase": self._phase, "frame_index": self._index,
            "protocol": copy.deepcopy(self.protocol),
            "normalizer": copy.deepcopy(self._normalizer),
            "models": models,
            "baseline_state": copy.deepcopy(self._monitor_predictions),
            "reference_hash": self._reference_hash,
            "fixed_hash": self._fixed_hash,
        }

    def summary(self) -> dict[str, Any]:
        """Return protocol, calibration, sealed-probe, and monitor evidence."""
        cumulative = {
            arm: self._summary_from_totals(
                self._monitor_sse[arm], self._monitor_scored_frames[arm])
            for arm in self.ARM_NAMES
        }
        for arm in self.ARM_NAMES:
            cumulative[arm]["alarm_count"] = self._monitor_alarms[arm]
            cumulative[arm]["alarms_per_minute"] = (
                self._monitor_alarms[arm]
                / (self._monitor_scored_frames[arm]
                   * float(self.protocol["frame_seconds"]) / 60.0)
                if self._monitor_scored_frames[arm] else None
            )
        adaptive_vs_fixed = {
            arm: {
                "block_mse_delta": [
                    block["arms"][arm]["mse"]
                    - block["arms"]["fixed_snn"]["mse"]
                    for block in self._blocks
                    if block["arms"][arm]["mse"] is not None
                    and block["arms"]["fixed_snn"]["mse"] is not None
                ],
                "cumulative_mse_delta": (
                    cumulative[arm]["mse"] - cumulative["fixed_snn"]["mse"]
                    if cumulative[arm]["mse"] is not None
                    and cumulative["fixed_snn"]["mse"] is not None
                    else None
                ),
                "relative_improvement": (
                    (cumulative["fixed_snn"]["mse"]
                     - cumulative[arm]["mse"])
                    / cumulative["fixed_snn"]["mse"]
                    if cumulative[arm]["mse"] is not None
                    and cumulative["fixed_snn"]["mse"] not in (None, 0)
                    else None
                ),
            }
            for arm in self.ARM_NAMES if arm != "fixed_snn"
        }
        return {
            "protocol": copy.deepcopy(self.protocol),
            "phase": self._phase,
            "frame_count": self._index,
            "normalizer": copy.deepcopy(self._normalizer),
            "reference_hash": self._reference_hash,
            "fixed_hash": self._fixed_hash,
            "thresholds": dict(self._thresholds),
            "calibration": {
                arm: self._error_summary(self._cal_errors[arm])
                for arm in self.ARM_NAMES
            },
            "sealed_probe": copy.deepcopy(self._probe_result),
            "monitor_frames": self._monitor_frames,
            "monitor_alarms": dict(self._monitor_alarms),
            "observed_alarms_per_minute": {
                arm: cumulative[arm]["alarms_per_minute"]
                for arm in self.ARM_NAMES
            },
            "cumulative": cumulative,
            "blocks": copy.deepcopy(self._blocks),
            "adaptive_vs_fixed": adaptive_vs_fixed,
            "checkpoints": copy.deepcopy(self._checkpoints),
            "model_metrics": self._model_metrics(),
            "work_accounting": {
                "live_training_banks": "fit, online updates, and replay",
                "calibration_copies": copy.deepcopy(self._calibration_work),
                "sealed_probe_copies": copy.deepcopy(self._probe_work),
                "wall_seconds_scope": "all observe processing",
            },
        }

    def _build_models(self, rows: list[list[float]],
                      seed: int) -> dict[str, AudioLearner]:
        """Fit default SNN and ANN banks and create the fixed copy."""
        adaptive_snn = self._train(
            AudioLearner(seed=seed, spiking=True), rows)
        adaptive_ann = self._train(
            AudioLearner(seed=seed, spiking=False), rows)
        return {
            "adaptive_snn": adaptive_snn,
            "adaptive_ann": adaptive_ann,
            "fixed_snn": adaptive_snn.evaluation_copy(),
        }

    def _clone_calibration_models(self) -> dict[str, AudioLearner]:
        """Create independent calibration copies of each neural arm."""
        return {name: copy.deepcopy(model)
                for name, model in self._models.items()}

    def _model_metrics(self) -> dict[str, Any]:
        """Return work evidence for every live neural arm."""
        return {name: model.metrics() for name, model in self._models.items()}

    @staticmethod
    def _counter_delta(before: dict[str, Any],
                       after: dict[str, Any]) -> dict[str, int]:
        counters = (
            "sessions", "predictions", "scored_targets", "updates",
            "neuron_ticks", "spikes", "readout_synapse_visits",
            "input_synapse_visits", "changed_readout_parameters",
            "changed_input_parameters", "replay_passes", "replay_targets",
        )
        return {
            name: max(0, int(after.get(name, 0)) - int(before.get(name, 0)))
            for name in counters
        }

    @classmethod
    def _accumulate_work(cls, target: dict[str, dict[str, int]], arm: str,
                         metrics: dict[str, Any]) -> None:
        delta = cls._counter_delta({}, metrics)
        current = target.setdefault(arm, {name: 0 for name in delta})
        for name, value in delta.items():
            current[name] += value

    def _after_fit_row(self) -> None:
        if len(self._fit_rows) != int(self.protocol["fit_frames"]):
            return
        self._normalizer = fit_band_normalizer(self._fit_rows)
        normalized = [self._normalize(row["band_dbfs"])
                      for row in self._fit_rows]
        seed = int(self.protocol["seed"])
        self._models = self._build_models(normalized, seed)
        self._reference_hash = self._bank_hash(self._models["fixed_snn"])
        self._fixed_hash = self._reference_hash
        self._cal_models = self._clone_calibration_models()
        self._cal_work_start = {
            name: model.metrics()
            for name, model in self._cal_models.items()
        }
        self._phase = "calibration"

    def _calibration_step(self, timestamp: float,
                          values: list[float]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        frame_position = len(self._cal_rows)
        for arm in self.ARM_NAMES:
            if arm in self._cal_models:
                model = self._cal_models[arm]
                before = self._cal_predictions.get(arm)
                if before is None:
                    after = model.start_session(timestamp, values)
                    result[arm] = self._arm_row(None, after, None, False,
                                                self._bank_hash(model),
                                                self._bank_hash(model))
                else:
                    old_hash = self._bank_hash(model)
                    errors = self._errors(before, values)
                    score = self._score(errors)
                    scored = frame_position >= int(
                        self.protocol["warmup_frames"])
                    if scored:
                        self._cal_errors[arm].append(score)
                    after = model.advance(timestamp, values)
                    result[arm] = self._arm_row(
                        before, after, score, scored, old_hash,
                        self._bank_hash(model), errors)
                self._cal_predictions[arm] = after
            else:
                before = self._cal_predictions.get(arm)
                after = self._baseline_next(arm, before, values)
                if before is None:
                    result[arm] = self._arm_row(None, after, None, False,
                                                "baseline", "baseline")
                else:
                    errors = self._errors(before, values)
                    score = self._score(errors)
                    scored = frame_position >= int(
                        self.protocol["warmup_frames"])
                    if scored:
                        self._cal_errors[arm].append(score)
                    result[arm] = self._arm_row(before, after, score, scored,
                                                "baseline", "baseline", errors)
                self._cal_predictions[arm] = after
        return result

    def _after_calibration_row(self) -> None:
        if len(self._cal_rows) != int(self.protocol["calibration_frames"]):
            return
        for name, model in self._cal_models.items():
            self._calibration_work[name] = self._counter_delta(
                self._cal_work_start.get(name, {}), model.metrics())
            if model.active:
                model.end_session()
        for arm in self.ARM_NAMES:
            errors = self._cal_errors[arm]
            if not errors:
                raise RuntimeError("calibration produced no scored targets")
            self._thresholds[arm] = max(1e-9, max(errors))
        self._cal_models = {}
        self._cal_work_start = {}
        self._cal_predictions = {}
        self._phase = "probe"

    def _after_probe_row(self) -> None:
        if len(self._probe_rows) != int(self.protocol["probe_frames"]):
            return
        self._probe_result = self._evaluate_probe(self._probe_rows)
        self._checkpoints.append({
            "monitor_frame": 0,
            "sealed_probe": copy.deepcopy(self._probe_result),
            "fixed_hash": self._bank_hash(self._models["fixed_snn"]),
        })
        self._phase = "monitor"
        self._monitor_predictions = {}

    def _monitor_step(self, timestamp: float,
                      values: list[float]) -> dict[str, Any]:
        position = self._monitor_frames
        result: dict[str, Any] = {}
        for arm in self.ARM_NAMES:
            if arm in self._models:
                model = self._models[arm]
                before = self._monitor_predictions.get(arm)
                old_hash = self._bank_hash(model)
                if before is None:
                    after = model.start_session(timestamp, values)
                    result[arm] = self._arm_row(None, after, None, False,
                                                old_hash,
                                                self._bank_hash(model))
                else:
                    errors = self._errors(before, values)
                    score = self._score(errors)
                    scored = position >= int(self.protocol["warmup_frames"])
                    alarm = self._alarm(arm, score, scored)
                    if scored:
                        self._record_monitor_score(arm, errors)
                        self._block_errors[arm].append(score)
                    after = model.advance(timestamp, values)
                    result[arm] = self._arm_row(
                        before, after, score, scored, old_hash,
                        self._bank_hash(model), errors, alarm)
                self._monitor_predictions[arm] = after
            else:
                before = self._monitor_predictions.get(arm)
                after = self._baseline_next(arm, before, values)
                if before is None:
                    result[arm] = self._arm_row(None, after, None, False,
                                                "baseline", "baseline")
                else:
                    errors = self._errors(before, values)
                    score = self._score(errors)
                    scored = position >= int(self.protocol["warmup_frames"])
                    alarm = self._alarm(arm, score, scored)
                    if scored:
                        self._record_monitor_score(arm, errors)
                        self._block_errors[arm].append(score)
                    result[arm] = self._arm_row(
                        before, after, score, scored, "baseline", "baseline",
                        errors, alarm)
                self._monitor_predictions[arm] = after
        self._monitor_frames += 1
        if self._monitor_frames % int(self.protocol["block_frames"]) == 0:
            self._close_block()
            self._checkpoint()
        return result

    def _evaluate_probe(self, rows: list[list[float]]) -> dict[str, Any]:
        warmup = int(self.protocol["warmup_frames"])
        fixed_hash_before = self._bank_hash(self._models["fixed_snn"])
        sums = {arm: 0.0 for arm in self.ARM_NAMES}
        counts = {arm: 0 for arm in self.ARM_NAMES}
        for arm, live in self._models.items():
            evaluator = live.clone_for_evaluation()
            predictions: list[float] | None = None
            for position, values in enumerate(rows):
                timestamp = (position + 1) * float(
                    self.protocol["frame_seconds"])
                if predictions is None:
                    predictions = evaluator.start_session(timestamp, values)
                    continue
                errors = self._errors(predictions, values)
                if position >= warmup:
                    sums[arm] += sum(error * error for error in errors)
                    counts[arm] += len(errors)
                predictions = evaluator.advance(timestamp, values)
            evaluator.end_session()
            self._accumulate_work(
                self._probe_work, arm, evaluator.metrics())
        for arm in (arm for arm in self.ARM_NAMES
                    if arm not in self._models):
            predictions = None
            baseline_state: dict[str, Any] = {}
            for position, values in enumerate(rows):
                if predictions is None:
                    predictions = self._baseline_next(
                        arm, None, values, baseline_state)
                    continue
                errors = self._errors(predictions, values)
                if position >= warmup:
                    sums[arm] += sum(error * error for error in errors)
                    counts[arm] += len(errors)
                predictions = self._baseline_next(
                    arm, predictions, values, baseline_state)
        return {
            "frames": len(rows), "warmup_frames": warmup,
            "arms": {arm: {
                "sse": sums[arm], "count": counts[arm],
                "frame_count": counts[arm] // 6,
                "feature_target_count": counts[arm],
                "mse": sums[arm] / counts[arm] if counts[arm] else None,
            } for arm in self.ARM_NAMES},
            "fixed_hash_before": fixed_hash_before,
            "fixed_hash_after": self._bank_hash(self._models["fixed_snn"]),
        }

    def _checkpoint(self) -> None:
        probe = self._evaluate_probe(self._probe_rows)
        fixed_hash = self._bank_hash(self._models["fixed_snn"])
        if fixed_hash != self._fixed_hash:
            raise RuntimeError(
                "fixed SNN parameters changed during monitoring")
        self._checkpoints.append({
            "monitor_frame": self._monitor_frames,
            "sealed_probe": probe,
            "fixed_hash": fixed_hash,
        })

    def _close_block(self) -> None:
        arms: dict[str, Any] = {}
        for arm in self.ARM_NAMES:
            errors = self._block_errors[arm]
            summary = self._error_summary(errors)
            summary["alarm_count"] = self._block_alarms[arm]
            summary["alarms_per_minute"] = (
                self._block_alarms[arm]
                / (len(errors)
                   * float(self.protocol["frame_seconds"]) / 60.0)
                if errors else None
            )
            arms[arm] = summary
            self._block_errors[arm] = []
            self._block_alarms[arm] = 0
        self._blocks.append({"block": len(self._blocks) + 1,
                             "frames": self._monitor_frames,
                             "arms": arms,
                             "relative_improvement_vs_fixed": {
                                 arm: (
                                     (arms["fixed_snn"]["mse"]
                                      - arms[arm]["mse"])
                                     / arms["fixed_snn"]["mse"]
                                     if arms["fixed_snn"]["mse"] not in
                                     (None, 0)
                                     and arms[arm]["mse"] is not None
                                     else None
                                 )
                                 for arm in self.ARM_NAMES
                                 if arm != "fixed_snn"
                             }})

    def _alarm(self, arm: str, score: float, scored: bool) -> bool:
        if not scored:
            return False
        if self._monitor_cooldown[arm] > 0:
            self._monitor_cooldown[arm] -= 1
            return False
        if score > self._thresholds[arm]:
            self._monitor_alarms[arm] += 1
            self._block_alarms[arm] += 1
            self._monitor_cooldown[arm] = max(
                0, int(self.protocol["cooldown_frames"]) - 1)
            return True
        return False

    def _baseline_next(self, arm: str, previous: list[float] | None,
                       current: list[float],
                       state: dict[str, Any] | None = None) -> list[float]:
        if previous is None:
            if arm == "ema":
                if state is not None:
                    state["ema"] = list(current)
            return list(current)
        if arm == "persistence":
            return list(current)
        alpha = float(self.protocol["ema_alpha"])
        ema = state.get("ema") if state is not None else previous
        if ema is None:
            ema = previous
        updated = [alpha * actual + (1.0 - alpha) * prior
                   for actual, prior in zip(current, ema)]
        if state is not None:
            state["ema"] = updated
        return updated

    def _train(self, learner: AudioLearner,
               rows: list[list[float]]) -> AudioLearner:
        if not rows:
            raise ValueError("training rows must not be empty")
        predictions = learner.start_session(
            float(self.protocol["frame_seconds"]), rows[0])
        for position, values in enumerate(rows[1:], 1):
            predictions = learner.advance(
                (position + 1) * float(self.protocol["frame_seconds"]), values)
        learner.end_session()
        return learner

    def _normalize(self, band_dbfs: list[float]) -> list[float]:
        if self._normalizer is None:
            raise RuntimeError("normalizer is not fitted")
        return normalize_bands(band_dbfs, self._normalizer)

    def _phase_index(self) -> int:
        if self._phase == "fit":
            return len(self._fit_rows)
        if self._phase == "calibration":
            return len(self._cal_rows)
        if self._phase == "probe":
            return len(self._probe_rows)
        return self._monitor_frames

    @staticmethod
    def _validate_band_values(values: Any) -> list[float]:
        if not isinstance(values, (list, tuple)) or len(values) != 6:
            raise ValueError("band_dbfs must contain six values")
        result = [AudioSession._finite(value, "band_dbfs")
                  for value in values]
        return result

    @staticmethod
    def _finite(value: Any, name: str) -> float:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value))):
            raise ValueError(f"{name} must be finite")
        return float(value)

    @staticmethod
    def _errors(prediction: list[float], actual: list[float]) -> list[float]:
        return [guess - target for guess, target in zip(prediction, actual)]

    @staticmethod
    def _score(errors: list[float]) -> float:
        return math.sqrt(sum(error * error for error in errors) / len(errors))

    @staticmethod
    def _arm_row(before: list[float] | None, after: list[float],
                 score: float | None, scored: bool, hash_before: str,
                 hash_after: str, errors: list[float] | None = None,
                 alarm: bool = False) -> dict[str, Any]:
        return {
            "prediction_before": before, "prediction_next": after,
            "score": score, "sse": (sum(value * value for value in errors)
                                      if errors is not None else None),
            "scored": scored, "alarm": alarm,
            "hash_before": hash_before, "hash_after": hash_after,
        }

    @staticmethod
    def _error_summary(values: list[float]) -> dict[str, Any]:
        sum_squared_error = sum(value * value for value in values) * 6
        return AudioSession._summary_from_totals(sum_squared_error,
                                                 len(values))

    @staticmethod
    def _summary_from_totals(sse: float, frames: int) -> dict[str, Any]:
        target_count = frames * 6
        return {
            "count": frames, "frame_count": frames,
            "feature_target_count": target_count,
            "sse": sse,
            "mse": (sse / target_count
                    if target_count else None),
        }

    def _record_monitor_score(self, arm: str, errors: list[float]) -> None:
        self._monitor_sse[arm] += sum(error * error for error in errors)
        self._monitor_scored_frames[arm] += 1

    @staticmethod
    def _bank_hash(model: AudioLearner) -> str:
        encoded = json.dumps(model.weight_hash(), separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _validate_protocol(self) -> None:
        for key in ("fit_frames", "calibration_frames", "probe_frames",
                    "warmup_frames", "cooldown_frames", "block_frames"):
            value = self.protocol[key]
            if type(value) is not int or value < 0:
                raise ValueError(f"{key} must be a non-negative integer")
        for key in ("fit_frames", "calibration_frames", "probe_frames",
                    "block_frames"):
            if self.protocol[key] < 1:
                raise ValueError(f"{key} must be positive")
        if (self.protocol["calibration_frames"]
                <= self.protocol["warmup_frames"]):
            raise ValueError(
                "calibration must have scored frames after warmup")
        if self.protocol["probe_frames"] <= self.protocol["warmup_frames"]:
            raise ValueError("probe must have frames after warmup")
        self._finite(self.protocol["frame_seconds"], "frame_seconds")
        if self.protocol["frame_seconds"] != 0.2:
            raise ValueError("frame_seconds is fixed at 0.2")
        self._finite(self.protocol["ema_alpha"], "ema_alpha")
        if self.protocol["frame_seconds"] <= 0:
            raise ValueError("frame_seconds must be positive")
        if not 0.0 < self.protocol["ema_alpha"] <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if type(self.protocol["seed"]) is not int or self.protocol["seed"] < 0:
            raise ValueError("seed must be a non-negative integer")
