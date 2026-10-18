"""Frozen predict-before-update sensing comparison on recorded fan data."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from rsi_programmer.online_benchmark import source_manifest
from rsi_programmer.sensing_baseline import SensingBaseline
from rsi_programmer.sensing_data import (
    fit_scale, load_recordings, normalize, rms_windows,
)
from rsi_programmer.sensing_detector import (
    SensingAlarm, calibrated_threshold, event_metrics,
)
from rsi_programmer.sensing_predictor import SensingPredictor
from rsi_programmer.sensing_summary import ARMS, summarize


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "fixtures" / "sensing_fan_v1.json"
PROTOCOL_SHA256 = (
    "b604d5cb2a88f9dcc7278c0f3ffd6e7437ccf798956958283235e8663a31d191")
TRAIN_SPEEDS = (20, 35, 50, 65, 80, 95)
TEST_SPEEDS = (25, 30, 40, 45, 55, 60, 70, 75, 85, 90, 100)


def digest(payload: bytes) -> str:
    """Hash exact bytes for source and output provenance."""
    return hashlib.sha256(payload).hexdigest()


def state_hash(payload: dict[str, Any]) -> str:
    """Hash canonical boundary state without publishing learned weights."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    return digest(encoded)


def load_protocol() -> dict[str, Any]:
    """Reject changes to the prospective comparison."""
    raw = PROTOCOL.read_bytes()
    if digest(raw) != PROTOCOL_SHA256:
        raise ValueError("sensing protocol SHA mismatch")
    return json.loads(raw)


def make_model(arm: str, seed: int) -> Any:
    """Construct one declared model, without task or configuration labels."""
    if arm == "frozen_snn":
        raise ValueError("frozen SNN inherits the fully trained SNN")
    if arm in ("persistence", "ema", "nlms"):
        return SensingBaseline(mode=arm)
    if arm not in ("plastic_snn", "readout_snn", "plastic_ann"):
        raise ValueError("unknown sensing arm")
    return SensingPredictor(seed=seed, spiking=arm != "plastic_ann",
                            encoder_learning=arm != "readout_snn")


def run_sequence(model: Any, values: list[float],
                 threshold: float | None = None) -> dict[str, Any]:
    """Only values, time and a calibrated threshold enter this loop."""
    if len(values) < 6:
        raise ValueError("a sequence needs scored observations")
    windows = []
    pending: float | None = None
    alarm = SensingAlarm(threshold) if threshold is not None else None
    processing_cpu = 0.0
    provenance_cpu = 0.0
    tick = time.process_time()
    before_metrics = model.metrics()
    provenance_cpu += time.process_time() - tick
    for index, value in enumerate(values):
        timestamp = (index + 1) * 0.2
        tick = time.process_time()
        before = model.weight_hash()
        provenance_cpu += time.process_time() - tick
        tick = time.process_time()
        score = abs(value - pending) if pending is not None else None
        fired = alarm.observe(index, score) if alarm is not None else False
        if index == 0:
            following = model.start_session(timestamp, value)
        else:
            following = model.advance(timestamp, value)
        processing_cpu += time.process_time() - tick
        tick = time.process_time()
        after = model.weight_hash()
        provenance_cpu += time.process_time() - tick
        windows.append({
            "i": index, "timestamp": timestamp, "value": value,
            "prediction_before": pending, "score": score,
            "scored": index >= 5, "alarm": fired,
            "prediction_next": following,
            "model_hash_before": before, "model_hash_after": after,
        })
        pending = following
    tick = time.process_time()
    model.end_session()
    processing_cpu += time.process_time() - tick
    tick = time.process_time()
    metrics = model.metrics()
    snapshot_hash = state_hash(model.snapshot())
    weight_hash = model.weight_hash()
    provenance_cpu += time.process_time() - tick
    work = {key: value - before_metrics[key]
            for key, value in metrics.items() if type(value) is int}
    return {
        "windows": windows, "end_weight_hash": weight_hash,
        "end_snapshot_hash": snapshot_hash, "work": work,
        "processing_cpu_seconds": processing_cpu,
        "provenance_cpu_seconds": provenance_cpu,
    }


def prepared_features() -> tuple[dict[tuple[int, int], list[float]],
                                 dict[str, Any]]:
    """Fit only training prefixes; retain no condition labels in features."""
    recordings = load_recordings()
    features = {key: rms_windows(samples)
                for key, samples in recordings.items()}
    train = [value for speed in TRAIN_SPEEDS
             for value in features[(1, speed)][:150]]
    center, scale = fit_scale(train)
    normalized = {key: [normalize(value, center, scale) for value in values]
                  for key, values in features.items()}
    return normalized, {
        "center": center, "scale": scale, "training_feature_count": 900,
        "fit_groups": [[1, speed, 0, 1500] for speed in TRAIN_SPEEDS],
    }


def train_models(features: dict[tuple[int, int], list[float]]
                 ) -> tuple[dict[tuple[str, int], Any],
                            list[dict[str, Any]]]:
    """Fit exactly one pass and derive the frozen control from full SNN."""
    models: dict[tuple[str, int], Any] = {}
    records = []
    for seed in (0, 1, 2):
        for arm in ARMS:
            if arm == "frozen_snn":
                model = models[("plastic_snn", seed)].evaluation_copy()
                sequences = []
            else:
                model = make_model(arm, seed)
                sequences = []
                for speed in TRAIN_SPEEDS:
                    result = run_sequence(model, features[(1, speed)][:150])
                    sequences.append({"speed": speed, **result})
            models[(arm, seed)] = model
            record = {
                "arm": arm, "seed": seed, "sequences": sequences,
                "trained_weight_hash": model.weight_hash(),
                "trained_snapshot_hash": state_hash(model.snapshot()),
            }
            if arm == "frozen_snn":
                record["inherited_from"] = "plastic_snn"
            records.append(record)
    return models, records


def calibrate(models: dict[tuple[str, int], Any],
              features: dict[tuple[int, int], list[float]]
              ) -> tuple[dict[tuple[str, int], float],
                         list[dict[str, Any]]]:
    """Discard calibration parameter updates; retain only frozen thresholds."""
    thresholds = {}
    records = []
    for (arm, seed), trained in models.items():
        scores = []
        sequences = []
        for speed in TRAIN_SPEEDS:
            model = copy.deepcopy(trained)
            result = run_sequence(model, features[(1, speed)][150:225])
            scores.extend(row["score"] for row in result["windows"]
                          if row["scored"])
            sequences.append({"speed": speed, **result})
        threshold = calibrated_threshold(scores)
        thresholds[(arm, seed)] = threshold
        records.append({"arm": arm, "seed": seed, "sequences": sequences,
                        "scores": scores, "threshold": threshold})
    return thresholds, records


def stream_spec(speed: int, kind: str, order: int
                ) -> tuple[list[dict[str, int]], list[int]]:
    """Return frozen source slices and evaluator-only event indices."""
    if speed not in TEST_SPEEDS or order not in (0, 1):
        raise ValueError("unknown held stream identity")
    if kind == "positive":
        configurations = ((1, 2, 1, 3), (1, 3, 1, 2))[order]
        quarters = (0, 0, 1, 0)
        events = [75, 150, 225]
    elif kind == "sham":
        configurations = (1, 1, 1, 1)
        quarters = ((0, 2, 1, 3), (3, 1, 2, 0))[order]
        events = []
    else:
        raise ValueError("unknown stream kind")
    slices = [{"config": config, "speed": speed,
               "start": quarter * 750, "stop": (quarter + 1) * 750}
              for config, quarter in zip(configurations, quarters)]
    return slices, events


def stream_values(features: dict[tuple[int, int], list[float]],
                  slices: list[dict[str, int]]) -> list[float]:
    """Compile complete causal windows without joining labels to inputs."""
    return [value for part in slices
            for value in features[(part["config"], part["speed"])][
                part["start"] // 10:part["stop"] // 10]]


def _write_json(path: Path, value: Any) -> None:
    """Write finite JSON evidence once."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def run_benchmark(output: Path) -> dict[str, Any]:
    """Run the prospective matrix and bind source-specific evidence."""
    protocol = load_protocol()
    if output.exists():
        raise ValueError("refusing to overwrite a prior measurement bundle")
    output.mkdir(parents=True)
    sources = source_manifest()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    wall_start, cpu_start = time.monotonic(), time.process_time()
    tick = time.process_time()
    features, normalization = prepared_features()
    data_cpu = time.process_time() - tick
    models, training = train_models(features)
    thresholds, calibration = calibrate(models, features)
    _write_json(output / "training.json", training)
    _write_json(output / "calibration.json", calibration)
    compact = []
    raw_path = output / "streams.jsonl.gz"
    with gzip.open(raw_path, "wt", encoding="utf-8") as stream:
        for (arm, seed), trained in models.items():
            for speed in TEST_SPEEDS:
                for kind in ("positive", "sham"):
                    for order in (0, 1):
                        slices, events = stream_spec(speed, kind, order)
                        values = stream_values(features, slices)
                        result = run_sequence(copy.deepcopy(trained), values,
                                              thresholds[(arm, seed)])
                        metrics = event_metrics(result["windows"], events)
                        identity = {"arm": arm, "seed": seed, "speed": speed,
                                    "kind": kind, "order": order}
                        row = {**identity, "source_slices": slices,
                               "events": events,
                               "threshold": thresholds[(arm, seed)],
                               **result, "metrics": metrics}
                        stream.write(json.dumps(row, sort_keys=True,
                                                allow_nan=False) + "\n")
                        compact.append({**identity, "metrics": metrics,
                                        "processing_cpu_seconds": result[
                                            "processing_cpu_seconds"],
                                        "provenance_cpu_seconds": result[
                                            "provenance_cpu_seconds"]})
        # Trained models remain unchanged by every calibration/test copy.
    for record in training:
        model = models[(record["arm"], record["seed"])]
        if state_hash(model.snapshot()) != record["trained_snapshot_hash"]:
            raise RuntimeError("held evaluation mutated a trained source")
    summary = summarize(compact)
    _write_json(output / "summary.json", summary)
    if source_manifest() != sources:
        raise RuntimeError("source changed while measuring")
    manifest = {
        "version": 1, "git_head": head, "protocol_sha256": PROTOCOL_SHA256,
        "source_files": sources, "source_unchanged": True,
        "trained_sources_unchanged": True,
        "dataset": protocol["dataset"], "normalization": normalization,
        "stream_rows": len(compact), "windows": len(compact) * 300,
        "python": platform.python_version(),
        "data_preparation_cpu_seconds": data_cpu,
        "full_process_cpu_seconds": time.process_time() - cpu_start,
        "wall_seconds": time.monotonic() - wall_start,
        "cost_scope": "Processing covers model, residual and alarm calls, "
        "including session reset. Provenance hashes are separate. "
        "Full process includes data/copies/evaluator/output/provenance; "
        "no device-energy measurement.",
        "files": {name: digest((output / name).read_bytes()) for name in (
            "training.json", "calibration.json", "streams.jsonl.gz",
            "summary.json")},
    }
    _write_json(output / "manifest.json", manifest)
    return manifest
