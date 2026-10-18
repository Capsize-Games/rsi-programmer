"""Validated access to the frozen UCI fan-vibration recordings."""
from __future__ import annotations

import csv
import hashlib
import math
import statistics
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "sensing_fan_v1" / "accelerometer (1).csv"
CSV_SHA256 = (
    "296a94eaf40643e1188012a4ee37d0b2c397bbc04decd39cfbe9952b8c841674")
EXPECTED_COLUMNS = ("wconfid", "pctid", "x", "y", "z")
CONFIGURATIONS = (1, 2, 3)
SPEEDS = tuple(range(20, 101, 5))
ROWS_PER_GROUP = 3000
WINDOW_SAMPLES = 10
MIN_SCALE = 0.000001
Recordings = dict[tuple[int, int], list[float]]


def _finite_float(value: object, label: str) -> float:
    """Convert a value to a finite float or raise a clear input error."""
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def load_recordings() -> Recordings:
    """Load x-axis samples after validating the complete pinned CSV."""
    payload = CSV_PATH.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != CSV_SHA256:
        raise ValueError("sensing CSV SHA-256 does not match protocol")

    recordings: Recordings = {}
    group_order: list[tuple[int, int]] = []
    group_counts: dict[tuple[int, int], int] = {}
    with CSV_PATH.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
            raise ValueError("sensing CSV columns do not match protocol")
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(row.get(key) is None
                                  for key in EXPECTED_COLUMNS):
                raise ValueError(f"malformed CSV row {line_number}")
            try:
                config = int(row["wconfid"])
                speed = int(row["pctid"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"non-integer group ID at row {line_number}") from error
            group = (config, speed)
            if config not in CONFIGURATIONS or speed not in SPEEDS:
                raise ValueError(f"unexpected group ID at row {line_number}")
            if not group_order or group_order[-1] != group:
                group_order.append(group)
                if group in group_counts:
                    raise ValueError(f"non-contiguous source group {group}")
                group_counts[group] = 0
                recordings[group] = []
            values: dict[str, float] = {}
            for axis in ("x", "y", "z"):
                values[axis] = _finite_float(
                    row[axis], f"{axis} at row {line_number}")
            recordings[group].append(values["x"])
            group_counts[group] += 1

    expected_order = [(config, speed) for config in CONFIGURATIONS
                      for speed in SPEEDS]
    if group_order != expected_order:
        raise ValueError("sensing CSV source group order changed")
    if set(recordings) != set(expected_order):
        raise ValueError("sensing CSV group set changed")
    if any(count != ROWS_PER_GROUP
           for count in group_counts.values()):
        raise ValueError("sensing CSV group row count changed")
    return recordings


def rms_windows(samples: Sequence[float]) -> list[float]:
    """Return RMS for complete, disjoint ten-sample windows."""
    values = [_finite_float(value, "sample") for value in samples]
    window_count = len(values) // WINDOW_SAMPLES
    features = []
    for start in range(0, window_count * WINDOW_SAMPLES, WINDOW_SAMPLES):
        mean_square = sum(value * value for value in values[start:start + 10])
        feature = math.sqrt(mean_square / WINDOW_SAMPLES)
        if not math.isfinite(feature):
            raise ValueError("derived RMS feature must be finite")
        features.append(feature)
    return features


def fit_scale(features: Sequence[float]) -> tuple[float, float]:
    """Fit median center and floored robust scale from supplied features."""
    values = [_finite_float(value, "feature") for value in features]
    if not values:
        raise ValueError("features must be nonempty and finite")
    center = _finite_float(statistics.median(values), "feature center")
    mad_value = statistics.median(abs(value - center) for value in values)
    mad = _finite_float(mad_value, "feature MAD")
    scale = max(1.4826 * mad, MIN_SCALE)
    if not math.isfinite(scale):
        raise ValueError("derived feature scale must be finite")
    return center, scale


def normalize(value: float, center: float, scale: float) -> float:
    """Map one feature to the protocol's bounded normalization range."""
    value = _finite_float(value, "value")
    center = _finite_float(center, "center")
    scale = _finite_float(scale, "scale")
    if scale <= 0:
        raise ValueError("scale must be positive")
    difference = value - center
    denominator = scale + abs(difference)
    if not math.isfinite(difference) or not math.isfinite(denominator):
        raise ValueError("derived normalization values must be finite")
    normalized = 0.5 + 0.5 * difference / denominator
    if not math.isfinite(normalized):
        raise ValueError("normalized feature must be finite")
    return normalized
