"""Descriptive frame-level audio measurements for acoustic sensing."""

from __future__ import annotations

import math
from statistics import median
from typing import Any

import torch


BAND_EDGES_HZ = (80.0, 160.0, 320.0, 640.0, 1280.0, 2560.0, 6000.0)
NORMALIZER_MIN_SCALE_DB = 3.0


def extract_frame(samples: list[float], sample_rate: int) -> dict[str, Any]:
    """Return descriptive level, spectral-bin and band-energy measurements."""
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
        raise TypeError("sample_rate must be an integer")
    if sample_rate < 2 * int(BAND_EDGES_HZ[-1]):
        raise ValueError("sample_rate must support the highest band edge")
    if len(samples) < 2:
        raise ValueError("at least two samples are required")
    signal = torch.tensor(samples, dtype=torch.float64)
    if not bool(torch.isfinite(signal).all()):
        raise ValueError("samples must be finite")
    if bool(torch.any((signal < -1.0) | (signal > 1.0))):
        raise ValueError("PCM samples must be in [-1, 1]")
    dc_removed = signal - signal.mean()
    rms = float(torch.sqrt(torch.mean(signal.square())))
    peak = float(torch.max(torch.abs(signal)))
    clip_fraction = float(torch.mean((torch.abs(signal) >= 0.999).double()))
    window = torch.hann_window(
        len(samples), periodic=False, dtype=torch.float64
    )
    spectrum = torch.fft.rfft(dc_removed * window)
    power = spectrum.abs().square()
    if len(samples) % 2 == 0:
        power[1:-1] *= 2.0
    else:
        power[1:] *= 2.0
    frequencies = torch.fft.rfftfreq(
        len(samples), d=1.0 / sample_rate, dtype=torch.float64
    )
    denominator = len(samples) * float(torch.sum(window.square()))
    total_power = float(torch.sum(power)) / denominator
    peak_index = int(torch.argmax(power[1:])) + 1
    peak_hz = float(frequencies[peak_index])
    tone_power = float(power[peak_index]) / denominator
    noise_power = max(total_power - tone_power, 1e-24)
    # This is the strongest FFT bin against all remaining bins; it does not
    # estimate a physical tone or identify its source.
    tone_snr_db = 10.0 * math.log10(max(tone_power, 1e-24) / noise_power)

    band_dbfs: list[float] = []
    for low_hz, high_hz in zip(BAND_EDGES_HZ, BAND_EDGES_HZ[1:]):
        mask = (frequencies >= low_hz) & (frequencies < high_hz)
        band_power = float(torch.sum(power[mask])) / denominator
        band_dbfs.append(10.0 * math.log10(max(band_power, 1e-24)))

    return {
        "rms_dbfs": 20.0 * math.log10(max(rms, 1e-12)),
        "peak_dbfs": 20.0 * math.log10(max(peak, 1e-12)),
        "clip_fraction": clip_fraction,
        "peak_hz": peak_hz,
        "tone_snr_db": tone_snr_db,
        "band_dbfs": band_dbfs,
    }


def fit_band_normalizer(
    training_rows: list[dict[str, Any]],
) -> dict[str, list[float]]:
    """Fit robust per-band centers and scales from training rows only."""
    if not training_rows:
        raise ValueError("training_rows must not be empty")
    values = [row["band_dbfs"] for row in training_rows]
    band_count = len(BAND_EDGES_HZ) - 1
    if any(len(row) != band_count for row in values):
        raise ValueError(
            f"each band_dbfs row must contain {band_count} values"
        )
    if any(
        not math.isfinite(float(item))
        for row in values
        for item in row
    ):
        raise ValueError("training band values must be finite")
    centers: list[float] = []
    scales: list[float] = []
    for index in range(band_count):
        column = [float(row[index]) for row in values]
        center = median(column)
        deviation = median([abs(item - center) for item in column]) * 1.4826
        centers.append(center)
        scales.append(max(NORMALIZER_MIN_SCALE_DB, deviation))
    return {"centers_db": centers, "scales_db": scales}


def normalize_bands(
    band_dbfs: list[float], normalizer: dict[str, list[float]]
) -> list[float]:
    """Map band deviations smoothly into [0, 1] using a fitted normalizer."""
    centers = normalizer["centers_db"]
    scales = normalizer["scales_db"]
    if len(band_dbfs) != len(centers) or len(centers) != len(scales):
        raise ValueError("band values and normalizer dimensions must match")
    if any(
        not math.isfinite(float(value))
        for value in (*band_dbfs, *centers, *scales)
    ):
        raise ValueError("band values and normalizer values must be finite")
    output: list[float] = []
    for value, center, scale in zip(band_dbfs, centers, scales):
        if scale < NORMALIZER_MIN_SCALE_DB:
            raise ValueError("normalizer scales must be at least 3 dB")
        deviation = float(value) - center
        output.append(0.5 + 0.5 * deviation / (scale + abs(deviation)))
    return output
