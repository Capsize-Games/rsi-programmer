"""Compact, local features for short sound examples."""

from __future__ import annotations

import math
from typing import Any

import torch


FEATURE_VERSION = 1
SAMPLE_RATE = 16000
WINDOW_SECONDS = 1.2
TEMPORAL_BINS = 12
FREQUENCY_BANDS = 16
FEATURES_PER_BIN = FREQUENCY_BANDS + 4
FEATURE_COUNT = TEMPORAL_BINS * FEATURES_PER_BIN
MIN_FREQUENCY_HZ = 80.0
MAX_FREQUENCY_HZ = 7600.0


def extract_sound(
    samples: list[float], sample_rate: int = SAMPLE_RATE
) -> dict[str, Any]:
    """Describe mono PCM as 12 time bins of spectral and rhythm features.

    The returned descriptor contains 240 normalized floats and no audio.
    Inputs are clipped in time to the first 1.2 seconds and zero-padded when
    shorter. Each temporal bin has 16 relative log-frequency powers, followed
    by relative level, zero-crossing rate, spectral centroid and spectral
    change. It is intended for nearest-example matching, not source identity.
    """
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
        raise TypeError("sample_rate must be an integer")
    if sample_rate < 2 * MAX_FREQUENCY_HZ:
        raise ValueError(
            "sample_rate must support the feature frequency range"
        )
    if not samples:
        raise ValueError("samples must not be empty")
    if len(samples) > sample_rate * 30:
        raise ValueError("sound clips must be at most 30 seconds")
    signal = torch.as_tensor(samples, dtype=torch.float64)
    if signal.ndim != 1:
        raise ValueError("samples must be mono")
    if not bool(torch.isfinite(signal).all()):
        raise ValueError("samples must be finite")
    if bool(torch.any((signal < -1.0) | (signal > 1.0))):
        raise ValueError("PCM samples must be in [-1, 1]")

    target_size = round(WINDOW_SECONDS * SAMPLE_RATE)
    source_size = round(WINDOW_SECONDS * sample_rate)
    signal = signal[:source_size]
    if sample_rate != SAMPLE_RATE:
        new_size = max(1, round(signal.numel() * SAMPLE_RATE / sample_rate))
        signal = torch.nn.functional.interpolate(
            signal.reshape(1, 1, -1), size=new_size, mode="linear",
            align_corners=False,
        ).reshape(-1)
    signal = signal[:target_size]
    if signal.numel() < target_size:
        signal = torch.nn.functional.pad(
            signal, (0, target_size - signal.numel())
        )

    band_centers = torch.logspace(
        math.log10(MIN_FREQUENCY_HZ), math.log10(MAX_FREQUENCY_HZ),
        FREQUENCY_BANDS, dtype=torch.float64,
    )
    log_band_centers = torch.log(band_centers)
    band_width = float(log_band_centers[1] - log_band_centers[0])
    bin_size = target_size // TEMPORAL_BINS
    raw_rows: list[list[float]] = []
    levels: list[float] = []
    previous_spectrum: torch.Tensor | None = None
    for index in range(TEMPORAL_BINS):
        frame = signal[index * bin_size:(index + 1) * bin_size]
        centered = frame - frame.mean()
        rms = float(torch.sqrt(torch.mean(frame.square())))
        levels.append(rms)
        window = torch.hann_window(
            bin_size, periodic=False, dtype=torch.float64
        )
        spectrum = torch.fft.rfft(centered * window).abs().square()
        frequencies = torch.fft.rfftfreq(bin_size, 1.0 / SAMPLE_RATE,
                                         dtype=torch.float64)
        in_range = ((frequencies >= MIN_FREQUENCY_HZ)
                    & (frequencies <= MAX_FREQUENCY_HZ))
        log_frequencies = torch.log(
            frequencies.clamp(min=MIN_FREQUENCY_HZ)
        )
        band_powers: list[float] = []
        for band in range(FREQUENCY_BANDS):
            weights = torch.clamp(
                1.0 - torch.abs(log_frequencies - log_band_centers[band])
                / band_width,
                min=0.0,
            ) * in_range
            band_powers.append(float((spectrum * weights).sum()))
        total = sum(band_powers) + 1e-24
        shares = [power / total for power in band_powers]
        crossings = float(torch.mean(
            (frame[1:] * frame[:-1] < 0).to(torch.float64)
        ))
        centroid = float((frequencies * spectrum).sum()
                         / (spectrum.sum() + 1e-24)) / SAMPLE_RATE
        normalized_spectrum = spectrum / (spectrum.sum() + 1e-24)
        if previous_spectrum is None:
            change = 0.0
        else:
            change = float(
                torch.abs(normalized_spectrum - previous_spectrum).sum() / 2.0
            )
        previous_spectrum = normalized_spectrum
        raw_rows.append(shares + [rms, crossings, centroid, change])

    level_scale = max(max(levels), 1e-12)
    vector: list[float] = []
    for row, level in zip(raw_rows, levels):
        vector.extend(row[:FREQUENCY_BANDS])
        vector.extend([
            level / level_scale,
            row[FREQUENCY_BANDS + 1],
            row[FREQUENCY_BANDS + 2],
            row[FREQUENCY_BANDS + 3],
        ])
    return {
        "version": FEATURE_VERSION,
        "sample_rate": SAMPLE_RATE,
        "duration_seconds": WINDOW_SECONDS,
        "active": max(levels) >= 0.002,
        "vector": vector,
    }
