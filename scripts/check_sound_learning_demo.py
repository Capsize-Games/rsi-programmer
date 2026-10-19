"""Synthetic end-to-end check for teachable sound memory.

This demonstrates deterministic local feature matching. It is not evidence
of real-world microphone accuracy.
"""

from __future__ import annotations

import math
import random
import tempfile
from pathlib import Path
from typing import Any

from rsi_programmer.sound_features import extract_sound
from rsi_programmer.sound_memory import SoundMemory


SAMPLE_RATE = 16000
DURATION = 1.2
LABEL_ACTIONS = {
    "bell": "wake",
    "knock": "timer",
    "whistle": "sleep",
    "chime": "dance",
}


def make_sound(kind: str, seed: int) -> list[float]:
    """Make one noisy, level-varied example from a simple sound family."""
    rng = random.Random(seed)
    base = {
        "bell": 880.0,
        "knock": 180.0,
        "whistle": 1760.0,
        "chime": 1320.0,
    }[kind]
    frequency = base * rng.uniform(0.96, 1.04)
    phase = rng.uniform(0.0, 2.0 * math.pi)
    gain = rng.uniform(0.20, 0.75)
    tempo = rng.uniform(0.92, 1.08)
    samples: list[float] = []
    for index in range(round(SAMPLE_RATE * DURATION)):
        time_seconds = index / SAMPLE_RATE
        if kind == "bell":
            active = (time_seconds * tempo) % 0.6 < 0.19
            envelope = math.exp(-((time_seconds * tempo) % 0.6) * 5.0)
        elif kind == "knock":
            active = (time_seconds * tempo) % 0.6 < 0.075
            envelope = math.exp(-((time_seconds * tempo) % 0.6) * 24.0)
        elif kind == "whistle":
            active = (time_seconds * tempo) % 0.8 < 0.42
            envelope = 1.0
        else:
            active = (time_seconds * tempo) % 0.8 < 0.30
            envelope = 0.5 + 0.5 * math.sin(
                math.pi * ((time_seconds * tempo) % 0.8) / 0.30)
        carrier = math.sin(2.0 * math.pi * frequency * time_seconds + phase)
        signal = gain * envelope * carrier if active else 0.0
        samples.append(signal + rng.gauss(0.0, 0.0015))
    return samples


def descriptor(kind: str, seed: int) -> dict[str, Any]:
    return extract_sound(make_sound(kind, seed), SAMPLE_RATE)


def evaluate(
    memory: SoundMemory,
    phase: str,
    expected_by_kind: dict[str, str],
    seed_offset: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, (kind, expected) in enumerate(expected_by_kind.items()):
        result = memory.recognize(descriptor(kind, seed_offset + index))
        memory.record_probe(
            expected, descriptor(kind, seed_offset + index))
        rows.append({"expected": expected, **result})
    return summarize_rows(phase, rows)


def summarize_rows(phase: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = [row for row in rows if row["label"] == row["expected"]]
    missed = [row for row in rows
              if row["label"] not in (None, row["expected"])]
    unknown = [row for row in rows if row["label"] is None]
    return {"phase": phase, "correct": correct, "missed": missed,
            "unknown": unknown, "false_matches": missed}


def run_demo(path: Path) -> dict[str, Any]:
    memory = SoundMemory(path)
    teaching_seeds = {
        "bell": (100, 101, 102),
        "knock": (200, 201, 202),
        "whistle": (300, 301, 302),
    }
    for kind, seeds in teaching_seeds.items():
        for seed in seeds:
            memory.teach(kind, descriptor(kind, seed), LABEL_ACTIONS[kind])

    held_out = {kind: kind for kind in teaching_seeds}
    before_fourth = evaluate(
        memory, "held_out_before_fourth", held_out, 1000)
    old_fingerprints = memory.summary()["label_fingerprints"]

    memory.teach("chime", descriptor("chime", 400), LABEL_ACTIONS["chime"])
    fourth_class_holdout = {**held_out, "chime": "chime"}
    after_fourth = evaluate(
        memory, "held_out_after_fourth", fourth_class_holdout, 1200)
    retained = {
        label: memory.summary()["label_fingerprints"].get(label) == fingerprint
        for label, fingerprint in old_fingerprints.items()
    }

    reloaded = SoundMemory(path)
    after_restart = evaluate(
        reloaded, "held_out_after_restart", fourth_class_holdout, 1200)
    silence = extract_sound([0.0] * round(SAMPLE_RATE * DURATION), SAMPLE_RATE)
    noise_rng = random.Random(9000)
    noise = extract_sound(
        [noise_rng.gauss(0.0, 0.04) for _ in range(
            round(SAMPLE_RATE * DURATION))], SAMPLE_RATE
    )
    unknown_inputs = {
        "silence": reloaded.recognize(silence),
        "noise": reloaded.recognize(noise),
    }
    return {
        "notice": (
            "Synthetic demonstration only; not real microphone accuracy."
        ),
        "held_out_before_fourth": before_fourth,
        "held_out_after_fourth": after_fourth,
        "held_out_after_restart": after_restart,
        "old_label_fingerprints_retained": retained,
        "silence_and_noise": unknown_inputs,
        "memory": reloaded.summary(),
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="sound-demo-") as directory:
        report = run_demo(Path(directory) / "memory.json")
    import json

    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
