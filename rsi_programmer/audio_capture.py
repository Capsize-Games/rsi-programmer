"""Explicit-source, fixed-frame PulseAudio/PipeWire microphone capture."""

from __future__ import annotations

import json
import math
import os
import select
import subprocess
import time
from typing import Any


class PulseCapture:
    """Read mono signed-16-bit PCM from one explicitly selected source."""

    def __init__(
        self,
        source: str,
        sample_rate: int = 16000,
        frame_seconds: float = 0.2,
    ) -> None:
        if not source:
            raise ValueError("an explicit audio source is required")
        if source.endswith(".monitor"):
            raise ValueError("monitor sources cannot be used for capture")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
            raise TypeError("sample_rate must be an integer")
        if (
            isinstance(frame_seconds, bool)
            or not isinstance(frame_seconds, (int, float))
            or not math.isfinite(frame_seconds)
        ):
            raise TypeError("frame_seconds must be a finite number")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if frame_seconds <= 0:
            raise ValueError("sample_rate and frame_seconds must be positive")
        if round(sample_rate * frame_seconds) <= 0:
            raise ValueError("frame duration must contain at least one sample")
        self.source = source
        self.sample_rate = sample_rate
        self.frame_seconds = frame_seconds
        self._process: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> PulseCapture:
        if self._process is not None:
            raise RuntimeError("capture is already open")
        self._process = subprocess.Popen(
            [
                "parec",
                "--device", self.source,
                "--format", "s16le",
                "--rate", str(self.sample_rate),
                "--channels", "1",
                "--raw",
                "--latency-msec=100",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdout is not None:
            process.stdout.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def read_frame(self) -> tuple[list[float], float]:
        """Read one exact frame, returning normalized PCM and end timestamp."""
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("capture is not open")
        sample_count = round(self.sample_rate * self.frame_seconds)
        wanted = sample_count * 2
        deadline = time.monotonic() + max(2.0, self.frame_seconds * 4.0)
        chunks = bytearray()
        fd = process.stdout.fileno()
        while len(chunks) < wanted:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out reading microphone frame")
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                raise TimeoutError("timed out reading microphone frame")
            chunk = os.read(fd, wanted - len(chunks))
            if not chunk:
                raise EOFError("microphone stream ended before a full frame")
            chunks.extend(chunk)
        timestamp = time.monotonic()
        samples = [
            int.from_bytes(chunks[i:i + 2], "little", signed=True) / 32768.0
            for i in range(0, wanted, 2)
        ]
        return samples, timestamp


def list_sources() -> list[dict[str, str]]:
    """List usable pactl sources without exposing device properties."""
    result = subprocess.run(
        ["pactl", "-f", "json", "list", "sources"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    records: list[dict[str, Any]] = json.loads(result.stdout)
    return [
        {
            "name": str(record.get("name", "")),
            "description": str(record.get("description", "")),
        }
        for record in records
        if record.get("name") and not str(record["name"]).endswith(".monitor")
    ]
