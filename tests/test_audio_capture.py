"""Unit tests for explicit-source microphone frame capture."""

from __future__ import annotations

import json
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from rsi_programmer.audio_capture import PulseCapture, list_sources


class FakeProcess:
    def __init__(self, raw: bytes) -> None:
        self.stdout = tempfile.TemporaryFile()
        self.stdout.write(raw)
        self.stdout.seek(0)
        self.terminated = False

    def poll(self) -> int | None:
        return None if not self.terminated else 0

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        return 0


def pcm_bytes(values: list[int]) -> bytes:
    return b"".join(struct.pack("<h", value) for value in values)


class AudioCaptureTests(unittest.TestCase):
    def test_reads_exact_frame_and_uses_explicit_source(self) -> None:
        process = FakeProcess(pcm_bytes([1000, -1000, 2000, -2000]))
        with patch("rsi_programmer.audio_capture.subprocess.Popen",
                   return_value=process) as popen:
            capture = PulseCapture("alsa_input.usb-mic", 20, 0.2)
            with capture:
                samples, timestamp = capture.read_frame()
            command = popen.call_args.args[0]
        self.assertEqual(command[2], "alsa_input.usb-mic")
        self.assertIn("--latency-msec=100", command)
        self.assertEqual(len(samples), 4)
        self.assertAlmostEqual(samples[0], 1000 / 32768.0)
        self.assertGreater(timestamp, 0.0)
        self.assertTrue(process.terminated)

    def test_requires_source(self) -> None:
        with self.assertRaises(ValueError):
            PulseCapture("")

    def test_returns_silent_and_clipped_frames(self) -> None:
        for raw in (pcm_bytes([0] * 4), pcm_bytes([32767] * 4)):
            process = FakeProcess(raw)
            with patch("rsi_programmer.audio_capture.subprocess.Popen",
                       return_value=process):
                with PulseCapture("mic", 20, 0.2) as capture:
                    samples, _ = capture.read_frame()
                    self.assertEqual(len(samples), 4)

    def test_rejects_monitor_source_and_empty_frame_size(self) -> None:
        with self.assertRaises(ValueError):
            PulseCapture("sink.monitor")
        with self.assertRaises(ValueError):
            PulseCapture("mic", 1, 0.1)

    def test_list_sources_excludes_monitor_sources(self) -> None:
        sources = [
            {"name": "usb_input", "description": "Blue Yeti",
             "properties": {"device.serial": "private"}},
            {"name": "sink.monitor", "description": "Monitor"},
        ]
        result = subprocess.CompletedProcess([], 0, json.dumps(sources), "")
        with patch("rsi_programmer.audio_capture.subprocess.run",
                   return_value=result):
            listed = list_sources()
        self.assertEqual(
            listed,
            [{"name": "usb_input", "description": "Blue Yeti"}],
        )


if __name__ == "__main__":
    unittest.main()
