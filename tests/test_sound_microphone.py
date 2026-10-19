"""Tests for the background microphone capture state machine."""
from __future__ import annotations

import fcntl
import os
import queue
import threading
import tempfile
import time
import unittest
from types import SimpleNamespace
from typing import Any

from rsi_programmer.sound_microphone import SoundMicrophone


class SoundMicrophoneTests(unittest.TestCase):
    """Exercise capture events with deterministic in-memory PCM sources."""

    def _capture_factory(
        self,
        sound_enabled: threading.Event,
        closed: threading.Event,
        fail_after_ready: bool = False,
        block_after_ready: bool = False,
    ) -> Any:
        frame_index = 0

        def read_frame() -> tuple[list[float], float]:
            nonlocal frame_index
            if closed.wait(0.001):
                return [0.0] * 640, time.monotonic()
            frame_index += 1
            if block_after_ready and frame_index > 30:
                closed.wait()
                return [0.0] * 640, time.monotonic()
            if fail_after_ready and frame_index > 30:
                raise OSError("fake capture detached")
            if sound_enabled.is_set():
                samples = [0.04 if i % 2 else -0.04 for i in range(640)]
            else:
                samples = [0.0] * 640
            return samples, time.monotonic()

        capture = SimpleNamespace()
        capture.__enter__ = lambda: capture
        capture.close = closed.set
        capture.__exit__ = lambda *_args: closed.set()
        capture.read_frame = read_frame
        return capture

    def _start(
        self,
        sound_enabled: threading.Event | None = None,
        fail_after_ready: bool = False,
        block_after_ready: bool = False,
        capture_factory: Any | None = None,
    ) -> tuple[SoundMicrophone, queue.Queue[dict[str, Any]],
               threading.Event]:
        sound = sound_enabled or threading.Event()
        closed = threading.Event()
        events: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=64)
        lock_dir = tempfile.TemporaryDirectory()
        self.addCleanup(lock_dir.cleanup)
        factory = capture_factory or (
            lambda *_args, **_kwargs: self._capture_factory(
                sound, closed, fail_after_ready, block_after_ready))
        microphone = SoundMicrophone(
            events, "test-input", capture_factory=factory,
            lock_path=os.path.join(lock_dir.name, "builder.lock"))
        self.addCleanup(microphone.stop)
        microphone.start()
        return microphone, events, closed

    def _next_event(
        self,
        events: queue.Queue[dict[str, Any]],
        kind: str,
        timeout: float = 2.0,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                event = events.get(
                    timeout=max(0.01, deadline - time.monotonic()))
            except queue.Empty:
                continue
            if event["kind"] == kind:
                return event
        self.fail(f"did not receive {kind} event")
        return {}

    def test_live_onset_emits_a_bounded_clip_and_stops_cleanly(self) -> None:
        sound = threading.Event()
        microphone, events, closed = self._start(sound)
        try:
            self._next_event(events, "connected")
            microphone.set_listening(True)
            sound.set()
            clip = self._next_event(events, "clip")
            self.assertIsNone(clip["request_id"])
            self.assertEqual(len(clip["samples"]), 19200)
        finally:
            microphone.stop()
        self.assertTrue(closed.is_set())
        self.assertFalse(microphone.alive)

    def test_requested_record_emits_matching_request_id(self) -> None:
        sound = threading.Event()
        microphone, events, _closed = self._start(sound)
        try:
            self._next_event(events, "connected")
            microphone.record(41)
            sound.set()
            clip = self._next_event(events, "clip")
            self.assertEqual(clip["request_id"], 41)
            self.assertEqual(len(clip["samples"]), 19200)
        finally:
            microphone.stop()

    def test_requested_record_times_out_without_onset(self) -> None:
        microphone, events, _closed = self._start()
        try:
            self._next_event(events, "connected")
            microphone.record(7)
            event = self._next_event(events, "timeout")
            self.assertEqual(event["request_id"], 7)
        finally:
            microphone.stop()

    def test_detached_capture_reports_error_without_restart(self) -> None:
        microphone, events, _closed = self._start(fail_after_ready=True)
        self.assertEqual(self._next_event(events, "connected")["kind"],
                         "connected")
        error = self._next_event(events, "error")
        self.assertIn("detached", error["message"])
        self._next_event(events, "stopped")
        self.assertFalse(microphone.alive)

    def test_stop_closes_a_blocked_capture(self) -> None:
        microphone, events, closed = self._start(block_after_ready=True)
        self._next_event(events, "connected")
        microphone.stop()
        self.assertTrue(closed.is_set())
        self._next_event(events, "stopped")
        self.assertFalse(microphone.alive)

    def test_full_event_queue_keeps_the_clip_event(self) -> None:
        events: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=2)
        events.put({"kind": "level", "db": -40.0})
        events.put({"kind": "status", "message": "working"})
        microphone = SoundMicrophone(events, "test-input")
        microphone._emit({"kind": "clip", "request_id": None,
                          "samples": [0.0] * 19200})
        queued = list(events.queue)
        self.assertIn("clip", [event["kind"] for event in queued])
        self.assertNotIn("level", [event["kind"] for event in queued])

    def test_pausing_discards_a_partial_live_clip(self) -> None:
        closed = threading.Event()
        sound_on = threading.Event()
        second_loud_read = threading.Event()
        release_read = threading.Event()
        loud_reads = 0

        def create_capture(*_args: Any, **_kwargs: Any) -> Any:
            def read_frame() -> tuple[list[float], float]:
                nonlocal loud_reads
                time.sleep(0.001)
                if closed.is_set():
                    return [0.0] * 640, time.monotonic()
                if sound_on.is_set() and loud_reads < 2:
                    loud_reads += 1
                    if loud_reads == 2:
                        second_loud_read.set()
                        release_read.wait()
                    values = [0.04 if i % 2 else -0.04
                              for i in range(640)]
                else:
                    values = [0.0] * 640
                return values, time.monotonic()

            capture = SimpleNamespace()
            capture.__enter__ = lambda: capture
            capture.close = closed.set
            capture.__exit__ = lambda *_args: closed.set()
            capture.read_frame = read_frame
            return capture

        microphone, events, _closed = self._start(
            capture_factory=create_capture)
        try:
            self._next_event(events, "connected")
            microphone.set_listening(True)
            sound_on.set()
            self.assertTrue(second_loud_read.wait(2.0))
            microphone.set_listening(False)
            release_read.set()
            time.sleep(0.02)
            microphone.set_listening(True)
            time.sleep(0.02)
            self.assertFalse(any(
                event["kind"] == "clip" for event in list(events.queue)))
        finally:
            release_read.set()
            microphone.stop()

    def test_recording_interrupts_a_partial_live_clip(self) -> None:
        closed = threading.Event()
        sound_on = threading.Event()
        second_loud_read = threading.Event()
        release_read = threading.Event()
        loud_reads = 0

        def create_capture(*_args: Any, **_kwargs: Any) -> Any:
            def read_frame() -> tuple[list[float], float]:
                nonlocal loud_reads
                time.sleep(0.001)
                if closed.is_set():
                    return [0.0] * 640, time.monotonic()
                if sound_on.is_set():
                    loud_reads += 1
                    if loud_reads == 2:
                        second_loud_read.set()
                        release_read.wait()
                    values = [0.04 if i % 2 else -0.04
                              for i in range(640)]
                else:
                    values = [0.0] * 640
                return values, time.monotonic()

            capture = SimpleNamespace()
            capture.__enter__ = lambda: capture
            capture.close = closed.set
            capture.__exit__ = lambda *_args: closed.set()
            capture.read_frame = read_frame
            return capture

        microphone, events, _closed = self._start(
            capture_factory=create_capture)
        try:
            self._next_event(events, "connected")
            microphone.set_listening(True)
            sound_on.set()
            self.assertTrue(second_loud_read.wait(2.0))
            microphone.record(22)
            release_read.set()
            clips: list[dict[str, Any]] = []
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                try:
                    event = events.get(timeout=0.05)
                except queue.Empty:
                    continue
                if event["kind"] == "clip":
                    clips.append(event)
                    if event.get("request_id") == 22:
                        break
            time.sleep(0.05)
        finally:
            release_read.set()
            microphone.stop()
        queued_clips = [event for event in list(events.queue)
                        if event["kind"] == "clip"]
        clips.extend(queued_clips)
        self.assertEqual([event.get("request_id") for event in clips], [22])

    def test_existing_builder_lock_fails_before_capture_starts(self) -> None:
        lock_dir = tempfile.TemporaryDirectory()
        self.addCleanup(lock_dir.cleanup)
        lock_path = os.path.join(lock_dir.name, "builder.lock")
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.addCleanup(os.close, lock_fd)
        self.addCleanup(fcntl.flock, lock_fd, fcntl.LOCK_UN)
        capture_started = threading.Event()
        events: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=8)

        def create_capture(*_args: Any, **_kwargs: Any) -> Any:
            capture_started.set()
            raise AssertionError("capture must not start while lock is held")

        microphone = SoundMicrophone(
            events, "test-input", capture_factory=create_capture,
            lock_path=lock_path)
        self.addCleanup(microphone.stop)
        microphone.start()
        error = self._next_event(events, "error")
        self.assertIn("busy", error["message"])
        self._next_event(events, "stopped")
        self.assertFalse(capture_started.is_set())


if __name__ == "__main__":
    unittest.main()
