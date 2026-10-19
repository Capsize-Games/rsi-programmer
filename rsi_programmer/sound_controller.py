"""Coordinate explicit teaching, held-out checks and sound actions."""
from __future__ import annotations

import queue
import math
import subprocess
import time
import tkinter as tk
from collections.abc import Callable
from typing import Any

from rsi_programmer.audio_runtime import resolve_source
from rsi_programmer.sound_features import extract_sound
from rsi_programmer.sound_memory import LABEL_PATTERN, SoundMemory
from rsi_programmer.sound_microphone import SoundMicrophone
from rsi_programmer.sound_window import SoundWindow

ACTION_IDS = {
    "Wake": "wake", "Sleep": "sleep", "Dance": "dance",
    "Start timer": "timer", "wake": "wake", "sleep": "sleep",
    "dance": "dance", "timer": "timer",
}


class SoundController:
    """Keep microphone work off the Tk thread and tests out of training."""

    def __init__(
        self, root: tk.Tk, memory: SoundMemory,
        source: str | None = None,
        window_factory: Callable[..., Any] = SoundWindow,
        microphone_factory: Callable[..., Any] = SoundMicrophone,
    ) -> None:
        self.root = root
        self.memory = memory
        self.source = source
        self.events: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=100)
        self._microphone_factory = microphone_factory
        self.microphone: Any = None
        self.connected = False
        self.listening = False
        self._closed = False
        self._pending: dict[str, Any] | None = None
        self._next_request = 0
        self._resume_listening = False
        self._last_descriptor: dict[str, Any] | None = None
        self._last_action_time = -10.0
        self._timer_deadline: float | None = None
        self._microphone_error: str | None = None
        self.window = window_factory(root, {
            "connect": self.connect,
            "toggle_listening": self.toggle_listening,
            "teach": self.teach,
            "test": self.test,
            "correction": self.correct,
            "delete": self.delete,
        })
        self.window.set_memory(self.memory.summary())
        error = self.memory.summary().get("load_error")
        if error:
            self.window.set_status("Saved memory needs repair; file preserved")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(60, self.poll)

    def connect(self) -> None:
        """Open the selected physical microphone after an explicit click."""
        if self.microphone is not None and self.microphone.alive:
            return
        if self.memory.summary().get("load_error"):
            self.window.set_status("Memory could not load; file preserved")
            return
        try:
            source, _description = resolve_source(self.source)
            self.microphone = self._microphone_factory(self.events, source)
            self._microphone_error = None
            self.window.set_busy(True)
            self.window.set_status(
                "Calibrating background; stay quiet briefly")
            self.microphone.start()
        except (OSError, ValueError, RuntimeError,
                subprocess.SubprocessError) as error:
            self.window.set_busy(False)
            self.window.set_status(f"Microphone unavailable: {error}")

    def toggle_listening(self) -> None:
        """Arm actions only when connected, idle and holding taught sounds."""
        if not self.connected or self._pending is not None:
            return
        if not self.memory.summary()["total_examples"]:
            self.window.set_status("Teach a sound before starting listening")
            return
        self._set_listening(not self.listening)

    def _set_listening(self, enabled: bool) -> None:
        self.listening = enabled and self.connected
        if self.microphone is not None:
            self.microphone.set_listening(self.listening)
        self.window.set_listening(self.listening)
        self.window.set_status(
            "Listening for taught sounds" if self.listening
            else "Microphone enabled; sound actions paused")

    def teach(self, label: str, action: str) -> None:
        """Capture a new labeled example; do not dispatch its action."""
        label = label.strip()
        if not LABEL_PATTERN.fullmatch(label):
            self.window.set_status("Use a sound name of 1 to 32 characters")
            return
        action_id = ACTION_IDS.get(action)
        if action_id is None:
            self.window.set_status("Choose one of the listed actions")
            return
        existing = next((entry for entry in self.memory.summary()["labels"]
                         if entry["name"] == label), None)
        if existing and existing["action"] != action_id:
            self.window.set_status("Use this sound's existing action")
            return
        self._begin("teach", label, action_id)

    def test(self, expected: str) -> None:
        """Record a new probe without adding it to sound memory."""
        labels = {entry["name"] for entry in self.memory.summary()["labels"]}
        if expected not in labels:
            self.window.set_status("Choose a taught sound to check")
            return
        self._begin("test", expected, None)

    def _begin(self, mode: str, label: str, action: str | None) -> None:
        if not self.connected or self._pending is not None:
            return
        self._resume_listening = self.listening
        self._last_descriptor = None
        self.window.set_can_correct(False)
        self._set_listening(False)
        self._next_request += 1
        self._pending = {
            "id": self._next_request, "mode": mode,
            "label": label, "action": action,
        }
        self.window.set_busy(True)
        self.window.set_status("Get ready; make your sound after the cue")
        self.microphone.record(self._next_request)

    def correct(self, label: str) -> None:
        """Teach the recent clip only after the user selects its true label."""
        if self._last_descriptor is None or self._pending is not None:
            return
        entry = next((entry for entry in self.memory.summary()["labels"]
                      if entry["name"] == label), None)
        if entry is None:
            return
        try:
            self.memory.teach(label, self._last_descriptor, entry["action"])
            self._last_descriptor = None
            self.window.set_can_correct(False)
            self.window.set_memory(self.memory.summary())
            self.window.set_status(f"Correction saved as {label}")
        except (OSError, ValueError, RuntimeError) as error:
            self.window.set_status(f"Correction not saved: {error}")

    def delete(self, label: str) -> None:
        """Forget the selected label while preserving all other examples."""
        if self._pending is not None:
            return
        try:
            self.memory.delete_label(label)
            self.window.set_memory(self.memory.summary())
            if not self.memory.summary()["total_examples"]:
                self._set_listening(False)
            self.window.set_status(f"Forgot {label}")
        except (OSError, ValueError, RuntimeError) as error:
            self.window.set_status(f"Memory unchanged: {error}")

    def poll(self) -> None:
        """Apply worker events and timer updates on the Tk thread."""
        if self._closed:
            return
        for _ in range(100):
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            self.handle_event(event)
        if self._timer_deadline is not None:
            remaining = math.ceil(max(
                0.0, self._timer_deadline - time.monotonic()))
            self.window.set_timer(remaining)
            if remaining == 0:
                self._timer_deadline = None
                self.window.set_status("Timer finished")
                self.root.bell()
        self.root.after(60, self.poll)

    def handle_event(self, event: dict[str, Any]) -> None:
        """Process one bounded worker message without implicit teaching."""
        kind = event.get("kind")
        if kind == "connected":
            self.connected = True
            self.window.set_connected(True)
            self.window.set_busy(False)
            self.window.set_status("Microphone ready; teach your first sound")
        elif kind == "level":
            self.window.set_level(float(event["db"]))
        elif kind == "status":
            self.window.set_status(str(event["message"]))
        elif kind in ("error", "stopped"):
            if kind == "error":
                self._microphone_error = str(event.get(
                    "message", "Microphone capture failed"))
            self.connected = False
            self.listening = False
            self._pending = None
            self.window.set_connected(False)
            self.window.set_listening(False)
            self.window.set_busy(False)
            self.window.set_status(self._microphone_error or str(
                event.get("message", "Microphone off")))
        elif kind == "timeout":
            if (self._pending is not None
                    and event.get("request_id") == self._pending["id"]):
                self._complete_capture()
                self.window.set_status("No distinct sound; try again closer")
        elif kind == "clip":
            self._receive_clip(event)

    def _receive_clip(self, event: dict[str, Any]) -> None:
        request_id = event.get("request_id")
        pending = self._pending
        if request_id is None and (not self.listening or pending is not None):
            return
        if request_id is not None and (
                pending is None or request_id != pending["id"]):
            return
        try:
            descriptor = extract_sound(event["samples"])
            if pending is not None:
                if pending["mode"] == "teach":
                    result = self.memory.teach(
                        pending["label"], descriptor, pending["action"])
                    message = (f"Saved {result['label']}: "
                               f"{result['examples']} examples")
                else:
                    result = self.memory.record_probe(
                        pending["label"], descriptor)
                    self.window.set_test_result(result)
                    message = ("Test matched" if result["correct"]
                               else "Test did not match; memory unchanged")
                self.window.set_memory(self.memory.summary())
                self._complete_capture()
                self.window.set_status(message)
            else:
                self._last_descriptor = descriptor
                self.window.set_can_correct(bool(descriptor["active"]))
                result = self.memory.recognize(descriptor)
                self.window.set_recognition(result)
                now = time.monotonic()
                if result["action"] and now - self._last_action_time >= 2.0:
                    self._last_action_time = now
                    self.window.perform_action(result["action"])
                    self.window.set_status(
                        f"{result['label']} → {result['action']}")
                    if result["action"] == "timer":
                        self._timer_deadline = now + 60.0
                        self.window.set_timer(60)
        except (OSError, ValueError, TypeError, RuntimeError) as error:
            if pending is not None:
                self._complete_capture()
            self.window.set_status(f"Sound not saved or acted on: {error}")

    def _complete_capture(self) -> None:
        self._pending = None
        self.window.set_busy(False)
        self._set_listening(self._resume_listening)

    def close(self) -> None:
        """Release capture before closing the native window."""
        self._closed = True
        if self.microphone is not None:
            self.microphone.stop()
        self.root.destroy()
