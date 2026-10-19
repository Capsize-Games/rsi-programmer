"""Background microphone capture for the native sound companion UI."""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
import fcntl
import math
import os
import queue
import threading
import time
from typing import Any

from rsi_programmer.audio_capture import PulseCapture


class SoundMicrophone:
    """Capture fixed microphone frames and publish bounded UI events."""

    SAMPLE_RATE = 16000
    FRAME_SECONDS = 0.04
    FRAME_SAMPLES = 640
    CLIP_FRAMES = 30
    CLIP_SAMPLES = FRAME_SAMPLES * CLIP_FRAMES
    AMBIENT_FRAMES = 30
    PREBUFFER_FRAMES = 3
    LOCK_PATH = "/tmp/rsi-programmer-builder.lock"
    EVENT_LIMIT = 64

    def __init__(
        self,
        events: queue.Queue[dict[str, Any]],
        source: str,
        capture_factory: Callable[..., Any] = PulseCapture,
        clock: Callable[[], float] = time.monotonic,
        lock_path: str = LOCK_PATH,
    ) -> None:
        if not source:
            raise ValueError("an explicit audio source is required")
        self.events = events
        if self.events.maxsize <= 0:
            self.events.maxsize = self.EVENT_LIMIT
        self.source = source
        self._capture_factory = capture_factory
        self._clock = clock
        self._lock_path = lock_path
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._command_lock = threading.Lock()
        self._listening = False
        self._pending_request: int | None = None
        self._thread: threading.Thread | None = None
        self._capture: Any | None = None
        self._active_request: int | None = None
        self._lock_fd: int | None = None

    @property
    def alive(self) -> bool:
        """Whether the capture worker is still running."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start one capture worker if it is not already running."""
        if self.alive:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="sound-microphone", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop capture, closing the stream to interrupt a blocked read."""
        self._stop_event.set()
        capture = self._capture
        if capture is not None:
            try:
                capture.close()
            except Exception:
                pass
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    def set_listening(self, listening: bool) -> None:
        """Enable or disable live-onset clips without changing capture."""
        with self._state_lock:
            self._listening = bool(listening)

    def record(self, request_id: int) -> None:
        """Queue one deliberate clip request for the worker to capture."""
        if isinstance(request_id, bool) or not isinstance(request_id, int):
            raise TypeError("request_id must be an integer")
        with self._command_lock:
            if (self._pending_request is not None
                    or self._active_request is not None):
                self._emit({"kind": "status",
                            "message": "A recording is already in progress"})
                return
            self._pending_request = request_id

    def _emit(self, event: dict[str, Any]) -> None:
        """Publish bounded events while prioritizing clips and worker state."""
        kind = event.get("kind")
        critical = kind in {"connected", "clip", "timeout", "error",
                            "stopped"}
        with self.events.mutex:
            if self.events._qsize() < self.events.maxsize:
                self.events._put(event)
                self.events.unfinished_tasks += 1
                self.events.not_empty.notify()
                return
            if kind == "level":
                return
            removable = next((queued for queued in self.events.queue
                              if queued.get("kind") == "level"), None)
            if removable is None and critical:
                removable = self.events.queue[0]
            if removable is None:
                removable = next((queued for queued in self.events.queue
                                  if queued.get("kind") == "status"), None)
            if removable is None:
                return
            self.events.queue.remove(removable)
            if self.events.unfinished_tasks:
                self.events.unfinished_tasks -= 1
            self.events._put(event)
            self.events.unfinished_tasks += 1
            self.events.not_empty.notify()
            self.events.not_full.notify()

    def _run(self) -> None:
        lock_fd: int | None = None
        capture: Any | None = None
        entered = False
        try:
            lock_fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Microphone is busy with another capture "
                                   "operation") from exc
            self._lock_fd = lock_fd
            capture = self._capture_factory(
                self.source, sample_rate=self.SAMPLE_RATE,
                frame_seconds=self.FRAME_SECONDS)
            self._capture = capture
            capture.__enter__()
            entered = True
            self._calibrate(capture)
            if not self._stop_event.is_set():
                self._capture_loop(capture)
        except Exception as exc:
            if not self._stop_event.is_set():
                self._emit({"kind": "error", "message": str(exc) or
                            "Microphone capture failed"})
        finally:
            if capture is not None:
                try:
                    if entered:
                        capture.__exit__(None, None, None)
                    else:
                        capture.close()
                except Exception:
                    pass
            self._capture = None
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                finally:
                    os.close(lock_fd)
            self._lock_fd = None
            self._emit({"kind": "stopped"})

    def _calibrate(self, capture: Any) -> None:
        levels: list[float] = []
        for _ in range(self.AMBIENT_FRAMES):
            if self._stop_event.is_set():
                return
            samples, _timestamp = capture.read_frame()
            levels.append(self._rms(samples))
        ambient_db = self._db(self._rms_from_levels(levels))
        self._ambient_db = ambient_db
        if not self._stop_event.is_set():
            self._emit({"kind": "connected", "ambient_db": ambient_db})

    def _capture_loop(self, capture: Any) -> None:
        prebuffer: deque[list[float]] = deque(
            maxlen=self.PREBUFFER_FRAMES)
        frame_index = 0
        last_level_at = self._clock()
        request_id: int | None = None
        arm_frames = 0
        onset_frames = 0
        clip: list[float] = []
        collecting_live = False
        await_release = False
        cooldown_frames = 0
        quiet_frames = 0
        threshold_db = max(-58.0, self._ambient_db + 5.0)
        while not self._stop_event.is_set():
            self._accept_request()
            if self._active_request is not None and request_id is None:
                request_id = self._active_request
                arm_frames = round(1.0 / self.FRAME_SECONDS)
                onset_frames = 0
                clip = []
                collecting_live = False
                self._emit({"kind": "status", "message": "Get ready"})
            samples, _timestamp = capture.read_frame()
            frame_index += 1
            level_db = self._db(self._rms(samples))
            now = self._clock()
            if now - last_level_at >= 0.2:
                self._emit({"kind": "level", "db": level_db,
                            "ambient_db": self._ambient_db})
                last_level_at = now
            if request_id is not None:
                if arm_frames > 0:
                    arm_frames -= 1
                    prebuffer.append(list(samples))
                    if arm_frames == 0:
                        onset_frames = round(5.0 / self.FRAME_SECONDS)
                        self._emit({"kind": "status",
                                    "message": "Make sound now"})
                elif level_db >= threshold_db:
                    clip = [part for buffered in prebuffer
                            for part in buffered]
                    clip.extend(samples)
                    clip.extend(self._take_clip(
                        capture, self.CLIP_SAMPLES - len(clip)))
                    if not self._stop_event.is_set():
                        self._emit({"kind": "clip", "request_id": request_id,
                                    "samples": clip[:self.CLIP_SAMPLES]})
                    request_id = None
                    with self._command_lock:
                        self._active_request = None
                    prebuffer.clear()
                    await_release = True
                    quiet_frames = 0
                else:
                    onset_frames -= 1
                    prebuffer.append(list(samples))
                    if onset_frames <= 0:
                        self._emit({"kind": "timeout",
                                    "request_id": request_id})
                        request_id = None
                        with self._command_lock:
                            self._active_request = None
                        prebuffer.clear()
                continue

            listening = self._is_listening()
            if not listening:
                prebuffer.clear()
                collecting_live = False
                clip = []
                continue
            loud = level_db >= threshold_db
            if collecting_live:
                clip.extend(samples)
                if len(clip) >= self.CLIP_SAMPLES:
                    self._emit({"kind": "clip", "request_id": None,
                                "samples": clip[:self.CLIP_SAMPLES]})
                    collecting_live = False
                    clip = []
                    await_release = True
                    quiet_frames = 0
                continue
            if await_release:
                quiet_frames = 0 if loud else quiet_frames + 1
                if quiet_frames >= 3:
                    await_release = False
                    cooldown_frames = round(0.4 / self.FRAME_SECONDS)
                    prebuffer.clear()
                continue
            if cooldown_frames > 0:
                cooldown_frames -= 1
                quiet_frames = 0 if loud else quiet_frames + 1
                if cooldown_frames == 0 and quiet_frames >= 3:
                    prebuffer.clear()
                continue
            if loud:
                clip = [part for buffered in prebuffer for part in buffered]
                clip.extend(samples)
                collecting_live = True
                prebuffer.clear()
            else:
                prebuffer.append(list(samples))
            if frame_index % 5 == 0:
                prebuffer = deque(prebuffer, maxlen=self.PREBUFFER_FRAMES)

    def _accept_request(self) -> None:
        with self._command_lock:
            if (self._pending_request is not None
                    and self._active_request is None):
                self._active_request = self._pending_request
                self._pending_request = None

    def _take_clip(self, capture: Any, sample_count: int) -> list[float]:
        remaining = sample_count
        clip: list[float] = []
        while remaining > 0 and not self._stop_event.is_set():
            samples, _timestamp = capture.read_frame()
            take = min(remaining, len(samples))
            clip.extend(samples[:take])
            remaining -= take
        return clip

    def _is_listening(self) -> bool:
        with self._state_lock:
            return self._listening

    @staticmethod
    def _rms(samples: list[float]) -> float:
        if not samples:
            return 0.0
        return math.sqrt(
            sum(value * value for value in samples) / len(samples))

    @staticmethod
    def _rms_from_levels(levels: list[float]) -> float:
        if not levels:
            return 0.0
        return math.sqrt(sum(value * value for value in levels) / len(levels))

    @staticmethod
    def _db(rms: float) -> float:
        return 20.0 * math.log10(max(rms, 1e-8))
