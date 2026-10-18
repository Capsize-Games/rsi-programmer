"""Private process and file management for live microphone sessions."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rsi_programmer.audio_capture import PulseCapture, list_sources
from rsi_programmer.audio_features import extract_frame
from rsi_programmer.audio_session import AudioSession


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / "fixtures" / "audio_fan_live_v1.json"
BUILDER_LOCK = Path("/tmp/rsi-programmer-builder.lock")
STATE_PATH = Path.home() / ".local/state/rsi-programmer/microphone"
STATUS_EVERY_FRAMES = 5
SNAPSHOT_EVERY_MONITOR_FRAMES = 300
PREFLIGHT_FRAMES = 10


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def atomic_write_json(path: Path, value: Any) -> None:
    """Replace a JSON file atomically with owner-only permissions."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_json_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def resolve_source(source: str | None) -> tuple[str, str]:
    """Resolve a physical source identity and its friendly description."""
    sources = list_sources()
    if source is None:
        result = subprocess.run(
            ["pactl", "get-default-source"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        source = result.stdout.strip()
    if source.endswith(".monitor"):
        raise ValueError("monitor sources cannot be used for capture")
    match = next((row for row in sources if row["name"] == source), None)
    if match is None:
        raise ValueError("selected source is not an available physical input")
    return source, match["description"]


def new_session_dir(root: Path = STATE_PATH) -> Path:
    """Create a unique private output directory outside the repository."""
    root = root.expanduser().resolve()
    if root == PROJECT_ROOT or PROJECT_ROOT in root.parents:
        raise ValueError("audio state root must be outside the repository")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for counter in range(1000):
        suffix = f"-{counter:03d}" if counter else ""
        target = root / f"session-{stamp}{suffix}"
        try:
            target.mkdir(mode=0o700)
            return target
        except FileExistsError:
            continue
    raise RuntimeError("could not allocate a unique session directory")


class AudioRuntime:
    """Capture and persist one private live protocol session."""

    def __init__(
        self,
        output_dir: Path,
        source_identity: str,
        source_description: str,
        minutes: float | None = None,
        protocol_path: Path = PROTOCOL_PATH,
    ) -> None:
        self.output_dir = output_dir.expanduser().resolve()
        self.source_identity = source_identity
        self.source_description = source_description
        self.minutes = minutes
        self.protocol_path = protocol_path
        self._stop = False
        self._stop_reason = "requested"
        self._error: str | None = None
        self._session: AudioSession | None = None
        self._trace: Any = None
        self._quality_issues = 0
        self._last_signal: dict[str, float] | None = None
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._start_monotonic = time.monotonic()
        self._protocol_hash = ""
        self._git_head = self._read_git_head()
        self._original_handlers: dict[int, Any] = {}

    def _read_git_head(self) -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                check=True, capture_output=True, text=True, timeout=3.0,
            )
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    @property
    def status_path(self) -> Path:
        return self.output_dir / "status.json"

    @property
    def trace_path(self) -> Path:
        return self.output_dir / "feature_trace.jsonl"

    def _signal_handler(self, signum: int, frame: Any) -> None:
        del frame
        self._stop = True
        self._stop_reason = signal.Signals(signum).name

    def _install_handlers(self) -> None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._original_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._signal_handler)

    def _restore_handlers(self) -> None:
        for signum, handler in self._original_handlers.items():
            signal.signal(signum, handler)
        self._original_handlers.clear()

    def _runtime_status(self, state: str) -> dict[str, Any]:
        return {
            "status": state,
            "pid": os.getpid(),
            "session_dir": str(self.output_dir),
            "source_description": self.source_description,
            "source_identity": self.source_identity,
            "started_at": self._started_at,
            "delivery_seconds": max(
                0.0, time.monotonic() - self._start_monotonic
            ),
            "last_signal": self._last_signal,
            "quality_issue_count": self._quality_issues,
            "protocol_sha256": self._protocol_hash,
            "git_head": self._git_head,
            "error": self._error,
            "stop_reason": self._stop_reason if state == "stopped" else None,
        }

    def _publish(self, state: str) -> None:
        summary = self._session.summary() if self._session else {
            "phase": "initializing", "frame_count": 0,
            "monitor_frames": 0, "blocks": [], "checkpoints": [],
        }
        atomic_write_json(self.status_path, {
            **summary,
            "runtime": self._runtime_status(state),
        })
        atomic_write_json(self.output_dir / "pid.json", {
            "pid": os.getpid(),
            "session_dir": str(self.output_dir),
        })

    def _write_trace(self, row: dict[str, Any]) -> None:
        assert self._trace is not None
        self._trace.write(json.dumps(
            row, sort_keys=True, separators=(",", ":"), allow_nan=False
        ) + "\n")
        self._trace.flush()

    @staticmethod
    def _usable(features: dict[str, Any]) -> bool:
        return (
            float(features["rms_dbfs"]) > -80.0
            and float(features["clip_fraction"]) <= 0.01
        )

    @staticmethod
    def _aggregate_quality(
        features: list[dict[str, Any]],
    ) -> tuple[float, float]:
        rms_power = sum(
            10.0 ** (float(row["rms_dbfs"]) / 10.0) for row in features
        ) / len(features)
        clip_fraction = sum(
            float(row["clip_fraction"]) for row in features
        ) / len(features)
        rms_dbfs = 10.0 * math.log10(max(rms_power, 1e-30))
        return rms_dbfs, clip_fraction

    def _feed(
        self,
        features: dict[str, Any],
        delivery_seconds: float,
    ) -> None:
        assert self._session is not None
        self._last_signal = {
            "rms_dbfs": float(features["rms_dbfs"]),
            "peak_hz": float(features["peak_hz"]),
            "clip_fraction": float(features["clip_fraction"]),
        }
        if not self._usable(features):
            self._quality_issues += 1
            raise RuntimeError(
                "unusable microphone frame; run stopped without advancing "
                f"the model clock: rms_dbfs={features['rms_dbfs']:.2f}, "
                f"clip_fraction={features['clip_fraction']:.6f}"
            )
        previous_phase = self._session.phase
        row = self._session.observe(features, delivery_seconds)
        self._write_trace(row)
        if previous_phase == "fit" and self._session.phase == "calibration":
            snapshot = self._session.snapshot()
            fixed = snapshot["models"].get("fixed_snn")
            if fixed is not None:
                atomic_write_json(self.output_dir / "reference.json", {
                    "protocol_sha256": self._protocol_hash,
                    "fixed_hash": snapshot["fixed_hash"],
                    "model": fixed,
                })
        summary = self._session.summary()
        monitor_frames = int(summary["monitor_frames"])
        if self._session.ready and monitor_frames == 0:
            self._write_snapshot(monitor_frames)
        elif (
            self._session.ready
            and monitor_frames > 0
            and monitor_frames % SNAPSHOT_EVERY_MONITOR_FRAMES == 0
        ):
            self._write_snapshot(monitor_frames)
        block_frames = int(self._session.protocol["block_frames"])
        checkpoint = self._session.ready and (
            monitor_frames == 0
            or monitor_frames > 0 and monitor_frames % block_frames == 0
        )
        publish = (
            int(summary["frame_count"]) % STATUS_EVERY_FRAMES == 0
            or checkpoint
        )
        if publish:
            self._publish("running")
        # Retain early audits, then double the interval. Full-prefix replay
        # on every minute would accumulate quadratic background work.
        block_number = monitor_frames // block_frames
        should_audit = (block_number == 0
                        or block_number & (block_number - 1) == 0)
        if checkpoint and should_audit:
            self._audit_checkpoint(monitor_frames)

    def _write_snapshot(self, monitor_frames: int) -> None:
        assert self._session is not None
        value = {
            "monitor_frame": monitor_frames,
            "protocol_sha256": self._protocol_hash,
            "snapshot": self._session.snapshot(),
        }
        checkpoint = self.output_dir / "snapshots" / (
            f"monitor-{monitor_frames:09d}.json"
        )
        atomic_write_json(checkpoint, value)
        atomic_write_json(self.output_dir / "model_snapshot.json", value)

    def _audit_checkpoint(self, monitor_frames: int) -> None:
        """Independently audit the flushed trace and published summary."""
        from scripts.audit_audio_session import audit_session

        started = time.monotonic()
        result = audit_session(self.trace_path, self.status_path)
        value = {
            "monitor_frame": monitor_frames,
            "protocol_sha256": self._protocol_hash,
            "audit": result,
            "wall_seconds": time.monotonic() - started,
        }
        checkpoint = self.output_dir / "audits" / (
            f"monitor-{monitor_frames:09d}.json"
        )
        atomic_write_json(checkpoint, value)
        atomic_write_json(self.output_dir / "audit.json", value)

    def run(self) -> int:
        """Run capture until stopped, timed out, or the signal fails."""
        if (
            self.output_dir == PROJECT_ROOT
            or PROJECT_ROOT in self.output_dir.parents
        ):
            raise ValueError(
                "audio session output must be outside the repository"
            )
        if self.minutes is not None and (
            not math.isfinite(self.minutes) or self.minutes <= 0
        ):
            raise ValueError("minutes must be a positive finite number")
        if self.output_dir.exists():
            allowed = {"status.json", "run.log", "pid.json"}
            if any(item.name not in allowed
                   for item in self.output_dir.iterdir()):
                raise FileExistsError("session output directory is not empty")
            os.chmod(self.output_dir, 0o700)
        else:
            self.output_dir.mkdir(parents=True, mode=0o700)
        os.chmod(self.output_dir, 0o700)
        self._install_handlers()
        lock_stream = None
        capture: PulseCapture | None = None
        state = "initializing"
        try:
            lock_stream = BUILDER_LOCK.open("a+b")
            try:
                fcntl.flock(
                    lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                )
            except BlockingIOError as exc:
                raise RuntimeError(
                    "builder lock is held by another process"
                ) from exc
            payload = self.protocol_path.read_bytes()
            protocol = json.loads(payload.decode("utf-8"))
            if not isinstance(protocol, dict):
                raise ValueError("protocol must be a JSON object")
            self._protocol_hash = hashlib.sha256(payload).hexdigest()
            self._trace = self.trace_path.open("x", encoding="utf-8")
            os.chmod(self.trace_path, 0o600)
            self._publish("initializing")
            torch_set_threads()
            session = AudioSession(protocol)
            self._session = session
            capture = PulseCapture(
                self.source_identity,
                int(protocol["sample_rate"]),
                float(protocol["frame_seconds"]),
            )
            capture.__enter__()
            preflight: list[tuple[dict[str, Any], float]] = []
            for _ in range(PREFLIGHT_FRAMES):
                samples, delivered = capture.read_frame()
                features = extract_frame(samples, int(protocol["sample_rate"]))
                preflight.append((features, delivered))
            aggregate_rms, aggregate_clip = self._aggregate_quality(
                [item[0] for item in preflight]
            )
            if aggregate_rms <= -80.0 or aggregate_clip > 0.01:
                raise RuntimeError(
                    "microphone signal check failed: "
                    f"rms_dbfs={aggregate_rms:.2f}, "
                    f"clip_fraction={aggregate_clip:.6f}"
                )
            frame_seconds = float(protocol["frame_seconds"])
            frame_budget = (
                math.ceil(self.minutes * 60.0 / frame_seconds)
                if self.minutes is not None else None
            )
            for features, delivered in preflight:
                self._feed(features, delivered - self._start_monotonic)
            monitor_frames = int(session.summary()["monitor_frames"])
            if frame_budget is not None and monitor_frames >= frame_budget:
                self._stop_reason = "duration_elapsed"
                self._stop = True
            while not self._stop:
                samples, delivered = capture.read_frame()
                features = extract_frame(samples, int(protocol["sample_rate"]))
                self._feed(features, delivered - self._start_monotonic)
                monitor_frames = int(session.summary()["monitor_frames"])
                if frame_budget is not None and monitor_frames >= frame_budget:
                    self._stop_reason = "duration_elapsed"
                    self._stop = True
            state = "stopped"
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            state = "error"
        finally:
            if capture is not None:
                capture.close()
            if self._session is not None:
                try:
                    self._session.finish()
                    final_monitor_frames = int(
                        self._session.summary()["monitor_frames"]
                    )
                    final = {
                        "monitor_frame": final_monitor_frames,
                        "protocol_sha256": self._protocol_hash,
                        "snapshot": self._session.snapshot(),
                    }
                    atomic_write_json(
                        self.output_dir / "final_snapshot.json", final)
                    atomic_write_json(
                        self.output_dir / "model_snapshot.json", final)
                except Exception as exc:
                    self._error = self._error or (
                        f"final snapshot failed: {type(exc).__name__}: {exc}"
                    )
            if self._trace is not None:
                self._trace.close()
            try:
                self._publish(state)
            except Exception as exc:
                self._error = self._error or (
                    f"status write failed: {type(exc).__name__}: {exc}"
                )
            if lock_stream is not None:
                lock_stream.close()
            self._restore_handlers()
        return 0 if state == "stopped" else 1


def torch_set_threads() -> None:
    """Constrain CPU thread count for predictable background work."""
    import torch

    torch.set_num_threads(1)
