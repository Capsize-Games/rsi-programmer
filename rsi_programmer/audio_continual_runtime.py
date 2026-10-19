"""Private runtime for the frozen continual-adaptation experiment."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from rsi_programmer.audio_runtime import AudioRuntime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTINUAL_PROTOCOL_PATH = (
    PROJECT_ROOT / "fixtures" / "audio_continual_v1.json"
)
CONTINUAL_PROTOCOL_SHA256 = (
    "ff4a161882904aca6ba31a8498a859a1347addaa289e989ec0ea1e231a70fb58"
)
CONTINUAL_MONITOR_FRAMES = 9000
CONTINUAL_MINUTES = 30.0


class AudioContinualRuntime(AudioRuntime):
    """Run the immutable continual-memory comparison for 30 minutes."""

    def __init__(
        self,
        output_dir: Path,
        source_identity: str,
        source_description: str,
        minutes: float | None = None,
    ) -> None:
        if minutes is not None and (
            isinstance(minutes, bool)
            or not isinstance(minutes, (int, float))
            or not math.isfinite(float(minutes))
            or float(minutes) != CONTINUAL_MINUTES
        ):
            raise ValueError("continual variant runs for exactly 30 minutes")
        super().__init__(
            output_dir=output_dir,
            source_identity=source_identity,
            source_description=source_description,
            minutes=CONTINUAL_MINUTES,
            protocol_path=CONTINUAL_PROTOCOL_PATH,
        )

    def _make_session(self, protocol: dict[str, Any]) -> Any:
        """Create the continual session only for the frozen protocol bytes."""
        if self._protocol_hash != CONTINUAL_PROTOCOL_SHA256:
            raise ValueError(
                "continual protocol hash does not match frozen v1"
            )
        if (
            protocol.get("monitor_frames") != CONTINUAL_MONITOR_FRAMES
            or protocol.get("frame_seconds") != 0.2
            or protocol.get("seed") != 0
        ):
            raise ValueError(
                "continual protocol settings do not match frozen v1"
            )
        from rsi_programmer.audio_continual import AudioContinualSession

        return AudioContinualSession(protocol)

    def _runtime_status(self, state: str) -> dict[str, Any]:
        return {**super()._runtime_status(state), "variant": "continual"}

    def _audit_checkpoint(self, monitor_frames: int) -> None:
        """Audit continual-arm rows with their independent protocol checker."""
        from scripts.audit_audio_continual import audit_session

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
        from rsi_programmer.audio_runtime import atomic_write_json

        atomic_write_json(checkpoint, value)
        atomic_write_json(self.output_dir / "audit.json", value)

    def _should_audit(self, block_number: int,
                      monitor_frames: int) -> bool:
        """Audit periodic checkpoints and the frozen run's terminal bound."""
        return (
            super()._should_audit(block_number, monitor_frames)
            or monitor_frames == CONTINUAL_MONITOR_FRAMES
        )
