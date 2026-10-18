"""Unit tests for private runtime state and detached process controls."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import tempfile
import time
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from rsi_programmer.audio_runtime import (
    AudioRuntime,
    PROJECT_ROOT,
    atomic_write_json,
    new_session_dir,
    resolve_source,
)
from scripts.fan_microphone import command_start, command_stop


class AudioRuntimeTests(unittest.TestCase):
    def test_bad_frame_fails_without_advancing_session(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(
                AudioRuntime, "_read_git_head", return_value=None
            ):
                runtime = AudioRuntime(
                    Path(temporary) / "session", "physical-input", "Yeti"
                )
            runtime._session = Mock(phase="fit")
            bad = {
                "rms_dbfs": -90.0,
                "peak_hz": 100.0,
                "clip_fraction": 0.0,
                "band_dbfs": [-100.0] * 6,
            }
            with self.assertRaisesRegex(RuntimeError, "without advancing"):
                runtime._feed(bad, 0.2)
            runtime._session.observe.assert_not_called()
            self.assertEqual(runtime._quality_issues, 1)

    def test_state_root_inside_repo_is_rejected_before_creation(self) -> None:
        target = PROJECT_ROOT / ".runtime-test-must-not-exist"
        with self.assertRaises(ValueError):
            new_session_dir(target)
        self.assertFalse(target.exists())

    def test_mocked_runtime_reaches_one_audited_monitor_block(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_path = root / "protocol.json"
            protocol = {
                "sample_rate": 16000,
                "frame_seconds": 0.2,
                "fit_frames": 3,
                "calibration_frames": 2,
                "probe_frames": 2,
                "block_frames": 3,
            }
            protocol_path.write_text(json.dumps(protocol))
            output = root / "session"
            features = {
                "rms_dbfs": -24.0,
                "peak_dbfs": -12.0,
                "clip_fraction": 0.0,
                "peak_hz": 150.0,
                "tone_snr_db": 10.0,
                "band_dbfs": [-50.0] * 6,
            }
            state: dict[str, object] = {"count": 0, "phase": "fit"}
            session = Mock()
            session.protocol = protocol
            session.phase = "fit"
            session.ready = False

            def observe(
                row_features: dict[str, object],
                received_seconds: float,
            ) -> dict[str, object]:
                count = int(state["count"]) + 1
                phase = str(state["phase"])
                phase = {
                    3: "calibration", 5: "probe", 7: "monitor"
                }.get(count, phase)
                state.update({"count": count, "phase": phase})
                session.phase = phase
                session.ready = phase == "monitor"
                return {
                    "i": count - 1,
                    "phase": phase,
                    "features": row_features,
                    "received_seconds": received_seconds,
                }

            def summary() -> dict[str, object]:
                count = int(state["count"])
                monitor_count = max(0, count - 7)
                blocks = [{"block": 1}] if monitor_count >= 3 else []
                return {
                    "phase": state["phase"],
                    "frame_count": count,
                    "monitor_frames": monitor_count,
                    "blocks": blocks,
                    "checkpoints": [],
                    "adaptive_vs_fixed": {},
                }

            def snapshot() -> dict[str, object]:
                return {
                    "phase": state["phase"],
                    "models": {"fixed_snn": {"weights": []}},
                    "fixed_hash": "fixed-hash",
                }

            def finish() -> None:
                state["phase"] = "finished"
                session.phase = "finished"
                session.ready = False

            session.observe.side_effect = observe
            session.summary.side_effect = summary
            session.snapshot.side_effect = snapshot
            session.finish.side_effect = finish
            capture = MagicMock()
            capture.__enter__.return_value = capture
            frame_index = {"value": 0}

            def read_frame() -> tuple[list[float], float]:
                frame_index["value"] += 1
                delivered = time.monotonic() + frame_index["value"] * 0.2
                return [0.1] * 3200, delivered

            capture.read_frame.side_effect = read_frame
            with (
                patch("rsi_programmer.audio_runtime.BUILDER_LOCK",
                      root / "runtime.lock"),
                patch("rsi_programmer.audio_runtime.PulseCapture",
                      return_value=capture),
                patch("rsi_programmer.audio_runtime.extract_frame",
                      return_value=features),
                patch("rsi_programmer.audio_runtime.AudioSession",
                      return_value=session),
                patch("rsi_programmer.audio_runtime.torch_set_threads"),
                patch.object(
                    AudioRuntime, "_read_git_head", return_value=None
                ),
                patch(
                    "scripts.audit_audio_session.audit_session",
                    return_value={"valid": True},
                ) as audit,
            ):
                runtime = AudioRuntime(
                    output, "physical-input", "Yeti Microphone",
                    minutes=0.01, protocol_path=protocol_path,
                )
                self.assertEqual(runtime.run(), 0)
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["frame_count"], 10)
            self.assertEqual(status["monitor_frames"], 3)
            self.assertEqual(status["runtime"]["status"], "stopped")
            rows = (output / "feature_trace.jsonl").read_text().splitlines()
            self.assertEqual(len(rows), 10)
            self.assertTrue((output / "reference.json").exists())
            self.assertTrue((output / "model_snapshot.json").exists())
            self.assertTrue((output / "audit.json").exists())
            self.assertEqual(audit.call_count, 2)

    def test_atomic_json_uses_private_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "private"
            path = root / "status.json"
            atomic_write_json(path, {"status": "running"})
            self.assertEqual(
                json.loads(path.read_text()), {"status": "running"}
            )
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)

    def test_resolves_explicit_physical_source(self) -> None:
        with patch(
            "rsi_programmer.audio_runtime.list_sources",
            return_value=[{
                "name": "alsa_input.usb-mic",
                "description": "Yeti Stereo Microphone",
            }],
        ):
            self.assertEqual(
                resolve_source("alsa_input.usb-mic"),
                ("alsa_input.usb-mic", "Yeti Stereo Microphone"),
            )

    def test_rejects_monitor_or_unknown_source(self) -> None:
        with patch(
            "rsi_programmer.audio_runtime.list_sources", return_value=[]
        ):
            with self.assertRaises(ValueError):
                resolve_source("sink.monitor")
            with self.assertRaises(ValueError):
                resolve_source("missing-input")

    def test_start_launches_detached_run_and_records_latest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "state"
            process = Mock(pid=4321)
            args = Namespace(
                source="alsa_input.usb-mic",
                minutes=0.5,
                state_root=str(root),
            )
            with (
                patch(
                    "scripts.fan_microphone.resolve_source",
                    return_value=("alsa_input.usb-mic", "Yeti Microphone"),
                ),
                patch("scripts.fan_microphone.subprocess.Popen",
                      return_value=process) as popen,
                patch("builtins.print"),
            ):
                self.assertEqual(command_start(args), 0)
            call = popen.call_args
            self.assertTrue(call.kwargs["start_new_session"])
            self.assertEqual(call.kwargs["stdin"], subprocess.DEVNULL)
            latest = json.loads((root / "latest.json").read_text())
            self.assertEqual(latest["pid"], 4321)
            self.assertTrue(Path(latest["session_dir"]).is_dir())

    def test_stop_signals_only_a_validated_owned_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "state"
            session_dir = root / "session-one"
            session_dir.mkdir(parents=True)
            record = {"pid": 1234, "session_dir": str(session_dir)}
            (root / "latest.json").write_text(json.dumps(record))
            (session_dir / "pid.json").write_text(json.dumps(record))
            args = Namespace(state_root=str(root))
            with (
                patch("scripts.fan_microphone._owned_run", return_value=False),
                patch("scripts.fan_microphone.os.kill") as kill,
                patch("builtins.print"),
            ):
                self.assertEqual(command_stop(args), 0)
                kill.assert_not_called()
            with (
                patch("scripts.fan_microphone._owned_run", return_value=True),
                patch("scripts.fan_microphone.os.kill") as kill,
                patch("builtins.print"),
            ):
                self.assertEqual(command_stop(args), 0)
                kill.assert_called_once_with(1234, 15)


if __name__ == "__main__":
    unittest.main()
