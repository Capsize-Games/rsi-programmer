"""Mocked integration checks for the continual runtime selection path."""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock, patch

from rsi_programmer.audio_continual_runtime import (
    CONTINUAL_MINUTES,
    CONTINUAL_PROTOCOL_SHA256,
    AudioContinualRuntime,
)
from rsi_programmer.audio_runtime import AudioRuntime
from scripts.fan_microphone import command_run, command_start


class AudioContinualRuntimeTests(unittest.TestCase):
    def test_default_runtime_factory_keeps_original_patch_point(self) -> None:
        session = Mock()
        with (
            patch.object(AudioRuntime, "_read_git_head", return_value=None),
            patch("rsi_programmer.audio_runtime.AudioSession",
                  return_value=session) as factory,
        ):
            runtime = AudioRuntime(Path("/tmp/audio-original"), "mic", "Mic")
            self.assertIs(runtime._make_session({"seed": 0}), session)
        factory.assert_called_once_with({"seed": 0})

    def test_continual_factory_requires_frozen_protocol_hash(self) -> None:
        session = Mock()
        module = types.ModuleType("rsi_programmer.audio_continual")
        module.AudioContinualSession = Mock(return_value=session)
        with (
            patch.object(AudioRuntime, "_read_git_head", return_value=None),
            patch.dict(
                sys.modules, {"rsi_programmer.audio_continual": module}
            ),
        ):
            runtime = AudioContinualRuntime(
                Path("/tmp/audio-continual"), "mic", "Mic"
            )
            runtime._protocol_hash = CONTINUAL_PROTOCOL_SHA256
            self.assertIs(runtime._make_session({"monitor_frames": 9000,
                                                 "frame_seconds": 0.2,
                                                 "seed": 0}), session)
            runtime._protocol_hash = "0" * 64
            with self.assertRaisesRegex(ValueError, "protocol hash"):
                runtime._make_session({"monitor_frames": 9000,
                                       "frame_seconds": 0.2, "seed": 0})

    def test_continual_runtime_has_fixed_30_minute_budget(self) -> None:
        with patch.object(AudioRuntime, "_read_git_head", return_value=None):
            runtime = AudioContinualRuntime(
                Path("/tmp/audio-continual"), "mic", "Mic"
            )
            self.assertEqual(runtime.minutes, CONTINUAL_MINUTES)
        with self.assertRaisesRegex(ValueError, "exactly 30 minutes"):
            AudioContinualRuntime(
                Path("/tmp/audio-continual"), "mic", "Mic", minutes=29.0
            )

    def test_run_command_selects_continual_runtime(self) -> None:
        runtime = Mock()
        runtime.run.return_value = 0
        factory = Mock(return_value=runtime)
        args = Namespace(
            variant="continual",
            minutes=None,
            output="/tmp/audio-continual",
            source="alsa_input.usb-mic",
        )
        with (
            patch(
                "scripts.fan_microphone._runtime_type",
                return_value=factory,
            ),
            patch(
                "scripts.fan_microphone.resolve_source",
                return_value=("alsa_input.usb-mic", "Yeti Microphone"),
            ),
        ):
            self.assertEqual(command_run(args), 0)
        factory.assert_called_once()
        runtime.run.assert_called_once_with()

    def test_continual_audit_uses_its_protocol_checker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "session"
            with patch.object(
                AudioRuntime, "_read_git_head", return_value=None
            ):
                runtime = AudioContinualRuntime(
                    output, "physical-input", "Yeti"
                )
            runtime._protocol_hash = CONTINUAL_PROTOCOL_SHA256
            auditor = Mock(return_value={"valid": True})
            module = types.ModuleType("scripts.audit_audio_continual")
            module.audit_session = auditor
            with patch.dict(
                sys.modules, {"scripts.audit_audio_continual": module}
            ):
                runtime._audit_checkpoint(0)
            value = json.loads((output / "audit.json").read_text())
            self.assertEqual(value["protocol_sha256"],
                             CONTINUAL_PROTOCOL_SHA256)
            self.assertEqual(value["audit"], {"valid": True})
            auditor.assert_called_once_with(
                runtime.trace_path, runtime.status_path
            )

    def test_start_propagates_continual_variant(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "state"
            process = Mock(pid=9876)
            args = Namespace(
                source="alsa_input.usb-mic",
                minutes=None,
                state_root=str(root),
                variant="continual",
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
            command = popen.call_args.args[0]
            self.assertIn("--variant", command)
            self.assertIn("continual", command)
            self.assertTrue(popen.call_args.kwargs["start_new_session"])
            latest = json.loads((root / "latest.json").read_text())
            self.assertEqual(latest["variant"], "continual")

    def test_original_quality_failure_preserves_model_clock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(
                AudioRuntime, "_read_git_head", return_value=None
            ):
                runtime = AudioRuntime(
                    Path(temporary) / "session", "mic", "Mic"
                )
            runtime._session = Mock(phase="fit")
            features = {
                "rms_dbfs": -90.0,
                "peak_hz": 100.0,
                "clip_fraction": 0.0,
                "band_dbfs": [-100.0] * 6,
            }
            with self.assertRaisesRegex(RuntimeError, "without advancing"):
                runtime._feed(features, 0.2)
            runtime._session.observe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
