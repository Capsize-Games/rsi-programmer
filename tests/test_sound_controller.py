"""Verify that teaching, tests and armed actions remain separate."""
from __future__ import annotations

import math
import unittest
from unittest.mock import Mock, patch

from rsi_programmer.sound_controller import SoundController
from rsi_programmer.sound_features import extract_sound
from rsi_programmer.sound_memory import SoundMemory


def tone(frequency: float) -> list[float]:
    return [0.3 * math.sin(2 * math.pi * frequency * index / 16000)
            for index in range(19200)]


class SoundControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Mock()
        self.view = Mock()
        self.memory = SoundMemory()
        self.memory.teach("low", extract_sound(tone(440)), "wake")
        self.memory.teach("high", extract_sound(tone(1760)), "sleep")
        self.controller = SoundController(
            self.root, self.memory,
            window_factory=lambda *_args: self.view)
        self.controller.microphone = Mock()
        self.controller.connected = True

    def _complete_request(self, samples: list[float]) -> None:
        self.controller.handle_event({
            "kind": "clip", "samples": samples,
            "request_id": self.controller._next_request,
        })

    def test_teaching_adds_memory_without_firing_an_action(self) -> None:
        self.controller.teach("low", "Wake")
        self._complete_request(tone(440))
        self.assertEqual(self.memory.summary()["total_examples"], 3)
        self.view.perform_action.assert_not_called()

    def test_probe_cannot_be_reused_as_a_correction_example(self) -> None:
        self.controller.test("low")
        self._complete_request(tone(1760))
        before = self.memory.fingerprint()
        self.controller.correct("low")
        self.assertEqual(self.memory.fingerprint(), before)
        self.view.set_can_correct.assert_called_with(False)
        self.assertFalse(self.controller.listening)

    def test_incorrect_probe_is_not_training_or_an_action(self) -> None:
        before = self.memory.fingerprint()
        self.controller.test("low")
        self._complete_request(tone(1760))
        result = self.view.set_test_result.call_args.args[0]
        self.assertEqual(result["recognized"], "high")
        self.assertFalse(result["correct"])
        self.assertIsNone(result["action"])
        self.assertEqual(self.memory.fingerprint(), before)
        self.view.perform_action.assert_not_called()

    def test_live_sound_only_acts_when_listening_is_armed(self) -> None:
        event = {"kind": "clip", "request_id": None,
                 "samples": tone(440)}
        self.controller.handle_event(event)
        self.view.perform_action.assert_not_called()
        self.controller.toggle_listening()
        self.controller.handle_event(event)
        self.view.perform_action.assert_called_once_with("wake")

    def test_quiet_unknown_sound_never_acts(self) -> None:
        self.controller.toggle_listening()
        self.controller.handle_event({
            "kind": "clip", "request_id": None,
            "samples": [0.0] * 19200,
        })
        result = self.view.set_recognition.call_args.args[0]
        self.assertIsNone(result["label"])
        self.view.perform_action.assert_not_called()

    def test_stale_request_cannot_teach_after_timeout(self) -> None:
        self.controller.teach("low", "Wake")
        request_id = self.controller._next_request
        self.controller.handle_event({"kind": "timeout",
                                      "request_id": request_id})
        self.controller.handle_event({"kind": "clip",
                                      "request_id": request_id,
                                      "samples": tone(440)})
        self.assertEqual(self.memory.summary()["total_examples"], 2)

    def test_correction_uses_explicit_selected_label(self) -> None:
        self.controller.toggle_listening()
        self.controller.handle_event({"kind": "clip", "request_id": None,
                                      "samples": tone(1760)})
        old_high = self.memory.label_fingerprints()["high"]
        self.controller.correct("low")
        counts = {row["name"]: row["examples"]
                  for row in self.memory.summary()["labels"]}
        self.assertEqual(counts, {"high": 1, "low": 2})
        self.assertEqual(self.memory.label_fingerprints()["high"], old_high)

    def test_capture_error_disarms_pending_and_live_actions(self) -> None:
        self.controller.toggle_listening()
        self.controller.test("low")
        self.controller.handle_event({"kind": "error", "message": "Lost mic"})
        self.assertFalse(self.controller.connected)
        self.assertFalse(self.controller.listening)
        self.assertIsNone(self.controller._pending)
        self.view.set_listening.assert_called_with(False)

    def test_timer_finishes_and_close_releases_capture(self) -> None:
        with patch("rsi_programmer.sound_controller.time.monotonic",
                   return_value=100.5):
            self.controller._timer_deadline = 100.0
            self.controller.poll()
        self.assertIsNone(self.controller._timer_deadline)
        self.root.bell.assert_called_once()
        self.controller.close()
        self.controller.microphone.stop.assert_called_once()
        self.root.destroy.assert_called_once()

    def test_failed_source_resolution_is_displayed(self) -> None:
        self.controller.microphone = None
        with patch("rsi_programmer.sound_controller.resolve_source",
                   side_effect=OSError("No source")):
            self.controller.connect()
        self.assertIn("unavailable", self.view.set_status.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
