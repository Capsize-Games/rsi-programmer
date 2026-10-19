"""Exercise real Tk controls on a private test display."""
from __future__ import annotations

import tkinter as tk
import unittest
from unittest.mock import Mock

from rsi_programmer.sound_window import SoundWindow


class SoundWindowTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk display unavailable: {error}")
        self.root.withdraw()
        self.callbacks = {name: Mock() for name in (
            "connect", "toggle_listening", "teach", "test",
            "correction", "delete")}
        self.window = SoundWindow(self.root, self.callbacks)
        self.window.set_memory({"labels": [
            {"name": "tap", "action": "wake", "examples": 3},
            {"name": "whistle", "action": "sleep", "examples": 3},
        ], "revision": 6})
        self.root.update_idletasks()
        self.addCleanup(self.root.destroy)

    def test_record_controls_require_connected_microphone(self) -> None:
        self.assertIn("disabled", self.window.teach_button.state())
        self.window.set_connected(True)
        self.window.name_var.set("tap")
        self.window.action_var.set("Wake")
        self.window.teach_button.invoke()
        self.callbacks["teach"].assert_called_once_with("tap", "Wake")

    def test_correction_uses_selected_true_label(self) -> None:
        self.window.set_recognition({"label": "tap", "action": "wake"})
        self.window.set_can_correct(True)
        self.window.memory_tree.selection_set("whistle")
        self.window._selection_changed(None)
        self.window.correct_button.invoke()
        self.callbacks["correction"].assert_called_once_with("whistle")

    def test_wrong_recognition_is_counted_as_incorrect(self) -> None:
        self.window.set_test_result({"expected": "tap",
                                     "recognized": "whistle",
                                     "matched": True, "correct": False})
        self.assertIn("0 correct", self.window.test_history_var.get())
        self.assertIn("Incorrect", self.window.feedback_var.get())

    def test_unknown_result_does_not_change_sleep_action(self) -> None:
        self.window.perform_action("sleep")
        self.window.set_recognition({"label": None, "action": None})
        self.assertEqual(self.window.creature.pose, "sleep")
        self.assertIn("Unknown", self.window.latest_var.get())

    def test_busy_state_disables_teaching_and_listening(self) -> None:
        self.window.set_connected(True)
        self.window.set_busy(True)
        self.assertIn("disabled", self.window.teach_button.state())
        self.assertIn("disabled", self.window.toggle_button.state())
        self.window.set_busy(False)
        self.assertNotIn("disabled", self.window.teach_button.state())

    def test_timer_is_a_real_controller_value(self) -> None:
        self.window.set_timer(42)
        self.assertIn("42", self.window.timer_var.get())
        self.window.set_timer(0)
        self.assertIn("0", self.window.timer_var.get())


if __name__ == "__main__":
    unittest.main()
