"""Exercise real Tk controls on a private test display."""
from __future__ import annotations

import tkinter as tk
import unittest
from unittest.mock import Mock

from rsi_programmer.sound_window import SoundWindow


class SoundWindowTests(unittest.TestCase):
    def test_minimum_window_shows_comparison_and_model_without_errors(
        self,
    ) -> None:
        self.root.report_callback_exception = Mock()
        self.window.set_memory({
            "backend": "trained_snn", "labels": [], "revision": 12,
            "learning": {"trained": True, "model_hash": "abcdef012345",
                         "updates": 1200},
            "comparison": {"snn": {"correct": 48, "total": 48},
                           "matcher": {"correct": 48, "total": 48}},
        })
        self.root.geometry("760x800")
        self.root.deiconify()
        self.root.update()
        self.root.report_callback_exception.assert_not_called()
        variables = {str(self.window.comparison_var),
                     str(self.window.learning_var)}

        def descendants(widget: tk.Misc) -> list[tk.Misc]:
            children = widget.winfo_children()
            return children + [child for item in children
                               for child in descendants(item)]

        found = 0
        for widget in descendants(self.root):
            if ("textvariable" in widget.keys()
                    and str(widget.cget("textvariable")) in variables):
                found += 1
                self.assertGreater(widget.winfo_height(), 0)
                self.assertLessEqual(
                    widget.winfo_y() + widget.winfo_height(),
                    widget.master.winfo_height())
                self.assertLessEqual(
                    widget.winfo_x() + widget.winfo_width(),
                    widget.master.winfo_width())
        self.assertEqual(found, 2)

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

    def test_memory_shows_snn_learning_and_probe_comparison(self) -> None:
        self.window.set_memory({
            "labels": [], "revision": 7, "backend": "trained_snn",
            "comparison": {
                "snn": {"correct": 4, "total": 5},
                "matcher": {"correct": 3, "total": 5},
            },
            "learning": {
                "trained": True,
                "model_hash": "0123456789abcdef",
                "updates": 12, "spikes": 480,
            },
        })
        self.assertIn("SNN controls the creature",
                      self.window.backend_var.get())
        self.assertIn("SNN 4/5", self.window.comparison_var.get())
        self.assertIn("Example matcher 3/5",
                      self.window.comparison_var.get())
        self.assertIn("0123456789", self.window.learning_var.get())
        self.assertIn("12 updates total", self.window.learning_var.get())

    def test_probe_result_can_show_baseline_comparison(self) -> None:
        self.window.set_test_result({
            "expected": "tap", "recognized": "tap", "correct": True,
            "baseline_recognized": "whistle",
            "baseline_correct": False,
        })
        self.assertIn("SNN 1/1", self.window.comparison_var.get())
        self.assertIn("Example matcher 0/1",
                      self.window.comparison_var.get())

    def test_backend_label_reflects_empty_and_unavailable_snn(self) -> None:
        self.window.set_memory({
            "backend": "trained_snn", "learning": {"trained": False},
            "labels": [],
        })
        self.assertEqual(self.window.backend_var.get(),
                         "SNN ready; teach a sound")
        self.window.set_memory({
            "backend": "trained_snn", "load_error": "corrupt",
            "learning": {"trained": False}, "labels": [],
        })
        self.assertEqual(self.window.backend_var.get(),
                         "SNN unavailable; saved memory needs repair")


if __name__ == "__main__":
    unittest.main()
