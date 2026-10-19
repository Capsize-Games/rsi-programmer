"""Check that SNN fitting stays outside Tk and keeps sound actions gated."""
from __future__ import annotations

from threading import Event, Thread, get_ident
import unittest
from unittest.mock import Mock, call, patch

from rsi_programmer.sound_controller import SoundController
from tests.sound_memory_fake import BlockingSoundMemory


class SoundSnnControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Mock()
        self.view = Mock()
        self.memory = BlockingSoundMemory()
        self.controller = SoundController(
            self.root, self.memory,
            window_factory=lambda *_args: self.view)
        self.controller.microphone = Mock()
        self.controller.connected = True
        self.descriptor = {"active": True, "features": [0.25, 0.75]}
        self.addCleanup(self._release_worker)

    def _release_worker(self) -> None:
        self.memory.release.set()
        thread = self.controller._learning_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2)

    def _start_teaching(self) -> None:
        self.controller.teach("tap", "Wake")
        with patch("rsi_programmer.sound_controller.extract_sound",
                   return_value=self.descriptor):
            self.controller.handle_event({
                "kind": "clip", "request_id": self.controller._next_request,
                "samples": [0.1],
            })
        self.assertTrue(self.memory.started.wait(1))

    def _finish_learning(self) -> None:
        thread = self.controller._learning_thread
        self.memory.release.set()
        self.assertIsNotNone(thread)
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.controller.poll()

    def test_teaching_worker_keeps_timer_live_and_blocks_new_operations(
        self,
    ) -> None:
        self._start_teaching()
        operation_id = self.controller._learning_operation_id

        self.controller.teach("new", "Dance")
        self.controller.test("tap")
        self.controller.correct("tap")
        self.controller.delete("tap")
        self.controller.toggle_listening()
        self.assertEqual(self.controller._learning_operation_id,
                         operation_id)
        self.assertEqual(len(self.memory.teach_calls), 1)
        self.assertEqual(self.memory.delete_calls, [])

        self.controller._timer_deadline = 100.5
        with patch("rsi_programmer.sound_controller.time.monotonic",
                   return_value=100.0):
            self.controller.poll()
        self.view.set_timer.assert_called_with(1)
        self.root.after.assert_called()
        self._finish_learning()
        self.assertIsNone(self.controller._learning_operation_id)

    def test_deletion_retrains_in_worker_and_gates_other_mutations(
        self,
    ) -> None:
        self.memory.blocked_operation = "delete"
        self.controller.listening = True
        self.controller.delete("tap")
        self.assertTrue(self.memory.started.wait(1))
        self.assertFalse(self.controller.listening)
        self.controller.delete("click")
        self.controller.test("tap")
        self.assertEqual(self.memory.delete_calls, ["tap"])
        self._finish_learning()
        self.assertNotIn("tap", self.memory.labels)
        self.assertTrue(self.controller.listening)

    def test_probe_evaluation_stays_on_calling_thread(self) -> None:
        self.controller.test("tap")
        main_thread = get_ident()
        with patch("rsi_programmer.sound_controller.extract_sound",
                   return_value=self.descriptor):
            self.controller.handle_event({
                "kind": "clip", "request_id": self.controller._next_request,
                "samples": [0.1],
            })
        self.assertEqual(self.memory.probe_thread_id, main_thread)
        self.view.perform_action.assert_not_called()

    def test_correction_trains_only_the_selected_label_without_action(
        self,
    ) -> None:
        self.controller.listening = True
        self.controller._last_descriptor = self.descriptor
        self.controller.correct("click")
        self.assertTrue(self.memory.started.wait(1))
        label, descriptor, action = self.memory.teach_calls[0]
        self.assertEqual((label, action), ("click", "sleep"))
        self.assertEqual(descriptor, self.descriptor)
        self.assertIsNot(descriptor, self.descriptor)
        self.view.perform_action.assert_not_called()
        self._finish_learning()

    def test_paused_correction_does_not_resume_stale_listening_state(
        self,
    ) -> None:
        self.controller.listening = False
        self.controller._resume_listening = True
        self.controller._last_descriptor = self.descriptor
        self.controller.correct("click")
        self.assertTrue(self.memory.started.wait(1))
        self._finish_learning()
        self.assertFalse(self.controller.listening)
        calls = self.controller.microphone.set_listening.call_args_list
        self.assertNotIn(call(True), calls)

    def test_microphone_failure_invalidates_resume_but_refreshes_saved_memory(
        self,
    ) -> None:
        self.controller.listening = True
        self._start_teaching()
        self.controller.handle_event({"kind": "error", "message": "Lost mic"})
        self.assertFalse(self.controller.connected)
        self.assertIsNotNone(self.controller._learning_operation_id)
        self.assertTrue(self.controller._pending)
        self._finish_learning()
        refreshed = self.view.set_memory.call_args.args[0]
        self.assertEqual(refreshed["total_examples"], 3)
        self.view.set_listening.assert_called_with(False)
        self.view.perform_action.assert_not_called()
        self.assertFalse(self.controller.listening)

    def test_close_joins_learning_before_destroy_and_keeps_saved_example(
        self,
    ) -> None:
        self._start_teaching()
        stopped = Event()
        self.controller.microphone.stop.side_effect = stopped.set
        closer = Thread(target=self.controller.close)
        closer.start()
        self.assertTrue(stopped.wait(1))
        self.assertTrue(closer.is_alive())
        self.root.destroy.assert_not_called()
        self.memory.release.set()
        closer.join(timeout=2)
        self.assertFalse(closer.is_alive())
        self.root.destroy.assert_called_once()
        self.assertEqual(self.memory.total_examples, 3)


if __name__ == "__main__":
    unittest.main()
