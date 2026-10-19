"""A blocking in-memory stand-in for the async SNN memory backend."""
from __future__ import annotations

from threading import Event, get_ident
from typing import Any


class BlockingSoundMemory:
    """Small async backend fake; it never imports or trains a model."""

    async_training = True

    def __init__(self, blocked_operation: str = "teach") -> None:
        self.blocked_operation = blocked_operation
        self.started = Event()
        self.release = Event()
        self.labels = {
            "tap": {"action": "wake", "examples": 1},
            "click": {"action": "sleep", "examples": 1},
        }
        self.total_examples = 2
        self.revision = 0
        self.teach_calls: list[tuple[str, dict[str, Any], str]] = []
        self.delete_calls: list[str] = []
        self.probe_thread_id: int | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "backend": "trained_snn", "load_error": None,
            "labels": [
                {"name": name, "action": value["action"],
                 "examples": value["examples"]}
                for name, value in self.labels.items()
            ],
            "total_examples": self.total_examples,
            "revision": self.revision,
            "learning": {"trained": True, "model_hash": "fakehash"},
            "comparison": {
                "snn": {"correct": 0, "total": 0},
                "matcher": {"correct": 0, "total": 0},
            },
        }

    def _wait_if_blocked(self, operation: str) -> None:
        if operation == self.blocked_operation:
            self.started.set()
            self.release.wait(5)

    def teach(self, label: str, descriptor: dict[str, Any],
              action: str) -> dict[str, Any]:
        self.teach_calls.append((label, descriptor, action))
        self._wait_if_blocked("teach")
        value = self.labels.setdefault(
            label, {"action": action, "examples": 0})
        value["examples"] += 1
        self.total_examples += 1
        self.revision += 1
        return {"label": label, "examples": value["examples"]}

    def delete_label(self, label: str) -> bool:
        self.delete_calls.append(label)
        self._wait_if_blocked("delete")
        value = self.labels.pop(label, None)
        if value is None:
            return False
        self.total_examples -= value["examples"]
        self.revision += 1
        return True

    def recognize(self, _descriptor: dict[str, Any]) -> dict[str, Any]:
        return {"label": "tap", "action": "wake", "reason": "matched"}

    def record_probe(self, expected: str,
                     _descriptor: dict[str, Any]) -> dict[str, Any]:
        self.probe_thread_id = get_ident()
        return {
            "expected": expected, "recognized": "tap",
            "correct": expected == "tap", "action": None,
            "baseline_recognized": "tap",
            "baseline_correct": expected == "tap",
        }
