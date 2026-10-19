"""Persistent nearest-example memory for teachable short sounds."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from rsi_programmer.sound_features import (
    FEATURE_COUNT,
    FEATURE_VERSION,
    FEATURES_PER_BIN,
    SAMPLE_RATE,
    WINDOW_SECONDS,
)


MEMORY_VERSION = 1
MAX_EXAMPLES_PER_LABEL = 20
MAX_LABELS = 100
MIN_GAP = 0.025
SINGLE_EXAMPLE_RADIUS = 0.16
MIN_MULTI_EXAMPLE_RADIUS = SINGLE_EXAMPLE_RADIUS
MAX_ACCEPTANCE_RADIUS = 0.38
TRAINING_RADIUS_MARGIN = 0.02
TRAINING_RADIUS_SCALE = 1.5
ALLOWED_ACTIONS = frozenset({
    "dance", "sleep", "timer", "wake",
})
LABEL_PATTERN = re.compile(r"^[\w][\w -]{0,31}$", re.UNICODE)


class SoundMemory:
    """Store labeled feature vectors and match with explicit distances."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._labels: dict[str, dict[str, Any]] = {}
        self._probes: list[dict[str, Any]] = []
        self._revision = 0
        self._load_error: str | None = None
        if self.path is not None and self.path.exists():
            self._load()

    def teach(
        self, label: str, descriptor: dict[str, Any], action: str
    ) -> dict[str, Any]:
        """Add one independently retained example for a label."""
        label = self._validate_label(label)
        if action not in ALLOWED_ACTIONS:
            raise ValueError("action is not in the sound action allowlist")
        if descriptor.get("active") is False:
            raise ValueError("quiet audio cannot be taught as a sound")
        vector = self._validate_descriptor(descriptor)
        if label not in self._labels and len(self._labels) >= MAX_LABELS:
            raise ValueError("sound memory has reached its label limit")
        previous_labels = copy.deepcopy(self._labels)
        previous_revision = self._revision
        entry = self._labels.setdefault(
            label, {"action": action, "examples": []}
        )
        if entry["action"] != action:
            raise ValueError(
                "a label cannot be taught with a different action"
            )
        if len(entry["examples"]) >= MAX_EXAMPLES_PER_LABEL:
            raise ValueError("label has reached its example limit")
        entry["examples"].append(vector)
        self._revision += 1
        try:
            self._save()
        except Exception:
            self._labels = previous_labels
            self._revision = previous_revision
            raise
        return {"label": label, "action": action,
                "examples": len(entry["examples"]), "revision": self._revision}

    def recognize(self, descriptor: dict[str, Any]) -> dict[str, Any]:
        """Return a nearest-example label, or reject uncertain inputs."""
        vector = self._validate_descriptor(descriptor)
        if descriptor.get("active") is False:
            return self._rejection("quiet")
        by_label: list[tuple[float, str]] = []
        for label, entry in self._labels.items():
            distance = min(self._distance(vector, example)
                           for example in entry["examples"])
            by_label.append((distance, label))
        if not by_label:
            return self._rejection("no_examples")
        by_label.sort()
        best_distance, best_label = by_label[0]
        best_entry = self._labels[best_label]
        gap = (by_label[1][0] - best_distance
               if len(by_label) > 1 else math.inf)
        if best_distance > self._acceptance_radius(best_entry["examples"]):
            return self._rejection("too_different", best_distance, gap)
        if gap < MIN_GAP:
            return self._rejection("ambiguous", best_distance, gap)
        return {"label": best_label,
                "action": self._labels[best_label]["action"],
                "distance": best_distance, "gap": gap, "reason": "matched"}

    def record_probe(
        self, expected: str, descriptor: dict[str, Any]
    ) -> dict[str, Any]:
        """Record a summary-only query outcome without teaching from it."""
        expected = self._validate_label(expected)
        result = self.recognize(descriptor)
        previous_probes = list(self._probes)
        previous_revision = self._revision
        record = {"expected": expected, "recognized": result["label"],
                  "predicted": result["label"],
                  "matched": result["label"] is not None,
                  "correct": result["label"] == expected,
                  "action": None, "reason": result["reason"]}
        self._probes.append(record)
        if len(self._probes) > 500:
            self._probes = self._probes[-500:]
        self._revision += 1
        try:
            self._save()
        except Exception:
            self._probes = previous_probes
            self._revision = previous_revision
            raise
        return dict(record)

    def remove_last(self, label: str) -> bool:
        """Remove the newest taught example for a label."""
        label = self._validate_label(label)
        entry = self._labels.get(label)
        if entry is None:
            return False
        previous_labels = copy.deepcopy(self._labels)
        previous_revision = self._revision
        entry["examples"].pop()
        if not entry["examples"]:
            del self._labels[label]
        self._revision += 1
        try:
            self._save()
        except Exception:
            self._labels = previous_labels
            self._revision = previous_revision
            raise
        return True

    def delete_label(self, label: str) -> bool:
        """Delete all examples associated with one label."""
        label = self._validate_label(label)
        if label not in self._labels:
            return False
        previous_labels = copy.deepcopy(self._labels)
        previous_revision = self._revision
        del self._labels[label]
        self._revision += 1
        try:
            self._save()
        except Exception:
            self._labels = previous_labels
            self._revision = previous_revision
            raise
        return True

    def leave_one_out_report(self) -> dict[str, Any]:
        """Report training-example leave-one-out results, clearly labeled."""
        rows: list[dict[str, Any]] = []
        for label, entry in self._labels.items():
            for index, vector in enumerate(entry["examples"]):
                candidates: list[tuple[float, str]] = []
                for other_label, other_entry in self._labels.items():
                    examples = list(other_entry["examples"])
                    if other_label == label:
                        examples.pop(index)
                    if examples:
                        candidates.append((min(self._distance(vector, item)
                                               for item in examples),
                                           other_label))
                predicted = min(candidates)[1] if candidates else None
                rows.append({"expected": label, "predicted": predicted,
                             "correct": predicted == label})
        correct = sum(bool(row["correct"]) for row in rows)
        return {"kind": "training_only_leave_one_out",
                "examples": len(rows), "correct": correct,
                "accuracy": correct / len(rows) if rows else None,
                "rows": rows}

    def summary(self) -> dict[str, Any]:
        """Return counts and fingerprints without exposing stored vectors."""
        labels = [{"name": name, "action": value["action"],
                   "examples": len(value["examples"])}
                  for name, value in sorted(self._labels.items())]
        return {"labels": labels,
                "total_examples": sum(item["examples"] for item in labels),
                "probe_count": len(self._probes),
                "revision": self._revision,
                "fingerprint": self.fingerprint(),
                "label_fingerprints": self.label_fingerprints(),
                "load_error": self._load_error}

    def fingerprint(self) -> str:
        """Hash all labels and examples in stable order."""
        payload = self._canonical({"labels": self._labels})
        return hashlib.sha256(payload).hexdigest()

    def label_fingerprints(self) -> dict[str, str]:
        """Hash each label's action and examples for retention checks."""
        return {label: hashlib.sha256(self._canonical(value)).hexdigest()
                for label, value in sorted(self._labels.items())}

    def _load(self) -> None:
        try:
            with self.path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
            if (not isinstance(payload, dict)
                    or payload.get("version") != MEMORY_VERSION):
                raise ValueError("unsupported memory format")
            labels = payload.get("labels")
            probes = payload.get("probes", [])
            revision = payload.get("revision", 0)
            if not isinstance(labels, dict) or len(labels) > MAX_LABELS:
                raise ValueError("invalid labels")
            if not isinstance(probes, list) or len(probes) > 500:
                raise ValueError("invalid probe history")
            if type(revision) is not int or revision < 0:
                raise ValueError("invalid revision")
            loaded: dict[str, dict[str, Any]] = {}
            for label, entry in labels.items():
                label = self._validate_label(label)
                if not isinstance(entry, dict):
                    raise ValueError("invalid label entry")
                action = entry.get("action")
                examples = entry.get("examples")
                if (action not in ALLOWED_ACTIONS
                        or not isinstance(examples, list)):
                    raise ValueError("invalid label action or examples")
                if not examples or len(examples) > MAX_EXAMPLES_PER_LABEL:
                    raise ValueError("invalid example count")
                loaded[label] = {"action": action,
                                 "examples": [self._validate_vector(item)
                                              for item in examples]}
            self._labels = loaded
            self._probes = [self._validate_probe(item) for item in probes]
            self._revision = revision
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            self._labels = {}
            self._probes = []
            self._revision = 0
            self._load_error = str(error)

    def _save(self) -> None:
        if self._load_error is not None:
            raise OSError("refusing to overwrite invalid sound memory")
        if self.path is None:
            return
        parent_was_missing = not self.path.parent.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if parent_was_missing:
            os.chmod(self.path.parent, 0o700)
        payload = {"version": MEMORY_VERSION, "revision": self._revision,
                   "labels": self._labels, "probes": self._probes}
        data = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        fd, temporary = tempfile.mkstemp(prefix=".sound-memory-",
                                         dir=self.path.parent)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _validate_label(label: str) -> str:
        if (not isinstance(label, str)
                or not LABEL_PATTERN.fullmatch(label.strip())):
            raise ValueError(
                "label must be 1 to 32 letters, numbers, spaces or hyphens"
            )
        return label.strip()

    @staticmethod
    def _validate_descriptor(descriptor: dict[str, Any]) -> list[float]:
        if not isinstance(descriptor, dict):
            raise TypeError("descriptor must be a mapping")
        if descriptor.get("version") != FEATURE_VERSION:
            raise ValueError("unsupported sound descriptor version")
        if descriptor.get("sample_rate") != SAMPLE_RATE:
            raise ValueError("descriptor sample rate is incompatible")
        if descriptor.get("duration_seconds") != WINDOW_SECONDS:
            raise ValueError("descriptor duration is incompatible")
        if type(descriptor.get("active")) is not bool:
            raise ValueError("descriptor active flag must be boolean")
        vector = SoundMemory._validate_vector(descriptor.get("vector"))
        if descriptor["active"] and not any(value > 0.0 for value in vector):
            raise ValueError("active descriptor cannot be all zero")
        return vector

    @staticmethod
    def _validate_vector(vector: Any) -> list[float]:
        if (not isinstance(vector, list) or len(vector) != FEATURE_COUNT
                or any(isinstance(value, bool) for value in vector)):
            raise ValueError("descriptor vector has the wrong shape")
        result = [float(value) for value in vector]
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0
               for value in result):
            raise ValueError("descriptor values must be finite and in [0, 1]")
        return result

    @staticmethod
    def _validate_probe(probe: Any) -> dict[str, Any]:
        if not isinstance(probe, dict):
            raise ValueError("invalid probe record")
        expected = SoundMemory._validate_label(probe.get("expected"))
        predicted = probe.get("recognized", probe.get("predicted"))
        if predicted is not None:
            predicted = SoundMemory._validate_label(predicted)
        correct = probe.get("correct")
        if type(correct) is not bool:
            raise ValueError("invalid probe result")
        reason = probe.get("reason")
        if reason not in {
            "matched", "no_examples", "too_different", "ambiguous", "quiet"
        }:
            raise ValueError("invalid probe reason")
        return {"expected": expected, "recognized": predicted,
                "predicted": predicted, "matched": predicted is not None,
                "correct": correct, "action": None, "reason": reason}

    @staticmethod
    def _distance(left: list[float], right: list[float]) -> float:
        """Return a one-bin-constrained DTW distance over active time bins."""
        bin_count = FEATURE_COUNT // FEATURES_PER_BIN
        feature_weights = [
            1.0 if index < 16 else 0.45
            for index in range(FEATURES_PER_BIN)
        ]
        weight_sum = sum(feature_weights)
        left_active = sum(
            left[index * FEATURES_PER_BIN + 16] > 0.05
            for index in range(bin_count)
        )
        right_active = sum(
            right[index * FEATURES_PER_BIN + 16] > 0.05
            for index in range(bin_count)
        )
        active_count = max(left_active, right_active)
        if active_count == 0:
            return 0.0

        infinity = float("inf")
        costs = [[infinity] * bin_count for _ in range(bin_count)]
        for left_index in range(bin_count):
            left_start = left_index * FEATURES_PER_BIN
            for right_index in range(
                max(0, left_index - 1), min(bin_count, left_index + 2)
            ):
                right_start = right_index * FEATURES_PER_BIN
                local_cost = 0.0
                left_level = left[left_start + 16]
                right_level = right[right_start + 16]
                if max(left_level, right_level) > 0.05:
                    for feature_index, weight in enumerate(feature_weights):
                        delta = (
                            left[left_start + feature_index]
                            - right[right_start + feature_index]
                        )
                        local_cost += weight * delta * delta
                if left_index == 0 and right_index == 0:
                    costs[left_index][right_index] = local_cost
                    continue
                predecessors: list[float] = []
                if left_index > 0:
                    predecessors.append(costs[left_index - 1][right_index])
                if right_index > 0:
                    predecessors.append(costs[left_index][right_index - 1])
                if left_index > 0 and right_index > 0:
                    predecessors.append(
                        costs[left_index - 1][right_index - 1]
                    )
                if predecessors:
                    costs[left_index][right_index] = (
                        local_cost + min(predecessors)
                    )
        return math.sqrt(
            costs[-1][-1] / (active_count * weight_sum)
        )

    @classmethod
    def _acceptance_radius(cls, examples: list[list[float]]) -> float:
        """Use positive-neighbor spread, with the one-example floor."""
        if len(examples) < 2:
            return SINGLE_EXAMPLE_RADIUS
        positive_distances = [
            min(cls._distance(example, other)
                for other_index, other in enumerate(examples)
                if other_index != index)
            for index, example in enumerate(examples)
        ]
        positive_distances.sort()
        middle = len(positive_distances) // 2
        if len(positive_distances) % 2:
            typical_distance = positive_distances[middle]
        else:
            typical_distance = (
                positive_distances[middle - 1] + positive_distances[middle]
            ) / 2.0
        radius = max(
            MIN_MULTI_EXAMPLE_RADIUS,
            TRAINING_RADIUS_SCALE * typical_distance
            + TRAINING_RADIUS_MARGIN,
        )
        return min(MAX_ACCEPTANCE_RADIUS, radius)

    @staticmethod
    def _canonical(value: Any) -> bytes:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")

    @staticmethod
    def _rejection(reason: str, distance: float | None = None,
                   gap: float | None = None) -> dict[str, Any]:
        return {"label": None, "action": None, "distance": distance,
                "gap": gap, "reason": reason}
