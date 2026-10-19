"""SNN sound decisions with a paired example-matcher comparison."""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from rsi_programmer.sound_memory import MIN_GAP, SoundMemory
from rsi_programmer.sound_spiking import SoundSpiking


class SoundComparison:
    """Train an SNN on taught examples; keep probes evaluation-only.

    The SNN chooses the label. A separate feature-distance support guard
    can reject that choice, but never substitutes the matcher's label.
    All taught examples are replayed on explicit teaching to retain older
    classes. Both the examples and trained parameters persist atomically.
    """

    async_training = True

    def __init__(self, path: Path | None = None, seed: int = 0) -> None:
        self.path = (path.with_name(path.stem + "-snn-v1" + path.suffix)
                     if path is not None else None)
        self.seed = seed
        self._matcher = SoundMemory()
        self._model = SoundSpiking(seed)
        self._model_training_fingerprint = self._matcher.fingerprint()
        self._probes: list[dict[str, Any]] = []
        self._teach_events = 0
        self._train_seconds = 0.0
        self._matcher_train_seconds = 0.0
        self._training_updates = 0
        self._load_error: str | None = None
        self._migrated = False
        if self.path is not None and self.path.exists():
            try:
                self._restore(json.loads(self.path.read_text()))
            except (OSError, ValueError, TypeError, KeyError) as error:
                self._load_error = str(error)
        elif path is not None and path.exists():
            legacy = SoundMemory(path)
            error = legacy.summary()["load_error"]
            if error:
                self._load_error = error
            else:
                self._matcher.restore(legacy.snapshot())
                self._fit()
                self._migrated = True
                self._save()

    def teach(self, label: str, descriptor: dict[str, Any],
              action: str) -> dict[str, Any]:
        """Fit both synapse stages from taught examples, never probes."""
        self._check_writable()
        previous = self._snapshot()
        try:
            started = time.perf_counter()
            result = self._matcher.teach(label, descriptor, action)
            self._matcher_train_seconds += time.perf_counter() - started
            self._fit()
            self._teach_events += 1
            self._save()
            return result
        except Exception:
            self._restore(previous)
            raise

    def recognize(self, descriptor: dict[str, Any]) -> dict[str, Any]:
        """Classify through trained spikes, with no matcher fallback."""
        vector = SoundMemory._validate_descriptor(descriptor)
        if self._load_error:
            raise RuntimeError("saved SNN memory needs repair")
        if not descriptor["active"]:
            return SoundMemory._rejection("quiet")
        labels = self._matcher.snapshot()["labels"]
        names = sorted(labels)
        if not names:
            return SoundMemory._rejection("no_examples")
        logits = self._model.scores(vector)
        index = max(range(len(names)), key=lambda item: logits[item])
        label = names[index]
        entry = labels[label]
        distances = {
            name: min(SoundMemory._distance(vector, example)
                      for example in value["examples"])
            for name, value in labels.items()
        }
        distance = distances[label]
        # This guard only rejects; the SNN's selected label is never changed.
        radius = SoundMemory._acceptance_radius(entry["examples"])
        others = [value for name, value in distances.items() if name != label]
        gap = min(others) - distance if others else None
        if distance > radius:
            return {**SoundMemory._rejection("too_different", distance, gap),
                    "neural_label": label}
        if gap is not None and gap < MIN_GAP:
            return {**SoundMemory._rejection("ambiguous", distance, gap),
                    "neural_label": label}
        return {"label": label, "action": entry["action"],
                "distance": distance, "gap": gap, "reason": "matched",
                "backend": "trained_snn", "neural_label": label}

    def recognize_baseline(self, descriptor: dict[str, Any]) -> dict[str, Any]:
        """Evaluate the original matcher without training or firing actions."""
        return self._matcher.recognize(descriptor)

    def record_probe(self, expected: str,
                     descriptor: dict[str, Any]) -> dict[str, Any]:
        """Score the same fresh recording on both backends without fitting."""
        self._check_writable()
        expected = SoundMemory._validate_label(expected)
        before = self._model.weight_hash()
        result = self.recognize(descriptor)
        baseline = self._matcher.recognize(descriptor)
        record = {
            "expected": expected, "recognized": result["label"],
            "snn_recognized": result["label"],
            "predicted": result["label"],
            "matched": result["label"] is not None,
            "correct": result["label"] == expected,
            "action": None, "reason": result["reason"],
            "baseline_recognized": baseline["label"],
            "baseline_correct": baseline["label"] == expected,
            "baseline_reason": baseline["reason"],
            "model_hash_before": before,
            "model_hash_after": self._model.weight_hash(),
        }
        old_probes = self._probes[:]
        self._probes = (self._probes + [record])[-500:]
        try:
            self._save()
        except Exception:
            self._probes = old_probes
            raise
        return dict(record)

    def delete_label(self, label: str) -> bool:
        """Forget one label and retrain only the retained teaching set."""
        self._check_writable()
        previous = self._snapshot()
        try:
            removed = self._matcher.delete_label(label)
            if removed:
                self._fit()
                self._save()
            return removed
        except Exception:
            self._restore(previous)
            raise

    def fingerprint(self) -> str:
        """Fingerprint the remembered teaching examples, without probes."""
        return self._matcher.fingerprint()

    def label_fingerprints(self) -> dict[str, str]:
        """Fingerprint each retained label's examples."""
        return self._matcher.label_fingerprints()

    def summary(self) -> dict[str, Any]:
        """Expose only aggregate teaching, comparison and learning evidence."""
        result = self._matcher.summary()
        metrics = self._model.metrics()
        result.update({
            "backend": "trained_snn", "load_error": self._load_error,
            "probe_count": len(self._probes),
            "comparison": {
                "snn": {"correct": sum(p["correct"] for p in self._probes),
                        "total": len(self._probes)},
                "matcher": {
                    "correct": sum(p["baseline_correct"]
                                   for p in self._probes),
                    "total": len(self._probes)},
            },
            "learning": {
                **metrics, "model_hash": self._model.weight_hash(),
                "updates": self._training_updates,
                "spikes": metrics["actual_spikes"],
                "train_seconds": self._train_seconds,
                "matcher_train_seconds": self._matcher_train_seconds,
                "teach_events": self._teach_events,
                "retention": "replay_all_taught_examples",
            },
            "legacy_examples_migrated": self._migrated,
        })
        return result

    def _fit(self) -> None:
        labels = self._matcher.snapshot()["labels"]
        self._model = SoundSpiking(self.seed)
        if not labels:
            self._model_training_fingerprint = self._matcher.fingerprint()
            return
        vectors: list[list[float]] = []
        targets: list[int] = []
        for index, name in enumerate(sorted(labels)):
            examples = labels[name]["examples"]
            vectors.extend(examples)
            targets.extend([index] * len(examples))
        started = time.perf_counter()
        self._model.fit(vectors, targets, len(labels))
        self._train_seconds += time.perf_counter() - started
        self._training_updates += self._model.metrics()["epochs"]
        self._model_training_fingerprint = self._matcher.fingerprint()

    def _snapshot(self) -> dict[str, Any]:
        return {
            "version": 1, "seed": self.seed,
            "memory": self._matcher.snapshot(),
            "network": self._model.snapshot(),
            "model_hash": self._model.weight_hash(),
            "model_training_fingerprint": self._model_training_fingerprint,
            "probes": copy.deepcopy(self._probes),
            "teach_events": self._teach_events,
            "train_seconds": self._train_seconds,
            "matcher_train_seconds": self._matcher_train_seconds,
            "training_updates": self._training_updates,
            "legacy_examples_migrated": self._migrated,
        }

    def _restore(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("unsupported SNN memory format")
        if payload.get("seed") != self.seed:
            raise ValueError("SNN memory seed mismatch")
        matcher = SoundMemory()
        matcher.restore(payload["memory"])
        model = SoundSpiking(self.seed)
        model.restore(payload["network"])
        if model.seed != self.seed:
            raise ValueError("SNN network seed mismatch")
        if model.weight_hash() != payload.get("model_hash"):
            raise ValueError("SNN memory parameter hash mismatch")
        if matcher.fingerprint() != payload.get("model_training_fingerprint"):
            raise ValueError("SNN training data identity mismatch")
        if model.metrics()["class_count"] != len(matcher.summary()["labels"]):
            raise ValueError("SNN class count does not match sound memory")
        probes = payload.get("probes")
        if not isinstance(probes, list) or len(probes) > 500:
            raise ValueError("invalid paired probe history")
        for probe in probes:
            SoundMemory._validate_probe(probe)
            if type(probe.get("baseline_correct")) is not bool:
                raise ValueError("invalid baseline probe result")
            baseline = probe.get("baseline_recognized")
            if baseline is not None:
                SoundMemory._validate_label(baseline)
            recognized, expected = probe["recognized"], probe["expected"]
            if (probe["correct"] != (recognized == expected)
                    or probe["baseline_correct"] != (baseline == expected)
                    or probe.get("snn_recognized") != recognized
                    or probe.get("predicted") != recognized
                    or probe.get("matched") != (recognized is not None)
                    or probe.get("action") is not None
                    or probe.get("model_hash_before")
                    != probe.get("model_hash_after")):
                raise ValueError("inconsistent paired probe history")
        events = payload.get("teach_events")
        seconds = payload.get("train_seconds")
        matcher_seconds = payload.get("matcher_train_seconds")
        updates = payload.get("training_updates")
        if type(events) is not int or events < 0:
            raise ValueError("invalid teaching count")
        for value in (seconds, matcher_seconds):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError("invalid training duration")
        if type(updates) is not int or updates < 0:
            raise ValueError("invalid update count")
        self._matcher, self._model = matcher, model
        self._model_training_fingerprint = matcher.fingerprint()
        self._probes = copy.deepcopy(probes)
        self._teach_events = events
        self._train_seconds = float(seconds)
        self._matcher_train_seconds = float(matcher_seconds)
        self._training_updates = updates
        self._migrated = bool(payload.get("legacy_examples_migrated"))

    def _check_writable(self) -> None:
        if self._load_error:
            raise OSError("refusing to overwrite invalid SNN sound memory")

    def _save(self) -> None:
        self._check_writable()
        if self.path is None:
            return
        missing = not self.path.parent.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if missing:
            os.chmod(self.path.parent, 0o700)
        data = json.dumps(self._snapshot(), sort_keys=True, allow_nan=False)
        fd, temporary = tempfile.mkstemp(prefix=".sound-snn-",
                                       dir=self.path.parent)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
