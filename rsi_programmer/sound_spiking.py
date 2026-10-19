"""Trainable spiking neural classifier for compact sound descriptors."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any

from snntorch import surrogate
import torch
import torch.nn.functional as F

from rsi_programmer.sound_features import (
    FEATURE_COUNT,
    FEATURES_PER_BIN,
    TEMPORAL_BINS,
)


SNAPSHOT_VERSION = 1
HIDDEN_SIZE = 64
MAX_CLASSES = 100
TRAINING_EPOCHS = 100
TRAINING_LEARNING_RATE = 0.025
LIF_BETA = 0.88
LIF_THRESHOLD = 0.8
SURROGATE_SLOPE = 12


class SoundSpiking:
    """Fit a temporal LIF network to explicitly taught sound features."""

    def __init__(self, seed: int = 0) -> None:
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.seed = seed
        self._class_count = 0
        self._input_weight = torch.empty((0, 0), dtype=torch.float32)
        self._input_bias = torch.empty((0,), dtype=torch.float32)
        self._readout_weight = torch.empty((0, 0), dtype=torch.float32)
        self._readout_bias = torch.empty((0,), dtype=torch.float32)
        self._metrics: dict[str, Any] = {
            "trained": False,
            "examples": 0,
            "epochs": 0,
            "loss": None,
            "accuracy": None,
            "actual_spikes": 0,
            "initialized_encoder_hash": None,
            "initialized_readout_hash": None,
            "changed_input_parameters": 0,
            "changed_readout_parameters": 0,
        }

    def fit(
        self,
        vectors: list[list[float]],
        targets: list[int],
        class_count: int,
    ) -> dict[str, Any]:
        """Train from all supplied labeled examples using a fixed recipe."""
        vectors, targets, class_count = self._validate_training_data(
            vectors, targets, class_count
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        input_weight = torch.randn(
            (HIDDEN_SIZE, FEATURES_PER_BIN), generator=generator,
            dtype=torch.float32,
        ) * 0.08
        input_bias = torch.full((HIDDEN_SIZE,), 0.14, dtype=torch.float32)
        output_count = max(2, class_count)
        readout_weight = torch.randn(
            (output_count, HIDDEN_SIZE), generator=generator,
            dtype=torch.float32,
        ) * 0.08
        readout_bias = torch.zeros((output_count,), dtype=torch.float32)
        input_weight.requires_grad_()
        input_bias.requires_grad_()
        readout_weight.requires_grad_()
        readout_bias.requires_grad_()
        initial_input_weight = input_weight.detach().clone()
        initial_input_bias = input_bias.detach().clone()
        initial_readout_weight = readout_weight.detach().clone()
        initial_readout_bias = readout_bias.detach().clone()
        optimizer = torch.optim.Adam(
            [input_weight, input_bias, readout_weight, readout_bias],
            lr=TRAINING_LEARNING_RATE,
        )
        input_tensor = torch.tensor(vectors, dtype=torch.float32).reshape(
            len(vectors), TEMPORAL_BINS, FEATURES_PER_BIN
        )
        target_tensor = torch.tensor(targets, dtype=torch.long)
        if class_count == 1:
            target_tensor = torch.zeros_like(target_tensor)
            class_weights = torch.ones((output_count,), dtype=torch.float32)
        else:
            counts = torch.bincount(target_tensor, minlength=class_count)
            if bool(torch.any(counts == 0)):
                raise ValueError("each class must have a taught example")
            class_weights = len(targets) / (class_count * counts.float())
        spike_fn = surrogate.fast_sigmoid(slope=SURROGATE_SLOPE)
        final_loss = 0.0
        final_spikes = 0
        for _ in range(TRAINING_EPOCHS):
            optimizer.zero_grad(set_to_none=True)
            membrane = torch.zeros(
                (len(vectors), HIDDEN_SIZE), dtype=torch.float32
            )
            spike_sum = torch.zeros_like(membrane)
            spike_count = 0
            for step in range(TEMPORAL_BINS):
                current = F.linear(
                    input_tensor[:, step, :], input_weight, input_bias
                )
                membrane = LIF_BETA * membrane + current
                spikes = spike_fn(membrane - LIF_THRESHOLD)
                membrane = membrane - LIF_THRESHOLD * spikes
                spike_sum = spike_sum + spikes
                spike_count += int(spikes.detach().sum().item())
            logits = F.linear(
                spike_sum / TEMPORAL_BINS, readout_weight, readout_bias
            )
            loss = F.cross_entropy(
                logits, target_tensor, weight=class_weights
            )
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().item())
            final_spikes = spike_count

        with torch.no_grad():
            logits = self._forward_tensor(
                input_tensor, input_weight, input_bias,
                readout_weight, readout_bias,
            )
            predicted = logits.argmax(dim=1)
            accuracy = float((predicted == target_tensor).float().mean().item())
        metrics = {
            "trained": True,
            "examples": len(vectors),
            "epochs": TRAINING_EPOCHS,
            "loss": final_loss,
            "accuracy": accuracy,
            "actual_spikes": final_spikes,
            "initialized_encoder_hash": self._hash_tensors(
                initial_input_weight, initial_input_bias
            ),
            "initialized_readout_hash": self._hash_tensors(
                initial_readout_weight, initial_readout_bias
            ),
            "changed_input_parameters": int(
                torch.count_nonzero(
                    input_weight.detach() != initial_input_weight
                ).item()
                + torch.count_nonzero(
                    input_bias.detach() != initial_input_bias
                ).item()
            ),
            "changed_readout_parameters": int(
                torch.count_nonzero(
                    readout_weight.detach() != initial_readout_weight
                ).item()
                + torch.count_nonzero(
                    readout_bias.detach() != initial_readout_bias
                ).item()
            ),
        }
        self._class_count = class_count
        self._input_weight = input_weight.detach().clone()
        self._input_bias = input_bias.detach().clone()
        self._readout_weight = readout_weight.detach().clone()
        self._readout_bias = readout_bias.detach().clone()
        self._metrics = metrics
        return self.metrics()

    def scores(self, vector: list[float]) -> list[float]:
        """Return trained readout scores, resetting LIF state for this clip."""
        vector = self._validate_vector(vector)
        if not self._class_count:
            raise ValueError("sound classifier has not been trained")
        tensor = torch.tensor(vector, dtype=torch.float32).reshape(
            1, TEMPORAL_BINS, FEATURES_PER_BIN
        )
        with torch.no_grad():
            logits = self._forward_tensor(
                tensor, self._input_weight, self._input_bias,
                self._readout_weight, self._readout_bias,
            )[0]
        if self._class_count == 1:
            return [float((logits[0] - logits[1]).item())]
        return [float(value) for value in logits[:self._class_count].tolist()]

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe representation of the trained network."""
        return {
            "version": SNAPSHOT_VERSION,
            "seed": self.seed,
            "feature_count": FEATURE_COUNT,
            "features_per_bin": FEATURES_PER_BIN,
            "temporal_bins": TEMPORAL_BINS,
            "hidden_size": HIDDEN_SIZE,
            "training_recipe": self._training_recipe(),
            "class_count": self._class_count,
            "input_weight": self._input_weight.tolist(),
            "input_bias": self._input_bias.tolist(),
            "readout_weight": self._readout_weight.tolist(),
            "readout_bias": self._readout_bias.tolist(),
            "metrics": self.metrics(),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Validate a snapshot completely before replacing current state."""
        if not isinstance(snapshot, dict):
            raise ValueError("invalid sound classifier snapshot")
        try:
            if (snapshot.get("version") != SNAPSHOT_VERSION
                    or snapshot.get("feature_count") != FEATURE_COUNT
                    or snapshot.get("features_per_bin") != FEATURES_PER_BIN
                    or snapshot.get("temporal_bins") != TEMPORAL_BINS
                    or snapshot.get("hidden_size") != HIDDEN_SIZE
                    or snapshot.get("training_recipe")
                    != self._training_recipe()):
                raise ValueError("unsupported sound classifier snapshot")
            seed = snapshot.get("seed")
            class_count = snapshot.get("class_count")
            if type(seed) is not int or seed < 0:
                raise ValueError("invalid snapshot seed")
            if (type(class_count) is not int
                    or class_count < 0 or class_count > MAX_CLASSES):
                raise ValueError("invalid snapshot class count")
            if class_count:
                output_count = max(2, class_count)
                input_weight = self._snapshot_tensor(
                    snapshot.get("input_weight"),
                    (HIDDEN_SIZE, FEATURES_PER_BIN),
                )
                input_bias = self._snapshot_tensor(
                    snapshot.get("input_bias"), (HIDDEN_SIZE,)
                )
                readout_weight = self._snapshot_tensor(
                    snapshot.get("readout_weight"),
                    (output_count, HIDDEN_SIZE),
                )
                readout_bias = self._snapshot_tensor(
                    snapshot.get("readout_bias"), (output_count,)
                )
            else:
                tensor_names = (
                    "input_weight", "input_bias", "readout_weight",
                    "readout_bias",
                )
                if any(snapshot.get(name) != [] for name in tensor_names):
                    raise ValueError("untrained snapshot has parameters")
                input_weight = torch.empty((0, 0), dtype=torch.float32)
                input_bias = torch.empty((0,), dtype=torch.float32)
                readout_weight = torch.empty((0, 0), dtype=torch.float32)
                readout_bias = torch.empty((0,), dtype=torch.float32)
            metrics = self._validate_metrics(snapshot.get("metrics"))
            if bool(metrics["trained"]) != bool(class_count):
                raise ValueError("snapshot trained state does not match")
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError("invalid sound classifier snapshot") from error
        self.seed = seed
        self._class_count = class_count
        self._input_weight = input_weight
        self._input_bias = input_bias
        self._readout_weight = readout_weight
        self._readout_bias = readout_bias
        self._metrics = metrics

    def weight_hash(self) -> str:
        """Hash all trained parameters and their network dimensions."""
        payload = {
            "class_count": self._class_count,
            "input_weight": self._input_weight.tolist(),
            "input_bias": self._input_bias.tolist(),
            "readout_weight": self._readout_weight.tolist(),
            "readout_bias": self._readout_bias.tolist(),
        }
        return hashlib.sha256(self._canonical(payload)).hexdigest()

    def readout_hash(self) -> str:
        """Hash the trained hidden-to-class synapses."""
        return self._hash_tensors(self._readout_weight, self._readout_bias)

    def encoder_hash(self) -> str:
        """Hash the feature-to-hidden synapses used as the encoder."""
        return self._hash_tensors(self._input_weight, self._input_bias)

    def metrics(self) -> dict[str, Any]:
        """Return counters, training results, and parameter fingerprints."""
        result = copy.deepcopy(self._metrics)
        result["class_count"] = self._class_count
        result["weight_hash"] = self.weight_hash()
        result["encoder_hash"] = self.encoder_hash()
        result["readout_hash"] = self.readout_hash()
        return result

    @staticmethod
    def _forward_tensor(
        inputs: torch.Tensor,
        input_weight: torch.Tensor,
        input_bias: torch.Tensor,
        readout_weight: torch.Tensor,
        readout_bias: torch.Tensor,
    ) -> torch.Tensor:
        """Run a fresh temporal LIF pass and return readout logits."""
        spike_fn = surrogate.fast_sigmoid(slope=SURROGATE_SLOPE)
        membrane = torch.zeros(
            (inputs.shape[0], HIDDEN_SIZE), dtype=torch.float32
        )
        spike_sum = torch.zeros_like(membrane)
        for step in range(TEMPORAL_BINS):
            current = F.linear(inputs[:, step, :], input_weight, input_bias)
            membrane = LIF_BETA * membrane + current
            spikes = spike_fn(membrane - LIF_THRESHOLD)
            membrane = membrane - LIF_THRESHOLD * spikes
            spike_sum = spike_sum + spikes
        return F.linear(
            spike_sum / TEMPORAL_BINS, readout_weight, readout_bias
        )

    @staticmethod
    def _validate_training_data(
        vectors: list[list[float]], targets: list[int], class_count: int
    ) -> tuple[list[list[float]], list[int], int]:
        if not isinstance(vectors, list) or not vectors:
            raise ValueError("at least one sound example is required")
        if not isinstance(targets, list) or len(targets) != len(vectors):
            raise ValueError("targets must match the sound examples")
        if (type(class_count) is not int
                or class_count < 1 or class_count > MAX_CLASSES):
            raise ValueError("class_count must be between 1 and 100")
        normalized = [SoundSpiking._validate_vector(vector)
                      for vector in vectors]
        target_values: list[int] = []
        for target in targets:
            if type(target) is not int or target < 0 or target >= class_count:
                raise ValueError("target is outside the class range")
            target_values.append(target)
        if class_count > 1 and len(set(target_values)) < 2:
            raise ValueError("multiple classes require examples per class")
        if class_count > 1 and set(target_values) != set(range(class_count)):
            raise ValueError("each class must have a taught example")
        return normalized, target_values, class_count

    @staticmethod
    def _validate_vector(vector: list[float]) -> list[float]:
        if not isinstance(vector, list) or len(vector) != FEATURE_COUNT:
            raise ValueError(f"sound vectors must contain {FEATURE_COUNT} values")
        result: list[float] = []
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("sound vector values must be numbers")
            numeric = float(value)
            if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                raise ValueError("sound vector values must be finite in [0, 1]")
            result.append(numeric)
        return result

    @staticmethod
    def _snapshot_tensor(
        value: Any, shape: tuple[int, ...]
    ) -> torch.Tensor:
        tensor = torch.tensor(value, dtype=torch.float32)
        if tuple(tensor.shape) != shape or not bool(torch.isfinite(tensor).all()):
            raise ValueError("invalid snapshot tensor")
        return tensor

    @staticmethod
    def _validate_metrics(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("invalid snapshot metrics")
        trained = value.get("trained")
        examples = value.get("examples")
        epochs = value.get("epochs")
        loss = value.get("loss")
        accuracy = value.get("accuracy")
        actual_spikes = value.get("actual_spikes")
        initialized_encoder_hash = value.get("initialized_encoder_hash")
        initialized_readout_hash = value.get("initialized_readout_hash")
        changed_input = value.get("changed_input_parameters")
        changed_readout = value.get("changed_readout_parameters")
        if type(trained) is not bool:
            raise ValueError("invalid trained metric")
        for name, number in (("examples", examples), ("epochs", epochs),
                             ("actual_spikes", actual_spikes),
                             ("changed_input_parameters", changed_input),
                             ("changed_readout_parameters", changed_readout)):
            if type(number) is not int or number < 0:
                raise ValueError(f"invalid {name} metric")
        for name, number in (("loss", loss), ("accuracy", accuracy)):
            if number is not None and (
                    isinstance(number, bool)
                    or not isinstance(number, (int, float))
                    or not math.isfinite(float(number))):
                raise ValueError(f"invalid {name} metric")
        if accuracy is not None and not 0.0 <= float(accuracy) <= 1.0:
            raise ValueError("invalid accuracy metric")
        for name, fingerprint in (
                ("initialized_encoder_hash", initialized_encoder_hash),
                ("initialized_readout_hash", initialized_readout_hash)):
            if fingerprint is not None and (
                    not isinstance(fingerprint, str)
                    or len(fingerprint) != 64):
                raise ValueError(f"invalid {name} metric")
        return {
            "trained": trained,
            "examples": examples,
            "epochs": epochs,
            "loss": None if loss is None else float(loss),
            "accuracy": None if accuracy is None else float(accuracy),
            "actual_spikes": actual_spikes,
            "initialized_encoder_hash": initialized_encoder_hash,
            "initialized_readout_hash": initialized_readout_hash,
            "changed_input_parameters": changed_input,
            "changed_readout_parameters": changed_readout,
        }

    @staticmethod
    def _canonical(payload: dict[str, Any]) -> bytes:
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

    @staticmethod
    def _training_recipe() -> dict[str, Any]:
        return {
            "optimizer": "Adam",
            "learning_rate": TRAINING_LEARNING_RATE,
            "epochs": TRAINING_EPOCHS,
            "class_balanced": True,
            "lif_beta": LIF_BETA,
            "lif_threshold": LIF_THRESHOLD,
            "surrogate": "fast_sigmoid",
            "surrogate_slope": SURROGATE_SLOPE,
        }

    @staticmethod
    def _hash_tensors(*tensors: torch.Tensor) -> str:
        payload = [tensor.detach().tolist() for tensor in tensors]
        return hashlib.sha256(
            SoundSpiking._canonical({"tensors": payload})
        ).hexdigest()
