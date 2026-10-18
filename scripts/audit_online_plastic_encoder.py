"""Independently replay and audit the frozen online plastic-encoder pilot."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
import copy
from collections import defaultdict
from pathlib import Path
from typing import Any

from rsi_programmer.online_graph import OPERATORS
from rsi_programmer.online_spiking import _nested_lists
from scripts.audit_online_graph import audit_episode_row, verify_source_hashes

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "fixtures" / "online_plastic_encoder_v1.json"
PROTOCOL_SHA256 = (
    "8fd37e1cd6904ab28dae47d48baf95beab68e0819b5a554d920c3d63aa1ebbf9")
CORPUS_SHA256 = (
    "a4f9af2dfc0d8c4539d700b032dcc223c82aa272f02497902024d9f371096140")
SCHEDULE_SHA256 = (
    "0ffbd7eadac93aa63b8df68d9ba250cd4da5a55a178e8339f9505d81638af3f1")
ARMS = (
    "persistent_snn", "frozen_snn", "episode_reset_snn",
    "online_nonspiking", "grammar_streaming", "plastic_snn",
    "reset_plastic_snn", "plastic_nonspiking",
)
SEEDS = (0, 1, 2)
BUDGETS = (144, 432)
NEURAL_ARMS = tuple(arm for arm in ARMS if arm != "grammar_streaming")
RESET_ARMS = ("episode_reset_snn", "reset_plastic_snn")
PLASTIC_ARMS = ("plastic_snn", "reset_plastic_snn",
                "plastic_nonspiking")
BOUNDARIES = (0, 16, 32, 48)
ACTION_ORDER = ("add", "sub", "mul")
FEATURE_COUNT, HIDDEN_COUNT, STEPS = 6, 16, 6
CONTEXT_COUNT, CONTEXT_SCALE = 18, 18.0
BETA, THRESHOLD = 0.8, 1.0
BASELINE_RATE, READOUT_LR = 0.05, 0.05
READOUT_MAX_DELTA, WEIGHT_LIMIT = 0.05, 5.0
ENCODER_LR, ENCODER_MAX_DELTA, SURROGATE = 0.01, 0.05, 0.3
PROJECTION_SEED_PREFIX = "online-observation-context-v1:"
TOLERANCE = 1e-12


def _json_bytes(value: Any) -> bytes:
    """Canonical compact JSON bytes used by the published state hashes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _sha256(payload: bytes) -> str:
    """Return the lowercase SHA-256 digest of bytes."""
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: Any) -> Any:
    """Normalize integral floats as the plastic snapshot format does."""
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _model_hash(value: Any) -> str:
    """Hash a projection/readout payload with integral-float normalization."""
    return _sha256(_json_bytes(_canonical(value)))


def _readout_hash(weights: list[list[float]], bias: list[float]) -> str:
    """Hash the inherited readout in its original float representation."""
    return _sha256(_json_bytes({"weights": weights, "bias": bias}))


def _close(actual: Any, expected: float, label: str) -> None:
    """Reject nonnumeric, nonfinite, or materially different values."""
    if (isinstance(actual, bool) or not isinstance(actual, (int, float))
            or not math.isfinite(float(actual))
            or not math.isclose(float(actual), expected,
                                rel_tol=TOLERANCE,
                                abs_tol=TOLERANCE)):
        raise ValueError(f"{label} mismatch")


def _equal(actual: Any, expected: Any, label: str) -> None:
    """Compare nested exact fields and floating values with tight tolerance."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{label} keys mismatch")
        for key, value in expected.items():
            _equal(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, (list, tuple)):
        if (not isinstance(actual, (list, tuple))
                or len(actual) != len(expected)):
            raise ValueError(f"{label} length mismatch")
        for index, value in enumerate(expected):
            _equal(actual[index], value, f"{label}[{index}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        _close(actual, float(expected), label)
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{label} mismatch")


def _matrix_hash(matrix: list[list[float]]) -> str:
    """Hash one projection matrix using the model's canonical float rules."""
    return _model_hash(matrix)


def _make_initial_state(seed: int) -> tuple[
    random.Random, list[list[float]], list[list[int]]
]:
    """Rebuild both seeded projections and the independent action stream."""
    action_rng = random.Random(seed)
    encoder = [[float(action_rng.choice((-1, 0, 1)))
                for _ in range(FEATURE_COUNT)]
               for _ in range(HIDDEN_COUNT)]
    for row in encoder:
        if not any(row):
            row[action_rng.randrange(FEATURE_COUNT)] = 1.0
    context_rng = random.Random(f"{PROJECTION_SEED_PREFIX}{seed}")
    projection = [[context_rng.choice((-1, 0, 1))
                   for _ in range(HIDDEN_COUNT)]
                  for _ in range(CONTEXT_COUNT)]
    return action_rng, encoder, projection


class ReferencePlasticPolicy:
    """Independent policy emulator used only by the offline auditor.

    This class intentionally does not inherit from or call the runtime policy.
    It implements action sampling, the fixed/context encoders, truncated
    six-tick eligibility, reward-modulated updates, snapshots, and metrics so
    the raw action sequence and learned state can be replayed independently.
    """

    def __init__(self, arm: str, seed: int) -> None:
        """Initialize the exact frozen mode and seed for one matrix cell."""
        if arm not in NEURAL_ARMS:
            raise ValueError("reference policy requires a neural arm")
        self.arm, self.seed = arm, seed
        self.learning = arm != "frozen_snn"
        self.spiking = arm not in ("online_nonspiking", "plastic_nonspiking")
        self.encoder_learning = arm in PLASTIC_ARMS
        self.lr, self.encoder_lr = READOUT_LR, ENCODER_LR
        self.rng, self.initial_encoder, self.initial_projection = (
            _make_initial_state(seed))
        self.encoder = [row[:] for row in self.initial_encoder]
        self.projection = [row[:] for row in self.initial_projection]
        self.weights = [[0.0] * HIDDEN_COUNT for _ in ACTION_ORDER]
        self.bias = [0.0] * len(ACTION_ORDER)
        self.baseline = 0.0
        self.membrane = [0.0] * HIDDEN_COUNT
        self.pending_context: list[float] | None = None
        self.active_context: list[float] | None = None
        self.context_current: list[float] | None = None
        self.episode_id: str | None = None
        self.pending_action: int | None = None
        self.pending_readout_eligibility: list[list[float]] | None = None
        self.pending_bias_eligibility: list[float] | None = None
        self.pending_hidden_signal: list[float] | None = None
        self.pending_rate_derivative = [0.0] * HIDDEN_COUNT
        self.pending_features = [0.0] * FEATURE_COUNT
        self.metrics_state = {
            "decisions": 0, "neuron_ticks": 0, "encoder_mac": 0,
            "readout_mac": 0, "updates": 0, "update_synapses": 0,
            "update_biases": 0, "update_synapse_visits": 0, "spikes": 0,
            "reward_observations": 0, "episodes": 0,
            "observation_scalars": 0, "projection_mac": 0,
            "context_adds": 0,
            "plastic_context_projection_macs": 0,
            "plastic_context_projection_recomputes": 0,
            "plastic_feedback_readout_macs": 0,
            "plastic_feedback_signal_terms": 0,
            "plastic_encoder_updates": 0,
            "plastic_eligibility_neuron_ticks": 0,
            "plastic_history_update_synapse_visits": 0,
            "plastic_context_update_synapse_visits": 0,
            "plastic_encoder_changed_synapses": 0,
        }

    def reset_learning(self) -> None:
        """Restore initial projections/readout while preserving action RNG."""
        if self.episode_id is not None or self.pending_action is not None:
            raise RuntimeError("cannot reset an active reference policy")
        self.encoder = [row[:] for row in self.initial_encoder]
        self.projection = [row[:] for row in self.initial_projection]
        self.weights = [[0.0] * HIDDEN_COUNT for _ in ACTION_ORDER]
        self.bias = [0.0] * len(ACTION_ORDER)
        self.baseline = 0.0
        self.membrane = [0.0] * HIDDEN_COUNT
        self.pending_context = self.active_context = None
        self.context_current = None
        self._clear_pending()
        for key in self.metrics_state:
            self.metrics_state[key] = 0

    def set_observed_rows(
        self, rows: tuple[tuple[int, int, int], ...]
    ) -> None:
        """Install current observed examples, never held labels or metadata."""
        if self.episode_id is not None or self.pending_context is not None:
            raise RuntimeError("reference context boundary violation")
        if len(rows) != 6:
            raise ValueError("reference policy requires six observed rows")
        if any(any(type(value) is not int for value in row) for row in rows):
            raise ValueError("reference policy observed rows must be integers")
        context = [value / (1 + abs(value))
                   for row in rows for value in row]
        self.pending_context = context
        self.context_current = self._project_context(context)
        self.metrics_state["observation_scalars"] += CONTEXT_COUNT
        self.metrics_state["projection_mac"] += CONTEXT_COUNT * HIDDEN_COUNT

    def start_episode(self, episode_id: str) -> None:
        """Start one task with zero membrane and no cross-task eligibility."""
        if self.episode_id is not None:
            raise RuntimeError("reference episode already active")
        if self.pending_context is None or not episode_id.strip():
            raise ValueError("reference task needs current observed context")
        self.active_context = self.pending_context
        self.pending_context = None
        self.episode_id = episode_id
        self.membrane = [0.0] * HIDDEN_COUNT
        self._clear_pending()
        self.metrics_state["episodes"] += 1

    def choose(
        self, features: tuple[float, ...], available: tuple[str, ...]
    ) -> str:
        """Run independent forward/sample and retain local credit terms."""
        if self.episode_id is None or self.pending_action is not None:
            raise RuntimeError("reference action lifecycle violation")
        if self.active_context is None or len(features) != FEATURE_COUNT:
            raise ValueError("reference feature/context shape invalid")
        legal = tuple(index for index, action in enumerate(ACTION_ORDER)
                      if action in available)
        if (not legal or any(action not in ACTION_ORDER
                             for action in available)):
            raise ValueError("reference legal mask invalid")
        rates, spikes, derivative = self._encode(list(features))
        logits = [sum(weight * rate for weight, rate
                      in zip(self.weights[index], rates)) + self.bias[index]
                  for index in range(len(ACTION_ORDER))]
        probabilities = self._softmax(logits, legal)
        draw = self.rng.random()
        cumulative = 0.0
        picked = legal[-1]
        for index in legal:
            cumulative += probabilities[index]
            if draw < cumulative:
                picked = index
                break
        readout_eligibility = [[0.0] * HIDDEN_COUNT
                               for _ in ACTION_ORDER]
        bias_eligibility = [0.0] * len(ACTION_ORDER)
        if self.learning:
            for index in legal:
                coefficient = (1.0 if index == picked else 0.0) \
                    - probabilities[index]
                bias_eligibility[index] = coefficient
                for hidden, rate in enumerate(rates):
                    readout_eligibility[index][hidden] = coefficient * rate
        signal = [0.0] * HIDDEN_COUNT
        if self.learning and self.encoder_learning:
            for hidden in range(HIDDEN_COUNT):
                signal[hidden] = sum(
                    self.weights[index][hidden]
                    * ((1.0 if index == picked else 0.0)
                       - probabilities.get(index, 0.0))
                    for index in legal)
            self.metrics_state["plastic_feedback_readout_macs"] += (
                len(ACTION_ORDER) * HIDDEN_COUNT)
            self.metrics_state["plastic_feedback_signal_terms"] += (
                len(legal) * HIDDEN_COUNT)
        self.pending_action = picked
        self.pending_readout_eligibility = readout_eligibility
        self.pending_bias_eligibility = bias_eligibility
        self.pending_hidden_signal = signal
        self.pending_rate_derivative = derivative
        self.pending_features = list(features)
        self.metrics_state["decisions"] += 1
        self.metrics_state["spikes"] += spikes
        self.metrics_state["neuron_ticks"] += STEPS * HIDDEN_COUNT
        self.metrics_state["encoder_mac"] += (
            STEPS * FEATURE_COUNT * HIDDEN_COUNT)
        self.metrics_state["readout_mac"] += (
            len(ACTION_ORDER) * HIDDEN_COUNT)
        self.metrics_state["context_adds"] += STEPS * HIDDEN_COUNT
        return ACTION_ORDER[picked]

    def observe(self, reward: float) -> float:
        """Update readout and input projections from this action's outcome."""
        if (self.episode_id is None or self.pending_action is None
                or self.pending_readout_eligibility is None
                or self.pending_bias_eligibility is None):
            raise RuntimeError("reference reward without pending action")
        clipped = max(-1.0, min(1.0, float(reward)))
        advantage = clipped - self.baseline
        if self.learning:
            changed, changed_biases = 0, 0
            for action in range(len(ACTION_ORDER)):
                for hidden in range(HIDDEN_COUNT):
                    delta = self._clip_update(
                        self.lr * advantage
                        * self.pending_readout_eligibility[action][hidden])
                    if delta:
                        self.weights[action][hidden] = max(
                            -WEIGHT_LIMIT, min(
                                WEIGHT_LIMIT,
                                self.weights[action][hidden] + delta))
                        changed += 1
                delta = self._clip_update(
                    self.lr * advantage
                    * self.pending_bias_eligibility[action])
                if delta:
                    self.bias[action] = max(
                        -WEIGHT_LIMIT,
                        min(WEIGHT_LIMIT, self.bias[action] + delta))
                    changed_biases += 1
            self.baseline += BASELINE_RATE * advantage
            self.metrics_state["updates"] += 1
            self.metrics_state["update_synapses"] += changed
            self.metrics_state["update_biases"] += changed_biases
            self.metrics_state["update_synapse_visits"] += (
                len(ACTION_ORDER) * (HIDDEN_COUNT + 1))
        self.metrics_state["reward_observations"] += 1
        if self.learning and self.encoder_learning:
            if self.pending_hidden_signal is None:
                raise RuntimeError("reference encoder signal missing")
            changed = self._update_projection(
                self.encoder, self.pending_hidden_signal,
                self.pending_rate_derivative, self.pending_features,
                1.0, advantage, transposed=False)
            if self.active_context is None:
                raise RuntimeError("reference context missing at feedback")
            changed += self._update_projection(
                self.projection, self.pending_hidden_signal,
                self.pending_rate_derivative, self.active_context,
                CONTEXT_SCALE, advantage, transposed=True)
            self.metrics_state["plastic_encoder_updates"] += 1
            self.metrics_state["plastic_history_update_synapse_visits"] += (
                HIDDEN_COUNT * FEATURE_COUNT)
            self.metrics_state["plastic_context_update_synapse_visits"] += (
                HIDDEN_COUNT * CONTEXT_COUNT)
            self.metrics_state["plastic_encoder_changed_synapses"] += changed
        self.pending_action = None
        self._clear_pending()
        return advantage

    def end_episode(self) -> None:
        """End a completed task and clear membrane/context/credit state."""
        if self.episode_id is None or self.pending_action is not None:
            raise RuntimeError("reference episode cannot end")
        self.episode_id = None
        self.active_context = None
        self.context_current = None
        self.membrane = [0.0] * HIDDEN_COUNT
        self._clear_pending()

    def _encode(self, features: list[float]
                ) -> tuple[list[float], int, list[float]]:
        """Compute rates and the independent scalar eligibility recurrence."""
        if self.active_context is None:
            raise RuntimeError("reference context not active")
        if self.encoder_learning and self.learning:
            self.context_current = self._project_context(self.active_context)
            self.metrics_state["plastic_context_projection_macs"] += (
                CONTEXT_COUNT * HIDDEN_COUNT)
            self.metrics_state["plastic_context_projection_recomputes"] += 1
            self.metrics_state["plastic_eligibility_neuron_ticks"] += (
                STEPS * HIDDEN_COUNT)
        elif self.context_current is None:
            self.context_current = self._project_context(self.active_context)
        rate_sums = [0.0] * HIDDEN_COUNT
        derivative_sums = [0.0] * HIDDEN_COUNT
        spikes_total = 0
        for hidden in range(HIDDEN_COUNT):
            derivative_membrane = 0.0
            for _step in range(STEPS):
                base_current = sum(
                    weight * value for weight, value
                    in zip(self.encoder[hidden], features))
                current = base_current + self.context_current[hidden]
                voltage = BETA * self.membrane[hidden] + current
                if self.encoder_learning and self.learning:
                    derivative_current = BETA * derivative_membrane + 1.0
                if self.spiking:
                    spike = int(voltage >= THRESHOLD)
                    if self.encoder_learning and self.learning:
                        psi = SURROGATE * max(
                            0.0, 1.0 - abs(voltage - THRESHOLD))
                        ds = psi * derivative_current
                        derivative_membrane = (
                            (1 - spike) * derivative_current - voltage * ds)
                        derivative_sums[hidden] += ds
                    self.membrane[hidden] = 0.0 if spike else voltage
                    rate_sums[hidden] += spike
                    spikes_total += spike
                else:
                    rate = math.tanh(voltage)
                    if self.encoder_learning and self.learning:
                        ds = (1.0 - rate * rate) * derivative_current
                        derivative_membrane = derivative_current
                        derivative_sums[hidden] += ds
                    self.membrane[hidden] = voltage
                    rate_sums[hidden] += rate
        rates = [value / STEPS for value in rate_sums]
        derivatives = [value / STEPS for value in derivative_sums]
        return rates, spikes_total, derivatives

    def _project_context(self, context: list[float]) -> list[float]:
        """Project normalized observed triples with current context weights."""
        return [sum(self.projection[index][hidden] * value
                    for index, value in enumerate(context)) / CONTEXT_SCALE
                for hidden in range(HIDDEN_COUNT)]

    def _update_projection(
        self, matrix: list[list[float]], signal: list[float],
        derivative: list[float], inputs: list[float], scale: float,
        advantage: float, *, transposed: bool,
    ) -> int:
        """Apply the frozen per-synapse clipping and weight bounds."""
        changed = 0
        for hidden in range(HIDDEN_COUNT):
            hidden_signal = signal[hidden] * derivative[hidden]
            for index, value in enumerate(inputs):
                delta = self._clip_update(
                    ENCODER_LR * advantage * hidden_signal * value / scale)
                if delta:
                    row, column = ((index, hidden) if transposed
                                   else (hidden, index))
                    matrix[row][column] = max(
                        -WEIGHT_LIMIT,
                        min(WEIGHT_LIMIT, matrix[row][column] + delta))
                    changed += 1
        return changed

    def _softmax(self, logits: list[float],
                 legal: tuple[int, ...]) -> dict[int, float]:
        """Stable softmax over only the available root-operation lanes."""
        maximum = max(logits[index] for index in legal)
        exponential = {index: math.exp(logits[index] - maximum)
                       for index in legal}
        total = sum(exponential.values())
        return {index: value / total for index, value in exponential.items()}

    @staticmethod
    def _clip_update(value: float) -> float:
        """Clip one parameter delta to the protocol bound."""
        return max(-READOUT_MAX_DELTA, min(READOUT_MAX_DELTA, value))

    def _clear_pending(self) -> None:
        """Clear action-local eligibility and pending hidden credit."""
        self.pending_action = None
        self.pending_readout_eligibility = None
        self.pending_bias_eligibility = None
        self.pending_hidden_signal = None
        self.pending_rate_derivative = [0.0] * HIDDEN_COUNT
        self.pending_features = [0.0] * FEATURE_COUNT

    def _config(self) -> dict[str, Any]:
        """Return exact immutable configuration fields from the protocol."""
        config = {
            "seed": self.seed, "learning": self.learning,
            "spiking": self.spiking, "lr": READOUT_LR,
            "eligibility_decay": 0.0, "feature_count": FEATURE_COUNT,
            "hidden_count": HIDDEN_COUNT, "steps": STEPS,
            "beta": BETA, "threshold": THRESHOLD,
            "baseline_rate": BASELINE_RATE,
            "max_update": READOUT_MAX_DELTA, "max_weight": WEIGHT_LIMIT,
            "temperature": 1.0,
            "observed_context_version": 1, "observed_rows": 6,
            "observed_context_dimensions": CONTEXT_COUNT,
            "observed_context_scale": CONTEXT_SCALE,
            "observed_context_transform": (
                "v/(1+abs(v)); row-major x,y,target"),
            "observed_context_projection_seed": PROJECTION_SEED_PREFIX,
            "observed_context_projection_sha256": _matrix_hash(
                self.projection),
            "plastic_encoder_version": 1,
            "encoder_learning": self.encoder_learning,
            "encoder_lr": ENCODER_LR,
            "encoder_max_update": ENCODER_MAX_DELTA,
            "encoder_max_weight": WEIGHT_LIMIT,
            "encoder_surrogate_scale": SURROGATE,
            "encoder_surrogate": "triangular threshold surrogate",
            "encoder_projection_dimensions": [HIDDEN_COUNT, FEATURE_COUNT],
            "context_projection_dimensions": [CONTEXT_COUNT, HIDDEN_COUNT],
            "encoder_initial_sha256": _matrix_hash(self.initial_encoder),
            "context_initial_sha256": _matrix_hash(
                self.initial_projection),
        }
        return config

    def encoder_hash(self) -> str:
        """Hash both learned input projections."""
        return _model_hash({"encoder": self.encoder,
                            "context_projection": self.projection})

    def weight_hash(self) -> str:
        """Hash both learned input projections and the readout."""
        return _model_hash({"encoder": self.encoder,
                            "context_projection": self.projection,
                            "weights": self.weights, "bias": self.bias})

    def metrics(self) -> dict[str, Any]:
        """Return runtime-compatible counters and state hashes."""
        return {**self.metrics_state,
                "encoder_mode": ("binary_lif" if self.spiking
                                 else "continuous_leaky"),
                "weight_hash": self.weight_hash(),
                "encoder_hash": self.encoder_hash()}

    def snapshot(self) -> dict[str, Any]:
        """Build the full version-2 boundary state for independent hashing."""
        if self.episode_id is not None or self.pending_context is not None:
            raise RuntimeError("reference snapshot requires boundary")
        return {
            "version": 2,
            "feature_encoding": "online-plastic-encoder-context-v1",
            "actions": list(ACTION_ORDER), "config": self._config(),
            "encoder": [row[:] for row in self.encoder],
            "context_projection": [row[:] for row in self.projection],
            "weights": [row[:] for row in self.weights],
            "bias": self.bias[:], "baseline": self.baseline,
            "rng_state": _nested_lists(self.rng.getstate()),
            "metrics": dict(self.metrics_state),
            "readout_hash": _readout_hash(self.weights, self.bias),
            "encoder_hash": self.encoder_hash(),
            "weight_hash": self.weight_hash(),
        }

    def full_state_hash(self) -> str:
        """Hash the exact full boundary snapshot including metrics and RNG."""
        return _sha256(_json_bytes(self.snapshot()))

    def evaluation_copy(self) -> ReferencePlasticPolicy:
        """Return a cold frozen reference copy for retention probes."""
        if self.episode_id is not None or self.pending_context is not None:
            raise RuntimeError("reference evaluation copy requires boundary")
        clone = ReferencePlasticPolicy(self.arm, self.seed)
        clone.learning = False
        clone.encoder_learning = False
        clone.rng.setstate(self.rng.getstate())
        clone.encoder = [row[:] for row in self.encoder]
        clone.projection = [row[:] for row in self.projection]
        clone.weights = [row[:] for row in self.weights]
        clone.bias = self.bias[:]
        clone.baseline = self.baseline
        clone.metrics_state = dict(self.metrics_state)
        clone._clear_pending()
        return clone


def _protocol() -> dict[str, Any]:
    """Check byte pin and frozen inputs before any bundle replay."""
    payload = PROTOCOL_PATH.read_bytes()
    if _sha256(payload) != PROTOCOL_SHA256:
        raise ValueError("plastic-encoder protocol SHA mismatch")
    protocol = json.loads(payload)["protocol"]
    if (protocol["source_corpus"]["sha256"] != CORPUS_SHA256
            or protocol["source_schedule"]["sha256"] != SCHEDULE_SHA256
            or tuple(protocol["arms"]) != ARMS
            or tuple(protocol["model_seeds"]) != SEEDS
            or tuple(protocol["budgets"]) != BUDGETS
            or protocol["matrix"]["episode_rows"] != 13824):
        raise ValueError("plastic-encoder protocol matrix changed")
    return protocol


def _schedule() -> dict[str, Any]:
    """Load the nested schedule fixture after checking its frozen bytes."""
    path = ROOT / "fixtures" / "online_counterbalanced_v1.json"
    payload = path.read_bytes()
    if _sha256(payload) != SCHEDULE_SHA256:
        raise ValueError("frozen schedule SHA mismatch")
    schedule = json.loads(payload)["protocol"]
    if (not isinstance(schedule.get("orders"), list)
            or len(schedule["orders"]) != 6):
        raise ValueError("frozen schedule order matrix mismatch")
    return schedule


def _manifest_float(manifest: dict[str, Any], key: str) -> float:
    """Read a finite nonnegative process timing field."""
    value = manifest.get(key)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or value < 0):
        raise ValueError(f"invalid manifest timing: {key}")
    return float(value)


def _validate_context_work(row: dict[str, Any]) -> None:
    """Check observed-context digest and setup counter deltas."""
    work = row.get("context_work")
    expected = {
        name: (CONTEXT_COUNT if name == "observation_scalars"
               else CONTEXT_COUNT * HIDDEN_COUNT
               if name == "projection_mac" else 0)
        for name in ReferencePlasticPolicy("persistent_snn", 0).metrics_state
    }
    if work != expected:
        raise ValueError("observed context setup counters mismatch")


def _audit_task_rows(
    bundle: Path,
    protocol: dict[str, Any],
    tasks: list[dict[str, Any]],
    orders: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, int, int, str, int],
                                     ReferencePlasticPolicy]]:
    """Replay every ordered task with the independent policy emulator."""
    expected = [(order, budget, seed, arm, task_id)
                for order in orders for budget in BUDGETS for seed in SEEDS
                for arm in ARMS for task_id in order["task_ids"]]
    path = bundle / "episodes.jsonl.gz"
    compact: list[dict[str, Any]] = []
    boundary_states: dict[
        tuple[str, int, int, str, int], ReferencePlasticPolicy] = {}
    policies: dict[tuple[str, int, int, str],
                   ReferencePlasticPolicy | None] = {}
    offsets: dict[tuple[str, int, int, str], int] = {}
    counts: dict[tuple[str, int, int, str], int] = {}
    task_by_id = {task["id"]: task for task in tasks}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line_number > len(expected):
                raise ValueError("extra plastic-encoder episode row")
            order, budget, seed, arm, task_id = expected[line_number - 1]
            row = json.loads(line)
            identity = (order["order_id"], budget, seed, arm, task_id)
            actual = (row.get("order_id"), row.get("budget"),
                      row.get("seed"), row.get("arm"), row.get("task_id"))
            if actual != identity:
                raise ValueError(f"episode sequence mismatch at {line_number}")
            group = order["order_id"], budget, seed, arm
            if group not in policies:
                policies[group] = (None if arm == "grammar_streaming"
                                   else ReferencePlasticPolicy(arm, seed))
                offsets[group] = 0
                counts[group] = 0
            policy = policies[group]
            if policy is not None and counts[group] == 0:
                boundary_states[(*group, 0)] = copy.deepcopy(policy)
            if row.get("phase_sequence") != order["phase_sequence"]:
                raise ValueError("episode phase-order metadata mismatch")
            stream_index = counts[group]
            if (row.get("stream_index") != stream_index
                    or row.get("phase_position") != stream_index // 16
                    or row.get("phase_ordinal") != stream_index % 16):
                raise ValueError("episode stream position mismatch")
            if policy is not None:
                if arm in RESET_ARMS:
                    policy.reset_learning()
                before = policy.snapshot()
                _equal(row.get("readout_hash_before"),
                       before["readout_hash"], "pre-task readout hash")
                _equal(row.get("encoder_hash_before"),
                       before["encoder_hash"], "pre-task encoder hash")
                _equal(row.get("learning_model_hash_before"),
                       before["weight_hash"], "pre-task model hash")
                _equal(row.get("full_state_sha256_before"),
                       _sha256(_json_bytes(before)), "pre-task full state")
                if row.get("rng_draw_offset_before") != offsets[group]:
                    raise ValueError("pre-task RNG offset mismatch")
                task = task_by_id[task_id]
                before_context_metrics = policy.metrics()
                observed = tuple(tuple(item) for item in task["observed"])
                policy.set_observed_rows(observed)
                expected_context = [value / (1 + abs(value))
                                    for item in observed for value in item]
                if policy.pending_context != expected_context:
                    raise ValueError("reference normalized context mismatch")
                _validate_context_work(row)
                context_work = {
                    name: policy.metrics_state[name]
                    - before_context_metrics[name]
                    for name in policy.metrics_state}
                if row.get("context_work") != context_work:
                    raise ValueError("context work does not replay")
                synthetic = dict(row)
                synthetic["weight_before"] = before["weight_hash"]
                synthetic["weight_after"] = row.get(
                    "learning_model_hash_after")
                # The graph auditor validates its original CPU-field contract;
                # preserve the runner's actual measured values where present.
                synthetic.setdefault("episode_wall_seconds", 0.0)
                synthetic.setdefault("policy_cpu_seconds", 0.0)
                synthetic.setdefault("update_cpu_seconds", 0.0)
                audit_episode_row(synthetic, task, budget, policy)
                after = policy.snapshot()
                _equal(row.get("readout_hash_after"),
                       after["readout_hash"], "post-task readout hash")
                _equal(row.get("encoder_hash_after"),
                       after["encoder_hash"], "post-task encoder hash")
                _equal(row.get("learning_model_hash_after"),
                       after["weight_hash"], "post-task model hash")
                _equal(row.get("full_state_sha256_after"),
                       _sha256(_json_bytes(after)), "post-task full state")
                if row.get("rng_draw_offset_after") != (
                        offsets[group] + row["decision_calls"]):
                    raise ValueError("post-task RNG offset mismatch")
                offsets[group] = row["rng_draw_offset_after"]
                if arm in ("frozen_snn", "episode_reset_snn"):
                    initial = ReferencePlasticPolicy(arm, seed)
                    if (after["encoder"] != initial.encoder
                            or after["context_projection"]
                            != initial.projection):
                        raise ValueError("fixed legacy encoder changed")
                if arm == "frozen_snn" and after["weight_hash"] != (
                        ReferencePlasticPolicy(arm, seed).weight_hash()):
                    raise ValueError("frozen readout changed")
            else:
                if row.get("policy_work") != {}:
                    raise ValueError("streaming row reports policy work")
                if any(row.get(key) is not None for key in (
                        "readout_hash_before", "readout_hash_after",
                        "encoder_hash_before", "encoder_hash_after",
                        "learning_model_hash_before",
                        "learning_model_hash_after",
                        "full_state_sha256_before",
                        "full_state_sha256_after")):
                    raise ValueError("streaming row contains model state")
                if row.get("rng_draw_offset_before") != 0 \
                        or row.get("rng_draw_offset_after") != 0:
                    raise ValueError("streaming row contains RNG draws")
                task = task_by_id[task_id]
                synthetic = dict(row)
                synthetic.setdefault("episode_wall_seconds", 0.0)
                synthetic.setdefault("policy_cpu_seconds", 0.0)
                synthetic.setdefault("update_cpu_seconds", 0.0)
                audit_episode_row(synthetic, task, budget, None)
            counts[group] += 1
            if policy is not None and counts[group] in BOUNDARIES[1:]:
                boundary_states[(*group, counts[group])] = copy.deepcopy(
                    policy)
            if row.get("decision_calls") != len(row.get("attempts", [])) \
                    * (policy is not None):
                raise ValueError("decision call count mismatch")
            for name in ("state_hash_cpu_seconds", "reset_cpu_seconds",
                         "context_cpu_seconds", "episode_cpu_seconds",
                         "checker_cpu_seconds", "engine_episode_cpu_seconds",
                         "engine_checker_cpu_seconds"):
                value = row.get(name)
                if (isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value)) or value < 0):
                    raise ValueError(f"invalid episode timing: {name}")
            if row.get("model_state_json_bytes", 0) < 0:
                raise ValueError("negative serialized state size")
            compact.append({key: value for key, value in row.items()
                            if key != "attempts"})
    if len(compact) != len(expected):
        raise ValueError(f"incomplete episode matrix {len(compact)}"
                         f"/{len(expected)}")
    return compact, boundary_states


def _audit_retention(
    bundle: Path,
    rows: dict[tuple[str, int, int, str, int], ReferencePlasticPolicy],
    tasks: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Independently replay every cold context probe and no-mutation hash."""
    path = bundle / "retention.jsonl.gz"
    digest = _sha256(path.read_bytes())
    if digest != manifest.get("retention_raw_sha256"):
        raise ValueError("retention stream SHA mismatch")
    expected = [(order, budget, seed, arm, boundary)
                for order in _schedule()["orders"]
                for budget in BUDGETS for seed in SEEDS
                for arm in NEURAL_ARMS for boundary in BOUNDARIES]
    expected_tasks = tasks
    output: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line_number > len(expected):
                raise ValueError("extra retention probe group")
            order, budget, seed, arm, boundary = expected[line_number - 1]
            order_id = order["order_id"]
            row = json.loads(line)
            key = (order_id, budget, seed, arm, boundary)
            state = rows.get(key)
            if state is None:
                raise ValueError("missing retained model boundary")
            for name, value in zip(
                    ("order_id", "budget", "seed", "arm",
                     "completed_tasks"), key):
                if row.get(name) != value:
                    raise ValueError("retention row order/identity mismatch")
            if row.get("phase_sequence") != order["phase_sequence"]:
                raise ValueError("retention phase-order metadata mismatch")
            state_hash = state.full_state_hash()
            _equal(row.get("source_state_sha256_before"), state_hash,
                   "retention source before")
            _equal(row.get("source_state_sha256_after"), state_hash,
                   "retention source after")
            evaluation = state.evaluation_copy()
            eval_snapshot = evaluation.snapshot()
            _equal(row.get("evaluation_copy_state_sha256"),
                   _sha256(_json_bytes(eval_snapshot)),
                   "evaluation copy state")
            _equal(row.get("readout_hash"), eval_snapshot["readout_hash"],
                   "evaluation readout hash")
            _equal(row.get("encoder_hash"), eval_snapshot["encoder_hash"],
                   "evaluation encoder hash")
            _equal(row.get("weight_hash"), eval_snapshot["weight_hash"],
                   "evaluation weight hash")
            contexts = row.get("contexts")
            if not isinstance(contexts, list) or len(contexts) != len(tasks):
                raise ValueError("retention context set incomplete")
            code_set: set[tuple[float, ...]] = set()
            work: dict[str, int] = {}
            for probe, task in zip(contexts, expected_tasks):
                if probe.get("task_id") != task["id"]:
                    raise ValueError("retention contexts not canonical order")
                observed = tuple(tuple(item) for item in task["observed"])
                context_sha = _sha256(_json_bytes(observed))
                if probe.get("context_sha256") != context_sha:
                    raise ValueError("retention context digest mismatch")
                evaluation.set_observed_rows(observed)
                evaluation.start_episode(f"retention:{task['id']}")
                rates, _spikes, _derivative = evaluation._encode(
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
                logits = [sum(weight * rate for weight, rate in zip(
                    evaluation.weights[index], rates))
                    + evaluation.bias[index]
                    for index in range(len(ACTION_ORDER))]
                probs_by = evaluation._softmax(logits, (0, 1, 2))
                probabilities = [probs_by[index] for index in range(3)]
                root_index = ACTION_ORDER.index(task["root"])
                expected_probe = {
                    "task_id": task["id"], "phase": task["phase"],
                    "root": task["root"], "context_sha256": context_sha,
                    "rates": rates, "probabilities": probabilities,
                    "root_probability": probabilities[root_index],
                    "argmax": ACTION_ORDER[max(
                        range(3), key=probabilities.__getitem__)],
                }
                _equal(probe, expected_probe, "retention context probe")
                code_set.add(tuple(round(value, 12) for value in rates))
                evaluation.end_episode()
                for name in evaluation.metrics_state:
                    delta = (evaluation.metrics_state[name]
                             - eval_snapshot["metrics"][name])
                    # Each evaluation copy starts from identical source
                    # counters, so accumulate the per-probe deltas below.
                    work[name] = work.get(name, 0) + delta
                eval_snapshot = evaluation.snapshot()
            if row.get("distinct_rate_codes") != len(code_set):
                raise ValueError("retention distinct rate-code count mismatch")
            if row.get("probe_count") != len(tasks):
                raise ValueError("retention probe count mismatch")
            if row.get("probe_work_counters") != work:
                raise ValueError("retention probe work counters mismatch")
            explicit = row.get("explicit_probe_work")
            expected_explicit = {
                "evaluation_copies": 48,
                "observed_context_sets": 48,
                "cold_feature_forwards": 48,
                "root_softmaxes": 48,
                "neuron_ticks": 48 * STEPS * HIDDEN_COUNT,
                "base_encoder_multiply_accumulates": (
                    48 * STEPS * HIDDEN_COUNT * FEATURE_COUNT),
                "context_additions": 48 * STEPS * HIDDEN_COUNT,
                "readout_multiply_accumulates": (
                    48 * len(ACTION_ORDER) * HIDDEN_COUNT),
                "context_projection_multiply_accumulates": (
                    work.get("projection_mac", 0)),
            }
            if explicit != expected_explicit:
                raise ValueError("retention explicit work missing")
            if row.get("probe_cpu_seconds") is None:
                raise ValueError("retention CPU charge missing")
            if row.get("source_state_sha256_before") != (
                    row.get("source_state_sha256_after")):
                raise ValueError("retention probe mutated source state")
            output.append(row)
    if len(output) != len(expected):
        raise ValueError(f"retention matrix incomplete {len(output)}"
                         f"/{len(expected)}")
    return output


def _sum_counters(rows: list[dict[str, Any]], field: str
                  ) -> dict[str, int]:
    """Sum sparse nonnegative integer work-counter maps."""
    result: dict[str, int] = defaultdict(int)
    for row in rows:
        counters = row.get(field)
        if not isinstance(counters, dict):
            raise ValueError(f"{field} must be an object")
        for name, value in counters.items():
            if type(value) is not int or value < 0:
                raise ValueError(f"invalid {field} counter {name}")
            result[name] += value
    return dict(result)


def _derive_seed(rows: list[dict[str, Any]], setup: float,
                 probe: float) -> dict[str, Any]:
    """Recompute per-seed totals directly from raw episode records."""
    search = setup + sum(
        row[name] for row in rows
        for name in ("reset_cpu_seconds", "context_cpu_seconds",
                     "episode_cpu_seconds", "checker_cpu_seconds",
                     "state_hash_cpu_seconds"))
    return {
        "tasks": len(rows),
        "accepted": sum(bool(row["accepted"]) for row in rows),
        "first_lane_correct": sum(
            bool(row["first_lane_correct"]) for row in rows),
        "candidate_evaluations": sum(
            int(row["candidate_evaluations"]) for row in rows),
        "updates": sum(row["policy_work"].get("updates", 0)
                       for row in rows),
        "plastic_encoder_updates": sum(
            row["policy_work"].get("plastic_encoder_updates", 0)
            for row in rows),
        "node_row_operations": sum(row["node_row_operations"]
                                    for row in rows),
        "interpreter_rows": sum(row["interpreter_rows"] for row in rows),
        "policy_work_counters": _sum_counters(rows, "policy_work"),
        "context_work_counters": _sum_counters(rows, "context_work"),
        "setup_cpu_seconds": setup,
        "search_only_cpu_seconds": search,
        "retention_probe_cpu_seconds": probe,
        "total_cpu_seconds": search + probe,
    }


def _derive_summary(
    compact: list[dict[str, Any]], manifest: dict[str, Any],
    orders: list[dict[str, Any]],
) -> dict[str, Any]:
    """Independently rebuild the published per-order gates and summaries."""
    grouped: dict[tuple[str, int, str, int], list[dict[str, Any]]]
    grouped = defaultdict(list)
    for row in compact:
        grouped[(row["order_id"], row["budget"], row["arm"],
                 row["seed"])].append(row)
    setup_map, probe_map = (manifest.get("setup_cpu_seconds"),
                            manifest.get("probe_cpu_seconds"))
    if not isinstance(setup_map, dict) or not isinstance(probe_map, dict):
        raise ValueError("setup/probe CPU maps missing")

    per_order = []
    for order in orders:
        order_id = order["order_id"]
        seed_metrics: dict[tuple[int, str], dict[int, dict[str, Any]]]
        seed_metrics = defaultdict(dict)
        for budget in BUDGETS:
            for arm in ARMS:
                for seed in SEEDS:
                    key = f"{order_id}:{budget}:{arm}:{seed}"
                    items = grouped.get((order_id, budget, arm, seed), [])
                    if len(items) != 48:
                        raise ValueError("summary source cell is incomplete")
                    if key not in setup_map or key not in probe_map:
                        raise ValueError("per-cell CPU charge missing")
                    for field in (setup_map[key], probe_map[key]):
                        if (isinstance(field, bool)
                                or not isinstance(field, (int, float))
                                or not math.isfinite(float(field))
                                or field < 0):
                            raise ValueError("invalid per-cell CPU charge")
                    values = _derive_seed(
                        items, float(setup_map[key]), float(probe_map[key]))
                    values["seed"] = seed
                    seed_metrics[(budget, arm)][seed] = values
        per_seed = []
        pooled = []
        aggregate: dict[tuple[int, str], dict[str, Any]] = {}
        for (budget, arm), by_seed in sorted(seed_metrics.items()):
            combined = {
                "budget": budget, "arm": arm,
                "tasks": sum(item["tasks"] for item in by_seed.values()),
                "seed_metrics": {str(seed): item
                                 for seed, item in by_seed.items()},
            }
            for metric in ("accepted", "first_lane_correct",
                           "candidate_evaluations", "updates",
                           "plastic_encoder_updates",
                           "search_only_cpu_seconds",
                           "retention_probe_cpu_seconds",
                           "total_cpu_seconds"):
                combined[metric] = sum(item[metric]
                                       for item in by_seed.values())
            aggregate[(budget, arm)] = combined
            per_seed.extend({"budget": budget, "arm": arm, **item}
                            for item in by_seed.values())
            pooled.append(combined)
        complete = all(len(grouped.get((order_id, budget, arm, seed), []))
                       == 48 for budget in BUDGETS for arm in ARMS
                       for seed in SEEDS)
        primary = aggregate[(144, "plastic_snn")]
        fixed = aggregate[(144, "persistent_snn")]
        reset = aggregate[(144, "reset_plastic_snn")]
        ann = aggregate[(144, "plastic_nonspiking")]

        def margin(ours: dict[str, Any], other: dict[str, Any],
                   each_seed: bool) -> bool:
            if (ours["first_lane_correct"] - other["first_lane_correct"]
                    < 0.10 * ours["tasks"]):
                return False
            return (not each_seed or all(
                ours["seed_metrics"][str(seed)]["first_lane_correct"]
                > other["seed_metrics"][str(seed)]["first_lane_correct"]
                for seed in SEEDS))

        gates = {
            "plastic_snn_vs_persistent_snn": (
                margin(primary, fixed, True)
                and primary["accepted"] >= fixed["accepted"]
                and primary["accepted"] >= reset["accepted"]),
            "plastic_snn_coverage_vs_reset_plastic_snn": (
                primary["accepted"] >= reset["accepted"]),
            "plastic_snn_vs_plastic_nonspiking": (
                margin(primary, ann, False)
                and primary["accepted"] >= ann["accepted"]
                and primary["total_cpu_seconds"]
                <= ann["total_cpu_seconds"]),
            "streaming_deployment": all(
                aggregate[(budget, "plastic_snn")]["accepted"]
                >= aggregate[(budget, "grammar_streaming")]["accepted"]
                and aggregate[(budget, "plastic_snn")]["total_cpu_seconds"]
                <= aggregate[(budget, "grammar_streaming")]
                ["total_cpu_seconds"]
                for budget in BUDGETS),
        }
        if not complete:
            gates = {name: False for name in gates}
        per_order.append({
            "order_id": order_id,
            "matrix_complete": complete,
            "phase_sequence": order["phase_sequence"],
            "per_seed": sorted(
                per_seed,
                key=lambda item: (item["budget"], item["arm"],
                                  item["seed"])),
            "pooled": pooled,
            "gates": gates,
        })
    gate_names = tuple(per_order[0]["gates"] if per_order else ())
    return {
        "per_order": per_order,
        "all_order_gates": {
            name: all(order["gates"][name] for order in per_order)
            for name in gate_names
        },
        "scope": "One 48-task stream repeated over six orders, three seeds, "
                 "and two budgets; task repetitions are not independent.",
    }


def _retention_summary(
    rows: list[dict[str, Any]], orders: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rebuild phase-end versus final root-probability summaries."""
    indexed = {(row["order_id"], row["budget"], row["seed"], row["arm"],
                row["completed_tasks"]): row for row in rows}
    result = []
    for order in orders:
        for budget in BUDGETS:
            for seed in SEEDS:
                for arm in NEURAL_ARMS:
                    final = indexed[(order["order_id"], budget, seed, arm, 48)]
                    final_map = {item["task_id"]: item
                                 for item in final["contexts"]}
                    for phase_index, phase in enumerate(
                            order["phase_sequence"]):
                        boundary = (phase_index + 1) * 16
                        checkpoint = indexed[(order["order_id"], budget,
                                              seed, arm, boundary)]
                        context_map = {item["task_id"]: item
                                       for item in checkpoint["contexts"]}
                        selected = [item for item in context_map.values()
                                    if item["phase"] == phase]
                        if len(selected) != 16:
                            raise ValueError("retention phase size mismatch")
                        ids = [item["task_id"] for item in selected]
                        result.append({
                            "order_id": order["order_id"], "budget": budget,
                            "seed": seed, "arm": arm, "phase": phase,
                            "phase_end_completed_tasks": boundary,
                            "tasks": 16,
                            "mean_phase_end_root_probability": sum(
                                context_map[task_id]["root_probability"]
                                for task_id in ids) / 16,
                            "mean_final_root_probability": sum(
                                final_map[task_id]["root_probability"]
                                for task_id in ids) / 16,
                        })
    return result


def audit_bundle(bundle: Path, repo_root: Path = ROOT) -> dict[str, Any]:
    """Audit provenance, the ordered matrix, retained probes, and summary."""
    protocol = _protocol()
    corpus_path = repo_root / "fixtures" / "online_graph_v1.json"
    corpus_payload = corpus_path.read_bytes()
    schedule_payload = (
        repo_root / "fixtures" / "online_counterbalanced_v1.json").read_bytes()
    if (_sha256(corpus_payload) != CORPUS_SHA256
            or _sha256(schedule_payload) != SCHEDULE_SHA256):
        raise ValueError("frozen corpus or schedule SHA mismatch")
    corpus = json.loads(corpus_payload)
    schedule = json.loads(schedule_payload)["protocol"]
    tasks, orders = corpus["tasks"], schedule["orders"]
    if len(tasks) != 48 or len(orders) != 6:
        raise ValueError("frozen corpus/schedule dimensions mismatch")
    manifest = json.loads((bundle / "manifest.json").read_text())
    if (manifest.get("protocol_sha256") != PROTOCOL_SHA256
            or manifest.get("corpus_sha256") != CORPUS_SHA256
            or manifest.get("schedule_sha256") != SCHEDULE_SHA256
            or manifest.get("source_unchanged") is not True):
        raise ValueError("manifest provenance mismatch")
    source_files = manifest.get("source_files")
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("source hash manifest missing")
    verify_source_hashes(source_files, repo_root)
    episode_path = bundle / "episodes.jsonl.gz"
    episode_bytes = episode_path.read_bytes()
    if _sha256(episode_bytes) != manifest.get("episodes_raw_sha256"):
        raise ValueError("compressed episode stream SHA mismatch")
    expected_count = len(orders) * len(BUDGETS) * len(SEEDS) * len(ARMS) * 48
    if (manifest.get("episode_rows") != expected_count
            or expected_count != 13824):
        raise ValueError("manifest episode count mismatch")
    compact, boundaries = _audit_task_rows(
        bundle, protocol, tasks, orders)
    for order in orders:
        for budget in BUDGETS:
            for seed in SEEDS:
                for arm in NEURAL_ARMS:
                    group = (order["order_id"], budget, seed, arm)
                    expected_boundaries = {
                        (*group, boundary) for boundary in BOUNDARIES}
                    if not expected_boundaries.issubset(boundaries):
                        raise ValueError("retention boundary replay missing")
    raw_retention = bundle / "retention.jsonl.gz"
    retained_rows = _audit_retention(bundle, boundaries, tasks, manifest)
    summary = json.loads((bundle / "summary.json").read_text())
    expected_summary = _derive_summary(compact, manifest, orders)
    expected_summary["retention"] = _retention_summary(retained_rows, orders)
    _equal(summary, expected_summary, "summary")
    for name in ("wall_seconds", "full_process_cpu_seconds"):
        _manifest_float(manifest, name)
    peak = manifest.get("whole_process_peak_rss_kib")
    if type(peak) is not int or peak <= 0:
        raise ValueError("invalid whole-process peak RSS")
    if manifest.get("episode_rows") != len(compact):
        raise ValueError("episode raw row count mismatch")
    return {
        "status": "passed",
        "episode_rows": len(compact),
        "retention_rows": len(retained_rows),
        "protocol_sha256": PROTOCOL_SHA256,
        "corpus_sha256": CORPUS_SHA256,
        "schedule_sha256": SCHEDULE_SHA256,
        "episodes_raw_sha256": _sha256(episode_bytes),
        "retention_raw_sha256": _sha256(raw_retention.read_bytes()),
        "source_file_count": len(source_files),
        "source_unchanged": True,
    }


def main() -> None:
    """Run the independent plastic-encoder bundle audit."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_bundle(args.bundle), sort_keys=True))


if __name__ == "__main__":
    main()
