"""Independent surrogate-VJP checks for the plastic encoder audit."""
from __future__ import annotations

import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from rsi_programmer.online_plastic_benchmark import (
    _attempt_row,
    load_protocol,
    policy_for,
    probe_retention,
)
from rsi_programmer.online_benchmark import load_corpus
from rsi_programmer.online_counterbalanced import load_schedule
from scripts.audit_online_plastic_encoder import (
    _audit_retention,
    _audit_task_rows,
    PROTOCOL_SHA256,
    PROTOCOL_PATH,
    ReferencePlasticPolicy,
    _protocol,
    _schedule,
    _sha256,
)
from rsi_programmer.online_plastic_encoder import OnlinePlasticEncoderPolicy

ROWS = ((1, 2, 3), (2, 1, 4), (3, 2, 5), (4, 3, 6), (5, 4, 7),
        (6, 5, 8))
FEATURES = (1.0, 0.0, 0.0, 0.0, 0.25, 0.5)
AVAILABLE = ("add", "mul")


def _torch_vjp(
    encoder: list[list[float]],
    projection: list[list[float]],
    readout: list[list[float]],
    bias: list[float],
    context: list[float],
    features: tuple[float, ...],
    initial_membrane: list[float],
    action: str,
) -> tuple[list[list[float]], list[list[float]]]:
    """Differentiate log pi(action) with an explicit hard-forward surrogate."""
    e = torch.tensor(encoder, dtype=torch.float64, requires_grad=True)
    c = torch.tensor(projection, dtype=torch.float64, requires_grad=True)
    w = torch.tensor(readout, dtype=torch.float64)
    b = torch.tensor(bias, dtype=torch.float64)
    x = torch.tensor(features, dtype=torch.float64)
    h = torch.tensor(context, dtype=torch.float64)
    initial = torch.tensor(initial_membrane, dtype=torch.float64)
    current = e @ x + (c.T @ h) / 18.0
    membrane = initial
    rate = torch.zeros(16, dtype=torch.float64)
    for _ in range(6):
        voltage = 0.8 * membrane + current
        hard = (voltage >= 1.0).to(torch.float64)
        psi = (0.3 * (1.0 - torch.abs(voltage - 1.0)).clamp(min=0.0)
               ).detach()
        spike = hard + (voltage - voltage.detach()) * psi
        rate = rate + spike / 6.0
        membrane = voltage * (1.0 - spike)
    logits = w @ rate + b
    allowed = torch.tensor([0, 2], dtype=torch.long)
    log_probs = torch.log_softmax(logits[allowed], dim=0)
    action_index = (0 if action == "add" else 1)
    grad_e, grad_c = torch.autograd.grad(
        log_probs[action_index], (e, c), retain_graph=False,
    )
    return grad_e.detach().tolist(), grad_c.detach().tolist()


class PlasticEncoderAuditMathTests(unittest.TestCase):
    """Check the runtime update against an independent autograd VJP."""

    def test_lif_encoder_and_context_updates_match_surrogate_autograd(
        self,
    ) -> None:
        policy = OnlinePlasticEncoderPolicy(seed=43, encoder_lr=0.01)
        policy._encoder = [
            [0.64 + 0.01 * neuron, 0.0, 0.0, 0.0, 0.0, 0.0]
            for neuron in range(16)
        ]
        policy._context_projection = [
            [((index * 7 + neuron * 3) % 5 - 2) * 0.1
             for neuron in range(16)]
            for index in range(18)
        ]
        policy._context_projection_hash = policy._matrix_hash(
            policy._context_projection)
        policy._weights = [
            [((action - 1) * (neuron + 1)) * 0.015
             for neuron in range(16)]
            for action in range(3)
        ]
        policy._bias = [0.1, -0.15, 0.05]
        policy._baseline = 0.15
        encoder_before = [row[:] for row in policy._encoder]
        projection_before = [row[:] for row in policy._context_projection]
        readout_before = [row[:] for row in policy._weights]
        bias_before = policy._bias[:]
        baseline_before = policy._baseline
        context = [value / (1 + abs(value))
                   for row in ROWS for value in row]

        policy.set_observed_rows(ROWS)
        policy.start_episode("autograd-vjp")
        policy._membrane = [0.1 + 0.02 * neuron
                            for neuron in range(16)]
        membrane_before = policy._membrane[:]
        action = policy.choose(FEATURES, AVAILABLE)
        grad_encoder, grad_context = _torch_vjp(
            encoder_before, projection_before, readout_before, bias_before,
            context, FEATURES, membrane_before, action,
        )
        reward = 0.85
        advantage = reward - baseline_before
        observed_advantage = policy.observe(reward)
        self.assertAlmostEqual(observed_advantage, advantage, places=14)

        for neuron in range(16):
            for feature in range(6):
                expected_delta = 0.01 * advantage * grad_encoder[
                    neuron][feature]
                expected_delta = max(-0.05, min(0.05, expected_delta))
                expected = max(-5.0, min(
                    5.0, encoder_before[neuron][feature] + expected_delta))
                self.assertAlmostEqual(
                    policy._encoder[neuron][feature], expected, places=10,
                )
        for index in range(18):
            for neuron in range(16):
                expected_delta = 0.01 * advantage * grad_context[
                    index][neuron]
                expected_delta = max(-0.05, min(0.05, expected_delta))
                expected = max(-5.0, min(
                    5.0, projection_before[index][neuron] + expected_delta))
                self.assertAlmostEqual(
                    policy._context_projection[index][neuron], expected,
                    places=10,
                )
        self.assertTrue(any(
            abs(policy._membrane[index]) == 0.0
            for index in range(16)))
        self.assertGreater(policy.metrics()["spikes"], 0)
        policy.end_episode()

    def test_reference_replays_all_policy_modes_and_boundaries(
        self,
    ) -> None:
        """Match independent action, learning, and snapshot trajectories."""
        features = FEATURES
        for arm in (
            "persistent_snn", "frozen_snn", "episode_reset_snn",
            "online_nonspiking", "plastic_snn", "reset_plastic_snn",
            "plastic_nonspiking",
        ):
            runtime = OnlinePlasticEncoderPolicy(
                seed=2,
                learning=arm != "frozen_snn",
                spiking=arm not in ("online_nonspiking",
                                    "plastic_nonspiking"),
                encoder_learning=arm in (
                    "plastic_snn", "reset_plastic_snn",
                    "plastic_nonspiking"),
            )
            reference = ReferencePlasticPolicy(arm, 2)
            for episode in range(2):
                if arm in ("episode_reset_snn", "reset_plastic_snn"):
                    runtime.reset_learning()
                    reference.reset_learning()
                observed = ROWS if episode == 0 else tuple(
                    tuple(reversed(row)) for row in ROWS)
                runtime.set_observed_rows(observed)
                reference.set_observed_rows(observed)
                runtime.start_episode(f"replay-{episode}")
                reference.start_episode(f"replay-{episode}")
                for action_index in range(3):
                    active_features = (
                        features if action_index == 0 else
                        (1.0, 1.0, 0.0, 0.0, 0.2, 0.5)
                    )
                    available = ("add", "mul") if action_index == 0 \
                        else ("add", "sub", "mul")
                    runtime_action = runtime.choose(
                        active_features, available)
                    reference_action = reference.choose(
                        active_features, available)
                    self.assertEqual(runtime_action, reference_action)
                    reward = (0.2, 0.9, 0.4)[action_index]
                    self.assertAlmostEqual(
                        runtime.observe(reward),
                        reference.observe(reward), places=14,
                    )
                runtime.end_episode()
                reference.end_episode()
                self.assertEqual(runtime.snapshot(), reference.snapshot())

    def test_auditor_loads_the_pinned_frozen_protocol(self) -> None:
        """Catch protocol parser/path drift before any bundle replay."""
        self.assertEqual(_sha256(PROTOCOL_PATH.read_bytes()), PROTOCOL_SHA256)
        protocol = _protocol()
        schedule = _schedule()
        self.assertEqual(protocol["matrix"]["episode_rows"], 13824)
        self.assertEqual(len(schedule["orders"]), 6)
        self.assertEqual(tuple(protocol["arms"]), (
            "persistent_snn", "frozen_snn", "episode_reset_snn",
            "online_nonspiking", "grammar_streaming", "plastic_snn",
            "reset_plastic_snn", "plastic_nonspiking",
        ))

    def test_valid_episode_prefix_replays_before_incomplete_matrix_error(
        self,
    ) -> None:
        """Prove one runner row replays before the completeness check."""
        protocol = load_protocol()
        corpus, schedule = load_corpus(), load_schedule()
        tasks = corpus["tasks"]
        task_by_id = {task["id"]: task for task in tasks}
        orders = schedule["orders"]
        order = orders[0]
        task_id = order["task_ids"][0]
        runtime = policy_for("persistent_snn", 0)
        assert runtime is not None
        row, draw_offset = _attempt_row(
            task_by_id[task_id], order_id=order["order_id"],
            phase_sequence=list(order["phase_sequence"]), stream_index=0,
            arm="persistent_snn", seed=0, budget=144, policy=runtime,
            draw_offset=0,
        )
        self.assertEqual(draw_offset, row["rng_draw_offset_after"])
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            with gzip.open(bundle / "episodes.jsonl.gz", "wt",
                           encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            with self.assertRaisesRegex(
                    ValueError, "incomplete episode matrix 1/13824"):
                _audit_task_rows(bundle, protocol, tasks, orders)

    def test_valid_retention_prefix_replays_before_incomplete_matrix_error(
        self,
    ) -> None:
        """Replay every cold probe in one valid boundary before EOF check."""
        corpus, schedule = load_corpus(), load_schedule()
        tasks = corpus["tasks"]
        order = schedule["orders"][0]
        runtime = policy_for("persistent_snn", 0)
        assert runtime is not None
        row = probe_retention(
            runtime, tasks, order_id=order["order_id"],
            phase_sequence=list(order["phase_sequence"]), budget=144,
            seed=0, arm="persistent_snn", completed_tasks=0,
        )
        reference = ReferencePlasticPolicy("persistent_snn", 0)
        key = (order["order_id"], 144, 0, "persistent_snn", 0)
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            raw = bundle / "retention.jsonl.gz"
            with gzip.open(raw, "wt", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            manifest = {"retention_raw_sha256": hashlib.sha256(
                raw.read_bytes()).hexdigest()}
            with self.assertRaisesRegex(
                    ValueError, "retention matrix incomplete 1/1008"):
                _audit_retention(bundle, {key: reference}, tasks, manifest)


if __name__ == "__main__":
    unittest.main()
