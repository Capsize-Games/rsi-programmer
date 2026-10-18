"""Contracts for the frozen plastic-encoder experiment harness."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from rsi_programmer import online_benchmark as base
from rsi_programmer.online_plastic_benchmark import (
    ARMS,
    BUDGETS,
    NEURAL_ARMS,
    PROTOCOL_SHA256,
    load_protocol,
    policy_for,
    probe_retention,
)
from rsi_programmer.online_plastic_summary import (
    summarize,
    summarize_retention,
)


class TestOnlinePlasticBenchmark(unittest.TestCase):
    """Check protocol pins, matched arm state, probes, and gate accounting."""

    def test_protocol_is_pinned_to_existing_inputs_and_matrix(self) -> None:
        """The new protocol must pin the original stream and full matrix."""
        protocol = load_protocol()
        self.assertEqual(protocol["source_corpus"]["sha256"],
                         base.CORPUS_SHA256)
        expected_schedule = (
            "0ffbd7eadac93aa63b8df68d9ba250cd4da5a55a178e8339f9505d81638af3f1")
        self.assertEqual(protocol["source_schedule"]["sha256"],
                         expected_schedule)
        self.assertEqual(tuple(protocol["arms"]), ARMS)
        self.assertEqual(tuple(protocol["budgets"]), BUDGETS)
        self.assertEqual(protocol["matrix"]["episode_rows"], 13824)
        self.assertEqual(len(PROTOCOL_SHA256), 64)

    def test_protocol_digest_rejects_changed_bytes(self) -> None:
        """Any protocol edit requires a new explicit freeze."""
        source = Path(__file__).resolve().parents[1] / "fixtures"
        payload = (source / "online_plastic_encoder_v1.json").read_text()
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "protocol.json"
            changed.write_text(payload + " ")
            with self.assertRaisesRegex(ValueError, "protocol hash"):
                load_protocol(changed)

    def test_all_neural_arms_start_from_matched_projection_and_readout(
            self) -> None:
        """Arm flags must not alter the seeded initial model or RNG state."""
        policies = [policy_for(arm, 1) for arm in NEURAL_ARMS]
        self.assertTrue(all(policy is not None for policy in policies))
        snapshots = [policy.snapshot() for policy in policies if policy]
        self.assertEqual(len({item["encoder_hash"] for item in snapshots}), 1)
        self.assertEqual(len({item["readout_hash"] for item in snapshots}), 1)
        self.assertEqual(len({json.dumps(item["rng_state"])
                              for item in snapshots}), 1)
        self.assertTrue(policy_for("grammar_streaming", 1) is None)

    def test_retention_probe_covers_canonical_contexts_without_mutation(
            self) -> None:
        """Read-only cold probes cover each task and preserve full source."""
        tasks = base.load_corpus()["tasks"]
        policy = policy_for("plastic_snn", 0)
        assert policy is not None
        before = policy.snapshot()
        row = probe_retention(
            policy, tasks, order_id="order-test",
            phase_sequence=[2, 0, 1], budget=144, seed=0,
            arm="plastic_snn", completed_tasks=0,
        )
        after = policy.snapshot()
        self.assertEqual(before, after)
        self.assertEqual(row["source_state_sha256_before"],
                         row["source_state_sha256_after"])
        self.assertEqual(row["probe_count"], 48)
        self.assertEqual([item["task_id"] for item in row["contexts"]],
                         [task["id"] for task in tasks])
        self.assertTrue(all(len(item["rates"]) == 16
                            and len(item["probabilities"]) == 3
                            for item in row["contexts"]))
        self.assertEqual(row["explicit_probe_work"]["evaluation_copies"], 48)

    @staticmethod
    def _synthetic_rows() -> tuple[list[dict[str, Any]],
                                   dict[str, float], dict[str, float]]:
        """Build one complete synthetic order for summary-contract tests."""
        rows: list[dict[str, Any]] = []
        setup: dict[str, float] = {}
        probes: dict[str, float] = {}
        order_id = "order-0-1-2"
        accepted = {
            "persistent_snn": 120, "frozen_snn": 100,
            "episode_reset_snn": 100, "online_nonspiking": 110,
            "grammar_streaming": 120, "plastic_snn": 130,
            "reset_plastic_snn": 125, "plastic_nonspiking": 110,
        }
        first = {
            "persistent_snn": 100, "frozen_snn": 80,
            "episode_reset_snn": 80, "online_nonspiking": 80,
            "grammar_streaming": 80, "plastic_snn": 120,
            "reset_plastic_snn": 80, "plastic_nonspiking": 80,
        }
        for budget in BUDGETS:
            for arm in ARMS:
                for seed in (0, 1, 2):
                    key = f"{order_id}:{budget}:{arm}:{seed}"
                    setup[key] = 0.01
                    probes[key] = 0.02 if arm != "grammar_streaming" else 0.0
                    arm_accept = accepted[arm]
                    arm_first = first[arm]
                    if budget == 432:
                        arm_accept = 144
                    for index in range(48):
                        rows.append({
                            "order_id": order_id, "budget": budget,
                            "arm": arm, "seed": seed,
                            "accepted": index < arm_accept // 3,
                            "first_lane_correct": index < arm_first // 3,
                            "candidate_evaluations": 4,
                            "policy_work": {"updates": 2,
                                            "plastic_encoder_updates": 1},
                            "context_work": {"projection_mac": 288},
                            "node_row_operations": 12,
                            "interpreter_rows": 4,
                            "reset_cpu_seconds": 0.0,
                            "context_cpu_seconds": 0.0,
                            "episode_cpu_seconds": 0.01,
                            "checker_cpu_seconds": 0.001,
                            "state_hash_cpu_seconds": 0.001,
                        })
        return rows, setup, probes

    def test_summary_charges_probes_and_enforces_each_frozen_gate(
            self) -> None:
        """Probe cost belongs in total CPU; gate denominators are per order."""
        rows, setup, probes = self._synthetic_rows()
        orders = [{"order_id": "order-0-1-2",
                   "phase_sequence": [0, 1, 2]}]
        summary = summarize(rows, setup, probes, orders)
        order = summary["per_order"][0]
        self.assertTrue(order["matrix_complete"])
        self.assertTrue(order["gates"]["plastic_snn_vs_persistent_snn"])
        self.assertTrue(order["gates"]["plastic_snn_vs_plastic_nonspiking"])
        plastic = next(item for item in order["per_seed"]
                       if item["budget"] == 144
                       and item["arm"] == "plastic_snn"
                       and item["seed"] == 0)
        self.assertAlmostEqual(
            plastic["total_cpu_seconds"],
            plastic["search_only_cpu_seconds"] + 0.02,
        )
        self.assertAlmostEqual(
            plastic["search_only_cpu_seconds"], 0.01 + 48 * 0.012,
        )

    def test_partial_matrix_cannot_pass_gates(self) -> None:
        """A sparse result set cannot satisfy any predeclared comparison."""
        row = {
            "order_id": "order-0-1-2", "budget": 144,
            "arm": "plastic_snn", "seed": 0,
            "accepted": True, "first_lane_correct": True,
            "candidate_evaluations": 1, "policy_work": {},
            "context_work": {}, "node_row_operations": 0,
            "interpreter_rows": 0,
            "reset_cpu_seconds": 0.0, "context_cpu_seconds": 0.0,
            "episode_cpu_seconds": 0.0, "checker_cpu_seconds": 0.0,
            "state_hash_cpu_seconds": 0.0,
        }
        summary = summarize(
            [row], {}, {},
            [{"order_id": "order-0-1-2", "phase_sequence": [0, 1, 2]}],
        )
        result = summary["per_order"][0]
        self.assertFalse(result["matrix_complete"])
        self.assertFalse(any(result["gates"].values()))

    def test_retention_summary_pairs_phase_end_with_final_same_contexts(
            self) -> None:
        """Retention compares each phase's contexts at two boundaries."""
        tasks = base.load_corpus()["tasks"]
        rows = []
        for arm in NEURAL_ARMS:
            for budget in BUDGETS:
                for seed in (0, 1, 2):
                    for boundary in (0, 16, 32, 48):
                        probability = boundary / 100.0
                        rows.append({
                            "order_id": "order-0-1-2", "budget": budget,
                            "seed": seed, "arm": arm,
                            "completed_tasks": boundary,
                            "contexts": [{
                                "task_id": task["id"],
                                "phase": task["phase"],
                                "root_probability": probability,
                            } for task in tasks],
                        })
        result = summarize_retention(
            rows, [{"order_id": "order-0-1-2",
                    "phase_sequence": [0, 1, 2]}])
        self.assertEqual(len(result), len(NEURAL_ARMS) * 2 * 3 * 3)
        sample = next(item for item in result
                      if item["arm"] == "plastic_snn"
                      and item["phase"] == 1)
        self.assertEqual(sample["tasks"], 16)
        self.assertAlmostEqual(sample["mean_phase_end_root_probability"],
                               0.32)
        self.assertAlmostEqual(sample["mean_final_root_probability"], 0.48)


if __name__ == "__main__":
    unittest.main()
