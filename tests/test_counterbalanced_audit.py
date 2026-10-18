"""Focused controls for the counterbalanced matrix auditor."""
from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

from rsi_programmer.online_counterbalanced import summarize
from rsi_programmer.online_graph import (
    check_result,
    graph_polynomial,
    iter_candidates,
    run_episode,
)
from rsi_programmer.online_spiking import OnlineSpikingPolicy
from rsi_programmer.probe import interpret
from scripts.audit_online_counterbalanced import (
    SCHEDULE,
    _diagnostic,
    _audit_summary,
    _validate_counterbalanced_row,
    _validate_schedule,
)


def _streaming_row() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Construct one exact grammar-streaming task without a matrix run."""
    graph = next(iter_candidates("add"))
    observed = ((-2, 3, interpret(graph, -2, 3)),
                (4, -1, interpret(graph, 4, -1)))
    held = ((1, 5, interpret(graph, 1, 5)),)
    polynomial = graph_polynomial(graph)
    task = {"id": "counterbalance-audit", "phase": 0, "root": "add",
            "observed": [list(row) for row in observed],
            "held": [list(row) for row in held],
            "polynomial": [list(term) for term in polynomial]}
    out = run_episode(None, observed, budget=1, episode_id=task["id"])
    verdict = check_result(out["graph"], held, polynomial)
    order = {"order_id": "0-1-2", "phase_sequence": [0, 1, 2]}
    row = {
        **out, **verdict,
        "order_id": order["order_id"],
        "phase_sequence": order["phase_sequence"],
        "stream_index": 0, "phase_position": 0, "phase_ordinal": 0,
        "arm": "grammar_streaming", "seed": 0, "budget": 1,
        "task_id": task["id"], "phase": 0, "root": "add",
        "first_lane_correct": True, "policy_work": {},
        "weight_before": None, "weight_after": None,
        "rng_draw_offset_before": 0, "rng_draw_offset_after": 0,
        "diagnostic_before": _diagnostic(None),
        "diagnostic_cpu_seconds": 0.0, "reset_cpu_seconds": 0.0,
        "reward_sum": out["attempts"][0]["reward"],
        "mean_reward": out["attempts"][0]["reward"],
        "checker_cpu_seconds": 0.0,
    }
    return row, order, task


class CounterbalancedAuditTests(unittest.TestCase):
    """Reject schedule, state-diagnostic, and draw-offset corruption."""

    def test_frozen_schedule_has_six_original_task_permutations(self) -> None:
        protocol = json.loads(SCHEDULE.read_text())["protocol"]
        corpus = json.loads((SCHEDULE.parents[0]
                             / "online_graph_v1.json").read_text())
        orders = _validate_schedule(protocol, corpus["tasks"])
        self.assertEqual(len(orders), 6)
        bad = copy.deepcopy(protocol)
        bad["orders"][1]["task_ids"][0] = "missing-task"
        with self.assertRaisesRegex(ValueError, "phase task order"):
            _validate_schedule(bad, corpus["tasks"])

    def test_diagnostic_is_pure_and_matches_probe_helper(self) -> None:
        from rsi_programmer.online_diagnostics import inspect_policy

        policy = OnlineSpikingPolicy(seed=5)
        state = (policy.weight_hash(), policy._rng.getstate(),
                 policy._membrane[:], policy._baseline)
        expected = inspect_policy(policy)
        self.assertEqual(_diagnostic(policy), expected)
        self.assertEqual((policy.weight_hash(), policy._rng.getstate(),
                          policy._membrane[:], policy._baseline), state)

    def test_accepts_valid_stream_row_and_rejects_corruption(self) -> None:
        row, order, task = _streaming_row()
        self.assertEqual(
            _validate_counterbalanced_row(row, order, task, 0, 1, 0, None),
            0,
        )
        changes = (
            lambda data: data.__setitem__("phase_position", 1),
            lambda data: data.__setitem__("rng_draw_offset_before", 1),
            lambda data: data["diagnostic_before"].__setitem__(
                "entropy_nats", 1.0),
            lambda data: data.__setitem__("diagnostic_cpu_seconds", -1.0),
            lambda data: data.__setitem__("mean_reward", 0.25),
        )
        for mutate in changes:
            with self.subTest(mutation=mutate):
                corrupt = copy.deepcopy(row)
                mutate(corrupt)
                with self.assertRaises(ValueError):
                    _validate_counterbalanced_row(
                        corrupt, order, task, 0, 1, 0, None)

    def test_diagnostic_replay_agrees_for_learning_policy(self) -> None:
        from rsi_programmer.online_diagnostics import inspect_policy

        policy = OnlineSpikingPolicy(seed=12)
        policy.start_episode("diagnostic-purity")
        policy.choose((1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                      ("add", "sub", "mul"))
        policy.observe(0.5)
        policy.end_episode()
        self.assertEqual(_diagnostic(policy), inspect_policy(policy))

    def test_summary_recomputation_covers_all_orders_and_cpu(self) -> None:
        """Exercise summary audit, including phase strata and all gates."""
        corpus = json.loads((SCHEDULE.parent / "online_graph_v1.json")
                            .read_text())
        protocol = json.loads(SCHEDULE.read_text())["protocol"]
        orders = _validate_schedule(protocol, corpus["tasks"])
        tasks = {task["id"]: task for task in corpus["tasks"]}
        rates = {"persistent_snn": (10, 9), "frozen_snn": (5, 6),
                 "episode_reset_snn": (5, 6),
                 "online_nonspiking": (5, 9),
                 "grammar_streaming": (6, 7)}
        episode_cpu = {"persistent_snn": 0.01, "frozen_snn": 0.012,
                       "episode_reset_snn": 0.012,
                       "online_nonspiking": 0.02,
                       "grammar_streaming": 0.003}
        rows: list[dict[str, Any]] = []
        setup: dict[str, float] = {}
        for order in orders:
            for budget in (144, 432):
                for seed in (0, 1, 2):
                    for arm in rates:
                        group = f"{order['order_id']}:{budget}:{arm}:{seed}"
                        setup[group] = 0.0001
                        for index, task_id in enumerate(order["task_ids"]):
                            task = tasks[task_id]
                            phase_ordinal = index % 16
                            first_cut, accept_cut = rates[arm]
                            if arm == "online_nonspiking" and seed == 0:
                                first_cut = 11
                            candidate_count = 2
                            rows.append({
                                "order_id": order["order_id"],
                                "budget": budget, "seed": seed, "arm": arm,
                                "task_id": task_id, "phase": task["phase"],
                                "phase_position": index // 16,
                                "accepted": phase_ordinal < accept_cut,
                                "observed_solved": phase_ordinal < accept_cut,
                                "first_lane_correct": (
                                    phase_ordinal < first_cut),
                                "candidate_evaluations": candidate_count,
                                "node_row_operations": 1,
                                "episode_cpu_seconds": episode_cpu[arm],
                                "checker_cpu_seconds": 0.0001,
                                "reset_cpu_seconds": 0.0001,
                                "diagnostic_cpu_seconds": 0.001,
                                "policy_cpu_seconds": 0.001,
                                "update_cpu_seconds": 0.001,
                                "policy_work": {"updates": 0},
                                "diagnostic_before": {
                                    "probabilities": [1 / 3] * 3,
                                    "entropy_nats": math.log(3),
                                },
                                "reward_sum": 0.6, "mean_reward": 0.3,
                            })
        summary = summarize(rows, setup, orders)
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            (bundle / "summary.json").write_text(
                json.dumps(summary, sort_keys=True))
            audited = _audit_summary(
                bundle, {"setup_cpu_seconds": setup,
                         "raw_sha256": "synthetic"}, rows, orders,
                protocol)
        self.assertEqual(audited["episode_rows"], 6 * 2 * 3 * 5 * 48)
        self.assertFalse(audited["diagnosis"]["pooled_regression"])
        self.assertTrue(summary["all_order_gates"]["adaptation"])
        self.assertTrue(summary["all_order_gates"]["snn_specific"])
        self.assertFalse(summary["all_order_gates"]["streaming_deployment"])
        group = next(item for item in summary["per_order"][0]["per_seed"]
                     if item["budget"] == 144
                     and item["arm"] == "persistent_snn"
                     and item["seed"] == 0)
        self.assertAlmostEqual(group["total_cpu_seconds"],
                               0.0001 + 48 * (0.01 + 0.0001
                                              + 0.0001 + 0.001))


if __name__ == "__main__":
    unittest.main()
