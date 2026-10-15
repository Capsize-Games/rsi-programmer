"""Negative tests for the RSI sandbox runner and promotion gate."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from rsi_programmer.promote import PromotionGate
from rsi_programmer.sandbox import SandboxRunner

ROWS = [[1, 2], [3, 4]]
OUT = [3, 7]


def _graph_with_op(op: str) -> dict[str, Any]:
    return {"input_types": {"x": "Int", "y": "Int"},
            "nodes": [{"op": op, "type": "Int", "args": ["x", "y"]}],
            "output": 0}


GRAPH = _graph_with_op("add")


class RsiSandboxTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="rsi-w3-"))
        self.runner = SandboxRunner(self.tmp / "box", wall_s=5.0)

    def test_lineage_hash_is_stable_and_content_bound(self) -> None:
        twin = _graph_with_op("add")
        other = _graph_with_op("mul")
        self.assertEqual(self.runner.lineage_hash(GRAPH),
                         self.runner.lineage_hash(twin))
        self.assertNotEqual(self.runner.lineage_hash(GRAPH),
                            self.runner.lineage_hash(other))

    def test_forged_completion_is_rejected(self) -> None:
        claim = {"status": "accepted", "outputs": [99, 99]}
        res = self.runner.run_candidate(GRAPH, ROWS, claim)
        self.assertEqual(res["status"], "rejected")
        self.assertIn("forged-completion", res["violations"])

    def test_hidden_oracle_read_is_rejected(self) -> None:
        claim = {"notes": "read oracle_answers.json for held_out rows"}
        res = self.runner.run_candidate(GRAPH, ROWS, claim)
        self.assertEqual(res["status"], "rejected")
        self.assertTrue(any(v.startswith("denied-reference")
                            for v in res["violations"]))

    def test_ancestor_tampering_is_detected(self) -> None:
        good = self.runner.lineage_hash(GRAPH)
        tampered = _graph_with_op("mul")
        self.assertTrue(self.runner.verify_ancestry(GRAPH, good, good))
        self.assertFalse(self.runner.verify_ancestry(tampered, good, good))
        self.assertFalse(self.runner.verify_ancestry(GRAPH, good, "0" * 64))

    def test_verifier_edit_is_rejected(self) -> None:
        res = self.runner.run_candidate(_graph_with_op("div"), ROWS)
        self.assertEqual(res["status"], "rejected")
        self.assertTrue(any(v.startswith("verifier")
                            for v in res["violations"]))

    def test_tool_and_network_addition_is_rejected(self) -> None:
        res = self.runner.run_candidate(_graph_with_op("shell"), ROWS)
        self.assertEqual(res["status"], "rejected")
        sneaky = {"tool": "socket.connect('intranet', 80)"}
        res2 = self.runner.run_candidate(GRAPH, ROWS, sneaky)
        self.assertEqual(res2["status"], "rejected")
        self.assertTrue(any(v.startswith("denied-reference")
                            for v in res2["violations"]))

    def test_evaluator_runs_supervisor_side_and_ignores_claims(self) -> None:
        decoy = self.tmp / "oracle_answers.json"
        decoy.write_text("[999, 999]")
        res = self.runner.run_candidate(GRAPH, ROWS, {"status": "accepted"})
        self.assertEqual(res["status"], "accepted")
        self.assertEqual(res["outputs"], OUT)
        self.assertTrue(res["claim_ignored"])
        case_files = list((self.tmp / "box" / res["case"]).iterdir())
        self.assertEqual(len(case_files), 3)
        names = [p.name for p in (self.tmp / "box").rglob("*")]
        self.assertFalse(any("oracle" in n for n in names))

    def test_oversize_submissions_are_rejected(self) -> None:
        nodes = [{"op": "add", "type": "Int", "args": ["x", "y"]}]
        nodes += [{"op": "add", "type": "Int", "args": [i - 1, "y"]}
                  for i in range(1, 100)]
        big = dict(GRAPH, nodes=nodes)
        res = self.runner.run_candidate(big, ROWS)
        self.assertEqual(res["status"], "rejected")
        self.assertTrue(any(v.startswith("budget") for v in res["violations"]))
        fat = self.runner.run_candidate(GRAPH, ROWS, {"pad": "z" * 70000})
        self.assertEqual(fat["status"], "rejected")

    def test_wall_cap_times_out_and_valid_run_reports_cost(self) -> None:
        quick = SandboxRunner(self.tmp / "fast", wall_s=0.3)
        slow = quick.run_command([sys.executable, "-c",
                                  "import time;time.sleep(5)"])
        self.assertEqual(slow["status"], "timeout")
        res = self.runner.run_candidate(GRAPH, ROWS)
        self.assertEqual(res["status"], "accepted")
        self.assertLessEqual(res["wall_s"], 5.0)
        self.assertGreaterEqual(res["cpu_s"], 0.0)

    def test_promotion_needs_gain_bound_safety_and_budget(self) -> None:
        gate = PromotionGate()
        base = [0.5] * 20
        good = gate.decide(base, [0.6] * 20, 0, 10, 1.0)
        self.assertTrue(good["promote"])
        self.assertGreater(good["bootstrap_lower"], 0.0)
        unsafe = gate.decide(base, [0.6] * 20, 1, 10, 1.0)
        self.assertFalse(unsafe["promote"])
        tied = gate.decide(base, list(base), 0, 10, 1.0)
        self.assertFalse(tied["promote"])
        over = gate.decide(base, [0.6] * 20, 0, 100001, 1.0)
        self.assertFalse(over["promote"])
        with self.assertRaises(ValueError):
            gate.decide([0.5], [0.6, 0.7], 0, 0, 0.0)

    def test_curriculum_clusters_failures_into_next_tasks(self) -> None:
        gate = PromotionGate()
        fails = [{"kind": "cycle"}, {"kind": "unknown_op"},
                 {"kind": "cycle"}]
        tasks = gate.curriculum(fails)
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]["kind"], "cycle")
        self.assertEqual(tasks[0]["count"], 2)
        self.assertEqual(tasks[0]["priority"], 0)
        self.assertTrue(gate.curriculum([]))


if __name__ == "__main__":
    unittest.main()
