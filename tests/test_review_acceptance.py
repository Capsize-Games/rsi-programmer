"""Focused regressions for acceptance and promotion review findings."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from rsi_programmer.autonomy import _dist, agency_profile
from rsi_programmer.autonomy import autonomy_ratio
from rsi_programmer.edits import DEFAULT_CONFIG
from rsi_programmer.loop_trainer import ACTIVE_FIELDS, EPOCHS, HELD
from rsi_programmer.loop_trainer import SHADOW, _solves, train_scores
from rsi_programmer.promote import PromotionGate
from rsi_programmer.self_gate import SelfGate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "scripts"))
from rsi_loop import RsiLoop


class AcceptanceReviewTests(unittest.TestCase):
    @staticmethod
    def _observe_default_trainer(
            config: dict[str, Any]) -> dict[str, Any]:
        """Capture the trainer's config-dependent call arguments."""
        groups = [[([0.0], 1)]]
        with patch("rsi_programmer.loop_trainer.SearchPolicy") as factory:
            with patch("rsi_programmer.loop_trainer.refine_operands") \
                    as refine:
                with patch("rsi_programmer.loop_trainer._solves",
                           return_value=[0.0]) as solves:
                    policy = factory.return_value
                    policy.trace_groups.return_value = groups
                    scores = train_scores(config, 7, "unused")
        return {"factory": factory, "policy": policy,
                "refine": refine, "solves": solves,
                "groups": groups, "scores": scores}

    def test_gate_rejects_inadequate_and_nonfinite_evidence(self) -> None:
        gate = PromotionGate()
        with self.assertRaisesRegex(ValueError, "3 paired"):
            gate.decide([0.0], [1.0], 0, 0, 0.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            gate.decide([0.0, 0.0, 0.0],
                        [0.1, math.nan, 0.1], 0, 0, 0.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            gate.decide([0.0] * 3, [0.1, "invalid", 0.1], 0, 0,
                        0.0)
        with self.assertRaisesRegex(ValueError, "spend"):
            gate.decide([0.0] * 3, [0.1] * 3, 0, 0, math.inf)
        with self.assertRaisesRegex(ValueError, "spend"):
            gate.decide([0.0] * 3, [0.1] * 3, 0, "invalid", 0.0)
        with self.assertRaisesRegex(ValueError, "safety"):
            gate.decide([0.0] * 3, [0.1] * 3, "invalid", 0, 0.0)
        self.assertTrue(PromotionGate(min_pairs=2).decide(
            [0.0, 0.0], [0.1, 0.1], 0, 0, 0.0)["promote"])
        with self.assertRaises(ValueError):
            PromotionGate(min_pairs=0)

    def test_self_gate_records_nonfinite_refusal_and_finite_gain(
            self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gate = SelfGate(tmp + "/fence", tmp + "/ledger.db",
                            budget=4)
            base = dict(DEFAULT_CONFIG)
            edit = {"op": "set", "field": "lr", "value": 0.1}

            def nonfinite(config: dict[str, Any], seed: int,
                          workdir: str) -> dict[str, Any]:
                values = ([math.nan] * 3 if config["lr"] == 0.1
                          else [0.0] * 3)
                return {"held": values}

            try:
                rejected = gate.propose(base, edit, nonfinite)
                self.assertEqual(rejected["decision"], "refuse")
                rows = gate._log.commands_in("self-gate", "prepared")
                self.assertEqual(rows[-1]["digest"], rejected["lineage"])
                self.assertIn("finite", rejected["reason"])

                def no_gain(config: dict[str, Any], seed: int,
                            workdir: str) -> dict[str, Any]:
                    return {"held": [0.0, 0.0, 0.0]}

                refused = gate.propose(base, edit, no_gain)
                self.assertEqual(refused["decision"], "refuse")
                self.assertEqual(refused["mean_gain"], 0.0)

                def nonnumeric(config: dict[str, Any], seed: int,
                               workdir: str) -> dict[str, Any]:
                    values = (["invalid"] * 3
                              if config["lr"] == 0.1 else [0.0] * 3)
                    return {"held": values}

                invalid = gate.propose(base, edit, nonnumeric)
                self.assertEqual(invalid["decision"], "refuse")
                self.assertIn("finite", invalid["reason"])
                rows = gate._log.commands_in("self-gate", "prepared")
                self.assertEqual(rows[-1]["digest"], invalid["lineage"])

                def missing_scores(config: dict[str, Any], seed: int,
                                   workdir: str) -> dict[str, Any]:
                    return {"held": None}

                missing = gate.propose(base, edit, missing_scores)
                self.assertEqual(missing["decision"], "refuse")
                self.assertTrue(missing["lineage"])
                self.assertTrue(missing["reason"])
                rows = gate._log.commands_in("self-gate", "prepared")
                self.assertEqual(rows[-1]["digest"], missing["lineage"])
            finally:
                gate.close()

    def test_default_scores_require_held_out_oracle(
            self) -> None:
        task = {"id": "one", "train": [[2, 0, 2]],
                "held_out": [[3, 0, 3]], "max_tree_ops": 1}
        graph = {"input_types": {"x": "Int", "y": "Int"},
                 "nodes": [], "output": "x"}
        rejected = {"input_types": {"x": "Int", "y": "Int"},
                    "nodes": [], "output": "y"}
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "family.json"
            fixture.write_text(json.dumps({"tasks": [task]}),
                               encoding="utf-8")
            def search(*args: Any,
                       **kwargs: Any) -> dict[str, Any]:
                self.assertFalse(kwargs["oracle"](rejected))
                self.assertTrue(kwargs["oracle"](graph))
                return {"solved": True, "tried": 1, "graph": graph}

            with patch("rsi_programmer.loop_trainer.guided_search",
                       side_effect=search):
                self.assertGreater(_solves(object(), str(fixture))[0],
                                   0.0)

    def test_summary_marks_singletons_and_checks_dimensions(self) -> None:
        self.assertEqual(autonomy_ratio({"one": [1.0],
                                         "many": [0.0, 0.0]})
                         ["singleton_groups"], 1)
        rows = agency_profile([{"generation": 0, "weights": [1.0],
                                "fitness": {"correctness": 1.0}}])
        self.assertTrue(rows[0]["singleton"])
        self.assertFalse(rows[0]["informative"])
        self.assertEqual(rows[0]["direction"], "non-informative")
        with self.assertRaisesRegex(ValueError, "equal dimensions"):
            _dist([0.0], [0.0, 1.0])

    def test_loop_excludes_unmeasured_gains_from_summaries(self) -> None:
        cases = (("nonfinite", [math.nan, math.nan, math.nan]),
                 ("insufficient", [0.0, 0.0]),
                 ("missing", None))
        for name, held in cases:
            with self.subTest(case=name):
                with tempfile.TemporaryDirectory() as tmp:
                    loop = RsiLoop(
                        tmp + "/fence", tmp + "/gate.db",
                        tmp + "/trace.jsonl", cycles=1,
                        active_fields=("lr",))

                    def trainer(config: dict[str, Any], seed: int,
                                workdir: str) -> dict[str, Any]:
                        return {"held": held}

                    try:
                        report = loop.run(dict(DEFAULT_CONFIG), trainer)
                    finally:
                        loop.close()
                record = report["records"][0]
                self.assertEqual(record["decision"], "refuse")
                self.assertNotIn("mean_gain", record)
                self.assertEqual(
                    report["outcome_association"]["samples"], 0)
                self.assertEqual(report["agency"], [])

    def test_loop_preserves_finite_negative_refusal_gain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = RsiLoop(tmp + "/fence", tmp + "/gate.db",
                           tmp + "/trace.jsonl", cycles=1,
                           active_fields=("lr",))

            def trainer(config: dict[str, Any], seed: int,
                        workdir: str) -> dict[str, Any]:
                level = (1.0 if config["lr"] == DEFAULT_CONFIG["lr"]
                         else 0.0)
                return {"held": [level] * 3}

            try:
                report = loop.run(dict(DEFAULT_CONFIG), trainer)
            finally:
                loop.close()
        record = report["records"][0]
        self.assertEqual(record["decision"], "refuse")
        self.assertEqual(record["mean_gain"], -1.0)
        self.assertEqual(report["outcome_association"]["samples"], 1)
        self.assertEqual(report["agency"][0]["members"], 1)
        self.assertFalse(report["agency"][0]["informative"])

    def test_default_active_fields_are_trainer_surface_only(self) -> None:
        self.assertEqual(set(ACTIVE_FIELDS),
                         {"hidden", "steps", "lr", "joint_lambda",
                          "joint_epochs"})
        self.assertIn("beta", DEFAULT_CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            loop = RsiLoop(tmp + "/fence", tmp + "/gate.db",
                           tmp + "/trace.jsonl", cycles=1,
                           active_fields=ACTIVE_FIELDS)
            try:
                pool, _ = loop._pool(dict(DEFAULT_CONFIG))
                self.assertTrue(pool)
                self.assertTrue(all(edit["field"] in ACTIVE_FIELDS
                                    for edit in pool))
            finally:
                loop.close()

    def test_default_trainer_consumes_active_fields(self) -> None:
        changed = {"hidden": 8, "steps": 4, "lr": 0.01,
                   "joint_lambda": 0.5, "joint_epochs": 10}
        for field in ACTIVE_FIELDS:
            with self.subTest(field=field):
                config = dict(DEFAULT_CONFIG)
                config[field] = changed[field]
                observed = self._observe_default_trainer(config)
                factory = observed["factory"]
                policy = observed["policy"]
                constructor = factory.call_args.kwargs
                self.assertEqual(constructor["seed"], 7)
                self.assertEqual(constructor["hidden"], config["hidden"])
                self.assertEqual(constructor["steps"], config["steps"])
                self.assertEqual(constructor["lr"], config["lr"])
                if field in ("hidden", "steps", "lr"):
                    self.assertNotEqual(constructor[field],
                                        DEFAULT_CONFIG[field])
                policy.fit.assert_called_once_with(
                    observed["groups"], EPOCHS)
                joint = observed["refine"].call_args
                self.assertEqual(joint.args,
                                 (policy, observed["groups"], SHADOW))
                self.assertEqual(joint.kwargs["lam"],
                                 config["joint_lambda"])
                self.assertEqual(joint.kwargs["epochs"],
                                 config["joint_epochs"])
                if field in ("joint_lambda", "joint_epochs"):
                    argument = "lam" if field == "joint_lambda" \
                        else "epochs"
                    self.assertNotEqual(joint.kwargs[argument],
                                        DEFAULT_CONFIG[field])
                self.assertEqual(
                    [call.args[1]
                     for call in observed["solves"].call_args_list],
                    [SHADOW, HELD])
                self.assertEqual(observed["scores"],
                                 {"train": [0.0], "held": [0.0]})

    def test_default_trainer_ignores_unsupported_fields(self) -> None:
        base = self._observe_default_trainer(dict(DEFAULT_CONFIG))
        changed = dict(DEFAULT_CONFIG)
        changed.update({"beta": 0.8, "margin": 2.0, "pop": 8,
                        "sigma": 0.2, "generations": 5,
                        "evolve_budget_s": 480})
        other = self._observe_default_trainer(changed)
        self.assertEqual(base["factory"].call_args,
                         other["factory"].call_args)
        self.assertEqual(base["policy"].fit.call_args.args[1:],
                         other["policy"].fit.call_args.args[1:])
        self.assertEqual(base["refine"].call_args.args[1:],
                         other["refine"].call_args.args[1:])
        self.assertEqual(base["refine"].call_args.kwargs,
                         other["refine"].call_args.kwargs)
        self.assertEqual(
            [call.args[1]
             for call in base["solves"].call_args_list],
            [call.args[1]
             for call in other["solves"].call_args_list])
        self.assertEqual(base["scores"], other["scores"])


if __name__ == "__main__":
    unittest.main()
