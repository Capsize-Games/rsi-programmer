"""Frozen LLM bank: counts, rows, exclusions, pins and seal guard."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from rsi_programmer import statement_source, statement_syntax
from rsi_programmer.benchmark_v2_corpus import polynomial
from rsi_programmer.llm_bank import (
    HELD_BOUNDARY_N,
    HELD_N,
    INT64_LIMIT,
    QUERY_N,
    VISIBLE_N,
    LlmBankGenerator,
)

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "fixtures" / "llm_bank_v1"
FAMILIES = ("A6", "A7", "A8", "S1", "S2", "S3")
PER_FAMILY = 15


def _identities(tasks: list[dict[str, Any]]) -> set[str]:
    """Collect bank identity keys from task specs."""
    return {task["identity_key"] for task in tasks}


def _tree_ops(node: dict[str, Any]) -> int:
    """Count operation nodes in an arithmetic witness."""
    if node["op"] == "input":
        return 0
    return 1 + _tree_ops(node["left"]) + _tree_ops(node["right"])


class LlmBankTests(unittest.TestCase):
    """Validate the frozen bank without opening sealed choices."""

    @classmethod
    def setUpClass(cls) -> None:
        gen = LlmBankGenerator()
        cls.gen = gen
        cls.protocol, cls.tasks, cls.visible, cls.query = \
            gen.load_public()
        cls.held = gen.load_held(unseal=True)
        cls.by_id = {task["id"]: task for task in cls.tasks}

    def test_counts_and_split(self) -> None:
        self.assertEqual(len(self.tasks), 180)
        for pool in ("dev", "sealed"):
            members = [t for t in self.tasks if t["pool"] == pool]
            self.assertEqual(len(members), 90)
            for family in FAMILIES:
                count = sum(1 for t in members
                            if t["family"] == family)
                self.assertEqual(count, PER_FAMILY)

    def test_identity_disjoint_pools(self) -> None:
        dev = _identities([t for t in self.tasks
                           if t["pool"] == "dev"])
        sealed = _identities([t for t in self.tasks
                              if t["pool"] == "sealed"])
        self.assertEqual(len(dev), 90)
        self.assertEqual(len(sealed), 90)
        self.assertFalse(dev & sealed)
        self.assertEqual(len(_identities(self.tasks)), 180)

    def test_row_discipline(self) -> None:
        for task in self.tasks:
            ident = task["id"]
            vis = self.visible[ident]
            qry = self.query[ident]
            hld = self.held[ident]
            self.assertEqual(len(vis), VISIBLE_N)
            self.assertEqual(len(qry), QUERY_N)
            self.assertEqual(len(hld), HELD_N)
            if task["interface"] == "pair":
                ins = [tuple(r[:2]) for rows in (vis, qry, hld)
                       for r in rows]
            else:
                ins = [tuple(r[0]) for rows in (vis, qry, hld)
                       for r in rows]
            self.assertEqual(len(set(ins)), len(ins))
            for rows in (vis, qry, hld):
                for row in rows:
                    self.assertLessEqual(abs(row[-1]),
                                         INT64_LIMIT)

    def test_held_boundary_structure(self) -> None:
        for task in self.tasks:
            hld = self.held[task["id"]]
            head = hld[:HELD_BOUNDARY_N]
            self.assertEqual(len(hld), HELD_N)
            if task["interface"] == "pair":
                self.assertIn([0, 0, head[0][-1]], head)
                self.assertEqual(len(head), HELD_BOUNDARY_N)
            else:
                inputs = [r[0] for r in head]
                self.assertIn([], inputs)
                self.assertTrue(any(len(v) == 256
                                    for v in inputs))

    def test_arithmetic_witnesses_execute(self) -> None:
        for task in self.tasks:
            if task["interface"] != "pair":
                continue
            node = task["witness"]["node"]
            want = int(task["family"][1])
            self.assertEqual(_tree_ops(node), want)
            poly = polynomial(node)
            recorded = tuple(tuple(t) for t in
                             task["witness"]["polynomial"])
            self.assertEqual(poly, recorded)
            for rows in (self.visible[task["id"]],
                         self.query[task["id"]],
                         self.held[task["id"]]):
                for x, y, expected in rows:
                    value = sum(c * x ** xp * y ** yp
                                for xp, yp, c in poly)
                    self.assertEqual(value, expected)

    def test_statement_witnesses_execute_python(self) -> None:
        for task in self.tasks:
            if task["interface"] != "list":
                continue
            tokens = tuple(task["witness"]["tokens"])
            statement_syntax.parse(tokens)
            source = statement_source.python_source(tokens)
            self.assertIn("def process", source)
            curlee = statement_source.curlee_source(tokens)
            self.assertIn("fn process", curlee)
            namespace: dict[str, Any] = {}
            exec(compile(source, task["id"], "exec"),
                 namespace)
            for rows in (self.visible[task["id"]],
                         self.query[task["id"]],
                         self.held[task["id"]]):
                for items, expected in rows:
                    self.assertEqual(namespace["process"](
                        list(items)), expected)

    def test_exclusion_disjointness(self) -> None:
        keys, _ = self.gen.load_exclusions()
        bank = _identities(self.tasks)
        self.assertFalse(bank & keys)
        record = self.protocol["exclusions"]
        self.assertEqual(record["excluded_token_programs"], 72)
        self.assertGreaterEqual(
            record["excluded_polynomials"], 19000)
        self.assertIn("teacher_functions",
                      record["declared"])
        self.assertEqual(len(record["declared"][
            "statement_dev_names"]["names"]), 11)

    def test_hash_pins(self) -> None:
        for name in ("tasks", "visible", "query", "held"):
            digest = LlmBankGenerator._sha256(
                (BANK / f"{name}.json").read_bytes())
            self.assertEqual(self.protocol["files"][name],
                             digest)
        for source, digest in self.protocol[
                "generator_sources"].items():
            actual = LlmBankGenerator._sha256(
                (ROOT / source).read_bytes())
            self.assertEqual(actual, digest)
        for task in self.tasks:
            digest = LlmBankGenerator._sha256(
                LlmBankGenerator._canonical({
                    "id": task["id"], "pool": task["pool"],
                    "family": task["family"],
                    "interface": task["interface"],
                    "witness": task["witness"]}))
            self.assertEqual(task["sha256"], digest)

    def test_sealed_held_file_guard(self) -> None:
        self.assertTrue(self.protocol["seal"]["sealed"])
        with self.assertRaises(ValueError):
            self.gen.load_held()
        held = self.gen.load_held(unseal=True)
        self.assertEqual(set(held), set(self.visible))
        for task in self.tasks:
            self.assertEqual(len(held[task["id"]]), HELD_N)


if __name__ == "__main__":
    unittest.main()
