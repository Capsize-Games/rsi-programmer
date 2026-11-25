#!/usr/bin/env python3
"""Tests for the Spark 1.3 proposer adapter and candidate gate."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from rsi_programmer import sandbox
from rsi_programmer.llm_proposer import DENIED_TOKENS
from rsi_programmer.llm_proposer import MAX_BYTES, MAX_NODES
from rsi_programmer.llm_proposer import MODEL_ID, MODEL_VERSION
from rsi_programmer.llm_proposer import CandidateStore
from rsi_programmer.llm_proposer import ProposerAdapter
from rsi_programmer.sandbox import SandboxRunner


def _graph(op: str = "add") -> dict[str, Any]:
    return {"input_types": {"x": "Int", "y": "Int"},
            "nodes": [{"op": op, "type": "Int",
                       "args": ["x", "y"]}],
            "output": 0}


def _spec(**extra: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "task_id": "dev-a6-001",
        "kind": "graph",
        "visible_rows": [[1, 2], [3, 4]],
    }
    base.update(extra)
    return base


STATEMENT = "return acc stop"


class ProposerProvenanceTests(unittest.TestCase):
    def test_proposal_records_complete_provenance(self) -> None:
        """Every proposal carries model, prompt hash and costs."""
        store = CandidateStore()
        adapter = ProposerAdapter(store)
        text = json.dumps(_graph())
        got = adapter.propose(
            _spec(params={"temperature": 0.2}),
            text,
            token_usage={"prompt": 120, "completion": 40},
            cpu=1.5)
        self.assertEqual(got["model"], MODEL_ID)
        self.assertEqual(got["model_version"], MODEL_VERSION)
        self.assertEqual(got["candidate_text"], text)
        self.assertEqual(got["params"], {"temperature": 0.2})
        self.assertEqual(got["token_usage"]["prompt"], 120)
        self.assertEqual(got["token_usage"]["completion"], 40)
        self.assertEqual(got["cpu"], 1.5)
        want = hashlib.sha256(
            adapter.build_prompt(_spec(
                params={"temperature": 0.2})).encode(
                    "utf-8")).hexdigest()
        self.assertEqual(got["prompt_sha256"], want)
        self.assertEqual(len(store), 1)
        self.assertEqual(store.entries()[0], got)

    def test_prompt_hash_binds_visible_rows(self) -> None:
        """Changing visible rows changes the recorded prompt hash."""
        adapter = ProposerAdapter()
        first = adapter.propose(_spec(), json.dumps(_graph()))
        rows = [[9, 9], [8, 8]]
        second = adapter.propose(_spec(visible_rows=rows),
                                 json.dumps(_graph()))
        self.assertNotEqual(first["prompt_sha256"],
                            second["prompt_sha256"])

    def test_store_is_append_only(self) -> None:
        """History entries survive later proposals and copies."""
        adapter = ProposerAdapter()
        first = adapter.propose(_spec(), json.dumps(_graph()))
        adapter.propose(_spec(), json.dumps(_graph("mul")))
        self.assertEqual(len(adapter.store), 2)
        entries = adapter.store.entries()
        entries[0]["model"] = "rewritten"
        self.assertEqual(adapter.store.entries()[0], first)
        self.assertEqual(adapter.store.entries()[0]["model"],
                         MODEL_ID)

    def test_model_identity_is_spark_workers(self) -> None:
        """The recorded identity names the supervised workers."""
        self.assertEqual(MODEL_ID, "spark-1.3")
        self.assertEqual(MODEL_VERSION,
                         "Spark 1.3 supervised background "
                         "workers")


class GateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = ProposerAdapter()

    def test_gate_accepts_valid_graph(self) -> None:
        """A well-typed arithmetic graph passes the gate."""
        res = self.adapter.gate(json.dumps(_graph()), "graph")
        self.assertEqual(res["status"], "accepted")
        self.assertEqual(res["violations"], [])

    def test_gate_accepts_valid_statement(self) -> None:
        """A well-typed token program passes the gate."""
        res = self.adapter.gate(STATEMENT, "statement")
        self.assertEqual(res["status"], "accepted")

    def test_gate_rejects_malformed_candidates(self) -> None:
        """Bad JSON, unknown ops and bad tokens are refused."""
        res = self.adapter.gate("not json {", "graph")
        self.assertEqual(res["status"], "rejected")
        res = self.adapter.gate(
            json.dumps(_graph(op="shell")), "graph")
        self.assertEqual(res["status"], "rejected")
        res = self.adapter.gate("return", "statement")
        self.assertEqual(res["status"], "rejected")
        res = self.adapter.gate("frobnicate now", "statement")
        self.assertEqual(res["status"], "rejected")
        res = self.adapter.gate(json.dumps(_graph()),
                                 "nope")
        self.assertEqual(res["status"], "rejected")

    def test_gate_rejects_over_cap_candidates(self) -> None:
        """Node, byte and token caps match the sandbox bounds."""
        nodes = [{"op": "add", "type": "Int",
                  "args": ["x", "y"]}]
        nodes += [{"op": "add", "type": "Int",
                   "args": [i - 1, "y"]}
                  for i in range(1, MAX_NODES + 5)]
        big = dict(_graph(), nodes=nodes)
        res = self.adapter.gate(json.dumps(big), "graph")
        self.assertEqual(res["status"], "rejected")
        self.assertTrue(any(v.startswith("budget")
                            for v in res["violations"]))
        fat = "x" * (MAX_BYTES + 1)
        res = self.adapter.gate(fat, "graph")
        self.assertEqual(res["status"], "rejected")
        many = " ".join(["set"] * 40)
        res = self.adapter.gate(many, "statement")
        self.assertEqual(res["status"], "rejected")
        self.assertTrue(any(v.startswith("budget")
                            for v in res["violations"]))

    def test_gate_rejects_fenced_tokens(self) -> None:
        """Oracle/held/ledger and tool tokens never pass."""
        for sneak in ("oracle_answers", "held_out rows",
                      "socket.connect", "subprocess run",
                      "__import__", "open(/etc/passwd)"):
            text = json.dumps(_graph()) + " " + sneak
            res = self.adapter.gate(text, "graph")
            self.assertEqual(res["status"], "rejected",
                             sneak)
            self.assertTrue(any(v.startswith(
                "denied-reference")
                for v in res["violations"]), sneak)

    def test_sandbox_constant_parity(self) -> None:
        """Gate caps and denylist match the real SandboxRunner."""
        self.assertEqual(DENIED_TOKENS, sandbox._DENIED)
        self.assertEqual(MAX_NODES, 64)
        self.assertEqual(MAX_BYTES, 65536)
        tmp = Path(tempfile.mkdtemp(prefix="rsi-llm-"))
        runner = SandboxRunner(tmp / "box")
        nodes = [{"op": "add", "type": "Int",
                  "args": ["x", "y"]}]
        nodes += [{"op": "add", "type": "Int",
                   "args": [i - 1, "y"]}
                  for i in range(1, MAX_NODES + 1)]
        big = dict(_graph(), nodes=nodes)
        mine = self.adapter.gate(json.dumps(big), "graph")
        theirs = runner.run_candidate(big, [[1, 2]])
        self.assertEqual(mine["status"], "rejected")
        self.assertEqual(theirs["status"], "rejected")


class DedupTests(unittest.TestCase):
    def test_same_outputs_dedup_on_op_sig(self) -> None:
        """Identical (op, sig) notes once; then reads duplicate."""
        store = CandidateStore()
        adapter = ProposerAdapter(store)
        rows = [[1, 2], [3, 4]]
        key = adapter.dedup_key(
            adapter.graph_root_op(_graph()),
            adapter.graph_signature(_graph(), rows))
        self.assertEqual(key[0], "add")
        self.assertEqual(key[1], (3, 7))
        self.assertTrue(store.note(key))
        self.assertFalse(store.note(key))
        self.assertTrue(store.seen(key))

    def test_different_programs_keep_distinct_keys(self) -> None:
        """A different root op with new outputs is not a dup."""
        store = CandidateStore()
        adapter = ProposerAdapter(store)
        rows = [[1, 2], [3, 4]]
        first = adapter.dedup_key(
            adapter.graph_root_op(_graph()),
            adapter.graph_signature(_graph(), rows))
        other = _graph(op="mul")
        second = adapter.dedup_key(
            adapter.graph_root_op(other),
            adapter.graph_signature(other, rows))
        self.assertNotEqual(first, second)
        self.assertTrue(store.note(first))
        self.assertTrue(store.note(second))


class SpecRefusalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = ProposerAdapter()

    def test_held_query_and_oracle_specs_refused(self) -> None:
        """Held, query, oracle and sealed fields never prompt."""
        for bad in ("held_rows", "query_rows", "oracle",
                    "sealed", "held_out"):
            with self.assertRaises(ValueError, msg=bad):
                self.adapter.propose(
                    _spec(**{bad: [[1, 2]]}),
                    json.dumps(_graph()))
        with self.assertRaises(ValueError):
            self.adapter.propose({"mystery": 1},
                                 json.dumps(_graph()))
        self.assertEqual(len(self.adapter.store), 0)

    def test_unqualified_fragments_refused(self) -> None:
        """Fragments without provenance are not accepted."""
        with self.assertRaises(ValueError):
            self.adapter.propose(
                _spec(fragments=[{"text": "x+y"}]),
                json.dumps(_graph()))
        with self.assertRaises(ValueError):
            self.adapter.propose(
                _spec(fragments=["x+y"]),
                json.dumps(_graph()))

    def test_qualified_fragments_and_visible_cases_pass(self
                                                       ) -> None:
        """Qualified fragments and visible cases build prompts."""
        fragment = {"content": "add(x, y)",
                    "provenance": {"source": "bank-v0",
                                   "model": MODEL_ID}}
        got = self.adapter.propose(
            _spec(fragments=[fragment],
                  counterexamples=[[5, 5]]),
            json.dumps(_graph()))
        self.assertEqual(got["model"], MODEL_ID)
        self.assertEqual(len(self.adapter.store), 1)


if __name__ == "__main__":
    unittest.main()
