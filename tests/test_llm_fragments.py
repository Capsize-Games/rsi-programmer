#!/usr/bin/env python3
"""Tests for the qualified fragment library and retriever."""
from __future__ import annotations

import unittest
from typing import Any

from rsi_programmer.cost_ledger import CostLedger
from rsi_programmer.llm_fragments import FragmentLibrary
from rsi_programmer.llm_proposer import ProposerAdapter
from rsi_programmer.llm_retrieval import FragmentRetriever
from rsi_programmer.statement_fragments import mine

SOURCE = "return acc stop"
KIND = "statement"
FAMILY = "S1"
COMPILER = "compiler-sha-256-pinned"
SOURCE_TWO = ("for if gt value zero set add acc value "
              "end end return acc stop")
DONORS = [("t1", tuple(SOURCE_TWO.split())),
          ("t2", tuple(SOURCE_TWO.split()))]


def _provenance(**extra: Any) -> dict[str, Any]:
    """Complete task/arm/model/prompt-hash/artifact provenance."""
    body: dict[str, Any] = {
        "task": "llm_dev_045",
        "arm": "single",
        "model": "spark-1.3",
        "prompt_hash": "prompt-sha-256",
        "artifact": "batch/summary.json",
    }
    body.update(extra)
    return body


def _evidence(source: str = SOURCE, **extra: Any
              ) -> dict[str, Any]:
    """Held-final acceptance evidence with full re-execution."""
    body: dict[str, Any] = {
        "candidate_text": source,
        "verdict": {
            "accepted": True,
            "stage": "final",
            "valid": True,
            "python_correct": True,
            "curlee_correct": True,
            "rows_checked": 40,
            "compiler": {"sha256": COMPILER},
            "signature": {"op": KIND,
                          "sig": [0, 0, 0]},
        },
        "compiler_before": COMPILER,
        "compiler_after": COMPILER,
        "wrong_control_accepted": False,
        "python_rows": 40,
        "curlee_rows": 40,
    }
    body.update(extra)
    return body


def _library() -> FragmentLibrary:
    """One library holding a single qualified S1 fragment."""
    library = FragmentLibrary()
    library.add_entry(SOURCE, KIND, FAMILY,
                      _provenance(), _evidence())
    return library


class AcceptTests(unittest.TestCase):
    def test_held_accepted_program_qualifies(self) -> None:
        """Full evidence stores source, family and coverage."""
        entry = _library().entries()[0]
        self.assertEqual(entry["source"], SOURCE)
        self.assertEqual(entry["kind"], KIND)
        self.assertEqual(entry["family"], FAMILY)
        self.assertEqual(entry["compiler_sha"], COMPILER)
        self.assertEqual(entry["coverage"], 40)
        self.assertEqual(entry["mining"], "none")
        self.assertEqual(entry["provenance"], _provenance())

    def test_construction_cost_hits_donor_phase(self) -> None:
        """Each qualification records donor_library_creation."""
        library = FragmentLibrary(ledger=CostLedger())
        library.add_entry(SOURCE, KIND, FAMILY,
                          _provenance(), _evidence())
        stats = library.ledger.phase_stats(
            "donor_library_creation")
        self.assertEqual(stats["n_cpu_seconds"], 1.0)

    def test_duplicate_source_refused(self) -> None:
        """The same kind plus source qualifies exactly once."""
        library = _library()
        with self.assertRaises(ValueError):
            library.add_entry(SOURCE, KIND, FAMILY,
                              _provenance(), _evidence())


class RejectTests(unittest.TestCase):
    def test_tampered_byte_source_blocked(self) -> None:
        """One changed byte fails the byte-identical match."""
        library = FragmentLibrary()
        with self.assertRaisesRegex(ValueError, "byte-match"):
            library.add_entry(SOURCE + " ", KIND, FAMILY,
                              _provenance(), _evidence())

    def test_compiler_change_blocked(self) -> None:
        """A rotated compiler binary voids the prior verdict."""
        library = FragmentLibrary()
        bad = _evidence(compiler_after="other-sha")
        with self.assertRaisesRegex(ValueError,
                                    "compiler hash changed"):
            library.add_entry(SOURCE, KIND, FAMILY,
                              _provenance(), bad)

    def test_wrong_control_pass_blocked(self) -> None:
        """A passing wrong-expected control voids acceptance."""
        library = FragmentLibrary()
        bad = _evidence(wrong_control_accepted=True)
        with self.assertRaisesRegex(
                ValueError, "wrong-expected control"):
            library.add_entry(SOURCE, KIND, FAMILY,
                              _provenance(), bad)

    def test_partial_reexecution_blocked(self) -> None:
        """Both backends must re-run every held case."""
        library = FragmentLibrary()
        bad = _evidence(python_rows=39)
        with self.assertRaisesRegex(ValueError,
                                    "re-execution"):
            library.add_entry(SOURCE, KIND, FAMILY,
                              _provenance(), bad)

    def test_visible_stage_verdict_blocked(self) -> None:
        """Repair-visible acceptance never qualifies donors."""
        library = FragmentLibrary()
        verdict = dict(_evidence()["verdict"])
        verdict["stage"] = "visible"
        bad = _evidence(verdict=verdict)
        with self.assertRaisesRegex(ValueError,
                                    "judge_final verdict"):
            library.add_entry(SOURCE, KIND, FAMILY,
                              _provenance(), bad)

    def test_rejected_verdict_blocked(self) -> None:
        """A mismatching final verdict never qualifies."""
        library = FragmentLibrary()
        verdict = dict(_evidence()["verdict"])
        verdict["accepted"] = False
        bad = _evidence(verdict=verdict)
        with self.assertRaisesRegex(ValueError,
                                    "judge_final verdict"):
            library.add_entry(SOURCE, KIND, FAMILY,
                              _provenance(), bad)


class ProvenanceTests(unittest.TestCase):
    def test_missing_key_refused(self) -> None:
        """Every provenance key is required for construction."""
        library = FragmentLibrary()
        partial = _provenance()
        del partial["artifact"]
        with self.assertRaisesRegex(ValueError, "provenance"):
            library.add_entry(SOURCE, KIND, FAMILY,
                              partial, _evidence())

    def test_empty_provenance_refused(self) -> None:
        """An empty provenance mapping qualifies nothing."""
        library = FragmentLibrary()
        with self.assertRaisesRegex(ValueError, "provenance"):
            library.add_entry(SOURCE, KIND, FAMILY,
                              {}, _evidence())

    def test_human_code_never_qualifies(self) -> None:
        """Human or JOE-authored code is refused at entry."""
        library = FragmentLibrary()
        for origin in ("human", "JOE"):
            with self.assertRaisesRegex(
                    ValueError, "human-authored", msg=origin):
                library.add_entry(
                    SOURCE, KIND, FAMILY,
                    _provenance(origin=origin), _evidence())


class MiningTests(unittest.TestCase):
    def test_mined_contexts_reuse_mine(self) -> None:
        """Classical contexts come straight from mine()."""
        library = FragmentLibrary()
        self.assertEqual(library.mined_contexts(DONORS),
                         mine(DONORS))
        self.assertTrue(library.mined_contexts(DONORS))

    def test_mining_recorded_on_entry(self) -> None:
        """Classically mined entries say so; others say none."""
        library = FragmentLibrary()
        mined = library.add_entry(
            SOURCE, KIND, FAMILY, _provenance(),
            _evidence(), mined=True)
        self.assertEqual(mined["mining"],
                         "statement_fragments.mine")
        plain = library.add_entry(
            SOURCE_TWO, KIND, FAMILY, _provenance(),
            _evidence(SOURCE_TWO), mined=False)
        self.assertEqual(plain["mining"], "none")


class RetrievalTests(unittest.TestCase):
    def test_same_task_and_version_same_ids(self) -> None:
        """Two retrievals serve identical fragment ids."""
        retriever = FragmentRetriever(_library())
        first, shown_first = retriever.retrieve(
            FAMILY, {"op": KIND})
        second, shown_second = retriever.retrieve(
            FAMILY, {"op": KIND})
        self.assertEqual(shown_first, shown_second)
        self.assertEqual(
            [item["fragment_id"] for item in first],
            [item["fragment_id"] for item in second])

    def test_top_k_caps_served_fragments(self) -> None:
        """Top-k cuts the id-sorted candidate list."""
        library = _library()
        library.add_entry(SOURCE_TWO, KIND, FAMILY,
                          _provenance(), _evidence(SOURCE_TWO))
        retriever = FragmentRetriever(library)
        fragments, shown = retriever.retrieve(
            FAMILY, {"op": KIND}, top_k=1)
        self.assertEqual(len(fragments), 1)
        self.assertEqual(len(shown), 1)

    def test_family_mismatch_serves_nothing(self) -> None:
        """Other families never leak into this task's prompt."""
        retriever = FragmentRetriever(_library())
        fragments, shown = retriever.retrieve("A6", None)
        self.assertEqual(fragments, [])
        self.assertEqual(shown, [])

    def test_only_qualified_fragments_injected(self) -> None:
        """Injected fragments carry provenance the gate wants."""
        retriever = FragmentRetriever(_library())
        spec = {"task_id": "llm_dev_045", "kind": KIND,
                "visible_rows": [[[1], 0]]}
        body, shown = retriever.inject(
            spec, FAMILY, {"op": KIND})
        self.assertEqual(len(shown), 1)
        adapter = ProposerAdapter()
        prompt = adapter.build_prompt(body)
        self.assertIn("llm_dev_045", prompt)
        self.assertEqual(spec.get("fragments", []), [])
        self.assertEqual(
            body["fragments"][0]["fragment_id"], shown[0])

    def test_shown_ids_logged_with_version(self) -> None:
        """Every serving records version, family and ids."""
        library = _library()
        retriever = FragmentRetriever(library)
        _, shown = retriever.retrieve(FAMILY, {"op": KIND})
        (record,) = retriever.shown_log()
        self.assertEqual(record["version"],
                         library.version_hash())
        self.assertEqual(record["family"], FAMILY)
        self.assertEqual(record["ids"], shown)

    def test_unqualified_source_refused(self) -> None:
        """The retriever binds a qualified library only."""
        with self.assertRaises(ValueError):
            FragmentRetriever(object())  # type: ignore[arg-type]


class FreezeTests(unittest.TestCase):
    def test_version_hash_stable(self) -> None:
        """An unchanged library freezes to one version."""
        library = _library()
        self.assertEqual(library.version_hash(),
                         library.version_hash())

    def test_version_hash_changes_on_add(self) -> None:
        """A second entry moves the frozen version."""
        library = _library()
        before = library.version_hash()
        library.add_entry(SOURCE_TWO, KIND, FAMILY,
                          _provenance(), _evidence(SOURCE_TWO))
        self.assertNotEqual(library.version_hash(), before)


if __name__ == "__main__":
    unittest.main()
