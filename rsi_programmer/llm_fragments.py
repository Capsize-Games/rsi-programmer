"""Held-gated reusable fragment library for Spark 1.3 proposals.

An entry exists only for a program accepted by
:meth:`LlmJudge.judge_final` on sealed held rows, proven by four
re-checked gates: the stored source byte-matches the judged
candidate, the pinned CURLEE compiler hash is identical before
and after re-execution, a deliberately-wrong control candidate
is rejected, and every held case is re-executed in both
backends. Classical condition contexts from
:func:`statement_fragments.mine` may be attached; such reuse is
recorded on the entry instead of claimed as learned.
Construction cost is charged to the ``donor_library_creation``
ledger phase. The library freezes to a version hash over its
entry ids and hashes, so adding an entry changes the version
retrieval logs against.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from rsi_programmer.cost_ledger import CostLedger
from rsi_programmer.statement_fragments import mine

FAMILIES = ("A6", "A7", "A8", "S1", "S2", "S3")
KINDS = ("graph", "statement")
MINING_RECORDER = "statement_fragments.mine"
PROVENANCE_KEYS = ("task", "arm", "model",
                   "prompt_hash", "artifact")


def _entry_id(kind: str, source: str) -> str:
    """Deterministic 16-hex id over kind plus exact source."""
    body = kind + "\x00" + source
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


class FragmentLibrary:
    """Store only held-accepted programs with re-checked proof."""

    def __init__(self, ledger: CostLedger | None = None
                 ) -> None:
        """Bind an optional cost ledger for construction samples."""
        self._entries: dict[str, dict[str, Any]] = {}
        self._ledger = ledger if ledger is not None \
            else CostLedger()

    @property
    def ledger(self) -> CostLedger:
        """Ledger carrying ``donor_library_creation`` samples."""
        return self._ledger

    def __len__(self) -> int:
        """Number of qualified entries currently stored."""
        return len(self._entries)

    def entries(self) -> list[dict[str, Any]]:
        """Copies in id order; callers cannot rewrite history."""
        return [dict(self._entries[key])
                for key in sorted(self._entries)]

    @staticmethod
    def _require_provenance(provenance: Mapping[str, Any]
                            ) -> dict[str, Any]:
        """Copy provenance or raise when any required key is off."""
        if not isinstance(provenance, Mapping):
            raise ValueError(
                "complete provenance (task, arm, model, "
                "prompt-hash, artifact) required")
        missing = [key for key in PROVENANCE_KEYS
                   if not isinstance(provenance.get(key), str)
                   or not provenance[key]]
        if missing:
            raise ValueError(
                "complete provenance (task, arm, model, "
                f"prompt-hash, artifact) required: {missing}")
        origin = str(provenance.get("origin", "")).lower()
        if origin in ("human", "joe"):
            raise ValueError(
                "human-authored code is never a library entry")
        return dict(provenance)

    @staticmethod
    def _require_evidence(source: str, kind: str,
                          evidence: Mapping[str, Any]
                          ) -> dict[str, Any]:
        """Check the four held-acceptance gates with specifics."""
        if not isinstance(evidence, Mapping):
            raise ValueError("judge_final evidence required")
        if evidence.get("candidate_text") != source:
            raise ValueError(
                "fragment source does not byte-match the "
                "held-accepted candidate")
        verdict = evidence.get("verdict")
        if not isinstance(verdict, Mapping) \
                or verdict.get("accepted") is not True \
                or verdict.get("stage") != "final" \
                or verdict.get("valid") is not True \
                or verdict.get("python_correct") is not True \
                or verdict.get("curlee_correct") is not True:
            raise ValueError(
                "held-accepted judge_final verdict required")
        rows = verdict.get("rows_checked")
        if not isinstance(rows, int) or rows <= 0:
            raise ValueError(
                "held-accepted judge_final verdict required")
        compiler = verdict.get("compiler")
        if not isinstance(compiler, Mapping) \
                or not compiler.get("sha256"):
            raise ValueError(
                "pinned compiler hash missing from verdict")
        pinned = str(compiler["sha256"])
        if evidence.get("compiler_before") != pinned \
                or evidence.get("compiler_after") != pinned:
            raise ValueError(
                "compiler hash changed between judge_final "
                "and re-execution")
        if evidence.get("wrong_control_accepted") is not False:
            raise ValueError(
                "wrong-expected control was not rejected")
        for key in ("python_rows", "curlee_rows"):
            if evidence.get(key) != rows:
                raise ValueError(
                    "held re-execution incomplete in both "
                    "backends")
        return dict(verdict)

    def add_entry(self, source: str, kind: str,
                  family: str,
                  provenance: Mapping[str, Any],
                  evidence: Mapping[str, Any],
                  mined: bool = False) -> dict[str, Any]:
        """Qualify one program and store its reusable fragment."""
        if kind not in KINDS:
            raise ValueError(f"unknown fragment kind: {kind!r}")
        if family not in FAMILIES:
            raise ValueError(
                f"unknown spec family: {family!r}")
        if not isinstance(source, str) or not source:
            raise ValueError("fragment source must be text")
        kept_provenance = self._require_provenance(provenance)
        with self._ledger.measure("donor_library_creation"):
            verdict = self._require_evidence(
                source, kind, evidence)
            signature = verdict.get("signature")
            entry = {
                "id": _entry_id(kind, source),
                "source": source,
                "kind": kind,
                "family": family,
                "signature": dict(signature)
                if isinstance(signature, Mapping)
                else None,
                "provenance": kept_provenance,
                "compiler_sha": str(
                    verdict["compiler"]["sha256"]),
                "coverage": int(verdict["rows_checked"]),
                "mining": MINING_RECORDER if mined else "none",
            }
            if entry["id"] in self._entries:
                raise ValueError(
                    "fragment entry already qualified: "
                    + str(entry["id"]))
            self._entries[entry["id"]] = entry
            return dict(entry)

    def mined_contexts(
            self, donors: list[tuple[str, tuple[str, ...]]]
    ) -> list[dict[str, Any]]:
        """Classical condition contexts; charged to the library."""
        with self._ledger.measure("donor_library_creation"):
            return mine(donors)

    def version_hash(self) -> str:
        """Freeze id list plus hashes; any new entry changes it."""
        lines = []
        for key in sorted(self._entries):
            entry = self._entries[key]
            digest = hashlib.sha256(
                entry["source"].encode("utf-8")).hexdigest()
            lines.append(f"{key}:{digest}:{entry['compiler_sha']}")
        return hashlib.sha256(
            "\n".join(lines).encode("utf-8")).hexdigest()
