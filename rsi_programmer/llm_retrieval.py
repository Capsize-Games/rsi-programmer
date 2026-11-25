"""Deterministic retrieval over the qualified fragment library.

The retriever reads only from a :class:`FragmentLibrary`, so
unqualified, human-authored, or otherwise unvetted code can
never be injected: every served fragment carries the stored
provenance the proposer gate requires. Selection is a pure
function of spec family, visible-row signature op, and the
library version hash, sorted by fragment id and cut to top-k,
with every served id list logged for audit.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rsi_programmer.llm_fragments import FragmentLibrary


class FragmentRetriever:
    """Serve qualified fragments deterministically with id logs."""

    def __init__(self, library: FragmentLibrary) -> None:
        """Bind the only source this retriever may ever serve."""
        if not isinstance(library, FragmentLibrary):
            raise ValueError(
                "retrieval serves a qualified library only")
        self._library = library
        self._shown: list[dict[str, Any]] = []

    @property
    def library(self) -> FragmentLibrary:
        """The qualified library backing every retrieval."""
        return self._library

    def shown_log(self) -> list[dict[str, Any]]:
        """Copies of every (version, family, ids) serving record."""
        return [dict(record) for record in self._shown]

    @staticmethod
    def _signature_op(signature: Mapping[str, Any] | None
                      ) -> str | None:
        """Visible-row signature op, or None when unknown."""
        if isinstance(signature, Mapping):
            op = signature.get("op")
            if isinstance(op, str) and op:
                return op
        return None

    def _candidates(
            self, family: str,
            signature: Mapping[str, Any] | None
    ) -> list[dict[str, Any]]:
        """Qualified family matches, narrowed by signature op."""
        want = self._signature_op(signature)
        kept = []
        for entry in self._library.entries():
            if entry["family"] != family:
                continue
            if not entry.get("provenance"):
                continue
            have = self._signature_op(entry.get("signature"))
            if want is not None and have is not None \
                    and want != have:
                continue
            kept.append(entry)
        kept.sort(key=lambda item: str(item["id"]))
        return kept

    def retrieve(self, family: str,
                 signature: Mapping[str, Any] | None,
                 top_k: int = 3
                 ) -> tuple[list[dict[str, Any]], list[str]]:
        """Serve top-k fragments; log served ids with the version."""
        if top_k <= 0:
            raise ValueError(
                "top_k must be a positive request count")
        version = self._library.version_hash()
        picked = self._candidates(family, signature)[:top_k]
        fragments = [{
            "content": entry["source"],
            "kind": entry["kind"],
            "family": entry["family"],
            "fragment_id": entry["id"],
            "provenance": dict(entry["provenance"]),
        } for entry in picked]
        shown = [str(item["fragment_id"]) for item in fragments]
        self._shown.append({"version": version,
                            "family": family, "ids": list(shown)})
        return fragments, shown

    def inject(self, spec: Mapping[str, Any], family: str,
               signature: Mapping[str, Any] | None,
               top_k: int = 3
               ) -> tuple[dict[str, Any], list[str]]:
        """Copy a spec with only qualified fragments appended."""
        if not isinstance(spec, Mapping):
            raise ValueError("spec must be a mapping")
        fragments, shown = self.retrieve(family, signature,
                                         top_k)
        body = dict(spec)
        prior = body.get("fragments", [])
        if not isinstance(prior, list):
            raise ValueError("spec fragments must be a list")
        body["fragments"] = [*prior, *fragments]
        return body, shown
