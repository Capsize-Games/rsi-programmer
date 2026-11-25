"""Append-only retention of proposals and seen dedup keys.

Every proposal the Spark 1.3 proposer records stays visible;
:meth:`CandidateStore.note` makes duplicate (op, sig) submissions
observable instead of silently dropping them. The judge decides;
retention alone proves nothing.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class CandidateStore:
    """Append-only retention of every proposal and seen keys."""

    def __init__(self) -> None:
        self._proposals: list[dict[str, Any]] = []
        self._seen: set[tuple[str, tuple[int, ...]]] = set()

    def append(self, proposal: Mapping[str, Any]) -> int:
        """Retain a copy; return its index; never mutate history."""
        index = len(self._proposals)
        self._proposals.append(dict(proposal))
        return index

    def __len__(self) -> int:
        return len(self._proposals)

    def entries(self) -> list[dict[str, Any]]:
        """Copies, so callers cannot rewrite retained history."""
        return [dict(entry) for entry in self._proposals]

    def seen(self, key: tuple[str, tuple[int, ...]]) -> bool:
        """True once a key was noted; duplicates stay visible."""
        return key in self._seen

    def note(self, key: tuple[str, tuple[int, ...]]) -> bool:
        """Record a key; False when it was already recorded."""
        if key in self._seen:
            return False
        self._seen.add(key)
        return True
