"""Lineage store for RSI training runs."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


class LineageError(Exception):
    """Duplicate mission, unknown mission, or invalid identity."""


_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS missions(mission_id TEXT PRIMARY KEY,"
    "revision INTEGER,authority TEXT,status TEXT,created_ns INTEGER,"
    "fence_term INTEGER)",
    "CREATE TABLE IF NOT EXISTS commands(command_id TEXT PRIMARY KEY,"
    "mission_id TEXT,authority_term INTEGER,generation INTEGER,"
    "digest TEXT,intent TEXT,receipt TEXT,state TEXT)",
)


def _doc(intent: dict[str, Any]) -> tuple[str, str]:
    """Canonical JSON and its sha256 for one intent entry."""
    text = json.dumps(intent, sort_keys=True, separators=(",", ":"))
    return text, hashlib.sha256(text.encode()).hexdigest()


def mission_ledger(ledger: str | Path, mission: str) -> LineageStore:
    """Store with ``mission`` created exactly once, ready to use."""
    path = Path(ledger)
    setup = LineageStore(path)
    try:
        setup.create_mission(mission, 1, {"task": mission})
    except LineageError:
        pass
    setup.close()
    return LineageStore(path)


class LineageStore:
    """Mission intents with read-back rows for Pareto archives."""

    def __init__(self, path: str | Path) -> None:
        """Open (creating parents) the lineage database file."""
        self._file = Path(path)
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._file))
        with self._lock:
            for stmt in _SCHEMA:
                self._conn.execute(stmt)
            self._conn.commit()

    def close(self) -> None:
        """Flush and close the database handle."""
        with self._lock:
            self._conn.commit()
            self._conn.close()

    def create_mission(self, mission_id: str, revision: int,
                       authority: dict[str, Any]) -> None:
        """Register a mission once; duplicates raise LineageError."""
        if not mission_id or not isinstance(revision, int):
            raise LineageError("mission identity is invalid")
        if revision < 1:
            raise LineageError("mission identity is invalid")
        doc = json.dumps(authority, sort_keys=True,
                         separators=(",", ":"))
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO missions(mission_id,revision,"
                    "authority,status,created_ns,fence_term)"
                    "VALUES(?, ?, ?, 'running', ?, 0)",
                    (mission_id, revision, doc, time.time_ns()))
                self._conn.commit()
            except sqlite3.IntegrityError as exc:
                raise LineageError("mission exists already") from exc

    def record_intent(self, command_id: str, mission_id: str,
                      term: int, generation: int,
                      intent: dict[str, Any]) -> None:
        """Append one prepared intent row for later read-back."""
        if not command_id or term < 0 or generation < 1:
            raise LineageError("command identity is invalid")
        doc, digest = _doc(intent)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO commands(command_id,"
                "mission_id,authority_term,generation,digest,intent,"
                "receipt,state)VALUES(?, ?, ?, ?, ?, ?, NULL,"
                "'prepared')",
                (command_id, mission_id, term, generation,
                 digest, doc))
            self._conn.commit()

    def commands_in(self, mission_id: str,
                    state: str) -> list[dict[str, Any]]:
        """Ordered rows for one mission and state, intents as JSON."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT command_id,authority_term,generation,"
                "digest,intent,receipt FROM commands WHERE "
                "mission_id=? AND state=? ORDER BY command_id",
                (mission_id, state)).fetchall()
        return [{"command_id": r[0], "term": r[1],
                 "generation": r[2], "digest": r[3],
                 "intent": r[4], "receipt": r[5]} for r in rows]
