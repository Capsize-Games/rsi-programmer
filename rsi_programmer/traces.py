#!/usr/bin/env python3
"""Frozen shadow-trace recorder for the SNN RSI stage 0 dataset.

Runs the untouched probe enumerator in
scripts/rsi_programmer.probe.py and freezes each
search-state to enumerator-choice pair. Train/held-out splits and
expected outputs come from the frozen family fixture, so oracles
stay independent of the search that produced the candidate.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import cast

from rsi_programmer.probe import OPS, Expr, enumerate_programs
from rsi_programmer.probe import interpret, lower_to_dag
from rsi_programmer.probe import signature, terminals

FIXTURE = "fixtures/rsi_shadow_tasks.json"
Rows = tuple[tuple[int, int], ...]
Vals = tuple[int, ...]
Pair = dict[str, object]
Task = dict[str, object]
Graph = dict[str, object]
Levels = dict[int, list[Expr]]
Seen = set[tuple[str, Vals]]


class TraceRecorder:
    """Records probe search traces over a frozen task family."""

    def __init__(self, fixture: str = FIXTURE) -> None:
        """Bind the frozen family fixture path (no I/O yet)."""
        self._fixture: str = fixture

    def load_family(self) -> dict[str, object]:
        """Load the frozen task family fixture from disk."""
        text = Path(self._fixture).read_text(encoding="utf-8")
        return cast(dict[str, object], json.loads(text))

    @staticmethod
    def split_hash(task: Task) -> str:
        """Hash the frozen train/held-out split of one task."""
        canonical = json.dumps({"held_out": task["held_out"],
                                "id": task["id"],
                                "train": task["train"]},
                               separators=(",", ":"),
                               sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def oracle_holds(self, task: Task, graph: Graph) -> bool:
        """Check a graph against frozen expectations only."""
        cases = cast(list[list[int]], task["train"])
        cases = cases + cast(list[list[int]], task["held_out"])
        for case in cases:
            if interpret(graph, case[0], case[1]) != case[2]:
                return False
        return True

    def record_task(self, task: Task) -> dict[str, object]:
        """Run the probe enumerator; freeze state-choice pairs."""
        train = cast(list[list[int]], task["train"])
        rows = tuple((case[0], case[1]) for case in train)
        expected = tuple(case[2] for case in train)
        bound = int(cast(int, task.get("max_tree_ops", 4)))
        start = time.perf_counter()
        expr, tried, unique = enumerate_programs(rows, expected,
                                                 bound)
        wall = time.perf_counter() - start
        pairs = self._replay(rows, expected, bound)
        graph = lower_to_dag(expr) if expr is not None else None
        solved = graph is not None and self.oracle_holds(task,
                                                         graph)
        return {"choice_pairs": pairs, "solved": solved,
                "split_sha256": self.split_hash(task),
                "task": task["id"], "tried": tried,
                "unique": unique, "wall_s": wall}

    def record_all(self) -> dict[str, object]:
        """Record every family task; summarize the baseline."""
        family = self.load_family()
        tasks = cast(list[Task], family["tasks"])
        records = [self.record_task(task) for task in tasks]
        solved = sum(1 for item in records if item["solved"])
        tried = [int(item["tried"]) for item in records]
        walls = [float(item["wall_s"]) for item in records]
        return {"baseline": {"candidates_tried": tried,
                             "solve_rate": solved / len(records),
                             "solved": solved,
                             "tasks": len(records),
                             "total_wall_s": sum(walls)},
                "records": records}

    def save(self, path: str) -> str:
        """Write frozen records plus split hashes; return path."""
        text = json.dumps(self.record_all(), indent=2,
                          sort_keys=True)
        Path(path).write_text(text, encoding="utf-8")
        return path

    def _cell(self, rows: Rows, left: Expr, right: Expr,
              size: int, current: list[Expr], seen: Seen,
              pairs: list[Pair], tried: int) -> int:
        """Freeze every op decision for one operand pair."""
        for op in OPS:
            tried += 1
            pairs.append({"choice": {"op": op},
                          "state": {"left": repr(left),
                                    "right": repr(right),
                                    "size": size, "tried": tried}})
            expr = Expr(op, left, right, size)
            key = (op, signature(expr, rows))
            if key not in seen:
                seen.add(key)
                current.append(expr)
        return tried

    def _layer(self, rows: Rows, levels: Levels, size: int,
               current: list[Expr], seen: Seen,
               pairs: list[Pair], tried: int) -> int:
        """Freeze one size layer of enumerator op decisions."""
        for left_size in range(size):
            right_size = size - left_size - 1
            for left in levels[left_size]:
                for right in levels[right_size]:
                    tried = self._cell(rows, left, right, size,
                                       current, seen, pairs,
                                       tried)
        return tried

    def _replay(self, rows: Rows, expected: Vals,
                bound: int) -> list[Pair]:
        """Mirror probe enumeration while freezing decisions."""
        levels: Levels = {0: list(terminals())}
        pairs: list[Pair] = []
        tried = 0
        for size in range(1, bound + 1):
            current: list[Expr] = []
            seen: Seen = set()
            tried = self._layer(rows, levels, size, current,
                                seen, pairs, tried)
            levels[size] = current
            for expr in current:
                if signature(expr, rows) == expected:
                    return pairs
        return pairs
