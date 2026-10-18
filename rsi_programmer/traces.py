#!/usr/bin/env python3
"""Frozen shadow-trace recorder for the SNN RSI stage 0 dataset.

Observes the probe enumerator and freezes each search-state to
enumerator-choice pair without replaying signature computations.
Train/held-out splits and
expected outputs come from the frozen family fixture, so oracles
stay independent of the search that produced the candidate.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import cast

from rsi_programmer.probe import Expr, enumerate_programs
from rsi_programmer.probe import interpret, lower_to_dag

FIXTURE = "fixtures/rsi_shadow_tasks.json"
Rows = tuple[tuple[int, int], ...]
Vals = tuple[int, ...]
Pair = dict[str, object]
Task = dict[str, object]
Graph = dict[str, object]


class TraceRecorder:
    """Records probe traces over a frozen family.

    Reported enumeration wall time includes the trace-capture callback.
    """

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
        """Enumerate once, capturing choices and candidate signatures.

        ``wall_s`` includes the per-candidate trace-capture callback and is
        not a pure-enumerator timing.
        """
        train = cast(list[list[int]], task["train"])
        rows = tuple((case[0], case[1]) for case in train)
        expected = tuple(case[2] for case in train)
        bound = int(cast(int, task.get("max_tree_ops", 4)))
        pairs: list[Pair] = []

        def record(expr: Expr, sig: Vals, tried: int) -> None:
            pairs.append({"choice": {"op": expr.op},
                          "state": {"left": repr(expr.left),
                                    "right": repr(expr.right),
                                    "size": expr.size, "tried": tried}})

        start = time.perf_counter()
        expr, tried, unique = enumerate_programs(rows, expected,
                                                 bound,
                                                 on_candidate=record)
        wall = time.perf_counter() - start
        graph = lower_to_dag(expr) if expr is not None else None
        solved = graph is not None and self.oracle_holds(task,
                                                         graph)
        return {"choice_pairs": pairs, "solved": solved,
                "split_sha256": self.split_hash(task),
                "task": task["id"], "tried": tried,
                "signature_evaluations": tried,
                "tried_unit": "candidate_signature_evaluations",
                "unique": unique, "wall_s": wall,
                "wall_scope": "enumeration_with_trace_capture"}

    def record_all(self) -> dict[str, object]:
        """Record every family task; summarize the baseline."""
        family = self.load_family()
        tasks = cast(list[Task], family["tasks"])
        records = [self.record_task(task) for task in tasks]
        solved = sum(1 for item in records if item["solved"])
        tried = [int(item["tried"]) for item in records]
        walls = [float(item["wall_s"]) for item in records]
        return {"baseline": {"candidates_tried": tried,
                             "signature_evaluations": tried,
                             "cost_unit":
                             "candidate_signature_evaluations",
                             "solve_rate": solved / len(records),
                             "solved": solved,
                             "tasks": len(records),
                             "total_wall_s": sum(walls),
                             "wall_scope":
                             "enumeration_with_trace_capture"},
                "records": records}

    def save(self, path: str) -> str:
        """Write frozen records plus split hashes; return path."""
        text = json.dumps(self.record_all(), indent=2,
                          sort_keys=True)
        Path(path).write_text(text, encoding="utf-8")
        return path
