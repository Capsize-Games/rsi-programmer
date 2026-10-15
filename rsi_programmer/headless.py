#!/usr/bin/env python3
"""Hands-off RSI headless loop: SNN proposes, sandbox disposes (W4).

Wires PolicyLoop guided search (stages 0-2) with SandboxRunner
verdicts and PromotionGate decisions into one run behind budget
flags: max wall seconds, max candidates, max generations. Each
decision appends one record to a JSONL trace; budget exhaustion
stops the loop with status stopped-not-failed. No per-decision
outside input and no network; oracle and promotion files never
enter candidate case directories.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, cast

from rsi_programmer.evolve import guided_search
from rsi_programmer.policy import SearchPolicy
from rsi_programmer.promote import PromotionGate
from rsi_programmer.sandbox import SandboxRunner
from rsi_programmer.traces import TraceRecorder

Rows = tuple[tuple[int, int], ...]
Vals = tuple[int, ...]
Pairs = list[list[int]]
Claim = dict[str, Any] | None
Hook = Callable[[str, dict[str, Any] | None], Claim]
Tasks = list[dict[str, Any]]
BAD = ("error", "timeout", "execution-mismatch")


class HeadlessLoop:
    """One hands-off sweep of the frozen family within budgets."""

    def __init__(self, fixture: str, trace: str, workdir: str,
                 seed: int = 0, max_wall_s: float = 600.0,
                 max_cands: int = 32, max_gens: int = 1, epochs: int = 40,
                 policy: SearchPolicy | None = None) -> None:
        """Bind family, budgets, trace path, sandbox, and gate."""
        if max_wall_s <= 0 or max_cands < 1 or max_gens < 1:
            raise ValueError("headless budgets must be positive")
        self._fixture = fixture
        self._trace = trace
        self._recorder = TraceRecorder(fixture)
        self._box = SandboxRunner(Path(workdir))
        self._gate = PromotionGate()
        self._seed = seed
        self._max_wall_s = max_wall_s
        self._max_cands = max_cands
        self._max_gens = max_gens
        self._epochs = epochs
        self._given = policy
        self._spent = 0

    def _split(self, task: dict[str, Any]
               ) -> tuple[Rows, Vals, int, Pairs]:
        """Train rows in search, sandbox, and bound forms."""
        train = task["train"]
        pairs = [[c[0], c[1]] for c in train]
        rows = tuple((c[0], c[1]) for c in train)
        target = tuple(c[2] for c in train)
        return rows, target, int(task.get("max_tree_ops", 4)), pairs

    def _baseline(self) -> dict[str, Any]:
        """Frozen W1 enumerator solve-rate, cost, and budget."""
        report = self._recorder.record_all()
        records = report["records"]
        solved = [1.0 if r["solved"] else 0.0 for r in records]
        cost = sum(int(r["tried"]) for r in records)
        rate = report["baseline"]["solve_rate"]
        return {"solve_rate": rate, "cost": cost,
                "solved": solved, "budget": cost}

    def _trained(self) -> SearchPolicy:
        """Train hands-off on frozen W1 traces unless given."""
        if self._given is not None:
            return self._given
        policy = SearchPolicy(seed=self._seed + 1)
        policy.fit(policy.trace_groups(self._fixture), self._epochs)
        return policy

    def _over(self, start: float) -> bool:
        """True once any predeclared budget is exhausted."""
        return (self._spent >= self._max_cands
                or time.perf_counter() - start >= self._max_wall_s)

    def _append(self, record: dict[str, Any]) -> None:
        """Append one decision record to the JSONL trace file."""
        with open(self._trace, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _minimize(self, graph: dict[str, Any]) -> dict[str, Any]:
        """Keep only nodes reachable from the output reference."""
        remap: dict[int, int] = {}; kept: list[dict[str, Any]] = []

        def visit(ref: object) -> object:
            if not isinstance(ref, int):
                return ref
            if ref not in remap:
                node = graph["nodes"][ref]
                args = [visit(a) for a in node["args"]]
                remap[ref] = len(kept)
                kept.append({"args": args, "op": node["op"],
                             "type": node["type"],
                             **({"value": node["value"]}
                                if "value" in node else {})})
            return remap[ref]

        root = visit(graph["output"])
        return {"input_types": {"x": "Int", "y": "Int"},
                "nodes": kept, "output": root}

    def _judge(self, graph: dict[str, Any] | None,
               pairs: Pairs, claim: Claim,
               task: dict[str, Any]
               ) -> tuple[dict[str, Any] | None, bool, str, bool]:
        """Sandbox verdict plus supervisor-side oracle check."""
        if graph is None:
            return None, False, "", False
        verdict = self._box.run_candidate(graph, pairs, claim)
        holds = bool(self._recorder.oracle_holds(task, graph))
        good = verdict["status"] == "accepted" and holds
        line = self._box.lineage_hash(graph)
        return verdict, holds, line, good

    def _propose(self, policy: SearchPolicy, task: dict[str, Any],
                 gen: int, hook: Hook | None,
                 budget: int) -> dict[str, Any]:
        """Search once; sandbox verdict recomputed supervisor-side."""
        rows, target, bound, pairs = self._split(task)
        out = guided_search(policy, rows, target, bound,
                            budget, str(task["id"]))
        graph = out["graph"] if out["solved"] else None
        mini = self._minimize(graph) if graph is not None else None
        claim: Claim = hook(str(task["id"]), mini) if hook else None
        verdict, holds, lineage, solved = self._judge(mini, pairs, claim,
                                                     task)
        status = verdict["status"] if verdict else "no-candidate"
        viol = verdict["violations"] if verdict else []
        record = {"gen": gen, "task": str(task["id"]),
                  "tried": out["tried"], "solved": solved,
                  "sandbox": status, "violations": list(viol),
                  "lineage": lineage, "oracle_holds": holds}
        self._append(record)
        return record

    def _gate_decide(self, ref: list[float], cand: list[float],
                     bad: int, cost: int, wall: float
                     ) -> dict[str, Any]:
        """Paired gate verdict; empty runs never promote."""
        if not ref:
            return {"promote": False,
                    "reasons": ["no decisions completed"]}
        return self._gate.decide(ref, cand, bad, cost, wall)

    def run(self, hook: Hook | None = None) -> dict[str, Any]:
        """Sweep the family within budgets; budgets stop, not fail."""
        start = time.perf_counter()
        base = self._baseline()
        policy = self._trained()
        family = self._recorder.load_family()
        tasks = cast(Tasks, family["tasks"])
        decisions: list[dict[str, Any]] = []
        for gen in range(self._max_gens):
            for task in tasks:
                if self._over(start):
                    break
                decisions.append(self._propose(policy, task, gen,
                                               hook, base["budget"]))
                self._spent += 1
            if self._over(start):
                break
        return self._summarize(start, base, tasks, decisions)

    def _summarize(self, start: float, base: dict[str, Any],
                   tasks: Tasks,
                   decisions: list[dict[str, Any]]) -> dict[str, Any]:
        """Solve-rate/cost vs baseline plus the gate verdict."""
        wall = time.perf_counter() - start
        cand = [1.0 if d["solved"] else 0.0 for d in decisions]
        ref = [base["solved"][i % len(tasks)] for i in range(len(cand))]
        bad = sum(1 for d in decisions if d["sandbox"] in BAD)
        cost = sum(int(d["tried"]) for d in decisions)
        rate = sum(cand) / len(cand) if cand else 0.0
        gate = self._gate_decide(ref, cand, bad, cost, wall)
        done = len(decisions) == len(tasks) * self._max_gens
        status = "complete" if done else "stopped-not-failed"
        return {"status": status, "decisions": decisions,
                "solve_rate": rate,
                "baseline_solve_rate": base["solve_rate"],
                "cost_tried": cost, "baseline_cost": base["cost"],
                "safety_failures": bad, "gate": gate,
                "wall_s": wall, "trace": self._trace}
