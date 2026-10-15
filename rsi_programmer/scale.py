#!/usr/bin/env python3
"""Scale the W2 search-control policy to harder families (#35).

Retrains the W2 policy with richer search-state features
against both families, evolves more generations inside a
predeclared CPU budget, grows the Pareto archive through
LineageStore, then runs a headless acceptance sweep over
all eight tasks. Fidelity controls from rsi_rich honestly
record caps and biases in every report.
"""

from __future__ import annotations

import json
import time
from typing import Any

from rsi_programmer.lineage import mission_ledger
from rsi_programmer.evolve import OracleHook, guided_search
from rsi_programmer.policy import Group, SearchPolicy, Task
from rsi_programmer.pool import founders, make_entry, next_pool, persist_entry
from rsi_programmer.rich import RICH, bonus_for, family_pairs, fit_rich
from rsi_programmer.rich import fitkey4, fitness_of, rollup
from rsi_programmer.traces import TraceRecorder
from rsi_programmer.probe import enumerate_programs
from rsi_programmer.probe import lower_to_dag

CPU_BUDGET_S = 480.0
EVOLVE_BUDGET_S = 240.0
GENERATIONS = 3
EPOCHS = 40
MISSION = "rsi-scale"


class ScaleUp:
    """Retrain and evolve the W2 policy against harder families."""

    def __init__(self, shadow: str, hard: str, ledger: str,
                 seed: int = 7) -> None:
        """Bind both families and the lineage ledger mission."""
        self._shadow = shadow
        self._seed = seed
        self._sh_rec = TraceRecorder(shadow)
        self._ha_rec = TraceRecorder(hard)
        self._sh_tasks = self._sh_rec.load_family()["tasks"]
        self._ha_tasks = self._ha_rec.load_family()["tasks"]
        self._ledger = mission_ledger(ledger, MISSION)
        self._rich = [0.0, 0.0, 0.0, 0.0]
        self._best: SearchPolicy = SearchPolicy(seed=seed)
        self._archive: list[dict[str, Any]] = []
        self._exprs: dict[str, Any] = {}

    @staticmethod
    def _parts(task: Task) -> tuple[Any, Any, int]:
        """Train rows, target, and op bound for one task."""
        rows = tuple((c[0], c[1]) for c in task["train"])
        return rows, tuple(c[2] for c in task["train"]), int(
            task.get("max_tree_ops", 4))

    @staticmethod
    def _fam(records: list[dict[str, Any]]) -> dict[str, Any]:
        """Solve-rate, cost, and per-task numbers for one family."""
        solved = sum(1 for r in records if r["solved"])
        cost = sum(int(r["tried"]) for r in records)
        tasks = {r["task"]: {"solved": bool(r["solved"]),
                             "tried": int(r["tried"])}
                 for r in records}
        return {"solve_rate": solved / len(records), "cost": cost,
                "solved": solved, "tasks": tasks}

    def _fast_hard(self) -> list[dict[str, Any]]:
        """One enumerator pass per hard task; keep solutions."""
        records = []
        for task in self._ha_tasks:
            rows, target, bound = self._parts(task)
            start = time.perf_counter()
            expr, tried, _ = enumerate_programs(rows, target,
                                                bound)
            wall = time.perf_counter() - start
            graph = lower_to_dag(expr) if expr is not None else None
            ok = graph is not None and self._ha_rec.oracle_holds(
                task, graph)
            records.append({"task": task["id"], "solved": ok,
                            "tried": tried, "wall_s": wall})
            self._exprs[str(task["id"])] = expr
        return records

    def baselines(self) -> dict[str, Any]:
        """Recorded per-family solve-rate/cost; keep solutions."""
        shadow = self._sh_rec.record_all()
        w1 = self._fam(shadow["records"])
        hard = self._fam(self._fast_hard())
        self._base = {"w1": w1, "hard": hard}
        self._w1cost = w1["cost"] or 1
        return {"w1": w1, "hard": hard}

    def surrogate_groups(self) -> tuple[list[Group],
                                        list[Group]]:
        """W1 groups, hard surrogates, disjoint selection pairs."""
        w1 = SearchPolicy(self._seed).trace_groups(self._shadow)
        tasks = {t["id"]: t for t in self._ha_tasks}
        tries = {t["id"]: int(self._base["hard"]["tasks"][
            t["id"]]["tried"]) for t in self._ha_tasks}
        more12, morex, sel = family_pairs(tasks, self._exprs,
                                          tries, self._parts)
        self._sel = sel
        return list(w1) + more12, morex

    def _arm(self, policy: SearchPolicy) -> SearchPolicy:
        """Attach the rich-bonus rank hook to one policy."""
        rich = list(self._rich)
        policy._bonus = lambda feats: bonus_for(rich, feats)
        return policy

    def _w1_jobs(self) -> list[Any]:
        """Full guided-search jobs over the W1 family."""
        return [self._parts(t) + (self._w1cost, str(t["id"]))
                for t in self._sh_tasks]

    def _score_pop(self, pop: list[Any], gen: int) -> None:
        """Fitness and lineage for one generation in place."""
        for item in pop:
            self._arm(item["policy"])
            fit = fitness_of(item["policy"], self._w1_jobs(),
                             self._sel, self._rich, self._w1cost,
                             self._seed)
            item["fitness"] = fit
            persist_entry(self._ledger, MISSION, item, gen, fit,
                          self._rich, self._archive)

    def _advance(self, pop: list[Any], gen: int,
                 groups: list[Group]) -> list[Any]:
        """Elite plus the next front children."""
        ordered = sorted(pop, key=fitkey4, reverse=True)
        elite = make_entry("g%d-e" % (gen + 1),
                           ordered[0]["policy"],
                           ordered[0]["id"], "elite")
        return [elite] + next_pool(ordered, gen, groups,
                                   self._seed)

    def evolve(self, groups: list[Group]) -> dict[str, Any]:
        """Elitist generations inside the evolve time budget."""
        start = time.perf_counter()
        trained = SearchPolicy(seed=self._seed + 1)
        trained.fit(groups, EPOCHS)
        pop = founders(self._seed, trained)
        history = []
        gen = 0
        while True:
            self._score_pop(pop, gen)
            history.append(dict(max(pop, key=fitkey4)["fitness"]))
            over = time.perf_counter() - start >= EVOLVE_BUDGET_S
            if gen >= GENERATIONS or over:
                break
            pop = self._advance(pop, gen, groups)
            gen += 1
        self._best = max(pop, key=fitkey4)["policy"]
        stopped = time.perf_counter() - start >= EVOLVE_BUDGET_S
        return {"history": history, "generations": gen,
                "stopped": stopped}

    @staticmethod
    def _oracle(rec: TraceRecorder, task: Task) -> OracleHook:
        """Post-hoc held-out filter; never a search signal."""
        def holds(graph: dict[str, Any]) -> bool:
            return rec.oracle_holds(task, graph)
        return holds

    def _accept(self) -> dict[str, Any]:
        """Full-search headless sweep over all eight tasks."""
        jobs = [(t, self._sh_rec, "w1") for t in self._sh_tasks]
        jobs += [(t, self._ha_rec, "hard") for t in self._ha_tasks]
        per = []
        for task, rec, fam in jobs:
            rows, target, bound = self._parts(task)
            total = int(self._base[fam]["tasks"][task["id"]][
                "tried"]) or 1
            out = guided_search(self._best, rows, target, bound,
                                total, str(task["id"]),
                                oracle=self._oracle(rec, task))
            ok = out["solved"] and rec.oracle_holds(task, out["graph"])
            base = int(self._base[fam]["tasks"][task["id"]][
                "tried"])
            per.append({"task": task["id"], "family": fam,
                        "solved": out["solved"] and ok,
                        "tried": out["tried"], "base_tried": base,
                        "oracle": ok, "truncated": False})
        return {"per_task": per, "w1": rollup(per, "w1"),
                "hard": rollup(per, "hard")}

    def _consts(self, evo: dict[str, Any]) -> dict[str, Any]:
        """Predeclared budgets plus fidelity caps and biases."""
        budgets = {"cpu_budget_s": CPU_BUDGET_S,
                   "evolve_budget_s": EVOLVE_BUDGET_S,
                   "generations": GENERATIONS, "pop": 4,
                   "epochs": EPOCHS}
        fidelity = {"richer": list(RICH),
                    "rich_weights": list(self._rich),
                    "stopped_early": bool(evo["stopped"]),
                    "bias": ["no full hard search in evolution",
                             "selection shares one positive",
                             "early layers provably match-free",
                             "ties count against the solver",
                             "oracle retry is post-hoc only"]}
        return {"budgets": budgets, "fidelity": fidelity}

    def _prepare(self) -> list[Group]:
        """Surrogates plus fitted rich weights; returns groups."""
        groups, extra = self.surrogate_groups()
        self._rich = fit_rich(extra)
        return groups

    def run(self) -> dict[str, Any]:
        """Baselines, retrain, evolve, and headless acceptance."""
        start = time.perf_counter()
        before = len(self.archive())
        base = self.baselines()
        evo = self.evolve(self._prepare())
        acc = self._accept()
        after = len(self.archive())
        wall = time.perf_counter() - start
        reg = (acc["w1"]["solve_rate"], acc["hard"]["solve_rate"])
        low = (base["w1"]["solve_rate"], base["hard"]["solve_rate"])
        rep = self._consts(evo)
        rep.update({"baseline": base, "acceptance": acc,
                    "evolution": evo, "wall_s": wall,
                    "regression": reg < low,
                    "archive": {"before": before, "after": after},
                    "status": "over-budget" if wall > CPU_BUDGET_S
                    else "complete"})
        return rep

    def archive(self) -> list[dict[str, Any]]:
        """Reload lineage intents; parse the stored payloads."""
        rows = self._ledger.commands_in(MISSION, "prepared")
        return [json.loads(r["intent"]) for r in rows]

    def close(self) -> None:
        """Close the ledger handle."""
        self._ledger.close()
