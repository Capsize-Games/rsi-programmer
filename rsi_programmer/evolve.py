#!/usr/bin/env python3
"""Evolutionary policy loop with Pareto archive (RSI W2)."""

from __future__ import annotations
import json
import uuid
from typing import Any
from rsi_programmer.lineage import mission_ledger
from rsi_programmer.policy import Rows, SearchPolicy, Task, Vals
from rsi_programmer.search import OracleHook as OracleHook
from rsi_programmer.search import guided_search
from rsi_programmer.traces import TraceRecorder
COMPS = ("correctness", "efficiency", "recovery")
POP = 4
SIGMA = 0.1
PROBE_TASKS = 2
PROBE_BUDGET = 2000
def _dom(a: dict[str, float], b: dict[str, float]) -> bool:
    return all(a[k] >= b[k] for k in COMPS) and any(
        a[k] > b[k] for k in COMPS)
def _fitkey(entry: dict[str, Any]) -> tuple[float, ...]:
    if entry.get("screen_skipped", False):
        return (float("-inf"),) * 4
    fit = entry["fitness"]
    return (fit["correctness"], fit.get("refinability", 0.0),
            fit["efficiency"], fit["recovery"])

def _search_cost(result: dict[str, Any]) -> int:
    """Canonical candidate-signature work, with legacy stub fallback."""
    value = result.get("signature_evaluations")
    if value is None:
        value = result["tried"]
    return int(value)

def _baseline_cost(record: dict[str, Any]) -> int:
    """Read canonical baseline work, accepting older trace records."""
    value = record.get("signature_evaluations")
    if value is None:
        value = record["tried"]
    return int(value)
def _front(pop: list[dict[str, Any]]) -> list[dict[str, Any]]:
    measured = [e for e in pop if not e.get("screen_skipped", False)]
    return [e for e in measured if not any(
        _dom(o["fitness"], e["fitness"]) for o in pop
        if o is not e and not o.get("screen_skipped", False))]
class PolicyLoop:
    """Evolve the policy; Pareto archive on LineageStore."""
    def __init__(self, fixture: str, ledger: str,
                 seed: int = 0, *,
                 screen: Any = None, stop_after: int = 2,
                 probe_rounds: int = 0
                 ) -> None:
        """Bind the frozen family, baseline, and intent log."""
        self._fixture = fixture
        self._screen = screen
        self._stop_after = stop_after
        self._probe_rounds = probe_rounds
        self._recorder = TraceRecorder(fixture)
        self._tasks = self._recorder.load_family()["tasks"]
        base = self._recorder.record_all()["records"]
        self._base = {r["task"]: r for r in base}
        self._budget = sum(_baseline_cost(record) for record in base)
        self._seed = seed
        self._ledger = mission_ledger(ledger, "rsi-w2")
        self._archive: list[dict[str, Any]] = []
        self._run_id = uuid.uuid4().hex
    def _rows(self, task: Task, part: str) -> tuple[Rows, Vals, int]:
        rows = tuple((c[0], c[1]) for c in task[part])
        target = tuple(c[2] for c in task[part])
        return rows, target, int(task.get("max_tree_ops", 4))
    def _recovery(self, policy: SearchPolicy,
                  task: dict[str, Any], cost: int) -> float:
        noisy = SearchPolicy(seed=self._seed)
        noisy.restore(policy.snapshot())
        noisy.perturb(SIGMA, self._seed)
        rows, target, bound = self._rows(task, "train")
        out = guided_search(noisy, rows, target, bound,
                            self._budget, str(task["id"]))
        if (not out["solved"] or
                not self._recorder.oracle_holds(task, out["graph"])):
            return 0.0
        baseline = _baseline_cost(self._base[str(task["id"])])
        return min(1.0, max(0.0, 1.0 -
                            (_search_cost(out) - cost) /
                            max(1, baseline)))
    def _score_task(self, policy: SearchPolicy,
                    task: dict[str, Any]) -> dict[str, Any]:
        rows, target, bound = self._rows(task, "train")
        out = guided_search(policy, rows, target, bound,
                            self._budget, str(task["id"]))
        cost = _search_cost(out)
        if not out["solved"]:
            return {"solved": False, "cost": cost,
                    "gate": out["verdict"] or "miss", "recovery": 0.0}
        if not self._recorder.oracle_holds(task, out["graph"]):
            return {"solved": False, "cost": cost,
                    "gate": "oracle", "recovery": 0.0}
        return {"solved": True, "cost": cost, "gate": "",
                "recovery": self._recovery(policy, task, cost)}
    def evaluate(self, policy: SearchPolicy) -> dict[str, Any]:
        """Gated correctness/refinability/efficiency/recovery.

        Correctness dominates; refinability (EXTRA-refined
        author solves on the fixed probe tasks) is the second
        objective and tiebreak; efficiency then recovery.
        Pareto dominance stays on the original three comps.
        """
        parts = [self._score_task(policy, t)
                 for t in self._tasks]
        gate = next((p["gate"] for p in parts if p["gate"]),
                    "")
        cost = sum(p["cost"] for p in parts)
        solved = sum(1 for p in parts if p["solved"])
        measured = {"correctness": solved / len(parts),
                    "efficiency": max(0.0, 1.0 - cost /
                                      max(1, self._budget)),
                    "recovery": sum(p["recovery"] for p in parts) /
                                len(parts),
                    "signature_evaluations": cost}
        if gate or cost > self._budget:
            return {"gates_pass": False, "gate": gate or "over-budget",
                    "correctness": 0.0, "efficiency": 0.0, "recovery": 0.0,
                    "refinability": 0.0, "measured": measured}
        return {"gates_pass": True, "gate": "",
                "correctness": measured["correctness"],
                "recovery": measured["recovery"],
                "refinability": self._probe(policy),
                "efficiency": measured["efficiency"],
                "measured": measured}

    def _probe_rows(self) -> list[Any]:
        """Fixed probe set: first tasks as author TaskRows."""
        rows = []
        for task in self._tasks[:PROBE_TASKS]:
            rows.append((str(task["id"]),
                         tuple((c[0], c[1]) for c in task["train"]),
                         tuple(c[2] for c in task["train"]),
                         int(task.get("max_tree_ops", 4))))
        return rows

    def _probe(self, policy: SearchPolicy) -> float:
        """EXTRA-refined author solves share on the probe set."""
        if self._probe_rounds <= 0:
            return 0.0
        from rsi_programmer.author import refine_author
        rows = self._probe_rows()
        _, report = refine_author(policy, rows, seed=self._seed,
                                 rounds=self._probe_rounds,
                                 budget=PROBE_BUDGET)
        return report["solved"] / len(rows)
    def _founders(self, groups: list[Any]
                  ) -> list[dict[str, Any]]:
        from rsi_programmer.author import train_author
        trained = SearchPolicy(seed=self._seed + 1)
        trained.fit(groups, 40)
        authored = SearchPolicy(seed=self._seed + 100)
        train_author(authored, self._fixture, epochs=40)
        pop = [{"id": "p0", "policy": SearchPolicy(
            seed=self._seed), "parent": "", "operator":
            "null"}, {"id": "p1", "policy": trained,
                      "parent": "", "operator": "train"},
               {"id": "p2", "policy": authored, "parent": "",
                "operator": "author"}]
        for i in range(POP - 3):
            child = SearchPolicy(seed=self._seed + 2 + i)
            child.restore(trained.snapshot())
            child.perturb(SIGMA, self._seed + i)
            pop.append({"id": "p%d" % (i + 3), "policy": child,
                        "parent": "p1", "operator": "perturb"})
        return pop
    def _spawn(self, parent: dict[str, Any], i: int, gen: int,
               groups: list[Any]) -> dict[str, Any]:
        child = SearchPolicy(self._seed + 5000 + gen * 64 + i)
        child.restore(parent["policy"].snapshot())
        op = ("gradient", "perturb")[i % 2]
        if op == "gradient":
            child.fit(groups, 8)
        else:
            child.perturb(SIGMA, self._seed + gen + i)
        return {"id": "g%d-%d" % (gen + 1, i), "policy": child,
                "parent": parent["id"], "operator": op}
    def _next_pop(self, pop: list[dict[str, Any]], gen: int,
                  groups: list[Any]) -> list[dict[str, Any]]:
        order = sorted(pop, key=_fitkey, reverse=True)
        front = _front(order)
        best = order[0]
        kids = [{"id": "g%d-e" % (gen + 1),
                 "policy": best["policy"],
                 "parent": best["id"], "operator": "elite"}]
        for i in range(POP - 1):
            kids.append(self._spawn(front[i % len(front)], i,
                                    gen, groups))
        return kids
    def _persist(self, item: dict[str, Any], gen: int,
                 fitness: dict[str, Any] | None) -> None:
        snap = item["policy"].snapshot()
        entry = {"policy_id": item["id"],
                 "parent_id": item["parent"], "generation": gen,
                 "operator": item["operator"], "fitness": fitness,
                 "measured": fitness is not None,
                 "inherited_fitness": item.get("inherited_fitness"),
                 "screen_skipped": bool(item.get("screen_skipped",
                                                False)),
                 "weights": snap["weights"], "spec": snap["spec"],
                 "run_id": self._run_id}
        cmd = "rsi-w2-%s-g%d-%s" % (self._run_id, gen, item["id"])
        self._ledger.record_intent(cmd, "rsi-w2", 0, gen + 1,
                                   entry)
        self._archive.append(entry)
    def load_archive(self) -> list[dict[str, Any]]:
        """Reload lineage intents; parse the stored payloads."""
        rows = self._ledger.commands_in("rsi-w2", "prepared")
        return [json.loads(r["intent"]) for r in rows]

    def _evaluate_pop(self, pop: list[dict[str, Any]],
                      parents: list[dict[str, Any]] | None,
                      gen: int) -> dict[str, Any] | None:
        """Full eval in screen order; decisive top picks skip."""
        if self._screen is None:
            order, scored = list(pop), None
        else:
            scored = self._screen.rank_policies(
                {i["id"]: i["policy"] for i in pop})
            order = sorted(pop, key=lambda i: scored[i["id"]],
                           reverse=True)
        by_id = {i["id"]: i for i in (parents or [])}
        early: list[bool] = []
        for item in order:
            decisive = self._screen is not None and \
                self._stop_after > 0 and \
                len(early) >= self._stop_after and all(early)
            if decisive and parents is not None:
                item["inherited_fitness"] = dict(
                    by_id[item["parent"]]["fitness"])
                item["fitness"] = None
                item["screen_skipped"] = True
                self._persist(item, gen, None)
                continue
            fitness = self.evaluate(item["policy"])
            item["fitness"] = fitness
            item["screen_skipped"] = False
            self._persist(item, gen, fitness)
            if len(early) < self._stop_after:
                early.append(bool(fitness.get("gates_pass")))
        if scored is None:
            return None
        full = {i["id"]: i["fitness"]["correctness"]
                for i in order if not i["screen_skipped"]}
        pact = self._screen.fidelity(scored, full)
        return {"order": [i["id"] for i in order],
                "skipped": [i["id"] for i in order
                             if i["screen_skipped"]],
                "pact": pact}

    def run(self, generations: int = 2) -> dict[str, Any]:
        """Elitist loop; every entry persists to the ledger."""
        self._run_id = uuid.uuid4().hex
        groups = SearchPolicy(self._seed).trace_groups(
            self._fixture)
        pop = self._founders(groups)
        history = []
        screened: list[Any] = []
        parents = None
        for gen in range(generations + 1):
            screened.append(self._evaluate_pop(pop, parents,
                                               gen))
            measured = [item for item in pop
                        if not item.get("screen_skipped", False)]
            history.append(dict(max(measured, key=_fitkey)["fitness"]))
            parents = pop
            if gen < generations:
                pop = self._next_pop(pop, gen, groups)
        return {"history": history, "archive": self._archive,
                "screened": screened}
    def close(self) -> None:
        """Close the ledger handle."""
        self._ledger.close()
