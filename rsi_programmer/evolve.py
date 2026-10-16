#!/usr/bin/env python3
"""Evolutionary policy loop with Pareto archive (RSI W2)."""

from __future__ import annotations
import json
from typing import Any, Callable
from rsi_programmer.lineage import mission_ledger
from rsi_programmer.probe import OPS, Expr, signature
from rsi_programmer.probe import terminals
from rsi_programmer.actions import ActionSpace
from rsi_programmer.policy import Rows, SearchPolicy, Task, Triple, Vals
from rsi_programmer.policy import encode, layer_triples
from rsi_programmer.traces import TraceRecorder
COMPS = ("correctness", "efficiency", "recovery")
POP = 4
SIGMA = 0.1
OracleHook = Callable[[dict[str, Any]], bool]
def _dom(a: dict[str, float], b: dict[str, float]) -> bool:
    return all(a[k] >= b[k] for k in COMPS) and any(
        a[k] > b[k] for k in COMPS)
def _fitkey(entry: dict[str, Any]) -> tuple[float, ...]:
    fit = entry["fitness"]
    return (fit["correctness"], fit["efficiency"],
            fit["recovery"])
def _front(pop: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in pop if not any(
        _dom(o["fitness"], e["fitness"]) for o in pop
        if o is not e)]
def _new_ctx() -> dict[str, Any]:
    levels: dict[int, list[Expr]] = {0: list(terminals())}
    refs = {id(term): term.left for term in levels[0]}
    graph: dict[str, Any] = {"input_types": {"x": "Int",
                                             "y": "Int"},
                             "nodes": [], "output": "x"}
    return {"space": ActionSpace(), "graph": graph, "levels": levels,
            "refs": refs, "interned": {}, "tried": 0, "rejected": []}
def _intern(ctx: dict[str, Any], rows: Rows,
            triples: list[Triple], size: int) -> list[Vals]:
    graph, refs = ctx["graph"], ctx["refs"]
    sigs = []
    for op, left, right in triples:
        expr = Expr(op, left, right, size)
        sigs.append(signature(expr, rows))
        key = (op, refs[id(left)], refs[id(right)])
        if key not in ctx["interned"]:
            node = {"args": [key[1], key[2]], "op": op}
            node["type"] = "Int"
            graph["nodes"].append(node)
            ctx["interned"][key] = len(graph["nodes"]) - 1
    return sigs
def _members(ctx: dict[str, Any], triples: list[Triple],
             sigs: list[Vals], size: int) -> None:
    seen: set[Any] = set()
    ctx["levels"][size] = []
    for (op, left, right), sig in zip(triples, sigs):
        if (op, sig) not in seen:
            seen.add((op, sig))
            member = Expr(op, left, right, size)
            key = (op, ctx["refs"][id(left)], ctx["refs"][id(right)])
            ctx["refs"][id(member)] = ctx["interned"][key]
            ctx["levels"][size].append(member)
def _oracle_kept(ctx: dict[str, Any], found: Any, sig: Vals,
                 task_id: str, oracle: OracleHook | None) -> bool:
    """Keep a train match the post-hoc oracle accepts."""
    if oracle is None:
        return True
    got = _accept(ctx, found)
    if got["graph"] is not None and oracle(got["graph"]):
        return True
    ctx["rejected"].append({"task": task_id, "op": found["op"],
                            "verdict": got["verdict"] or
                            "oracle-rejected",
                            "signature": list(sig)})
    return False


def _scan(ctx: dict[str, Any], triples: list[Triple],
          sigs: list[Vals], order: list[int], target: Vals,
          task_id: str, oracle: OracleHook | None = None) -> Any:
    for idx in order:
        op, left, right = triples[idx]
        ctx["tried"] += 1
        if sigs[idx] != target:
            ctx["rejected"].append({"task": task_id, "op": op,
                                    "verdict":
                                    "signature-mismatch"})
            continue
        found = {"kind": "apply", "op": op,
                 "left": ctx["refs"][id(left)],
                 "right": ctx["refs"][id(right)]}
        if _oracle_kept(ctx, found, sigs[idx], task_id, oracle):
            return found
    return None
def _layer_gap(sigs: list[Vals], target: Vals) -> float:
    """Open fraction: one minus the best row-match rate."""
    total = len(target) or 1
    best = max((sum(1 for a, b in zip(sig, target) if a == b)
                for sig in sigs), default=0)
    return 1.0 - best / total


def _step(policy: SearchPolicy, ctx: dict[str, Any],
          rows: Rows, target: Vals, size: int, bound: int,
          budget: int, task_id: str, oracle: OracleHook | None
          ) -> Any:
    triples = layer_triples(ctx["levels"], size)
    sigs = _intern(ctx, rows, triples, size)
    _members(ctx, triples, sigs, size)
    feats = [encode(op, signature(left, rows),
                    signature(right, rows), target,
                    left.size, right.size, size, bound,
                    ctx["tried"] + i + 1, budget)
             for i, (op, left, right) in enumerate(triples)]
    gap = _layer_gap(sigs, target)
    return _scan(ctx, triples, sigs, policy.rank(feats, triples,
                                                gap),
                 target, task_id, oracle)
def _accept(ctx: dict[str, Any], found: Any) -> dict[str, Any]:
    try:
        graph = ctx["space"].apply(ctx["graph"], found)
    except ValueError as exc:
        return {"solved": False, "tried": ctx["tried"],
                "graph": None, "rejected": ctx["rejected"],
                "verdict": "validator: " + str(exc)}
    return {"solved": True, "tried": ctx["tried"],
            "graph": graph, "rejected": ctx["rejected"],
            "verdict": ""}
def guided_search(policy: SearchPolicy, rows: Rows,
                  expected: Vals, bound: int, budget: int,
                  task_id: str = "",
                  oracle: OracleHook | None = None) -> dict[str, Any]:
    """Policy-ranked search; oracle vetoes retry past near-misses."""
    ctx = _new_ctx()
    for size in range(1, bound + 1):
        found = _step(policy, ctx, rows, expected, size, bound,
                      budget, task_id, oracle)
        if found is not None:
            return _accept(ctx, found)
    return {"solved": False, "tried": ctx["tried"],
            "graph": None, "rejected": ctx["rejected"],
            "verdict": "budget-exhausted"}
class PolicyLoop:
    """Evolve the policy; Pareto archive on LineageStore."""
    def __init__(self, fixture: str, ledger: str,
                 seed: int = 0, *,
                 screen: Any = None, stop_after: int = 2
                 ) -> None:
        """Bind the frozen family, baseline, and intent log."""
        self._fixture = fixture
        self._screen = screen
        self._stop_after = stop_after
        self._recorder = TraceRecorder(fixture)
        self._tasks = self._recorder.load_family()["tasks"]
        base = self._recorder.record_all()["records"]
        self._base = {r["task"]: r for r in base}
        self._budget = sum(r["tried"] for r in base)
        self._seed = seed
        self._ledger = mission_ledger(ledger, "rsi-w2")
        self._archive: list[dict[str, Any]] = []
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
        if not out["solved"]:
            return 0.0
        return max(0.0, 1.0 - (out["tried"] - cost) /
                   self._budget)
    def _score_task(self, policy: SearchPolicy,
                    task: dict[str, Any]) -> dict[str, Any]:
        rows, target, bound = self._rows(task, "train")
        out = guided_search(policy, rows, target, bound,
                            self._budget, str(task["id"]))
        if not out["solved"]:
            return {"solved": False, "cost": out["tried"],
                    "gate": out["verdict"] or "miss", "recovery": 0.0}
        if not self._recorder.oracle_holds(task, out["graph"]):
            return {"solved": False, "cost": out["tried"],
                    "gate": "oracle", "recovery": 0.0}
        return {"solved": True, "cost": out["tried"], "gate": "",
                "recovery": self._recovery(policy, task, out["tried"])}
    def evaluate(self, policy: SearchPolicy) -> dict[str, Any]:
        """Gated correctness/efficiency/recovery; no scalar."""
        parts = [self._score_task(policy, t)
                 for t in self._tasks]
        gate = next((p["gate"] for p in parts if p["gate"]),
                    "")
        cost = sum(p["cost"] for p in parts)
        if gate or cost > self._budget:
            return {"gates_pass": False, "gate": gate or "over-budget",
                    "correctness": 0.0, "efficiency": 0.0, "recovery": 0.0}
        solved = sum(1 for p in parts if p["solved"])
        rec = sum(p["recovery"] for p in parts) / len(parts)
        return {"gates_pass": True, "gate": "",
                "correctness": solved / len(parts), "recovery": rec,
                "efficiency": max(0.0, 1.0 - cost / self._budget)}
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
                 fitness: dict[str, Any]) -> None:
        snap = item["policy"].snapshot()
        entry = {"policy_id": item["id"],
                 "parent_id": item["parent"], "generation": gen,
                 "operator": item["operator"], "fitness": fitness,
                 "screen_skipped": bool(item.get("screen_skipped",
                                                False)),
                 "weights": snap["weights"], "spec": snap["spec"]}
        cmd = "rsi-w2-g%d-%s" % (gen, item["id"])
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
                item["fitness"] = dict(
                    by_id[item["parent"]]["fitness"])
                item["screen_skipped"] = True
                self._persist(item, gen, item["fitness"])
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
        groups = SearchPolicy(self._seed).trace_groups(
            self._fixture)
        pop = self._founders(groups)
        history = []
        screened: list[Any] = []
        parents = None
        for gen in range(generations + 1):
            screened.append(self._evaluate_pop(pop, parents,
                                               gen))
            history.append(dict(max(pop, key=_fitkey)["fitness"]))
            parents = pop
            if gen < generations:
                pop = self._next_pop(pop, gen, groups)
        return {"history": history, "archive": self._archive,
                "screened": screened}
    def close(self) -> None:
        """Close the ledger handle."""
        self._ledger.close()
