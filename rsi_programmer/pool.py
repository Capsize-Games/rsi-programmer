#!/usr/bin/env python3
"""Evolution pool mechanics for the scale-up loop (#35).

Founder/child construction, per-generation scoring with
ledger lineage, and elite-plus-front child creation. The
fitness itself lives in rsi_rich; this module moves the
population, never judging it.
"""

from __future__ import annotations

from typing import Any

from rsi_programmer.evolve import POP, SIGMA
from rsi_programmer.policy import Group, SearchPolicy
from rsi_programmer.rich import fitkey4, front4

CHILD_EPOCHS = 8


def make_entry(pid: str, policy: SearchPolicy, parent: str,
               op: str) -> dict[str, Any]:
    """One lineage pool entry before fitness is known."""
    return {"id": pid, "policy": policy, "parent": parent,
            "operator": op}


def founders(seed: int, trained: SearchPolicy) -> list[Any]:
    """Null, trained, and one perturbed founder policy."""
    pop = [make_entry("p0", SearchPolicy(seed=seed), "", "null"),
           make_entry("p1", trained, "", "train")]
    child = SearchPolicy(seed=seed + 2)
    child.restore(trained.snapshot())
    child.perturb(SIGMA, seed)
    pop.append(make_entry("p2", child, "p1", "perturb"))
    return pop


def persist_entry(ledger: Any, mission: str, item: dict[str, Any],
                  gen: int, fitness: dict[str, float],
                  rich: list[float],
                  archive: list[dict[str, Any]],
                  run_id: str) -> None:
    """Persist one lineage entry through the LineageStore."""
    snap = item["policy"].snapshot()
    entry = {"policy_id": item["id"], "parent_id": item["parent"],
             "generation": gen, "operator": item["operator"],
             "fitness": fitness, "rich": list(rich),
             "weights": snap["weights"], "spec": snap["spec"],
             "run_id": run_id}
    command = "rsi-scale-%s-g%d-%s" % (run_id, gen, item["id"])
    ledger.record_intent(command,
                         mission, 0, gen + 1, entry)
    archive.append(entry)


def next_pool(order: list[Any], gen: int, groups: list[Group],
              seed: int) -> list[Any]:
    """Gradient/perturb children of the front; elite added by caller."""
    front = front4(order)
    kids: list[Any] = []
    for i in range(POP - 1):
        parent = front[i % len(front)]
        child = SearchPolicy(seed + 5000 + gen * 64 + i)
        child.restore(parent["policy"].snapshot())
        op = ("gradient", "perturb")[i % 2]
        if op == "gradient":
            child.fit(groups, CHILD_EPOCHS)
        else:
            child.perturb(SIGMA, seed + gen + i)
        kids.append(make_entry("g%d-%d" % (gen + 1, i), child,
                               parent["id"], op))
    return kids
