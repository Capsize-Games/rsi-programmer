#!/usr/bin/env python3
"""Frozen-policy structural-OOD expression benchmark (#12).

Trains nothing here: two policies arrive pre-trained (one on the
train family only, one on train plus novel), get evaluated frozen
on the mirrored OOD family with weights write-locked, and the
report compares solve-rates plus tried-cost against an
enumerator-from-scratch baseline. Expression-only throughout:
no promotion, no placement, no gradient claims.
"""
from __future__ import annotations

import hashlib
from typing import Any

import torch

from rsi_programmer.author import beam_search, refine_author
from rsi_programmer.policy import SearchPolicy, encode
from rsi_programmer.policy import layer_triples
from rsi_programmer.probe import CONSTS, enumerate_programs
from rsi_programmer.probe import signature, terminals
from rsi_programmer.traces import TraceRecorder

OOD_FIXTURE = "fixtures/rsi_ood_tasks.json"

CAVEAT = ("v2.4.0 section 9.1: behavioral adaptation alone is "
          "neither the definition of intelligence nor sufficient "
          "demonstration of it. This report measures expression "
          "only and places nothing on any gradient.")


def weights_hash(policy: SearchPolicy) -> str:
    """SHA-256 over sorted policy weight bytes."""
    digest = hashlib.sha256()
    state = policy._net.state_dict()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(bytes(state[key].detach().cpu().numpy()))
    return digest.hexdigest()


def assert_frozen(before: str, policy: SearchPolicy) -> None:
    """Abort when evaluation moved the locked weights."""
    after = weights_hash(policy)
    if after != before:
        raise RuntimeError("frozen weights moved during OOD eval")


def _parts(task: dict[str, Any]) -> tuple[Any, Any, int]:
    """Train rows, target outputs, and op bound for one task."""
    rows = tuple((c[0], c[1]) for c in task["train"])
    return rows, tuple(c[2] for c in task["train"]), int(
        task.get("max_tree_ops", 4))


def enumerator_cost(task: dict[str, Any]) -> int:
    """From-scratch exhaustive tried-count for one OOD task."""
    rows, target, bound = _parts(task)
    feats = list(task.get("features", []))
    _, tried, _ = enumerate_programs(
        rows, target, bound,
        consts=CONSTS if "const" in feats else (),
        branch="branch" in feats)
    return int(tried)


def evaluate(policy: SearchPolicy, tasks: list[dict[str, Any]],
             recorder: TraceRecorder, budget: int, *,
             order: str = "fifo", widen: int = 0,
             weights: list[float] | None = None
             ) -> list[dict[str, Any]]:
    """Frozen beam sweep; solved needs train + held-out hold."""
    per: list[dict[str, Any]] = []
    for task in tasks:
        rows, target, bound = _parts(task)
        out = beam_search(policy, rows, target, bound,
                          budget, str(task["id"]), fallback=False,
                          order=order, widen=widen,
                          weights=weights)
        holds = out["solved"] and recorder.oracle_holds(
            task, out["graph"])
        per.append({"task": str(task["id"]),
                    "solved": bool(out["solved"] and holds),
                    "tried": int(out["tried"]),
                    "oracle": bool(holds),
                    "path": str(out.get("path", "beam"))})
    return per


def _probe_layer() -> tuple[list[Any], int]:
    """Size-1 triple layer over terminals, plus its count."""
    levels = {0: list(terminals())}
    triples = layer_triples(levels, 1)
    return triples, len(triples) or 1


def _probe_feats(task: dict[str, Any], triples: list[Any],
                 total: int) -> list[list[float]]:
    """Encode one task's probe layer against its own target."""
    rows = tuple((c[0], c[1]) for c in task["train"])
    target = tuple(c[2] for c in task["train"])
    bound = int(task.get("max_tree_ops", 4))
    return [encode(op, signature(left, rows),
                   signature(right, rows), target,
                   left.size, right.size, 1, bound, 0, total)
            for op, left, right in triples]


def contrast_tensors(twin: list[dict[str, Any]],
                     shifted: list[dict[str, Any]]
                     ) -> list[tuple[torch.Tensor,
                                     torch.Tensor]]:
    """Aligned probe-layer tensors for contrastive training."""
    triples, total = _probe_layer()
    return [(torch.tensor(_probe_feats(ref, triples, total)),
             torch.tensor(_probe_feats(alt, triples, total)))
            for ref, alt in zip(twin, shifted)]


def guidance_similarity(policy: SearchPolicy,
                        twin: list[dict[str, Any]],
                        shifted: list[dict[str, Any]]
                        ) -> dict[str, Any]:
    """Cosine of guidance scores across row-aligned pairs."""
    triples, total = _probe_layer()
    sims: list[float] = []
    for ref, alt in zip(twin, shifted):
        vecs = []
        for task in (ref, alt):
            feats = _probe_feats(task, triples, total)
            vecs.append(policy.scores(feats))
        num = sum(a * b for a, b in zip(vecs[0], vecs[1]))
        left = sum(a * a for a in vecs[0]) ** 0.5
        right = sum(b * b for b in vecs[1]) ** 0.5
        sims.append(num / (left * right)
                    if left > 0 and right > 0 else 0.0)
    return {"mean": round(sum(sims) / len(sims), 4) if sims
            else 0.0,
            "per_task": [round(s, 4) for s in sims],
            "pairs": len(sims)}


def _refine_rows(tasks: list[dict[str, Any]]
                ) -> list[tuple[str, Any, Any, int]]:
    """Refinement inputs: train rows only, never held-out."""
    return [(str(task["id"]),) + _parts(task) for task in tasks]


def _refine_locked(policy: SearchPolicy,
                   tasks: list[dict[str, Any]]) -> list[float]:
    """EXTRA refinement on train rows; returns locked weights."""
    weights, _ = refine_author(policy, _refine_rows(tasks),
                               seed=0)
    return weights


def benchmark(narrow: SearchPolicy, broad: SearchPolicy,
              tasks: list[dict[str, Any]], budget: int,
              name: str = "ood", order: str = "fifo",
              widen: int = 0, refine: bool = False
              ) -> dict[str, Any]:
    """Frozen OOD runs plus enumerator baseline; prints gap."""
    recorder = TraceRecorder(OOD_FIXTURE)
    extras = {}
    if refine:
        extras["narrow"] = _refine_locked(narrow, tasks)
        extras["broad"] = _refine_locked(broad, tasks)
    locked = {name: weights_hash(pol)
              for name, pol in (("narrow", narrow),
                                ("broad", broad))}
    narrow_per = evaluate(narrow, tasks, recorder, budget,
                          order=order, widen=widen,
                          weights=extras.get("narrow"))
    assert_frozen(locked["narrow"], narrow)
    broad_per = evaluate(broad, tasks, recorder, budget,
                         order=order, widen=widen,
                         weights=extras.get("broad"))
    assert_frozen(locked["broad"], broad)
    base = [enumerator_cost(task) for task in tasks]
    report = {"name": name, "tasks": len(tasks),
              "budget": budget, "refined": refine,
              "narrow_solve": round(
                  sum(1 for r in narrow_per if r["solved"]) /
                  len(tasks), 4),
              "broad_solve": round(
                  sum(1 for r in broad_per if r["solved"]) /
                  len(tasks), 4),
              "mean_enumerator_tried": round(
                  sum(base) / len(base), 1),
              "mean_narrow_tried": round(
                  sum(r["tried"] for r in narrow_per) /
                  len(narrow_per), 1),
              "mean_broad_tried": round(
                  sum(r["tried"] for r in broad_per) /
                  len(broad_per), 1),
              "per_task": [{"task": n["task"],
                            "narrow": n["solved"],
                            "broad": b["solved"],
                            "enumerator_tried": e}
                           for n, b, e in
                           zip(narrow_per, broad_per, base)],
              "weights_unchanged": True, "caveat": CAVEAT}
    report["expression_gap"] = round(
        report["broad_solve"] - report["narrow_solve"], 4)
    print("%s: narrow=%.3f broad=%.3f gap=%+.3f "
          "enum_tried=%.0f" % (name, report["narrow_solve"],
                               report["broad_solve"],
                               report["expression_gap"],
                               report["mean_enumerator_tried"]))
    print(CAVEAT)
    return report
