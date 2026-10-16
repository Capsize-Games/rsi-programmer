#!/usr/bin/env python3
"""Organization metrics the paper asks for but never implements.

Three quantities, all computable here: the autonomy ratio
(history-attributable behavioral variance over total variance, by
the law of total variance), partition divergence (whole vs
severed-graph output change over explicit TopologySpec edges), and
the agency temporal profile (per-generation convergence or
divergence from lineage archives). Expression metrics throughout:
they describe the process, never place it on any gradient.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import torch
from spikeforge.topology.builder import build_module
from spikeforge.topology.spec import TopologySpec

Tensor = torch.Tensor
Readout = Callable[[dict[str, Tensor]], Tensor]


def autonomy_ratio(histories: dict[str, list[float]]
                   ) -> dict[str, Any]:
    """Between-history variance over total behavioral variance."""
    groups = {h: [float(v) for v in vals]
              for h, vals in histories.items() if vals}
    flat = [v for vals in groups.values() for v in vals]
    if len(groups) < 2 or not flat:
        return {"ratio": 0.0, "between": 0.0, "total": 0.0,
                "histories": len(groups), "samples": len(flat)}
    mean = sum(flat) / len(flat)
    total = sum((v - mean) ** 2 for v in flat)
    between = sum(len(vals) * (sum(vals) / len(vals) - mean) ** 2
                   for vals in groups.values())
    if total <= 0:
        return {"ratio": 0.0, "between": 0.0, "total": 0.0,
                "histories": len(groups), "samples": len(flat)}
    return {"ratio": round(between / total, 4),
            "between": round(between, 4),
            "total": round(total, 4),
            "histories": len(groups), "samples": len(flat)}


def _cross_edges(spec: TopologySpec) -> tuple[list[str], list]:
    """Stages carrying cross-partition signal, and the edges."""
    doc = spec.to_dict()
    targets: dict[str, int] = {}
    for edge in doc["edges"]:
        targets[edge["target"]] = targets.get(edge["target"],
                                              0) + 1
    stages = {s["name"]: s for s in doc["stages"]}
    sever: list[str] = []
    for edge in doc["edges"]:
        if edge.get("delayed"):
            src = edge["source"]
            if stages[src].get("kind") == "linear":
                sever.append(src)
        elif targets[edge["target"]] > 1:
            kind = stages[edge["target"]].get("kind")
            if kind == "linear":
                sever.append(edge["target"])
    return sorted(set(sever)), doc["edges"]


def _run_rates(spec: TopologySpec, weights: dict[str, Tensor],
               feats: list[list[float]], steps: int,
               readout: Readout, sever: tuple[str, ...]
               ) -> list[float]:
    """Readout rates with cross edges intact or severed."""
    net = build_module(spec)
    net.load_state_dict(dict(weights))
    with torch.no_grad():
        for name, param in net.named_parameters():
            stage = name.split(".")[0]
            if stage in sever:
                param.zero_()
        mat = torch.tensor(feats)
        state = None
        acc = None
        for _ in range(steps):
            outs, state = net.step(mat, state)
            vals = readout(outs)
            acc = vals if acc is None else acc + vals
        assert acc is not None
        return [float(v) for v in acc / steps]


def integration_divergence(policy: Any,
                           feats: list[list[float]],
                           readout: Readout) -> dict[str, Any]:
    """Mean output change when cross-partition edges sever."""
    spec = policy._spec
    weights = {k: v.detach().cpu() for k, v in
               policy._net.state_dict().items()}
    sever, _ = _cross_edges(spec)
    base = int(getattr(policy, "_steps", 6))
    steps = 2 * base
    whole = _run_rates(spec, weights, feats, steps, readout,
                       ())
    if not sever:
        return {"divergence": 0.0, "severed": [],
                "reason": "feedforward chain"}
    cut = _run_rates(spec, weights, feats, steps, readout,
                     tuple(sever))
    diff = sum(abs(a - b) for a, b in zip(whole, cut))
    diff = diff / len(whole) if whole else 0.0
    return {"divergence": round(diff, 6), "severed": sever,
            "reason": "cross edges severed"}


def _flat(weights: Any) -> list[float]:
    """Flatten nested weight lists to one vector."""
    if isinstance(weights, (int, float)):
        return [float(weights)]
    if isinstance(weights, dict):
        out: list[float] = []
        for key in sorted(weights):
            out += _flat(weights[key])
        return out
    out = []
    for item in weights:
        out += _flat(item)
    return out


def _dist(left: list[float], right: list[float]) -> float:
    """Euclidean distance between two weight vectors."""
    return math.sqrt(sum((a - b) ** 2 for a, b in
                         zip(left, right)))


def agency_profile(entries: list[dict[str, Any]]
                   ) -> list[dict[str, Any]]:
    """Per-generation diversity, spread, and direction."""
    gens: dict[int, list[dict[str, Any]]] = {}
    for entry in entries:
        gens.setdefault(int(entry.get("generation", 0)),
                        []).append(entry)
    profile: list[dict[str, Any]] = []
    prev = None
    for gen in sorted(gens):
        group = gens[gen]
        vecs = [_flat(e.get("weights", [])) for e in group]
        divs = [_dist(a, b) for i, a in enumerate(vecs)
                for b in vecs[i + 1:]]
        diversity = sum(divs) / len(divs) if divs else 0.0
        fits = [float((e.get("fitness", {}) or {}).get(
            "correctness", 0.0)) for e in group]
        spread = max(fits) - min(fits) if fits else 0.0
        if prev is None:
            direction = "start"
        elif diversity < prev[0] and spread <= prev[1]:
            direction = "converging"
        elif diversity > prev[0]:
            direction = "diverging"
        else:
            direction = "stable"
        prev = (diversity, spread)
        profile.append({"generation": gen,
                        "diversity": round(diversity, 4),
                        "spread": round(spread, 4),
                        "direction": direction,
                        "members": len(group)})
    return profile
