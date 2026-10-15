#!/usr/bin/env python3
"""SNN search-control policy trained on W1 shadow traces (RSI W2)."""
from __future__ import annotations
import copy
from typing import Any
import torch
from spikeforge.topology.builder import build_module
from spikeforge.topology.edge import Edge
from spikeforge.topology.spec import TopologySpec
from spikeforge.topology.stage import Stage
from rsi_programmer.probe import OPS, Expr, signature
from rsi_programmer.probe import terminals
from rsi_programmer.traces import TraceRecorder
DIM = 12
HIDDEN = 16
STEPS = 6
OP_ORDER = tuple(OPS.keys())
Rows = tuple[tuple[int, int], ...]
Vals = tuple[int, ...]
Task = dict[str, Any]
Group = list[tuple[list[float], int]]
Triple = tuple[str, Expr, Expr]
def encode(op: str, left_sig: Vals, right_sig: Vals,
           target: Vals, lsize: int, rsize: int, size: int,
           bound: int, tried: int, total: int) -> list[float]:
    """Twelve search-state features for one apply choice."""
    total_n = len(target) or 1
    feats = [1.0 if op == name else 0.0 for name in OP_ORDER]
    norm = float(bound) or 1.0
    feats += [lsize / norm, rsize / norm, size / norm]
    for sig in (left_sig, right_sig):
        hit = sum(1 for a, b in zip(sig, target) if a == b)
        low, high = min(target), max(target)
        inside = sum(1 for a in sig if low <= a <= high)
        feats += [hit / total_n, inside / total_n]
    frac = tried / (total or 1)
    return feats + [frac, 1.0 - frac]
def layer_triples(levels: dict[int, list[Expr]],
             size: int) -> list[Triple]:
    """Probe-order (op, left, right) triples for one layer."""
    triples: list[Triple] = []
    for left_size in range(size):
        right_size = size - left_size - 1
        for left in levels[left_size]:
            for right in levels[right_size]:
                for op in OPS:
                    triples.append((op, left, right))
    return triples
def replay_layers(rows: Rows, expected: Vals,
                  bound: int) -> tuple[list[list[Triple]], bool]:
    """Mirror probe order; stop after the solving layer."""
    levels: dict[int, list[Expr]] = {0: list(terminals())}
    layers: list[list[Triple]] = []
    for size in range(1, bound + 1):
        triples = layer_triples(levels, size)
        seen: set[Any] = set()
        levels[size] = []
        for op, left, right in triples:
            expr = Expr(op, left, right, size)
            key = (op, signature(expr, rows))
            if key not in seen:
                seen.add(key)
                levels[size].append(expr)
        layers.append(triples)
        if any(signature(e, rows) == expected
               for e in levels[size]):
            return layers, True
    return layers, False
def _label_final(final: list[Triple], rows: Rows,
                 target: Vals, size: int, bound: int,
                 offset: int, total: int) -> Group:
    """Label the solving layer: 1 where the spec is matched."""
    group = []
    for i, (op, left, right) in enumerate(final):
        feats = encode(op, signature(left, rows),
                       signature(right, rows), target,
                       left.size, right.size, size, bound,
                       offset + i + 1, total)
        expr = Expr(op, left, right, size)
        label = 1 if signature(expr, rows) == target else 0
        group.append((feats, label))
    return group
def _clone(value: Any) -> Any:
    """Clone nested temporal state into detached tensors."""
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _clone(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone(v) for v in value)
    if isinstance(value, list):
        return [_clone(v) for v in value]
    return copy.deepcopy(value)
class SearchPolicy:
    """Spikeforge SNN mapping search states to choice rankings."""
    def __init__(self, seed: int = 0, hidden: int = HIDDEN,
                 steps: int = STEPS, lr: float = 0.05) -> None:
        """Build the TopologySpec SNN and its gradient entry."""
        torch.manual_seed(seed)
        self._seed = seed
        self._steps = steps
        self._spec = self._spec_for(hidden)
        self._net = build_module(self._spec)
        self._opt = torch.optim.Adam(self._net.parameters(),
                                     lr=lr)
        self._temporal: Any = None
    @staticmethod
    def _spec_for(hidden: int) -> TopologySpec:
        """Linear/leaky/readout topology over twelve features."""
        stages = [Stage("in_", "add", {})]
        stages.append(Stage("fc", "linear", {"in_features": DIM,
                                             "out_features": hidden}))
        stages.append(Stage("lif", "leaky", {"beta": 0.9,
                                             "threshold": 1.0}))
        stages.append(Stage("out", "linear",
                            {"in_features": hidden,
                             "out_features": 1}))
        edges = [Edge("in_", "fc"), Edge("fc", "lif")]
        spec = TopologySpec(stages, edges + [Edge("lif",
                                                  "out")],
                            "in_", "out")
        spec.validate()
        return spec
    def _state(self, batch: int) -> Any:
        """Reuse temporal state when the batch shape agrees."""
        try:
            mem = self._temporal["lif"][0]
        except (KeyError, TypeError, IndexError):
            return None
        if torch.is_tensor(mem) and mem.shape[0] == batch:
            return self._temporal
        return None
    def _forward(self, mat: torch.Tensor) -> torch.Tensor:
        """Mean readout over constant-drive steps."""
        acc = torch.zeros(mat.shape[0])
        state = self._state(mat.shape[0])
        for _ in range(self._steps):
            outs, state = self._net.step(mat, state)
            acc = acc + outs["out"][:, 0]
        self._temporal = _clone(state)
        return acc / self._steps
    def scores(self, feats: list[list[float]]) -> list[float]:
        """Rank scores from one batched SNN drive."""
        with torch.no_grad():
            out = self._forward(torch.tensor(feats))
        return [float(v) for v in out]
    def _bonus(self, feats: list[list[float]]) -> list[float]:
        """Additive per-choice bonus; scale-up assigns its own."""
        return [0.0] * len(feats)

    def rank(self, feats: list[list[float]]) -> list[int]:
        """Descending score order over choices, stable on ties."""
        scored = self.scores(feats)
        extra = self._bonus(feats)
        order = sorted(range(len(feats)))
        return sorted(order,
                      key=lambda i: scored[i] + extra[i],
                      reverse=True)
    def _epoch(self, groups: list[Any]) -> float:
        """One pairwise margin pass over final-layer groups."""
        self._opt.zero_grad()
        loss = torch.zeros(())
        for pos, neg in groups:
            if not len(pos) or not len(neg):
                continue
            gap = self._forward(pos).unsqueeze(1)
            gap = gap - self._forward(neg).unsqueeze(0)
            margin = torch.nn.functional.softplus(1.0 - gap)
            loss = loss + margin.mean()
        loss.backward()
        self._opt.step()
        return float(loss.detach())
    def fit(self, groups: list[Group],
            epochs: int = 40) -> float:
        """Gradient operator: rank solvers above the rest."""
        tensed = [(torch.tensor([f for f, y in g if y == 1]),
                   torch.tensor([f for f, y in g if y == 0]))
                  for g in groups]
        loss_val = 0.0
        for _ in range(epochs):
            loss_val = self._epoch(tensed)
        return loss_val
    def trace_groups(self, fixture: str) -> list[Group]:
        """Final-layer groups from the frozen W1 traces."""
        recorder = TraceRecorder(fixture)
        family = recorder.load_family()
        frozen = {r["task"]: r
                  for r in recorder.record_all()["records"]}
        return [self._task_group(t, frozen)
                for t in family["tasks"]]
    def _task_group(self, task: dict[str, Any],
                    frozen: dict[str, Any]) -> Group:
        """One group; triple count must match the W1 traces."""
        rows = tuple((c[0], c[1]) for c in task["train"])
        target = tuple(c[2] for c in task["train"])
        bound = int(task.get("max_tree_ops", 4))
        layers, solved = replay_layers(rows, target, bound)
        assert solved
        total = len(frozen[task["id"]]["choice_pairs"])
        assert sum(map(len, layers)) == total
        return _label_final(layers[-1], rows, target,
                            len(layers), bound,
                            total - len(layers[-1]), total)
    def train_on_traces(self, fixture: str,
                        epochs: int = 40) -> int:
        """Fit the ranking on frozen W1 traces; triple count."""
        groups = self.trace_groups(fixture)
        self.fit(groups, epochs)
        return sum(len(g) for g in groups)
    def perturb(self, sigma: float = 0.05,
                seed: int = 0) -> None:
        """Evolutionary operator: Gaussian weight noise."""
        rng = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for param in self._net.parameters():
                noise = torch.randn(param.shape, generator=rng)
                param.add_(noise * sigma)
    def weights(self) -> dict[str, Any]:
        """Detached CPU state dict of the policy network."""
        return {k: v.detach().cpu()
                for k, v in self._net.state_dict().items()}

    def snapshot(self) -> dict[str, Any]:
        """Clone weights, spec, and temporal state for turnover."""
        weights = {k: v.detach().cpu().tolist()
                   for k, v in self._net.state_dict().items()}
        return {"spec": self._spec.to_dict(),
                "weights": weights,
                "temporal": _clone(self._temporal),
                "seed": self._seed, "steps": self._steps}
    def restore(self, snap: dict[str, Any]) -> None:
        """Reload a snapshot; reuse temporal state when valid."""
        spec = TopologySpec.from_dict(snap["spec"])
        spec.validate()
        self._spec = spec
        self._net = build_module(spec)
        state = {k: torch.tensor(v)
                 for k, v in snap["weights"].items()}
        self._net.load_state_dict(state)
        self._steps = int(snap.get("steps", STEPS))
        self._temporal = _clone(snap.get("temporal"))
