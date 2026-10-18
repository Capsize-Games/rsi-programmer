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
FEATURE_ENCODING = "layer-ordinal-v1"
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
    """Encode one choice; tried/total are its layer ordinal/width.

    Search budgets and the eventual solving-layer trace length are not
    feature denominators. Callers use the candidate's generation index
    within the current layer, before any ranking or truncation.
    """
    if not target:
        raise ValueError("non-empty target required")
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
                  size: int, *, limit: int | None = None
                  ) -> list[Triple]:
    """Probe-order triples, optionally stopping after ``limit`` items."""
    triples: list[Triple] = []
    if limit is not None and limit <= 0:
        return triples
    for left_size in range(size):
        right_size = size - left_size - 1
        for left in levels[left_size]:
            for right in levels[right_size]:
                for op in OPS:
                    triples.append((op, left, right))
                    if limit is not None and len(triples) >= limit:
                        return triples
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
                       i + 1, len(final))
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
        self._rich_bonus = [0.0] * 4
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
    def _forward(self, mat: torch.Tensor) -> torch.Tensor:
        """Mean readout over steps, fresh state per drive.

        The end state is kept only for snapshot
        serialization; scoring never consumes it, so scores
        depend on features and weights alone (#52).
        """
        acc = torch.zeros(mat.shape[0])
        state: Any = None
        for _ in range(self._steps):
            outs, state = self._net.step(mat, state)
            acc = acc + outs["out"][:, 0]
        self._temporal = _clone(state)
        return acc / self._steps
    def scores(self, feats: list[list[float]]) -> list[float]:
        """Rank scores from one batched SNN drive."""
        with torch.no_grad():
            out = self._forward(torch.tensor(feats))
        extra = self._bonus(feats)
        return [float(v) + b for v, b in zip(out, extra)]
    def set_rich_bonus(self, weights: list[float]) -> None:
        """Set the four explicit, serializable classical score weights."""
        import math
        if len(weights) != 4 or not all(map(math.isfinite, weights)):
            raise ValueError("four finite bonus weights required")
        self._rich_bonus = list(map(float, weights))
    def _bonus(self, feats: list[list[float]]) -> list[float]:
        """Classical bonus shared by scores and rank, never doubled."""
        return [sum(w * x for w, x in zip(self._rich_bonus,
                    (f[5], abs(f[3] - f[4]), (f[6] + f[8]) / 2,
                     (f[7] + f[9]) / 2))) for f in feats]

    def rank(self, feats: list[list[float]],
             triples: list[Triple] | None = None,
             gap: float | None = None) -> list[int]:
        """Descending score order over choices, stable on ties."""
        scored = self.scores(feats)
        order = sorted(range(len(feats)))
        return sorted(order,
                      key=lambda i: scored[i],
                      reverse=True)
    def _epoch(self, groups: list[Any],
               contrast: list[tuple[torch.Tensor,
                                    torch.Tensor]] | None = None,
               lam: float = 1.0) -> float:
        """Margin pass plus mirror-separation penalty."""
        self._opt.zero_grad()
        loss = torch.zeros(())
        for pos, neg in groups:
            if not len(pos) or not len(neg):
                continue
            gap = self._forward(pos).unsqueeze(1)
            gap = gap - self._forward(neg).unsqueeze(0)
            margin = torch.nn.functional.softplus(1.0 - gap)
            loss = loss + margin.mean()
        for first, second in contrast or []:
            vec = self._forward(first)
            other = self._forward(second)
            denom = torch.sqrt((vec * vec).sum() *
                               (other * other).sum())
            cos = (vec * other).sum() / (denom + 1e-8)
            band = torch.relu(cos - 0.3) + \
                torch.relu(0.1 - cos)
            loss = loss + lam * band
        loss.backward()
        self._opt.step()
        return float(loss.detach())
    def fit(self, groups: list[Group],
            epochs: int = 40, *,
            contrast: list[tuple[torch.Tensor,
                                 torch.Tensor]] | None = None,
            lam: float = 1.0) -> float:
        """Gradient operator: rank solvers above the rest."""
        tensed = [(torch.tensor([f for f, y in g if y == 1]),
                   torch.tensor([f for f, y in g if y == 0]))
                  for g in groups]
        loss_val = 0.0
        for _ in range(epochs):
            loss_val = self._epoch(tensed, contrast, lam)
        return loss_val
    def trace_groups(self, fixture: str) -> list[Group]:
        """Final-layer groups from the frozen W1 traces."""
        return list(self.trace_groups_by_task(fixture).values())

    def trace_groups_by_task(self, fixture: str) -> dict[str, Group]:
        """Final-layer groups keyed by their frozen task identifiers."""
        recorder = TraceRecorder(fixture)
        family = recorder.load_family()
        records = recorder.record_all()["records"]
        frozen = {str(r["task"]): r for r in records}
        tasks = family["tasks"]
        task_ids = [str(task["id"]) for task in tasks]
        if (len(task_ids) != len(set(task_ids))
                or len(frozen) != len(records)
                or set(task_ids) != set(frozen)):
            raise ValueError("trace tasks and records must match one-to-one")
        return {str(task["id"]): self._task_group(task, frozen)
                for task in tasks}
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
                "seed": self._seed, "steps": self._steps,
                "optimizer": self._optimizer_options(),
                "feature_encoding": FEATURE_ENCODING,
                "rich_bonus": list(self._rich_bonus)}
    def _optimizer_options(self) -> dict[str, Any]:
        """Clone supported Adam options without parameter references."""
        group = self._opt.param_groups[0]
        return {key: copy.deepcopy(group.get(key, value))
                for key, value in self._opt.defaults.items()}

    def restore(self, snap: dict[str, Any], *,
                optimizer_from_snapshot: bool = False) -> None:
        """Restore weights with a fresh recipient-configured optimizer.

        This clones a policy for a child; it is not an Adam checkpoint
        resume. Recipient Adam options are preserved and moments reset.
        Set optimizer_from_snapshot to opt into donor options. Untagged
        legacy snapshots load, but their feature contract is unverified.
        """
        encoding = snap.get("feature_encoding", FEATURE_ENCODING)
        if encoding != FEATURE_ENCODING:
            raise ValueError("incompatible feature encoding")
        options = self._optimizer_options()
        if optimizer_from_snapshot:
            options.update(snap.get("optimizer", {}))
        spec = TopologySpec.from_dict(snap["spec"])
        spec.validate()
        self._spec = spec
        self._net = build_module(spec)
        state = {k: torch.tensor(v)
                 for k, v in snap["weights"].items()}
        self._net.load_state_dict(state)
        self._opt = torch.optim.Adam(self._net.parameters(), **options)
        self._steps = int(snap.get("steps", STEPS))
        self._temporal = _clone(snap.get("temporal"))
        self.set_rich_bonus(snap.get("rich_bonus", [0.0] * 4))
