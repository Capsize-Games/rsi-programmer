#!/usr/bin/env python3
"""Factored transmitting SNN: encoder, op, operand, structure heads.

One shared TaskEncoder drive fans out to four spiking populations
connected by additive edges plus a delayed feedback projection, so
modules converse across steps through spikes and membranes rather
than one scalar per choice. Combined head rates rank choices; each
head trains on its own labels and reports its own metric.
"""
from __future__ import annotations

from typing import Any

import torch
from spikeforge.topology.builder import build_module
from spikeforge.topology.edge import Edge
from spikeforge.topology.spec import TopologySpec
from spikeforge.topology.stage import Stage

from rsi_programmer.policy import DIM, OP_ORDER, STEPS, Triple
from rsi_programmer.policy import _clone

MOD_DIM = DIM + 1
MOD_HIDDEN = 16
N_OP = len(OP_ORDER)
EMIT, STOP = 0, 1


def with_gap(feats: list[list[float]],
             gap: float | None) -> torch.Tensor:
    """Append the open-gap drive channel to each feature row."""
    value = 1.0 if gap is None else float(gap)
    return torch.tensor([row + [value] for row in feats])


class ModularPolicy:
    """Four-head SNN with rank-compatible choice ordering."""

    def __init__(self, seed: int = 0, hidden: int = MOD_HIDDEN,
                 steps: int = STEPS, lr: float = 0.05) -> None:
        """Build the factored topology and its gradient entry."""
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
        """Shared encoder, four spiking heads, delayed feedback."""
        stages = [Stage("in_", "add", {})]
        stages.append(Stage("enc_fc", "linear",
                            {"in_features": MOD_DIM,
                             "out_features": hidden}))
        stages.append(Stage("enc", "leaky", {"beta": 0.9,
                                             "threshold": 1.0}))
        stages.append(Stage("op_fc", "linear",
                            {"in_features": hidden,
                             "out_features": N_OP}))
        stages.append(Stage("op", "leaky", {"beta": 0.9,
                                            "threshold": 1.0}))
        for name in ("lo", "ro"):
            stages.append(Stage(name + "_fc", "linear",
                                {"in_features": hidden,
                                 "out_features": 1}))
            stages.append(Stage(name, "leaky", {"beta": 0.9,
                                                "threshold": 1.0}))
        stages.append(Stage("st_fc", "linear",
                            {"in_features": hidden,
                             "out_features": 2}))
        stages.append(Stage("st", "leaky", {"beta": 0.9,
                                            "threshold": 1.0}))
        stages.append(Stage("fb_fc", "linear",
                            {"in_features": 2,
                             "out_features": hidden}))
        edges = [Edge("in_", "enc_fc"), Edge("enc_fc", "enc"),
                 Edge("enc", "op_fc"), Edge("op_fc", "op"),
                 Edge("enc", "lo_fc"), Edge("lo_fc", "lo"),
                 Edge("enc", "ro_fc"), Edge("ro_fc", "ro"),
                 Edge("enc", "st_fc"), Edge("st_fc", "st"),
                 Edge("st", "fb_fc"),
                 Edge("fb_fc", "enc", delayed=True)]
        spec = TopologySpec(stages, edges, "in_", "op")
        spec.validate()
        return spec

    def _state(self, batch: int) -> Any:
        """Reuse temporal state when the batch shape agrees."""
        try:
            mem = self._temporal["enc"][0]
        except (KeyError, TypeError, IndexError):
            return None
        if torch.is_tensor(mem) and mem.shape[0] == batch:
            return self._temporal
        return None

    def _drive(self, mat: torch.Tensor) -> dict[str, torch.Tensor]:
        """Mean spike rates per head over the step window."""
        acc = {k: torch.zeros(mat.shape[0], n)
               for k, n in (("op", N_OP), ("lo", 1),
                            ("ro", 1), ("st", 2))}
        state = self._state(mat.shape[0])
        for _ in range(self._steps):
            outs, state = self._net.step(mat, state)
            for key in acc:
                acc[key] = acc[key] + outs[key]
        self._temporal = _clone(state)
        return {k: v / self._steps for k, v in acc.items()}

    def head_rates(self, feats: list[list[float]],
                   gap: float | None = None) -> dict[str, list]:
        """Per-head rates without gradients, as plain lists."""
        with torch.no_grad():
            out = self._drive(with_gap(feats, gap))
        return {k: v.tolist() for k, v in out.items()}

    def scores(self, feats: list[list[float]]) -> list[float]:
        """Compat-only ranking; rank() with triples is exact."""
        with torch.no_grad():
            out = self._drive(with_gap(feats, None))
        mean_op = out["op"].mean(dim=1)
        total = mean_op + out["lo"][:, 0] + out["ro"][:, 0]
        return [float(v) for v in total + out["st"][:, EMIT]]

    def rank(self, feats: list[list[float]],
             triples: list[Triple] | None = None,
             gap: float | None = None) -> list[int]:
        """Descending combined head order, stable on ties."""
        if not feats:
            return []
        with torch.no_grad():
            out = self._drive(with_gap(feats, gap))
        order = sorted(range(len(feats)))
        if triples is None:
            scored = self.scores(feats)
            return sorted(order, key=scored.__getitem__,
                          reverse=True)
        idx = [OP_ORDER.index(op) for op, _, _ in triples]
        own = out["op"][range(len(feats)), idx]
        total = own + out["lo"][:, 0] + out["ro"][:, 0]
        total = total + out["st"][:, EMIT]
        scored = [float(v) for v in total]
        return sorted(order, key=scored.__getitem__,
                      reverse=True)

    def fit(self, groups: list[dict[str, Any]],
            epochs: int = 8) -> float:
        """One joint pass: choice margin plus per-head losses."""
        loss_val = 0.0
        for _ in range(epochs):
            self._opt.zero_grad()
            terms = [term for term in
                     (self._group_loss(group) for group in groups)
                     if term is not None]
            if not terms:
                continue
            loss = terms[0]
            for extra in terms[1:]:
                loss = loss + extra
            loss.backward()
            self._opt.step()
            loss_val = float(loss.detach())
        return loss_val

    def _group_loss(self, group: dict[str, Any]
                    ) -> torch.Tensor | None:
        """Margin on combined choice plus each head's own term."""
        if not group["pos"] or not group["neg"]:
            return None
        mat = with_gap(group["feats"], group.get("gap"))
        out = self._drive(mat)
        pos = torch.tensor(group["pos"])
        neg = torch.tensor(group["neg"])
        cm = self._combined(out, group["ops"])
        gap = cm[pos].unsqueeze(1) - cm[neg].unsqueeze(0)
        total = torch.nn.functional.softplus(1.0 - gap).mean()
        own = out["op"][range(len(group["ops"])), group["ops"]]
        ogap = own[pos].unsqueeze(1) - own[neg].unsqueeze(0)
        total = total + torch.nn.functional.softplus(
            1.0 - ogap).mean()
        for key in ("lo", "ro"):
            vec = out[key][:, 0]
            hgap = vec[pos].unsqueeze(1) - vec[neg].unsqueeze(0)
            total = total + torch.nn.functional.softplus(
                1.0 - hgap).mean()
        emit = out["st"][:, EMIT]
        egap = emit[pos].unsqueeze(1) - emit[neg].unsqueeze(0)
        total = total + torch.nn.functional.softplus(
            1.0 - egap).mean()
        return total

    def _combined(self, out: dict[str, torch.Tensor],
                  ops: list[int]) -> torch.Tensor:
        """Summed head rates behind one choice ranking."""
        own = out["op"][range(len(ops)), ops]
        total = own + out["lo"][:, 0] + out["ro"][:, 0]
        return total + out["st"][:, EMIT]

    def perturb(self, sigma: float = 0.05,
                seed: int = 0) -> None:
        """Evolutionary operator: Gaussian weight noise."""
        rng = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for param in self._net.parameters():
                noise = torch.randn(param.shape, generator=rng)
                param.add_(noise * sigma)

    def weights(self) -> dict[str, Any]:
        """Detached CPU state dict of the modular network."""
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
