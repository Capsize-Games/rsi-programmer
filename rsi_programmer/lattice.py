#!/usr/bin/env python3
"""Bit-slice lattice: frozen shared encoder, four 1-bit heads.

Head i drives bit i (weight 2**i) of every 4-bit word; the
shared EMIT/STOP struct gate commits words on EMIT-majority.
Encoder, struct, and feedback stages load from a trained
modular snapshot by stage-name prefix; bit heads train fresh
in v32 with the encoder write-locked.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as func
from spikeforge.topology.builder import build_module
from spikeforge.topology.edge import Edge
from spikeforge.topology.spec import TopologySpec
from spikeforge.topology.stage import Stage

from rsi_programmer.author import contrastive_groups
from rsi_programmer.lineage import mission_ledger
from rsi_programmer.modules import MOD_DIM, with_gap
from rsi_programmer.policy import OP_ORDER, STEPS, _clone
from rsi_programmer.words import OPCODES, pack, unpack

N_BITS = 4
EMIT, STOP = 0, 1
ENCODER_PREFIXES = ("enc_fc", "enc", "st_fc", "st", "fb_fc")
HEAD_PREFIXES = ("b0", "b1", "b2", "b3", "st")


def _head_stages(name: str, hidden: int) -> list[Stage]:
    """One linear plus leaky spiking head stage pair."""
    return [Stage(name + "_fc", "linear",
                  {"in_features": hidden, "out_features": 1}),
            Stage(name, "leaky", {"beta": 0.9, "threshold": 1.0})]


class BitLattice:
    """Four 1-bit SNN heads over one shared encoder."""

    def __init__(self, seed: int = 0, hidden: int = 16,
                 steps: int = STEPS, lr: float = 0.05) -> None:
        """Build the lattice topology and its gradient entry."""
        torch.manual_seed(seed)
        self._seed = seed
        self._steps = steps
        self._hidden = hidden
        self._spec = self._spec_for(hidden)
        self._net = build_module(self._spec)
        self._opt = torch.optim.Adam(self._net.parameters(),
                                     lr=lr)
        self._temporal: Any = None

    @staticmethod
    def _spec_for(hidden: int) -> TopologySpec:
        """Shared encoder, four bit heads, struct, feedback."""
        stages = [Stage("in_", "add", {})]
        stages.append(Stage("enc_fc", "linear",
                            {"in_features": MOD_DIM,
                             "out_features": hidden}))
        stages.append(Stage("enc", "leaky", {"beta": 0.9,
                                             "threshold": 1.0}))
        names = ["b%d" % i for i in range(N_BITS)]
        for name in names:
            stages += _head_stages(name, hidden)
        stages.append(Stage("st_fc", "linear",
                            {"in_features": hidden,
                             "out_features": 2}))
        stages.append(Stage("st", "leaky", {"beta": 0.9,
                                            "threshold": 1.0}))
        stages.append(Stage("fb_fc", "linear",
                            {"in_features": 2,
                             "out_features": hidden}))
        edges = [Edge("in_", "enc_fc"), Edge("enc_fc", "enc")]
        for name in names:
            edges += [Edge("enc", name + "_fc"),
                      Edge(name + "_fc", name)]
        edges += [Edge("enc", "st_fc"), Edge("st_fc", "st"),
                  Edge("st", "fb_fc"),
                  Edge("fb_fc", "enc", delayed=True)]
        spec = TopologySpec(stages, edges, "in_", "st")
        spec.validate()
        return spec

    def reset(self) -> None:
        """Drop temporal state; next drive starts fresh."""
        self._temporal = None

    def _state(self, batch: int) -> Any:
        """Reuse temporal state when the batch shape agrees."""
        try:
            mem = self._temporal["enc"][0]
        except (KeyError, TypeError, IndexError):
            return None
        if torch.is_tensor(mem) and mem.shape[0] == batch:
            return self._temporal
        return None

    def _drive(self, mat: torch.Tensor) -> dict[str, Any]:
        """Mean spike rates per bit head and struct gate."""
        keys = ["b%d" % i for i in range(N_BITS)] + ["st"]
        acc = {k: torch.zeros(mat.shape[0], 1 if k != "st"
                              else 2) for k in keys}
        state = self._state(mat.shape[0])
        for _ in range(self._steps):
            outs, state = self._net.step(mat, state)
            for key in keys:
                acc[key] = acc[key] + outs[key]
        self._temporal = _clone(state)
        return {k: v / self._steps for k, v in acc.items()}

    def head_rates(self, feats: list[list[float]],
                   gap: float | None = None) -> dict[str, list]:
        """Per-bit and struct rates, no gradients, plain lists."""
        with torch.no_grad():
            out = self._drive(with_gap(feats, gap))
        return {k: v.tolist() for k, v in out.items()}

    def bits(self, feats: list[list[float]],
             gap: float | None = None) -> list[int]:
        """One packed word per row; head rate >= 0.5 is 1."""
        rates = self.head_rates(feats, gap)
        words = []
        for row in range(len(feats)):
            bits = [1 if rates["b%d" % i][row][0] >= 0.5
                    else 0 for i in range(N_BITS)]
            words.append(pack(bits))
        return words

    def struct(self, feats: list[list[float]],
               gap: float | None = None) -> list[bool]:
        """EMIT-majority commit flag per row (ties commit)."""
        rates = self.head_rates(feats, gap)
        return [row[EMIT] >= row[STOP] for row in rates["st"]]

    def load_encoder(self, weights: dict[str, Any]) -> int:
        """Copy encoder/struct/feedback stages by key prefix."""
        own = self._net.state_dict()
        src = {k: (v if torch.is_tensor(v) else torch.tensor(v))
               for k, v in weights.items()}
        kept = {k: v for k, v in src.items()
                if k.split(".")[0] in ENCODER_PREFIXES
                and k in own and own[k].shape == v.shape}
        own.update(kept)
        self._net.load_state_dict(own)
        return len(kept)

    def set_encoder_frozen(self, frozen: bool) -> None:
        """Write-lock (or unlock) the shared encoder stages."""
        for name, param in self._net.named_parameters():
            if name.split(".")[0] in ENCODER_PREFIXES:
                param.requires_grad = not frozen

    def perturb(self, sigma: float = 0.05, seed: int = 0) -> None:
        """Evolutionary operator: noise on heads only."""
        rng = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for name, param in self._net.named_parameters():
                if name.split(".")[0] not in HEAD_PREFIXES:
                    continue
                noise = torch.randn(param.shape, generator=rng)
                param.add_(noise * sigma)

    def weights(self) -> dict[str, Any]:
        """Detached CPU state dict of the lattice network."""
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
                "hidden": self._hidden}

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
        self._hidden = int(snap.get("hidden", 16))
        self._temporal = _clone(snap.get("temporal"))


def _bit_targets(groups: list[Any]
                 ) -> tuple[list[list[float]], list[int]]:
    """Opcode bits for positives, reserved word for negatives."""
    feats: list[list[float]] = []
    bits: list[int] = []
    for group in groups:
        for row, label in group:
            op = OP_ORDER[max(range(len(OP_ORDER)),
                              key=lambda i: row[i])]
            word = unpack(OPCODES[op]) if label == 1 else \
                [1, 1, 1, 1]
            feats.append(row)
            bits += word
    return feats, bits


def train_lattice(lat: BitLattice, fixture: str, ledger: str,
                  epochs: int = 8,
                  lam: float = 0.2) -> dict[str, Any]:
    """Train bit heads on opcode words; encoder stays frozen.

    Struct gate targets solution membership. A worst-bit
    joint term couples the four heads per word. Records one
    lineage intent; returns per-head and word-exact metrics.
    """
    groups = contrastive_groups(fixture)
    feats, flat = _bit_targets(groups)
    labels = [row[1] for group in groups for row in group]
    mat = with_gap(feats, None)
    want = torch.tensor(flat, dtype=torch.float32)
    want = want.reshape(len(feats), N_BITS)
    membership = torch.tensor(labels, dtype=torch.float32)
    lat.set_encoder_frozen(True)
    lat.reset()
    for _ in range(epochs):
        lat._opt.zero_grad()
        out = lat._drive(mat)
        bit = [func.binary_cross_entropy(out["b%d" % i]
                                         [:, 0], want[:, i],
                                         reduction="none")
               for i in range(N_BITS)]
        emit = func.binary_cross_entropy(out["st"][:, EMIT],
                                         membership)
        stop = func.binary_cross_entropy(out["st"][:, STOP],
                                         1.0 - membership)
        per_sample = torch.stack(bit, dim=1)
        worst = per_sample.max(dim=1).values.mean()
        loss = per_sample.mean() + emit + stop + lam * worst
        loss.backward()
        lat._opt.step()
    report = _report(lat, mat, want, fixture, epochs,
                     len(feats))
    store = mission_ledger(ledger, "rsi-lattice")
    store.record_intent("train-seed-%d" % lat._seed,
                        "rsi-lattice", 0, 1, report)
    store.close()
    return report


def _report(lat: BitLattice, mat: torch.Tensor,
            want: torch.Tensor, fixture: str, epochs: int,
            rounds: int) -> dict[str, Any]:
    """Per-head and word-exact metrics, no gradients."""
    with torch.no_grad():
        out = lat._drive(mat)
    got = torch.stack([out["b%d" % i][:, 0] >= 0.5
                       for i in range(N_BITS)], dim=1)
    truth = want > 0.5
    acc = [(got[:, i] == truth[:, i]).float().mean().item()
           for i in range(N_BITS)]
    exact = (got == truth).all(dim=1).float().mean()
    return {"fixture": fixture, "epochs": epochs,
            "rounds": rounds, "head_accuracy": acc,
            "word_exact": exact.item(),
            "emit_rate": out["st"][:, EMIT].mean().item()}
