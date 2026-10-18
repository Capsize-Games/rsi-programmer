#!/usr/bin/env python3
"""Bit-slice lattice: four independent 1-bit rate heads.

Head i drives bit i (weight 2**i) of every 4-bit word; the
struct output is trained on membership and its complement, then
decodes a commit when EMIT rate clears STOP rate by its calibrated
margin. Encoder and feedback stages can load from a modular snapshot.
"""
from __future__ import annotations

from typing import Any

import torch
from spikeforge.topology.builder import build_module
from spikeforge.topology.edge import Edge
from spikeforge.topology.spec import TopologySpec
from spikeforge.topology.stage import Stage

from rsi_programmer.modules import MOD_DIM, with_gap
from rsi_programmer.policy import FEATURE_ENCODING, STEPS, _clone
from rsi_programmer.words import pack

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
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError("lattice steps must be positive")
        torch.manual_seed(seed)
        self._seed = seed
        self._steps = steps
        self._hidden = hidden
        self._spec = self._spec_for(hidden)
        self._net = build_module(self._spec)
        self._opt = torch.optim.Adam(self._net.parameters(),
                                     lr=lr)
        self._temporal: Any = None
        self.thresholds = {"b%d" % i: 0.5
                            for i in range(N_BITS)}
        self.thresholds["st"] = 0.0

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

    def _drive(self, mat: torch.Tensor) -> dict[str, Any]:
        """Mean spike rates, fresh state per drive (#52).

        The end state is kept only for snapshot
        serialization; scoring never consumes it. reset()
        stays as an explicit no-history marker for callers.
        """
        keys = ["b%d" % i for i in range(N_BITS)] + ["st"]
        acc = {k: torch.zeros(mat.shape[0], 1 if k != "st"
                              else 2) for k in keys}
        state: Any = None
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
        """One packed word per row; learned head thresholds."""
        rates = self.head_rates(feats, gap)
        words = []
        for row in range(len(feats)):
            bits = [1 if rates["b%d" % i][row][0] >=
                    self.thresholds["b%d" % i]
                    else 0 for i in range(N_BITS)]
            words.append(pack(bits))
        return words

    def struct(self, feats: list[list[float]],
               gap: float | None = None) -> list[bool]:
        """Commit when EMIT clears STOP by the calibrated margin.

        Lattice training fits EMIT on membership and STOP on its
        complement. Threshold calibration tunes their difference.
        """
        rates = self.head_rates(feats, gap)
        margin = self.thresholds["st"]
        return [row[EMIT] >= row[STOP] + margin
                for row in rates["st"]]

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
        """Clone weights, locks and options; Adam moments are omitted."""
        weights = {k: v.detach().cpu().tolist()
                   for k, v in self._net.state_dict().items()}
        return {"spec": self._spec.to_dict(),
                "weights": weights,
                "temporal": _clone(self._temporal),
                "feature_encoding": FEATURE_ENCODING,
                "frozen": [name for name, param in
                           self._net.named_parameters()
                           if not param.requires_grad],
                "optimizer": self._optimizer_options(),
                "thresholds": dict(self.thresholds),
                "seed": self._seed, "steps": self._steps,
                "hidden": self._hidden}

    def _optimizer_options(self) -> dict[str, Any]:
        """Capture Adam options while leaving its moments fresh."""
        keys = ("lr", "betas", "eps", "weight_decay", "amsgrad",
                "maximize", "foreach", "capturable", "differentiable",
                "fused")
        group = self._opt.param_groups[0]
        return {key: group[key] for key in keys if key in group}

    def restore(self, snap: dict[str, Any], *,
                optimizer_from_snapshot: bool = False) -> None:
        """Reload weights and runtime state, resetting Adam moments.

        Recipient optimizer options are retained unless the caller explicitly
        opts into the donor snapshot's saved options. Legacy untagged snapshots
        remain loadable, with no verifiable feature-encoding contract.
        """
        encoding = snap.get("feature_encoding")
        if encoding is not None and encoding != FEATURE_ENCODING:
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
        frozen = set(snap.get("frozen", []))
        for name, param in self._net.named_parameters():
            param.requires_grad_(name not in frozen)
        self._opt = torch.optim.Adam(self._net.parameters(), **options)
        self._steps = int(snap.get("steps", STEPS))
        self._hidden = int(snap.get("hidden", 16))
        self._temporal = _clone(snap.get("temporal"))
        if "thresholds" in snap:
            self.thresholds = dict(snap["thresholds"])
