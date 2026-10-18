"""Feed-forward neural control for the frozen search benchmark."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from rsi_programmer.benchmark_policy import (
    DIM,
    HIDDEN,
    Group,
    eligible_groups,
    margin_loss,
    rank_scores,
    state_fingerprint,
    validate_features,
    validate_fit,
)


class MLPControl:
    """Train a 12 -> 16 ReLU -> 1 ranking function."""

    def __init__(self, seed: int = 0, lr: float = 0.05) -> None:
        """Initialize reproducibly while restoring the caller's torch RNG."""
        if not math.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be finite and positive")
        self._seed = seed
        self._lr = float(lr)
        self._steps = 1
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self._net = nn.Sequential(
                nn.Linear(DIM, HIDDEN),
                nn.ReLU(),
                nn.Linear(HIDDEN, 1),
            )
        self._opt = torch.optim.Adam(self._net.parameters(), lr=self._lr)

    def _forward(self, mat: torch.Tensor) -> torch.Tensor:
        """Score rows without carrying state between ranking drives."""
        return self._net(mat).squeeze(-1)

    def rank(self, feats: list[list[float]], triples: Any = None,
             gap: float | None = None) -> list[int]:
        """Return stable descending MLP score order."""
        del triples, gap
        mat = validate_features(feats)
        with torch.no_grad():
            return rank_scores(self._forward(mat))

    def fit(self, groups: list[Group], epochs: int) -> float:
        """Fit the same summed per-group softplus margin objective."""
        validate_fit(epochs)
        tensors = eligible_groups(groups)
        loss_value = 0.0
        for _ in range(epochs):
            self._opt.zero_grad()
            loss = margin_loss(self._forward, tensors)
            loss.backward()
            self._opt.step()
            loss_value = float(loss.detach())
        return loss_value

    def weights(self) -> dict[str, torch.Tensor]:
        """Return detached copies of the trained state dict."""
        return {name: value.detach().cpu().clone()
                for name, value in self._net.state_dict().items()}

    def metadata(self) -> dict[str, Any]:
        """Report architecture, capacity and stateless scoring semantics."""
        return {
            "name": "mlp_control",
            "architecture": "12->16_relu->1",
            "trainable_parameters": sum(
                parameter.numel() for parameter in self._net.parameters()
            ),
            "simulation_steps": 1,
            "state_semantics": "stateless_per_candidate_batch",
            "fit_mode": f"adam_margin_lr_{self._lr:g}",
            "learning_rate": self._lr,
            "seed": self._seed,
            "feature_encoding": "layer-ordinal-v1",
            "feature_dimension": 12,
            "capacity_note": (
                "225 parameters, matching the SNN count; activation and "
                "temporal dynamics differ, so count does not establish "
                "equivalent capacity or energy"
            ),
        }

    def fingerprint(self) -> str:
        """Hash architecture and current trained weights."""
        return state_fingerprint(self.weights(), self.metadata())
