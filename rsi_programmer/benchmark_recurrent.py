"""Nonspiking recurrent neural control for the frozen benchmark."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from rsi_programmer.benchmark_policy import (
    DIM,
    HIDDEN,
    STEPS,
    Group,
    eligible_groups,
    margin_loss,
    rank_scores,
    state_fingerprint,
    validate_features,
    validate_fit,
)


class RecurrentControl(nn.Module):
    """Apply a tanh RNN to one row for six fresh-state scoring ticks.

    Recurrence models within-candidate dynamics. Every forward drive starts
    from zero state; it does not remember prior candidates or task episodes.
    """

    def __init__(self, seed: int = 0, lr: float = 0.05) -> None:
        """Initialize reproducibly while restoring the caller's torch RNG."""
        super().__init__()
        if not math.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be finite and positive")
        self._seed = seed
        self._lr = float(lr)
        self._steps = STEPS
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self._input = nn.Linear(DIM, HIDDEN)
            self._recurrent = nn.Linear(HIDDEN, HIDDEN)
            self._readout = nn.Linear(HIDDEN, 1)
        self._opt = torch.optim.Adam(self.parameters(), lr=self._lr)

    def _forward(self, mat: torch.Tensor) -> torch.Tensor:
        """Return a mean readout with fresh zero state on every call."""
        state = torch.zeros((mat.shape[0], HIDDEN), dtype=mat.dtype,
                            device=mat.device)
        total = torch.zeros(mat.shape[0], dtype=mat.dtype,
                            device=mat.device)
        drive = self._input(mat)
        for _ in range(STEPS):
            state = torch.tanh(drive + self._recurrent(state))
            total = total + self._readout(state).squeeze(-1)
        return total / STEPS

    def rank(self, feats: list[list[float]], triples: Any = None,
             gap: float | None = None) -> list[int]:
        """Return stable descending recurrent score order."""
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
                for name, value in self.state_dict().items()}

    def metadata(self) -> dict[str, Any]:
        """Report architecture, capacity and recurrence boundaries."""
        return {
            "name": "nonspiking_recurrent_control",
            "architecture": "12->16_tanh_rnn->mean_readout_1",
            "trainable_parameters": sum(
                parameter.numel() for parameter in self.parameters()
            ),
            "simulation_steps": STEPS,
            "state_semantics": (
                "fresh_zero_state_per_candidate_drive; recurrence is within "
                "candidate only and carries no task-episode memory"
            ),
            "fit_mode": f"adam_margin_lr_{self._lr:g}",
            "learning_rate": self._lr,
            "seed": self._seed,
            "feature_encoding": "layer-ordinal-v1",
            "feature_dimension": 12,
            "capacity_note": (
                "parameter count differs from the spikeforge SNN; "
                "this is not an energy comparison"
            ),
        }

    def fingerprint(self) -> str:
        """Hash architecture and current trained weights."""
        return state_fingerprint(self.weights(), self.metadata())
