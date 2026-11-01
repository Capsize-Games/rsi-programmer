"""Factorized ANN/SNN decisions over a partial executable program."""
from __future__ import annotations

import math
from typing import Any

from snntorch import surrogate
import torch
from torch import nn

from rsi_programmer.benchmark_policy import state_fingerprint
from rsi_programmer.construction_features import Batch, FEATURE_DIM

OPERATIONS = ("add", "sub", "mul", "stop")


class ConstructionPolicy(nn.Module):
    """Choose an opcode and conditional operand pointers, then STOP.

    ANN uses a tanh row encoder. The SNN alternative uses six fresh-state
    LIF ticks and binary spikes before the same pooling and pointer heads.
    Equal parameter counts do not imply equal capacity or compute.
    """

    def __init__(self, seed: int = 0, backend: str = "ann",
                 hidden: int = 64, steps: int = 6) -> None:
        super().__init__()
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if backend not in ("ann", "snn"):
            raise ValueError("backend must be ann or snn")
        if type(hidden) is not int or hidden < 4:
            raise ValueError("hidden size must be at least four")
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be positive")
        self.seed, self.backend = seed, backend
        self.hidden, self.steps = hidden, steps
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.row_input = nn.Linear(FEATURE_DIM, hidden)
            self.row_hidden = nn.Linear(hidden, hidden)
            nn.init.constant_(self.row_hidden.bias, 0.2)
            self.reference = nn.Linear(2 * hidden, hidden)
            self.context = nn.Linear(3 * hidden + 2, hidden)
            self.operation = nn.Linear(hidden, len(OPERATIONS))
            self.op_embedding = nn.Embedding(len(OPERATIONS), hidden)
            self.left_query = nn.Linear(hidden, hidden)
            self.right_query = nn.Linear(2 * hidden, hidden)
            self.keys = nn.Linear(hidden, hidden)
        self.reset_work()

    def reset_work(self) -> None:
        self.work = {"encoder_calls": 0, "encoded_rows": 0,
                     "neuron_ticks": 0, "spikes": 0}

    def _rows(self, features: torch.Tensor) -> torch.Tensor:
        drive = self.row_hidden(torch.tanh(self.row_input(features)))
        ticks = self.steps if self.backend == "snn" else 1
        self.work["encoder_calls"] += 1
        count = features.numel() // FEATURE_DIM
        self.work["encoded_rows"] += count
        self.work["neuron_ticks"] += count * self.hidden * ticks
        if self.backend == "ann":
            return torch.tanh(drive)
        membrane = torch.zeros_like(drive)
        total = torch.zeros_like(drive)
        spike_fn = surrogate.fast_sigmoid(slope=12)
        for _ in range(self.steps):
            membrane = 0.8 * membrane + drive
            spikes = spike_fn(membrane - 0.5)
            membrane = membrane - 0.5 * spikes
            total = total + spikes
        self.work["spikes"] += int(total.detach().sum().item())
        return total / self.steps

    def encode(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode task content and partial state with fresh neural state."""
        features, mask, outputs, progress = batch
        rows = self._rows(features)
        pooled = torch.cat((rows.mean(dim=2), rows.amax(dim=2)), dim=-1)
        references = torch.tanh(self.reference(pooled))
        valid = mask.unsqueeze(-1)
        mean = (references * valid).sum(dim=1) / valid.sum(dim=1)
        maximum = references.masked_fill(~valid, -torch.inf).amax(dim=1)
        output = references[torch.arange(len(outputs)), outputs]
        context = torch.tanh(self.context(torch.cat(
            (mean, maximum, output, progress), dim=-1
        )))
        return context, references

    def operation_logits(self, context: torch.Tensor) -> torch.Tensor:
        return self.operation(context)

    def left_logits(self, context: torch.Tensor, references: torch.Tensor,
                    mask: torch.Tensor, operations: torch.Tensor
                    ) -> torch.Tensor:
        query = self.left_query(torch.tanh(
            context + self.op_embedding(operations)
        ))
        return self._pointers(query, references, mask)

    def right_logits(self, context: torch.Tensor, references: torch.Tensor,
                     mask: torch.Tensor, operations: torch.Tensor,
                     left: torch.Tensor) -> torch.Tensor:
        selected = references[torch.arange(len(left)), left]
        query = self.right_query(torch.cat((
            torch.tanh(context + self.op_embedding(operations)), selected,
        ), dim=-1))
        return self._pointers(query, references, mask)

    def _pointers(self, query: torch.Tensor, references: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
        scores = (self.keys(references) * query.unsqueeze(1)).sum(dim=-1)
        return (scores / math.sqrt(self.hidden)).masked_fill(
            ~mask, -torch.inf
        )

    def metadata(self) -> dict[str, Any]:
        return {"backend": self.backend, "seed": self.seed,
                "hidden": self.hidden, "feature_dimension": FEATURE_DIM,
                "steps": self.steps if self.backend == "snn" else 1,
                "parameter_count": sum(p.numel() for p in self.parameters()),
                "state": "observed_partial_program; fresh neural drive",
                "feature_encoding": "observed-reference-rows-v1"}

    def fingerprint(self) -> str:
        return state_fingerprint(self.state_dict(), self.metadata())
