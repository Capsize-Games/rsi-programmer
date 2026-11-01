"""Observed-row and live-reference features for a construction controller."""
from __future__ import annotations

import math

import torch

from rsi_programmer.construction_state import ConstructionState

FEATURE_DIM = 14
ORIGINS = ("x", "y", "add", "sub", "mul")
Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def encode_states(states: list[ConstructionState]) -> Batch:
    """Encode actual task rows and already executed references, not actions.

    Rows are pooled by the policy after a learned transformation. A batch
    requires equal row counts, but row order has no semantic identity.
    Candidate root results, generator witnesses and held data are absent.
    """
    if not states or any(state.stopped for state in states):
        raise ValueError("nonempty unstopped construction states required")
    row_count = len(states[0].observed)
    if any(len(state.observed) != row_count for state in states):
        raise ValueError("a batch requires equal observed row counts")
    width = max(len(state.vectors()) for state in states)
    features = torch.zeros(len(states), width, row_count, FEATURE_DIM)
    mask = torch.zeros(len(states), width, dtype=torch.bool)
    outputs = torch.tensor([state.output_slot for state in states])
    progress = torch.tensor([
        [state.node_count / state.bound, state.bound / 8.0]
        for state in states
    ])
    for index, state in enumerate(states):
        rows = state.observed
        input_scale = max(1, *(abs(value) for x, y, _ in rows
                               for value in (x, y)))
        scale = max(1, *(abs(target) for _, _, target in rows))
        graph = state.graph()
        for slot, vector in enumerate(state.vectors()):
            origin = (ORIGINS[slot] if slot < 2
                      else graph["nodes"][slot - 2]["op"])
            metadata = [float(origin == name) for name in ORIGINS]
            metadata += [state.costs()[slot] / state.bound,
                         float(slot == state.output_slot),
                         math.log1p(scale) / 8.0]
            values = [
                [x / input_scale, y / input_scale, target / scale,
                 value / scale, (target - value) / scale,
                 float(value == target), *metadata]
                for (x, y, target), value in zip(rows, vector)
            ]
            features[index, slot] = torch.tensor(values)
            mask[index, slot] = True
    if not bool(torch.isfinite(features).all()):
        raise ValueError("construction features must be finite")
    return features, mask, outputs, progress


def slice_batch(batch: Batch, indices: torch.Tensor) -> Batch:
    """Select already encoded teacher states."""
    return (batch[0][indices], batch[1][indices],
            batch[2][indices], batch[3][indices])
