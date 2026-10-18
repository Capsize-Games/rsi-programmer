"""Shared helpers for frozen benchmark ranking controls."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import torch

DIM = 12
HIDDEN = 16
STEPS = 6
Group = list[tuple[list[float], int]]


def validate_features(feats: list[list[float]]) -> torch.Tensor:
    """Convert and validate the operand-only 12-field encoding."""
    if any(len(row) != DIM for row in feats):
        raise ValueError(f"each feature row must have {DIM} values")
    mat = (torch.tensor(feats, dtype=torch.float32) if feats else
           torch.empty((0, DIM), dtype=torch.float32))
    if mat.numel() and not bool(torch.isfinite(mat).all()):
        raise ValueError("features must be finite")
    return mat


def rank_scores(scores: torch.Tensor) -> list[int]:
    """Return descending score order, preserving input order on ties."""
    values = scores.detach().cpu().reshape(-1).tolist()
    return sorted(range(len(values)), key=lambda i: values[i], reverse=True)


def state_fingerprint(state: dict[str, Any], metadata: dict[str, Any]
                      ) -> str:
    """Hash state values and declared architecture with stable JSON."""
    values: dict[str, Any] = {}
    for name, value in sorted(state.items()):
        if torch.is_tensor(value):
            values[name] = {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "values": value.detach().cpu().tolist(),
            }
        else:
            values[name] = value
    payload = json.dumps({"metadata": metadata, "weights": values},
                         sort_keys=True, separators=(",", ":"),
                         allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def eligible_groups(groups: list[Group]
                    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Make positive and negative tensors, dropping incomplete groups."""
    result: list[tuple[torch.Tensor, torch.Tensor]] = []
    for group in groups:
        pos = [features for features, label in group if label == 1]
        neg = [features for features, label in group if label == 0]
        if pos and neg:
            result.append((validate_features(pos), validate_features(neg)))
    if not result:
        raise ValueError("fit requires at least one group with both labels")
    return result


def margin_loss(forward: Any,
                groups: list[tuple[torch.Tensor,
                                   torch.Tensor]]) -> torch.Tensor:
    """Sum per-group mean softplus ranking margins."""
    loss = torch.zeros(())
    for pos, neg in groups:
        gap = forward(pos).unsqueeze(1) - forward(neg).unsqueeze(0)
        loss = loss + torch.nn.functional.softplus(1.0 - gap).mean()
    return loss


def validate_fit(epochs: int) -> None:
    """Reject a request that would silently perform no training."""
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("epochs must be a positive integer")


def finite_json(value: Any) -> None:
    """Raise if metadata cannot be fingerprinted as finite JSON."""
    json.dumps(value, allow_nan=False)

