"""Small direct decoder for early development measurements."""
from __future__ import annotations

from typing import Any

import torch

from rsi_programmer.construction_features import encode_states
from rsi_programmer.construction_policy import OPERATIONS, ConstructionPolicy
from rsi_programmer.construction_state import ConstructionState, Rows


def greedy_construct(policy: ConstructionPolicy, observed: Rows,
                     bound: int = 4) -> dict[str, Any]:
    """Choose op and operands directly; no candidate enumeration or retry."""
    state = ConstructionState(observed, bound)
    actions: list[dict[str, str | int]] = []
    with torch.inference_mode():
        for _ in range(bound + 1):
            batch = encode_states([state])
            context, references = policy.encode(batch)
            operation = policy.operation_logits(context).argmax(dim=1)
            op = OPERATIONS[int(operation[0])]
            if op == "stop":
                actions.append({"op": "stop"})
                stopped = state.apply({"op": "stop"})
                return {"graph": stopped.graph(), "stopped": True,
                        "observed_match": stopped.observed_match(),
                        "actions": actions, "error": None}
            left = policy.left_logits(
                context, references, batch[1], operation
            ).argmax(dim=1)
            right = policy.right_logits(
                context, references, batch[1], operation, left
            ).argmax(dim=1)
            action = {"op": op, "left": int(left[0]),
                      "right": int(right[0])}
            actions.append(action)
            try:
                state = state.apply(action)
            except ValueError as error:
                return {"graph": None, "stopped": False,
                        "observed_match": False, "actions": actions,
                        "error": str(error)}
    return {"graph": None, "stopped": False, "observed_match": False,
            "actions": actions, "error": "decision budget exhausted"}
