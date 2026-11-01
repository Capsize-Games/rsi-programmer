"""Bounded search over factorized construction actions for development."""
from __future__ import annotations

import math
import random
from typing import Any

import torch

from rsi_programmer.construction_features import encode_states
from rsi_programmer.construction_policy import OPERATIONS, ConstructionPolicy
from rsi_programmer.construction_state import Action, ConstructionState, Rows


def _best(logits: torch.Tensor, count: int = 2
          ) -> list[tuple[int, float]]:
    values = torch.log_softmax(logits, dim=-1).flatten().tolist()
    order = sorted(range(len(values)), key=lambda i: -values[i])
    return [(index, values[index]) for index in order[:count]
            if math.isfinite(values[index])]


def _choices(policy: ConstructionPolicy | None, state: ConstructionState,
             rng: random.Random) -> list[tuple[float, Action]]:
    costs = state.costs()
    count = len(costs)
    if policy is not None:
        batch = encode_states([state])
        context, references = policy.encode(batch)
        operations = _best(policy.operation_logits(context))
    else:
        operations = [(index, -math.log(4)) for index in
                      rng.sample(range(len(OPERATIONS)), 2)]
    choices: list[tuple[float, Action]] = []
    for op_index, op_score in operations:
        op = OPERATIONS[op_index]
        if op == "stop":
            choices.append((op_score, {"op": "stop"}))
            continue
        if state.node_count >= state.bound:
            continue
        allowed_left = [index for index, cost in enumerate(costs)
                        if cost < state.bound]
        if policy is not None:
            operation = torch.tensor([op_index])
            mask = torch.tensor([[i in allowed_left for i in range(count)]])
            left_choices = _best(policy.left_logits(
                context, references, mask, operation
            ))
        else:
            left_choices = [(index, -math.log(len(allowed_left))) for index
                            in rng.sample(allowed_left,
                                          min(2, len(allowed_left)))]
        for left, left_score in left_choices:
            allowed_right = [index for index, cost in enumerate(costs)
                             if costs[left] + cost + 1 <= state.bound]
            if policy is not None:
                mask = torch.tensor([[i in allowed_right
                                      for i in range(count)]])
                right_choices = _best(policy.right_logits(
                    context, references, mask, operation,
                    torch.tensor([left]),
                ))
            else:
                right_choices = [(index, -math.log(len(allowed_right)))
                                 for index in rng.sample(
                                     allowed_right,
                                     min(2, len(allowed_right)),
                                 )]
            for right, right_score in right_choices:
                choices.append((op_score + left_score + right_score,
                                {"op": op, "left": left, "right": right}))
    return choices


def beam_construct(policy: ConstructionPolicy | None, observed: Rows,
                   bound: int = 4, budget: int = 256, width: int = 8,
                   seed: int = 0) -> dict[str, Any]:
    """Return the first explicit observed-correct stop, never a held retry.

    None uses random factorized actions with identical beam and masks.
    Root execution caps exclude neural/feature/stop work; CPU must be
    reported separately. Every non-stop child is executed before pruning.
    """
    if type(budget) is not int or budget < 0 or width < 1:
        raise ValueError("nonnegative budget and positive width required")
    rng = random.Random(seed)
    initial = ConstructionState(observed, bound)
    beam: list[tuple[float, ConstructionState, list[Action]]] = [
        (0.0, initial, [])
    ]
    executions = stops = 0
    with torch.inference_mode():
        for _ in range(bound + 1):
            proposed = [(score + extra, state, [*actions, action], action)
                        for score, state, actions in beam
                        for extra, action in _choices(policy, state, rng)]
            proposed.sort(key=lambda item: -item[0])
            children: list[tuple[float, ConstructionState, list[Action]]] = []
            seen: set[str] = set()
            for score, state, actions, action in proposed:
                if action["op"] == "stop":
                    stops += 1
                    child = state.apply(action)
                    if child.observed_match():
                        return {"graph": child.graph(), "stopped": True,
                                "actions": actions, "executions": executions,
                                "stop_attempts": stops, "error": None}
                    continue
                if executions >= budget:
                    return {"graph": None, "stopped": False,
                            "actions": [], "executions": executions,
                            "stop_attempts": stops, "error": "root budget"}
                child = state.apply(action)
                executions += 1
                key = str(child.graph())
                if key not in seen:
                    seen.add(key)
                    children.append((score, child, actions))
            beam = children[:width]
            if not beam:
                break
    return {"graph": None, "stopped": False, "actions": [],
            "executions": executions, "stop_attempts": stops,
            "error": "beam exhausted"}
