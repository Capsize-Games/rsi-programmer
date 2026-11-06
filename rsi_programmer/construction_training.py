"""Supervised construction learning from training-only execution traces."""
from __future__ import annotations

import time
from typing import Any

import torch
from torch.nn import functional as F

from rsi_programmer.construction_corpus import teacher_steps
from rsi_programmer.construction_features import (
    Batch, encode_states, slice_batch,
)
from rsi_programmer.construction_policy import OPERATIONS, ConstructionPolicy
from rsi_programmer.construction_state import Action, ConstructionState

Example = tuple[ConstructionState, Action]
Prepared = tuple[Batch, torch.Tensor, torch.Tensor, torch.Tensor]


def _check_schedule(epochs: int, batch_size: int, lr: float) -> None:
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be positive")
    if type(batch_size) is not int or batch_size < 1 or lr <= 0:
        raise ValueError("positive batch size and learning rate required")


def _prepare_tensors(examples: list[Example]) -> Prepared:
    """Encode states once; order matches the caller-supplied examples."""
    batch = encode_states([state for state, _ in examples])
    operations = torch.tensor([OPERATIONS.index(str(action["op"]))
                               for _, action in examples])
    left = torch.tensor([int(action.get("left", 0))
                         for _, action in examples])
    right = torch.tensor([int(action.get("right", 0))
                          for _, action in examples])
    return batch, operations, left, right


def _pointer_loss(policy: ConstructionPolicy, context: torch.Tensor,
                  references: torch.Tensor, inputs: Batch,
                  selected: torch.Tensor, chosen: torch.Tensor,
                  left: torch.Tensor, right: torch.Tensor
                  ) -> torch.Tensor:
    """Add operand-pointer losses over the non-stop rows."""
    active = selected != OPERATIONS.index("stop")
    mask = inputs[1][active]
    loss = F.cross_entropy(policy.left_logits(
        context[active], references[active], mask,
        selected[active]), left[chosen][active])
    return loss + F.cross_entropy(policy.right_logits(
        context[active], references[active], mask, selected[active],
        left[chosen][active]), right[chosen][active])


def _train_step(policy: ConstructionPolicy,
                optimizer: torch.optim.Adam, prepared: Prepared,
                chosen: torch.Tensor) -> float:
    batch, operations, left, right = prepared
    inputs = slice_batch(batch, chosen)
    context, references = policy.encode(inputs)
    selected = operations[chosen]
    loss = F.cross_entropy(policy.operation_logits(context), selected)
    if bool((selected != OPERATIONS.index("stop")).any()):
        loss = loss + _pointer_loss(policy, context, references,
                                    inputs, selected, chosen, left,
                                    right)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())


def _fit_loop(policy: ConstructionPolicy, optimizer: torch.optim.Adam,
              prepared: Prepared, generator: torch.Generator,
              epochs: int, batch_size: int) -> tuple[int, float]:
    count, updates, loss_value = len(prepared[1]), 0, 0.0
    for _ in range(epochs):
        order = torch.randperm(count, generator=generator)
        for offset in range(0, count, batch_size):
            loss_value = _train_step(policy, optimizer, prepared,
                                     order[offset:offset + batch_size])
            updates += 1
    return updates, loss_value


def _begin_fit(policy: ConstructionPolicy, lr: float
             ) -> tuple[torch.optim.Adam, torch.Generator]:
    """Enter training mode and seed the optimizer schedule."""
    policy.train()
    policy.reset_work()
    return (torch.optim.Adam(policy.parameters(), lr=lr),
            torch.Generator().manual_seed(policy.seed + 100))


def _fit_receipt(initial: str, policy: ConstructionPolicy, count: int,
                 epochs: int, updates: int, loss: float, ready: float,
                 started: float, cpu: float) -> dict[str, Any]:
    return {"initial_hash": initial, "final_hash": policy.fingerprint(),
            "examples": count, "epochs": epochs, "updates": updates,
            "final_loss": loss, "preparation_wall_seconds": ready,
            "complete_wall_seconds": time.perf_counter() - started,
            "complete_cpu_seconds": time.process_time() - cpu,
            "neural_work": dict(policy.work)}


def fit_on_examples(policy: ConstructionPolicy, examples: list[Example],
                    epochs: int = 100, batch_size: int = 64,
                    lr: float = 0.003) -> dict[str, Any]:
    """Fit heads on fixed examples; callers own train-split discipline."""
    _check_schedule(epochs, batch_size, lr)
    if not examples:
        raise ValueError("only nonempty example lists may fit")
    started, cpu = time.perf_counter(), time.process_time()
    prepared = _prepare_tensors(examples)
    ready, initial = time.perf_counter() - started, policy.fingerprint()
    optimizer, generator = _begin_fit(policy, lr)
    updates, loss = _fit_loop(policy, optimizer, prepared, generator,
                              epochs, batch_size)
    policy.eval()
    return _fit_receipt(initial, policy, len(examples), epochs,
                        updates, loss, ready, started, cpu)


def fit_constructor(policy: ConstructionPolicy, tasks: list[dict[str, Any]],
                    epochs: int = 100, batch_size: int = 64,
                    lr: float = 0.003) -> dict[str, Any]:
    """Fit chosen-operation/pointer losses; never generate held labels."""
    _check_schedule(epochs, batch_size, lr)
    if not tasks or any(task.get("split") != "train" for task in tasks):
        raise ValueError("only nonempty training tasks may supply labels")
    examples = [example for task in tasks for example in teacher_steps(task)]
    result = fit_on_examples(policy, examples, epochs, batch_size, lr)
    result["tasks"] = len(tasks)
    result["teacher_steps"] = result.pop("examples")
    return result
