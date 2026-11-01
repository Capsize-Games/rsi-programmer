"""Supervised construction learning from training-only execution traces."""
from __future__ import annotations

import time
from typing import Any

import torch
from torch.nn import functional as F

from rsi_programmer.construction_corpus import teacher_steps
from rsi_programmer.construction_features import encode_states, slice_batch
from rsi_programmer.construction_policy import OPERATIONS, ConstructionPolicy


def fit_constructor(policy: ConstructionPolicy, tasks: list[dict[str, Any]],
                    epochs: int = 100, batch_size: int = 64,
                    lr: float = 0.003) -> dict[str, Any]:
    """Fit chosen-operation/pointer losses; never generate held labels."""
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be positive")
    if type(batch_size) is not int or batch_size < 1 or lr <= 0:
        raise ValueError("positive batch size and learning rate required")
    started, cpu = time.perf_counter(), time.process_time()
    if not tasks or any(task.get("split") != "train" for task in tasks):
        raise ValueError("only nonempty training tasks may supply labels")
    examples = [example for task in tasks for example in teacher_steps(task)]
    batch = encode_states([state for state, _ in examples])
    operations = torch.tensor([OPERATIONS.index(str(action["op"]))
                               for _, action in examples])
    left = torch.tensor([int(action.get("left", 0))
                         for _, action in examples])
    right = torch.tensor([int(action.get("right", 0))
                          for _, action in examples])
    prepared = time.perf_counter() - started
    initial_hash = policy.fingerprint()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(policy.seed + 100)
    policy.train()
    policy.reset_work()
    updates, loss_value = 0, 0.0
    for _ in range(epochs):
        order = torch.randperm(len(examples), generator=generator)
        for offset in range(0, len(order), batch_size):
            chosen = order[offset:offset + batch_size]
            inputs = slice_batch(batch, chosen)
            context, references = policy.encode(inputs)
            op = operations[chosen]
            loss = F.cross_entropy(policy.operation_logits(context), op)
            active = op != OPERATIONS.index("stop")
            if bool(active.any()):
                mask = inputs[1][active]
                loss = loss + F.cross_entropy(policy.left_logits(
                    context[active], references[active], mask, op[active],
                ), left[chosen][active])
                loss = loss + F.cross_entropy(policy.right_logits(
                    context[active], references[active], mask, op[active],
                    left[chosen][active],
                ), right[chosen][active])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            updates += 1
            loss_value = float(loss.detach())
    policy.eval()
    return {"initial_hash": initial_hash, "final_hash": policy.fingerprint(),
            "tasks": len(tasks), "teacher_steps": len(examples),
            "epochs": epochs, "updates": updates, "final_loss": loss_value,
            "preparation_wall_seconds": prepared,
            "complete_wall_seconds": time.perf_counter() - started,
            "complete_cpu_seconds": time.process_time() - cpu,
            "neural_work": dict(policy.work)}
