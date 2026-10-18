"""Measure fitting work without changing optimizer or forward behavior."""
from __future__ import annotations

from typing import Any, Callable

from rsi_programmer.benchmark_policy import Group


def fit_with_work(model: Any, groups: list[Group], epochs: int,
                  steps: int) -> tuple[float, dict[str, int]]:
    """Fit a learned control and count its actual forward/update work.

    Counts include only groups with both labels, matching the controls'
    margin-loss helper. ``simulation_row_steps`` is scored rows multiplied
    by the model's declared per-row ticks; it is an operation count, not an
    estimate of multiply-accumulates, energy, or wall time.
    """
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    positive = [sum(label == 1 for _features, label in group)
                for group in groups]
    negative = [sum(label == 0 for _features, label in group)
                for group in groups]
    eligible = [(pos, neg) for pos, neg in zip(positive, negative)
                if pos and neg]
    counters = {
        "requested_epochs": epochs,
        "eligible_groups": len(eligible),
        "positive_features": sum(pos for pos, _neg in eligible),
        "negative_features": sum(neg for _pos, neg in eligible),
        "margin_pairs_per_epoch": sum(pos * neg
                                       for pos, neg in eligible),
        "forward_batches": 0,
        "scored_feature_rows": 0,
        "simulation_row_steps": 0,
        "optimizer_updates": 0,
    }

    forward_had_instance = "_forward" in model.__dict__
    forward_previous = model.__dict__.get("_forward")
    step_had_instance = "step" in model._opt.__dict__
    step_previous = model._opt.__dict__.get("step")
    forward: Callable[..., Any] = model._forward
    optimizer_step: Callable[..., Any] = model._opt.step

    def counted_forward(mat: Any, *args: Any,
                        **kwargs: Any) -> Any:
        counters["forward_batches"] += 1
        rows = int(mat.shape[0])
        counters["scored_feature_rows"] += rows
        counters["simulation_row_steps"] += rows * steps
        return forward(mat, *args, **kwargs)

    def counted_step(*args: Any, **kwargs: Any) -> Any:
        result = optimizer_step(*args, **kwargs)
        counters["optimizer_updates"] += 1
        return result

    model._forward = counted_forward
    model._opt.step = counted_step
    try:
        loss = model.fit(groups, epochs)
    finally:
        if forward_had_instance:
            model._forward = forward_previous
        else:
            del model._forward
        if step_had_instance:
            model._opt.step = step_previous
        else:
            del model._opt.step
    return float(loss), counters
