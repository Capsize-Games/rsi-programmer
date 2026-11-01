"""Learn library compositions from concrete input/output examples."""
from __future__ import annotations

import itertools
import math
import random
from typing import Any

import torch
from torch import nn

from rsi_programmer.utility_ops import OPS, apply_op, execute

TOKENS = (*OPS, "stop")
Pairs = list[tuple[list[int], list[int]]]


def _array(values: list[int], scale: int) -> list[float]:
    padded = [value / scale for value in values[:12]]
    padded += [0.0] * (12 - len(padded))
    length = len(values)
    return padded + [length / 12.0, min(values, default=0) / scale,
                     max(values, default=0) / scale,
                     sum(values) / (scale * 12.0),
                     sum(v > 0 for v in values) / 12.0,
                     sum(v < 0 for v in values) / 12.0,
                     sum(v == 0 for v in values) / 12.0,
                     len(set(values)) / 12.0,
                     float(values == sorted(values)),
                     float(values == sorted(values, reverse=True))]


def features(current: list[list[int]], target: list[list[int]],
             depth: int) -> torch.Tensor:
    scale = max([1, *(abs(value) for array in current + target
                      for value in array)])
    return torch.tensor([
        [*_array(first, scale), *_array(second, scale),
         float(first == second), depth / 3.0, math.log1p(scale) / 8.0]
        for first, second in zip(current, target)
    ])


def teacher_programs(seed: int = 0) -> list[tuple[str, ...]]:
    """Deduplicate teachers and exclude the target on 24 probe arrays."""
    rng = random.Random(seed)
    probes = [[], [0], [-1, 0, 1], [2, 2, -3, 0, 4, 1]] + [
        [rng.randrange(-9, 10) for _ in range(rng.randrange(4, 13))]
        for _ in range(20)
    ]
    goal = tuple(tuple(sorted({v for v in values if v > 0}))
                 for values in probes)
    seen = {tuple(tuple(values) for values in probes), goal}
    programs: list[tuple[str, ...]] = []
    for length in (1, 2, 3):
        for program in itertools.product(OPS, repeat=length):
            identity = tuple(tuple(execute(program, values))
                             for values in probes)
            if identity not in seen:
                seen.add(identity)
                programs.append(program)
    return programs


def training_data(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build teacher action states without the evaluation example pairs."""
    rng = random.Random(seed)
    # Preserve the original data generator's random-stream position.
    for _ in range(20):
        for _ in range(rng.randrange(4, 13)):
            rng.randrange(-9, 10)
    programs = teacher_programs(seed)
    matrices: list[torch.Tensor] = []
    labels: list[int] = []
    for program in programs:
        for _ in range(6):
            inputs = [[rng.randrange(-9, 10) for _ in
                       range(rng.randrange(5, 13))] for _ in range(3)]
            targets = [execute(program, values) for values in inputs]
            current = [values[:] for values in inputs]
            for depth, op in enumerate((*program, "stop")):
                matrices.append(features(current, targets, depth))
                labels.append(TOKENS.index(op))
                if op != "stop":
                    current = [apply_op(op, values) for values in current]
    return torch.stack(matrices), torch.tensor(labels), len(programs)


class UtilityPolicy(nn.Module):
    """Select each operation, rather than a prewritten complete utility."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = nn.Sequential(nn.Linear(47, 96), nn.Tanh(),
                                         nn.Linear(96, 96), nn.Tanh())
            self.readout = nn.Linear(96, len(TOKENS))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.readout(self.encoder(values).mean(dim=1))

    def fit(self, epochs: int = 40) -> dict[str, Any]:
        x, y, count = training_data()
        opt = torch.optim.Adam(self.parameters(), lr=0.003)
        rng = torch.Generator().manual_seed(100)
        loss_value = 0.0
        for _ in range(epochs):
            order = torch.randperm(len(y), generator=rng)
            for offset in range(0, len(y), 256):
                indices = order[offset:offset + 256]
                loss = nn.functional.cross_entropy(self(x[indices]),
                                                   y[indices])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                loss_value = float(loss.detach())
        return {"teacher_programs": count, "teacher_steps": len(y),
                "epochs": epochs, "loss": loss_value,
                "target_function_excluded_on_probe_suite": True}

    def synthesize(self, examples: Pairs, width: int = 32
                   ) -> tuple[str, ...] | None:
        """Choose a composition using examples only; no held-file feedback."""
        if not examples or width < 1:
            raise ValueError("examples and a positive beam width required")
        inputs = [first for first, _ in examples]
        targets = [second for _, second in examples]
        beam: list[tuple[float, tuple[str, ...], list[list[int]]]] = [
            (0.0, (), inputs)
        ]
        with torch.inference_mode():
            for depth in range(4):
                matrix = torch.stack([features(current, targets, depth)
                                      for _, _, current in beam])
                probs = self(matrix).log_softmax(dim=1)
                following = []
                for index, (score, program, current) in enumerate(beam):
                    choices = sorted(range(len(TOKENS)),
                                     key=lambda j: -float(probs[index, j]))[:4]
                    for choice in choices:
                        op = TOKENS[choice]
                        weight = score + float(probs[index, choice])
                        if op == "stop":
                            if current == targets:
                                return program
                        elif depth < 3:
                            following.append((weight, (*program, op), [
                                apply_op(op, values) for values in current
                            ]))
                following.sort(key=lambda item: -item[0])
                beam = following[:width]
                if not beam:
                    break
        return None
