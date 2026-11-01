"""Choose well-typed statement tokens from examples and the actual prefix."""
from __future__ import annotations

import random
import time
from typing import Any

import torch
from torch import nn

from rsi_programmer.statement_curriculum import teachers
from rsi_programmer.statement_execution import execute
from rsi_programmer.statement_partial import execution_features
from rsi_programmer.statement_syntax import MAX_TOKENS, TOKENS, analyze
from rsi_programmer.utility_policy import features


def _prefix(tokens: tuple[str, ...]) -> list[int]:
    return [*(TOKENS.index(token) + 1 for token in tokens),
            *([0] * (MAX_TOKENS - len(tokens)))]


class StatementPolicy(nn.Module):
    """Construct loop/branch/update syntax, with no whole-operation tokens."""

    def __init__(self, seed: int = 0,
                 execution_context: bool = False) -> None:
        super().__init__()
        self.execution_context = execution_context
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            size = 51 if execution_context else 47
            self.encoder = nn.Sequential(nn.Linear(size, 64), nn.Tanh(),
                                         nn.Linear(64, 64), nn.Tanh())
            self.embedding = nn.Embedding(len(TOKENS) + 1, 8, padding_idx=0)
            self.decoder = nn.Sequential(
                nn.Linear(64 + 8 * MAX_TOKENS, 128), nn.Tanh(),
                nn.Linear(128, len(TOKENS)))

    def forward(self, pairs: torch.Tensor, prefix: torch.Tensor
                ) -> torch.Tensor:
        context = self.encoder(pairs).mean(dim=1)
        sequence = self.embedding(prefix).flatten(start_dim=1)
        return self.decoder(torch.cat((context, sequence), dim=1))

    def observed_features(self, inputs: list[list[int]], targets: list[int],
                          prefix: tuple[str, ...]) -> torch.Tensor:
        base = features(inputs, [[target] for target in targets], 0)
        if not self.execution_context:
            return base
        extra = torch.tensor(execution_features(prefix, inputs, targets))
        return torch.cat((base, extra), dim=1)

    def fit(self, excluded: list[str], epochs: int = 30,
            stateful: bool = False,
            return_variants: bool = False) -> dict[str, Any]:
        started = time.monotonic()
        programs, _ = teachers(excluded, stateful=stateful,
                               return_variants=return_variants)
        rng = random.Random(681)
        matrices, prefixes, labels = [], [], []
        for program in programs:
            for _ in range(2):
                inputs = [[rng.randrange(-6, 7) for _ in
                           range(rng.randrange(3, 11))] for _ in range(3)]
                targets = [execute(program, values) for values in inputs]
                for index, token in enumerate(program):
                    matrices.append(self.observed_features(
                        inputs, targets, program[:index]))
                    prefixes.append(_prefix(program[:index]))
                    labels.append(TOKENS.index(token))
        x, p, y = torch.stack(matrices), torch.tensor(prefixes), torch.tensor(
            labels)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.003)
        generator = torch.Generator().manual_seed(100)
        loss_value = 0.0
        for _ in range(epochs):
            order = torch.randperm(len(y), generator=generator)
            for start in range(0, len(order), 256):
                indices = order[start:start + 256]
                loss = nn.functional.cross_entropy(
                    self(x[indices], p[indices]), y[indices])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_value = float(loss.detach())
        self.eval()
        return {"teacher_functions": len(programs), "teacher_states": len(y),
                "epochs": epochs, "loss": loss_value,
                "wall_seconds": time.monotonic() - started,
                "execution_context": self.execution_context,
                "curriculum": ("return-variants-v1" if return_variants else
                               "stateful-v1" if stateful else "basic-v1"),
                "excluded_functions": excluded,
                "exclusion_scope": "equivalence on 24 fixed probe arrays"}

    def synthesize(self, examples: list[tuple[list[int], int]], width: int = 32
                   ) -> dict[str, Any]:
        """Choose syntax using observed examples only, without held retries."""
        inputs = [values for values, _ in examples]
        targets = [expected for _, expected in examples]
        beam: list[tuple[float, tuple[str, ...]]] = [(0.0, ())]
        visited = completions = 0
        started = time.monotonic()
        with torch.inference_mode():
            for _ in range(MAX_TOKENS):
                prefixes = torch.tensor([_prefix(tokens)
                                         for _, tokens in beam])
                matrix = torch.stack([self.observed_features(
                    inputs, targets, tokens) for _, tokens in beam])
                scores = self(matrix, prefixes)
                following = []
                for index, (weight, tokens) in enumerate(beam):
                    _, allowed = analyze(tokens)
                    if not allowed:
                        continue
                    mask = torch.tensor([token in allowed for token in TOKENS])
                    probabilities = scores[index].masked_fill(
                        ~mask, -torch.inf).log_softmax(dim=0)
                    choices = sorted(
                        (TOKENS.index(token) for token in allowed),
                        key=lambda i: -float(probabilities[i]))[:4]
                    for choice in choices:
                        visited += 1
                        child = (*tokens, TOKENS[choice])
                        score = weight + float(probabilities[choice])
                        if TOKENS[choice] == "stop":
                            completions += 1
                            try:
                                correct = all(
                                    execute(child, values) == expected
                                    for values, expected in examples)
                            except ValueError:
                                correct = False
                            if correct:
                                return {"tokens": child, "visited": visited,
                                        "completions": completions,
                                        "wall_seconds": time.monotonic()
                                        - started}
                        else:
                            following.append((score, child))
                following.sort(key=lambda item: -item[0])
                beam = following[:width]
                if not beam:
                    break
        return {"tokens": None, "visited": visited, "completions": completions,
                "wall_seconds": time.monotonic() - started}
