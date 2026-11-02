"""Neural token construction for bounded list-producing statements."""
from __future__ import annotations

import random
import time
from typing import Any

import torch
from torch import nn

from rsi_programmer.array_curriculum import teachers
from rsi_programmer.array_execution import execute
from rsi_programmer.array_partial import outputs
from rsi_programmer.array_syntax import MAX_TOKENS, TOKENS, analyze
from rsi_programmer.utility_policy import features


def prefix_ids(tokens: tuple[str, ...]) -> list[int]:
    return [*(TOKENS.index(token) + 1 for token in tokens),
            *([0] * (MAX_TOKENS - len(tokens)))]


class ArrayPolicy(nn.Module):
    """Predict individual syntax tokens from list examples and a prefix."""

    def __init__(self, seed: int = 0,
                 execution_context: bool = False) -> None:
        super().__init__()
        self.execution_context = execution_context
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            size = 95 if execution_context else 47
            self.encoder = nn.Sequential(nn.Linear(size, 64), nn.Tanh(),
                                         nn.Linear(64, 64), nn.Tanh())
            self.embedding = nn.Embedding(len(TOKENS) + 1, 8, padding_idx=0)
            self.decoder = nn.Sequential(
                nn.Linear(64 + 8 * MAX_TOKENS, 128), nn.Tanh(),
                nn.Linear(128, len(TOKENS)))

    def forward(self, pairs: torch.Tensor, prefix: torch.Tensor
                ) -> torch.Tensor:
        encoded = self.encoder(pairs).mean(dim=1)
        sequence = self.embedding(prefix).flatten(start_dim=1)
        return self.decoder(torch.cat((encoded, sequence), dim=1))

    def observed_features(self, inputs: list[list[int]],
                          targets: list[list[int]], prefix: tuple[str, ...]
                          ) -> torch.Tensor:
        base = features(inputs, targets, 0)
        if not self.execution_context:
            return base
        actual, valid = outputs(prefix, inputs)
        return torch.cat((base, features(actual, targets, 0),
                          torch.tensor(valid).unsqueeze(1)), dim=1)

    def fit(self, epochs: int = 30) -> dict[str, Any]:
        started = time.monotonic()
        programs = teachers()
        rng = random.Random(723023)
        matrices, prefixes, labels = [], [], []
        rejected_draws = 0
        for tokens in programs:
            accepted = 0
            for _ in range(32):
                inputs = [[rng.randrange(-6, 7) for _ in
                           range(rng.randrange(3, 10))] for _ in range(3)]
                try:
                    targets = [execute(tokens, values) for values in inputs]
                except ValueError:
                    rejected_draws += 1
                    continue
                matrix = features(inputs, targets, 0)
                for index, token in enumerate(tokens):
                    matrices.append(self.observed_features(
                        inputs, targets, tokens[:index])
                        if self.execution_context else matrix)
                    prefixes.append(prefix_ids(tokens[:index]))
                    labels.append(TOKENS.index(token))
                accepted += 1
                if accepted == 2:
                    break
            if accepted != 2:
                raise ValueError("teacher lacks two valid training draws")
        x, p = torch.stack(matrices), torch.tensor(prefixes)
        y = torch.tensor(labels)
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
        return {"teacher_functions": len(programs), "states": len(y),
                "epochs": epochs, "loss": loss_value,
                "rejected_invalid_training_draws": rejected_draws,
                "execution_context": self.execution_context,
                "wall_seconds": time.monotonic() - started,
                "exclusion": "Three complete development functions "
                "excluded on 24 fixed probes; components may occur."}

    def synthesize(self, pairs: list[tuple[list[int], list[int]]],
                   width: int = 32, seconds: float = 3.0,
                   token_choices: int = 4) -> dict[str, Any]:
        if not pairs or width < 1 or seconds <= 0 or token_choices < 1:
            raise ValueError("examples and positive search limits required")
        inputs = [x for x, _ in pairs]
        targets = [y for _, y in pairs]
        beam: list[tuple[float, tuple[str, ...]]] = [(0.0, ())]
        started = time.monotonic()
        visited = completions = 0
        with torch.inference_mode():
            for _ in range(MAX_TOKENS):
                if time.monotonic() - started >= seconds:
                    break
                matrix = torch.stack([
                    self.observed_features(inputs, targets, t)
                    for _, t in beam])
                scores = self(matrix, torch.tensor(
                    [prefix_ids(t) for _, t in beam]))
                following = []
                for index, (weight, tokens) in enumerate(beam):
                    _, allowed = analyze(tokens)
                    if not allowed:
                        continue
                    mask = torch.tensor([t in allowed for t in TOKENS])
                    probabilities = scores[index].masked_fill(
                        ~mask, -torch.inf).log_softmax(dim=0)
                    choices = sorted((TOKENS.index(t) for t in allowed),
                                     key=lambda j: -float(probabilities[j]))
                    choices = choices[:token_choices]
                    for choice in choices:
                        visited += 1
                        child = (*tokens, TOKENS[choice])
                        score = weight + float(probabilities[choice])
                        if TOKENS[choice] == "stop":
                            completions += 1
                            try:
                                correct = all(execute(child, x) == y
                                              for x, y in pairs)
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
        return {"tokens": None, "visited": visited,
                "completions": completions,
                "wall_seconds": time.monotonic() - started}
