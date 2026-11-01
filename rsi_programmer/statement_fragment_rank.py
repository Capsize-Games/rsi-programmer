"""Rank reusable contexts using the existing neural statement policy."""
from __future__ import annotations

from typing import Any

import torch

from rsi_programmer.statement_policy import StatementPolicy, _prefix
from rsi_programmer.statement_syntax import TOKENS, analyze


def rank(model: StatementPolicy, examples: list[tuple[list[int], int]],
         fragments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score copied context tokens with a fixed, target-independent hole."""
    if not fragments:
        return []
    inputs = [values for values, _ in examples]
    targets = [expected for _, expected in examples]
    states, labels, owners = [], [], []
    for index, fragment in enumerate(fragments):
        prefix = tuple(fragment["prefix"])
        # This fixed condition is used only to score the copied structure.
        program = (*prefix, "eq", "acc", "acc", *fragment["suffix"])
        for offset, token in enumerate(program):
            if len(prefix) <= offset < len(prefix) + 3:
                continue
            states.append(program[:offset])
            labels.append(TOKENS.index(token))
            owners.append(index)
    with torch.inference_mode():
        matrix = torch.stack([model.observed_features(inputs, targets, tokens)
                              for tokens in states])
        logits = model(matrix, torch.tensor([_prefix(tokens)
                                            for tokens in states]))
        choices = [set(analyze(prefix)[1]) for prefix in states]
        masks = torch.tensor([[token in allowed for token in TOKENS]
                              for allowed in choices])
        scores = logits.masked_fill(~masks, -torch.inf).log_softmax(dim=1)
        totals = [0.0] * len(fragments)
        for index, (owner, label) in enumerate(zip(owners, labels)):
            totals[owner] += float(scores[index, label])
    order = sorted(range(len(fragments)), key=lambda i: -totals[i])
    return [{**fragments[index], "neural_context_score": totals[index]}
            for index in order]
