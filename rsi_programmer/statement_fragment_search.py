"""Neural filling of typed holes in verified reusable statement contexts."""
from __future__ import annotations

import time
from typing import Any

import torch

from rsi_programmer.statement_execution import execute
from rsi_programmer.statement_fragment_rank import rank
from rsi_programmer.statement_policy import StatementPolicy, _prefix
from rsi_programmer.statement_syntax import MAX_TOKENS, TOKENS, analyze, parse


def synthesize(model: StatementPolicy, examples: list[tuple[list[int], int]],
               fragments: list[dict[str, Any]], width: int = 32,
               rank_contexts: bool = False,
               budget_seconds: float | None = None,
               choices_per_prefix: int | None = 4
               ) -> dict[str, Any]:
    """Explore contexts classically; the neural controller fills each hole.

    This is fragment-guided synthesis, not learned abstraction discovery.
    Copied syntax, including the return/stop suffix, comes from the donor.
    No held result or task identifier enters the search.
    """
    if width < 1 or (budget_seconds is not None and budget_seconds <= 0):
        raise ValueError("positive width and search budget required")
    if choices_per_prefix is not None and choices_per_prefix < 1:
        raise ValueError("positive choice count or None required")
    started = time.monotonic()
    if rank_contexts:
        fragments = rank(model, examples, fragments)
    inputs = [values for values, _ in examples]
    targets = [expected for _, expected in examples]
    visited = completions = 0
    tried = []
    with torch.inference_mode():
        for fragment in fragments:
            prefix = tuple(fragment["prefix"])
            suffix = tuple(fragment["suffix"])
            beam: list[tuple[float, tuple[str, ...]]] = [(0.0, prefix)]
            for _ in range(MAX_TOKENS - len(prefix) - len(suffix)):
                if (budget_seconds is not None
                        and time.monotonic() - started >= budget_seconds):
                    return {"tokens": None, "fragment": None,
                            "visited": visited, "completions": completions,
                            "wall_seconds": time.monotonic() - started,
                            "tried_contexts": tried, "budget_exhausted": True}
                matrix = torch.stack([model.observed_features(
                    inputs, targets, tokens) for _, tokens in beam])
                scores = model(matrix, torch.tensor([_prefix(tokens)
                                                    for _, tokens in beam]))
                following = []
                for index, (score, tokens) in enumerate(beam):
                    _, allowed = analyze(tokens)
                    mask = torch.tensor([token in allowed for token in TOKENS])
                    probabilities = scores[index].masked_fill(
                        ~mask, -torch.inf).log_softmax(dim=0)
                    choices = sorted(
                        (TOKENS.index(token) for token in allowed),
                        key=lambda i: -float(probabilities[i]))
                    choices = choices[:choices_per_prefix]
                    for choice in choices:
                        visited += 1
                        child = (*tokens, TOKENS[choice])
                        if len(child) + len(suffix) > MAX_TOKENS:
                            continue
                        weight = score + float(probabilities[choice])
                        full = (*child, *suffix)
                        try:
                            parse(full)
                        except ValueError:
                            following.append((weight, child))
                            continue
                        completions += 1
                        try:
                            correct = all(execute(full, values) == expected
                                          for values, expected in examples)
                        except ValueError:
                            correct = False
                        if correct:
                            return {"tokens": full, "fragment": fragment["id"],
                                    "donors": fragment["donors"],
                                    "visited": visited,
                                    "completions": completions,
                                    "wall_seconds": time.monotonic() - started,
                                    "tried_contexts": tried + [fragment["id"]]}
                following.sort(key=lambda item: -item[0])
                beam = following[:width]
                if not beam:
                    break
            tried.append(fragment["id"])
    return {"tokens": None, "fragment": None, "visited": visited,
            "completions": completions,
            "wall_seconds": time.monotonic() - started,
            "tried_contexts": tried}
