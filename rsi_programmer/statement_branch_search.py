"""Bounded two-branch composition with a shared neural expression hole."""
from __future__ import annotations

import itertools
import time
from typing import Any

import torch

from rsi_programmer.statement_branch_library import instantiate, pattern
from rsi_programmer.statement_execution import execute
from rsi_programmer.statement_policy import StatementPolicy, _prefix
from rsi_programmer.statement_syntax import ARITHMETIC, LEAVES, TOKENS

LEAF = (*LEAVES, "value")


def _contexts(examples: list[tuple[list[int], int]],
              branches: list[dict[str, Any]], first_limit: int
              ) -> list[dict[str, Any]]:
    ranked = sorted(branches, key=lambda item: (sum(
        abs(execute(tuple(item["program"]), values) - expected)
        for values, expected in examples), item["donor"]))[:first_limit]
    contexts = []
    for first in ranked:
        following = sorted(branches, key=lambda item: (
            item["donor"] != first["donor"], item["donor"]))
        for second in following:
            update = second["update"]
            variants = ("add", "sub") if (update[0] in ("add", "sub")
                        and tuple(update[1]) == ("acc",)) else (None,)
            for variant in variants:
                template = pattern(first, second, variant)
                if "$arg" in template:
                    contexts.append({"template": template,
                                     "donors": [first["donor"],
                                                second["donor"]],
                                     "combine": variant})
    return contexts


def synthesize(model: StatementPolicy | None,
               examples: list[tuple[list[int], int]],
               branches: list[dict[str, Any]], width: int = 8,
               budget_seconds: float = 5.0, first_limit: int = 3
               ) -> dict[str, Any]:
    """Host chooses branch pairs; neural scores choose a shared Int argument.

    Argument grammar is a leaf or one binary operation on leaves. Passing
    None uses a classical size-interleaved control over the same contexts.
    No task name, oracle or held rows enter this API.
    """
    if not examples or width < 1 or first_limit < 1 or budget_seconds <= 0:
        raise ValueError("examples and positive search limits required")
    started = time.monotonic()
    contexts = _contexts(examples, branches, first_limit)
    visited = 0
    chosen_context = None

    def attempt(context: dict[str, Any], arg: tuple[str, ...]
                ) -> tuple[str, ...] | None:
        nonlocal visited, chosen_context
        visited += 1
        try:
            tokens = instantiate(context["template"], arg)
            if all(execute(tokens, values) == expected
                   for values, expected in examples):
                chosen_context = {"donors": context["donors"],
                                  "combine": context["combine"], "arg": arg}
                return tokens
        except ValueError:
            pass
        return None

    chosen = None
    if model is None:
        arguments = [(leaf,) for leaf in LEAF] + [
            (op, left, right) for op, left, right in itertools.product(
                ARITHMETIC, LEAF, LEAF)]
        for arg in arguments:
            for context in contexts:
                if time.monotonic() - started >= budget_seconds:
                    break
                chosen = attempt(context, arg)
                if chosen is not None:
                    break
            if chosen is not None or time.monotonic() - started >= (
                    budget_seconds):
                break
    else:
        inputs = [values for values, _ in examples]
        targets = [target for _, target in examples]
        with torch.inference_mode():
            for context in contexts:
                template = context["template"]
                head = template[:template.index("$arg")]
                beam: list[tuple[float, tuple[str, ...]]] = [(0.0, ())]
                for depth in range(3):
                    if time.monotonic() - started >= budget_seconds:
                        break
                    prefix = [(*head, *arg) for _, arg in beam]
                    if not prefix or any(len(p) > 32 for p in prefix):
                        break
                    matrix = torch.stack([model.observed_features(
                        inputs, targets, p) for p in prefix])
                    scores = model(matrix, torch.tensor([_prefix(p)
                                                        for p in prefix]))
                    following = []
                    allowed = (*LEAF, *ARITHMETIC) if depth == 0 else LEAF
                    for index, (weight, arg) in enumerate(beam):
                        logits = scores[index]
                        mask = torch.tensor([token in allowed for token in
                                             TOKENS])
                        probabilities = logits.masked_fill(
                            ~mask, -torch.inf).log_softmax(dim=0)
                        for token in sorted(allowed, key=lambda value:
                                -float(probabilities[TOKENS.index(value)])):
                            child = (*arg, token)
                            score = weight + float(probabilities[
                                TOKENS.index(token)])
                            if (len(child) == 1 and token in LEAF
                                    or len(child) == 3):
                                chosen = attempt(context, child)
                                if chosen is not None:
                                    break
                            else:
                                following.append((score, child))
                        if chosen is not None:
                            break
                    if chosen is not None:
                        break
                    following.sort(key=lambda item: -item[0])
                    beam = following[:width]
                if chosen is not None or time.monotonic() - started >= (
                        budget_seconds):
                    break
    return {"tokens": chosen, "composition": chosen_context,
            "visited": visited, "contexts": len(contexts),
            "wall_seconds": time.monotonic() - started,
            "budget_exhausted": time.monotonic() - started >= budget_seconds}
