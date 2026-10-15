#!/usr/bin/env python3
"""Replay proxy vs full eval on historical seeds; print fidelity.

Runs cheap W1-subset screening and full novel-task eval over N
historical candidate seeds, then prints one JSON report with rank
agreement, pair counts, both wall times, and the proven verdict.
The proxy judges nothing until the measured agreement clears its
threshold on enough pairs (see rsi_programmer.proxy).
"""
from __future__ import annotations

import argparse
import json
from typing import Any

from rsi_programmer.proxy import FULL_EPOCHS, ProxyScreen

W1 = "fixtures/rsi_shadow_tasks.json"
NOVEL = "fixtures/rsi_novel_tasks.json"
DEFAULT_SEEDS = 6


def build_report(shadow: str, novel: str, count: int,
                 full_epochs: int) -> dict[str, Any]:
    """Screen N seeds, full-eval them, and score fidelity."""
    if count < 2:
        raise ValueError("need at least two seeds to rank")
    screen = ProxyScreen(shadow, novel)
    cands = [{"id": "h%d" % i, "seed": i} for i in range(count)]
    screened = screen.screen(cands)
    full = screen.full_eval(cands, epochs=full_epochs)
    scores = screened["scores"]
    pact = screen.fidelity(scores, full["gains"])
    pick = screen.select(scores, full["gains"])
    return {"candidates": count, "full_epochs": full_epochs,
            "screen_wall_s": screened["wall_s"],
            "full_wall_s": full["wall_s"],
            "fidelity": pact, "selection": pick}


def main(argv: list[str] | None = None) -> int:
    """Parse flags, print the JSON fidelity report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int,
                        default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int,
                        default=FULL_EPOCHS)
    args = parser.parse_args(argv)
    print(json.dumps(build_report(W1, NOVEL, args.seeds,
                                  args.epochs),
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
