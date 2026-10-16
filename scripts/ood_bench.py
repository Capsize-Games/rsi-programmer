#!/usr/bin/env python3
"""Frozen-policy OOD benchmark run (Issue 12).

Trains a narrow policy (overfit on two shadow tasks) and a
broad one (all shadow tasks), then evaluates both frozen on
the unmirrored twin reference, the mirrored plain stratum,
and the full-IR boundary stratum with weights write-locked.
Prints the expression gap plus the v2.4.0 section 9.1
caveat; promotes nothing.
"""
from __future__ import annotations

import json
from typing import Any

from rsi_programmer.ood import benchmark, contrast_tensors
from rsi_programmer.ood import guidance_similarity
from rsi_programmer.policy import SearchPolicy

SHADOW = "fixtures/rsi_shadow_tasks.json"
TRAIN = "fixtures/rsi_plain_train.json"
TWIN = "fixtures/rsi_ood_twin.json"
PLAIN = "fixtures/rsi_ood_plain.json"
FULL = "fixtures/rsi_ood_tasks.json"
EPOCHS = 40
BUDGET = 3000


def _load(path: str) -> list[dict[str, Any]]:
    """Task list from one fixture file."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["tasks"]


def main() -> int:
    """Train narrow/broad, run all frozen strata, report."""
    twin_tasks = _load(TWIN)
    plain_tasks = _load(PLAIN)
    pairs = contrast_tensors(twin_tasks, plain_tasks)
    maker = SearchPolicy(seed=0)
    shadow = maker.trace_groups(SHADOW)
    extra = SearchPolicy(seed=0).trace_groups(TRAIN)
    narrow = SearchPolicy(seed=0)
    narrow.fit(shadow[:2], EPOCHS)
    broad = SearchPolicy(seed=0)
    broad.fit(shadow + extra, EPOCHS, contrast=pairs)
    ref = benchmark(narrow, broad, twin_tasks, BUDGET,
                    name="ood-twin", order="score", widen=6)
    mir = benchmark(narrow, broad, plain_tasks, BUDGET,
                    name="ood-plain", order="score", widen=6)
    benchmark(narrow, broad, _load(FULL), BUDGET,
              name="ood-full", order="score", widen=6)
    for who in ("narrow_solve", "broad_solve"):
        print("shift-bite %s: twin=%.3f mirrored=%.3f "
              "bite=%+.3f" % (who, ref[who], mir[who],
                               ref[who] - mir[who]))
    for who, pol in (("narrow", narrow), ("broad", broad)):
        sim = guidance_similarity(pol, twin_tasks, plain_tasks)
        print("guidance-blindness %s: mean-cosine=%.4f "
              "(1.0 = blind to the mirror)" % (who,
                                               sim["mean"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
