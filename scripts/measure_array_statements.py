"""First bounded development trial of neural list-statement construction."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random

import torch

from rsi_programmer.array_policy import ArrayPolicy
from rsi_programmer.array_tasks import CONTROL, TASKS, oracle
from rsi_programmer.benchmark_policy import state_fingerprint
from rsi_programmer.program_cost import CostClock


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    torch.set_num_threads(1)
    clock = CostClock()
    trained = ArrayPolicy(seed=0)
    fit = trained.fit()
    trained_hash = state_fingerprint(trained.state_dict(), {"seed": 0})
    control = ArrayPolicy(seed=0)
    control.eval()
    untrained_hash = state_fingerprint(control.state_dict(), {"seed": 0})
    observed = [[-3, 0, 1, 4, -1], [2, -2, 5, 1], [0, 3, -4, 2]]
    rows = []
    for label, model in (("neural", trained), ("untrained", control)):
        for name in TASKS:
            examples = [(values, oracle(name, values)) for values in observed]
            request = CostClock()
            chosen = model.synthesize(examples, width=32, seconds=3.0)
            rows.append({"method": label, "id": name, "chosen": chosen,
                         "observed": examples, "cost": request.elapsed()})
            print(json.dumps({"method": label, "id": name,
                              "emitted": chosen["tokens"] is not None}),
                  flush=True)
    for name in TASKS:
        rows.append({"method": "supplied_control", "id": name,
                     "chosen": {"tokens": CONTROL[name]}})
    assert trained_hash == state_fingerprint(trained.state_dict(), {"seed": 0})
    assert untrained_hash == state_fingerprint(control.state_dict(),
                                               {"seed": 0})
    # No synthesis or repair occurs after these independent inputs are made.
    rng = random.Random(723031)
    held = [[], [0], [-1], [1], [100] * 256, [-100] * 256] + [
        [rng.randrange(-100, 101) for _ in range(rng.randrange(13, 101))]
        for _ in range(14)]
    (root / "held-inputs.json").write_text(json.dumps(held) + "\n")
    sources = [*Path("rsi_programmer").glob("array_*.py"),
               Path(__file__)]
    report = {"scope": "Development trial: token construction of bounded "
                       "list statements. Supplied witnesses qualify the "
                       "backend only; they are not learner successes.",
              "rows": rows, "fit": fit, "trained_hash": trained_hash,
              "untrained_hash": untrained_hash, "cost": clock.elapsed(),
              "source_sha256": {str(path): hashlib.sha256(
                  path.read_bytes()).hexdigest() for path in sources},
              "execution_pending": True}
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"fit": fit, "cost": report["cost"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
