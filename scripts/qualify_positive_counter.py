"""Qualify a previously trained primitive before using it as a donor."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch

from rsi_programmer.benchmark_policy import state_fingerprint
from rsi_programmer.stateful_tasks import oracle
from rsi_programmer.statement_curriculum import teachers
from rsi_programmer.statement_execution import execute
from rsi_programmer.statement_fragment_library import qualify
from rsi_programmer.statement_fragment_search import synthesize
from scripts.measure_context_growth import _model

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    tasks = json.loads((ROOT / "fixtures/statement_tasks_v1.json").read_text())
    protocol = {"scope": "Qualify a training primitive; not an unseen "
                         "function or new learning result.",
                "held_seed": 461059, "held_per_task": 40,
                "source_hashes": {name: hashlib.sha256(
                    (ROOT / name).read_bytes()).hexdigest() for name in (
                        "scripts/qualify_positive_counter.py",
                        "rsi_programmer/stateful_tasks.py",
                        "rsi_programmer/statement_fragment_search.py")}}
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    training, probes = teachers([task["id"] for task in tasks["tasks"]])
    signature = tuple(oracle("positive_count", values) for values in probes)
    protocol["present_in_training_on_probes"] = any(
        tuple(execute(program, values) for values in probes) == signature
        for program in training)
    donors = ROOT / "artifacts/statement-programs-v1-execution-20261101"
    library = qualify(donors)
    torch.set_num_threads(1)
    model, metadata, fingerprint = _model("statement-repair-v1.pt", donors)
    inputs = [values for values, _ in tasks["tasks"][0]["observed"]]
    observed = [(values, oracle("positive_count", values))
                for values in inputs]
    chosen = synthesize(model, observed, library["fragments"],
                        budget_seconds=1.0)
    if state_fingerprint(model.state_dict(), metadata) != fingerprint:
        raise ValueError("primitive qualification changed weights")
    rng = random.Random(protocol["held_seed"])
    held = [[], [0], [1], [-1], [100] * 256, [-100] * 256] + [
        [rng.randrange(-100, 101) for _ in range(rng.randrange(13, 101))]
        for _ in range(34)]
    report = {"scope": protocol["scope"], "trained_hash": fingerprint,
              "rows": [{"id": "positive_count", "observed": observed,
                        "chosen": chosen, "role": "training_primitive"}],
              "tasks": 1, "held_per_task": 40, "execution_pending": True}
    for name, data in (("report.json", report), ("held-inputs.json", held),
                       ("protocol.json", protocol)):
        (root / name).write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({**chosen, "primitive_in_training":
                      protocol["present_in_training_on_probes"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
