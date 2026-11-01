"""One component-preserving fit, then freeze all programming decisions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch

from rsi_programmer.benchmark_policy import state_fingerprint
from rsi_programmer.statement_body_contexts import qualify_bodies
from rsi_programmer.statement_curriculum import oracle
from rsi_programmer.statement_fragment_search import synthesize
from rsi_programmer.statement_policy import StatementPolicy
from rsi_programmer.statement_syntax import TOKENS

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    old = json.loads((ROOT / "fixtures/statement_tasks_v1.json").read_text())
    new = json.loads((ROOT / "fixtures/stateful_tasks_v1.json").read_text())
    tasks = [*old["tasks"], *new["tasks"]]
    queries = json.loads((ROOT / "artifacts/stateful-queries-v1-20261101" /
                          "protocol.json").read_text())["query_inputs"][:3]
    protocol = {
        "scope": "Development component-transfer curriculum; complete "
                 "targets excluded on probes, components present in other "
                 "functions. No claim of algorithm invention.",
        "epochs": 30, "seed": 0, "beam_width": 128,
        "choices_per_prefix": None, "wall_seconds_per_search": 1.0,
        "additional_observed_inputs": queries,
        "held_seed": 461041, "held_per_task": 40,
        "source_hashes": {name: hashlib.sha256(
            (ROOT / name).read_bytes()).hexdigest() for name in (
                "rsi_programmer/statement_curriculum.py",
                "rsi_programmer/statement_stateful_curriculum.py",
                "rsi_programmer/statement_return_curriculum.py",
                "rsi_programmer/statement_policy.py",
                "rsi_programmer/statement_fragment_search.py",
                "rsi_programmer/statement_body_contexts.py",
                "scripts/measure_return_curriculum.py")}}
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    library = qualify_bodies(
        ROOT / "artifacts/statement-programs-v1-execution-20261101")
    torch.set_num_threads(1)
    model = StatementPolicy(seed=0, execution_context=True)
    fit = model.fit([task["id"] for task in tasks], epochs=30,
                    stateful=True, return_variants=True)
    metadata = {"seed": 0, "tokens": TOKENS, "max_tokens": 32,
                "execution_context": True, "curriculum": "return-variants-v1"}
    fingerprint = state_fingerprint(model.state_dict(), metadata)
    torch.save({"state": model.state_dict(), "metadata": metadata,
                "fit": fit, "trained_hash": fingerprint},
               ROOT / "weights/statement-return-v1.pt")
    print(json.dumps({"fit": fit, "trained_hash": fingerprint}), flush=True)
    rows = []
    for task in tasks:
        name = task["id"]
        observed = [*task["observed"], *[
            (values, oracle(name, values)) for values in queries]]
        chosen = synthesize(model, observed, library["fragments"],
                            width=128, budget_seconds=1.0,
                            choices_per_prefix=None)
        rows.append({"id": name, "observed": observed, "chosen": chosen,
                     "role": "retention" if task in old["tasks"] else "new"})
        print(json.dumps({"id": name, "tokens": chosen["tokens"]}), flush=True)
    if state_fingerprint(model.state_dict(), metadata) != fingerprint:
        raise ValueError("program construction changed frozen parameters")
    rng = random.Random(protocol["held_seed"])
    held = [[rng.randrange(-100, 101)
             for _ in range(rng.randrange(13, 101))] for _ in range(40)]
    report = {"scope": protocol["scope"], "fit": fit, "metadata": metadata,
              "trained_hash": fingerprint, "tasks": len(rows), "rows": rows,
              "held_per_task": len(held), "execution_pending": True}
    for name, data in (("report.json", report), ("held-inputs.json", held),
                       ("fragment-qualification.json", library)):
        (root / name).write_text(json.dumps(data, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
