"""First bounded branch-composition probe with frozen neural weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import torch

from rsi_programmer.benchmark_policy import state_fingerprint
from rsi_programmer.statement_branch_library import qualify_branches
from rsi_programmer.statement_branch_search import synthesize
from rsi_programmer.statement_policy import StatementPolicy
from scripts.measure_context_growth import _model

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--extra-donor", type=Path, action="append",
                        default=[])
    parser.add_argument("--held-seed", type=int, default=461053)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    fixture = json.loads(
        (ROOT / "fixtures/stateful_tasks_v1.json").read_text())
    tasks = fixture["tasks"][2:]
    queries = json.loads((ROOT / "artifacts/stateful-queries-v1-20261101" /
                          "protocol.json").read_text())["query_inputs"][:3]
    protocol = {
        "scope": "Development two-branch template composition. Host copies "
                 "branch pairs and enumerates add/sub variants; neural "
                 "scores choose a shared expression. No refit.",
        "task_ids": [task["id"] for task in tasks],
        "methods": ["trained", "untrained", "classical"],
        "additional_observed_inputs": queries,
        "width": args.width, "first_donors_by_residual": 3,
        "argument_grammar": "leaf or binary operation on two leaves",
        "wall_budget_per_task": 5.0, "held_seed": args.held_seed,
        "held_per_task": 40,
        "extra_donor_roots": [str(path) for path in args.extra_donor],
        "source_hashes": {name: hashlib.sha256(
            (ROOT / name).read_bytes()).hexdigest() for name in (
                "scripts/measure_branch_composition.py",
                "rsi_programmer/statement_branch_library.py",
                "rsi_programmer/statement_branch_search.py",
                "rsi_programmer/statement_policy.py")}}
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    donors = ROOT / "artifacts/statement-context-repair-v1-20261101"
    library = qualify_branches(donors)
    library["extra_qualifications"] = []
    for path in args.extra_donor:
        extra = qualify_branches(path.resolve())
        library["branches"].extend(extra["branches"])
        library["extra_qualifications"].append(extra["qualification"])
    (root / "branch-qualification.json").write_text(
        json.dumps(library, indent=2) + "\n")
    torch.set_num_threads(1)
    model, metadata, expected = _model(
        "statement-return-v1.pt",
        ROOT / "artifacts/statement-return-v1-20261101")
    untrained = StatementPolicy(seed=0, execution_context=True)
    untrained.eval()
    initial = state_fingerprint(untrained.state_dict(), {"seed": 0})
    from rsi_programmer.statement_curriculum import oracle
    results = []
    for method, active in (("trained", model), ("untrained", untrained),
                           ("classical", None)):
        rows = []
        for task in tasks:
            name = task["id"]
            observed = [*task["observed"], *[
                (values, oracle(name, values)) for values in queries]]
            cpu = time.process_time()
            chosen = synthesize(active, observed, library["branches"],
                                width=args.width)
            rows.append({"id": name, "observed": observed, "chosen": chosen,
                         "cpu_seconds": time.process_time() - cpu})
            print(json.dumps({"method": method, "id": name, **chosen}),
                  flush=True)
        results.append((method, rows))
    if state_fingerprint(model.state_dict(), metadata) != expected:
        raise ValueError("composition changed trained parameters")
    if state_fingerprint(untrained.state_dict(), {"seed": 0}) != initial:
        raise ValueError("composition changed untrained control")
    # All controller decisions end before any new held input is created.
    rng = random.Random(protocol["held_seed"])
    held = [[], [0], [1], [-1], [100] * 256, [-100] * 256] + [
        [rng.randrange(-100, 101) for _ in range(rng.randrange(13, 101))]
        for _ in range(34)]
    for method, rows in results:
        folder = root / method
        folder.mkdir()
        report = {"scope": protocol["scope"], "method": method,
                  "rows": rows, "tasks": len(rows), "held_per_task": 40,
                  "execution_pending": True,
                  "model_hash": expected if method == "trained" else
                  initial if method == "untrained" else None}
        (folder / "report.json").write_text(
            json.dumps(report, indent=2) + "\n")
        (folder / "held-inputs.json").write_text(json.dumps(held) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
