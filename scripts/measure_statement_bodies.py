"""Compare condition and body transfer with frozen neural controllers."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch

from rsi_programmer.benchmark_policy import state_fingerprint
from rsi_programmer.statement_body_contexts import qualify_bodies
from rsi_programmer.statement_fragment_search import synthesize
from rsi_programmer.statement_policy import StatementPolicy
from rsi_programmer.stateful_tasks import oracle

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-choices", action="store_true")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--budget", type=float, default=0.5)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = ROOT / "fixtures/stateful_tasks_v1.json"
    tasks = json.loads(source.read_text())["tasks"]
    queries = json.loads((ROOT / "artifacts/stateful-queries-v1-20261101" /
                          "protocol.json").read_text())["query_inputs"][:3]
    protocol = {"scope": "Development body-transfer probe, no refit. "
                         "Classical context mining, neural body tokens.",
                "additional_observed_inputs": queries,
                "wall_budget_per_search": args.budget,
                "beam_width": args.width,
                "choices_per_prefix": None if args.all_choices else 4,
                "held_seed": 461037, "held_per_task": 40,
                "task_fixture_sha256": hashlib.sha256(
                    source.read_bytes()).hexdigest(),
                "source_hashes": {str(path.relative_to(ROOT)):
                    hashlib.sha256(path.read_bytes()).hexdigest() for path in (
                        Path(__file__).resolve(),
                        ROOT / "rsi_programmer/statement_body_contexts.py",
                        ROOT / "rsi_programmer/statement_fragment_search.py")}}
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    library = qualify_bodies(
        ROOT / "artifacts/statement-programs-v1-execution-20261101")
    (root / "fragment-qualification.json").write_text(
        json.dumps(library, indent=2) + "\n")
    models = {}
    torch.set_num_threads(1)
    for name, filename, prior in (
            ("qualified", "statement-repair-v1.pt",
             "stateful-programs-v1-execution-20261101"),
            ("broader", "statement-stateful-v1.pt",
             "stateful-curriculum-v1-20261101")):
        data = torch.load(ROOT / "weights" / filename, weights_only=True)
        model = StatementPolicy(seed=0, execution_context=True)
        model.load_state_dict(data["state"])
        model.eval()
        fingerprint = state_fingerprint(model.state_dict(), data["metadata"])
        expected = json.loads((ROOT / "artifacts" / prior /
                               "report.json").read_text())["trained_hash"]
        if fingerprint != expected:
            raise ValueError("body-transfer checkpoint differs from prior")
        models[name] = (model, data["metadata"], fingerprint)
    untrained = StatementPolicy(seed=0, execution_context=True)
    untrained.eval()
    metadata = {"seed": 0, "execution_context": True}
    models["untrained"] = (
        untrained, metadata,
        state_fingerprint(untrained.state_dict(), metadata))
    methods = [("condition-qualified", "qualified", "condition_fragments"),
               ("body-qualified", "qualified", "fragments"),
               ("body-broader", "broader", "fragments"),
               ("body-untrained", "untrained", "fragments")]
    results = []
    for label, model_name, fragments in methods:
        model, metadata, fingerprint = models[model_name]
        rows = []
        for task in tasks:
            name = task["id"]
            observed = [*task["observed"], *[
                (values, oracle(name, values)) for values in queries]]
            chosen = synthesize(model, observed, library[fragments],
                                budget_seconds=args.budget, width=args.width,
                                choices_per_prefix=(
                                    None if args.all_choices else 4))
            rows.append({"id": name, "observed": observed, "chosen": chosen})
            print(json.dumps({"method": label, "task": name,
                              "tokens": chosen["tokens"]}), flush=True)
        if state_fingerprint(model.state_dict(), metadata) != fingerprint:
            raise ValueError("body transfer changed frozen parameters")
        results.append((label, fingerprint, rows))
    rng = random.Random(protocol["held_seed"])
    held = [[rng.randrange(-100, 101)
             for _ in range(rng.randrange(13, 101))] for _ in range(40)]
    for label, fingerprint, rows in results:
        folder = root / label
        folder.mkdir()
        report = {"scope": protocol["scope"], "model_hash": fingerprint,
                  "tasks": len(rows), "rows": rows, "execution_pending": True}
        (folder / "report.json").write_text(
            json.dumps(report, indent=2) + "\n")
        (folder / "held-inputs.json").write_text(json.dumps(held) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
