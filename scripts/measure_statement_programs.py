"""Learn statement syntax, then execute every generated Python utility."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import tempfile

import torch

from rsi_programmer.benchmark_policy import state_fingerprint
from rsi_programmer.statement_curriculum import oracle
from rsi_programmer.statement_execution import execute
from rsi_programmer.statement_policy import StatementPolicy
from rsi_programmer.statement_source import curlee_source, python_source
from rsi_programmer.statement_syntax import TOKENS
from scripts.measure_useful_program import run_cli

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_HASH = (
    "d732492f5cc2ade6412e8f2ca1bb46a98d442b9a9cbb88aa95c61d87da362fb0")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, choices=(30, 100))
    parser.add_argument("--execution-features", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    raw = (ROOT / "fixtures/statement_tasks_v1.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != FIXTURE_HASH:
        raise ValueError("frozen statement task suite changed")
    spec = json.loads(raw)
    torch.set_num_threads(1)
    model = StatementPolicy(seed=spec["seed"],
                            execution_context=args.execution_features)
    metadata = {"seed": spec["seed"], "max_tokens": spec["max_tokens"],
                "tokens": TOKENS,
                "execution_context": args.execution_features}
    initial_hash = state_fingerprint(model.state_dict(), metadata)
    initial = {task["id"]: model.synthesize(task["observed"])
               for task in spec["tasks"]}
    epochs = args.epochs if args.epochs is not None else spec["epochs"]
    fit = model.fit([task["id"] for task in spec["tasks"]], epochs)
    fingerprint = state_fingerprint(model.state_dict(), metadata)
    chosen = {task["id"]: model.synthesize(task["observed"])
              for task in spec["tasks"]}
    print(json.dumps({"fit": fit, "programs": chosen}), flush=True)
    rng = random.Random(spec["held_seed"])
    held = [[], [0], [-3, -1, 0], [2, 2, 0, -2, -2, 3],
            list(range(100, -101, -1))] + [
        [rng.randrange(-100, 101) for _ in range(rng.randrange(1, 101))]
        for _ in range(spec["held_per_task"] - 5)]
    (root / "held-inputs.json").write_text(json.dumps(held) + "\n")
    rows = []
    for task in spec["tasks"]:
        name = task["id"]
        tokens = chosen[name]["tokens"]
        expected = [oracle(name, values) for values in held]
        entry = {"id": name, "chosen": chosen[name],
                 "untrained": initial[name], "observed": task["observed"],
                 "expected": expected, "python_records": [],
                 "accepted": False}
        if tokens is not None:
            folder = root / name
            for directory in ("python", "curlee", "inputs"):
                (folder / directory).mkdir(parents=True)
            python = python_source(tokens)
            curlee = curlee_source(tokens)
            (folder / "python/clean_readings.py").write_text(python)
            (folder / "curlee/process.curlee").write_text(curlee)
            entry["python_sha256"] = hashlib.sha256(
                python.encode()).hexdigest()
            entry["curlee_sha256"] = hashlib.sha256(
                curlee.encode()).hexdigest()
            with tempfile.TemporaryDirectory(prefix="rsi-statements-") as temp:
                output = Path(temp)
                for index, (values, answer) in enumerate(zip(held, expected)):
                    filename = f"{index:03}.json"
                    (folder / "inputs" / filename).write_text(
                        json.dumps(values) + "\n")
                    result = run_cli(folder / "python", folder / "inputs",
                                     output, filename, filename)
                    path = output / filename
                    actual = (json.loads(path.read_text())
                              if path.exists() else None)
                    try:
                        interpreted = execute(tokens, values)
                    except ValueError:
                        interpreted = None
                    entry["python_records"].append({
                        "case": index, "exit_code": result.returncode,
                        "actual": actual, "expected": answer,
                        "interpreted": interpreted,
                        "correct": result.returncode == 0
                        and actual == answer and interpreted == answer,
                        "stderr": result.stderr})
            entry["accepted"] = all(row["correct"]
                                    for row in entry["python_records"])
        rows.append(entry)
        print(json.dumps({"id": name, "tokens": tokens,
                          "accepted": entry["accepted"]}), flush=True)
    if fingerprint != state_fingerprint(model.state_dict(), metadata):
        raise ValueError("evaluation changed model weights")
    report = {"scope": "Statement-construction development only. All six "
                       "target functions excluded on teacher probes. "
                       "Learner chooses loop, branch and expression tokens; "
                       "emitter supplies syntax lowering and CLI plumbing.",
              "fixture_sha256": FIXTURE_HASH, "fit": fit,
              "initial_hash": initial_hash, "trained_hash": fingerprint,
              "tasks": len(rows), "accepted": sum(row["accepted"]
                                                    for row in rows),
              "rows": rows}
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"tasks": len(rows), "accepted": report["accepted"]}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
