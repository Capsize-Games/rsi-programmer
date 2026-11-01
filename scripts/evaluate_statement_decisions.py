"""Execute already-frozen program choices on independent file inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

from rsi_programmer.statement_curriculum import oracle
from rsi_programmer.statement_execution import execute
from rsi_programmer.statement_source import curlee_source, python_source


def evaluate(root: Path, decisions: list[dict[str, Any]],
              inputs: list[list[int]]) -> list[dict[str, Any]]:
    """Evaluate fixed choices without a solver or repair path."""
    if not inputs or not decisions:
        raise ValueError("nonempty program decisions and held inputs required")
    from scripts.measure_useful_program import run_cli

    rows = []
    for decision in decisions:
        name, chosen = decision["id"], decision["chosen"]
        tokens = chosen["tokens"]
        entry = {**decision, "python_records": [], "accepted": False}
        if tokens is not None:
            tokens = tuple(tokens)
            folder = root / name
            for directory in ("python", "curlee"):
                (folder / directory).mkdir(parents=True, exist_ok=True)
            python, curlee = python_source(tokens), curlee_source(tokens)
            (folder / "python/clean_readings.py").write_text(python)
            (folder / "curlee/process.curlee").write_text(curlee)
            entry["python_sha256"] = hashlib.sha256(
                python.encode()).hexdigest()
            entry["curlee_sha256"] = hashlib.sha256(
                curlee.encode()).hexdigest()
            with tempfile.TemporaryDirectory(prefix="rsi-held-") as temp:
                work = Path(temp)
                input_dir, output_dir = work / "inputs", work / "outputs"
                input_dir.mkdir()
                output_dir.mkdir()
                for index, values in enumerate(inputs):
                    filename = f"{index:03}.json"
                    (input_dir / filename).write_text(
                        json.dumps(values) + "\n")
                    result = run_cli(folder / "python", input_dir, output_dir,
                                     filename, filename)
                    path = output_dir / filename
                    actual = (json.loads(path.read_text())
                              if path.exists() else None)
                    expected = oracle(name, values)
                    try:
                        interpreted = execute(tokens, values)
                    except ValueError:
                        interpreted = None
                    entry["python_records"].append({
                        "case": index, "exit_code": result.returncode,
                        "actual": actual, "expected": expected,
                        "interpreted": interpreted,
                        "correct": result.returncode == 0
                        and type(actual) is int and actual == expected
                        and interpreted == expected, "stderr": result.stderr})
            entry["accepted"] = (
                len(entry["python_records"]) == len(inputs)
                and all(record["correct"]
                        for record in entry["python_records"]))
        rows.append(entry)
        print(json.dumps({"id": name, "accepted": entry["accepted"]}),
              flush=True)
    return rows
