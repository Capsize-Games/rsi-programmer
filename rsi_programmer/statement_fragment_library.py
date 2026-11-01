"""Qualify unchanged donor evidence before promoting reusable code."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

from rsi_programmer.statement_curriculum import oracle
from rsi_programmer.statement_execution import execute
from rsi_programmer.statement_fragments import mine
from rsi_programmer.statement_source import curlee_harness, curlee_source
from rsi_programmer.statement_source import python_source


def qualify(root: Path) -> dict[str, Any]:
    """Reuse executions only when source and cases still match."""
    report_path = root / "report.json"
    proof_path = root / "curlee-execution.json"
    report, proof = (json.loads(path.read_text())
                     for path in (report_path, proof_path))
    control = proof["wrong_expected_control"]
    if (not proof["compiler_stable"]
            or proof["compiler_before"] != proof["compiler_after"]
            or not control["rejected"] or control["run"]["exit_code"] != 0
            or not re.search(r"curlee run: result 1\b",
                             control["run"]["stdout"])):
        raise ValueError("independent compiler/control qualification missing")
    inputs = json.loads((root / "held-inputs.json").read_text())
    if not inputs:
        raise ValueError("nonempty independent donor cases required")
    compiler_rows = {row["id"]: row for row in proof["rows"]}
    donors = []
    for row in report["rows"]:
        name = row["id"]
        compiled = compiler_rows[name]
        if not row["accepted"] or not compiled["accepted"]:
            continue
        tokens = tuple(row["chosen"]["tokens"])
        python = python_source(tokens)
        curlee = curlee_source(tokens)
        folder = root / name
        if ((folder / "python/clean_readings.py").read_text() != python
                or (folder / "curlee/process.curlee").read_text() != curlee
                or hashlib.sha256(python.encode()).hexdigest()
                != row["python_sha256"]
                or hashlib.sha256(curlee.encode()).hexdigest()
                != row["curlee_sha256"]):
            raise ValueError("donor function source changed")
        if (len(row["python_records"]) != len(inputs)
                or len(compiled["records"]) != len(inputs)):
            raise ValueError("donor case coverage incomplete")
        for index, values in enumerate(inputs):
            expected = oracle(name, values)
            py, vm = row["python_records"][index], compiled["records"][index]
            source = curlee_harness(tokens, values, expected)
            path = folder / "curlee" / f"case-{index:03}.curlee"
            valid = (py["case"] == index and vm["case"] == index
                     and py["exit_code"] == 0 and type(py["actual"]) is int
                     and py["actual"] == expected
                     and py["expected"] == expected
                     and execute(tokens, values) == expected
                     and vm["check"]["exit_code"] == 0
                     and vm["run"]["exit_code"] == 0 and bool(re.search(
                         r"curlee run: result 0\b", vm["run"]["stdout"]))
                     and path.read_text() == source
                     and hashlib.sha256(source.encode()).hexdigest()
                     == vm["source_sha256"])
            if not valid:
                raise ValueError("donor execution evidence does not match")
        donors.append((name, tokens))
    if not donors:
        raise ValueError("no qualified donor programs")
    return {"donors": [name for name, _ in donors], "fragments": mine(donors),
            "report_sha256": hashlib.sha256(
                report_path.read_bytes()).hexdigest(),
            "compiler_proof_sha256": hashlib.sha256(
                proof_path.read_bytes()).hexdigest()}
