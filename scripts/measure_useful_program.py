"""Learn and execute a standalone JSON readings-cleaning program."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import resource
import subprocess
import tempfile
import time
from typing import Any

import torch

from rsi_programmer.utility_harness import curlee_harness
from rsi_programmer.utility_policy import UtilityPolicy
from rsi_programmer.utility_source import emit_curlee, emit_python

EXAMPLES = [
    ([4, 0, -2, 4, 7, 3, -9, 3], [3, 4, 7]),
    ([0, -1, 9, 2, 9, -5, 6, 2], [2, 6, 9]),
    ([-3, 8, 5, 8, 0, 1, -2, 5], [1, 5, 8]),
]


def child_limits() -> None:
    """Bound the generated Python CLI process, including its descendants."""
    for kind, value in ((resource.RLIMIT_CPU, 3),
                        (resource.RLIMIT_AS, 256 * 1024 ** 2),
                        (resource.RLIMIT_FSIZE, 1024 ** 2),
                        (resource.RLIMIT_CORE, 0)):
        resource.setrlimit(kind, (value, value))


def run_cli(root: Path, inputs: Path, output: Path,
            name: str, result_name: str) -> subprocess.CompletedProcess[str]:
    """Run with no host home, credentials or network, and bounded resources."""
    command = [
        "bwrap", "--unshare-all", "--die-with-parent",
        "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib",
        "--ro-bind", "/lib64", "/lib64", "--proc", "/proc",
        "--dev", "/dev", "--tmpfs", "/tmp",
        "--ro-bind", str(root), "/app",
        "--ro-bind", str(inputs), "/inputs",
        "--bind", str(output), "/output", "--clearenv",
        "--setenv", "PATH", "/usr/bin:/bin",
        "/usr/bin/python3", "-I", "-S", "/app/clean_readings.py",
        "/inputs/" + name, "/output/" + result_name,
    ]
    return subprocess.run(command, capture_output=True, text=True,
                          timeout=5, preexec_fn=child_limits)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name in ("python", "curlee", "inputs"):
        (root / name).mkdir()
    torch.set_num_threads(1)
    model = UtilityPolicy()
    started = time.monotonic()
    untrained = model.synthesize(EXAMPLES)
    fit = model.fit()
    fit["wall_seconds"] = time.monotonic() - started
    program = model.synthesize(EXAMPLES)
    if not program:
        raise RuntimeError("learner did not produce a program")
    print(json.dumps({"selected_program": program, "fit": fit}), flush=True)
    python = emit_python(program)
    curlee = emit_curlee(program)
    (root / "python/clean_readings.py").write_text(python)
    (root / "curlee/clean_readings.curlee").write_text(curlee)
    # Generate evaluation inputs only after all learning and synthesis ends.
    rng = random.Random(971)
    held = [[], [0], [-4, -1, 0], [1] * 20,
            list(range(100, -101, -1))] + [
        [rng.randrange(-100, 101) for _ in range(rng.randrange(1, 101))]
        for _ in range(95)
    ]
    expected = [sorted({value for value in values if value > 0})
                for values in held]
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="rsi-utility-output-") as temp:
        output = Path(temp)
        for index, (values, answers) in enumerate(zip(held, expected)):
            name = f"{index:03}.json"
            (root / "inputs" / name).write_text(json.dumps(values) + "\n")
            result = run_cli(root / "python", root / "inputs", output,
                             name, name)
            path = output / name
            actual = json.loads(path.read_text()) if path.exists() else None
            records.append({"input": name, "exit_code": result.returncode,
                            "correct": result.returncode == 0
                            and actual == answers, "expected": answers,
                            "actual": actual, "stderr": result.stderr})
        controls = []
        for index, content in enumerate(("{}", "[true]", '[1,"x"]', "[")):
            name = f"invalid-{index}.json"
            (root / "inputs" / name).write_text(content)
            result = run_cli(root / "python", root / "inputs", output,
                             name, name)
            controls.append({"input": content,
                             "rejected": result.returncode != 0
                             and not (output / name).exists()})
    (root / "held-inputs.json").write_text(json.dumps(held) + "\n")
    harness = curlee_harness(program, held, expected)
    (root / "curlee/check_clean_readings.curlee").write_text(harness)
    wrong = [answers[:] for answers in expected]
    wrong[0] = [123]
    (root / "curlee/wrong_expected.curlee").write_text(
        curlee_harness(program, held, wrong))
    report = {
        "scope": "One development utility, composed from 13 supplied list "
                 "operations. The emitter supplies implementations and CLI "
                 "plumbing. No general source-code generation claim.",
        "program": program, "examples": EXAMPLES, "fit": fit,
        "untrained_program": untrained, "held_input_seed": 971,
        "python_sha256": hashlib.sha256(python.encode()).hexdigest(),
        "curlee_sha256": hashlib.sha256(curlee.encode()).hexdigest(),
        "python_files": len(records),
        "python_passed_files": sum(row["correct"] for row in records),
        "python_negative_controls": controls, "python_records": records,
        "curlee_execution": "Not run by this command; use guarded checker.",
    }
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"python_files": len(records), "passed_files":
                      report["python_passed_files"], "controls": controls}),
          flush=True)
    return int(not all(row["correct"] for row in records)
               or not all(row["rejected"] for row in controls))


if __name__ == "__main__":
    raise SystemExit(main())
