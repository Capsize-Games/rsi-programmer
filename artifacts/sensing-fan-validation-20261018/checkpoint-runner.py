"""Run one serial validation command with immutable source receipts."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path


def sources(root: Path) -> dict[str, str]:
    paths = set()
    for directory in ("rsi_programmer", "scripts", "tests", "fixtures"):
        paths.update(path for path in (root / directory).rglob("*")
                     if path.suffix in {".py", ".json", ".js"})
    paths.update(root / path for path in (
        "requirements.txt", "justfile", ".github/workflows/ci.yml"))
    return {str(path.relative_to(root)): hashlib.sha256(
        path.read_bytes()).hexdigest() for path in sorted(paths)}


def main() -> None:
    label, destination, *command = sys.argv[1:]
    if command[0] == "--":
        command = command[1:]
    root = Path.cwd()
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    log = output / (label + ".log")
    receipt = output / (label + "-receipt.json")
    if log.exists() or receipt.exists():
        raise FileExistsError("preserve the existing validation receipt")
    before = sources(root)
    started = time.perf_counter()
    print("Starting", label, flush=True)
    with log.open("w") as stream:
        result = subprocess.run(command, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    duration = time.perf_counter() - started
    after = sources(root)
    text = log.read_text()
    counts = re.findall(r"Ran (\d+) tests? in ([\d.]+)s", text)
    data = {
        "command": command, "exit_code": result.returncode,
        "wall_seconds": duration, "source_unchanged": before == after,
        "source_files": before,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        "unittest_runs": [{"tests": int(count), "seconds": float(seconds)}
                          for count, seconds in counts],
        "builder_lock": "/tmp/rsi-programmer-builder.lock",
    }
    receipt.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in data.items()
                      if key != "source_files"}), flush=True)
    print("\n".join(text.splitlines()[-30:]), flush=True)
    raise SystemExit(result.returncode if before == after else 99)


if __name__ == "__main__":
    main()
