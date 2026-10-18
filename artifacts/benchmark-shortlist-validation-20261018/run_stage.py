"""Retain command, process accounting and source hashes for each stage."""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import spikeforge

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot() -> dict[str, str]:
    dependency = Path(spikeforge.__file__).resolve().parent
    paths = set((ROOT / "rsi_programmer").rglob("*.py"))
    paths.update((ROOT / "tests").rglob("*.py"))
    paths.update((ROOT / "benchmarks").rglob("*.py"))
    paths.add(ROOT / "artifacts/benchmark-pre-eval-v2-20261018/training.json")
    paths.update((ROOT / "fixtures").rglob("*.json"))
    paths.update((ROOT / "scripts").glob("*.py"))
    paths.update((ROOT / "scripts").glob("*.js"))
    paths.update(dependency.rglob("*.py"))
    paths.update([ROOT / "justfile", ROOT / "requirements.txt",
                  Path(__file__).resolve()])
    return {
        str(p.relative_to(ROOT)) if p.is_relative_to(ROOT)
        else "spikeforge/" + str(p.relative_to(dependency)): digest(p)
        for p in sorted(paths)
    }


def main() -> int:
    name, *command = sys.argv[1:]
    log_path = ARTIFACT / (name + ".log")
    receipt = ARTIFACT / (name + ".json")
    if log_path.exists() or receipt.exists():
        raise ValueError("use a new stage name; previous evidence is retained")
    before = snapshot()
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    with log_path.open("w") as out:
        result = subprocess.run(command, cwd=ROOT, stdout=out,
                                stderr=subprocess.STDOUT)
    cli_wall = time.monotonic() - started
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    after = snapshot()
    user = usage_after.ru_utime - usage_before.ru_utime
    system = usage_after.ru_stime - usage_before.ru_stime
    record = {
        "command": command, "exit_code": result.returncode,
        "cli_wall_seconds": cli_wall,
        "stage_wall_seconds_including_post_snapshot":
        time.monotonic() - started,
        "child_user_cpu_seconds": user, "child_system_cpu_seconds": system,
        "child_total_cpu_seconds": user + system,
        "whole_child_scope": "CLI and waited descendants through exit; "
        "includes imports, setup, fit, hashes, final output I/O and isolated "
        "memory children. Excludes observer CPU.",
        "finished_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "files_before": before, "files_after": after,
        "source_unchanged": before == after,
        "scope": "All runtime/test Python, scripts, JSON fixtures, reference "
        "Python, model-reference JSON, requirements, justfile, imported "
        "SpikeForge source and observer itself; no source exclusions.",
    }
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items()
                      if not k.startswith("files_")}, indent=2), flush=True)
    print(log_path.read_text()[-4000:], flush=True)
    return result.returncode or (0 if before == after else 99)


if __name__ == "__main__":
    raise SystemExit(main())
