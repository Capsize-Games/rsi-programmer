"""List research tracks and run their declared test modules."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tracks" / "manifest.json"
BUILDER_LOCK = Path("/tmp/rsi-programmer-builder.lock")
LANES = ("fast", "slow", "retry")


def load_manifest() -> dict[str, Any]:
    """Read track ownership without importing a learner or GUI."""
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_modules(track: dict[str, Any], lane: str) -> list[str]:
    """Return modules in lane order, including every lane for all."""
    selected = LANES if lane == "all" else (lane,)
    return [module for name in selected for module in track["tests"][name]]


def run_tests(track: dict[str, Any], modules: list[str]) -> int:
    """Serialize tests and use an isolated display for native controls."""
    python = ROOT / track["python"]
    if not python.is_file():
        print(f"Missing interpreter: {python}")
        print("Run just setup-sound for the companion, or just setup.")
        return 2
    command = [str(python), "-m", "unittest", *modules, "-v"]
    if track["headless"]:
        command = ["xvfb-run", "-a", *command]
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with BUILDER_LOCK.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another RSI job holds the builder lock; tests did not run.")
            return 75
        print(shlex.join(command), flush=True)
        try:
            result = subprocess.run(command, cwd=ROOT, env=environment)
        except FileNotFoundError as error:
            print(f"Test command unavailable: {error.filename}")
            return 2
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    """List track status or execute an explicitly selected test scope."""
    manifest = load_manifest()
    tracks = manifest["tracks"]
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="show track status and test scopes")
    test = commands.add_parser("test", help="run one track's tests")
    test.add_argument("track", choices=tuple(tracks))
    test.add_argument("--lane", choices=(*LANES, "all"), default="fast")
    test.add_argument("--list", action="store_true", help="list without run")
    args = parser.parse_args(argv)
    if args.command == "list":
        for name, track in tracks.items():
            counts = ", ".join(
                f"{lane}={len(track['tests'][lane])}" for lane in LANES
            )
            print(f"{name}: {track['status']}; test modules: {counts}")
        return 0
    track = tracks[args.track]
    modules = test_modules(track, args.lane)
    if args.list:
        print("\n".join(modules))
        return 0
    if not modules:
        print(f"No {args.lane} test modules declared for {args.track}.")
        return 2
    return run_tests(track, modules)


if __name__ == "__main__":
    raise SystemExit(main())
