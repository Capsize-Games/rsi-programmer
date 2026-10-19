"""Run and inspect a private live microphone prediction session."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from rsi_programmer.audio_runtime import (
    AudioRuntime,
    PROJECT_ROOT,
    STATE_PATH,
    atomic_write_json,
    new_session_dir,
    read_json,
    resolve_source,
)


def _runtime_type(variant: str) -> type[AudioRuntime]:
    if variant == "original":
        return AudioRuntime
    if variant == "continual":
        from rsi_programmer.audio_continual_runtime import (
            AudioContinualRuntime,
        )

        return AudioContinualRuntime
    raise ValueError(f"unknown microphone runtime variant: {variant}")


def _validate_minutes(variant: str, minutes: float | None) -> None:
    if variant == "continual":
        if minutes is not None and minutes != 30.0:
            raise ValueError("continual variant runs for exactly 30 minutes")
        return
    if minutes is not None and (
        not math.isfinite(minutes) or minutes <= 0
    ):
        raise ValueError("minutes must be a positive finite number")


def _process_arguments(pid: int) -> list[str] | None:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    return [part.decode("utf-8", errors="replace")
            for part in raw.split(b"\0") if part]


def _owned_run(pid: int, session_dir: Path) -> bool:
    arguments = _process_arguments(pid)
    if not arguments:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return (
        "-m" in arguments
        and "scripts.fan_microphone" in arguments
        and "run" in arguments
        and "--output" in arguments
        and str(session_dir) in arguments
    )


def _latest_path(root: Path) -> Path:
    return root / "latest.json"


def _current(root: Path) -> tuple[dict[str, Any], Path]:
    latest = read_json(_latest_path(root))
    session_dir = Path(str(latest["session_dir"])).resolve()
    if root.resolve() not in session_dir.parents:
        raise ValueError(
            "latest session path is outside the private state root"
        )
    return latest, session_dir


def _status_starting(
    output_dir: Path,
    source_identity: str,
    source_description: str,
    variant: str = "original",
) -> None:
    runtime = {
        "status": "initializing",
        "pid": None,
        "session_dir": str(output_dir),
        "source_description": source_description,
        "source_identity": source_identity,
        "started_at": None,
        "delivery_seconds": 0.0,
        "last_signal": None,
        "protocol_sha256": None,
        "git_head": None,
        "error": None,
    }
    if variant != "original":
        runtime["variant"] = variant
    atomic_write_json(output_dir / "status.json", {
        "phase": "initializing",
        "frame_count": 0,
        "monitor_frames": 0,
        "blocks": [],
        "checkpoints": [],
        "runtime": runtime,
    })


def command_run(args: argparse.Namespace) -> int:
    variant = getattr(args, "variant", "original")
    _validate_minutes(variant, args.minutes)
    output_dir = Path(args.output).expanduser().resolve()
    source, description = resolve_source(args.source)
    runtime = _runtime_type(variant)(
        output_dir=output_dir,
        source_identity=source,
        source_description=description,
        minutes=args.minutes,
    )
    return runtime.run()


def command_start(args: argparse.Namespace) -> int:
    variant = getattr(args, "variant", "original")
    _validate_minutes(variant, args.minutes)
    root = Path(args.state_root).expanduser().resolve()
    source, description = resolve_source(args.source)
    latest_path = _latest_path(root)
    if latest_path.exists():
        try:
            latest, old_dir = _current(root)
            old_pid = int(latest["pid"])
            if _owned_run(old_pid, old_dir):
                print(
                    f"already running: pid={old_pid} session={old_dir}",
                    file=sys.stderr,
                )
                return 1
        except (KeyError, TypeError, ValueError, OSError):
            pass
    output_dir = new_session_dir(root)
    _status_starting(output_dir, source, description, variant)
    command = [
        sys.executable, "-m", "scripts.fan_microphone", "run",
        "--source", source,
        "--output", str(output_dir),
    ]
    if variant != "original":
        command.extend(["--variant", variant])
    if args.minutes is not None:
        command.extend(["--minutes", str(args.minutes)])
    log_path = output_dir / "run.log"
    with log_path.open("ab") as log_stream:
        os.chmod(log_path, 0o600)
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            cwd=PROJECT_ROOT,
        )
    atomic_write_json(output_dir / "pid.json", {
        "pid": process.pid,
        "session_dir": str(output_dir),
    })
    latest = {
        "pid": process.pid,
        "session_dir": str(output_dir),
    }
    if variant != "original":
        latest["variant"] = variant
    atomic_write_json(latest_path, latest)
    print(
        f"started microphone session pid={process.pid} "
        f"source={description} session={output_dir}"
    )
    return 0


def command_status(args: argparse.Namespace) -> int:
    root = Path(args.state_root).expanduser().resolve()
    latest, session_dir = _current(root)
    status = read_json(session_dir / "status.json")
    runtime = status.get("runtime", {})
    pid = int(latest.get("pid", runtime.get("pid") or 0))
    if runtime.get("status") in ("running", "initializing") and not (
        _owned_run(pid, session_dir)
    ):
        runtime = {**runtime, "status": "stale"}
        status = {**status, "runtime": runtime}
    if args.json:
        print(json.dumps(status, sort_keys=True, indent=2, allow_nan=False))
        return 0
    comparisons = status.get("adaptive_vs_fixed", {})
    deltas = ", ".join(
        f"{arm}={item.get('cumulative_mse_delta')}"
        for arm, item in comparisons.items()
    ) or "no monitor comparison yet"
    print(
        f"{runtime.get('status', 'unknown')} | "
        f"phase={status.get('phase', 'unknown')} "
        f"frames={status.get('frame_count', 0)} "
        f"monitor={status.get('monitor_frames', 0)} "
        f"blocks={len(status.get('blocks', []))} "
        f"checkpoints={len(status.get('checkpoints', []))}"
    )
    if runtime.get("source_description"):
        print(f"source={runtime['source_description']}")
    print(f"paired cumulative MSE delta: {deltas}")
    if runtime.get("error"):
        print(f"error={runtime['error']}")
    print(f"session={session_dir}")
    return 0


def command_stop(args: argparse.Namespace) -> int:
    root = Path(args.state_root).expanduser().resolve()
    latest, session_dir = _current(root)
    pid_data = read_json(session_dir / "pid.json")
    pid = int(pid_data["pid"])
    if pid_data.get("session_dir") != str(session_dir):
        raise ValueError("PID record does not match the latest session")
    if not _owned_run(pid, session_dir):
        print(f"no owned microphone process is running for {session_dir}")
        return 0
    os.kill(pid, signal.SIGTERM)
    print(f"stopping microphone session pid={pid} session={session_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run the capture process")
    run_parser.add_argument("--source")
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--minutes", type=float)
    run_parser.add_argument(
        "--variant", choices=("original", "continual"), default="original"
    )
    run_parser.set_defaults(handler=command_run)

    start_parser = subparsers.add_parser("start", help="start in background")
    start_parser.add_argument("--source")
    start_parser.add_argument("--minutes", type=float)
    start_parser.add_argument(
        "--variant", choices=("original", "continual"), default="original"
    )
    start_parser.add_argument("--state-root", default=str(STATE_PATH))
    start_parser.set_defaults(handler=command_start)

    status_parser = subparsers.add_parser("status", help="show latest session")
    status_parser.add_argument("--json", action="store_true")
    status_parser.add_argument("--state-root", default=str(STATE_PATH))
    status_parser.set_defaults(handler=command_status)

    stop_parser = subparsers.add_parser("stop", help="stop latest session")
    stop_parser.add_argument("--state-root", default=str(STATE_PATH))
    stop_parser.set_defaults(handler=command_stop)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (
        OSError, ValueError, RuntimeError, subprocess.SubprocessError
    ) as exc:
        print(f"fan microphone: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
