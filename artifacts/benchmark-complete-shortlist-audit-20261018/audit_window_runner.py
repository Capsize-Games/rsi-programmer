"""Run the authorized metadata and static checks while holding the lock."""

from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path.cwd()
FOLDER = ROOT / "artifacts/benchmark-complete-shortlist-audit-20261018"
FILES = [
    FOLDER / name
    for name in (
        "complete_shortlist_audit.py",
        "metadata_preflight.py",
        "negative_controls.py",
        "selection_audit.py",
        "audit_window_runner.py",
    )
]
RECEIPT: dict[str, object] = {
    "lock": "/tmp/rsi-programmer-builder.lock",
    "evaluation_bundle_read": False,
    "evaluation_measurement_started_or_inferred": False,
    "steps": [],
}


def run(label: str, command: list[str], log: pathlib.Path) -> None:
    started = time.monotonic()
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    step = {
        "label": label,
        "command": " ".join(command),
        "exit_code": proc.returncode,
        "duration_seconds": round(time.monotonic() - started, 6),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    RECEIPT["steps"].append(step)  # type: ignore[union-attr]
    log.write_text(
        proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr else "")
    )
    if proc.returncode:
        raise RuntimeError(json.dumps(step, indent=2))


def static_check(label: str) -> None:
    details = {}
    for path in FILES:
        source = path.read_text()
        ast.parse(source, filename=str(path))
        violations = [
            (number, len(line))
            for number, line in enumerate(source.splitlines(), 1)
            if len(line) > 79
        ]
        details[path.name] = {
            "ast": "pass",
            "width_pass": not violations,
            "violations": violations,
        }
    ok = all(item["width_pass"] for item in details.values())
    RECEIPT["steps"].append(  # type: ignore[union-attr]
        {"label": label, "exit_code": 0 if ok else 1, "details": details}
    )
    if not ok:
        raise RuntimeError(json.dumps(details, indent=2))


def main() -> int:
    run(
        "ruff format",
        [
            "uv",
            "tool",
            "run",
            "--from",
            "ruff",
            "ruff",
            "format",
            "--line-length",
            "79",
            *[str(path.relative_to(ROOT)) for path in FILES],
        ],
        FOLDER / "ruff-format.log",
    )
    static_check("AST and 79-column check after format")
    run(
        "independent train-only selector audit",
        [
            sys.executable,
            str(FOLDER / "selection_audit.py"),
            "--output",
            str(FOLDER / "selection-audit.json"),
        ],
        FOLDER / "selection-audit.log",
    )
    script_hash = hashlib.sha256(
        (FOLDER / "selection_audit.py").read_bytes()
    ).hexdigest()
    report_hash = hashlib.sha256(
        (FOLDER / "selection-audit.json").read_bytes()
    ).hexdigest()
    (FOLDER / "selection-audit.sha256").write_text(
        f"{script_hash}  "
        "artifacts/benchmark-complete-shortlist-audit-20261018/"
        "selection_audit.py\n"
        f"{report_hash}  "
        "artifacts/benchmark-complete-shortlist-audit-20261018/"
        "selection-audit.json\n"
    )
    preflight = FOLDER / "metadata_preflight.py"
    source = preflight.read_text()
    source, count = re.subn(
        r'(SELECTION_REPORT_SHA256 = \(\s*)"[0-9a-f]{64}"(\s*\))',
        rf'\g<1>"{report_hash}"\g<2>',
        source,
    )
    if count != 1:
        raise RuntimeError("could not update selection report hash pin")
    preflight.write_text(source)
    static_check("final independent AST and 79-column check")
    run(
        "metadata-only preflight",
        [
            sys.executable,
            str(preflight),
            "--output",
            str(FOLDER / "metadata-preflight.json"),
        ],
        FOLDER / "metadata-preflight.log",
    )
    RECEIPT["final_sha256"] = {  # type: ignore[index]
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in FILES
    }
    final_hashes = RECEIPT["final_sha256"]
    if not isinstance(final_hashes, dict):
        raise RuntimeError("final hash receipt is malformed")
    final_hashes["selection-audit.json"] = report_hash
    final_hashes["selection-audit.sha256"] = hashlib.sha256(
        (FOLDER / "selection-audit.sha256").read_bytes()
    ).hexdigest()
    RECEIPT["completed_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    (FOLDER / "audit-window-receipt.json").write_text(
        json.dumps(RECEIPT, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(RECEIPT, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
