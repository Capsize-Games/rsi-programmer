"""Reproduce missing CI auth and exercise real checks with token env."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
METHOD = (
    "tests.test_hub_verify.HubVerifyTests."
    "test_missing_hub_listing_recorded"
)


def main() -> int:
    receipt = HERE / "auth-probe.json"
    if receipt.exists():
        raise ValueError("previous evidence exists; use a new probe script")
    token_result = subprocess.run(
        ["gh", "auth", "token", "--hostname", "github.com"],
        capture_output=True, text=True, check=True)
    token = token_result.stdout.strip()
    if not token:
        raise ValueError("existing gh credential unavailable")
    stages: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="rsi-ci-auth-") as config:
        env = os.environ.copy()
        for name in ("GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN",
                     "GITHUB_ENTERPRISE_TOKEN", "GH_HOST", "GH_REPO"):
            env.pop(name, None)
        env.update(GH_CONFIG_DIR=config,
                   GITHUB_ACTIONS="true", GH_PROMPT_DISABLED="1")
        for stage, modules in (
            ("unauthenticated", [METHOD]),
            ("authenticated", ["tests.test_hub_release_lookup",
                               "tests.test_hub_verify"]),
        ):
            if stage == "authenticated":
                env["GH_TOKEN"] = token
            command = [sys.executable, "-m", "unittest", *modules, "-v"]
            started = time.monotonic()
            result = subprocess.run(
                command, cwd=ROOT, env=env, capture_output=True,
                text=True, timeout=600)
            output = result.stdout + result.stderr
            if token in output:
                raise ValueError("refusing to persist credential output")
            log = HERE / (stage + ".log")
            log.write_text(output)
            stages.append({
                "name": stage, "command": command,
                "exit_code": result.returncode,
                "wall_seconds": time.monotonic() - started,
                "log": log.name,
            })
            if stage == "unauthenticated":
                assert result.returncode == 1, output
                assert "AssertionError: 'FAIL' != 'PASS'" in output
                assert "FAILED (failures=1)" in output
            else:
                assert result.returncode == 0, output
                assert "Ran 6 tests" in output and "\nOK\n" in output
    record = {
        "passed": True,
        "credential_scope": "Existing local gh credential supplied only "
        "as the authenticated child GH_TOKEN; no credential saved. "
        "This does not mint or test the Actions token itself.",
        "isolation": "Fresh empty GH_CONFIG_DIR, no inherited token, "
        "GH_HOST or GH_REPO variables; GITHUB_ACTIONS=true for both "
        "child processes.",
        "stages": stages,
        "expected_negative_failures": 1,
        "authenticated_tests_passed": 6,
    }
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
