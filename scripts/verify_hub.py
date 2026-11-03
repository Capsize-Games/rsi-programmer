#!/usr/bin/env python3
"""Verify weight distribution: GitHub assets plus hub listing.

For every weights/*.manifest.json model: download the matching
GitHub release asset and require byte-identical weights, then query
the spikeforge hub for the published model. Prints PASS/FAIL per
model per source; any undownloadable or mismatched asset fails the
run. Held-out metrics comparison stays in the release notes as
test evidence; this script proves distribution, not training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

HUB_BASE = "https://hub.spikeforge.net"
HUB_HANDLE = "reference"
TIMEOUT_S = 60
GITHUB_READ_ATTEMPTS = 3
GITHUB_READ_BACKOFF_S = (1, 2)


def local_sha256(path: str) -> str:
    """Hex sha256 of one local file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _release_with_asset(repo: str, asset: str) -> str:
    """Tag of the newest release carrying one asset name."""
    command = ["gh", "api", "--paginate", "repos/%s/releases" % repo,
               "--jq", ".[] | {tag_name, assets: [.assets[].name]}"]
    output = _github_read(command, "release list")
    try:
        rows = [json.loads(line) for line in output.splitlines()
                if line.strip()]
    except ValueError as exc:
        raise ValueError("release list returned invalid JSON") from exc
    for row in rows:
        if (not isinstance(row, dict)
                or not isinstance(row.get("tag_name"), str)
                or not isinstance(row.get("assets"), list)):
            raise ValueError("release list returned invalid data")
        tag = row["tag_name"]
        if asset in row["assets"]:
            return tag
    raise ValueError("asset missing: " + asset)


def _transient_release_error(detail: str) -> bool:
    """Whether gh reported a retryable network or server failure."""
    lowered = detail.lower()
    if any(code in lowered for code in ("401", "403", "404")):
        return False
    if any(code in lowered for code in (
            "429", "500", "501", "502", "503", "504", "505", "506",
            "507", "508", "509", "510", "511")):
        return True
    return any(message in lowered for message in (
        "error connecting", "connection reset", "connection refused",
        "connection closed", "timed out", "timeout", "temporary failure",
    ))


def _github_read_once(command: list[str]) -> tuple[str | None, str, bool]:
    """Read through gh once; classify only transport and server failures."""
    try:
        out = subprocess.run(command, capture_output=True, text=True,
                             timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return None, "request timed out", True
    if out.returncode == 0:
        return out.stdout, "", False
    detail = out.stderr.strip() or out.stdout.strip() or (
        "exit status %d" % out.returncode)
    return None, detail, _transient_release_error(detail)


def _github_read(command: list[str], operation: str) -> str:
    """Bound retries for read-only release metadata and asset downloads."""
    detail = "no attempts"
    for attempt in range(GITHUB_READ_ATTEMPTS):
        output, detail, transient = _github_read_once(command)
        if output is not None:
            return output
        if not transient or attempt + 1 == GITHUB_READ_ATTEMPTS:
            break
        time.sleep(GITHUB_READ_BACKOFF_S[attempt])
    raise ValueError(operation + " failed: " + detail)


def github_asset_bytes(repo: str, asset: str,
                       workdir: str) -> bytes:
    """Download a release asset by name from any release."""
    tag = _release_with_asset(repo, asset)
    _github_read(
        ["gh", "release", "download", tag, "--repo", repo,
         "--pattern", asset, "--dir", workdir, "--clobber"],
        "asset download")
    data = Path(workdir, asset).read_bytes()
    if not data:
        raise ValueError("asset empty: " + asset)
    return data


def hub_model_status(base: str, handle: str,
                     slug: str) -> tuple[bool, str]:
    """Published listing and version detail, or why not."""
    url = "%s/v1/models/%s/%s" % (base.rstrip("/"), handle,
                                  slug)
    req = urllib.request.Request(
        url, headers={"User-Agent": "rsi-programmer-verify/1.0",
                      "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req,
                                    timeout=TIMEOUT_S) as res:
            body = res.read().decode()
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode()
        except Exception:
            return False, "hub error: %s" % exc
    except Exception as exc:
        return False, "hub unreachable: %s" % exc
    try:
        detail = json.loads(body)
    except ValueError:
        return False, "hub bad payload"
    if isinstance(detail, dict) and detail.get("status") == 404:
        return False, str(detail.get("detail", "not found"))
    return True, body[:200]


def verify_model(repo: str, weights_dir: str, manifest: str,
                 hub_base: str = HUB_BASE,
                 hub_handle: str = HUB_HANDLE) -> dict[str, Any]:
    """Byte and listing checks for one manifest; never raises."""
    info = json.loads(Path(weights_dir, manifest).read_text())
    name = str(info.get("name", ""))
    local = str(Path(weights_dir, name + ".pt"))
    result: dict[str, Any] = {"model": name, "github": "FAIL",
                              "hub": "FAIL", "detail": ""}
    try:
        want = local_sha256(local)
    except OSError as exc:
        result["detail"] = "local unreadable: %s" % exc
        return result
    with tempfile.TemporaryDirectory(
            prefix="rsi-hub-") as tmp:
        try:
            got = github_asset_bytes(repo, name + ".pt", tmp)
        except (ValueError, subprocess.SubprocessError,
                OSError) as exc:
            result["detail"] = str(exc)
            return result
        if hashlib.sha256(got).hexdigest() != want:
            result["detail"] = "asset bytes differ"
            return result
    result["github"] = "PASS"
    ok, detail = hub_model_status(hub_base, hub_handle, name)
    result["hub"] = "PASS" if ok else "FAIL"
    result["detail"] = detail
    return result


def main(argv: list[str] | None = None) -> int:
    """Verify every manifest model; fail on any bad asset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",
                        default="Capsize-Games/rsi-programmer")
    parser.add_argument("--weights-dir", default="weights")
    parser.add_argument("--hub-base", default=HUB_BASE)
    parser.add_argument("--hub-handle", default=HUB_HANDLE)
    args = parser.parse_args(argv)
    manifests = sorted(Path(args.weights_dir).glob("*.json"))
    if not manifests:
        print("FAIL: no manifests")
        return 1
    bad = 0
    for path in manifests:
        if path.name.endswith(".manifest.json"):
            result = verify_model(args.repo, args.weights_dir,
                                  path.name, args.hub_base,
                                  args.hub_handle)
            print("%s github=%s hub=%s %s"
                  % (result["model"], result["github"],
                     result["hub"], result["detail"][:120]))
            if result["github"] != "PASS" \
                    or result["hub"] != "PASS":
                bad += 1
    print("PASS" if not bad else "FAIL: %d model(s)" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
