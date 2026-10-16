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
import urllib.request
from pathlib import Path
from typing import Any

HUB_BASE = "https://hub.spikeforge.net"
HUB_HANDLE = "reference"
TIMEOUT_S = 60


def local_sha256(path: str) -> str:
    """Hex sha256 of one local file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _release_with_asset(repo: str, asset: str) -> str:
    """Tag of the newest release carrying one asset name."""
    out = subprocess.run(
        ["gh", "release", "list", "--repo", repo, "--limit", "50",
         "--json", "tagName"],
        capture_output=True, text=True, timeout=TIMEOUT_S)
    if out.returncode != 0:
        raise ValueError("release list failed")
    for row in json.loads(out.stdout or "[]"):
        tag = str(row.get("tagName", ""))
        view = subprocess.run(
            ["gh", "release", "view", tag, "--repo", repo,
             "--json", "assets", "--jq",
             ".assets[].name"],
            capture_output=True, text=True, timeout=TIMEOUT_S)
        if view.returncode == 0 and asset in view.stdout.split():
            return tag
    raise ValueError("asset missing: " + asset)


def github_asset_bytes(repo: str, asset: str,
                       workdir: str) -> bytes:
    """Download a release asset by name from any release."""
    tag = _release_with_asset(repo, asset)
    out = subprocess.run(
        ["gh", "release", "download", tag, "--repo", repo,
         "--pattern", asset, "--dir", workdir, "--clobber"],
        capture_output=True, text=True, timeout=TIMEOUT_S)
    if out.returncode != 0:
        raise ValueError("asset missing: " + asset)
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
