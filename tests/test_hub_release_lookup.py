"""Mocked regressions for paginated GitHub release asset lookup."""

from __future__ import annotations

import json
import subprocess
import unittest
from unittest.mock import patch

from scripts.verify_hub import _release_with_asset


class HubReleaseLookupTests(unittest.TestCase):
    def test_finds_asset_beyond_first_fifty_releases(self) -> None:
        asset = "rsi-search-policy-w2.pt"
        records = [{"tag_name": "code-%d" % index, "assets": []}
                   for index in range(51)]
        records.append({"tag_name": "weights-v3", "assets": [asset]})
        output = "\n".join(json.dumps(row) for row in records)
        result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=output, stderr="")
        with patch("scripts.verify_hub.subprocess.run",
                   return_value=result) as run:
            tag = _release_with_asset("owner/repo", asset)
        self.assertEqual(tag, "weights-v3")
        command = run.call_args.args[0]
        self.assertEqual(command[:3], ["gh", "api", "--paginate"])
        self.assertEqual(command[3], "repos/owner/repo/releases")
        run.assert_called_once()

    def test_release_api_error_fails_closed(self) -> None:
        result = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="API unavailable")
        with patch("scripts.verify_hub.subprocess.run",
                   return_value=result):
            with self.assertRaisesRegex(ValueError, "release list failed"):
                _release_with_asset("owner/repo", "weights.pt")

    def test_missing_asset_reports_missing(self) -> None:
        records = [{"tag_name": "code-v1", "assets": ["other.pt"]}]
        result = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=json.dumps(records[0]), stderr="")
        with patch("scripts.verify_hub.subprocess.run",
                   return_value=result):
            with self.assertRaisesRegex(ValueError, "asset missing"):
                _release_with_asset("owner/repo", "weights.pt")

    def test_malformed_release_json_fails_closed(self) -> None:
        result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="not-json", stderr="")
        with patch("scripts.verify_hub.subprocess.run",
                   return_value=result):
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                _release_with_asset("owner/repo", "weights.pt")


if __name__ == "__main__":
    unittest.main()
