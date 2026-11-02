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
                   return_value=result) as run, \
                patch("scripts.verify_hub.time.sleep") as sleep:
            with self.assertRaisesRegex(ValueError, "release list failed"):
                _release_with_asset("owner/repo", "weights.pt")
        run.assert_called_once()
        sleep.assert_not_called()

    def test_transient_server_error_retries_then_recovers(self) -> None:
        asset = "weights.pt"
        failed = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="HTTP 503")
        passed = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=json.dumps({"tag_name": "weights-v1",
                               "assets": [asset]}), stderr="")
        with patch("scripts.verify_hub.subprocess.run",
                   side_effect=[failed, passed]) as run, \
                patch("scripts.verify_hub.time.sleep") as sleep:
            tag = _release_with_asset("owner/repo", asset)
        self.assertEqual(tag, "weights-v1")
        self.assertEqual(run.call_count, 2)
        sleep.assert_called_once_with(1)

    def test_auth_error_is_not_retried(self) -> None:
        denied = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="HTTP 403")
        with patch("scripts.verify_hub.subprocess.run",
                   return_value=denied) as run, \
                patch("scripts.verify_hub.time.sleep") as sleep:
            with self.assertRaisesRegex(ValueError, "HTTP 403"):
                _release_with_asset("owner/repo", "weights.pt")
        run.assert_called_once()
        sleep.assert_not_called()

    def test_transient_connection_error_stops_after_three_attempts(
        self,
    ) -> None:
        unavailable = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="",
            stderr="error connecting to api.github.com")
        with patch("scripts.verify_hub.subprocess.run",
                   return_value=unavailable) as run, \
                patch("scripts.verify_hub.time.sleep") as sleep:
            with self.assertRaisesRegex(ValueError, "api.github.com"):
                _release_with_asset("owner/repo", "weights.pt")
        self.assertEqual(run.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list],
                         [1, 2])

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
                   return_value=result) as run, \
                patch("scripts.verify_hub.time.sleep") as sleep:
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                _release_with_asset("owner/repo", "weights.pt")
        run.assert_called_once()
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
