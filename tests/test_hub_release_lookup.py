"""Mocked regressions for paginated GitHub release asset lookup."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.verify_hub import _release_with_asset, github_asset_bytes


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

    def test_asset_download_retries_transport_failure(self) -> None:
        failed = subprocess.CompletedProcess([], 1, "", "HTTP 503")
        passed = subprocess.CompletedProcess([], 0, "", "")
        with tempfile.TemporaryDirectory() as workdir:
            Path(workdir, "weights.pt").write_bytes(b"opaque model bytes")
            with patch("scripts.verify_hub._release_with_asset",
                       return_value="weights-v1"), \
                    patch("scripts.verify_hub.subprocess.run",
                          side_effect=[failed, passed]) as run, \
                    patch("scripts.verify_hub.time.sleep") as sleep:
                data = github_asset_bytes("o/r", "weights.pt", workdir)
        self.assertEqual(data, b"opaque model bytes")
        self.assertEqual(run.call_count, 2)
        sleep.assert_called_once_with(1)

    def test_asset_timeout_is_bounded_and_explained(self) -> None:
        timeout = subprocess.TimeoutExpired("gh", 60)
        with patch("scripts.verify_hub._release_with_asset",
                   return_value="weights-v1"), \
                patch("scripts.verify_hub.subprocess.run",
                      side_effect=timeout) as run, \
                patch("scripts.verify_hub.time.sleep") as sleep:
            with self.assertRaisesRegex(ValueError,
                                        "asset download failed: "
                                        "request timed out"):
                github_asset_bytes("o/r", "weights.pt", "/unused")
        self.assertEqual(run.call_count, 3)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [1, 2])

    def test_asset_permission_and_missing_errors_do_not_retry(self) -> None:
        for error in ("HTTP 401", "HTTP 403", "HTTP 404"):
            with self.subTest(error=error), \
                    patch("scripts.verify_hub._release_with_asset",
                          return_value="weights-v1"), \
                    patch("scripts.verify_hub.subprocess.run",
                          return_value=subprocess.CompletedProcess(
                              [], 1, "", error)) as run, \
                    patch("scripts.verify_hub.time.sleep") as sleep:
                with self.assertRaisesRegex(ValueError, error):
                    github_asset_bytes("o/r", "weights.pt", "/unused")
                run.assert_called_once()
                sleep.assert_not_called()

    def test_successful_download_with_empty_file_still_fails(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            Path(workdir, "weights.pt").write_bytes(b"")
            with patch("scripts.verify_hub._release_with_asset",
                       return_value="weights-v1"), \
                    patch("scripts.verify_hub.subprocess.run",
                          return_value=subprocess.CompletedProcess(
                              [], 0, "", "")):
                with self.assertRaisesRegex(ValueError, "asset empty"):
                    github_asset_bytes("o/r", "weights.pt", workdir)


if __name__ == "__main__":
    unittest.main()
