#!/usr/bin/env python3
"""Tests for hub verification (#8)."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
REPO = "Capsize-Games/rsi-programmer"


class HubVerifyTests(unittest.TestCase):
    def test_wrong_checksum_fails(self) -> None:
        """A local weight mismatch fails against downloaded asset bytes."""
        from scripts.verify_hub import main as verify
        with tempfile.TemporaryDirectory(prefix="rsi-hub-") as tmp:
            for name in ("rsi-search-policy-w2.pt",
                         "rsi-search-policy-w2.manifest.json"):
                shutil.copy(str(ROOT / "weights" / name),
                            str(Path(tmp) / name))
            target = Path(tmp) / "rsi-search-policy-w2.pt"
            published = target.read_bytes()
            blob = bytearray(target.read_bytes())
            blob[0] ^= 0xFF
            target.write_bytes(bytes(blob))
            with patch("scripts.verify_hub.github_asset_bytes",
                       return_value=published):
                output = io.StringIO()
                with redirect_stdout(output):
                    status = verify(["--repo", REPO, "--weights-dir", tmp])
        self.assertEqual(status, 1)
        self.assertIn("github=FAIL", output.getvalue())
        self.assertIn("asset bytes differ", output.getvalue())

    def test_missing_hub_listing_recorded_with_mocked_services(self) -> None:
        """An unpublished listing is reported when assets are available."""
        from scripts.verify_hub import verify_model as check
        weights_dir = ROOT / "weights"
        asset = weights_dir / "rsi-search-policy-w2.pt"
        with patch("scripts.verify_hub.github_asset_bytes",
                   return_value=asset.read_bytes()), \
                patch("scripts.verify_hub.hub_model_status",
                      return_value=(False, "not published")):
            res = check(REPO, str(weights_dir),
                        "rsi-search-policy-w2.manifest.json")
        self.assertEqual(res["model"], "rsi-search-policy-w2")
        self.assertEqual(res["github"], "PASS", res["detail"])
        self.assertEqual(res["hub"], "FAIL")
        self.assertEqual(res["detail"], "not published")

    def test_live_release_asset_and_missing_hub_listing(self) -> None:
        """Verify the live GitHub release asset and hub listing."""
        from scripts.verify_hub import verify_model as check
        res = check(REPO, str(ROOT / "weights"),
                    "rsi-search-policy-w2.manifest.json")
        self.assertEqual(res["model"], "rsi-search-policy-w2")
        self.assertEqual(res["github"], "PASS", res["detail"])
        self.assertEqual(res["hub"], "FAIL")
        self.assertTrue(res["detail"])


if __name__ == "__main__":
    unittest.main()
