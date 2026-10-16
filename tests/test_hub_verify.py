#!/usr/bin/env python3
"""Tests for hub verification (#8)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "Capsize-Games/rsi-programmer"


class HubVerifyTests(unittest.TestCase):
    def test_wrong_checksum_fails(self) -> None:
        """Corrupted local weights fail against real assets."""
        tmp = tempfile.mkdtemp(prefix="rsi-hub-")
        for name in ("rsi-search-policy-w2.pt",
                     "rsi-search-policy-w2.manifest.json"):
            shutil.copy(str(ROOT / "weights" / name),
                        str(Path(tmp) / name))
        target = Path(tmp) / "rsi-search-policy-w2.pt"
        blob = bytearray(target.read_bytes())
        blob[0] ^= 0xFF
        target.write_bytes(bytes(blob))
        proc = subprocess.run(
            ["python3", str(ROOT / "scripts" / "verify_hub.py"),
             "--repo", REPO, "--weights-dir", tmp],
            capture_output=True, text=True, timeout=300)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("FAIL", proc.stdout)

    def test_missing_hub_listing_recorded(self) -> None:
        """Unpublished models report hub FAIL with the reason."""
        from scripts.verify_hub import verify_model as check
        res = check(REPO, str(ROOT / "weights"),
                    "rsi-search-policy-w2.manifest.json")
        self.assertEqual(res["model"], "rsi-search-policy-w2")
        self.assertEqual(res["github"], "PASS")
        self.assertEqual(res["hub"], "FAIL")
        self.assertTrue(res["detail"])


if __name__ == "__main__":
    unittest.main()
