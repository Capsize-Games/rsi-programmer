#!/usr/bin/env python3
"""Gains table smoke test (#18, stdlib only)."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = str(ROOT / "scripts" / "plot_gains.py")


class GainsTests(unittest.TestCase):
    def test_check_passes_offline(self) -> None:
        """CSV validates without matplotlib or network."""
        proc = subprocess.run(
            [sys.executable, SCRIPT, "--check"],
            capture_output=True, text=True, timeout=120,
            cwd=str(ROOT))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("gains rows=6", proc.stdout)


if __name__ == "__main__":
    unittest.main()
