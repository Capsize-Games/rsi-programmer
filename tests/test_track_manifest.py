"""Ensure track separation preserves discovery and existing test gates."""
from __future__ import annotations

from collections import Counter
import re
import subprocess
import sys
import unittest

from scripts.tracks import LANES, ROOT, load_manifest, test_modules


class TrackManifestTests(unittest.TestCase):
    """Detect missing tests, accidental overlap and changed lane coverage."""

    def test_every_discovered_module_has_declared_ownership(self) -> None:
        manifest = load_manifest()
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["primary_track"], "programmer")
        tracks = manifest["tracks"]
        self.assertEqual(set(tracks), {
            "programmer", "sensing", "sound-companion",
        })
        counts = Counter(module for track in tracks.values()
                         for module in test_modules(track, "all"))
        discovered = {f"tests.{path.stem}" for path in
                      (ROOT / "tests").glob("test_*.py")}
        self.assertEqual(set(counts), discovered)
        shared = set(manifest["shared_tests"])
        self.assertEqual({name for name, count in counts.items() if count > 1},
                         shared)
        for module in shared:
            self.assertEqual(counts[module], 2)
        for track in tracks.values():
            self.assertEqual(set(track["tests"]), set(LANES))
            modules = test_modules(track, "all")
            self.assertEqual(len(modules), len(set(modules)))

    def test_existing_gate_modules_are_preserved(self) -> None:
        content = (ROOT / "justfile").read_text(encoding="utf-8")
        tracks = load_manifest()["tracks"]
        for lane, recipe in (("fast", "test-fast"), ("slow", "test-slow"),
                             ("retry", "test-retry")):
            match = re.search(
                rf"^{recipe}:\n(.*?)(?=^[^\s#]|\Z)", content, re.M | re.S,
            )
            self.assertIsNotNone(match)
            assert match is not None
            legacy = set(re.findall(r"\btests\.test_[a-z0-9_]+\b",
                                    match.group(1)))
            self.assertTrue(legacy)
            declared = {module for track in tracks.values()
                        for module in track["tests"][lane]}
            extra: set[str] = set()
            if lane == "fast":
                extra = {"tests.test_sound_window",
                         "tests.test_install_sound_companion"}
            self.assertEqual(declared, legacy | extra)

    def test_programmer_excludes_sensor_and_companion_tests(
        self,
    ) -> None:
        tracks = load_manifest()["tracks"]
        self.assertEqual(tracks["programmer"]["status"], "primary")
        modules = test_modules(tracks["programmer"], "all")
        for module in modules:
            self.assertFalse(module.startswith((
                "tests.test_audio_", "tests.test_sensing_",
                "tests.test_sound_", "tests.test_install_sound_",
            )))
        self.assertIn("tests.test_rsi_loop", modules)
        self.assertIn("tests.test_online_graph", modules)

    def test_listing_works_without_installed_python_packages(self) -> None:
        command = [sys.executable, "-S", "-m", "scripts.tracks", "test",
                   "programmer", "--lane", "all", "--list"]
        result = subprocess.run(command, cwd=ROOT, capture_output=True,
                                text=True, timeout=5, check=True)
        expected = test_modules(load_manifest()["tracks"]["programmer"],
                                "all")
        self.assertEqual(result.stdout.splitlines(), expected)


if __name__ == "__main__":
    unittest.main()
