"""Tests for desktop entry generation without installing the shortcut."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
import tempfile
import unittest

from scripts.install_sound_companion import desktop_entry_path
from scripts.install_sound_companion import desktop_entry_text
from scripts.install_sound_companion import desktop_exec_arg


def _unescape_entry_string(value: str) -> str:
    """Apply the Desktop Entry string-value escaping pass."""
    escaped = {"s": " ", "n": "\n", "t": "\t", "r": "\r",
               "\\": "\\"}
    result: list[str] = []
    index = 0
    while index < len(value):
        if value[index] == "\\" and index + 1 < len(value):
            index += 1
            result.append(escaped[value[index]])
        else:
            result.append(value[index])
        index += 1
    return "".join(result)


def _split_exec(value: str) -> list[str]:
    """Use the Linux desktop's parser after entry-string decoding."""
    command = [
        "/usr/bin/python3", "-c",
        "import json,sys; from gi.repository import GLib; "
        "print(json.dumps(GLib.shell_parse_argv(sys.argv[1])[1]))",
        value,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=5.0)
    except FileNotFoundError:
        raise unittest.SkipTest("System Python unavailable")
    if "No module named 'gi'" in result.stderr:
        raise unittest.SkipTest("System GLib parser unavailable")
    if result.returncode:
        raise AssertionError(result.stderr)
    return [argument.replace("%%", "%")
            for argument in json.loads(result.stdout)]


class SoundCompanionInstallerTests(unittest.TestCase):
    """Check generated paths and Desktop Entry escaping."""

    def test_exec_argument_escapes_desktop_entry_special_characters(
        self,
    ) -> None:
        value = '/tmp/Sound "Companion" $home `x` \\\n100%'
        encoded = desktop_exec_arg(value)
        decoded = _unescape_entry_string(encoded)
        self.assertEqual(_split_exec(decoded), [value])

    def test_entry_uses_absolute_checkout_paths_with_spaces(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sound companion ") as tmp:
            root = Path(tmp) / "my checkout"
            text = desktop_entry_text(root)
            self.assertIn(
                f'Exec="{root.resolve()}/.venv-desktop/bin/python" '
                "-m scripts.sound_companion",
                text,
            )
            self.assertIn(f"Path={root.resolve()}", text)
            self.assertIn(
                f"Icon={root.resolve()}/assets/sound-companion.svg", text)
            self.assertIn("Terminal=false", text)
            self.assertIn("Name=Sound Companion", text)

    def test_entry_path_is_user_applications_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = desktop_entry_path(Path(tmp))
            self.assertEqual(path.parent.name, "applications")
            self.assertEqual(path.name, "sound-companion.desktop")


if __name__ == "__main__":
    unittest.main()
