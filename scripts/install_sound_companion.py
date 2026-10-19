"""Install a user-local desktop shortcut for Sound Companion."""
from __future__ import annotations

import os
from pathlib import Path


def desktop_exec_arg(argument: str) -> str:
    """Quote one argument according to the Desktop Entry Exec grammar."""
    escaped = argument.replace("\\", "\\\\")
    escaped = escaped.replace('"', '\\"')
    escaped = escaped.replace("`", "\\`").replace("$", "\\$")
    escaped = escaped.replace("%", "%%")
    return desktop_value(f'"{escaped}"')


def desktop_value(value: str) -> str:
    """Escape a plain value for a Desktop Entry string field."""
    return (value.replace("\\", "\\\\").replace("\n", "\\n")
            .replace("\t", "\\t").replace("\r", "\\r"))


def desktop_entry_text(repository_root: Path) -> str:
    """Return the desktop entry text for a resolved project checkout."""
    root = repository_root.expanduser().resolve()
    python = root / ".venv-desktop" / "bin" / "python"
    icon = root / "assets" / "sound-companion.svg"
    command = (f"{desktop_exec_arg(str(python))} "
               "-m scripts.sound_companion")
    return "\n".join((
        "[Desktop Entry]",
        "Type=Application",
        "Name=Sound Companion",
        f"Exec={command}",
        f"Path={desktop_value(str(root))}",
        f"Icon={desktop_value(str(icon))}",
        "Terminal=false",
        "Categories=Utility;",
        "",
    ))


def desktop_entry_path(data_home: Path | None = None) -> Path:
    """Return the user applications path without writing to it."""
    if data_home is None:
        configured = os.environ.get("XDG_DATA_HOME")
        data_home = (Path(configured).expanduser() if configured else
                     Path.home() / ".local" / "share")
    return data_home.expanduser().resolve() / "applications" / (
        "sound-companion.desktop")


def install_shortcut(repository_root: Path | None = None,
                     data_home: Path | None = None) -> Path:
    """Write the user-local application-menu shortcut and return its path."""
    root = repository_root or Path(__file__).resolve().parents[1]
    destination = desktop_entry_path(data_home)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(desktop_entry_text(root), encoding="utf-8")
    destination.chmod(0o644)
    return destination


def main() -> int:
    """Install the shortcut for this checkout."""
    destination = install_shortcut()
    print(f"Installed Sound Companion shortcut at {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
