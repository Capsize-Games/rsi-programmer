"""Launch the local, teachable sound companion desktop window."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="Physical microphone source")
    parser.add_argument(
        "--state-dir", type=Path,
        default=Path.home() / ".local/state/rsi-programmer/sound-companion")
    args = parser.parse_args()
    directory = args.state_dir.expanduser().resolve()
    repository = Path(__file__).resolve().parents[1]
    if directory == repository or repository in directory.parents:
        parser.error("sound memory must stay outside the repository")
    existed = directory.exists()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not existed:
        os.chmod(directory, 0o700)
    instance = (directory / "desktop.lock").open("a+b")
    os.fchmod(instance.fileno(), 0o600)
    try:
        fcntl.flock(instance.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("Sound Companion is already open for this memory folder")
    import tkinter as tk
    import torch

    from rsi_programmer.sound_controller import SoundController
    from rsi_programmer.sound_memory import SoundMemory

    torch.set_num_threads(1)
    root = tk.Tk(className="SoundCompanion")
    controller = SoundController(
        root, SoundMemory(directory / "memory.json"), args.source)
    try:
        root.mainloop()
    finally:
        if not controller._closed:
            controller.close()
        instance.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
