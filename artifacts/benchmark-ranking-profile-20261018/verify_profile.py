"""Recheck retained profile artifacts without training or search."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main() -> None:
    folder = Path(__file__).resolve().parent
    root = folder.parents[1]
    manifest = json.loads((folder / "manifest.json").read_text())
    receipt = json.loads((folder / "execution.json").read_text())
    rows = [json.loads(line) for line in
            (folder / "raw.jsonl").read_text().splitlines()]
    assert len(rows) == 480
    assert len(list((folder / "cprofile").glob("*.prof"))) == 240
    for name, key in (("raw.jsonl", "raw_jsonl_sha256"),
                      ("profile_summary.json", "profile_summary_sha256")):
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == (
            receipt[key])
    changed = []
    for name, expected in manifest["source_and_input"][
            "files_sha256"].items():
        path = Path(name) if Path(name).is_absolute() else root / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            changed.append(name)
    assert set(changed) == {
        "rsi_programmer/bounded_pre_eval.py",
        "scripts/profile_benchmark_ranking.py",
    }, changed
    print(json.dumps({
        "valid": True, "rows": len(rows), "profiles": 240,
        "changed_captured_files": changed,
        "active_baseline_sources_unchanged": True,
        "excluded_changes": "unused bounded engine; post-run profile report "
        "projection code; original binary profiles retained",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
