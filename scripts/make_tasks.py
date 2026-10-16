#!/usr/bin/env python3
"""Generate the frozen task corpus (Issue 06).

Writes fixtures/rsi_gen_train.json (24 tasks, seed 7) and
fixtures/rsi_gen_novel.json (12 tasks, seed 21) with bound 4 and
mixed const/branch requirements. Same seeds always yield
byte-identical files; verify with the split-hash test.
"""
from __future__ import annotations

from rsi_programmer.families import generate_family, write_family

SPECS = (("fixtures/rsi_gen_train.json", 7, 24, "gen_train"),
         ("fixtures/rsi_gen_novel.json", 21, 12, "gen_novel"))


def main() -> int:
    """Generate both families with their frozen seeds."""
    for path, seed, count, prefix in SPECS:
        family = generate_family(seed, count, 4, 2, prefix)
        write_family(path, family)
        print("wrote %s tasks=%d seed=%d"
              % (path, len(family["tasks"]), seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
