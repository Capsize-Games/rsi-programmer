#!/usr/bin/env python3
"""Generate the frozen task corpus (Issues 06 and 12).

Writes fixtures/rsi_gen_train.json (24 tasks, seed 7),
fixtures/rsi_gen_novel.json (12 tasks, seed 21),
fixtures/rsi_ood_tasks.json (8 mirrored tasks, seed 33), and
fixtures/rsi_ood_plain.json (8 mirrored plain-OPS tasks,
seed 34), and fixtures/rsi_ood_twin.json (8 unmirrored
row-aligned reference tasks, seed 34) with bound 4. Same
seeds always yield byte-identical files; verify with the
split-hash test.
"""
from __future__ import annotations

from rsi_programmer.families import generate_family, write_family

SPECS = (("fixtures/rsi_gen_train.json", 7, 24, "gen_train",
          False, True, 2),
         ("fixtures/rsi_gen_novel.json", 21, 12, "gen_novel",
          False, True, 2),
         ("fixtures/rsi_ood_tasks.json", 33, 8, "gen_ood",
          True, True, 2),
         ("fixtures/rsi_ood_plain.json", 34, 8, "gen_ood_p",
          True, False, 0),
         ("fixtures/rsi_ood_twin.json", 34, 8, "gen_ood_u",
          False, False, 0))


def main() -> int:
    """Generate all families with their frozen seeds."""
    for path, seed, count, prefix, mirror, consts, every in \
            SPECS:
        family = generate_family(seed, count, 4, every, prefix,
                                 mirror=mirror, consts=consts)
        write_family(path, family)
        print("wrote %s tasks=%d seed=%d mirror=%s consts=%s"
              % (path, len(family["tasks"]), seed, mirror,
                 consts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
