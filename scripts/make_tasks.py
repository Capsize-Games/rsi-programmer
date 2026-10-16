#!/usr/bin/env python3
"""Generate the frozen task corpus (Issues 06 and 12).

Writes fixtures/rsi_gen_train.json (24 tasks, seed 7),
fixtures/rsi_gen_novel.json (12 tasks, seed 21),
fixtures/rsi_ood_tasks.json (8 mirrored tasks, seed 33), and
fixtures/rsi_ood_plain.json (8 mirrored plain-OPS tasks,
seed 34), fixtures/rsi_ood_twin.json (8 unmirrored
row-aligned reference tasks, seed 34), and
fixtures/rsi_ood_v2.json (8 mirrored tasks with degenerate
branches rejected at generation, seed 35, strict), and
fixtures/rsi_plain_train.json (24 plain-OPS training tasks,
seed 55), fixtures/rsi_plain_hard.json (12 bound-5 plain
tasks, seed 56), and fixtures/rsi_plain_more.json (12
plain tasks, seed 57) with per-family bounds. Same seeds
always yield byte-identical files; verify with the
split-hash test.
"""
from __future__ import annotations

from rsi_programmer.families import generate_family, write_family

SPECS = (("fixtures/rsi_gen_train.json", 7, 24, "gen_train",
          False, True, 2, False, 4),
         ("fixtures/rsi_gen_novel.json", 21, 12, "gen_novel",
          False, True, 2, False, 4),
         ("fixtures/rsi_ood_tasks.json", 33, 8, "gen_ood",
          True, True, 2, False, 4),
         ("fixtures/rsi_ood_plain.json", 34, 8, "gen_ood_p",
          True, False, 0, False, 4),
         ("fixtures/rsi_ood_twin.json", 34, 8, "gen_ood_u",
          False, False, 0, False, 4),
         ("fixtures/rsi_ood_v2.json", 35, 8, "gen_ood2",
          True, True, 2, True, 4),
         ("fixtures/rsi_plain_train.json", 55, 24, "gen_plain",
          False, False, 0, False, 4),
         ("fixtures/rsi_plain_hard.json", 56, 12, "gen_plain_h",
          False, False, 0, False, 5),
         ("fixtures/rsi_plain_more.json", 57, 12, "gen_plain_m",
          False, False, 0, False, 4))


def main() -> int:
    """Generate all families with their frozen seeds."""
    for path, seed, count, prefix, mirror, consts, every, \
            strict, bound in SPECS:
        family = generate_family(seed, count, bound, every,
                                 prefix, mirror=mirror,
                                 consts=consts,
                                 strict=bool(strict))
        write_family(path, family)
        print("wrote %s tasks=%d seed=%d bound=%d mirror=%s "
              "consts=%s strict=%s"
              % (path, len(family["tasks"]), seed, bound,
                 mirror, consts, strict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
