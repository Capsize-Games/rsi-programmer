#!/usr/bin/env python3
"""Gains charts from committed measurements (Issue 18).

Reads docs/gains/gains.csv (measured release numbers only,
never backfilled) and emits PNGs beside it. --check validates
the CSV without plotting, so the test gate needs no
matplotlib; chart regen does (uv pip install matplotlib).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent / "docs" / "gains"
CSV = HERE / "gains.csv"
RELEASES = ("code-v12", "code-v13", "code-v17", "code-v20",
            "code-v21", "code-v26", "code-v27", "code-v29")
STRATA = ("twin", "plain", "full-ir")


def load() -> list[dict[str, str]]:
    """Parse and validate the committed gains table."""
    with open(CSV, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "gains.csv is empty"
    for row in rows:
        assert row["release"] in RELEASES, row
        assert row["stratum"] in STRATA, row
        for key in ("narrow_solve", "broad_solve"):
            assert 0.0 <= float(row[key]) <= 1.0, row
        for key in ("blind_narrow", "blind_broad"):
            assert -1.0 <= float(row[key]) <= 1.0, row
        assert float(row["enum_tried"]) >= 0, row
    return rows


def plot(rows: list[dict[str, str]]) -> list[str]:
    """Render solve bars and blindness trend; return paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out: list[str] = []
    width, pos = 0.2, {"twin": 0, "plain": 1, "full-ir": 2}
    series = [(rel, who) for rel in RELEASES
              for who in ("narrow_solve", "broad_solve")]
    off = {key: (num - (len(series) - 1) / 2) * width
           for num, key in enumerate(series)}
    tint = {key: "C%d" % num for num, key in
            enumerate(series)}
    fig, ax = plt.subplots(figsize=(8, 4))
    for row in rows:
        x = pos[row["stratum"]]
        for who in ("narrow_solve", "broad_solve"):
            key = (row["release"], who)
            ax.bar(x + off[key], float(row[who]), width,
                   color=tint[key],
                   label="%s %s" % (row["release"],
                                    who.split("_")[0])
                   if x == 0 else "")
    ax.set_xticks([0, 1, 2], ["twin", "plain", "full-ir"])
    ax.set_ylabel("frozen solve rate")
    ax.set_title("OOD solve rate per release (8 tasks each)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    solos = str(HERE / "gains_solves.png")
    fig.savefig(solos)
    out.append(solos)
    plt.close(fig)

    seen: dict[str, dict[str, str]] = {}
    for row in rows:
        seen.setdefault(row["release"], row)
    order = [r for r in RELEASES if r in seen]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(order, [float(seen[r]["blind_narrow"])
                    for r in order], marker="o",
            label="narrow")
    ax.plot(order, [float(seen[r]["blind_broad"])
                    for r in order], marker="o",
            label="broad (contrast from v13)")
    ax.set_ylabel("guidance-blindness (1.0 = blind)")
    ax.set_title("Guidance blindness across the mirror")
    ax.legend(fontsize=8)
    fig.tight_layout()
    blind = str(HERE / "gains_blindness.png")
    fig.savefig(blind)
    out.append(blind)
    plt.close(fig)
    return out


def main(argv: list[str]) -> int:
    """--check validates; default validates then plots."""
    rows = load()
    print("gains rows=%d releases=%s" % (len(rows), RELEASES))
    if "--check" in argv:
        return 0
    for path in plot(rows):
        print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
