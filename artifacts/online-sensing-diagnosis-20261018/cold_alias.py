#!/usr/bin/env python3
"""Audit saved cold-context rate-code aliases; never executes a model."""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def majority(contexts: list[dict[str, Any]], rounded: int | None = None
             ) -> tuple[int, int, list[dict[str, Any]]]:
    """Return majority-root count, distinct codes, and cross-root examples."""
    groups: dict[tuple[float, ...], list[dict[str, Any]]] = (
        collections.defaultdict(list))
    for row in contexts:
        rates = tuple(float(value) for value in row["rates"])
        code = rates if rounded is None else tuple(round(x, rounded)
                                                    for x in rates)
        groups[code].append(row)
    correct = 0
    clashes = []
    for code, rows in groups.items():
        roots = collections.Counter(row["root"] for row in rows)
        correct += max(roots.values())
        if len(roots) > 1:
            clashes.append({"rates": list(code), "roots": dict(roots),
                            "task_ids": [row["task_id"] for row in rows]})
    return correct, len(groups), clashes


def main() -> int:
    """Verify frozen retention provenance and save compact summaries."""
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=root / "artifacts" /
                        "online-plastic-encoder-v1-20261018")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("cold-alias.json"))
    args = parser.parse_args()
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    source = args.bundle / "retention.jsonl.gz"
    source_hash = digest(source)
    if source_hash != manifest["retention_raw_sha256"]:
        raise ValueError("retention SHA-256 does not match manifest")
    selected = []
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["budget"] == 144 and row["completed_tasks"] in (0, 48):
                selected.append(row)
    arms = {"persistent_snn", "frozen_snn", "episode_reset_snn",
            "online_nonspiking", "plastic_snn", "reset_plastic_snn",
            "plastic_nonspiking"}
    orders = {"-".join(map(str, order))
              for order in itertools.permutations(range(3))}
    expected = {(arm, order, seed, boundary) for arm in arms
                for order in orders for seed in range(3)
                for boundary in (0, 48)}
    identities = [(row["arm"], row["order_id"], row["seed"],
                   row["completed_tasks"]) for row in selected]
    if (len(selected) != 252 or set(identities) != expected
            or any(len(row["contexts"]) != 48 for row in selected)):
        raise ValueError("expected 252 groups with 48 contexts each")
    detail = []
    for row in selected:
        exact, codes, clashes = majority(row["contexts"])
        rounded, rounded_codes, _ = majority(row["contexts"], rounded=12)
        if exact != rounded or codes != rounded_codes:
            raise ValueError("exact and rounded12 code results disagree")
        detail.append({"arm": row["arm"], "order": row["order_id"],
                       "seed": row["seed"], "boundary": row[
                           "completed_tasks"], "exact_correct": exact,
                       "exact_accuracy": exact / 48, "exact_codes": codes,
                       "rounded12_correct": rounded,
                       "rounded12_codes": rounded_codes,
                       "rounded12_agrees": exact == rounded,
                       "cross_root_collisions": len(clashes),
                       "example": clashes[0] if clashes else None})
    initial = [row for row in detail if row["boundary"] == 0]
    final = [row for row in detail if row["boundary"] == 48]
    initial_by_seed = []
    for arm in ("plastic_snn", "online_nonspiking"):
        for seed in range(3):
            values = [row["exact_accuracy"] for row in initial
                      if row["arm"] == arm and row["seed"] == seed]
            initial_by_seed.append({"arm": arm, "seed": seed,
                                    "mean_accuracy": sum(values) /
                                    len(values)})
    final_by_order = []
    for arm in ("plastic_snn", "online_nonspiking"):
        for order in sorted({row["order"] for row in final}):
            values = [row["exact_accuracy"] for row in final
                      if row["arm"] == arm and row["order"] == order]
            final_by_order.append({"arm": arm, "order": order,
                                   "mean_accuracy": sum(values) /
                                   len(values)})
    result = {"source_sha256": source_hash, "filtered_groups": len(detail),
              "contexts_per_group": 48,
              "interpretation": "Optimistic lookup bound on cold context "
              "codes only; root labels are analysis-only, "
              "not training labels; "
              "not attained accuracy or a bound on the whole sequential "
              "policy.", "initial_by_seed": initial_by_seed,
              "final_by_order": final_by_order, "groups": detail}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output),
                      "filtered_groups": len(detail),
                      "all_rounded12_agree": all(
                          row["rounded12_agrees"] for row in detail)},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
