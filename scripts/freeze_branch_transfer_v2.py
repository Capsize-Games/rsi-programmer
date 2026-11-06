"""Freeze unseen branch functions for transfer bank v2.

Same two-branch grammar family as v1 (identical branch library,
pattern/instantiate and argument grammar), six functions per donor
group instead of two. Separation is stronger than v1's 24 probes:
see scripts/branch_transfer_v2_probes.py. No learner score or solve
result is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.branch_transfer_v2_admit import SHUFFLE_SEED
from scripts.branch_transfer_v2_admit import Bank
from scripts.branch_transfer_v2_admit import Probes
from scripts.branch_transfer_v2_admit import Rej
from scripts.branch_transfer_v2_admit import arguments
from scripts.branch_transfer_v2_admit import fill_group
from scripts.branch_transfer_v2_probes import BOUNDARY
from scripts.branch_transfer_v2_probes import EXTRA_PROBE_SEED
from scripts.branch_transfer_v2_probes import Sig
from scripts.branch_transfer_v2_probes import arrays
from scripts.branch_transfer_v2_probes import build_exclusion
from scripts.branch_transfer_v2_probes import canonical_probes
from scripts.branch_transfer_v2_probes import extended_probes
from scripts.branch_transfer_v2_probes import training_names
from scripts.branch_transfer_v2_probes import v1_inputs
from scripts.branch_transfer_v2_probes import v1_references
from rsi_programmer.statement_branch_library import qualify_branches

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "fixtures/frozen_branch_transfer_v1"
OBSERVED_SEED = 613131
QUERY_SEED = 613137
HELD_SEED = 613143


def _dev_programs() -> list[tuple[str, ...]]:
    report = json.loads((ROOT / "artifacts/statement-composed-route-v1"
                         "-20261101/report.json").read_text())
    return [tuple(row["chosen"]["tokens"]) for row in report["rows"]]


def _check_inputs(observed: Probes, queries: Probes,
                  held: Probes) -> None:
    visible = {tuple(v) for v in observed + queries}
    if visible & {tuple(v) for v in held}:
        raise ValueError("v2 visible and held inputs overlap")
    fresh = (visible | {tuple(v) for v in held}) - {
        tuple(v) for v in BOUNDARY}
    if fresh & v1_inputs():
        raise ValueError("v2 seeded inputs reuse v1 inputs; reseed")


def _exclusion_dict(names: list[str], training: int, probes: Probes,
                    canonical: int, v1_held_count: int) -> dict[str, Any]:
    return {
        "canonical_probes": canonical,
        "v1_held_probes": v1_held_count,
        "extra_probes": len(probes) - canonical - v1_held_count,
        "extra_probe_seed": EXTRA_PROBE_SEED,
        "total_probes": len(probes),
        "excluded_sources": {
            "curriculum_programs": training,
            "development_names": names,
            "v1_references": 18,
            "v1_tasks_sha256": hashlib.sha256(
                (V1 / "tasks.json").read_bytes()).hexdigest()},
        "exact_token_exclusion": {
            "v1_references": 18, "dev_programs": 11},
        "signature_rule": "Outputs compared per probe; an engine "
                          "error records 'ERR'. Two functions are "
                          f"distinct when their {len(probes)}-"
                          "signatures differ anywhere."}


def _bank_data(bank: Bank, queries: Probes, names: list[str],
               training: int, probes: Probes, rejected: Rej,
               canonical: int, v1_held_count: int) -> dict[str, Any]:
    scope = ("Frozen unseen-function transfer within the supplied "
             "two-branch grammar; not unrestricted programming.")
    return {
        "scope": scope, "tasks": bank, "query_inputs": queries,
        "exclusion": _exclusion_dict(names, training, probes,
                                     canonical, v1_held_count),
        "training_functions": training, "rejected": rejected,
        "seeds": {"observed": OBSERVED_SEED, "queries": QUERY_SEED,
                  "held": HELD_SEED, "shuffle": SHUFFLE_SEED}}


def _protocol(bank_size: int, held_size: int) -> dict[str, Any]:
    return {
        "source_checkpoint": "v2-freeze",
        "frozen_before_solver_use": True, "tasks": bank_size,
        "methods": ["classical", "neural", "untrained"],
        "seeds": {"neural": [0, 1, 2], "untrained": [0, 1, 2],
                  "classical": [None]},
        "request_cpu_budget_seconds": 1.0,
        "budget_overrun_tolerance_seconds": 0.05,
        "max_added_examples": 3, "branch_width": 8,
        "first_limit": 3, "held_per_task": held_size,
        "cost_policy": "End-to-end from an empty donor library, "
                       "including library creation, fits, search, "
                       "visible feedback, emission and actual checks.",
        "actual_fit_usage_volume": bank_size,
        "held_access": "Workers load tasks and visible queries only; "
                       "held inputs load after every arm finishes.",
        "generator_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest()}


def _write_bank(root: Path, bank: Bank, queries: Probes,
                held: Probes, names: list[str], training: int,
                probes: Probes, rejected: Rej, canonical: int,
                v1_held_count: int) -> dict[str, str]:
    data = _bank_data(bank, queries, names, training, probes,
                      rejected, canonical, v1_held_count)
    protocol = _protocol(len(bank), len(held))
    for name, value in (("tasks.json", data), ("held.json", held),
                        ("protocol.json", protocol)):
        (root / name).write_text(json.dumps(value, indent=2) + "\n")
    hashes = {n: hashlib.sha256((root / n).read_bytes()).hexdigest()
              for n in ("tasks.json", "held.json", "protocol.json")}
    (root / "sha256.json").write_text(json.dumps(hashes, indent=2)
                                      + "\n")
    return hashes


def _library() -> list[Any]:
    base = ROOT / "artifacts/statement-context-repair-v1-20261101"
    extra = ROOT / "artifacts/positive-counter-v1-20261101"
    branches = qualify_branches(base)["branches"]
    return branches + qualify_branches(extra)["branches"]


def _seeded_inputs() -> tuple[Probes, Probes, Probes]:
    observed = arrays(OBSERVED_SEED, 3, 3, 12, 8)
    queries = arrays(QUERY_SEED, 24, 3, 100, 100)
    extra = arrays(HELD_SEED, 14, 13, 100, 100)
    return observed, queries, extra


def _collect(branches: list[Any], excluded: set[Sig], probes: Probes,
             observed: Probes, queries: Probes, held: Probes
             ) -> tuple[Bank, Rej]:
    exact = set(v1_references()) | {tuple(p) for p in _dev_programs()}
    bank: Bank = []
    rejected: Rej = {"syntax_or_runtime": 0, "known_function": 0,
                     "exact_repeat": 0}
    args = arguments()
    for first in branches:
        fill_group(first, branches, args, excluded, probes, observed,
                   queries, held, exact, bank, rejected)
    return bank, rejected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    names = training_names()
    canonical = canonical_probes(names)
    probes = extended_probes(canonical)
    excluded, training = build_exclusion(names, probes)
    observed, queries, extra = _seeded_inputs()
    held = [list(v) for v in BOUNDARY] + extra
    _check_inputs(observed, queries, held)
    bank, rejected = _collect(_library(), excluded, probes, observed,
                              queries, held)
    v1_held = len(json.loads((V1 / "held.json").read_text()))
    hashes = _write_bank(root, bank, queries, held, names, training,
                         probes, rejected, len(canonical), v1_held)
    print(json.dumps({"tasks": len(bank), "hashes": hashes}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
