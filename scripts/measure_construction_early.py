"""Run a bounded construction development trial; never infer on final tasks."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch

from rsi_programmer.construction_beam import beam_construct
from rsi_programmer.construction_check import check_graph
from rsi_programmer.construction_greedy import greedy_construct
from rsi_programmer.construction_policy import ConstructionPolicy
from rsi_programmer.construction_training import fit_constructor
from rsi_programmer.pre_eval import pre_evaluation_search

ROOT = Path(__file__).resolve().parents[1]


def evaluate(tasks: list[dict[str, Any]], method: str,
             model: ConstructionPolicy | None
             ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Count independently correct functions, with fixed decoder settings."""
    if any(task["split"] != "dev" for task in tasks):
        raise ValueError("early measurements are restricted to development")
    before = model.fingerprint() if model is not None else None
    started = time.process_time()
    rows: list[dict[str, Any]] = []
    by_size: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for task in tasks:
        observed = tuple(tuple(row) for row in task["observed"])
        if method == "streaming":
            out = pre_evaluation_search(
                None, tuple((x, y) for x, y, _ in observed),
                tuple(value for _, _, value in observed), 4, 16384,
            )
        elif method == "greedy":
            assert model is not None
            out = greedy_construct(model, observed)
        else:
            out = beam_construct(model, observed, budget=256, width=8, seed=0)
        graph = out["graph"]
        check = check_graph(
            graph, observed, tuple(tuple(row) for row in task["held"]),
            tuple(tuple(term) for term in task["generator"]["polynomial"]),
        )
        size = str(task["generator"]["minimal_tree_ops"])
        by_size[size][0] += int(check["accepted"])
        by_size[size][1] += 1
        rows.append({"id": task["id"], "accepted": check["accepted"],
                     "graph": graph, "actions": out.get("actions"),
                     "executions": out.get("executions"),
                     "error": out.get("error"), "checker": check})
    cpu = time.process_time() - started
    if model is not None and model.fingerprint() != before:
        raise ValueError("evaluation changed neural weights")
    return {"accepted": sum(row["accepted"] for row in rows),
            "tasks": len(rows), "cpu_seconds": cpu,
            "by_minimal_tree_ops": dict(by_size)}, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("ann", "snn"), default="ann")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--epochs", type=int, choices=(20, 100), default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    corpus_path = ROOT / "fixtures" / "construction_corpus_v1.json"
    protocol = json.loads((ROOT / "fixtures/construction_v1.json").read_text())
    raw = corpus_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != protocol["corpus_sha256"]:
        raise ValueError("frozen construction corpus changed")
    corpus = json.loads(raw)
    model = ConstructionPolicy(seed=args.seed, backend=args.backend)
    untrained = ConstructionPolicy(seed=args.seed, backend=args.backend)
    print(f"Fitting {args.backend}, seed {args.seed}, {args.epochs} epochs.",
          flush=True)
    fit = fit_constructor(model, corpus["splits"]["train"], args.epochs)
    print(json.dumps({"fit": fit}), flush=True)
    report: dict[str, Any] = {
        "scope": "Development only, one seed; final split not inferred on.",
        "backend": args.backend, "seed": args.seed, "fit": fit,
        "corpus_sha256": protocol["corpus_sha256"],
        "timing": "CPU of evaluation loop including independent checks; "
                  "excludes fitting, initialization, hashing and report I/O.",
        "decoders": "Greedy rejects invalid edits; beam additionally masks "
                    "illegal operands. Their difference is not search alone.",
        "controls": {}, "rows": {},
    }
    for name, method, policy in (
        ("trained_greedy", "greedy", model),
        ("trained_beam", "beam", model),
        ("untrained_beam", "beam", untrained),
        ("random_beam", "beam", None),
        ("streaming", "streaming", None),
    ):
        summary, rows = evaluate(corpus["splits"]["dev"], method, policy)
        report["controls"][name], report["rows"][name] = summary, rows
        print(json.dumps({"control": name, **summary}), flush=True)
    sources = [*Path(ROOT / "rsi_programmer").glob("construction_*.py"),
               Path(__file__)]
    report["source_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sources
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2)
                                            + "\n")
    print("COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
