"""Frozen inputs for the CPU-matched benchmark: hashes and contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rsi_programmer import benchmark_v2_runner as v2runner
from rsi_programmer.cpu_matched_verdict import KILL_CRITERION

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "fixtures/benchmark_cpu_matched_v1.json"
CORPUS_PATH = ROOT / "fixtures/benchmark_cpu_matched_corpus_v1.json"
V1_CORPUS_PATH = ROOT / "fixtures/benchmark_pre_eval_corpus_v1.json"
V2_CORPUS_PATH = ROOT / "fixtures/benchmark_pre_eval_corpus_v2.json"
GENERATOR_PATHS = (
    ROOT / "scripts/generate_benchmark_cpu_matched_corpus.py",
    ROOT / "rsi_programmer/benchmark_v2_corpus.py",
    ROOT / "rsi_programmer/cpu_budget.py",
    ROOT / "rsi_programmer/cpu_matched_verdict.py",
)
PROTOCOL_SHA256 = (
    "a661eddce10671bea4666185d1c6e646ebb3dd5072a8da159b396ace6eb43a14")
CORPUS_SHA256 = (
    "494e5d04bee656afb090a06245a2cc45c1a048dddfa21bdeac2f1c0738a8fd6d")
ARMS = ("streaming", "heuristic", "seeded_random", "untrained_mlp",
        "trained_mlp", "trained_recurrent")


def sha256_bytes(data: bytes) -> str:
    """Return a lowercase SHA256 digest."""
    return hashlib.sha256(data).hexdigest()


def load_frozen_inputs(
        protocol_path: Path = PROTOCOL_PATH,
        corpus_path: Path = CORPUS_PATH,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify pinned bytes before parsing protocol and corpus."""
    protocol_bytes = protocol_path.read_bytes()
    corpus_bytes = corpus_path.read_bytes()
    if sha256_bytes(protocol_bytes) != PROTOCOL_SHA256:
        raise ValueError("frozen cpu-matched protocol hash mismatch")
    if sha256_bytes(corpus_bytes) != CORPUS_SHA256:
        raise ValueError("frozen cpu-matched corpus hash mismatch")
    protocol = json.loads(protocol_bytes)
    corpus = json.loads(corpus_bytes)
    for path in GENERATOR_PATHS:
        label = str(path.relative_to(ROOT))
        expected = protocol["corpus"]["generator_source"][label]
        if sha256_bytes(path.read_bytes()) != expected:
            raise ValueError(f"frozen generator hash mismatch: {label}")
    if protocol["corpus"]["sha256"] != CORPUS_SHA256:
        raise ValueError("protocol does not pin the frozen corpus")
    validate_protocol(protocol)
    v2runner.validate_corpus(corpus, protocol)
    check_disjoint(corpus, protocol)
    return protocol, corpus


def validate_protocol(protocol: dict[str, Any]) -> None:
    """Reject drift from the frozen CPU-matched contract."""
    if protocol.get("protocol_id") != "benchmark-cpu-matched-v1":
        raise ValueError("unexpected cpu-matched protocol ID")
    if tuple(protocol.get("arms", ())) != ARMS:
        raise ValueError("arm set differs from cpu-matched contract")
    if any("snn" in arm for arm in ARMS):
        raise ValueError("cpu-matched arms must exclude SNN")
    training = protocol["training"]
    if training["seeds"] != [0, 1, 2] or training["epochs"] != 40:
        raise ValueError("training seeds or epochs differ")
    if training["learning_rate"] != 0.05:
        raise ValueError("learning rate differs from contract")
    if training["positive_cap"] != 16 or training["negative_cap"] != 256:
        raise ValueError("training group caps differ")
    if training["sampling_seed"] != 83078:
        raise ValueError("training sampling seed differs")
    evaluation = protocol["evaluation"]
    if evaluation["tree_bound"] != 5:
        raise ValueError("tree bound differs from contract")
    if evaluation["candidate_backstop"] != 131072:
        raise ValueError("candidate backstop differs")
    cost = protocol["cost"]
    if not cost["cell_budget_cpu_seconds"] > 0.0:
        raise ValueError("cell CPU budget must be positive")
    if cost["amortization_volume_per_seed"] != 48:
        raise ValueError("amortization volume differs")
    rule = protocol["decision_rule"]
    if rule["verbatim_kill"] != KILL_CRITERION:
        raise ValueError("kill criterion text differs")
    if set(rule["learned_arms"]) != {"trained_mlp",
                                     "trained_recurrent"}:
        raise ValueError("learned arms differ from contract")


Poly = tuple[tuple[tuple[int, int], int], ...]


def _combine(left: Poly, right: Poly, op: str) -> Poly:
    """Add, subtract or multiply exact integer polynomials."""
    if op == "mul":
        result: dict[tuple[int, int], int] = {}
        for (lx, ly), lv in left:
            for (rx, ry), rv in right:
                key = (lx + rx, ly + ry)
                result[key] = result.get(key, 0) + lv * rv
        return tuple((x, y, value)
                     for (x, y), value in sorted(result.items())
                     if value)
    merged = dict(left)
    factor = 1 if op == "add" else -1
    for key, value in right:
        merged[key] = merged.get(key, 0) + factor * value
    return tuple((x, y, value) for (x, y), value in
                 sorted(merged.items()) if value)


def _canonical(node: Any) -> tuple[tuple[int, int, int], ...]:
    """Canonicalize a witness program without trusting stored forms."""
    if not isinstance(node, dict):
        raise ValueError("malformed witness program")
    if node.get("op") == "input":
        name = node.get("name")
        if name not in {"x", "y"}:
            raise ValueError("unknown witness input")
        return ((1, 0, 1),) if name == "x" else ((0, 1, 1),)
    op = node.get("op")
    if op not in {"add", "sub", "mul"}:
        raise ValueError("unknown witness operator")
    left = dict(((x, y), coefficient) for x, y, coefficient in
                _canonical(node.get("left")))
    right = dict(((x, y), coefficient) for x, y, coefficient in
                 _canonical(node.get("right")))
    return _combine(tuple(left.items()), tuple(right.items()), op)


def check_disjoint(corpus: dict[str, Any],
                   protocol: dict[str, Any]) -> dict[str, int]:
    """Prove zero exact-function overlap with the v1/v2 banks."""
    old: set[tuple[tuple[int, int, int], ...]] = set()
    for entry in protocol["corpus"]["disjoint_from"]:
        data = (ROOT / entry["path"]).read_bytes()
        if sha256_bytes(data) != entry["sha256"]:
            raise ValueError("disjoint bank hash mismatch")
        bank = json.loads(data)
        for tasks in bank["splits"].values():
            for task in tasks:
                old.add(_canonical(task["generator"]["program"]))
    new = [_canonical(task["generator"]["program"])
           for tasks in corpus["splits"].values() for task in tasks]
    if set(new) & old or len(set(new)) != len(new):
        raise ValueError("cpu-matched bank is not function-disjoint")
    return {"old_functions": len(old), "new_functions": len(new),
            "overlap": 0}
