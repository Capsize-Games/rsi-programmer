"""Frozen Phase 6 dev-diagnostic protocol and bar rule.

The task list, arms, caps and proceed/kill bars mirror Kanban
card ``bce2d353-1794-4466-b747-fee7d0a1a04`` and are hashed
before any execution starts. :func:`evaluate_bar` computes the
PROCEED/KILL verdict from final accepts with no post-run
threshold edits.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from rsi_programmer.llm_proposer import MODEL_ID, MODEL_VERSION
from rsi_programmer.llm_repair import CPU_CAP_S, QUERY_REVEAL_CAP
from rsi_programmer.llm_repair import R_MAX, TOKEN_CAP
from rsi_programmer.llm_repair import template_sha256

CARD_ID = "bce2d353-1794-4466-b747-fee7d0a1a04"
BANK_ID = "llm_bank_v1"
SEED = 20261125
TOP_K = 3
COST_MULTIPLE_M = 3
DEEP_FAMILIES = ("A7", "A8", "S2", "S3")
ARMS = ("a", "b", "c", "d", "e")
LLM_ARMS = ("a", "b", "c")
CANDIDATE_BUDGET_D = 50000
MAX_OPS_D = {"A6": 6, "A7": 7, "A8": 8}

FAMILY_TASKS: dict[str, tuple[str, ...]] = {
    "A6": ("llm_dev_004", "llm_dev_002", "llm_dev_011",
           "llm_dev_010", "llm_dev_009"),
    "A7": ("llm_dev_018", "llm_dev_022", "llm_dev_020",
           "llm_dev_016", "llm_dev_021"),
    "A8": ("llm_dev_039", "llm_dev_031", "llm_dev_030",
           "llm_dev_042", "llm_dev_043"),
    "S1": ("llm_dev_047", "llm_dev_050", "llm_dev_045",
           "llm_dev_049", "llm_dev_048"),
    "S2": ("llm_dev_068", "llm_dev_065", "llm_dev_062",
           "llm_dev_072", "llm_dev_073"),
    "S3": ("llm_dev_076", "llm_dev_088", "llm_dev_083",
           "llm_dev_081", "llm_dev_075"),
}
FROZEN_TASK_IDS: tuple[str, ...] = tuple(
    task for family in ("A6", "A7", "A8", "S1", "S2", "S3")
    for task in FAMILY_TASKS[family])


def _canonical(value: Any) -> bytes:
    """Canonical JSON bytes for hashing protocol records."""
    return (json.dumps(value, indent=2, sort_keys=True)
            + "\n").encode("utf-8")


@dataclass(frozen=True)
class DevProtocol:
    """Frozen Phase 6 protocol; hashed before any execution."""

    bank_id: str = BANK_ID
    card_id: str = CARD_ID
    seed: int = SEED
    task_ids: tuple[str, ...] = FROZEN_TASK_IDS
    arms: tuple[str, ...] = ARMS
    max_rounds: int = R_MAX
    query_cap: int = QUERY_REVEAL_CAP
    token_cap: int = TOKEN_CAP
    cpu_cap_s: float = CPU_CAP_S
    cost_multiple_m: int = COST_MULTIPLE_M
    deep_families: tuple[str, ...] = DEEP_FAMILIES
    model_id: str = MODEL_ID
    model_version: str = MODEL_VERSION
    repair_template_sha256: str = field(
        default_factory=template_sha256)
    proposer_prompt: str = "canonical-json-v1"
    max_ops_d: Mapping[str, int] = field(
        default_factory=lambda: dict(MAX_OPS_D))
    candidate_budget_d: int = CANDIDATE_BUDGET_D
    top_k: int = TOP_K
    library_plan: str = (
        "build in-run from dev-held-accepted a/b programs "
        "via FragmentLibrary.add_entry with judge_final "
        "evidence, then freeze (version_hash) before arm c")
    held_policy: str = (
        "dev tasks/visible/query plus dev held opened "
        "identically for all arms at final; sealed pool "
        "never loaded (ids must start with llm_dev_)")
    verdict_rule: str = (
        "PROCEED iff best LLM arm accepts >=1 deep-family "
        "task where streaming accepts 0 AND cost/accepted "
        "<=Mx streaming on shared-solved tasks AND b>a or "
        "c>b strictly on identical tasks; else KILL")

    def __post_init__(self) -> None:
        """Refuse empty or non-dev task lists at freeze time."""
        if not self.task_ids:
            raise ValueError("protocol task list is empty")
        for task_id in self.task_ids:
            if not str(task_id).startswith("llm_dev_"):
                raise ValueError(
                    "sealed pool is out of scope: "
                    f"{task_id!r}")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe copy of the frozen protocol fields."""
        body = asdict(self)
        body["task_ids"] = list(self.task_ids)
        body["arms"] = list(self.arms)
        body["deep_families"] = list(self.deep_families)
        body["max_ops_d"] = dict(self.max_ops_d)
        return body

    def sha256(self) -> str:
        """Hex digest identifying this exact protocol run."""
        return hashlib.sha256(
            _canonical(self.to_dict())).hexdigest()


def kind_for_interface(interface: str) -> str:
    """Candidate kind for a bank interface label."""
    if interface == "pair":
        return "graph"
    if interface == "list":
        return "statement"
    raise ValueError(f"unknown interface: {interface!r}")


def evaluate_bar(
        llm_accepted: Mapping[str, set[str]],
        streaming_accepted: set[str],
        family_of: Mapping[str, str],
        arm_cpu: Mapping[str, Mapping[str, float]],
        *,
        deep_families: tuple[str, ...] = DEEP_FAMILIES,
        multiple_m: int = COST_MULTIPLE_M) -> dict[str, Any]:
    """PROCEED/KILL from the frozen bar on final accepts.

    The best LLM arm holds the most final accepts (ties
    break on lower total CPU, then arm order a<b<c).
    Prong (i) needs a deep-family accept where streaming
    scores zero; prong (ii) bounds cost per accepted on
    shared-solved tasks within the frozen multiple; prong
    (iii) needs a strict b-vs-a or c-vs-b delta.
    """
    order = [arm for arm in LLM_ARMS if arm in llm_accepted]
    if not order:
        return {"verdict": "KILL", "best_arm": None,
                "prongs": {}, "reasons": ["no LLM arms ran"]}
    totals = {arm: sum(arm_cpu.get(arm, {}).values())
              for arm in order}
    best = min(order, key=lambda arm: (
        -len(llm_accepted[arm]), totals[arm],
        LLM_ARMS.index(arm)))
    best_set = set(llm_accepted[best])
    deep = sorted(task for task in best_set
                  if family_of.get(task) in deep_families
                  and task not in streaming_accepted)
    prong_deep = bool(deep)
    shared = sorted(task for task in best_set
                    if task in streaming_accepted)
    stream_cpu = arm_cpu.get("d", {})
    cost_best = sum(arm_cpu.get(best, {}).get(t, 0.0)
                    for t in shared)
    cost_stream = sum(stream_cpu.get(t, 0.0) for t in shared)
    if shared and cost_stream > 0:
        ratio: float | None = cost_best / cost_stream
        prong_cost = ratio <= multiple_m
    else:
        ratio = None
        prong_cost = False
    count_a = len(llm_accepted.get("a", set()))
    count_b = len(llm_accepted.get("b", set()))
    count_c = len(llm_accepted.get("c", set()))
    prong_delta = count_b > count_a or count_c > count_b
    verdict = ("PROCEED" if prong_deep and prong_cost
               and prong_delta else "KILL")
    reasons = []
    if prong_deep:
        reasons.append(f"deep reach: {best} holds {deep}")
    else:
        reasons.append(f"no deep reach: {best} adds nothing "
                       f"where streaming scores 0")
    if ratio is None:
        reasons.append("cost ratio undefined: no shared "
                       "solves or zero streaming cost")
    else:
        reasons.append(f"cost/accepted ratio {ratio:.3f} "
                       f"vs M={multiple_m}")
    reasons.append(f"counts a/b/c={count_a}/{count_b}/"
                   f"{count_c}")
    return {"verdict": verdict, "best_arm": best,
            "prongs": {"deep_reach": prong_deep,
                       "deep_tasks": deep,
                       "cost_ratio": ratio,
                       "shared_tasks": shared,
                       "multiple_m": multiple_m,
                       "delta": prong_delta,
                       "counts": {"a": count_a, "b": count_b,
                                  "c": count_c}},
            "reasons": reasons}
