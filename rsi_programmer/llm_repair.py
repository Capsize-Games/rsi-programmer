"""Bounded counterexample-guided repair loop (Spark 1.3 arm b).

Round 0 is the incoming single-shot proposal; each later round
sends the previous candidate plus sanitized judge diagnostics
plus ONE deterministic failing batch (first failing row plus
boundary rows) under a versioned, hashed prompt template that
instructs a general fix and restates i64 bounds. Query rows are
revealed only after a candidate passes every visible row, up to
a frozen per-task cap; the revealed set is logged per task and
checked disjoint from held rows with
:func:`assert_disjoint_from_held` (the loop itself never takes
held rows). Repair drives only the repair-facing judge API
(``judge_visible`` plus ``counterexamples`` scoped to visible
or query); any held scope raises before feedback is built.

Stop on accept, round cap ``R_MAX``, per-task token/CPU cap,
a repeated candidate hash (declared loop), or a round verdict
outside the value-mismatch set below (repair continues only
for executable candidates with wrong values, never for
structural, gate, timeout, or backend-error failures).
Every round is stored with full provenance, and the round-0
verdict stays recoverable as the single-shot outcome from the
same artifact. No network calls, no LLM calls, and no lineage
imports.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from typing import Any

from rsi_programmer.cost_ledger import CostLedger
from rsi_programmer.llm_proposer import ProposerAdapter

R_MAX = 3
QUERY_REVEAL_CAP = 3
TOKEN_CAP = 4000
CPU_CAP_S = 5.0
BOUNDARY_BATCH_N = 3
PROMPT_TEMPLATE_VERSION = "llm_repair_prompt_v1"

REPAIR_TEMPLATE = (
    "Propose a general fix for the previous candidate so it "
    "passes every visible row and the shown failing batch. "
    "Repair the underlying rule; do not special-case rows. "
    "Every expected and actual output must lie in i64 bounds "
    "[-2**63, 2**63 - 1]; bank rows stay within 2**62."
)

KINDS = ("graph", "statement")

VALUE_MISMATCH_TAXONOMIES = frozenset(
    {"python_mismatch", "curlee_mismatch",
     "identity_mismatch"})


def _is_value_miss(lite: Mapping[str, Any]) -> bool:
    """True when repair may continue from this round verdict."""
    taxonomy = lite.get("taxonomy") or []
    return len(taxonomy) == 1 and taxonomy[0] in \
        VALUE_MISMATCH_TAXONOMIES

_PAIR_BOUNDARY = ((0, 0), (6, 6), (-6, -6), (6, -6),
                  (-6, 6), (0, 6))
_LIST_BOUNDARY = ([], [0], [9], [-9], [0, 0],
                  [5, -3, 0, 7, -9, 2])

_DIAG_KEYS = ("task_id", "kind", "stage", "accepted",
              "valid", "python_correct", "curlee_correct",
              "identity_ok", "rows_checked", "mismatches",
              "first_mismatch", "taxonomy", "detail",
              "signature")

CandidateFn = Callable[
    [int, "str | None", list[dict[str, Any]]],
    tuple[str, Mapping[str, int], float]]


def template_sha256() -> str:
    """Hex sha256 of the frozen repair instruction template."""
    return hashlib.sha256(
        REPAIR_TEMPLATE.encode("utf-8")).hexdigest()


def row_key(row: list[Any]) -> str:
    """Stable identity key for one bank row of any shape."""
    return json.dumps(row, sort_keys=True,
                      separators=(",", ":"))


def assert_disjoint_from_held(
        revealed: list[list[Any]],
        held_rows: list[list[Any]]) -> None:
    """Refuse any revealed query row that also appears held."""
    held_keys = {row_key(row) for row in held_rows}
    overlap = [row for row in revealed
               if row_key(row) in held_keys]
    if overlap:
        raise ValueError(
            "revealed rows overlap held rows: "
            f"{len(overlap)} of {len(revealed)}")


def _input_of(kind: str, row: list[Any]) -> list[Any]:
    """Comparable input part of one bank row for a kind."""
    if kind == "graph":
        return [int(row[0]), int(row[1])]
    return [int(value) for value in list(row[0])]


def _boundary_inputs(kind: str) -> list[list[Any]]:
    """Frozen structural boundary inputs for a kind."""
    if kind == "graph":
        return [list(pair) for pair in _PAIR_BOUNDARY]
    return [list(items) for items in _LIST_BOUNDARY]


def _sanitize_verdict(
        verdict: Mapping[str, Any]) -> dict[str, Any]:
    """Allowlisted diagnostics; row values never leave here."""
    lite: dict[str, Any] = {}
    for key in _DIAG_KEYS:
        lite[key] = verdict.get(key)
    lite["detail"] = str(lite["detail"])[:500]
    lite["taxonomy"] = [str(code)
                        for code in lite["taxonomy"] or []]
    compiler = verdict.get("compiler", {})
    if isinstance(compiler, Mapping):
        lite["compiler"] = {
            "sha256": compiler.get("sha256"),
            "guard": compiler.get("guard")}
    else:
        lite["compiler"] = {"sha256": None, "guard": None}
    counter = verdict.get("counterexamples", [])
    lite["counterexamples"] = []
    lite["counterexample_count"] = len(counter) if isinstance(
        counter, list) else 0
    return lite


class RepairLoop:
    """Drive proposer plus repair-facing judge for one task."""

    def __init__(self, adapter: ProposerAdapter, judge: Any,
                 *, visible: Mapping[str, list[list[Any]]],
                 query: Mapping[str, list[list[Any]]],
                 fragments: Mapping[str, list[Any]] | None
                 = None,
                 max_rounds: int = R_MAX,
                 query_cap: int = QUERY_REVEAL_CAP,
                 token_cap: int = TOKEN_CAP,
                 cpu_cap: float = CPU_CAP_S,
                 ledger: CostLedger | None = None) -> None:
        """Bind adapter, judge, row pools and frozen caps."""
        if max_rounds < 0 or query_cap < 0:
            raise ValueError("round and query caps must hold")
        if token_cap < 0 or cpu_cap < 0:
            raise ValueError("token and CPU caps must hold")
        self._adapter = adapter
        self._judge = judge
        self._visible = {key: [list(row) for row in rows]
                         for key, rows in visible.items()}
        self._query = {key: [list(row) for row in rows]
                       for key, rows in query.items()}
        self._fragments = {key: list(rows)
                           for key, rows in
                           (fragments or {}).items()}
        self._max_rounds = max_rounds
        self._query_cap = query_cap
        self._token_cap = token_cap
        self._cpu_cap = cpu_cap
        self._ledger = ledger if ledger is not None \
            else CostLedger()

    @property
    def ledger(self) -> CostLedger:
        """Cost ledger holding per-round proposal charges."""
        return self._ledger

    def _failing_batch(
            self, kind: str, first: list[Any] | None,
            pool: list[list[Any]]) -> list[list[Any]]:
        """First failing row plus matching boundary rows."""
        batch: list[list[Any]] = []
        seen: set[str] = set()
        if first is not None:
            batch.append(list(first))
            seen.add(row_key(first))
        for want in _boundary_inputs(kind):
            if len(batch) >= 1 + BOUNDARY_BATCH_N:
                break
            for row in pool:
                if _input_of(kind, row) == want and \
                        row_key(row) not in seen:
                    batch.append(list(row))
                    seen.add(row_key(row))
                    break
        return batch

    def _build_prompt(
            self, task_id: str, kind: str,
            visible: list[list[Any]],
            frags: list[Any], text: str, digest: str,
            lite: dict[str, Any],
            failing: list[list[Any]],
            revealed: list[list[Any]]) -> str:
        """Versioned repair prompt JSON with a hashed template."""
        body = {"template_version": PROMPT_TEMPLATE_VERSION,
                "template_sha256": template_sha256(),
                "instruction": REPAIR_TEMPLATE,
                "task_id": task_id, "kind": kind,
                "visible_rows": visible, "fragments": frags,
                "previous_candidate": text,
                "previous_candidate_sha256": digest,
                "diagnostics": lite,
                "failing_batch": failing,
                "revealed_query_rows": revealed,
                "revealed_count": len(revealed),
                "query_cap": self._query_cap}
        return json.dumps(body, sort_keys=True,
                          separators=(",", ":"))

    def _reveal(self, task_id: str,
                visible_keys: set[str],
                failures: list[list[Any]],
                revealed: list[list[Any]],
                revealed_keys: set[str]) -> None:
        """Log fresh query failures up to the per-task cap."""
        pool_keys = {row_key(row)
                     for row in self._query[task_id]}
        room = self._query_cap - len(revealed)
        for row in failures:
            if room <= 0:
                break
            key = row_key(row)
            if key in visible_keys or key in revealed_keys:
                continue
            if key not in pool_keys:
                continue
            revealed.append(list(row))
            revealed_keys.add(key)
            room -= 1

    def repair(self, task_id: str, kind: str,
               candidate_fn: CandidateFn, *,
               params: Mapping[str, Any] | None = None
               ) -> dict[str, Any]:
        """Run rounds 0..max and return the stored artifact."""
        if kind not in KINDS:
            raise ValueError(f"unknown kind: {kind!r}")
        try:
            visible = [list(row)
                       for row in self._visible[task_id]]
        except KeyError:
            raise ValueError(
                f"unknown task: {task_id}") from None
        frags = list(self._fragments.get(task_id, []))
        extra = dict(params) if params else {}

        rounds: list[dict[str, Any]] = []
        seen: set[str] = set()
        revealed: list[list[Any]] = []
        revealed_keys: set[str] = set()
        prompt: str | None = None
        failing: list[list[Any]] = []
        status = "round_cap"
        accepted_round: int | None = None
        prompt_total = 0
        completion_total = 0
        cpu_total = 0.0
        text, usage, cpu = candidate_fn(0, None, [])

        for number in range(self._max_rounds + 1):
            digest = hashlib.sha256(
                text.encode("utf-8")).hexdigest()
            if digest in seen:
                entry = {"round": number,
                         "candidate_text": text,
                         "candidate_sha256": digest,
                         "spec": None, "proposal": None,
                         "prompt": prompt, "verdict": None,
                         "revealed_count": len(revealed),
                         "note": "loop: repeated hash"}
                rounds.append(entry)
                status = "loop"
                break
            spec = {"task_id": task_id, "kind": kind,
                    "visible_rows": visible, "fragments": frags,
                    "counterexamples": failing,
                    "params": extra}
            proposal = self._adapter.propose(
                spec, text,
                token_usage={str(k): int(v)
                             for k, v in dict(usage).items()},
                cpu=float(cpu))
            tokens = proposal["token_usage"]
            prompt_total += int(tokens.get("prompt", 0))
            completion_total += int(tokens.get(
                "completion", 0))
            cpu_total += float(proposal["cpu"])
            self._ledger.record("candidate_evaluation",
                                float(proposal["cpu"]), 0.0)
            verdict, _ = self._judge.judge_visible(
                task_id, kind, text, proposal=proposal)
            lite = _sanitize_verdict(verdict)
            entry = {"round": number,
                     "candidate_text": text,
                     "candidate_sha256": digest,
                     "spec": spec, "proposal": proposal,
                     "prompt": prompt, "verdict": lite,
                     "revealed_count": len(revealed),
                     "note": ""}
            rounds.append(entry)
            seen.add(digest)
            if lite["accepted"]:
                status = "accepted"
                accepted_round = number
                break
            if number >= self._max_rounds:
                status = "round_cap"
                break
            if not _is_value_miss(lite):
                status = "not_near_miss"
                break
            if prompt_total + completion_total > \
                    self._token_cap or \
                    cpu_total > self._cpu_cap:
                status = "cost_cap"
                break
            vis_fails = self._judge.counterexamples(
                task_id, kind, text, scope="visible")
            if vis_fails:
                failing_rows = [list(row)
                                for row in vis_fails]
            else:
                query_fails = self._judge.counterexamples(
                    task_id, kind, text, scope="query")
                self._reveal(task_id,
                             {row_key(row) for row in visible},
                             [list(row) for row in query_fails],
                             revealed, revealed_keys)
                failing_rows = [list(row)
                                for row in query_fails]
            first = failing_rows[0] if failing_rows else (
                visible[0] if visible else None)
            failing = self._failing_batch(
                kind, first, visible + revealed)
            prompt = self._build_prompt(
                task_id, kind, visible, frags, text,
                digest, lite, failing, revealed)
            text, usage, cpu = candidate_fn(
                number + 1, prompt, rounds)

        accepted = status == "accepted"
        return {
            "task_id": task_id, "kind": kind,
            "status": status, "accepted": accepted,
            "accepted_round": accepted_round,
            "single_shot_accepted": bool(
                rounds and rounds[0]["verdict"] is not None
                and rounds[0]["verdict"]["accepted"]),
            "single_shot": {
                "candidate_sha256": rounds[0][
                    "candidate_sha256"] if rounds else None,
                "accepted": bool(
                    rounds and
                    rounds[0]["verdict"] is not None
                    and rounds[0]["verdict"]["accepted"])},
            "rounds": rounds,
            "revealed_query_rows": revealed,
            "revealed_count": len(revealed),
            "totals": {"prompt_tokens": prompt_total,
                       "completion_tokens": completion_total,
                       "cpu_seconds": cpu_total},
            "cost_ledger": self._ledger.to_dict(),
            "template_version": PROMPT_TEMPLATE_VERSION,
            "template_sha256": template_sha256()}
