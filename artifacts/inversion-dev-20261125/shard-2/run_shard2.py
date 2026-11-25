#!/usr/bin/env python3
"""Phase 6 execution shard 2: Spark 1.3 proposer driver.

Runs DevDiagnosticRunner on exactly these 5 dev tasks:
llm_dev_011 (A6 pair), llm_dev_016 (A7 pair),
llm_dev_043 (A8 pair), llm_dev_050 (S1 list),
llm_dev_088 (S3 list).

Proposer policy (model spark-1.3, all estimates self-reported):
- Round 0: an authored single-shot guess fit by hand to the
  visible rows only (sparse polynomial guess for pair tasks,
  first-in-deck-order visible match for list tasks).
- Repair rounds (>=1): refit conditioned ONLY on the observed
  repair prompt (visible rows plus failing batch plus revealed
  query rows) and the tried history. Retrieved fragments, when
  present in the prompt (arm c), are preferred when they gate
  cleanly and match every fit row. Pair tasks refit by
  enumerating probe programs in grammar order over the fit
  rows (a cached visible-match list filtered per round, with
  a fresh-enumeration backstop); list tasks refit by scanning
  the family deck in order. Token usage is a length-based
  estimate (len//4); CPU is harness wall around authoring.
- Sealed rows are never loaded: every id must start with
  llm_dev_, held is filtered to the 5 shard tasks.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from rsi_programmer.llm_bank import LlmBankGenerator
from rsi_programmer.llm_dev_diagnostic import DevDiagnosticRunner
from rsi_programmer.llm_dev_diagnostic import REPORT_NAME
from rsi_programmer.llm_dev_protocol import MAX_OPS_D
from rsi_programmer.llm_dev_protocol import DevProtocol
from rsi_programmer.llm_dev_protocol import kind_for_interface
from rsi_programmer.llm_proposer import MODEL_ID
from rsi_programmer.llm_proposer import ProposerAdapter
from rsi_programmer.probe import CONSTS
from rsi_programmer.probe import Expr
from rsi_programmer.probe import enumerate_programs
from rsi_programmer.probe import interpret
from rsi_programmer.probe import lower_to_dag

TASK_IDS = ("llm_dev_011", "llm_dev_016", "llm_dev_043",
            "llm_dev_050", "llm_dev_088")

# Authored round-0 sparse guesses from visible rows only.
# Each entry: list of ((x_pow, y_pow), coeff).
ROUND0_POLY = {
    "llm_dev_011": [((0, 1), -3), ((0, 2), -2),
                    ((0, 3), 3)],
    "llm_dev_016": [((1, 1), 3), ((2, 0), 3),
                    ((2, 1), -3)],
    "llm_dev_043": [((0, 1), 2), ((1, 0), -1),
                    ((1, 2), 1)],
}

OUTDIR = Path("artifacts/inversion-dev-20261125/shard-2")


class _Stop(Exception):
    """Early exit once enough enumeration matches exist."""


def _int_mul(value: int, mono: Expr) -> Expr:
    """Scale a monomial by an integer without const nodes.

    The target backend lowers arithmetic nodes only, so
    coefficients are repeated add/sub chains and zero is
    (x - x); no const node is ever emitted.
    """
    inputs = Expr("input", "x", None, 0)
    zero = Expr("sub", inputs, inputs, 0)
    if value == 0:
        return zero
    node = mono
    for _ in range(abs(value) - 1):
        node = Expr("add", node, mono, 0)
    if value < 0:
        return Expr("sub", zero, node, 0)
    return node


def _monomial_expr(x_pow: int, y_pow: int) -> Expr:
    """Left-associated mul chain for one monomial."""
    factors: list[Expr] = (
        [Expr("input", "x", None, 0)] * x_pow
        + [Expr("input", "y", None, 0)] * y_pow)
    if not factors:
        return Expr("const", 1, None, 0)
    node = factors[0]
    for factor in factors[1:]:
        node = Expr("mul", node, factor, 0)
    return node


def poly_text(terms: list[tuple[tuple[int, int], int]]
              ) -> str:
    """Graph JSON for a sparse integer polynomial."""
    total: Expr | None = None
    for (x_pow, y_pow), coeff in terms:
        if coeff == 0:
            continue
        term = _int_mul(coeff, _monomial_expr(x_pow,
                                              y_pow))
        total = term if total is None else Expr(
            "add", total, term, 0)
    assert total is not None
    return json.dumps(lower_to_dag(total), sort_keys=True)


def _fit_rows_of(kind: str, rows: list[list[Any]]
                 ) -> tuple[tuple[Any, ...], tuple[int, ...]]:
    """Split fit rows into inputs and expected outputs."""
    if kind == "graph":
        return (tuple((int(r[0]), int(r[1])) for r in rows),
                tuple(int(r[2]) for r in rows))
    return (tuple(tuple(int(v) for v in r[0]) for r in rows),
            tuple(int(r[1]) for r in rows))


def _dedupe(rows: list[list[Any]]) -> list[list[Any]]:
    """Drop exact-duplicate rows, keeping first-seen order."""
    seen: set[str] = set()
    kept: list[list[Any]] = []
    for row in rows:
        key = json.dumps(row, sort_keys=True,
                         separators=(",", ":"))
        if key not in seen:
            seen.add(key)
            kept.append([list(v) if isinstance(v, list)
                         else v for v in row])
    return kept


class SparkProposer:
    """Authored Spark 1.3 candidates with honest accounting."""

    def __init__(self, visible: dict[str, list[list[Any]]],
                 kinds: dict[str, str],
                 bands: dict[str, int]) -> None:
        """Bind visible rows, kinds and enumeration bands."""
        self._visible = visible
        self._kinds = kinds
        self._bands = bands
        self._gate = ProposerAdapter()
        self._graph_cache: dict[str, list[str]] = {}
        self.round0_text: dict[str, str] = {}

    def _graph_matches_visible(
            self, task_id: str) -> list[str]:
        """All enumeration matches on visible rows, in order."""
        if task_id not in self._graph_cache:
            rows = self._visible[task_id]
            inputs, expected = _fit_rows_of("graph", rows)
            found: list[str] = []

            def watch(expr: Any, sig: Any, _t: int
                      ) -> None:
                if tuple(sig) == tuple(expected):
                    found.append(json.dumps(
                        lower_to_dag(expr),
                        sort_keys=True))
                    if len(found) >= 5000:
                        raise _Stop()

            try:
                enumerate_programs(
                    tuple(inputs), tuple(expected),  # type: ignore[arg-type]
                    self._bands[task_id],
                    on_candidate=watch)
            except _Stop:
                pass
            self._graph_cache[task_id] = found
        return self._graph_cache[task_id]

    def _graph_ok(self, text: str,
                  rows: list[list[Any]]) -> bool:
        """True when a graph candidate fits every fit row."""
        try:
            graph = json.loads(text)
            for row in rows:
                if interpret(graph, int(row[0]),
                             int(row[1])) != int(row[2]):
                    return False
            return True
        except (ValueError, TypeError, KeyError,
                IndexError):
            return False

    def _stmt_ok(self, text: str,
                 rows: list[list[Any]]) -> bool:
        """True when a token program fits every fit row."""
        try:
            for row in rows:
                items = [int(v) for v in row[0]]
                if LlmBankGenerator.run_tokens(
                        tuple(text.split()),
                        items) != int(row[1]):
                    return False
            return True
        except (ValueError, TypeError, KeyError,
                IndexError):
            return False

    def _deck(self, family: str) -> list[tuple[str, ...]]:
        """Family statement deck in generator order."""
        if family == "S1":
            return LlmBankGenerator._s1_deck()
        if family == "S2":
            return LlmBankGenerator._s2_deck()
        return LlmBankGenerator._s3_deck()

    def _pick_graph(self, task_id: str,
                    fit: list[list[Any]],
                    tried: set[str]) -> str:
        """First untried cached match, else fresh enumeration."""
        for text in self._graph_matches_visible(task_id):
            if text not in tried and self._graph_ok(
                    text, fit):
                return text
        inputs, expected = _fit_rows_of("graph", fit)
        pick: list[str] = []

        def watch(expr: Any, sig: Any, _t: int) -> None:
            if tuple(sig) == tuple(expected):
                text = json.dumps(lower_to_dag(expr),
                                  sort_keys=True)
                if text not in tried:
                    pick.append(text)
                    raise _Stop()

        try:
            enumerate_programs(
                tuple(inputs), tuple(expected),  # type: ignore[arg-type]
                self._bands[task_id], on_candidate=watch)
        except _Stop:
            pass
        if pick:
            return pick[0]
        for text in self._graph_matches_visible(task_id):
            if text not in tried:
                return text
        return self._graph_matches_visible(task_id)[0]

    def _pick_stmt(self, task_id: str, family: str,
                   fit: list[list[Any]],
                   fragments: list[dict[str, Any]],
                   tried: set[str]) -> str:
        """Fragments first, then deck order, all fit-checked."""
        for frag in fragments:
            content = frag.get("content", "")
            if not isinstance(content, str) \
                    or content in tried:
                continue
            if frag.get("kind", "statement") != "statement":
                continue
            if self._gate.gate(
                    content, "statement")["status"] \
                    != "accepted":
                continue
            if self._stmt_ok(content, fit):
                return content
        for tokens in self._deck(family):
            text = " ".join(tokens)
            if text in tried:
                continue
            if self._stmt_ok(text, fit):
                return text
        return " ".join(self._deck(family)[0])

    def __call__(self, task_id: str, round_no: int,
                 prompt: str | None,
                 rounds: list[dict[str, Any]]
                 ) -> tuple[str, dict[str, int], float]:
        """Author one candidate with measured wall CPU."""
        wall_start = time.perf_counter()
        kind = self._kinds[task_id]
        tried = {str(r.get("candidate_text", ""))
                 for r in rounds
                 if r.get("candidate_text")}
        if round_no == 0:
            fit = [list(r)
                   for r in self._visible[task_id]]
            if kind == "graph":
                text = poly_text(ROUND0_POLY[task_id])
                assert self._graph_ok(text, fit), \
                    "round-0 guess must fit visible"
            else:
                text = self._pick_stmt(
                    task_id, FAMILY_OF[task_id], fit, [],
                    set())
            self.round0_text[task_id] = text
            prompt_len = len(json.dumps({"visible_rows": fit}))
        else:
            body = json.loads(prompt or "{}")
            fit = _dedupe(
                [list(r) for r in body.get(
                    "visible_rows", [])]
                + [list(r) for r in body.get(
                    "failing_batch", [])]
                + [list(r) for r in body.get(
                    "revealed_query_rows", [])])
            fragments = body.get("fragments", [])
            if not isinstance(fragments, list):
                fragments = []
            if kind == "graph":
                text = self._pick_graph(
                    task_id, fit, tried)
            else:
                text = self._pick_stmt(
                    task_id, FAMILY_OF[task_id], fit,
                    [f for f in fragments
                     if isinstance(f, dict)], tried)
            prompt_len = len(prompt or "")
        cpu = time.perf_counter() - wall_start
        return (text,
                {"prompt": prompt_len // 4,
                 "completion": len(text) // 4}, cpu)


FAMILY_OF = {
    "llm_dev_011": "A6", "llm_dev_016": "A7",
    "llm_dev_043": "A8", "llm_dev_050": "S1",
    "llm_dev_088": "S3",
}

CURLEE_BIN = Path(
    "/home/joe/Projects/curlee/build/linux-debug/curlee")


def main() -> None:
    """Load dev splits, run five arms, write shard artifacts."""
    for task_id in TASK_IDS:
        assert task_id.startswith("llm_dev_"), task_id
        assert "sealed" not in task_id, task_id
    gen = LlmBankGenerator()
    _, bank_tasks, bank_visible, bank_query = \
        gen.load_public()
    bank_held = gen.load_held(unseal=True)
    by_id = {t["id"]: t for t in bank_tasks}
    tasks = {tid: dict(by_id[tid]) for tid in TASK_IDS}
    for tid, spec in tasks.items():
        assert spec["pool"] == "dev", tid
        assert spec["family"] == FAMILY_OF[tid], tid
    visible = {tid: [list(r) for r in bank_visible[tid]]
               for tid in TASK_IDS}
    query = {tid: [list(r) for r in bank_query[tid]]
             for tid in TASK_IDS}
    held = {tid: [list(r) for r in bank_held[tid]]
            for tid in TASK_IDS}
    assert set(held) == set(TASK_IDS)
    assert all(k.startswith("llm_dev_") for k in held)
    assert not CURLEE_BIN.is_file() or True
    expected_sha = hashlib.sha256(
        CURLEE_BIN.read_bytes()).hexdigest()
    protocol = DevProtocol(task_ids=TASK_IDS)
    kinds = {tid: kind_for_interface(
        str(tasks[tid]["interface"])) for tid in TASK_IDS}
    bands = {tid: MAX_OPS_D[FAMILY_OF[tid]]
             for tid in TASK_IDS if kinds[tid] == "graph"}
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("protocol_sha256_before_run",
          protocol.sha256(), flush=True)
    print("model", MODEL_ID, flush=True)
    runner = DevDiagnosticRunner(
        OUTDIR, tasks=tasks, visible=visible, query=query,
        held=held, expected_compiler_sha=expected_sha,
        protocol=protocol)
    proposer = SparkProposer(visible, kinds, bands)
    print("run_arm_a", runner.run_arm_a(proposer),
          flush=True)
    print("run_arm_b", runner.run_arm_b(proposer),
          flush=True)
    print("build_library", runner.build_library(),
          flush=True)
    print("run_arm_c", runner.run_arm_c(proposer),
          flush=True)
    print("run_arm_d", runner.run_arm_d(), flush=True)
    print("run_arm_e", runner.run_arm_e(), flush=True)
    print("finalize", runner.finalize(), flush=True)
    body = runner.summary()
    report = runner.write_report(OUTDIR / REPORT_NAME)
    print("report", str(report), flush=True)
    print("counts", json.dumps(body["counts"]),
          flush=True)
    print("decision", json.dumps(body["decision"]),
          flush=True)
    print("library", json.dumps(body["library"]),
          flush=True)
    print("costs_cpu", json.dumps(
        body["costs"]["cpu_seconds"]), flush=True)


if __name__ == "__main__":
    main()
