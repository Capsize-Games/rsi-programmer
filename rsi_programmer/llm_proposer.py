"""Spark 1.3 proposer adapter and untrusted-candidate gate.

The supervised-worker harness calls :class:`ProposerAdapter` to
record each proposal with provenance; the adapter performs no
network calls and executes no candidate. Every candidate passes
the structural gate (parse plus ``probe.validate_graph`` or
``statement_syntax.parse``, then the sandbox prescreen caps)
before any execution. The judge decides; proposer scores alone
prove nothing.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from rsi_programmer.candidate_store import CandidateStore
from rsi_programmer.probe import interpret, validate_graph
from rsi_programmer.statement_syntax import MAX_TOKENS, parse

__all__ = ["CandidateStore", "ProposerAdapter", "DENIED_TOKENS",
           "KINDS", "MAX_BYTES", "MAX_NODES", "MODEL_ID",
           "MODEL_VERSION"]

MODEL_ID = "spark-1.3"
MODEL_VERSION = "Spark 1.3 supervised background workers"

MAX_NODES = 64
MAX_BYTES = 65536

# Mirror of rsi_programmer.sandbox._DENIED; parity is asserted
# by tests.test_llm_proposer so the gate never drifts softer.
DENIED_TOKENS = (
    "oracle", "held_out", "heldout", "ledger", "promot",
    "verifier", "validate", "socket", "urllib", "http",
    "subprocess", "shell", "exec", "eval(", "open(",
    "__", "/proc", "connect",
)

KINDS = ("graph", "statement")

_SPEC_KEYS = ("task_id", "kind", "visible_rows", "fragments",
              "counterexamples", "params")
_FORBIDDEN_KEYS = ("held", "held_rows", "held_out", "heldout",
                   "query", "query_rows", "oracle", "sealed")


class ProposerAdapter:
    """Boundary plus provenance recorder for Spark 1.3 workers."""

    def __init__(self, store: CandidateStore | None = None,
                 params: Mapping[str, Any] | None = None
                 ) -> None:
        self._store = store if store is not None \
            else CandidateStore()
        self._params = dict(params) if params else {}

    @property
    def store(self) -> CandidateStore:
        """The append-only store backing this adapter."""
        return self._store

    @staticmethod
    def _refuse_held(spec: Mapping[str, Any]) -> None:
        for key in spec:
            if str(key).lower() in _FORBIDDEN_KEYS:
                raise ValueError(
                    "held/query data refused: %r" % (key,))
        for key in spec:
            if key not in _SPEC_KEYS:
                raise ValueError("unknown spec field: %r"
                                 % (key,))

    @staticmethod
    def _check_fragments(fragments: Any) -> None:
        if not isinstance(fragments, list):
            raise ValueError("fragments must be a list")
        for fragment in fragments:
            if not isinstance(fragment, Mapping):
                raise ValueError("fragment must be a mapping")
            provenance = fragment.get("provenance")
            if not isinstance(provenance, Mapping) \
                    or not provenance:
                raise ValueError(
                    "only qualified fragments with "
                    "provenance are accepted")

    def build_prompt(self, spec: Mapping[str, Any]) -> str:
        """Canonical prompt JSON over visible rows and extras."""
        if not isinstance(spec, Mapping):
            raise ValueError("spec must be a mapping")
        self._refuse_held(spec)
        kind = spec.get("kind", "graph")
        if kind not in KINDS:
            raise ValueError("unknown candidate kind: %r"
                             % (kind,))
        rows = spec.get("visible_rows", [])
        if not isinstance(rows, list) or not rows:
            raise ValueError("visible_rows must be a "
                             "non-empty list")
        fragments = list(spec.get("fragments", []))
        self._check_fragments(fragments)
        counterexamples = spec.get("counterexamples", [])
        if not isinstance(counterexamples, list):
            raise ValueError("counterexamples must be a list")
        body = {"task_id": spec.get("task_id", ""),
                "kind": kind, "visible_rows": rows,
                "fragments": fragments,
                "counterexamples": counterexamples}
        return json.dumps(body, sort_keys=True,
                          separators=(",", ":"))

    def propose(self, spec: Mapping[str, Any],
                candidate_text: str, *,
                token_usage: Mapping[str, int] | None = None,
                cpu: float = 0.0) -> dict[str, Any]:
        """Record one worker-produced candidate with provenance.

        No network call is made: the supervised-worker harness
        supplies ``candidate_text``; this method is the boundary
        plus provenance recorder.
        """
        if not isinstance(candidate_text, str) \
                or not candidate_text:
            raise ValueError("candidate_text must be "
                             "non-empty text")
        prompt = self.build_prompt(spec)
        usage = {"prompt": 0, "completion": 0}
        if token_usage is not None:
            for name, value in dict(token_usage).items():
                usage[str(name)] = int(value)
        params = dict(self._params)
        extra = spec.get("params", {})
        if not isinstance(extra, Mapping):
            raise ValueError("params must be a mapping")
        params.update(dict(extra))
        proposal = {
            "model": MODEL_ID,
            "model_version": MODEL_VERSION,
            "prompt_sha256": hashlib.sha256(
                prompt.encode("utf-8")).hexdigest(),
            "params": params,
            "candidate_text": candidate_text,
            "token_usage": usage,
            "cpu": float(cpu),
        }
        self._store.append(proposal)
        return dict(proposal)

    def gate(self, candidate_text: str, kind: str
             ) -> dict[str, Any]:
        """Structural gate; performs no candidate execution."""

        def rejected(violations: list[str]
                     ) -> dict[str, Any]:
            return {"status": "rejected", "kind": kind,
                    "violations": violations}

        if kind not in KINDS:
            return rejected(["structural: unknown kind %r"
                             % (kind,)])
        if not isinstance(candidate_text, str) \
                or not candidate_text:
            return rejected(["structural: empty candidate"])
        if len(candidate_text.encode("utf-8")) > MAX_BYTES:
            return rejected(["budget: payload bytes over "
                             "cap %d" % MAX_BYTES])
        lowered = candidate_text.lower()
        hits = ["denied-reference: " + token
                for token in DENIED_TOKENS if token in lowered]
        if hits:
            return rejected(hits)
        if kind == "statement":
            return self._gate_statement(candidate_text)
        return self._gate_graph(candidate_text)

    @staticmethod
    def _gate_statement(candidate_text: str
                        ) -> dict[str, Any]:
        tokens = tuple(candidate_text.split())
        if len(tokens) > MAX_TOKENS:
            return {"status": "rejected",
                    "kind": "statement",
                    "violations": [
                        "budget: tokens %d over cap %d"
                        % (len(tokens), MAX_TOKENS)]}
        try:
            parse(tokens)
        except ValueError as exc:
            return {"status": "rejected",
                    "kind": "statement",
                    "violations": ["structural: %s" % exc]}
        return {"status": "accepted", "kind": "statement",
                "violations": []}

    @staticmethod
    def _gate_graph(candidate_text: str) -> dict[str, Any]:
        try:
            graph = json.loads(candidate_text)
        except ValueError as exc:
            return {"status": "rejected", "kind": "graph",
                    "violations": ["structural: %s" % exc]}
        if not isinstance(graph, dict):
            return {"status": "rejected", "kind": "graph",
                    "violations": ["structural: graph must "
                                   "be an object"]}
        nodes = graph.get("nodes", [])
        if isinstance(nodes, list) \
                and len(nodes) > MAX_NODES:
            return {"status": "rejected", "kind": "graph",
                    "violations": ["budget: nodes %d over "
                                   "cap %d"
                                   % (len(nodes), MAX_NODES)]}
        try:
            validate_graph(graph)
        except (ValueError, TypeError, KeyError,
                IndexError) as exc:
            return {"status": "rejected", "kind": "graph",
                    "violations": ["structural: %s" % exc]}
        return {"status": "accepted", "kind": "graph",
                "violations": []}

    @staticmethod
    def graph_signature(graph: Mapping[str, Any],
                        rows: Sequence[Sequence[int]]
                        ) -> tuple[int, ...]:
        """Per-row outputs; the ``sig`` half of (op, sig)."""
        if not isinstance(graph, dict):
            raise ValueError("graph must be an object")
        validate_graph(graph)
        pairs = [(int(row[0]), int(row[1])) for row in rows]
        return tuple(interpret(graph, x, y) for x, y in pairs)

    @staticmethod
    def graph_root_op(graph: Mapping[str, Any]) -> str:
        """Root op; the ``op`` half of (op, sig) dedup keys."""
        if not isinstance(graph, dict):
            raise ValueError("graph must be an object")
        output = graph.get("output")
        if isinstance(output, str):
            return "input"
        nodes = graph.get("nodes", [])
        if not isinstance(output, int) or not isinstance(
                nodes, list) or not 0 <= output < len(nodes):
            raise ValueError("invalid output reference")
        op = nodes[output].get("op")
        if not isinstance(op, str):
            raise ValueError("operation must be a string")
        return op

    @staticmethod
    def dedup_key(op: str, sig: Sequence[int]
                  ) -> tuple[str, tuple[int, ...]]:
        """Mirror probe.enumerate_programs (op, sig) dedup."""
        return (op, tuple(sig))
