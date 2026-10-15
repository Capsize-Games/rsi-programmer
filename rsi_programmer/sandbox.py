#!/usr/bin/env python3
"""Disposable unprivileged runner for RSI DAG candidates.

Candidate programs are Expr/dict DAGs validated by the supervisor-side
allowlisted interpreter from rsi_programmer.probe; candidate
emitted source is never executed. The supervisor spawns one disposable
worker per run with wall/CPU caps enforced outside the candidate, a
scrubbed environment, an isolated cwd holding no oracle, ledger, or
promotion files, and recomputes every verdict supervisor-side.
"""
from __future__ import annotations

import hashlib
import json
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from rsi_programmer.probe import interpret
from rsi_programmer.probe import validate_graph

_DENIED = ("oracle", "held_out", "heldout", "ledger", "promot",
           "verifier", "validate", "socket", "urllib", "http",
           "subprocess", "shell", "exec", "eval(", "open(",
           "__", "/proc", "connect")

_CHILD = (
    "import json,sys;"
    "sys.path.insert(0,sys.argv[4]);"
    "from rsi_programmer.probe import interpret;"
    "g=json.load(open(sys.argv[1]));"
    "r=json.load(open(sys.argv[2]));"
    "json.dump([interpret(g,x,y) for x,y in r],open(sys.argv[3],'w'))"
)


class SandboxRunner:
    """Run DAG candidates in a capped disposable subprocess."""

    def __init__(self, workdir: Path, *, wall_s: float = 5.0,
                 cpu_s: int = 5, max_nodes: int = 64,
                 max_bytes: int = 65536) -> None:
        if wall_s <= 0 or cpu_s < 1 or max_nodes < 1 or max_bytes < 1:
            raise ValueError("sandbox budgets must be positive")
        self._workdir = workdir
        self._workdir.mkdir(parents=True, exist_ok=True)
        self._wall_s = wall_s
        self._cpu_s = cpu_s
        self._max_nodes = max_nodes
        self._max_bytes = max_bytes

    @staticmethod
    def lineage_hash(graph: dict[str, Any]) -> str:
        """Content hash binding a candidate to its exact DAG bytes."""
        canon = json.dumps(graph, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canon.encode()).hexdigest()

    def check_submission(self, graph: dict[str, Any],
                         claim: dict[str, Any] | None = None
                         ) -> list[str]:
        """List policy violations without executing anything."""
        payload = json.dumps({"graph": graph, "claim": claim},
                             sort_keys=True, default=str)
        hits = ["denied-reference: " + tok for tok in _DENIED
                if tok in payload.lower()]
        nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
        if isinstance(nodes, list) and len(nodes) > self._max_nodes:
            hits.append("budget: nodes %d over cap %d"
                        % (len(nodes), self._max_nodes))
        if len(payload) > self._max_bytes:
            hits.append("budget: payload bytes over cap")
        return hits

    def verify_ancestry(self, parent_graph: dict[str, Any],
                        claimed_hash: str, child_ref: str) -> bool:
        """True only if stored parent bytes match both hashes."""
        actual = self.lineage_hash(parent_graph)
        return actual == claimed_hash and actual == child_ref

    def _wait(self, argv: list[str], cwd: Path) -> dict[str, Any]:
        limit = self._cpu_s

        def _cap() -> None:
            resource.setrlimit(resource.RLIMIT_CPU, (limit, limit))
        env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8",
               "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            proc = subprocess.run(argv, cwd=cwd, env=env,
                                  capture_output=True, text=True,
                                  timeout=self._wall_s, preexec_fn=_cap)
        except subprocess.TimeoutExpired as exc:
            return {"status": "timeout", "code": None,
                    "err": str(exc)[:200]}
        if proc.returncode != 0:
            return {"status": "error", "code": proc.returncode,
                    "err": proc.stderr[-200:]}
        return {"status": "ok", "code": 0, "err": ""}

    def _spawn(self, argv: list[str], cwd: Path) -> dict[str, Any]:
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        start = time.perf_counter()
        outcome = self._wait(argv, cwd)
        wall = time.perf_counter() - start
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu = (after.ru_utime - before.ru_utime
               + after.ru_stime - before.ru_stime)
        return {"status": outcome["status"], "wall_s": wall,
                "cpu_s": cpu, "code": outcome["code"],
                "err": outcome["err"]}

    def run_command(self, argv: list[str]) -> dict[str, Any]:
        """Run argv under the same wall/CPU caps; timeout proof hook."""
        outcome = self._spawn(argv, self._workdir)
        return {"status": outcome["status"], "wall_s": outcome["wall_s"],
                "cpu_s": outcome["cpu_s"]}

    def _prepare_case(self, graph: dict[str, Any],
                      rows: list[list[int]]) -> Path:
        case = Path(tempfile.mkdtemp(dir=str(self._workdir)))
        (case / "graph.json").write_text(json.dumps(graph))
        (case / "rows.json").write_text(json.dumps(rows))
        return case

    def _launch(self, case: Path) -> dict[str, Any]:
        lib = str(Path(__file__).resolve().parent.parent)
        argv = [sys.executable, "-c", _CHILD, str(case / "graph.json"),
                str(case / "rows.json"), str(case / "outputs.json"),
                lib]
        return self._spawn(argv, case)

    def _prescreen(self, graph: dict[str, Any],
                   claim: dict[str, Any] | None) -> dict[str, Any] | None:
        pre = self.check_submission(graph, claim)
        if pre:
            return {"status": "rejected", "violations": pre,
                    "claim_ignored": True}
        try:
            validate_graph(graph)
        except ValueError as exc:
            return {"status": "rejected",
                    "violations": ["verifier: " + str(exc)],
                    "claim_ignored": True}
        return None

    def _judge(self, graph: dict[str, Any], rows: list[list[int]],
               case: Path, claim: dict[str, Any] | None,
               res: dict[str, Any]) -> dict[str, Any]:
        raw = json.loads((case / "outputs.json").read_text())
        supervised = [interpret(graph, x, y) for x, y in rows]
        base: dict[str, Any] = {"case": case.name,
                                "lineage": self.lineage_hash(graph),
                                "wall_s": res["wall_s"],
                                "cpu_s": res["cpu_s"],
                                "outputs": supervised,
                                "claim_ignored": True}
        if raw != supervised:
            return {**base, "status": "rejected",
                    "violations": ["execution-mismatch"]}
        claimed = claim.get("outputs") if isinstance(claim, dict) else None
        if claimed is not None and claimed != supervised:
            return {**base, "status": "rejected",
                    "violations": ["forged-completion"]}
        return {**base, "status": "accepted", "violations": []}

    def run_candidate(self, graph: dict[str, Any],
                      rows: list[list[int]],
                      claim: dict[str, Any] | None = None
                      ) -> dict[str, Any]:
        """Evaluate one DAG; the verdict is recomputed supervisor-side."""
        denied = self._prescreen(graph, claim)
        if denied is not None:
            return denied
        case = self._prepare_case(graph, rows)
        res = self._launch(case)
        if res["status"] != "ok":
            return {"status": res["status"], "case": case.name,
                    "lineage": self.lineage_hash(graph),
                    "wall_s": res["wall_s"], "cpu_s": res["cpu_s"],
                    "violations": ["execution: " + res["status"]]}
        return self._judge(graph, rows, case, claim, res)
