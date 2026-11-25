"""Dual-backend gated judge for Spark 1.3 bank candidates.

One entry point takes a bank task plus a gated candidate and
returns a structured verdict. A candidate is accepted only when
it is valid (structural gate plus a CURLEE check with the pinned
compiler) AND correct on every evaluated row in BOTH backends
(sandboxed Python plus a CURLEE run) within stated i64 bounds
AND, for arithmetic graphs, exactly matches the witness
polynomial identity. Timeouts are verdicts, never exceptions.

Split discipline: :meth:`LlmJudge.judge_visible` evaluates the
visible plus query rows and is the only stage repair-facing code
may reach, together with :meth:`LlmJudge.counterexamples` scoped
to ``visible`` or ``query``. :meth:`LlmJudge.judge_final`
evaluates the sealed held rows; the held accessor raises
:class:`SealedHeldError` whenever held data is unavailable, so
repair code constructed without held rows cannot reach it.

Verdict schema (all values JSON-safe)::

    {"task_id": str, "kind": "graph" | "statement",
     "stage": "visible" | "final", "accepted": bool,
     "valid": bool, "python_correct": bool | None,
     "curlee_correct": bool | None, "identity_ok": bool | None,
     "rows_checked": int, "mismatches": int,
     "first_mismatch": int | None, "taxonomy": [str],
     "detail": str, "signature": {"op": str, "sig": [int]} | None,
     "counterexamples": [[input, expected], ...],
     "compiler": {"path": str, "sha256": str, "guard": bool},
     "range_note": str,
     "costs": {phase: {"cpu_seconds": float,
                       "wall_seconds": float}},
     "tokens": {"prompt": int, "completion": int},
     "proposal_cpu": float}

``counterexamples`` carries row values only for the visible and
query splits; final-stage verdicts record counts and the first
mismatching index but never held row values. ``taxonomy`` uses a
fixed vocabulary: structural, denied_reference, budget,
curlee_lower, curlee_check, python_mismatch, python_error,
execution_mismatch, curlee_mismatch, curlee_error,
identity_mismatch, identity_error, int64_range, timeout.
``python_correct`` and ``curlee_correct`` stay ``None`` when an
earlier gate stops the pipeline before that backend runs, and
``identity_ok`` stays ``None`` for statement candidates whose
grammar has no polynomial identity to check.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rsi_programmer import statement_syntax
from rsi_programmer.cost_ledger import COST_DEFINITION
from rsi_programmer.cost_ledger import CostLedger
from rsi_programmer.llm_proposer import ProposerAdapter
from rsi_programmer.online_graph import graph_polynomial
from rsi_programmer.probe import validate_graph
from rsi_programmer.program_cost import CostClock
from rsi_programmer.sandbox import SandboxRunner
from rsi_programmer.statement_execution import execute
from rsi_programmer.statement_source import curlee_source
from rsi_programmer.target_source import emit_function

INT64_MIN = -(2 ** 63)
INT64_MAX = 2 ** 63 - 1
CURLEE_FUEL = "20000000"
CURLEE_TIMEOUT_S = 30.0
GUARD_SCRIPT = Path(
    "/home/joe/Projects/curlee/scripts/run_limited.py")
DEFAULT_COMPILER = Path(
    "/home/joe/Projects/curlee/build/linux-debug/curlee")
JUDGE_PHASES = ("proposer", "gate", "sandbox", "curlee_check",
                "curlee_run", "dedup")
RANGE_NOTE = (
    "Python ints are unbounded; CURLEE Ints are i64. Every "
    "expected and actual output must lie in "
    "[-2**63, 2**63 - 1]; bank rows stay within 2**62.")
EXCLUDED_COSTS = (
    "Historical donor-library creation outside this run",
    "LLM pretraining and proposal token generation outside "
    "the harness",
    "Sealed held handling outside judge_final",
    "Proposer CPU is reported per proposal; only gate and "
    "judge work is ledgered",
)
_RUN_RESULT = re.compile(r"curlee run: result (\d+)\b")

_STMT_CHILD = (
    "import json,sys;"
    "sys.path.insert(0,sys.argv[4]);"
    "from rsi_programmer.statement_execution import execute;"
    "tokens=json.load(open(sys.argv[1]));"
    "rows=json.load(open(sys.argv[2]));"
    "json.dump([execute(tuple(tokens),list(v)) for v in rows],"
    "open(sys.argv[3],'w'))"
)


class SealedHeldError(ValueError):
    """Raised when repair-facing code reaches for held rows."""


def _sha256_file(path: Path) -> str:
    """Hash a file's bytes with sha256 and return hex."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _zero_costs() -> dict[str, dict[str, float]]:
    """Fresh per-verdict CPU/wall slots for every judge phase."""
    return {phase: {"cpu_seconds": 0.0, "wall_seconds": 0.0}
            for phase in JUDGE_PHASES}


def _pair_harness(graph: Mapping[str, Any],
                  rows: list[list[int]]) -> str:
    """Lower one graph plus per-row asserts to CURLEE source."""
    lines = [emit_function(dict(graph), "curlee", "solve"),
             "fn main() -> Int {"]
    for x, y, want in rows:
        lines.append(f"    if (solve({x}, {y}) != {want}) {{")
        lines.append("        return 1;")
        lines.append("    }")
    lines += ["    return 0;", "}", ""]
    return "\n".join(lines)


def _statement_harness(tokens: tuple[str, ...],
                       rows: list[list[Any]]) -> str:
    """Lower one token program plus per-row asserts to CURLEE."""
    lines = [curlee_source(tokens),
             "fn main() -> Int {",
             "    let input: [Int; 256] = [0; 256];"]
    for items, want in rows:
        for index, value in enumerate(items):
            lines.append(f"    input[{index}] = {value};")
        lines.append(f"    if (process(&input, {len(items)})"
                     f" != {want}) {{")
        lines.append("        return 1;")
        lines.append("    }")
    lines += ["    return 0;", "}", ""]
    return "\n".join(lines)


def _public_verdict(verdict: dict[str, Any]) -> dict[str, Any]:
    """Copy a verdict for JSON artifacts without row values."""
    body = dict(verdict)
    counter = body.get("counterexamples", [])
    body["counterexamples"] = []
    body["counterexample_count"] = len(counter)
    return body


class LlmJudge:
    """Gate plus dual-backend verdicts with sealed held rows."""

    def __init__(self, workdir: Path, *,
                 tasks: Mapping[str, Mapping[str, Any]],
                 visible: Mapping[str, list[list[Any]]],
                 query: Mapping[str, list[list[Any]]],
                 held: Mapping[str, list[list[Any]]] | None,
                 expected_compiler_sha: str,
                 compiler: Path = DEFAULT_COMPILER,
                 wall_s: float = 5.0, cpu_s: int = 5,
                 curlee_timeout: float = CURLEE_TIMEOUT_S
                 ) -> None:
        """Bind row splits and pin the CURLEE compiler binary."""
        if wall_s <= 0 or curlee_timeout <= 0 or cpu_s < 1:
            raise ValueError("judge budgets must be positive")
        if not compiler.is_file():
            raise ValueError(f"CURLEE compiler missing: {compiler}")
        digest = _sha256_file(compiler)
        if digest != expected_compiler_sha:
            raise ValueError("CURLEE compiler hash mismatch: "
                             f"{digest}")
        self._workdir = workdir
        self._workdir.mkdir(parents=True, exist_ok=True)
        self._tasks = {key: dict(value)
                       for key, value in tasks.items()}
        self._visible = {key: [list(row) for row in rows]
                         for key, rows in visible.items()}
        self._query = {key: [list(row) for row in rows]
                       for key, rows in query.items()}
        self._held = ({key: [list(row) for row in rows]
                       for key, rows in held.items()}
                      if held is not None else None)
        self._compiler = compiler
        self._compiler_sha = digest
        self._guard = GUARD_SCRIPT.is_file()
        self._curlee_timeout = curlee_timeout
        self._gate = ProposerAdapter()
        self._runner = SandboxRunner(workdir / "sandbox",
                                     wall_s=wall_s,
                                     cpu_s=cpu_s)
        self.ledger = CostLedger()
        for excluded in EXCLUDED_COSTS:
            self.ledger.declare_excluded(excluded)

    @property
    def compiler_sha(self) -> str:
        """Pinned CURLEE compiler digest checked at construction."""
        return self._compiler_sha

    @property
    def compiler_path(self) -> Path:
        """Pinned CURLEE compiler binary path."""
        return self._compiler

    @property
    def uses_guard(self) -> bool:
        """True when compiler runs through the resource guard."""
        return self._guard

    def held_rows(self, task_id: str) -> list[list[Any]]:
        """Return sealed held rows; raises outside final use."""
        if self._held is None:
            raise SealedHeldError(
                "held rows are unavailable outside final")
        try:
            return [list(row) for row in self._held[task_id]]
        except KeyError:
            raise SealedHeldError(
                f"unknown task: {task_id}") from None

    def _task(self, task_id: str) -> dict[str, Any]:
        """Return one bank task record or raise a clear error."""
        try:
            return self._tasks[task_id]
        except KeyError:
            raise ValueError(
                f"unknown task: {task_id}") from None

    def _split_rows(self, task_id: str,
                    stage: str) -> list[list[Any]]:
        """Visible plus query rows, or sealed held rows for final."""
        if stage == "final":
            return self.held_rows(task_id)
        return ([list(row) for row in self._visible[task_id]]
                + [list(row) for row in self._query[task_id]])

    def _signature(self, kind: str, parsed: Any,
                   rows: list[list[Any]]
                   ) -> dict[str, Any] | None:
        """Visible-row (op, sig) key mirroring probe enumeration."""
        try:
            if kind == "graph":
                pairs = [[int(row[0]), int(row[1])]
                         for row in rows]
                sig = self._gate.graph_signature(parsed, pairs)
                return {"op": self._gate.graph_root_op(parsed),
                        "sig": list(sig)}
            inputs = [list(row[0]) for row in rows]
            outs = [execute(tuple(parsed), list(values))
                    for values in inputs]
            return {"op": "statement", "sig": outs}
        except (ValueError, TypeError, KeyError, IndexError):
            return None

    def _run_python_graph(
            self, graph: dict[str, Any], rows: list[list[Any]]
    ) -> tuple[str, list[int] | None, str]:
        """Sandbox one graph; supervisor compares outputs itself."""
        pairs = [[int(row[0]), int(row[1])] for row in rows]
        try:
            result = self._runner.run_candidate(graph, pairs)
        except TimeoutError as exc:
            return "timeout", None, f"python sandbox: {exc}"
        status = str(result.get("status", "error"))
        if status == "timeout":
            return "timeout", None, "python sandbox timed out"
        if status != "accepted":
            detail = ";".join(str(v) for v in
                               result.get("violations", []))
            return "error", None, f"python sandbox: {detail}"
        supervised = [int(v) for v in result["outputs"]]
        return "ok", supervised, ""

    def _run_python_statement(
            self, tokens: tuple[str, ...], rows: list[list[Any]]
    ) -> tuple[str, list[int] | None, str]:
        """Sandbox one token program with supervisor recompute."""
        case = Path(tempfile.mkdtemp(dir=str(self._workdir)))
        inputs = [list(row[0]) for row in rows]
        (case / "tokens.json").write_text(json.dumps(
            list(tokens)))
        (case / "rows.json").write_text(json.dumps(inputs))
        lib = str(Path(__file__).resolve().parents[1])
        argv = [sys.executable, "-c", _STMT_CHILD,
                str(case / "tokens.json"),
                str(case / "rows.json"),
                str(case / "outputs.json"), lib]
        try:
            outcome = self._runner.run_command(argv)
        except TimeoutError as exc:
            return "timeout", None, f"python sandbox: {exc}"
        if outcome["status"] == "timeout":
            return "timeout", None, "python sandbox timed out"
        if outcome["status"] != "ok":
            return "error", None, "python sandbox failed"
        try:
            raw = json.loads((case / "outputs.json").read_text())
            supervised = [execute(tokens, list(values))
                          for values in inputs]
        except (ValueError, TypeError, OSError) as exc:
            return "error", None, f"statement error: {exc}"
        if [int(v) for v in raw] != [int(v) for v in supervised]:
            return "error", None, "execution-mismatch"
        return "ok", [int(v) for v in supervised], ""

    def _curlee_command(self, stage: str, source: Path
                        ) -> list[str]:
        """Compiler invocation through the resource guard if present."""
        command = [str(self._compiler), stage]
        if stage == "run":
            command += ["--fuel", CURLEE_FUEL]
        command.append(str(source))
        if self._guard:
            return [sys.executable, str(GUARD_SCRIPT)] + command
        return command

    def _run_curlee_source(
            self, source_text: str, tag: str,
            costs: dict[str, dict[str, float]],
            commands: dict[str, dict[str, Any]]
    ) -> tuple[str, dict[str, Any]]:
        """Check then run one CURLEE harness under guard caps."""
        case = Path(tempfile.mkdtemp(dir=str(self._workdir)))
        path = case / f"{tag}.curlee"
        path.write_text(source_text)
        record: dict[str, Any] = {"source_sha256": _sha256_file(
            path), "check": {}, "run": {}, "passed": False}
        try:
            clock = CostClock()
            check = subprocess.run(
                self._curlee_command("check", path),
                capture_output=True, text=True,
                timeout=self._curlee_timeout)
            spent = clock.elapsed()
        except subprocess.TimeoutExpired:
            return "timeout", record
        costs["curlee_check"]["cpu_seconds"] += spent[
            "cpu_seconds"]
        costs["curlee_check"]["wall_seconds"] += spent[
            "wall_seconds"]
        self.ledger.record("backend_execution", spent["cpu_seconds"],
                           spent["wall_seconds"])
        check_rec = {"exit_code": check.returncode,
                     "stdout": check.stdout[-2000:],
                     "stderr": check.stderr[-2000:],
                     "wall_seconds": spent["wall_seconds"],
                     "cpu_seconds": spent["cpu_seconds"]}
        record["check"] = check_rec
        commands[f"curlee_check_{tag}"] = {
            "command": self._curlee_command("check", path),
            "exit_code": check.returncode,
            "wall_seconds": spent["wall_seconds"],
            "cpu_seconds": spent["cpu_seconds"]}
        if self._guard and check.returncode == 75:
            return "timeout", record
        if check.returncode != 0:
            return "check", record
        try:
            clock = CostClock()
            run = subprocess.run(
                self._curlee_command("run", path),
                capture_output=True, text=True,
                timeout=self._curlee_timeout)
            spent = clock.elapsed()
        except subprocess.TimeoutExpired:
            return "timeout", record
        costs["curlee_run"]["cpu_seconds"] += spent["cpu_seconds"]
        costs["curlee_run"]["wall_seconds"] += spent["wall_seconds"]
        self.ledger.record("backend_execution", spent["cpu_seconds"],
                           spent["wall_seconds"])
        match = _RUN_RESULT.search(run.stdout)
        record["run"] = {"exit_code": run.returncode,
                         "stdout": run.stdout[-2000:],
                         "stderr": run.stderr[-2000:],
                         "wall_seconds": spent["wall_seconds"],
                         "cpu_seconds": spent["cpu_seconds"]}
        commands[f"curlee_run_{tag}"] = {
            "command": self._curlee_command("run", path),
            "exit_code": run.returncode,
            "wall_seconds": spent["wall_seconds"],
            "cpu_seconds": spent["cpu_seconds"]}
        if self._guard and run.returncode == 75:
            return "timeout", record
        if run.returncode == 0 and match and match.group(1) == "0":
            record["passed"] = True
            return "ok", record
        if run.returncode == 0 and match and match.group(1) == "1":
            return "mismatch", record
        return "error", record

    def _parse(self, kind: str, text: str) -> Any:
        """Parse a gated candidate into graph or token objects."""
        if kind == "graph":
            graph = json.loads(text)
            validate_graph(graph)
            return graph
        tokens = tuple(text.split())
        statement_syntax.parse(tokens)
        return tokens

    def _check_identity(self, task: dict[str, Any],
                        graph: dict[str, Any]) -> str:
        """Exact polynomial identity for arithmetic candidates."""
        try:
            witness = task["witness"]["polynomial"]
            want = sorted((int(a), int(b), int(c))
                          for a, b, c in witness)
            got = sorted(tuple(int(v) for v in term)
                         for term in graph_polynomial(graph))
        except (ValueError, TypeError, KeyError):
            return "identity_error"
        if [list(term) for term in got] != [list(term)
                                            for term in want]:
            return "identity_mismatch"
        return ""

    def _base_verdict(self, task_id: str, kind: str, stage: str,
                      rows: list[list[Any]],
                      proposal: Mapping[str, Any] | None,
                      costs: dict[str, dict[str, float]]
                      ) -> dict[str, Any]:
        """Verdict skeleton with proposer costs already recorded."""
        tokens = {"prompt": 0, "completion": 0}
        cpu = 0.0
        if proposal is not None:
            usage = proposal.get("token_usage", {})
            if isinstance(usage, Mapping):
                for name in ("prompt", "completion"):
                    value = usage.get(name, 0)
                    if isinstance(value, bool):
                        continue
                    if isinstance(value, (int, float)):
                        tokens[name] = int(value)
            raw_cpu = proposal.get("cpu", 0.0)
            if isinstance(raw_cpu, bool):
                pass
            elif isinstance(raw_cpu, (int, float)):
                cpu = float(raw_cpu)
        costs["proposer"]["cpu_seconds"] += cpu
        return {"task_id": task_id, "kind": kind, "stage": stage,
                "accepted": False, "valid": False,
                "python_correct": None, "curlee_correct": None,
                "identity_ok": None, "rows_checked": len(rows),
                "mismatches": 0, "first_mismatch": None,
                "taxonomy": [], "detail": "", "signature": None,
                "counterexamples": [],
                "compiler": {"path": str(self._compiler),
                             "sha256": self._compiler_sha,
                             "guard": self._guard},
                "range_note": RANGE_NOTE, "costs": costs,
                "tokens": tokens, "proposal_cpu": cpu}

    def _evaluate(self, task_id: str, kind: str, text: str,
                  stage: str,
                  proposal: Mapping[str, Any] | None
                  ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        """Run gate, Python sandbox first, then CURLEE and identity.

        CURLEE check/run is skipped when the Python sandbox
        already fails the candidate, so broken proposals cost
        one backend instead of two. ``curlee_correct`` stays
        ``None`` for those verdicts.
        """
        task = self._task(task_id)
        rows = self._split_rows(task_id, stage)
        costs = _zero_costs()
        commands: dict[str, dict[str, Any]] = {}
        verdict = self._base_verdict(task_id, kind, stage, rows,
                                     proposal, costs)

        def finish(code: str, detail: str) -> tuple[
                dict[str, Any], dict[str, dict[str, Any]]]:
            if code:
                verdict["taxonomy"] = [code]
            verdict["detail"] = detail
            verdict["accepted"] = not verdict["taxonomy"]
            return verdict, commands

        clock = CostClock()
        gate = self._gate.gate(text, kind)
        spent = clock.elapsed()
        costs["gate"]["cpu_seconds"] += spent["cpu_seconds"]
        costs["gate"]["wall_seconds"] += spent["wall_seconds"]
        self.ledger.record("candidate_evaluation",
                           spent["cpu_seconds"],
                           spent["wall_seconds"])
        if gate["status"] != "accepted":
            verdict["valid"] = False
            joined = ";".join(str(v) for v in
                               gate["violations"])
            if "denied-reference" in joined:
                return finish("denied_reference", joined)
            if "budget" in joined:
                return finish("budget", joined)
            return finish("structural", joined)
        try:
            parsed = self._parse(kind, text)
        except (ValueError, TypeError) as exc:
            return finish("structural", f"parse: {exc}")
        verdict["valid"] = True
        verdict["signature"] = self._signature(
            kind, parsed,
            [list(row) for row in self._visible[task_id]])
        for row in rows:
            if abs(int(row[-1])) > INT64_MAX or \
                    int(row[-1]) < INT64_MIN:
                return finish("int64_range",
                              "expected row outside i64")
        try:
            if kind == "graph":
                source = _pair_harness(parsed, rows)
            else:
                source = _statement_harness(parsed, rows)
        except (ValueError, TypeError, KeyError) as exc:
            verdict["valid"] = False
            return finish("curlee_lower", f"lower: {exc}")
        tag = f"{task_id}_{stage}"
        clock = CostClock()
        if kind == "graph":
            state, outputs, detail = self._run_python_graph(
                parsed, rows)
        else:
            state, outputs, detail = self._run_python_statement(
                parsed, rows)
        spent = clock.elapsed()
        costs["sandbox"]["cpu_seconds"] += spent["cpu_seconds"]
        costs["sandbox"]["wall_seconds"] += spent["wall_seconds"]
        self.ledger.record("backend_execution", spent["cpu_seconds"],
                           spent["wall_seconds"])
        if state == "timeout":
            return finish("timeout", detail)
        if state != "ok" or outputs is None:
            verdict["python_correct"] = False
            if "execution-mismatch" in detail:
                return finish("execution_mismatch", detail)
            return finish("python_error", detail)
        for value in outputs:
            if value < INT64_MIN or value > INT64_MAX:
                verdict["python_correct"] = False
                return finish("int64_range",
                              "python output outside i64")
        want = [int(row[-1]) for row in rows]
        bad = [i for i, (got, exp) in
               enumerate(zip(outputs, want)) if got != exp]
        if bad:
            verdict["python_correct"] = False
            verdict["mismatches"] = len(bad)
            verdict["first_mismatch"] = bad[0]
            if stage != "final":
                verdict["counterexamples"] = [rows[i]
                                              for i in bad]
            return finish("python_mismatch",
                          f"{len(bad)} of {len(rows)} rows differ")
        verdict["python_correct"] = True
        status, record = self._run_curlee_source(
            source, tag, costs, commands)
        if status == "timeout":
            if record["check"] and \
                    record["check"].get("exit_code") == 75:
                return finish("timeout",
                              "resource guard refused job")
            return finish("timeout", "CURLEE stage timed out")
        if status == "check":
            check = record["check"]
            return finish("curlee_check",
                          f"check rc={check.get('exit_code')}")
        if status == "mismatch":
            verdict["curlee_correct"] = False
            verdict["mismatches"] = len(rows)
            verdict["first_mismatch"] = 0
            if stage != "final":
                verdict["counterexamples"] = [rows[0]]
            return finish("curlee_mismatch",
                          "CURLEE run disagrees with expected")
        if status == "error":
            verdict["curlee_correct"] = False
            return finish("curlee_error",
                          "CURLEE run stage failed")
        verdict["curlee_correct"] = True
        if kind == "graph":
            code = self._check_identity(task, parsed)
            if code:
                verdict["identity_ok"] = False
                return finish(code, "polynomial identity differs")
            verdict["identity_ok"] = True
        return finish("", "valid and correct in both backends")

    def judge_visible(self, task_id: str, kind: str, text: str,
                      proposal: Mapping[str, Any] | None = None
                      ) -> tuple[dict[str, Any],
                                 dict[str, dict[str, Any]]]:
        """Judge visible plus query rows; safe for repair loops."""
        try:
            return self._evaluate(task_id, kind, text, "visible",
                                  proposal)
        except TimeoutError as exc:
            costs = _zero_costs()
            verdict = self._base_verdict(
                task_id, kind, "visible",
                self._split_rows(task_id, "visible"), proposal,
                costs)
            verdict["taxonomy"] = ["timeout"]
            verdict["detail"] = f"budget exhausted: {exc}"
            return verdict, {}

    def judge_final(self, task_id: str, kind: str, text: str,
                    proposal: Mapping[str, Any] | None = None
                    ) -> tuple[dict[str, Any],
                               dict[str, dict[str, Any]]]:
        """Judge sealed held rows; raises without held data."""
        self.held_rows(task_id)
        try:
            return self._evaluate(task_id, kind, text, "final",
                                  proposal)
        except TimeoutError as exc:
            costs = _zero_costs()
            verdict = self._base_verdict(
                task_id, kind, "final",
                self.held_rows(task_id), proposal, costs)
            verdict["taxonomy"] = ["timeout"]
            verdict["detail"] = f"budget exhausted: {exc}"
            return verdict, {}

    def counterexamples(self, task_id: str, kind: str, text: str,
                        scope: str = "visible"
                        ) -> list[list[Any]]:
        """Failing rows for repair; held scope is always refused."""
        if scope == "held":
            raise SealedHeldError(
                "held rows are unavailable to repair")
        if scope == "query":
            rows = ([list(row)
                     for row in self._visible[task_id]]
                    + [list(row) for row in self._query[task_id]])
        elif scope == "visible":
            rows = [list(row) for row in self._visible[task_id]]
        else:
            raise ValueError(f"unknown scope: {scope!r}")
        try:
            parsed = self._parse(kind, text)
        except (ValueError, TypeError):
            return []
        try:
            if kind == "graph":
                _, outs_or_none, _ = self._run_python_graph(
                    parsed, rows)
            else:
                _, outs_or_none, _ = self._run_python_statement(
                    parsed, rows)
        except TimeoutError:
            return []
        if outs_or_none is None:
            return []
        outs = [int(v) for v in outs_or_none]
        return [list(row) for row, got in zip(rows, outs)
                if int(got) != int(row[-1])]

    def cluster(self, task_id: str,
                proposals: list[Mapping[str, Any]]
                ) -> tuple[list[dict[str, Any]], int,
                           dict[str, float]]:
        """Group proposals by visible (op, sig); keep cheapest."""
        clock = CostClock()
        rows = [list(row) for row in self._visible[task_id]]
        groups: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for index, proposal in enumerate(proposals):
            kind = str(proposal.get("kind", "graph"))
            text = str(proposal.get("candidate_text", ""))
            key = f"invalid:{index}"
            try:
                parsed = self._parse(kind, text)
                sig = self._signature(kind, parsed, rows)
                if sig is not None:
                    key = kind + ":" + sig["op"] + ":" + ",".join(
                        str(v) for v in sig["sig"])
            except (ValueError, TypeError):
                parsed = None
            cpu = proposal.get("cpu", 0.0)
            cpu_value = float(cpu) if isinstance(
                cpu, (int, float)) and not isinstance(
                    cpu, bool) else 0.0
            entry = {"kind": kind, "candidate_text": text,
                     "proposal": dict(proposal),
                     "cpu": cpu_value, "bytes": len(text.encode())}
            if key not in groups:
                groups[key] = entry
                order.append(key)
            else:
                prior = groups[key]
                if (cpu_value, entry["bytes"]) < (
                        prior["cpu"], prior["bytes"]):
                    groups[key] = entry
        spent = clock.elapsed()
        cost = {"cpu_seconds": spent["cpu_seconds"],
                "wall_seconds": spent["wall_seconds"]}
        self.ledger.record("candidate_evaluation",
                           spent["cpu_seconds"],
                           spent["wall_seconds"])
        kept = [groups[key] for key in order]
        return kept, len(proposals) - len(kept), cost

    @staticmethod
    def wrong_control(task: Mapping[str, Any],
                      visible_rows: list[list[Any]]
                      ) -> tuple[str, str]:
        """Deliberately-wrong candidate; must always be rejected."""
        if task.get("interface", "graph") == "list":
            want = int(visible_rows[0][-1])
            word = "two" if want == 1 else "one"
            return "statement", f"return {word} stop"
        x_value = int(visible_rows[0][0])
        y_value = int(visible_rows[0][1])
        want = int(visible_rows[0][-1])
        guesses = (("add", x_value + y_value),
                   ("mul", x_value * y_value),
                   ("sub", x_value - y_value))
        for op, guess in guesses:
            if guess != want:
                graph = {"input_types": {"x": "Int",
                                          "y": "Int"},
                         "nodes": [{"op": op, "type": "Int",
                                    "args": ["x", "y"]}],
                         "output": 0}
                return "graph", json.dumps(graph)
        raise ValueError("no wrong single-op graph available")
