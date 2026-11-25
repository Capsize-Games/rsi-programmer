"""Dev-bank diagnostic runner: every arm, one judge, frozen bar.

Phase 6 executes the frozen protocol from Kanban card
``bce2d353-1794-4466-b747-fee7d0a1a04`` on the 30 development
tasks (5 per family over A6/A7/A8/S1/S2/S3). Arms (a) LLM
single-shot, (b) bounded repair, (c) repair plus retrieval,
(d) streaming enumeration control and (e) seeded random valid
candidate all pass through one identical :class:`LlmJudge`;
dev held rows open identically for every arm only through
guarded final calls, and sealed rows are never loaded (every
task id must start with ``llm_dev_``).

Cost rule: proposer CPU is harness wall around generation,
tokens are worker-reported estimates, every judge phase is
ledgered per arm, and exclusions match the judge defaults
(donor library, pretraining, sealed handling, inference
estimates). The verdict is computed from the frozen bar with
no post-run threshold edits: PROCEED iff the best LLM arm
reaches a deep family where streaming scores zero AND its
cost per accepted program stays within M=3 of streaming on
shared-solved tasks AND repair or retrieval shows a strict
identical-task delta; otherwise KILL with no sealed run.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from rsi_programmer.cost_ledger import CostLedger
from rsi_programmer.llm_bank import BANK_DIRNAME
from rsi_programmer.llm_dev_protocol import ARMS, LLM_ARMS
from rsi_programmer.llm_dev_protocol import BANK_ID, CARD_ID
from rsi_programmer.llm_dev_protocol import CANDIDATE_BUDGET_D
from rsi_programmer.llm_dev_protocol import MAX_OPS_D, TOP_K
from rsi_programmer.llm_dev_protocol import DevProtocol
from rsi_programmer.llm_dev_protocol import FROZEN_TASK_IDS
from rsi_programmer.llm_dev_protocol import _canonical
from rsi_programmer.llm_dev_protocol import evaluate_bar
from rsi_programmer.llm_dev_protocol import kind_for_interface
from rsi_programmer.llm_fragments import FragmentLibrary
from rsi_programmer.llm_judge import LlmJudge
from rsi_programmer.llm_proposer import MODEL_ID
from rsi_programmer.llm_proposer import ProposerAdapter
from rsi_programmer.llm_repair import CandidateFn, RepairLoop
from rsi_programmer.llm_retrieval import FragmentRetriever
from rsi_programmer.probe import enumerate_programs, lower_to_dag
from rsi_programmer.program_cost import CostClock

__all__ = ["DevDiagnosticRunner", "DevProtocol",
           "FROZEN_TASK_IDS", "TaskCandidateFn",
           "evaluate_bar", "kind_for_interface"]

REPORT_NAME = "inversion-dev-results.md"

ARM_E_GRAPH_OPS = ("add", "sub", "mul")
ARM_E_STATEMENT_DECK = (
    "return acc stop",
    "return one stop",
    "for set add acc value end return acc stop",
    "for set add acc one end return acc stop",
)


def _lite_verdict(verdict: Mapping[str, Any]
                  ) -> dict[str, Any]:
    """Public copy of a verdict without row values."""
    body = dict(verdict)
    counter = body.get("counterexamples", [])
    body["counterexamples"] = []
    body["counterexample_count"] = len(counter) \
        if isinstance(counter, list) else 0
    return body


def _lite_proposal(proposal: Mapping[str, Any]
                   ) -> dict[str, Any]:
    """Provenance slice of a proposal, no candidate text."""
    usage = proposal.get("token_usage", {})
    return {"model": proposal.get("model"),
            "prompt_sha256": proposal.get("prompt_sha256"),
            "token_usage": dict(usage) if isinstance(
                usage, Mapping) else {},
            "cpu": float(proposal.get("cpu", 0.0))}


def _verdict_cpu(verdict: Mapping[str, Any]) -> float:
    """Sum every judge-phase CPU sample in one verdict."""
    costs = verdict.get("costs", {})
    total = 0.0
    if isinstance(costs, Mapping):
        for slot in costs.values():
            if isinstance(slot, Mapping):
                total += float(slot.get("cpu_seconds", 0.0))
    return total


def _verdict_tokens(verdict: Mapping[str, Any]
                    ) -> dict[str, int]:
    """Prompt plus completion tokens in one verdict."""
    tokens = verdict.get("tokens", {})
    if not isinstance(tokens, Mapping):
        return {"prompt": 0, "completion": 0}
    return {"prompt": int(tokens.get("prompt", 0)),
            "completion": int(tokens.get("completion", 0))}


class _Found(Exception):
    """Early exit carrying the first matching Expr."""

    def __init__(self, expr: Any) -> None:
        """Keep the matching expression for lowering."""
        super().__init__("streaming match")
        self.expr = expr


class _BudgetStop(Exception):
    """Early exit when arm (d) exhausts its budget."""


TaskCandidateFn = Callable[
    [str, int, str | None, list[dict[str, Any]]],
    tuple[str, Mapping[str, int], float]]
"""Task-aware proposer: (task, round, prompt, history)."""


def _bind_task(candidate_fn: TaskCandidateFn,
               task_id: str) -> CandidateFn:
    """Adapt a task-aware proposer to the repair-loop shape."""
    def run(round_no: int, prompt: str | None,
            rounds: list[dict[str, Any]]
            ) -> tuple[str, Mapping[str, int], float]:
        return candidate_fn(task_id, round_no, prompt, rounds)

    return run


class DevDiagnosticRunner:
    """Run all five arms through one judge on the dev bank."""

    def __init__(self, outdir: Path, *,
                 tasks: Mapping[str, Mapping[str, Any]],
                 visible: Mapping[str, list[list[Any]]],
                 query: Mapping[str, list[list[Any]]],
                 held: Mapping[str, list[list[Any]]],
                 expected_compiler_sha: str,
                 protocol: DevProtocol | None = None
                 ) -> None:
        """Bind bank splits; assert dev-only protocol ids."""
        self._protocol = protocol if protocol is not None \
            else DevProtocol()
        self._task_ids = tuple(self._protocol.task_ids)
        if not self._task_ids:
            raise ValueError("protocol task list is empty")
        self._tasks = {key: dict(value)
                       for key, value in tasks.items()}
        for task_id in self._task_ids:
            if not task_id.startswith("llm_dev_"):
                raise ValueError(
                    "sealed pool is out of scope: "
                    f"{task_id!r}")
            if task_id not in self._tasks:
                raise ValueError(
                    f"protocol task missing: {task_id}")
            if self._tasks[task_id].get("pool", "dev") \
                    != "dev":
                raise ValueError(
                    f"non-dev pool task: {task_id}")
        for key in held:
            if not str(key).startswith("llm_dev_"):
                raise ValueError(
                    f"sealed held row refused: {key!r}")
        missing = [task for task in self._task_ids
                   if task not in held]
        if missing:
            raise ValueError(
                f"dev held rows missing: {missing}")
        self._visible = {key: [list(row) for row in rows]
                         for key, rows in visible.items()}
        self._query = {key: [list(row) for row in rows]
                       for key, rows in query.items()}
        self._held = {key: [list(row) for row in rows]
                      for key, rows in held.items()
                      if str(key).startswith("llm_dev_")}
        self._outdir = outdir
        self._outdir.mkdir(parents=True, exist_ok=True)
        self._judge = LlmJudge(
            outdir / "judge-work", tasks=self._tasks,
            visible=self._visible, query=self._query,
            held=self._held,
            expected_compiler_sha=expected_compiler_sha)
        self._adapter = ProposerAdapter()
        self._ledgers = {arm: CostLedger() for arm in ARMS}
        for arm in ARMS:
            self._ledgers[arm].declare_excluded(
                f"arm-{arm}: donor library creation "
                "outside this run")
            self._ledgers[arm].declare_excluded(
                f"arm-{arm}: LLM pretraining and proposal "
                "tokens outside the harness")
            self._ledgers[arm].declare_excluded(
                f"arm-{arm}: sealed held handling outside "
                "judge_final")
            self._ledgers[arm].declare_excluded(
                f"arm-{arm}: proposer inference estimates")
        self._records: dict[str, dict[str, dict[str, Any]]] = {
            arm: {} for arm in ARMS}
        self._repair: dict[tuple[str, str], dict[str, Any]] = {}
        self._retrieval: dict[str, dict[str, Any]] = {}
        self._final_opened = False
        self._library: FragmentLibrary | None = None
        self._library_info: dict[str, Any] = {}

    @property
    def protocol(self) -> DevProtocol:
        """The frozen protocol this run executes."""
        return self._protocol

    @property
    def task_ids(self) -> tuple[str, ...]:
        """Protocol task ids in execution order."""
        return self._task_ids

    @property
    def judge(self) -> LlmJudge:
        """The single judge every arm is measured through."""
        return self._judge

    def protocol_sha256(self) -> str:
        """Record this digest before any execution starts."""
        return self._protocol.sha256()

    def family_of(self, task_id: str) -> str:
        """Bank family label for one protocol task."""
        return str(self._tasks[task_id].get("family", ""))

    def kind_of(self, task_id: str) -> str:
        """Candidate kind for one protocol task interface."""
        return kind_for_interface(
            str(self._tasks[task_id].get("interface", "")))

    def _check_commands(self, commands: Mapping[str, Any],
                        task_id: str, stage: str) -> bool:
        """True when the CURLEE check stage passed cleanly."""
        wanted = f"curlee_check_{task_id}_{stage}"
        for name, entry in commands.items():
            if isinstance(entry, Mapping) and name == wanted:
                return int(entry.get("exit_code", 1)) == 0
        return False

    def _store(self, arm: str, record: dict[str, Any],
               commands: Mapping[str, Any],
               stage: str) -> None:
        """File one arm/task record with its check outcome."""
        record["check_passed"] = self._check_commands(
            commands, str(record["task_id"]), stage)
        record["commands"] = {
            str(name): {"exit_code": entry.get("exit_code"),
                        "wall_seconds": entry.get(
                            "wall_seconds")}
            for name, entry in commands.items()
            if isinstance(entry, Mapping)}
        self._records[arm][str(record["task_id"])] = record

    def _confirm(self, arm: str, task_id: str, kind: str,
                 text: str, proposal: Mapping[str, Any]
                 ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Judge one best candidate; ledger its proposal CPU."""
        self._ledgers[arm].record(
            "candidate_evaluation",
            float(proposal.get("cpu", 0.0)), 0.0)
        verdict, commands = self._judge.judge_visible(
            task_id, kind, text, proposal=proposal)
        return verdict, commands

    def _base_record(self, arm: str, task_id: str, kind: str,
                     text: str, proposal: Mapping[str, Any],
                     verdict: Mapping[str, Any],
                     extra_cpu: float,
                     extra_tokens: Mapping[str, int],
                     repair: dict[str, Any] | None,
                     retrieval: dict[str, Any] | None,
                     note: str) -> dict[str, Any]:
        """One uniform arm/task record for tables and bars."""
        tokens = _verdict_tokens(verdict)
        tokens["prompt"] += int(extra_tokens.get("prompt", 0))
        tokens["completion"] += int(
            extra_tokens.get("completion", 0))
        return {
            "task_id": task_id,
            "family": self.family_of(task_id),
            "kind": kind, "arm": arm,
            "status": "accepted_visible"
            if verdict["accepted"] else "rejected_visible",
            "accepted_visible": bool(verdict["accepted"]),
            "accepted_final": None,
            "candidate_text": text,
            "candidate_sha256": hashlib.sha256(
                text.encode("utf-8")).hexdigest(),
            "proposal": _lite_proposal(proposal),
            "verdict_visible": _lite_verdict(verdict),
            "verdict_final": None,
            "cpu_seconds": _verdict_cpu(verdict) + extra_cpu,
            "tokens": tokens,
            "repair": repair, "retrieval": retrieval,
            "note": note}

    def run_arm_a(self, candidate_fn: TaskCandidateFn
                  ) -> dict[str, Any]:
        """Single-shot Spark 1.3 proposals, judged once each."""
        done = 0
        for task_id in self._task_ids:
            kind = self.kind_of(task_id)
            spec = {"task_id": task_id, "kind": kind,
                    "visible_rows": self._visible[task_id],
                    "fragments": [], "counterexamples": [],
                    "params": {}}
            text, usage, cpu = candidate_fn(
                task_id, 0, None, [])
            proposal = self._adapter.propose(
                spec, text,
                token_usage={str(k): int(v)
                             for k, v in dict(usage).items()},
                cpu=float(cpu))
            verdict, commands = self._confirm(
                "a", task_id, kind, text, proposal)
            self._store("a", self._base_record(
                "a", task_id, kind, text, proposal, verdict,
                0.0, {"prompt": 0, "completion": 0},
                None, None, ""), commands, "visible")
            done += 1
        return {"arm": "a", "tasks": done}

    def run_arm_b(self, candidate_fn: TaskCandidateFn
                  ) -> dict[str, Any]:
        """Bounded repair over visible/query rows per task."""
        done = 0
        for task_id in self._task_ids:
            kind = self.kind_of(task_id)
            loop = RepairLoop(
                self._adapter, self._judge,
                visible=self._visible, query=self._query,
                fragments={},
                max_rounds=self._protocol.max_rounds,
                query_cap=self._protocol.query_cap,
                token_cap=self._protocol.token_cap,
                cpu_cap=self._protocol.cpu_cap_s,
                ledger=self._ledgers["b"])
            artifact = loop.repair(
                task_id, kind,
                _bind_task(candidate_fn, task_id), params={})
            self._repair[("b", task_id)] = artifact
            rounds = artifact["rounds"]
            pick = next((entry for entry in rounds
                         if entry["verdict"] is not None
                         and entry["verdict"]["accepted"]),
                        rounds[0])
            text = str(pick["candidate_text"])
            proposal = dict(pick["proposal"])
            verdict, commands = self._confirm(
                "b", task_id, kind, text, proposal)
            totals = artifact["totals"]
            record = self._base_record(
                "b", task_id, kind, text, proposal, verdict,
                float(totals["cpu_seconds"]),
                {"prompt": int(totals["prompt_tokens"]),
                 "completion": int(
                     totals["completion_tokens"])},
                {"status": artifact["status"],
                 "accepted_round": artifact["accepted_round"],
                 "rounds": len(rounds),
                 "single_shot_accepted": artifact[
                     "single_shot_accepted"],
                 "revealed_count": artifact[
                     "revealed_count"]},
                None, "")
            self._store("b", record, commands, "visible")
            done += 1
        return {"arm": "b", "tasks": done}

    def _judge_final_guarded(
            self, task_id: str, kind: str, text: str,
            proposal: Mapping[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Sealed-stage call; marks dev held as opened."""
        self._final_opened = True
        return self._judge.judge_final(
            task_id, kind, text, proposal=proposal)

    def _qualify_evidence(self, task_id: str, kind: str,
                          text: str,
                          proposal: Mapping[str, Any]
                          ) -> dict[str, Any]:
        """Held-acceptance evidence for one library entry."""
        before = self._judge.compiler_sha
        verdict, _ = self._judge_final_guarded(
            task_id, kind, text, proposal)
        after = self._judge.compiler_sha
        wrong_kind, wrong_text = LlmJudge.wrong_control(
            self._tasks[task_id],
            [list(row) for row in self._visible[task_id]])
        wrong_verdict, _ = self._judge.judge_visible(
            task_id, wrong_kind, wrong_text)
        rerun, _ = self._judge_final_guarded(
            task_id, kind, text, proposal)
        rows = int(rerun.get("rows_checked", 0) or 0)
        python_rows = rows if rerun.get(
            "python_correct") is True else 0
        curlee_rows = rows if rerun.get(
            "curlee_correct") is True else 0
        return {"candidate_text": text, "verdict": verdict,
                "compiler_before": before,
                "compiler_after": after,
                "wrong_control_accepted": bool(
                    wrong_verdict.get("accepted")),
                "python_rows": python_rows,
                "curlee_rows": curlee_rows}

    def build_library(self) -> dict[str, Any]:
        """Qualify dev-held-accepted a/b programs, then freeze."""
        library = FragmentLibrary(
            ledger=self._ledgers["c"])
        entries: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for arm in ("a", "b"):
            for task_id in self._task_ids:
                record = self._records[arm].get(task_id)
                if record is None:
                    skipped.append({"task_id": task_id,
                                    "arm": arm,
                                    "reason": "no candidate"})
                    continue
                kind = str(record["kind"])
                text = str(record["candidate_text"])
                proposal = self._adapter.propose(
                    {"task_id": task_id, "kind": kind,
                     "visible_rows": self._visible[task_id],
                     "fragments": [], "counterexamples": [],
                     "params": {}},
                    text, token_usage={}, cpu=0.0)
                try:
                    evidence = self._qualify_evidence(
                        task_id, kind, text, proposal)
                    entry = library.add_entry(
                        text, kind,
                        self.family_of(task_id),
                        {"task": task_id, "arm": arm,
                         "model": MODEL_ID,
                         "prompt_hash": str(
                             proposal.get("prompt_sha256")),
                         "artifact": f"dev-diagnostic/arm-{arm}"},
                        evidence)
                except ValueError as exc:
                    skipped.append({"task_id": task_id,
                                    "arm": arm,
                                    "reason": str(exc)[:200]})
                    continue
                entries.append({"task_id": task_id,
                                "arm": arm,
                                "fragment_id": entry["id"]})
        version = library.version_hash()
        self._library = library
        self._library_info = {
            "version_hash": version,
            "entries": entries, "skipped": skipped,
            "entry_count": len(entries)}
        return dict(self._library_info)

    def run_arm_c(self, candidate_fn: TaskCandidateFn
                  ) -> dict[str, Any]:
        """Repair with retrieval from the frozen in-run library."""
        if self._library is None:
            raise ValueError(
                "build_library must run before arm (c)")
        retriever = FragmentRetriever(self._library)
        done = 0
        for task_id in self._task_ids:
            kind = self.kind_of(task_id)
            family = self.family_of(task_id)
            fragments, shown = retriever.retrieve(
                family, None, self._protocol.top_k)
            self._retrieval[task_id] = {
                "version": self._library.version_hash(),
                "family": family, "shown": shown}
            loop = RepairLoop(
                self._adapter, self._judge,
                visible=self._visible, query=self._query,
                fragments={task_id: fragments},
                max_rounds=self._protocol.max_rounds,
                query_cap=self._protocol.query_cap,
                token_cap=self._protocol.token_cap,
                cpu_cap=self._protocol.cpu_cap_s,
                ledger=self._ledgers["c"])
            artifact = loop.repair(
                task_id, kind,
                _bind_task(candidate_fn, task_id), params={})
            self._repair[("c", task_id)] = artifact
            rounds = artifact["rounds"]
            pick = next((entry for entry in rounds
                         if entry["verdict"] is not None
                         and entry["verdict"]["accepted"]),
                        rounds[0])
            text = str(pick["candidate_text"])
            proposal = dict(pick["proposal"])
            verdict, commands = self._confirm(
                "c", task_id, kind, text, proposal)
            totals = artifact["totals"]
            record = self._base_record(
                "c", task_id, kind, text, proposal, verdict,
                float(totals["cpu_seconds"]),
                {"prompt": int(totals["prompt_tokens"]),
                 "completion": int(
                     totals["completion_tokens"])},
                {"status": artifact["status"],
                 "accepted_round": artifact["accepted_round"],
                 "rounds": len(rounds),
                 "single_shot_accepted": artifact[
                     "single_shot_accepted"],
                 "revealed_count": artifact[
                     "revealed_count"]},
                {"version": self._library.version_hash(),
                 "shown": shown}, "")
            self._store("c", record, commands, "visible")
            done += 1
        return {"arm": "c", "tasks": done,
                "library_version": self._library_info.get(
                    "version_hash")}

    def run_arm_d(self) -> dict[str, Any]:
        """Streaming enumeration control, first visible match."""
        done = 0
        for task_id in self._task_ids:
            kind = self.kind_of(task_id)
            family = self.family_of(task_id)
            clock = CostClock()
            if kind != "graph":
                self._ledgers["d"].record(
                    "search", 0.0, 0.0)
                self._store("d", {
                    "task_id": task_id, "family": family,
                    "kind": kind, "arm": "d",
                    "status": "not_applicable",
                    "accepted_visible": False,
                    "accepted_final": None,
                    "candidate_text": "",
                    "candidate_sha256": "",
                    "proposal": _lite_proposal({}),
                    "verdict_visible": None,
                    "verdict_final": None,
                    "cpu_seconds": 0.0,
                    "tokens": {"prompt": 0,
                               "completion": 0},
                    "tried": 0, "repair": None,
                    "retrieval": None,
                    "note": "no enumeration grammar for "
                            "statement tasks"}, {}, "visible")
                done += 1
                continue
            rows = tuple(
                (int(row[0]), int(row[1]))
                for row in self._visible[task_id])
            expected = tuple(int(row[-1]) for row in
                             self._visible[task_id])
            max_ops = MAX_OPS_D.get(family)
            if max_ops is None:
                raise ValueError(
                    f"no streaming op band: {family}")
            budget = self._protocol.candidate_budget_d
            state: dict[str, Any] = {"tried": 0}

            def watch(expr: Any, sig: Any,
                      tried: int) -> None:
                state["tried"] = tried
                if tried > budget:
                    raise _BudgetStop()
                if tuple(sig) == expected:
                    raise _Found(expr)

            text = ""
            try:
                enumerate_programs(
                    rows, expected, max_ops,
                    on_candidate=watch)
                note = "no visible match within budget"
            except _Found as found:
                graph = lower_to_dag(found.expr)
                text = json.dumps(graph, sort_keys=True)
                note = ""
            except _BudgetStop:
                note = "candidate budget exhausted"
            spent = clock.elapsed()
            self._ledgers["d"].record(
                "search", spent["cpu_seconds"],
                spent["wall_seconds"])
            if not text:
                self._store("d", {
                    "task_id": task_id, "family": family,
                    "kind": kind, "arm": "d",
                    "status": "rejected_visible",
                    "accepted_visible": False,
                    "accepted_final": None,
                    "candidate_text": "",
                    "candidate_sha256": "",
                    "proposal": _lite_proposal({}),
                    "verdict_visible": None,
                    "verdict_final": None,
                    "cpu_seconds": spent["cpu_seconds"],
                    "tokens": {"prompt": 0,
                               "completion": 0},
                    "tried": state["tried"],
                    "repair": None, "retrieval": None,
                    "note": note}, {}, "visible")
                done += 1
                continue
            verdict, commands = self._judge.judge_visible(
                task_id, kind, text)
            self._ledgers["d"].record(
                "candidate_evaluation", 0.0, 0.0)
            self._store("d", {
                "task_id": task_id, "family": family,
                "kind": kind, "arm": "d",
                "status": "accepted_visible"
                if verdict["accepted"] else "rejected_visible",
                "accepted_visible": bool(
                    verdict["accepted"]),
                "accepted_final": None,
                "candidate_text": text,
                "candidate_sha256": hashlib.sha256(
                    text.encode("utf-8")).hexdigest(),
                "proposal": _lite_proposal({}),
                "verdict_visible": _lite_verdict(verdict),
                "verdict_final": None,
                "cpu_seconds": _verdict_cpu(verdict)
                + spent["cpu_seconds"],
                "tokens": {"prompt": 0, "completion": 0},
                "tried": state["tried"],
                "repair": None, "retrieval": None,
                "note": note}, commands, "visible")
            done += 1
        return {"arm": "d", "tasks": done}

    def run_arm_e(self) -> dict[str, Any]:
        """Seeded random valid candidate, gate-checked."""
        done = 0
        for pos, task_id in enumerate(self._task_ids):
            kind = self.kind_of(task_id)
            family = self.family_of(task_id)
            rng = random.Random(self._protocol.seed + pos)
            if kind == "graph":
                op = rng.choice(ARM_E_GRAPH_OPS)
                graph = {
                    "input_types": {"x": "Int", "y": "Int"},
                    "nodes": [{"op": op, "type": "Int",
                               "args": ["x", "y"]}],
                    "output": 0}
                text = json.dumps(graph, sort_keys=True)
            else:
                text = rng.choice(ARM_E_STATEMENT_DECK)
            gate = self._adapter.gate(text, kind)
            clock = CostClock()
            spent = clock.elapsed()
            self._ledgers["e"].record(
                "candidate_evaluation", spent["cpu_seconds"],
                spent["wall_seconds"])
            if gate["status"] != "accepted":
                self._store("e", {
                    "task_id": task_id, "family": family,
                    "kind": kind, "arm": "e",
                    "status": "rejected_visible",
                    "accepted_visible": False,
                    "accepted_final": None,
                    "candidate_text": text,
                    "candidate_sha256": hashlib.sha256(
                        text.encode("utf-8")).hexdigest(),
                    "proposal": _lite_proposal({}),
                    "verdict_visible": None,
                    "verdict_final": None,
                    "cpu_seconds": spent["cpu_seconds"],
                    "tokens": {"prompt": 0,
                               "completion": 0},
                    "repair": None, "retrieval": None,
                    "note": ";".join(
                        gate["violations"])[:200]},
                    {}, "visible")
                done += 1
                continue
            verdict, commands = self._judge.judge_visible(
                task_id, kind, text)
            self._store("e", {
                "task_id": task_id, "family": family,
                "kind": kind, "arm": "e",
                "status": "accepted_visible"
                if verdict["accepted"] else "rejected_visible",
                "accepted_visible": bool(
                    verdict["accepted"]),
                "accepted_final": None,
                "candidate_text": text,
                "candidate_sha256": hashlib.sha256(
                    text.encode("utf-8")).hexdigest(),
                "proposal": _lite_proposal({}),
                "verdict_visible": _lite_verdict(verdict),
                "verdict_final": None,
                "cpu_seconds": _verdict_cpu(verdict)
                + spent["cpu_seconds"],
                "tokens": {"prompt": 0, "completion": 0},
                "repair": None, "retrieval": None,
                "note": f"seed={self._protocol.seed + pos}"},
                commands, "visible")
            done += 1
        return {"arm": "e", "tasks": done}

    def finalize(self) -> dict[str, Any]:
        """Open dev held identically for every arm's best."""
        done = 0
        for arm in ARMS:
            for task_id in self._task_ids:
                record = self._records[arm].get(task_id)
                if record is None or not record.get(
                        "candidate_text"):
                    continue
                verdict, commands = \
                    self._judge_final_guarded(
                        task_id, str(record["kind"]),
                        str(record["candidate_text"]))
                record["accepted_final"] = bool(
                    verdict["accepted"])
                record["verdict_final"] = _lite_verdict(
                    verdict)
                record["status"] = "accepted_final" \
                    if verdict["accepted"] else "held_rejected"
                record["check_passed"] = \
                    record.get("check_passed", False) and \
                    self._check_commands(
                        commands, task_id, "final")
                record["commands"] = {
                    str(name): {
                        "exit_code": entry.get("exit_code"),
                        "wall_seconds": entry.get(
                            "wall_seconds")}
                    for name, entry in commands.items()
                    if isinstance(entry, Mapping)}
                record["cpu_seconds"] = float(
                    record["cpu_seconds"]) + _verdict_cpu(
                        verdict)
                done += 1
        return {"finalized": done,
                "held_opened": self._final_opened}

    def _accepted_sets(
            self, key: str) -> dict[str, set[str]]:
        """Final or visible accepts per arm over protocol tasks."""
        return {
            arm: {task for task in self._task_ids
                  if self._records[arm].get(task, {}).get(
                      key) is True}
            for arm in ARMS}

    def _taxonomy(self, key: str) -> dict[str, dict[str, int]]:
        """Rejection code counts per arm for one verdict key."""
        table: dict[str, dict[str, int]] = {}
        for arm in ARMS:
            counts: dict[str, int] = {}
            for task in self._task_ids:
                verdict = self._records[arm].get(
                    task, {}).get(key)
                if isinstance(verdict, Mapping):
                    for code in verdict.get("taxonomy", []):
                        counts[str(code)] = counts.get(
                            str(code), 0) + 1
            table[arm] = counts
        return table

    def summary(self) -> dict[str, Any]:
        """Aggregate tables, costs and the frozen-bar decision."""
        visible = self._accepted_sets("accepted_visible")
        final = self._accepted_sets("accepted_final")
        families = sorted({self.family_of(task)
                           for task in self._task_ids})
        by_family = {
            family: {
                arm: sum(1 for task in self._task_ids
                         if self.family_of(task) == family
                         and task in final[arm])
                for arm in ARMS}
            for family in families}
        cpu = {
            arm: sum(
                float(self._records[arm].get(task, {}).get(
                    "cpu_seconds", 0.0))
                for task in self._task_ids)
            for arm in ARMS}
        tokens = {arm: {
            "prompt": sum(int(self._records[arm].get(
                task, {}).get("tokens", {}).get("prompt", 0))
                for task in self._task_ids),
            "completion": sum(int(self._records[arm].get(
                task, {}).get("tokens", {}).get(
                    "completion", 0))
                for task in self._task_ids)}
            for arm in ARMS}
        per_accepted = {
            arm: (cpu[arm] / len(final[arm])
                  if final[arm] else None)
            for arm in ARMS}

        def marginal(top: str, base: str) -> float | None:
            """Added CPU per added accept over a simpler arm."""
            gain = len(final[top]) - len(final[base])
            if gain <= 0:
                return None
            return (cpu[top] - cpu[base]) / gain

        marginals: dict[str, float | None] = {
            "b_vs_a": marginal("b", "a"),
            "c_vs_b": marginal("c", "b")}
        finals = sum(1 for arm in ARMS for task in
                     self._task_ids
                     if self._records[arm].get(
                         task, {}).get("verdict_final"))
        checks = sum(
            1 for arm in ARMS for task in self._task_ids
            if self._records[arm].get(task, {}).get(
                "verdict_final") and self._records[arm][
                    task].get("check_passed"))
        converts = [task for task in self._task_ids
                    if (self._repair.get(("b", task), {}).get(
                        "status") == "accepted"
                        and not self._repair[("b", task)].get(
                            "single_shot_accepted"))]
        rejected0 = [task for task in self._task_ids
                     if ("b", task) in self._repair
                     and not self._repair[("b", task)].get(
                         "single_shot_accepted")]
        only_b = sorted(set(final["b"]) - set(final["a"]))
        only_a = sorted(set(final["a"]) - set(final["b"]))
        only_c = sorted(set(final["c"]) - set(final["b"]))
        only_c_nb = sorted(set(final["b"]) - set(final["c"]))
        decision = evaluate_bar(
            {arm: set(final[arm]) for arm in LLM_ARMS},
            set(final["d"]),
            {task: self.family_of(task)
             for task in self._task_ids},
            {arm: {task: float(self._records[arm].get(
                task, {}).get("cpu_seconds", 0.0))
                for task in self._task_ids} for arm in ARMS},
            deep_families=tuple(
                self._protocol.deep_families),
            multiple_m=self._protocol.cost_multiple_m)
        best = decision.get("best_arm")
        stream = set(final["d"])
        best_set = set(final[best]) if best else set()
        body = {
            "protocol": self._protocol.to_dict(),
            "protocol_sha256": self.protocol_sha256(),
            "bank_dir": BANK_DIRNAME,
            "tasks": list(self._task_ids),
            "arms": list(ARMS),
            "counts": {
                "visible_accepted": {
                    arm: len(visible[arm]) for arm in ARMS},
                "final_accepted": {
                    arm: len(final[arm]) for arm in ARMS}},
            "breakdown": {"by_family_final": by_family},
            "costs": {"cpu_seconds": cpu, "tokens": tokens,
                      "cpu_by_task": {
                          arm: {task: float(
                              self._records[arm].get(
                                  task, {}).get(
                                  "cpu_seconds", 0.0))
                              for task in self._task_ids}
                          for arm in ARMS},
                      "per_accepted_total": per_accepted,
                      "marginal_per_accepted": marginals,
                      "ledgers": {
                          arm: self._ledgers[arm].to_dict()
                          for arm in ARMS},
                      "judge_ledger": self._judge.ledger
                      .to_dict()},
            "curlee_check_pass_rate_final":
                (checks / finals if finals else None),
            "repair_conversion_b": {
                "converted": len(converts),
                "round0_rejected": len(rejected0),
                "rate": (len(converts) / len(rejected0)
                         if rejected0 else None),
                "tasks": converts},
            "retrieval_delta_c_vs_b": {
                "delta": len(final["c"]) - len(final["b"]),
                "c_only": only_c, "b_only": only_c_nb},
            "repair_delta_b_vs_a": {
                "delta": len(final["b"]) - len(final["a"]),
                "b_only": only_b, "a_only": only_a},
            "taxonomy": {"visible": self._taxonomy(
                "verdict_visible"),
                "final": self._taxonomy("verdict_final")},
            "discordance_best_vs_streaming": {
                "best_arm": best,
                "both": sorted(best_set & stream),
                "llm_only": sorted(best_set - stream),
                "streaming_only": sorted(stream - best_set),
                "neither": sorted(
                    set(self._task_ids) - best_set - stream)},
            "library": dict(self._library_info),
            "decision": decision,
            "held_opened": self._final_opened}
        path = self._outdir / "summary.json"
        path.write_text(_canonical(body).decode("utf-8"))
        return body

    def write_report(self, path: Path) -> Path:
        """Write inversion-dev-results.md tables plus verdict."""
        body = self.summary()
        lines = ["# RSI Inversion dev-diagnostic results",
                 "",
                 f"Protocol sha256: `{body['protocol_sha256']}`",
                 f"Card: `{CARD_ID}`, bank: `{BANK_ID}`",
                 f"Model: `{self._protocol.model_id}` "
                 f"({self._protocol.model_version})",
                 f"Tasks: {len(self._task_ids)} dev "
                 "(5 per family over A6/A7/A8/S1/S2/S3)",
                 f"Held opened identically: {body['held_opened']}",
                 "",
                 f"## Verdict: {body['decision']['verdict']}",
                 ""]
        for reason in body["decision"]["reasons"]:
            lines.append(f"- {reason}")
        lines += ["", "## Final accepts per arm/family", "",
                  "| family | a | b | c | d | e |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for family, counts in body["breakdown"][
                "by_family_final"].items():
            lines.append(
                f"| {family} | {counts['a']} | {counts['b']} | "
                f"{counts['c']} | {counts['d']} | "
                f"{counts['e']} |")
        lines += ["", "## Cost per accepted (CPU seconds)",
                  "",
                  "| arm | total CPU | accepts | "
                  "per accepted | marginal |",
                  "| --- | --- | --- | --- | --- |"]
        cpu = body["costs"]["cpu_seconds"]
        counts = body["counts"]["final_accepted"]
        per = body["costs"]["per_accepted_total"]
        marginals = body["costs"]["marginal_per_accepted"]
        margin_of = {"a": None, "b": marginals["b_vs_a"],
                     "c": marginals["c_vs_b"], "d": None,
                     "e": None}
        for arm in ARMS:
            total = f"{cpu[arm]:.3f}"
            take = f"{per[arm]:.3f}" if per[arm] is not None \
                else "n/a"
            marg = margin_of[arm]
            show = f"{marg:.3f}" if marg is not None else "n/a"
            lines.append(f"| {arm} | {total} | {counts[arm]} | "
                         f"{take} | {show} |")
        lines += ["",
                  "## CURLEE check-pass rate (final stage)",
                  "",
                  f"{body['curlee_check_pass_rate_final']}",
                  "",
                  "## Repair conversion (b vs a)",
                  "",
                  f"converted "
                  f"{body['repair_conversion_b']['converted']} "
                  f"of "
                  f"{body['repair_conversion_b']['round0_rejected']} "
                  f"round-0 rejections "
                  f"(rate "
                  f"{body['repair_conversion_b']['rate']})",
                  "",
                  "## Retrieval delta (c vs b, identical tasks)",
                  "",
                  f"delta {body['retrieval_delta_c_vs_b']['delta']}; "
                  f"c-only {body['retrieval_delta_c_vs_b']['c_only']}; "
                  f"b-only {body['retrieval_delta_c_vs_b']['b_only']}",
                  "",
                  "## Rejection taxonomy (final)",
                  ""]
        for arm, table in body["taxonomy"]["final"].items():
            lines.append(f"- {arm}: {table or 'no rejections'}")
        lines += ["",
                  "## Paired same-task discordance",
                  "",
                  f"best arm "
                  f"{body['discordance_best_vs_streaming']['best_arm']} "
                  f"vs streaming: both "
                  f"{body['discordance_best_vs_streaming']['both']}; "
                  f"llm-only "
                  f"{body['discordance_best_vs_streaming']['llm_only']}; "
                  f"streaming-only "
                  f"{body['discordance_best_vs_streaming']}"
                  "['streaming_only']; "
                  f"neither "
                  f"{body['discordance_best_vs_streaming']['neither']}",
                  "",
                  "## Honesty notes",
                  "",
                  "- Single run, one seed: development "
                  "evidence only, not a capability claim.",
                  "- Dev held rows are tunable; the sealed "
                  "pool was never loaded (dev-only ids "
                  "asserted in code).",
                  "- Costs: proposer CPU is harness wall, "
                  "tokens worker-reported, every judge phase "
                  "ledgered per arm; exclusions recorded in "
                  "the summary ledgers.",
                  ""]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")
        return path
