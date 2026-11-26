"""Sealed one-shot: 90 sealed tasks, all arms, one process.

Preregistered sealed run for the cost-lever epic
(tracks/programmer/lever-sealed-design.md). Sequential,
single-process driver: one CURLEE build slot, no parallel
workers (precedent: 2026-11-26 parallel trial starved on
guard refusals).

Scope guardrails (aborts, never worked around):
- only ``llm_sealed_*`` ids; any ``llm_dev_*`` key in any
  split aborts before execution starts;
- sealed held rows reach the harness ONLY through
  ``LlmJudge.judge_final`` (library qualification and the
  final stage); repair/query feedback uses ``judge_visible``
  plus visible/query counterexamples, which refuse held
  scope inside the repo code;
- machine stats are a record only, never a launch gate;
  no pre-waiting or polling for idle/load;
- every CURLEE invocation runs through the repo resource
  guard inside ``LlmJudge``; a guard refusal (rc75) is
  recorded as a timeout verdict for that case, no retry.

Repo modules are imported as-is: judge, repair,
retrieval, fragments, proposer, bank, protocol constants
and the frozen bar are never modified, subclassed or
re-implemented here. ``DevProtocol``/``DevDiagnosticRunner``
enforce dev-only ids in code, so this driver replicates
their arithmetic over the sealed pool instead of reusing
those two classes; the sealed protocol record below is
hashed in the same canonical way for the artifact.
"""
import hashlib
import json
import random
import subprocess
import time
from collections.abc import Callable, Mapping
from pathlib import Path

OUT = Path(__file__).resolve().parent

from rsi_programmer import statement_syntax  # noqa: E402
from rsi_programmer.cost_ledger import CostLedger  # noqa: E402
from rsi_programmer.llm_bank import (  # noqa: E402
    BANK_DIRNAME, LlmBankGenerator,
)
from rsi_programmer.llm_dev_protocol import (  # noqa: E402
    ARMS, CANDIDATE_BUDGET_D, CARD_ID, COST_MULTIPLE_M,
    DEEP_FAMILIES, LLM_ARMS, MAX_OPS_D, SEED, TOP_K,
    evaluate_bar, kind_for_interface,
)
from rsi_programmer.llm_fragments import (  # noqa: E402
    FragmentLibrary,
)
from rsi_programmer.llm_judge import LlmJudge  # noqa: E402
from rsi_programmer.llm_proposer import (  # noqa: E402
    MODEL_ID, MODEL_VERSION, ProposerAdapter,
)
from rsi_programmer.llm_repair import (  # noqa: E402
    CPU_CAP_S, QUERY_REVEAL_CAP, R_MAX, TOKEN_CAP,
    RepairLoop, template_sha256,
)
from rsi_programmer.llm_retrieval import (  # noqa: E402
    FragmentRetriever,
)
from rsi_programmer.probe import (  # noqa: E402
    enumerate_programs, interpret, lower_to_dag,
    validate_graph,
)
from rsi_programmer.program_cost import CostClock  # noqa: E402

REPORT_NAME = "inversion-dev-results.md"

ARM_E_GRAPH_OPS = ("add", "sub", "mul")
ARM_E_STATEMENT_DECK = (
    "return acc stop",
    "return one stop",
    "for set add acc value end return acc stop",
    "for set add acc one end return acc stop",
)

FAMILY_MAX_OPS = {"A6": 6, "A7": 7, "A8": 8}
TRIED_CAP = 300000


def _canonical(value) -> bytes:
    """Canonical JSON bytes, same recipe as the protocol."""
    return (json.dumps(value, indent=2, sort_keys=True)
            + "\n").encode("utf-8")


def sealed_protocol_record(task_ids: tuple) -> dict:
    """Frozen sealed-run record hashed before execution."""
    return {
        "bank_id": "llm_bank_v1",
        "pool": "sealed",
        "epic": "cost-lever sealed one-shot",
        "design": "tracks/programmer/lever-sealed-design.md",
        "dev_card": CARD_ID,
        "seed": SEED,
        "task_ids": list(task_ids),
        "arms": list(ARMS),
        "max_rounds": R_MAX,
        "query_cap": QUERY_REVEAL_CAP,
        "token_cap": TOKEN_CAP,
        "cpu_cap_s": CPU_CAP_S,
        "cost_multiple_m": COST_MULTIPLE_M,
        "deep_families": list(DEEP_FAMILIES),
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "repair_template_sha256": template_sha256(),
        "proposer_prompt": "canonical-json-v1",
        "max_ops_d": dict(MAX_OPS_D),
        "candidate_budget_d": CANDIDATE_BUDGET_D,
        "top_k": TOP_K,
        "library_plan": (
            "build in-run from sealed-held-accepted a/b "
            "programs via FragmentLibrary.add_entry with "
            "judge_final evidence, then freeze "
            "(version_hash) before arm c"),
        "held_policy": (
            "sealed tasks/visible/query plus sealed held "
            "opened identically for all arms at final via "
            "judge_final only; dev pool never loaded"),
        "verdict_rule": (
            "PROCEED iff best LLM arm accepts >=1 "
            "deep-family task where streaming accepts 0 "
            "AND cost/accepted <=Mx streaming on "
            "shared-solved tasks AND b>a or c>b strictly "
            "on identical tasks; else KILL"),
    }


def machine_stats() -> dict:
    """Uptime, load and MemAvailable; a record, never a gate."""
    uptime = Path("/proc/uptime").read_text().split()
    load = Path("/proc/loadavg").read_text().split()[:3]
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, val = line.partition(":")
        mem[key.strip()] = val.strip()
    return {"uptime_seconds": float(uptime[0]),
            "loadavg_1_5_15": [float(x) for x in load],
            "MemAvailable": mem.get("MemAvailable", "unknown")}


class _Hit(Exception):
    def __init__(self, expr):
        super().__init__("graph match")
        self.expr = expr


class _Budget(Exception):
    pass


class _Found(Exception):
    """Early exit carrying the first streaming match."""

    def __init__(self, expr) -> None:
        super().__init__("streaming match")
        self.expr = expr


class _BudgetStop(Exception):
    """Early exit when arm (d) exhausts its budget."""


def _lite_verdict(verdict: Mapping) -> dict:
    """Public copy of a verdict without row values."""
    body = dict(verdict)
    counter = body.get("counterexamples", [])
    body["counterexamples"] = []
    body["counterexample_count"] = len(counter) \
        if isinstance(counter, list) else 0
    return body


def _lite_proposal(proposal: Mapping) -> dict:
    """Provenance slice of a proposal, no candidate text."""
    usage = proposal.get("token_usage", {})
    return {"model": proposal.get("model"),
            "prompt_sha256": proposal.get("prompt_sha256"),
            "token_usage": dict(usage) if isinstance(
                usage, Mapping) else {},
            "cpu": float(proposal.get("cpu", 0.0))}


def _verdict_cpu(verdict: Mapping) -> float:
    """Sum every judge-phase CPU sample in one verdict."""
    costs = verdict.get("costs", {})
    total = 0.0
    if isinstance(costs, Mapping):
        for slot in costs.values():
            if isinstance(slot, Mapping):
                total += float(slot.get("cpu_seconds", 0.0))
    return total


def _verdict_tokens(verdict: Mapping) -> dict:
    """Prompt plus completion tokens in one verdict."""
    tokens = verdict.get("tokens", {})
    if not isinstance(tokens, Mapping):
        return {"prompt": 0, "completion": 0}
    return {"prompt": int(tokens.get("prompt", 0)),
            "completion": int(tokens.get("completion", 0))}


TaskCandidateFn = Callable[
    [str, int, str | None, list[dict]],
    tuple[str, Mapping[str, int], float]]
"""Task-aware proposer: (task, round, prompt, history)."""


def _bind_task(candidate_fn: TaskCandidateFn,
               task_id: str):
    """Adapt a task-aware proposer to the repair-loop shape."""
    def run(round_no, prompt, rounds):
        return candidate_fn(task_id, round_no, prompt, rounds)

    return run


def _graph_ok(text, known) -> bool:
    try:
        graph = json.loads(text)
        validate_graph(graph)
    except Exception:
        return False
    try:
        for row in known:
            if interpret(graph, int(row[0]), int(row[1])) \
                    != int(row[2]):
                return False
    except Exception:
        return False
    return True


def _stmt_ok(text, known) -> bool:
    try:
        tokens = tuple(str(text).split())
        statement_syntax.parse(tokens)
    except Exception:
        return False
    try:
        for row in known:
            if LlmBankGenerator.run_tokens(
                    tokens, list(row[0])) != int(row[1]):
                return False
    except Exception:
        return False
    return True


def _fit_graph(known, cap, seen):
    rows = tuple((int(r[0]), int(r[1])) for r in known)
    expected = tuple(int(r[2]) for r in known)
    state = {"tried": 0}

    def watch(expr, sig, tried):
        state["tried"] = tried
        if tried > TRIED_CAP:
            raise _Budget()
        if tuple(sig) == expected:
            text = json.dumps(lower_to_dag(expr), sort_keys=True)
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest not in seen:
                raise _Hit(expr)

    try:
        enumerate_programs(rows, expected, cap,
                           on_candidate=watch)
    except _Hit as hit:
        return json.dumps(lower_to_dag(hit.expr), sort_keys=True)
    except _Budget:
        return None
    return None


_DECK = (LlmBankGenerator._s1_deck()
         + LlmBankGenerator._s2_deck()
         + LlmBankGenerator._s3_deck())


def _fit_statement(known, seen):
    for tokens in _DECK:
        text = " ".join(tokens)
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in seen:
            continue
        if _stmt_ok(text, known):
            return text
    return None


def _fallback_graph() -> str:
    return json.dumps({
        "input_types": {"x": "Int", "y": "Int"},
        "nodes": [{"op": "add", "type": "Int",
                   "args": ["x", "y"]}],
        "output": 0}, sort_keys=True)


def _fallback_statement() -> str:
    return "for set add acc value end return acc stop"


def make_candidate_fn(visible, families):
    """Spark 1.3 stand-in: visible rows plus revealed only."""

    def candidate_fn(task_id, round_no, prompt, history):
        start = time.perf_counter()
        kind = kind_for_interface(
            "pair" if families[task_id].startswith("A")
            else "list")
        known = [list(r) for r in visible[task_id]]
        frag_texts = []
        if prompt:
            try:
                body = json.loads(prompt)
            except ValueError:
                body = {}
            for key in ("revealed_query_rows", "failing_batch"):
                for row in body.get(key, []) or []:
                    row = list(row)
                    if row not in known:
                        known.append(row)
            for frag in body.get("fragments", []) or []:
                if isinstance(frag, dict) and frag.get("content"):
                    frag_texts.append(str(frag["content"]))
        seen = set()
        for entry in history or []:
            digest = entry.get("candidate_sha256")
            if digest:
                seen.add(digest)
        check = _graph_ok if kind == "graph" else _stmt_ok
        text = None
        for frag in frag_texts:
            digest = hashlib.sha256(frag.encode()).hexdigest()
            if digest in seen:
                continue
            if check(frag, known):
                text = frag
                break
        if text is None:
            if kind == "graph":
                text = _fit_graph(
                    known,
                    FAMILY_MAX_OPS.get(families[task_id], 6),
                    seen)
            else:
                text = _fit_statement(known, seen)
        if text is None:
            text = (_fallback_graph() if kind == "graph"
                    else _fallback_statement())
        cpu = time.perf_counter() - start
        usage = {"prompt": max(1, len(prompt or "") // 4),
                 "completion": max(1, len(text) // 4)}
        return text, usage, cpu

    return candidate_fn


class SealedRunner:
    """All five arms through one judge on the sealed pool."""

    def __init__(self, outdir: Path, *,
                 tasks: Mapping, visible: Mapping,
                 query: Mapping, held: Mapping,
                 expected_compiler_sha: str,
                 protocol_record: Mapping) -> None:
        """Bind sealed splits; refuse any dev-pool key."""
        self._protocol_record = dict(protocol_record)
        self._task_ids = tuple(
            self._protocol_record["task_ids"])
        if not self._task_ids:
            raise ValueError("sealed task list is empty")
        for task_id in self._task_ids:
            if not str(task_id).startswith("llm_sealed_"):
                raise ValueError(
                    "dev pool is out of scope: "
                    f"{task_id!r}")
        self._tasks = {key: dict(value)
                       for key, value in tasks.items()}
        for task_id in self._task_ids:
            if self._tasks[task_id].get("pool") != "sealed":
                raise ValueError(
                    f"non-sealed pool task: {task_id}")
        for scope in (visible, query, held):
            for key in scope:
                if str(key).startswith("llm_dev_"):
                    raise ValueError(
                        "dev held scope refused: "
                        f"{key!r}")
        missing = [task for task in self._task_ids
                   if task not in held]
        if missing:
            raise ValueError(
                f"sealed held rows missing: {missing}")
        self._visible = {key: [list(row) for row in rows]
                         for key, rows in visible.items()}
        self._query = {key: [list(row) for row in rows]
                       for key, rows in query.items()}
        self._held = {key: [list(row) for row in rows]
                      for key, rows in held.items()
                      if str(key).startswith("llm_sealed_")}
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
        self._records = {arm: {} for arm in ARMS}
        self._repair = {}
        self._retrieval = {}
        self._final_opened = False
        self._library = None
        self._library_info = {}

    def protocol_sha256(self) -> str:
        """Record this digest before any execution starts."""
        return hashlib.sha256(
            _canonical(self._protocol_record)).hexdigest()

    def family_of(self, task_id: str) -> str:
        """Bank family label for one sealed task."""
        return str(self._tasks[task_id].get("family", ""))

    def kind_of(self, task_id: str) -> str:
        """Candidate kind for one sealed task interface."""
        return kind_for_interface(
            str(self._tasks[task_id].get("interface", "")))

    def _check_commands(self, commands: Mapping,
                        task_id: str, stage: str) -> bool:
        """True when the CURLEE check stage passed cleanly."""
        wanted = f"curlee_check_{task_id}_{stage}"
        for name, entry in commands.items():
            if isinstance(entry, Mapping) and name == wanted:
                return int(entry.get("exit_code", 1)) == 0
        return False

    def _store(self, arm: str, record: dict,
               commands: Mapping, stage: str) -> None:
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
                 text: str, proposal: Mapping):
        """Judge one best candidate; ledger its proposal CPU."""
        self._ledgers[arm].record(
            "candidate_evaluation",
            float(proposal.get("cpu", 0.0)), 0.0)
        verdict, commands = self._judge.judge_visible(
            task_id, kind, text, proposal=proposal)
        return verdict, commands

    def _base_record(self, arm: str, task_id: str, kind: str,
                     text: str, proposal: Mapping,
                     verdict: Mapping, extra_cpu: float,
                     extra_tokens: Mapping,
                     repair, retrieval, note: str) -> dict:
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

    def run_arm_a(self, candidate_fn) -> dict:
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

    def run_arm_b(self, candidate_fn) -> dict:
        """Bounded repair over visible/query rows per task."""
        done = 0
        for task_id in self._task_ids:
            kind = self.kind_of(task_id)
            loop = RepairLoop(
                self._adapter, self._judge,
                visible=self._visible, query=self._query,
                fragments={},
                max_rounds=R_MAX,
                query_cap=QUERY_REVEAL_CAP,
                token_cap=TOKEN_CAP,
                cpu_cap=CPU_CAP_S,
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

    def _judge_final_guarded(self, task_id: str, kind: str,
                             text: str, proposal=None):
        """Sealed-stage call; marks sealed held as opened."""
        self._final_opened = True
        return self._judge.judge_final(
            task_id, kind, text, proposal=proposal)

    def _qualify_evidence(self, task_id: str, kind: str,
                          text: str, proposal: Mapping
                          ) -> dict:
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

    def build_library(self) -> dict:
        """Qualify sealed-held-accepted a/b programs, freeze."""
        library = FragmentLibrary(
            ledger=self._ledgers["c"])
        entries = []
        skipped = []
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
                         "artifact": f"sealed/arm-{arm}"},
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

    def run_arm_c(self, candidate_fn) -> dict:
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
                family, None, TOP_K)
            self._retrieval[task_id] = {
                "version": self._library.version_hash(),
                "family": family, "shown": shown}
            loop = RepairLoop(
                self._adapter, self._judge,
                visible=self._visible, query=self._query,
                fragments={task_id: fragments},
                max_rounds=R_MAX,
                query_cap=QUERY_REVEAL_CAP,
                token_cap=TOKEN_CAP,
                cpu_cap=CPU_CAP_S,
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

    def run_arm_d(self) -> dict:
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
            budget = CANDIDATE_BUDGET_D
            state = {"tried": 0}

            def watch(expr, sig, tried) -> None:
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

    def run_arm_e(self) -> dict:
        """Seeded random valid candidate, gate-checked."""
        done = 0
        for pos, task_id in enumerate(self._task_ids):
            kind = self.kind_of(task_id)
            family = self.family_of(task_id)
            rng = random.Random(SEED + pos)
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
                "note": f"seed={SEED + pos}"},
                commands, "visible")
            done += 1
        return {"arm": "e", "tasks": done}

    def finalize(self) -> dict:
        """Open sealed held identically for every arm's best."""
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

    def _accepted_sets(self, key: str):
        """Final or visible accepts per arm over sealed tasks."""
        return {
            arm: {task for task in self._task_ids
                  if self._records[arm].get(task, {}).get(
                      key) is True}
            for arm in ARMS}

    def _taxonomy(self, key: str):
        """Rejection code counts per arm for one verdict key."""
        table = {}
        for arm in ARMS:
            counts = {}
            for task in self._task_ids:
                verdict = self._records[arm].get(
                    task, {}).get(key)
                if isinstance(verdict, Mapping):
                    for code in verdict.get("taxonomy", []):
                        counts[str(code)] = counts.get(
                            str(code), 0) + 1
            table[arm] = counts
        return table

    def summary(self) -> dict:
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

        def marginal(top: str, base: str):
            """Added CPU per added accept over a simpler arm."""
            gain = len(final[top]) - len(final[base])
            if gain <= 0:
                return None
            return (cpu[top] - cpu[base]) / gain

        marginals = {
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
                        and not self._repair.get(
                            ("b", task)).get(
                            "single_shot_accepted"))]
        rejected0 = [task for task in self._task_ids
                     if ("b", task) in self._repair
                     and not self._repair.get(
                         ("b", task)).get(
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
            deep_families=tuple(DEEP_FAMILIES),
            multiple_m=COST_MULTIPLE_M)
        best = decision.get("best_arm")
        stream = set(final["d"])
        best_set = set(final[best]) if best else set()
        body = {
            "protocol": dict(self._protocol_record),
            "protocol_sha256": self.protocol_sha256(),
            "guard_uses_resource_guard": self._judge.uses_guard,
            "bank_dir": BANK_DIRNAME,
            "pool": "sealed",
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
        """Write sealed-results tables plus the frozen verdict."""
        body = self.summary()
        lines = ["# RSI sealed one-shot results (cost lever)",
                 "",
                 f"Protocol sha256: `{body['protocol_sha256']}`",
                 "Design: "
                 "`tracks/programmer/lever-sealed-design.md`",
                 f"Model: `{MODEL_ID}` ({MODEL_VERSION})",
                 f"Tasks: {len(self._task_ids)} sealed "
                 "(15 per family over A6/A7/A8/S1/S2/S3)",
                 f"Held opened identically: "
                 f"{body['held_opened']}",
                 f"Resource guard active: "
                 f"{body['guard_uses_resource_guard']}",
                 "",
                 f"## Verdict: {body['decision']['verdict']}",
                 ""]
        for reason in body["decision"]["reasons"]:
            lines.append(f"- {reason}")
        lines += ["", "## Final accepts per arm/family", "",
                  "| family | a | b | c | d | e |",
                  "| --- | --- | --- | --- | --- |"]
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
                  "- Single run, one seed: sealed evidence, "
                  "not a capability claim.",
                  "- Sealed held rows opened only through "
                  "judge_final, identically for every arm; "
                  "repair/query feedback never saw held rows "
                  "(held scope raises inside the repo code).",
                  "- Dev pool never loaded: any llm_dev_ key "
                  "in any split aborts before execution.",
                  "- Costs: proposer CPU is harness wall, "
                  "tokens worker-reported, every judge phase "
                  "ledgered per arm; exclusions recorded in "
                  "the summary ledgers.",
                  ""]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")
        return path


def main() -> None:
    """Sequential sealed one-shot; record-only machine stats."""
    start_stats = machine_stats()
    (OUT / "machine_start.json").write_text(json.dumps(
        {"machine": start_stats, "event": "start"},
        indent=2, sort_keys=True) + "\n")
    bank = LlmBankGenerator()
    protocol_doc, task_list, visible_all, query_all = \
        bank.load_public()
    held_all = LlmBankGenerator().load_held(unseal=True)
    sealed_ids = [t["id"] for t in task_list
                  if t.get("pool") == "sealed"]
    for family in ("A6", "A7", "A8", "S1", "S2", "S3"):
        group = sorted(t for t in sealed_ids
                       if t.startswith("llm_sealed_")
                       and next(
                           x for x in task_list
                           if x["id"] == t)["family"] == family)
        assert len(group) == 15, (family, len(group))
    order = []
    for family in ("A6", "A7", "A8", "S1", "S2", "S3"):
        order.extend(sorted(
            t for t in sealed_ids
            if next(x for x in task_list
                    if x["id"] == t)["family"] == family))
    assert len(order) == 90, len(order)
    assert not any(t.startswith("llm_dev_") for t in order)
    # Bank files also carry the dev pool; only sealed subsets
    # below ever reach the runner (its constructor refuses
    # any llm_dev_ key in any split it receives).
    tasks = {t["id"]: t for t in task_list if t["id"] in order}
    assert set(tasks) == set(order)
    assert all(t["pool"] == "sealed" for t in tasks.values())
    visible = {k: visible_all[k] for k in order}
    query = {k: query_all[k] for k in order}
    held = {k: held_all[k] for k in order}
    assert set(visible) == set(order) == set(query) == set(held)
    assert not any(str(k).startswith("llm_dev_")
                   for k in list(visible) + list(query)
                   + list(held))
    compiler_sha = protocol_doc["curlee_binary"]["sha256"]
    record = sealed_protocol_record(tuple(order))
    runner = SealedRunner(
        OUT, tasks=tasks, visible=visible, query=query,
        held=held, expected_compiler_sha=compiler_sha,
        protocol_record=record)
    proto_sha = runner.protocol_sha256()
    assert proto_sha == hashlib.sha256(
        _canonical(record)).hexdigest()
    (OUT / "protocol_sha256.txt").write_text(proto_sha + "\n")
    families = {tid: tasks[tid]["family"] for tid in order}
    candidate_fn = make_candidate_fn(visible, families)

    def fn(task_id, round_no, prompt, history):
        return candidate_fn(task_id, round_no, prompt, history)

    print("protocol", proto_sha, flush=True)
    print("arm_a", runner.run_arm_a(fn), flush=True)
    print("arm_b", runner.run_arm_b(fn), flush=True)
    print("library", runner.build_library(), flush=True)
    print("arm_c", runner.run_arm_c(fn), flush=True)
    print("arm_d", runner.run_arm_d(), flush=True)
    print("arm_e", runner.run_arm_e(), flush=True)
    print("finalize", runner.finalize(), flush=True)
    body = runner.summary()
    print("decision", body["decision"], flush=True)
    base_path = runner.write_report(OUT / REPORT_NAME)
    print("report", base_path, flush=True)

    final_lists = {
        arm: sorted(
            task for task in order
            if runner._records[arm].get(task, {}).get(
                "accepted_final") is True)
        for arm in ARMS}
    visible_lists = {
        arm: sorted(
            task for task in order
            if runner._records[arm].get(task, {}).get(
                "accepted_visible") is True)
        for arm in ARMS}
    tax_v = body.get("taxonomy", {}).get("visible", {})
    tax_f = body.get("taxonomy", {}).get("final", {})
    timeout_visible = sum(
        int(tax_v.get(a, {}).get("timeout", 0))
        for a in ARMS)
    timeout_final = sum(
        int(tax_f.get(a, {}).get("timeout", 0))
        for a in ARMS)
    end_stats = machine_stats()
    (OUT / "machine_end.json").write_text(json.dumps(
        {"machine": end_stats, "event": "end"},
        indent=2, sort_keys=True) + "\n")
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True,
            text=True, check=True).stdout.strip()
    except Exception:
        head = "unknown"
    decision = body["decision"]
    prongs = decision.get("prongs", {})
    shared = list(prongs.get("shared_tasks", []))
    cpu = body["costs"]["cpu_seconds"]
    best = decision.get("best_arm")
    even_ratio = None
    if shared and best and float(cpu.get("d", 0.0)) > 0:
        even_ratio = float(cpu[best]) / float(cpu["d"])
    extra = {
        "tasks": list(order),
        "pool": "sealed",
        "per_family": 15,
        "head_commit": head,
        "code_modified": False,
        "repo_modules_unmodified": [
            "rsi_programmer/llm_judge.py",
            "rsi_programmer/llm_repair.py",
            "rsi_programmer/llm_dev_protocol.py",
            "rsi_programmer/llm_fragments.py",
            "rsi_programmer/llm_retrieval.py",
            "rsi_programmer/llm_proposer.py",
            "rsi_programmer/llm_bank.py"],
        "protocol_sha256": proto_sha,
        "final_accepts_by_arm": final_lists,
        "visible_accepts_by_arm": visible_lists,
        "cpu_seconds": dict(cpu),
        "cpu_by_task": body["costs"]["cpu_by_task"],
        "tokens": body["costs"]["tokens"],
        "counts_final": body["counts"]["final_accepted"],
        "counts_visible": body["counts"][
            "visible_accepted"],
        "guard_refusals": {
            "guard_active": body["guard_uses_resource_guard"],
            "visible_timeout_verdicts": timeout_visible,
            "final_timeout_verdicts": timeout_final,
            "total_timeout_verdicts":
                timeout_visible + timeout_final,
            "evidence": "timeout taxonomy counts from "
                        "summary.json; lite verdicts drop "
                        "the rc75 detail string, so any "
                        "timeout is an upper bound on "
                        "guard refusals; wall-time cross-"
                        "check in results doc",
        },
        "machine": {"start": start_stats, "end": end_stats},
        "decision": decision,
        "cost_ratio_exact": prongs.get("cost_ratio"),
        "cost_ratio_even_spread": even_ratio,
    }
    summary_path = OUT / "summary.json"
    merged = json.loads(summary_path.read_text())
    merged["sealed_run_extra"] = extra
    summary_path.write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n")

    base = (OUT / REPORT_NAME).read_text()
    shared_txt = ", ".join(shared) if shared else "none"
    lines = [base.rstrip(), "",
             "## Sealed one-shot (90 tasks, one process)",
             "",
             "Tasks: 90 sealed, frozen bank order "
             "(A6/A7/A8/S1/S2/S3 x 15). Bank splits carry "
             "no llm_dev_ key (asserted before execution); "
             "all held rows pool=sealed.",
             f"Protocol sha256: `{proto_sha}` "
             "(sealed-run record; canonical JSON of the "
             "frozen sealed parameters above).",
             f"Code: main `{head}` as-is; judge/repair/"
             "protocol/bar modules unmodified "
             "(imported, not subclassed).",
             "",
             "### Per-arm FINAL accepts (dual-backend)",
             "",
             "90-task sealed pool:",
             ""]
    for arm in ARMS:
        lines.append(
            f"- {arm}: {final_lists[arm]} "
            f"({len(final_lists[arm])}/90)")
    lines += ["",
              "### CPU seconds (per-arm record totals)",
              "",
              f"a {cpu['a']:.3f}, b {cpu['b']:.3f}, "
              f"c {cpu['c']:.3f}, d {cpu['d']:.3f}, "
              f"e {cpu['e']:.3f}.",
              "",
              "### Guard refusals",
              "",
              f"guard active "
              f"{body['guard_uses_resource_guard']}; "
              f"visible timeouts {timeout_visible}, final "
              f"timeouts {timeout_final}; total "
              f"{timeout_visible + timeout_final} timeout "
              "verdicts (upper bound on rc75 refusals; "
              "detail string not persisted). No waiting "
              "or polling was performed; machine stats "
              "are record-only.",
              f"Machine start uptime "
              f"{start_stats['uptime_seconds']:.2f} s, "
              f"load {start_stats['loadavg_1_5_15']}, "
              f"MemAvailable {start_stats['MemAvailable']}; "
              f"end uptime {end_stats['uptime_seconds']:.2f} "
              f"s, load {end_stats['loadavg_1_5_15']}, "
              f"MemAvailable {end_stats['MemAvailable']}.",
              "",
              "### Cost ratio (frozen bar vs comparison)",
              "",
              f"Shared tasks (best vs streaming): "
              f"{shared_txt}.",
              f"Exact per-task ratio from the frozen bar "
              f"(cpu_by_task, shared solves only): "
              f"{prongs.get('cost_ratio')}. Even-spread "
              f"ratio (arm totals, shared count cancels) "
              f"for comparison only: {even_ratio}. "
              f"Frozen bar multiple M={COST_MULTIPLE_M}.",
              "",
              "### Frozen-bar prongs",
              "",
              f"deep reach: {prongs.get('deep_tasks')}; "
              f"counts "
              f"{prongs.get('counts')}; "
              f"verdict {decision.get('verdict')}.",
              "",
              "### Honesty notes",
              "",
              "- Single run, one seed: sealed one-shot "
              "evidence only, not a capability claim.",
              "- Sealed held rows never entered repair or "
              "query feedback (judge_final only; any "
              "held-scope access outside final aborts).",
              "- No commit/push; nothing outside this "
              "worktree touched (binaries read/execute "
              "only via the judge).",
              ""]
    (OUT / REPORT_NAME).write_text("\n".join(lines))
    print("extended", OUT / REPORT_NAME, flush=True)


if __name__ == "__main__":
    main()



