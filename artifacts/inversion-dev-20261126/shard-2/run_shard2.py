"""Phase 6 shard-2 driver (2026-11-26 re-trial): 5 dev tasks, all arms.

Shard tasks: llm_dev_011 (A6), llm_dev_016 (A7), llm_dev_043 (A8),
llm_dev_050 (S1), llm_dev_088 (S3). Sealed pool never loaded: only
llm_dev_* splits are passed to the runner, which refuses other keys.

NEW-trial scope: measures the two committed cost levers at main
commit 0f29072 (python-first judge gating + near-miss repair gate)
on the dev set; current main code used as-is, no judge/repair/
protocol/bar edits.

No-gate procedure (same as shard-5 pattern): machine stats are a
record only, never a launch gate; no pre-waiting or polling for
idle; every CURLEE invocation runs through the repo resource guard
inside LlmJudge, and a guard refusal is recorded as a timeout
verdict for that case.
"""
import hashlib
import json
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent
SHARD = ("llm_dev_011", "llm_dev_016", "llm_dev_043",
         "llm_dev_050", "llm_dev_088")

from rsi_programmer import statement_syntax  # noqa: E402
from rsi_programmer.llm_bank import LlmBankGenerator  # noqa: E402
from rsi_programmer.llm_dev_diagnostic import (  # noqa: E402
    REPORT_NAME, DevDiagnosticRunner,
)
from rsi_programmer.llm_dev_protocol import (  # noqa: E402
    DevProtocol, kind_for_interface,
)
from rsi_programmer.probe import (  # noqa: E402
    enumerate_programs, interpret, lower_to_dag, validate_graph,
)

FAMILY_MAX_OPS = {"A6": 6, "A7": 7, "A8": 8}
TRIED_CAP = 300000


def machine_stats() -> dict:
    """Uptime plus MemAvailable; a record, never a launch gate."""
    uptime = Path("/proc/uptime").read_text().split()
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, val = line.partition(":")
        mem[key.strip()] = val.strip()
    return {"uptime_seconds": float(uptime[0]),
            "MemAvailable": mem.get("MemAvailable", "unknown")}


class _Hit(Exception):
    def __init__(self, expr):
        super().__init__("graph match")
        self.expr = expr


class _Budget(Exception):
    pass


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
    """Spark 1.3 proposer: visible rows plus revealed feedback only."""

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


def main() -> None:
    (OUT / "machine_start.json").write_text(json.dumps(
        {"machine": machine_stats(), "shard": list(SHARD),
         "event": "start"}, indent=2, sort_keys=True) + "\n")
    bank = LlmBankGenerator()
    protocol_doc, task_list, visible_all, query_all = \
        bank.load_public()
    held_all = LlmBankGenerator().load_held(unseal=True)
    tasks = {t["id"]: t for t in task_list
             if t["id"] in SHARD}
    assert set(tasks) == set(SHARD), set(SHARD) - set(tasks)
    assert all(t["pool"] == "dev" for t in tasks.values())
    visible = {k: visible_all[k] for k in SHARD}
    query = {k: query_all[k] for k in SHARD}
    held = {k: held_all[k] for k in SHARD}
    compiler_sha = protocol_doc["curlee_binary"]["sha256"]
    proto = DevProtocol(task_ids=tuple(SHARD))
    runner = DevDiagnosticRunner(
        OUT, tasks=tasks, visible=visible, query=query,
        held=held, expected_compiler_sha=compiler_sha,
        protocol=proto)
    (OUT / "protocol_sha256.txt").write_text(
        runner.protocol_sha256() + "\n")
    families = {tid: tasks[tid]["family"] for tid in SHARD}
    candidate_fn = make_candidate_fn(visible, families)

    from rsi_programmer.llm_dev_diagnostic import _bind_task
    task_fns = {tid: _bind_task(candidate_fn, tid)
                for tid in SHARD}

    def fn(task_id, round_no, prompt, history):
        return candidate_fn(task_id, round_no, prompt, history)

    print("protocol", runner.protocol_sha256(), flush=True)
    print("arm_a", runner.run_arm_a(fn), flush=True)
    print("arm_b", runner.run_arm_b(fn), flush=True)
    print("library", runner.build_library(), flush=True)
    print("arm_c", runner.run_arm_c(fn), flush=True)
    print("arm_d", runner.run_arm_d(), flush=True)
    print("arm_e", runner.run_arm_e(), flush=True)
    print("finalize", runner.finalize(), flush=True)
    body = runner.summary()
    print("decision", body["decision"], flush=True)
    path = runner.write_report(OUT / REPORT_NAME)
    print("report", path, flush=True)
    _ = task_fns
    (OUT / "machine_end.json").write_text(json.dumps(
        {"machine": machine_stats(), "shard": list(SHARD),
         "event": "end"}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
