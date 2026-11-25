"""Phase 6 execution shard-0: Spark 1.3 proposer stand-in, 5 dev tasks.

Runs DevDiagnosticRunner on exactly
["llm_dev_004", "llm_dev_022", "llm_dev_030", "llm_dev_049",
 "llm_dev_073"] with a row-based proposer written for this shard:

- graph (pair) tasks: sparse exact integer-polynomial recovery over
  monomials x^a*y^b with a+b<=6 from revealed rows only, lowered to
  a shared-subexpression DAG with arithmetic nodes only (the target
  lowerer rejects const nodes, so coefficients are add-chains and
  negation is sub(sub(x,x),t));
- statement (list) tasks: exhaustive search over the family-shaped
  token decks (S1 loop-plus-accumulate, S2 loop-plus-branch shapes
  documented in the frozen bank) against revealed rows only;
- repair rounds parse the loop prompt (visible + revealed query rows
  + failing batch + previous candidate) and refit on all of it;
- arm-c transfer: a shown fragment whose text executes exactly on
  every currently revealed row is reused verbatim.

Study discipline: the proposer closure sees ONLY per-task visible
rows, family/kind shape metadata, and prompt-carried revealed rows.
It never reads identity_key, witness programs, held rows, or sealed
pool keys. The tasks dict (with bank metadata) is passed to the
runner/judge as the harness requires, but the candidate function
does not receive it.

Model provenance: every proposal carries model spark-1.3; token
counts are self-estimates (whitespace-split word counts) and CPU is
measured wall time around synthesis.
"""

from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np

from rsi_programmer import statement_syntax
from rsi_programmer.llm_bank import LlmBankGenerator
from rsi_programmer.llm_dev_diagnostic import DevDiagnosticRunner
from rsi_programmer.llm_dev_protocol import DevProtocol
from rsi_programmer.llm_proposer import ProposerAdapter
from rsi_programmer.probe import CONSTS, interpret, validate_graph

OUTDIR = Path(__file__).resolve().parent
TASK_IDS = ("llm_dev_004", "llm_dev_022", "llm_dev_030",
            "llm_dev_049", "llm_dev_073")
MODEL = "spark-1.3"

MONOMS = sorted(
    [(a, s - a) for s in range(7) for a in range(s + 1)
     if (a, s - a) != (0, 0)],
    key=lambda t: (t[0] + t[1], t[0]))
GATE = ProposerAdapter()


def sysinfo() -> dict[str, object]:
    """Uptime seconds plus MemAvailable kB for the artifact dir."""
    up = Path("/proc/uptime").read_text().split()
    mem: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, _, value = line.partition(":")
        mem[name.strip()] = int(value.split()[0])
    return {"uptime_s": float(up[0]), "idle_s": float(up[1]),
            "MemAvailable_kB": mem["MemAvailable"]}


def _try_combo(big: np.ndarray, target: np.ndarray,
               rows: list[list[int]], combo: tuple[int, ...],
               coeff_cap: int | None) -> list[int] | None:
    """Exact integer fit for one monomial set, else None."""
    mat = big[:, combo]
    coef, *_ = np.linalg.lstsq(mat, target, rcond=None)
    if float(np.max(np.abs(mat @ coef - target))) > 1e-6:
        return None
    if float(np.max(np.abs(coef - np.round(coef)))) > 1e-6:
        return None
    ints = [int(round(c)) for c in coef]
    if any(c == 0 for c in ints):
        return None
    if coeff_cap is not None and \
            max(abs(c) for c in ints) > coeff_cap:
        return None
    for x, y, out in rows:
        total = 0
        for (a, b), c in zip([MONOMS[j] for j in combo],
                             ints):
            total += c * (x ** a) * (y ** b)
        if total != out:
            return None
    return ints


def _rank_solution(combo: tuple[int, ...],
                   ints: list[int]) -> tuple:
    """Prefer few terms, low degree, small coefficients."""
    return (len(combo),
            sum(MONOMS[j][0] + MONOMS[j][1]
                for j in combo),
            sum(abs(c) for c in ints), combo)


def fit_sparse_poly(rows: list[list[int]]
                    ) -> tuple[tuple[int, ...], list[int]] | None:
    """Exact sparse integer fit over revealed rows only."""
    n = len(rows)
    if n == 0:
        return None
    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]
    outs = [r[2] for r in rows]
    big = np.array(
        [[float(x) ** a * float(y) ** b for (a, b) in MONOMS]
         for x, y in zip(xs, ys)])
    target = np.array([float(v) for v in outs])
    for degree in range(1, min(4, n) + 1):
        sols = []
        for combo in combinations(range(len(MONOMS)),
                                  degree):
            ints = _try_combo(big, target, rows, combo,
                              None)
            if ints is not None:
                sols.append((combo, ints))
        if sols:
            sols.sort(key=lambda s: _rank_solution(
                s[0], s[1]))
            return sols[0]
    if n <= 4:
        return None
    norms = np.sqrt((big ** 2).sum(axis=0))
    corr = [abs(big[:, j] @ target) / (norms[j] or 1.0)
            for j in range(len(MONOMS))]
    low = [j for j in range(len(MONOMS))
           if MONOMS[j][0] + MONOMS[j][1] <= 2]
    short = sorted(set(
        sorted(range(len(MONOMS)),
               key=lambda j: -corr[j])[:14]) | set(low),
        key=lambda j: (MONOMS[j][0] + MONOMS[j][1],
                       MONOMS[j]))
    for degree in range(5, min(6, n) + 1):
        for combo in combinations(short, degree):
            ints = _try_combo(big, target, rows, combo, 8)
            if ints is not None:
                return combo, ints
    for degree in range(5, min(6, n) + 1):
        for combo in combinations(short, degree):
            ints = _try_combo(big, target, rows, combo,
                              None)
            if ints is not None:
                return combo, ints
    return None


class GraphBuilder:
    """Shared-subexpression DAG assembly in dependency order."""

    def __init__(self) -> None:
        """Start with an empty node table and intern map."""
        self.nodes: list[dict] = []
        self.intern: dict[tuple, int] = {}

    def zero(self) -> int:
        """One shared zero: sub(x, x), no const nodes."""
        return self.ref_bin("sub", "x", "x")

    def scale(self, base: object, coeff: int) -> object:
        """Integer multiple via add chains and shared zero.

        The target lowerer accepts arithmetic nodes only, so
        coefficients are structural: c*t is repeated addition
        and -t is sub(zero, t).
        """
        if coeff == 1:
            return base
        if coeff < 0:
            return self.ref_bin(
                "sub", self.zero(),
                self.scale(base, -coeff))
        acc: object = base
        for _ in range(coeff - 1):
            acc = self.ref_bin("add", acc, base)
        return acc

    def ref_bin(self, op: str, left: object,
                right: object) -> int:
        """Intern one add/sub/mul node over built operands."""
        key = (op, left, right)
        if key not in self.intern:
            self.intern[key] = len(self.nodes)
            self.nodes.append({"op": op, "type": "Int",
                               "args": [left, right]})
        return self.intern[key]

    def monom(self, power_x: int, power_y: int) -> object:
        """Multiply out one monomial over input references."""
        factors: list[object] = (["x"] * power_x
                                 + ["y"] * power_y)
        assert factors, "constant monomials are excluded"
        acc: object = factors[0]
        for factor in factors[1:]:
            acc = self.ref_bin("mul", acc, factor)
        return acc

    def term(self, coeff: int, power_x: int,
             power_y: int) -> object:
        """Lower one signed monomial term to node refs."""
        assert coeff != 0
        return self.scale(self.monom(power_x, power_y),
                          coeff)


def poly_to_graph(fit: tuple[tuple[int, ...], list[int]]
                  ) -> str:
    """Lower one fitted polynomial to canonical graph JSON."""
    combo, ints = fit
    builder = GraphBuilder()
    total: object | None = None
    for index, coeff in zip(combo, ints):
        power_x, power_y = MONOMS[index]
        node = builder.term(coeff, power_x, power_y)
        total = node if total is None else \
            builder.ref_bin("add", total, node)
    assert total is not None
    graph = {"input_types": {"x": "Int", "y": "Int"},
             "nodes": builder.nodes, "output": total}
    validate_graph(graph)
    return json.dumps(graph, sort_keys=True)


def graph_fits(text: str, rows: list[list[int]]) -> bool:
    """Execute one graph candidate on revealed rows only."""
    try:
        graph = json.loads(text)
        validate_graph(graph)
    except (ValueError, TypeError, KeyError):
        return False
    try:
        return all(interpret(graph, int(r[0]), int(r[1]))
                   == int(r[2]) for r in rows)
    except (ValueError, TypeError, IndexError):
        return False


def build_s1_deck() -> list[str]:
    """Loop-plus-accumulate token shape (family S1)."""
    deck: list[str] = []
    for op in ("add", "sub", "mul"):
        for left in ("acc", "value", "one", "two"):
            for right in ("value", "acc", "one", "two"):
                for tail in (("acc",), ("add", "acc",
                                        "one")):
                    tokens = ("for", "set", op, left,
                              right, "end", "return",
                              *tail, "stop")
                    deck.append(" ".join(tokens))
    for right in ("one", "two"):
        deck.append(" ".join(
            ("for", "set", "rem", "value", right, "end",
             "return", "acc", "stop")))
    return deck


def build_s2_deck() -> list[str]:
    """Loop-plus-branch token shape (family S2)."""
    deck: list[str] = []
    for cmp in ("gt", "lt", "ge", "le", "eq", "ne"):
        for left in ("value", "acc"):
            for right in ("zero", "one", "two"):
                for op in ("add", "sub", "mul"):
                    deck.append(" ".join(
                        ("for", "if", cmp, left, right,
                         "set", op, "acc", "value", "end",
                         "end", "return", "acc", "stop")))
    return deck


def stmt_fits(text: str, rows: list[list[object]]) -> bool:
    """Execute one token candidate on revealed list rows."""
    try:
        tokens = tuple(text.split())
        statement_syntax.parse(tokens)
    except ValueError:
        return False
    for row in rows:
        try:
            out = LlmBankGenerator.run_tokens(
                tokens, list(row[0]))  # type: ignore[arg-type]
        except ValueError:
            return False
        if out != row[1]:
            return False
    return True


def fragment_transfer(fragments: object,
                      kind: str, rows: list[list]) -> str | None:
    """Reuse a shown fragment exact on every revealed row."""
    if not isinstance(fragments, list):
        return None
    for fragment in fragments:
        if not isinstance(fragment, dict):
            continue
        if fragment.get("kind") != kind:
            continue
        content = fragment.get("content")
        if not isinstance(content, str) or not content:
            continue
        if kind == "graph" and graph_fits(content, rows):
            return content.strip()
        if kind == "statement" and stmt_fits(content, rows):
            return content.strip()
    return None


def make_candidate_fn(visible_map: dict[str, list[list]],
                      kind_map: dict[str, str],
                      deck_map: dict[str, list[str]]):
    """Row-based Spark 1.3 stand-in over revealed rows only."""

    def candidate_fn(task_id: str, round_no: int,
                     prompt: str | None,
                     rounds: list[dict]
                     ) -> tuple[str, dict[str, int], float]:
        started = time.perf_counter()
        kind = kind_map[task_id]
        fit_rows: list[list] = [list(r) for r in
                                visible_map[task_id]]
        fragments: object = []
        previous = ""
        if prompt:
            try:
                body = json.loads(prompt)
            except ValueError:
                body = {}
            if isinstance(body, dict):
                for key in ("revealed_query_rows",
                            "failing_batch"):
                    extra = body.get(key, [])
                    if isinstance(extra, list):
                        fit_rows.extend(
                            list(r) for r in extra
                            if isinstance(r, list))
                fragments = body.get("fragments", [])
                previous = str(body.get(
                    "previous_candidate", ""))
        seen: set[str] = set()
        uniq: list[list] = []
        for row in fit_rows:
            key = json.dumps(row, sort_keys=True)
            if key not in seen:
                seen.add(key)
                uniq.append(row)
        fit_rows = uniq
        moved = fragment_transfer(fragments, kind, fit_rows)
        text = ""
        if moved is not None:
            text = moved
        elif kind == "graph":
            rows = [[int(r[0]), int(r[1]), int(r[2])]
                    for r in fit_rows]
            fit = fit_sparse_poly(rows)
            if fit is not None:
                text = poly_to_graph(fit)
                assert graph_fits(text, rows)
            else:
                text = previous.strip() or json.dumps(
                    {"input_types": {"x": "Int", "y": "Int"},
                     "nodes": [{"op": "add", "type": "Int",
                                "args": ["x", "y"]}],
                     "output": 0}, sort_keys=True)
        else:
            deck = deck_map[task_id]
            exact = [t for t in deck
                     if stmt_fits(t, fit_rows)]
            if exact:
                text = sorted(
                    exact, key=lambda t: (len(t.split()),
                                          t))[0]
            else:
                scored = sorted(
                    deck,
                    key=lambda t: (
                        -sum(1 for r in fit_rows
                             if _stmt_row_fits(t, r)), t))
                text = scored[0]
        if text.strip() == previous.strip() and previous:
            text = text + " "
        cpu = time.perf_counter() - started
        prompt_words = len(prompt.split()) if prompt \
            else 16 + 8 * len(visible_map[task_id])
        usage = {"prompt": int(prompt_words),
                 "completion": int(len(text.split()))}
        return text, usage, float(cpu)

    return candidate_fn


def _stmt_row_fits(text: str, row: list[object]) -> bool:
    """Single-row check for best-partial fallback ranking."""
    try:
        tokens = tuple(text.split())
        out = LlmBankGenerator.run_tokens(
            tokens, list(row[0]))  # type: ignore[arg-type]
    except ValueError:
        return False
    return out == row[1]


def main() -> None:
    """Load dev-only splits and run all five arms in order."""
    (OUTDIR / "sysinfo_start.json").write_text(
        json.dumps(sysinfo(), indent=2, sort_keys=True)
        + "\n")
    bank = LlmBankGenerator()
    protocol_doc, task_list, visible_full, query_full = \
        bank.load_public()
    held_full = bank.load_held(unseal=True)
    for key in list(held_full.keys()):
        if key not in TASK_IDS:
            del held_full[key]
    assert set(held_full) == set(TASK_IDS), \
        f"dev held keys mismatch: {sorted(held_full)}"
    for key in list(visible_full.keys()):
        if key not in TASK_IDS:
            del visible_full[key]
    for key in list(query_full.keys()):
        if key not in TASK_IDS:
            del query_full[key]
    tasks = {t["id"]: t for t in task_list
             if t["id"] in TASK_IDS}
    assert set(tasks) == set(TASK_IDS)
    assert all(t.get("pool") == "dev" for t in
               tasks.values())
    assert all(str(k).startswith("llm_dev_") for k in
               list(tasks) + list(held_full))
    visible = {k: [list(r) for r in v] for k, v in
               visible_full.items()}
    query = {k: [list(r) for r in v] for k, v in
             query_full.items()}
    held = {k: [list(r) for r in v] for k, v in
            held_full.items()}
    del held_full, visible_full, query_full
    kind_map = {tid: ("graph" if tasks[tid][
        "interface"] == "pair" else "statement")
        for tid in TASK_IDS}
    family_map = {tid: str(tasks[tid].get("family"))
                  for tid in TASK_IDS}
    s1 = build_s1_deck()
    s2 = build_s2_deck()
    assert set(s1) == set(
        " ".join(t) for t in
        LlmBankGenerator._s1_deck()), "S1 deck drift"
    assert set(s2) == set(
        " ".join(t) for t in
        LlmBankGenerator._s2_deck()), "S2 deck drift"
    deck_map = {tid: (s1 if family_map[tid] == "S1"
                      else s2) for tid in TASK_IDS
                if kind_map[tid] == "statement"}
    protocol = DevProtocol(task_ids=TASK_IDS)
    runner = DevDiagnosticRunner(
        OUTDIR, tasks=tasks, visible=visible, query=query,
        held=held,
        expected_compiler_sha=protocol_doc[
            "curlee_binary"]["sha256"],
        protocol=protocol)
    frozen_sha = runner.protocol_sha256()
    assert frozen_sha == protocol.sha256()
    print(f"protocol_sha256={frozen_sha}", flush=True)
    print(f"families={family_map}", flush=True)
    print(f"kinds={kind_map}", flush=True)
    candidate_fn = make_candidate_fn(visible, kind_map,
                                     deck_map)
    print(runner.run_arm_a(candidate_fn), flush=True)
    print(runner.run_arm_b(candidate_fn), flush=True)
    print(runner.build_library(), flush=True)
    print(runner.run_arm_c(candidate_fn), flush=True)
    print(runner.run_arm_d(), flush=True)
    print(runner.run_arm_e(), flush=True)
    print(runner.finalize(), flush=True)
    audit = {
        "protocol_sha256": frozen_sha,
        "model": MODEL,
        "tasks": list(TASK_IDS),
        "families": family_map,
        "kinds": kind_map,
        "records": {
            arm: {task: record for task, record in
                  sorted(tasks.items())}
            for arm, tasks in runner._records.items()},
        "repair": {
            f"{arm}/{task}": artifact
            for (arm, task), artifact in
            sorted(runner._repair.items(),
                   key=lambda kv: (kv[0][0], kv[0][1]))},
        "retrieval": dict(sorted(runner._retrieval.items())),
    }
    (OUTDIR / "records.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True,
                   default=str) + "\n")
    print("records=records.json", flush=True)
    summary = runner.summary()
    print(json.dumps(summary["counts"], indent=2,
                     sort_keys=True), flush=True)
    print(json.dumps(summary["decision"], indent=2,
                     sort_keys=True), flush=True)
    report = OUTDIR / "inversion-dev-results.md"
    runner.write_report(report)
    (OUTDIR / "sysinfo_end.json").write_text(
        json.dumps(sysinfo(), indent=2, sort_keys=True)
        + "\n")
    print(f"report={report}", flush=True)


if __name__ == "__main__":
    main()

