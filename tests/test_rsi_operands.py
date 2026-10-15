#!/usr/bin/env python3
"""Operand discrimination in SNN authorship scores (#39).

Follow-up to #38: trained scores were identical across
same-op operand variants (op preference learned, operand
identity ignored) because pairwise margin training never
pressures operand weights. This suite mines
operand-contrastive pairs (right op, wrong operands) into
clone_groups, audits that operand-signature dims carry
variance and trained weight, and re-measures rank spread
plus beam wins with identical tried accounting.

Training head is 50 epochs: the matched-epoch sweep gave
base 6 wins vs contrastive 10 with zero flips; 100 epochs
gives 13 wins with one twice_product flip, so the
flip-free config ships and the sweep ships alongside.
"""
from __future__ import annotations
import statistics
import time
import unittest
from pathlib import Path
from typing import Any
from rsi_programmer.actions import ActionSpace
from rsi_programmer.author import OPDIMS, TaskRow, beam_search
from rsi_programmer.author import contrastive_groups, emit_edit
from rsi_programmer.author import refine_author, train_author
from rsi_programmer.evolve import guided_search
from rsi_programmer.policy import SearchPolicy
from rsi_programmer.traces import TraceRecorder
from rsi_programmer.probe import validate_graph
FIXTURE = (Path(__file__).resolve().parent.parent
           / "fixtures" / "rsi_shadow_tasks.json")
SEEDS = (0, 1, 2)
EPOCHS = 50
BUDGET = 10 ** 9
WIDTHS = (3, 6)
BASE_WON = {(0, 3, "twice_product"), (0, 6, "diff_squares"),
            (0, 6, "twice_product"), (1, 3, "twice_product"),
            (1, 6, "diff_squares"), (1, 6, "twice_product"),
            (2, 3, "twice_product"), (2, 6, "diff_squares"),
            (2, 6, "twice_product")}
def _load_tasks() -> list[TaskRow]:
    """Frozen W1 tasks as search-ready tuples."""
    rec = TraceRecorder(str(FIXTURE))
    out: list[TaskRow] = []
    for task in rec.load_family()["tasks"]:
        rows = tuple((c[0], c[1]) for c in task["train"])
        target = tuple(c[2] for c in task["train"])
        out.append((str(task["id"]), rows, target,
                    int(task.get("max_tree_ops", 4))))
    return out
def _tag(row: dict[str, Any]) -> str:
    """Win when beam solves with fewer tried than rank."""
    if row["bm_solved"] and (not row["rank_solved"]
                             or row["bm_tried"] < row["rank_tried"]):
        return "win"
    if row["rank_solved"] and (not row["bm_solved"]
                               or row["bm_tried"] > row["rank_tried"]):
        return "loss"
    return "tie"
def _row(seed: int, width: int, tid: str, ranked: dict[str, Any],
         beam: dict[str, Any]) -> dict[str, Any]:
    """One beam-vs-rank comparison row for the report."""
    return {"seed": seed, "width": width, "task": tid,
            "rank_tried": ranked["tried"], "rank_solved": ranked["solved"],
            "bm_tried": beam["tried"], "bm_solved": beam["solved"],
            "bm_path": beam["path"], "bm_log": beam["rejected"],
            "bm_graph": beam["graph"]}
Stats = tuple[float, float, int]
def _run_seed(seed: int, tasks: list[TaskRow]
              ) -> tuple[list[dict[str, Any]], Stats, Stats]:
    """Train with contrast pairs; read spread before search reuse."""
    policy = SearchPolicy(seed=seed)
    count = train_author(policy, str(FIXTURE), epochs=EPOCHS)
    print("seed=%d contrast triples=%d" % (seed, count))
    acc = _contrastive_stats(policy)
    audit = _audit(policy)
    rows: list[dict[str, Any]] = []
    for width in WIDTHS:
        weights, _ = refine_author(policy, tasks, seed=seed, topk=width)
        for tid, trows, target, bound in tasks:
            ranked = guided_search(policy, trows, target, bound,
                                   BUDGET, tid)
            beam = beam_search(policy, trows, target, bound, BUDGET,
                               tid, width, weights, fallback=False)
            rows.append(_row(seed, width, tid, ranked, beam))
    return rows, acc, audit
def _contrastive_stats(policy: SearchPolicy) -> tuple[float, float, int]:
    """Pooled top-1 accuracy vs chance over mined pairs."""
    accs, chances = [], []
    groups = contrastive_groups(str(FIXTURE))
    for group in groups:
        scored = policy.scores([f for f, _ in group])
        top = max(range(len(group)), key=lambda i: scored[i])
        accs.append(1.0 if group[top][1] == 1 else 0.0)
        chances.append(sum(1 for _, y in group if y == 1) / len(group))
    return sum(accs) / len(accs), sum(chances) / len(chances), len(groups)
def _audit(policy: SearchPolicy) -> tuple[float, float, int]:
    """Min operand-dim variance plus mean-ablation dependence."""
    pool = [f for g in contrastive_groups(str(FIXTURE)) for f, _ in g]
    var = [statistics.pvariance([f[d] for f in pool]) for d in OPDIMS]
    scored = policy.scores(pool)
    means = [sum(f[d] for f in pool) / len(pool) for d in range(len(pool[0]))]
    flat = [[means[d] if d in OPDIMS else v for d, v in enumerate(f)]
            for f in pool]
    muted = policy.scores(flat)
    depend = sum(abs(a - b) for a, b in zip(scored, muted)) / len(pool)
    return min(var), depend, len(pool)
class OperandTests(unittest.TestCase):
    rows: list[dict[str, Any]] = []
    @classmethod
    def setUpClass(cls) -> None:
        """Train three heads with contrast pairs; compare."""
        cls.start = time.perf_counter()
        cls.tasks = _load_tasks()
        fam = TraceRecorder(str(FIXTURE)).load_family()["tasks"]
        cls.by_id = {t["id"]: t for t in fam}
        cls.rows = []
        cls.acc: dict[int, tuple[float, float, int]] = {}
        cls.audit: dict[int, tuple[float, float, int]] = {}
        for seed in SEEDS:
            rows, acc, audit = _run_seed(seed, cls.tasks)
            cls.rows += rows
            cls.acc[seed] = acc
            cls.audit[seed] = audit
        cls.wall_s = time.perf_counter() - cls.start
    def test_contrastive_top1_above_chance(self) -> None:
        """Rank spread: top-1 beats chance on frozen pairs."""
        for seed in SEEDS:
            acc, chance, n = self.acc[seed]
            print("seed=%d contrast acc=%.3f chance=%.3f n=%d"
                  % (seed, acc, chance, n))
            self.assertGreater(acc, chance)
    def test_feature_audit_variance_nonzero(self) -> None:
        """Operand-signature dims vary over contrast pairs."""
        for seed in SEEDS:
            var, _, n = self.audit[seed]
            print("seed=%d min operand-dim var=%.5f n=%d"
                  % (seed, var, n))
            self.assertGreater(var, 0.0)
    def test_feature_audit_readout_dependence_nonzero(self) -> None:
        """Trained readout depends on operand-signature dims."""
        for seed in SEEDS:
            _, depend, n = self.audit[seed]
            print("seed=%d ablation dependence=%.4f n=%d"
                  % (seed, depend, n))
            self.assertGreater(depend, 0.0)
    def test_beam_wins_above_baseline_without_regressions(self) -> None:
        """Wins strictly above 9/24; every old win still wins."""
        tags = [_tag(r) for r in self.rows]
        for row, tag in zip(self.rows, tags):
            print("s%d w%d %-12s rank=%4d/%s bm=%3d/%s %s"
                  % (row["seed"], row["width"], row["task"],
                     row["rank_tried"], row["rank_solved"],
                     row["bm_tried"], row["bm_solved"], tag))
        wins = tags.count("win")
        flips = sorted((r["seed"], r["width"], r["task"])
                       for r, tag in zip(self.rows, tags)
                       if (r["seed"], r["width"], r["task"]) in BASE_WON
                       and tag != "win")
        print("wins=%d/24 flips=%s wall=%.1fs" % (wins, flips, self.wall_s))
        self.assertEqual(len(tags), 24)
        self.assertEqual({r["bm_path"] for r in self.rows}, {"beam"})
        self.assertGreater(wins, 9)
        self.assertEqual(flips, [])
    def test_beam_accounting_identical(self) -> None:
        """One scored triple is one tried, exactly like rank."""
        for row in self.rows:
            self.assertEqual(row["bm_tried"], len(row["bm_log"]))
    def test_zero_bad_executions(self) -> None:
        """Every edit verdict-carrying; solutions validate."""
        rec = TraceRecorder(str(FIXTURE))
        for row in self.rows:
            log = row["bm_log"]
            self.assertGreater(len(log), 0)
            for entry in log:
                self.assertEqual(set(entry),
                                 {"op", "signature", "task", "verdict"})
                if entry["verdict"] not in ("", "signature-mismatch"):
                    self.assertTrue(entry["verdict"].startswith(
                        "validator:"))
            blanks = [e for e in log if e["verdict"] == ""]
            self.assertEqual(len(blanks), 1 if row["bm_solved"] else 0)
            if not row["bm_solved"]:
                continue
            validate_graph(row["bm_graph"])
            self.assertTrue(rec.oracle_holds(self.by_id[row["task"]],
                                             row["bm_graph"]))
    def test_malformed_edits_rejected_pre_execution(self) -> None:
        """Validator gate rejects malformed edits pre-execution."""
        space = ActionSpace()
        empty: dict[str, Any] = {"input_types": {"x": "Int", "y": "Int"},
                                 "nodes": [], "output": "x"}
        bad = [{"kind": "apply", "op": "div", "left": "x", "right": "y"},
               {"kind": "apply", "op": "add", "left": 9, "right": "y"},
               {"kind": "input", "ref": "z"}, {"kind": "jump", "ref": "x"},
               {"kind": "apply", "op": "add", "left": "x"},
               {"kind": "apply", "op": "add", "left": 0, "right": "y"}]
        for action in bad:
            graph, rec, ok = emit_edit(space, dict(empty), action, "probe", ())
            self.assertFalse(ok)
            self.assertTrue(rec["verdict"].startswith("validator:"))
            self.assertEqual(graph, empty)
        good = {"kind": "apply", "op": "add", "left": "x", "right": "y"}
        graph, rec, ok = emit_edit(space, dict(empty), good,
                                   "probe", (1, 2))
        self.assertTrue(ok)
        validate_graph(graph)
    def test_runs_within_cpu_budget(self) -> None:
        """Whole module stays under 600 wall seconds."""
        print("operands wall=%.1fs" % self.wall_s)
        self.assertLess(self.wall_s, 600.0)
if __name__ == "__main__":
    unittest.main()
