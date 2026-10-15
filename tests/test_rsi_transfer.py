#!/usr/bin/env python3
"""Novel-task transfer with a policy frozen on W1 (#40).

Three brand-new closed-form rules (fixtures/rsi_novel_tasks.json)
never enter training: the policy trains exactly once on the W1
shadow fixture, then beam, ranked, and plain enumerator arms run
on each novel task with identical tried accounting.
"""
from __future__ import annotations
import hashlib
import json
import statistics
import time
import unittest
from pathlib import Path
from typing import Any, cast
from rsi_programmer.author import TaskRow, beam_search, refine_author
from rsi_programmer.author import train_author
from rsi_programmer.evolve import guided_search
from rsi_programmer.policy import Rows, SearchPolicy, Vals, encode
from rsi_programmer.policy import layer_triples
from rsi_programmer.traces import TraceRecorder
from rsi_programmer.probe import enumerate_programs
from rsi_programmer.probe import interpret, lower_to_dag
from rsi_programmer.probe import signature, terminals
from rsi_programmer.probe import validate_graph
ROOT = Path(__file__).resolve().parent.parent
W1 = ROOT / "fixtures" / "rsi_shadow_tasks.json"
NOVEL = ROOT / "fixtures" / "rsi_novel_tasks.json"
SEED = 0
EPOCHS = 50
BUDGET = 10 ** 9
WIDTH = 3
TRAIN_LOG: list[Any] = []


def _fam(path: str) -> list[dict[str, Any]]:
    """Raw task dicts from a family fixture file."""
    fam = TraceRecorder(path).load_family()
    return cast(list[dict[str, Any]], fam["tasks"])


def _as_rows(fam: list[dict[str, Any]]) -> list[TaskRow]:
    """Family dicts to (id, rows, target, bound) tuples."""
    out: list[TaskRow] = []
    for task in fam:
        rows = tuple((c[0], c[1]) for c in task["train"])
        out.append((str(task["id"]), rows,
                    tuple(c[2] for c in task["train"]),
                    int(task.get("max_tree_ops", 4))))
    return out


def _ids(path: str) -> set[str]:
    """Task ids frozen in one family fixture file."""
    return {str(t["id"]) for t in _fam(path)}


def _guarded_train(policy: SearchPolicy, fixture: str,
                   epochs: int) -> int:
    """Train only on W1; fail if a novel id is present."""
    assert fixture == str(W1), "training fixture must be W1"
    ids = {str(t["id"]) for t in _fam(fixture)}
    assert not ids & _ids(str(NOVEL)), "novel leak in train"
    TRAIN_LOG.append(("train_author", fixture, epochs))
    return train_author(policy, fixture, epochs=epochs)


def _guarded_refine(policy: SearchPolicy,
                    tasks: list[TaskRow], seed: int,
                    topk: int) -> list[float]:
    """Refine only on W1 rows; fail on novel task ids."""
    ids = {t[0] for t in tasks}
    assert not ids & _ids(str(NOVEL)), "novel leak in refine"
    assert ids <= _ids(str(W1)), "refine must use W1 tasks"
    TRAIN_LOG.append(("refine_author", sorted(ids), seed))
    weights, _ = refine_author(policy, tasks, seed=seed,
                               topk=topk)
    return weights


def _weights_hash(policy: SearchPolicy) -> str:
    """Canonical sha256 over the frozen policy weights."""
    snap = policy.snapshot()
    text = json.dumps(snap["weights"], sort_keys=True, separators=(",", ":"))

    return hashlib.sha256(text.encode()).hexdigest()


def _plain(rows: Rows, target: Vals,
           bound: int) -> dict[str, Any]:
    """Plain probe enumerator arm with tried accounting."""
    expr, tried, _ = enumerate_programs(rows, target, bound)
    graph = lower_to_dag(expr) if expr is not None else None
    return {"solved": expr is not None, "tried": tried,
            "graph": graph,
            "root": expr.op if expr is not None else ""}


def _run_arms(policy: SearchPolicy, weights: list[float],
              task: TaskRow) -> dict[str, Any]:
    """Ranked, beam, and plain arms with identical tried."""
    tid, rows, target, bound = task
    ranked = guided_search(policy, rows, target, bound,
                           BUDGET, tid)
    beam = beam_search(policy, rows, target, bound, BUDGET,
                       tid, WIDTH, weights, fallback=False)
    return {"task": tid, "rank": ranked, "beam": beam,
            "plain": _plain(rows, target, bound)}


def _oracle_ok(task: dict[str, Any],
               graph: dict[str, Any]) -> bool:
    """Frozen train+held expectations must all match."""
    cases = list(task["train"]) + list(task["held_out"])
    return all(interpret(graph, c[0], c[1]) == c[2]
               for c in cases)


def _wins(row: dict[str, Any]) -> bool:
    """A guided arm solves with fewer tried than enumerator."""
    plain = row["plain"]["tried"]
    rank, beam = row["rank"], row["beam"]
    return (rank["solved"] and rank["tried"] < plain) or \
        (beam["solved"] and beam["tried"] < plain)


def _mode_line(row: dict[str, Any]) -> str:
    """One per-arm solve/tried comparison line."""
    r, b, p = row["rank"], row["beam"], row["plain"]
    return "%-13s rank=%5d/%s beam=%5d/%s enum=%5d/%s" % (
        row["task"], r["tried"], r["solved"], b["tried"],
        b["solved"], p["tried"], p["solved"])


def _spread(policy: SearchPolicy, rows: Rows,
            target: Vals, bound: int) -> tuple[float, str]:
    """Score stdev and top op on the first choice layer."""
    triples = layer_triples({0: list(terminals())}, 1)
    feats = [encode(op, signature(left, rows),
                    signature(right, rows), target, 0, 0,
                    1, bound, i + 1, len(triples))
             for i, (op, left, right) in enumerate(triples)]
    scores = policy.scores(feats)
    top = triples[max(range(len(scores)),
                      key=scores.__getitem__)][0]
    return (statistics.pstdev(scores), top)


def _diagnose(spread: float, top: str, root: str) -> str:
    """Name the measured no-transfer failure mode."""
    if spread < 1e-6:
        return "scores-flat on novel signatures"
    if top != root:
        return "wrong-op-preference top=%s root=%s" % (top,
                                                       root)
    return "operand-blindness on unseen signatures"


class TransferTests(unittest.TestCase):
    """Frozen W1 policy evaluated on unseen novel rules."""

    rows: list[dict[str, Any]] = []

    @classmethod
    def setUpClass(cls) -> None:
        """Train once on W1; run three arms per novel task."""
        cls.start = time.perf_counter()
        w1raw = _fam(str(W1))
        cls.w1ids = {str(t["id"]) for t in w1raw}
        cls.w1rules = {str(t["task"]) for t in w1raw}
        cls.w1vecs = {str(t["id"]): tuple(c[2] for c in
                                          t["train"])
                      for t in w1raw}
        w1 = _as_rows(w1raw)
        cls.novel = _as_rows(_fam(str(NOVEL)))
        cls.raw = {str(t["id"]): t for t in _fam(str(NOVEL))}
        cls.policy = SearchPolicy(seed=SEED)
        _guarded_train(cls.policy, str(W1), EPOCHS)
        cls.weights = _guarded_refine(cls.policy, w1, SEED,
                                      WIDTH)
        cls.frozen = _weights_hash(cls.policy)
        cls.rows = [_run_arms(cls.policy, cls.weights, t)
                    for t in cls.novel]
        cls.wall_s = time.perf_counter() - cls.start

    def test_novelty_unseen(self) -> None:
        """Novel rules/vectors match no W1 fixture task."""
        for tid, task in self.raw.items():
            self.assertNotIn(str(task["task"]), self.w1rules)
            self.assertNotIn(tid, self.w1ids)
            vec = tuple(c[2] for c in task["train"])
            for wid, wvec in self.w1vecs.items():
                self.assertNotEqual(vec, wvec, "%s=%s" %
                                    (tid, wid))

    def test_frozen_weights_no_leak(self) -> None:
        """Exactly one W1 train; hash printed and stable."""
        print("frozen weights sha256=%s" % self.frozen)
        kinds = [e[0] for e in TRAIN_LOG]
        self.assertEqual(kinds, ["train_author", "refine_author"])
        self.assertEqual(TRAIN_LOG[0][1], str(W1))
        self.assertFalse(set(TRAIN_LOG[1][1]) & set(self.raw))
        self.assertEqual(len(self.frozen), 64)

    def _check_oracles(self, row: dict[str, Any]) -> None:
        """Solved graphs validate and meet frozen oracle."""
        for arm in (row["rank"], row["beam"], row["plain"]):
            if not arm["solved"]:
                continue
            validate_graph(arm["graph"])
            self.assertTrue(_oracle_ok(self.raw[row["task"]],
                                       arm["graph"]))
        self.assertEqual(row["beam"]["path"], "beam")
        self.assertEqual(row["beam"]["tried"],
                         len(row["beam"]["rejected"]))
        self.assertIn(row["rank"]["tried"] -
                      len(row["rank"]["rejected"]), (0, 1))

    def _failure(self) -> str:
        """Name the measured no-transfer failure mode."""
        _, rows, target, bound = self.novel[0]
        spread, top = _spread(self.policy, rows, target,
                              bound)
        root = str(_plain(rows, target, bound)["root"])
        mode = _diagnose(spread, top, root)
        return "%s spread=%.6f top=%s root=%s" % (mode,
                                                  spread,
                                                  top, root)

    def test_transfer_or_honest_no_transfer(self) -> None:
        """Solve/tried per arm; win or named failure mode."""
        for row in self.rows:
            print(_mode_line(row))
            self._check_oracles(row)
        wins = sum(1 for row in self.rows if _wins(row))
        print("frozen=%s wins=%d" % (self.frozen, wins))
        if not wins:
            print("NO-TRANSFER: %s" % self._failure())
        self.assertTrue(all(r["plain"]["solved"]
                            for r in self.rows))

    def test_runs_within_cpu_budget(self) -> None:
        """Whole module stays under 600 wall seconds."""
        print("transfer wall=%.1fs" % self.wall_s)
        self.assertLess(self.wall_s, 600.0)


if __name__ == "__main__":
    unittest.main()
