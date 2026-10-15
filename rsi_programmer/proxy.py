#!/usr/bin/env python3
"""Few-epoch few-seed proxy screening for transfer gain.

The proxy trains each candidate seed for a few epochs on a W1
subset and scores it on W1 holdback tasks, predicting the
full-eval transfer gain (full W1 training, novel-task eval).
Fidelity is the measured rank agreement between the proxy and
full rankings over historical candidates. The proxy judges
nothing until proven: select names a winner only when the
measured agreement clears its threshold on enough pairs.
"""
from __future__ import annotations

import json
import time
from typing import Any, cast

from rsi_programmer.evolve import guided_search
from rsi_programmer.policy import Group, Rows, SearchPolicy
from rsi_programmer.policy import Vals
from rsi_programmer.probe import enumerate_programs
from rsi_programmer.traces import TraceRecorder

Candidate = dict[str, Any]
Scores = dict[str, float]

PROXY_EPOCHS = 2
PROXY_SEEDS = (0, 1)
SUBSET_TASKS = 2
FULL_EPOCHS = 50
BUDGET = 10 ** 9
AGREE_THRESHOLD = 0.8
MIN_PAIRS = 3
REPORT_BUDGET = 4096


def _parts(task: dict[str, Any]) -> tuple[Rows, Vals, int]:
    """Train rows, target, and op bound for one task."""
    rows = tuple((c[0], c[1]) for c in task["train"])
    target = tuple(c[2] for c in task["train"])
    return rows, target, int(task.get("max_tree_ops", 4))


def _plain_tried(rows: Rows, target: Vals,
                 bound: int) -> int:
    """Plain enumerator tried; family tasks solve by design."""
    expr, tried, _ = enumerate_programs(rows, target, bound)
    assert expr is not None
    return tried


def _gain(solved: bool, tried: int, plain: int) -> float:
    """Relative tried saving; an unsolved task scores -1."""
    if not solved:
        return -1.0
    return (plain - tried) / (plain or 1)


def _task_gain(policy: SearchPolicy,
               task: dict[str, Any]) -> float:
    """Guided tried saving over the enumerator on one task."""
    rows, target, bound = _parts(task)
    out = guided_search(policy, rows, target, bound, BUDGET,
                        str(task["id"]))
    return _gain(out["solved"], out["tried"],
                 _plain_tried(rows, target, bound))


def _mean_gain(policy: SearchPolicy,
               tasks: list[dict[str, Any]]) -> float:
    """Mean tried saving across a task family slice."""
    gains = [_task_gain(policy, task) for task in tasks]
    return sum(gains) / len(gains)


def _trained(seed: int, groups: list[Group],
             epochs: int) -> SearchPolicy:
    """One policy trained for a fixed epoch count."""
    policy = SearchPolicy(seed=seed)
    policy.fit(groups, epochs=epochs)
    return policy


def _sign(value: float) -> int:
    """Sign of a score gap; zero marks a tied pair."""
    return (value > 0.0) - (value < 0.0)


def rank_agreement(proxy: Scores,
                   full: Scores) -> dict[str, Any]:
    """Pairwise rank agreement; ties count half, stated."""
    ids = [key for key in proxy if key in full]
    concord = discord = tied = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            side_p = _sign(proxy[ids[i]] - proxy[ids[j]])
            side_f = _sign(full[ids[i]] - full[ids[j]])
            if side_p == 0 or side_f == 0:
                tied += 1
            elif side_p == side_f:
                concord += 1
            else:
                discord += 1
    total = concord + discord + tied
    agree = (concord + 0.5 * tied) / total if total else 0.0
    return {"pairs": total, "concordant": concord,
            "discordant": discord, "tied": tied,
            "agreement": round(agree, 4)}


class ProxyScreen:
    """Cheap W1-subset screening with measured fidelity."""

    def __init__(self, shadow: str, novel: str,
                 subset_tasks: int = SUBSET_TASKS,
                 epochs: int = PROXY_EPOCHS,
                 seeds: tuple[int, ...] = PROXY_SEEDS,
                 threshold: float = AGREE_THRESHOLD,
                 min_pairs: int = MIN_PAIRS) -> None:
        """Bind the W1 subset/holdback split and threshold."""
        fam = TraceRecorder(shadow).load_family()
        tasks = cast(list[dict[str, Any]], fam["tasks"])
        assert 0 < subset_tasks < len(tasks)
        self._shadow = shadow
        self._novel = novel
        self._subset = tasks[:subset_tasks]
        self._hold = tasks[subset_tasks:]
        self._epochs = epochs
        self._seeds = seeds
        self._threshold = threshold
        self._min_pairs = min_pairs

    @staticmethod
    def _task_ids(tasks: list[dict[str, Any]]) -> list[str]:
        """Task ids for one family slice."""
        return [str(task["id"]) for task in tasks]

    def _groups(self) -> list[Group]:
        """Full W1 trace groups in family order."""
        return SearchPolicy(seed=0).trace_groups(self._shadow)

    def screen(self, candidates: list[Candidate]
               ) -> dict[str, Any]:
        """Proxy score per candidate; novel tasks untouched."""
        sub = self._groups()[:len(self._subset)]
        scores: Scores = {}
        start = time.perf_counter()
        for cand in candidates:
            base = int(cand.get("seed", 0))
            gains = [_mean_gain(_trained(base + k, sub,
                                        self._epochs),
                               self._hold)
                     for k in range(len(self._seeds))]
            scores[str(cand["id"])] = round(sum(gains)
                                            / len(gains), 4)
        wall = time.perf_counter() - start
        return {"scores": scores,
                "train_tasks": self._task_ids(self._subset),
                "eval_tasks": self._task_ids(self._hold),
                "epochs": self._epochs,
                "seeds": len(self._seeds),
                "wall_s": round(wall, 3)}

    def full_eval(self, candidates: list[Candidate],
                  epochs: int = FULL_EPOCHS) -> dict[str, Any]:
        """Full W1 training, novel-task gain per candidate."""
        groups = self._groups()
        fam = TraceRecorder(self._novel).load_family()
        novel = cast(list[dict[str, Any]], fam["tasks"])
        gains: Scores = {}
        start = time.perf_counter()
        for cand in candidates:
            policy = _trained(int(cand.get("seed", 0)),
                              groups, epochs)
            gains[str(cand["id"])] = round(
                _mean_gain(policy, novel), 4)
        wall = time.perf_counter() - start
        return {"gains": gains,
                "eval_tasks": self._task_ids(novel),
                "epochs": epochs,
                "wall_s": round(wall, 3)}

    def fidelity(self, proxy: Scores,
                 full: Scores) -> dict[str, Any]:
        """Measured agreement plus the proven verdict."""
        pact = rank_agreement(proxy, full)
        pact.update({"threshold": self._threshold,
                     "min_pairs": self._min_pairs,
                     "proven": pact["agreement"] >=
                     self._threshold and pact["pairs"] >=
                     self._min_pairs})
        return pact

    def select(self, proxy: Scores,
               full: Scores) -> dict[str, Any]:
        """Winner only when the proxy has proven itself."""
        pact = self.fidelity(proxy, full)
        if not proxy or not pact["proven"]:
            return {"winner": None, "proven": False,
                    "reason": "unproven-proxy",
                    "fidelity": pact}
        top = max(proxy, key=proxy.__getitem__)
        return {"winner": top, "proven": True,
                "reason": "agreement-proven",
                "fidelity": pact}

    def report(self, candidates: list[Candidate],
               historical: Scores | None = None,
               epochs: int = FULL_EPOCHS) -> dict[str, Any]:
        """Screening, fidelity, and gated selection in one."""
        screened = self.screen(candidates)
        if historical is not None:
            full: dict[str, Any] = {"gains": dict(historical),
                                    "epochs": epochs,
                                    "historical": True}
        else:
            full = self.full_eval(candidates, epochs)
        scores = screened["scores"]
        payload: dict[str, Any] = {
            "screen": screened, "full": full,
            "fidelity": self.fidelity(scores, full["gains"]),
            "selection": self.select(scores, full["gains"])}
        size = len(json.dumps(payload, sort_keys=True).encode())
        payload["report_bytes"] = size
        total = len(json.dumps(payload, sort_keys=True).encode())
        payload["within_budget"] = total < REPORT_BUDGET
        return payload
