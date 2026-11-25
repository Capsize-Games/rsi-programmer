"""Batch runner: task lists plus arm sources to verdict aggregates.

owns one :class:`LlmJudge` and, for every task and arm, clusters
that arm's canned candidates by visible-row (op, sig), judges only
the cheapest representative of each cluster, keeps the cheapest
accepted representative (or the cheapest rejection), and embeds
one deliberately-wrong control candidate per task whose rejection
is asserted. The returned summary is JSON-safe and mirrors the
cost-ledger artifact shape (``cost_definition``,
``excluded_costs``, ``command_costs``, ``measured_total_*``)
with an additional ``judge_phases`` breakdown over the six judge
phases. Counterexample row values never enter the summary; final
acceptance stays a per-candidate :meth:`LlmJudge.judge_final`
call outside this repair-facing loop.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from rsi_programmer.cost_ledger import COST_DEFINITION
from rsi_programmer.llm_judge import JUDGE_PHASES, RANGE_NOTE
from rsi_programmer.llm_judge import LlmJudge
from rsi_programmer.llm_judge import _public_verdict

ArmFn = Callable[[Mapping[str, Any]], list[Mapping[str, Any]]]


class JudgeBatchRunner:
    """Judge task lists across arm sources with control asserts."""

    def __init__(self, outdir: Path, *,
                 tasks: Mapping[str, Mapping[str, Any]],
                 visible: Mapping[str, list[list[Any]]],
                 query: Mapping[str, list[list[Any]]],
                 arms: Mapping[str, ArmFn],
                 expected_compiler_sha: str,
                 bank_id: str = "llm_bank_v1") -> None:
        """Bind output, row splits, arms and the compiler pin."""
        if not arms:
            raise ValueError("at least one arm is required")
        self._outdir = outdir
        self._outdir.mkdir(parents=True, exist_ok=True)
        self._tasks = {key: dict(value)
                       for key, value in tasks.items()}
        self._visible = visible
        self._query = query
        self._arms = dict(arms)
        self._bank_id = bank_id
        self._judge = LlmJudge(
            outdir / "judge-work", tasks=tasks, visible=visible,
            query=query, held=None,
            expected_compiler_sha=expected_compiler_sha)

    @property
    def judge(self) -> LlmJudge:
        """The repair-facing judge owned by this runner."""
        return self._judge

    @staticmethod
    def _pick(reps: list[tuple[dict[str, Any], dict[str, Any]]]
              ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Cheapest accepted representative, else cheapest."""
        def key(item: tuple[dict[str, Any], dict[str, Any]]
                ) -> tuple[bool, float, int]:
            verdict, entry = item
            return (not verdict["accepted"], entry["cpu"],
                    entry["bytes"])

        return min(reps, key=key)

    def _judge_arm(self, task_id: str, arm: str,
                   proposals: list[Mapping[str, Any]]
                   ) -> tuple[dict[str, Any], int,
                              dict[str, float],
                              dict[str, dict[str, Any]]]:
        """Cluster, judge representatives and pick the task verdict."""
        kept, dropped, dedup = self._judge.cluster(
            task_id, proposals)
        judged: list[tuple[dict[str, Any], dict[str, Any]]] = []
        commands: dict[str, dict[str, Any]] = {}
        for entry in kept:
            verdict, extra = self._judge.judge_visible(
                task_id, entry["kind"], entry["candidate_text"],
                proposal=entry["proposal"])
            judged.append((verdict, entry))
            commands.update(extra)
        best, entry = self._pick(judged)
        best["costs"]["dedup"]["cpu_seconds"] += dedup[
            "cpu_seconds"]
        best["costs"]["dedup"]["wall_seconds"] += dedup[
            "wall_seconds"]
        lite = _public_verdict(best)
        lite["arm"] = arm
        lite["candidate_bytes"] = entry["bytes"]
        lite["cluster_size"] = len(kept)
        return lite, dropped, dedup, commands

    def run(self, task_ids: "list[str] | None" = None
            ) -> dict[str, Any]:
        """Judge tasks, assert controls, write and return summary."""
        ids = list(task_ids) if task_ids is not None else sorted(
            self._tasks)
        per_task: dict[str, Any] = {}
        taxonomy: dict[str, int] = {}
        arms_accepted = {name: 0 for name in self._arms}
        by_family: dict[str, Any] = {}
        by_interface: dict[str, Any] = {}
        phases: dict[str, dict[str, float]] = {
            phase: {"cpu_seconds": 0.0, "wall_seconds": 0.0,
                    "n": 0} for phase in JUDGE_PHASES}
        commands: dict[str, dict[str, Any]] = {}
        dropped_total = 0
        controls_bad: list[str] = []
        controls_total = 0

        def note_costs(costs: dict[str, dict[str, float]]
                       ) -> None:
            for phase in JUDGE_PHASES:
                slot = costs[phase]
                phases[phase]["cpu_seconds"] += slot[
                    "cpu_seconds"]
                phases[phase]["wall_seconds"] += slot[
                    "wall_seconds"]
                phases[phase]["n"] += 1

        for task_id in ids:
            task = self._tasks[task_id]
            entry: dict[str, Any] = {
                "family": task.get("family", ""),
                "interface": task.get("interface", ""),
                "arms": {}, "dropped_equivalents": 0}
            for name, arm in self._arms.items():
                proposals = [dict(p) for p in arm(task)]
                lite, dropped, dedup, extra = self._judge_arm(
                    task_id, name, proposals)
                entry["arms"][name] = lite
                entry["dropped_equivalents"] += dropped
                dropped_total += dropped
                commands.update(extra)
                for code in lite["taxonomy"]:
                    taxonomy[code] = taxonomy.get(code, 0) + 1
                if lite["accepted"]:
                    arms_accepted[name] += 1
                note_costs(lite["costs"])
            kind, text = LlmJudge.wrong_control(
                task, [list(row)
                       for row in self._visible[task_id]])
            control, extra = self._judge.judge_visible(
                task_id, kind, text)
            commands.update(extra)
            controls_total += 1
            lite_control = _public_verdict(control)
            entry["control"] = lite_control
            note_costs(control["costs"])
            for code in control["taxonomy"]:
                taxonomy[code] = taxonomy.get(code, 0) + 1
            if control["accepted"]:
                controls_bad.append(task_id)
            for table, label in ((by_family, "family"),
                                 (by_interface, "interface")):
                slot = table.setdefault(
                    str(task.get(label, "")),
                    {"tasks": 0, "accepted": {
                        name: 0 for name in self._arms}})
                slot["tasks"] += 1
                for name in self._arms:
                    if entry["arms"][name]["accepted"]:
                        slot["accepted"][name] += 1
            per_task[task_id] = entry
        after = self._judge.compiler_sha
        summary = {
            "batch": "llm_judge_batch_v1",
            "bank_id": self._bank_id,
            "range_note": RANGE_NOTE,
            "tasks": ids, "arms": sorted(self._arms),
            "per_task": per_task,
            "counts": {"tasks": len(ids),
                       "arms": dict(arms_accepted),
                       "controls_rejected":
                       controls_total - len(controls_bad),
                       "controls_total": controls_total,
                       "dropped_equivalents": dropped_total},
            "breakdown": {"by_family": by_family,
                          "by_interface": by_interface},
            "taxonomy": taxonomy,
            "compiler": {
                "path": str(self._judge.compiler_path),
                "sha256_before": self._judge.compiler_sha,
                "sha256_after": after,
                "stable": after == self._judge.compiler_sha,
                "guard": self._judge.uses_guard},
            "costs": {
                "cost_definition": COST_DEFINITION,
                "excluded_costs": list(
                    self._judge.ledger.to_dict()[
                        "excluded_costs"]),
                "judge_phases": phases,
                "command_costs": commands,
                "measured_total_cpu_seconds": sum(
                    slot["cpu_seconds"]
                    for slot in phases.values()),
                "measured_total_wall_seconds": sum(
                    slot["wall_seconds"]
                    for slot in phases.values()),
                "cost_ledger": self._judge.ledger.to_dict()},
        }
        path = self._outdir / "summary.json"
        path.write_text(json.dumps(summary, indent=2) + "\n")
        if controls_bad:
            raise AssertionError(
                "wrong controls accepted: "
                + ",".join(sorted(controls_bad)))
        return summary
