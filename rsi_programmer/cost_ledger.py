"""End-to-end CPU cost ledger for programmer trials.

Records named trial phases in CPU seconds including completed child
processes, keeps repeated measurements with spread, amortizes one-off
costs over a declared request volume, and serializes to the JSON shape
artifacts already use (``command_costs`` entries, ``measured_total_*``
totals, ``excluded_costs``, ``cost_definition``).

Intended caller (issue #91 runner): create one ledger per trial, wrap
each phase with :meth:`CostLedger.measure` or :meth:`CostLedger.run`
for subprocesses, declare anything not measured with
:meth:`CostLedger.declare_excluded`, then :meth:`CostLedger.write_json`.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import resource
import statistics
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from rsi_programmer.program_cost import CostClock, cpu_seconds

PHASES: tuple[str, ...] = (
    "donor_library_creation",
    "fit",
    "mining",
    "qualification",
    "oracle_labelling",
    "search",
    "emission",
    "backend_execution",
    "candidate_evaluation",
)

COST_DEFINITION = (
    "Sum of process CPU including completed child descendants "
    "(resource RUSAGE_SELF plus RUSAGE_CHILDREN) across every "
    "recorded phase. Largest child RSS is a maximum, not an "
    "aggregate simultaneous memory measurement."
)


def _command_entry(command: Sequence[str], elapsed: dict[str, float],
                   rss: int, code: int) -> dict[str, Any]:
    return {"command": list(command), "exit_code": code,
            "wall_seconds": elapsed["wall_seconds"],
            "cpu_seconds": elapsed["cpu_seconds"],
            "whole_process_cpu_seconds": cpu_seconds(),
            "largest_child_rss_kib": rss,
            "accounting_source_sha256": hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest()}


def _stats(values: list[float]) -> dict[str, float]:
    total = sum(values)
    count = len(values)
    spread = statistics.stdev(values) if count > 1 else 0.0
    return {"n": float(count), "total": total,
            "mean": total / count if count else 0.0,
            "stdev": spread,
            "minimum": min(values) if values else 0.0,
            "maximum": max(values) if values else 0.0}


class CostLedger:
    """Accumulate per-phase CPU/wall samples and emit artifact JSON."""

    def __init__(self, cost_definition: str = COST_DEFINITION,
                 excluded: Sequence[str] | None = None) -> None:
        self._definition = cost_definition
        self._cpu: dict[str, list[float]] = {p: [] for p in PHASES}
        self._wall: dict[str, list[float]] = {p: [] for p in PHASES}
        self._commands: dict[str, dict[str, Any]] = {}
        self._excluded: list[str] = list(excluded or [])
        self._peak_rss_kib = 0

    @property
    def phases(self) -> tuple[str, ...]:
        return PHASES

    def _check(self, phase: str) -> None:
        if phase not in PHASES:
            raise ValueError(f"unknown ledger phase: {phase!r}")

    @contextlib.contextmanager
    def measure(self, phase: str) -> Iterator[None]:
        """Time one synchronous region and append its sample."""
        self._check(phase)
        clock = CostClock()
        yield
        elapsed = clock.elapsed()
        self._cpu[phase].append(elapsed["cpu_seconds"])
        self._wall[phase].append(elapsed["wall_seconds"])

    def record(self, phase: str, cpu: float, wall: float) -> None:
        """Append one externally measured sample to a phase."""
        self._check(phase)
        if cpu < 0 or wall < 0:
            raise ValueError("ledger samples must be non-negative")
        self._cpu[phase].append(cpu)
        self._wall[phase].append(wall)

    def run(self, phase: str, command: Sequence[str]) -> int:
        """Run a subprocess, ledger its incl-children cost, return code."""
        self._check(phase)
        clock = CostClock()
        result = subprocess.run(list(command), check=False)
        elapsed = clock.elapsed()
        rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        self._peak_rss_kib = max(self._peak_rss_kib, rss)
        self._cpu[phase].append(elapsed["cpu_seconds"])
        self._wall[phase].append(elapsed["wall_seconds"])
        key = phase if phase not in self._commands else (
            f"{phase}_{len(self._commands)}")
        self._commands[key] = _command_entry(
            command, elapsed, rss, result.returncode)
        return result.returncode

    def declare_excluded(self, description: str) -> None:
        """Name a cost outside this ledger with no measured sample."""
        if description not in self._excluded:
            self._excluded.append(description)

    def phase_stats(self, phase: str) -> dict[str, float]:
        """Return n/total/mean/stdev/min/max CPU for one phase."""
        self._check(phase)
        stats = _stats(self._cpu[phase])
        return {f"{key}_cpu_seconds": value
                for key, value in stats.items()}

    def totals(self) -> dict[str, float]:
        """Return measured CPU/wall summed across every phase."""
        return {"measured_total_cpu_seconds": sum(
            sum(v) for v in self._cpu.values()),
            "measured_total_wall_seconds": sum(
            sum(v) for v in self._wall.values())}

    def amortized(self, volume: int) -> dict[str, Any]:
        """Spread every phase total over a declared request volume."""
        if volume <= 0:
            raise ValueError("amortization volume must be positive")
        total = self.totals()["measured_total_cpu_seconds"]
        return {"volume": volume,
                "cpu_per_request_seconds": total / volume,
                "per_phase_cpu_per_request_seconds": {
                    p: sum(self._cpu[p]) / volume for p in PHASES
                    if self._cpu[p]}}

    def to_dict(self, volume: int | None = None) -> dict[str, Any]:
        """Serialize, extending the frozen-summary shape in place."""
        body: dict[str, Any] = {
            "cost_definition": self._definition,
            "excluded_costs": list(self._excluded),
            "phases": {p: {**self.phase_stats(p),
                            "total_wall_seconds": sum(self._wall[p])}
                       for p in PHASES if self._cpu[p]},
            "command_costs": dict(self._commands),
            **self.totals(),
            "largest_single_child_rss_kib": self._peak_rss_kib}
        if volume is not None:
            body["amortized"] = self.amortized(volume)
        return body

    def write_json(self, path: Path,
                   volume: int | None = None) -> Path:
        """Write :meth:`to_dict` to path and return the path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(volume), indent=2) + "\n")
        return path

    @classmethod
    def from_legacy(cls, legacy: Mapping[str, Any]) -> "CostLedger":
        """Ingest a frozen-summary-style dict as single-sample phases.

        Uses whole-process CPU to match artifact totals. The legacy
        search command already contains the fresh fit, so the search
        sample is netted by ``fresh_fit_cpu_seconds`` to keep phases
        partitioned; the fit wall is approximated by its CPU value.
        """
        ledger = cls(
            cost_definition=str(legacy.get("cost_definition",
                                           COST_DEFINITION)),
            excluded=[str(item) for item in
                      legacy.get("excluded_costs", [])])
        commands = legacy.get("command_costs", {})
        fit = legacy.get("fresh_fit_cpu_seconds")
        fit_cpu = float(fit) if isinstance(fit, (int, float)) else 0.0
        if isinstance(commands, Mapping):
            _ingest_commands(ledger, commands, fit_cpu)
        if fit_cpu and not ledger._cpu["fit"]:
            ledger.record("fit", fit_cpu, fit_cpu)
        return ledger


_LEGACY_COMMAND_PHASES = {"search": "search",
                           "python": "backend_execution",
                           "curlee": "backend_execution"}


def _legacy_phase(key: str) -> str | None:
    base = key.split("_")[0]
    if base in PHASES:
        return base
    return _LEGACY_COMMAND_PHASES.get(base)


def _ingest_commands(ledger: CostLedger,
                     commands: Mapping[str, Any],
                     fit_cpu: float = 0.0) -> None:
    for key, entry in commands.items():
        if not isinstance(entry, Mapping):
            continue
        phase = _legacy_phase(str(key))
        if phase is None:
            continue
        cpu = entry.get("whole_process_cpu_seconds",
                        entry.get("cpu_seconds"))
        wall = entry.get("wall_seconds", cpu)
        if not isinstance(cpu, (int, float)) or not isinstance(
                wall, (int, float)):
            continue
        value = float(cpu)
        if phase == "search" and fit_cpu:
            value = max(0.0, value - fit_cpu)
        ledger.record(phase, value, float(wall))
    ledger._commands.update(
        {str(k): dict(v) for k, v in commands.items()
         if isinstance(v, Mapping)})
