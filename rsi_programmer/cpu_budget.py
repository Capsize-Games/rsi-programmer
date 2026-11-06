"""Enforce per-arm CPU-second budgets with process CPU timers."""

from __future__ import annotations

import signal
from types import FrameType
from typing import Any


class CpuBudgetExceeded(TimeoutError):
    """Raised when a cell spends more CPU than its frozen budget."""


def _raise_budget(signum: int, frame: FrameType | None) -> None:
    """Convert the virtual-timer signal into a catchable error."""
    del signum, frame
    raise CpuBudgetExceeded("cell exceeded its CPU-second budget")


class CpuBudget:
    """Cap CPU time (user plus system) for one search cell.

    Uses ``ITIMER_VIRTUAL`` so only CPU consumed counts, never wall
    waiting. Single-threaded use only: entering saves and leaving
    restores the previous timer and handler.
    """

    def __init__(self, budget_seconds: float) -> None:
        """Store a strictly positive CPU-second budget."""
        if not isinstance(budget_seconds, (int, float)):
            raise TypeError("budget must be numeric")
        if not budget_seconds > 0.0:
            raise ValueError("budget must be positive")
        self._budget = float(budget_seconds)
        self._previous: Any = None
        self._handler: Any = None

    def __enter__(self) -> CpuBudget:
        """Arm the virtual timer and install the raising handler."""
        self._handler = signal.getsignal(signal.SIGVTALRM)
        self._previous = signal.getitimer(signal.ITIMER_VIRTUAL)
        signal.signal(signal.SIGVTALRM, _raise_budget)
        signal.setitimer(signal.ITIMER_VIRTUAL, self._budget)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any,
                 traceback: Any) -> bool:
        """Disarm the timer and restore the previous handler state."""
        signal.setitimer(signal.ITIMER_VIRTUAL, 0.0)
        signal.signal(signal.SIGVTALRM, self._handler)
        previous_delay, previous_interval = self._previous
        if previous_delay > 0.0 or previous_interval > 0.0:
            signal.setitimer(signal.ITIMER_VIRTUAL, previous_delay,
                             previous_interval)
        return False
