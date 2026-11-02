"""Local CPU/wall accounting for a process and its completed children."""
from __future__ import annotations

import resource
import time


def cpu_seconds() -> float:
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


class CostClock:
    """Measure elapsed wall and CPU without changing a task's authority."""

    def __init__(self) -> None:
        self.wall = time.monotonic()
        self.cpu = cpu_seconds()

    def elapsed(self) -> dict[str, float]:
        return {"wall_seconds": time.monotonic() - self.wall,
                "cpu_seconds": cpu_seconds() - self.cpu}
