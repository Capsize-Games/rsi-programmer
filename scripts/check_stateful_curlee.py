"""Check a stateful probe or its supplied grammar controls under CURLEE."""
from __future__ import annotations

from rsi_programmer.stateful_tasks import oracle
from scripts.check_statement_curlee import main


if __name__ == "__main__":
    raise SystemExit(main(task_oracle=oracle))
