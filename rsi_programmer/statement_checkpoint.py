"""Reuse a local, fingerprint-checked statement policy without retraining."""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import torch

from rsi_programmer.benchmark_policy import state_fingerprint
from rsi_programmer.statement_policy import StatementPolicy
from rsi_programmer.statement_syntax import TOKENS


def load_or_fit(path: Path, expected_hash: str, task_ids: list[str]
                ) -> tuple[StatementPolicy, dict[str, Any]]:
    started = time.monotonic()
    model = StatementPolicy(seed=0, execution_context=True)
    metadata = {"seed": 0, "max_tokens": 32, "tokens": TOKENS,
                "execution_context": True}
    reused = path.exists()
    if reused:
        checkpoint = torch.load(path, weights_only=True)
        if checkpoint["metadata"] != metadata:
            raise ValueError("checkpoint policy metadata changed")
        model.load_state_dict(checkpoint["state"])
        fit = checkpoint["fit"]
    else:
        fit = model.fit(task_ids, 30)
    fingerprint = state_fingerprint(model.state_dict(), metadata)
    if fingerprint != expected_hash:
        raise ValueError("checkpoint differs from qualified policy")
    if not reused:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": model.state_dict(), "metadata": metadata,
                    "trained_hash": fingerprint, "fit": fit}, path)
    model.eval()
    return model, {"fit": fit, "trained_hash": fingerprint,
                   "metadata": metadata,
                   "checkpoint_reused": reused,
                   "load_or_fit_wall_seconds": time.monotonic() - started}
