#!/usr/bin/env python3
"""Fixed spike-to-edit decoder: populations in, one typed edit out.

The SNN transmits per-population spike trains over the policy step
window; this module deterministically decodes them into a single
ActionSpace action. Winner-take-all with explicit thresholds: ties,
sub-threshold activity, and out-of-alphabet tokens reject with a
logged reason instead of guessing. The decoder learns nothing; the
alphabet it accepts is the safety boundary.
"""
from __future__ import annotations

import torch

from rsi_programmer.actions import ActionSpace
from rsi_programmer.policy import OP_ORDER, STEPS
from rsi_programmer.probe import CMP, CONSTS, SELECT

INPUT = "input"
CONST = "const"
OP_NAMES = tuple(OP_ORDER) + tuple(CMP) + (SELECT, INPUT, CONST)
STRUCT = ("emit", "stop")


def _rate(train: torch.Tensor) -> torch.Tensor:
    """Mean firing rate per unit over the step window."""
    return train.float().mean(dim=0)


def _winner(rate: torch.Tensor, threshold: float,
            tie_eps: float) -> tuple[int | None, str]:
    """Index of the outright winner, or None with a reason."""
    if rate.numel() == 0:
        return None, "unknown-token"
    top = torch.argmax(rate).item()
    if float(rate[top]) < threshold:
        return None, "sub-threshold"
    if rate.numel() > 1:
        rest = torch.cat([rate[:top], rate[top + 1:]])
        if float(rate[top]) - float(torch.max(rest)) <= tie_eps:
            return None, "tie"
    return int(top), ""


class SpikeDecoder:
    """WTA decoder from spike populations to ActionSpace edits."""

    def __init__(self, space: ActionSpace | None = None,
                 op_threshold: float = 0.5,
                 ptr_threshold: float = 0.5,
                 struct_threshold: float = 0.5,
                 tie_eps: float = 1e-6) -> None:
        """Bind thresholds; the alphabet is fixed, never learned."""
        self._space = space or ActionSpace()
        self._op_threshold = op_threshold
        self._ptr_threshold = ptr_threshold
        self._struct_threshold = struct_threshold
        self._tie_eps = tie_eps

    def _pop(self, spikes: dict[str, torch.Tensor],
             name: str, size: int) -> torch.Tensor | None:
        """Rate vector for one population, or None when malformed."""
        train = spikes.get(name)
        if train is None or train.dim() != 2:
            return None
        if train.shape[0] != STEPS or train.shape[1] != size:
            return None
        return _rate(train)

    def decode(self, spikes: dict[str, torch.Tensor],
               refs: list[str | int],
               kinds: list[str] | None = None
               ) -> tuple[dict | None, dict]:
        """One action or None plus a reject record with reasons."""
        table = list(kinds) if kinds is not None else ["Int"] * len(
            refs)
        struct = self._pop(spikes, "struct", len(STRUCT))
        if struct is None:
            return None, {"verdict": "unknown-token"}
        gate, reason = _winner(struct, self._struct_threshold,
                               self._tie_eps)
        if gate is None or STRUCT[gate] != "emit":
            return None, {"verdict": reason or "stop"}
        ops = self._pop(spikes, "op", len(OP_NAMES))
        if ops is None:
            return None, {"verdict": "unknown-token"}
        pick, reason = _winner(ops, self._op_threshold,
                               self._tie_eps)
        if pick is None:
            return None, {"verdict": reason}
        return self._emit(OP_NAMES[pick], spikes, refs, table)

    def _kind(self, ref: str | int, refs: list[str | int],
              table: list[str]) -> str | None:
        """Static IR type of one referenced element."""
        if isinstance(ref, str):
            return "Int" if ref in self._space.inputs() else None
        try:
            return table[refs.index(ref)]
        except (ValueError, IndexError):
            return None

    def _typed(self, ref: str | int | None,
               refs: list[str | int], table: list[str],
               want: str) -> bool:
        """True when the reference exists with the wanted type."""
        return ref is not None and self._kind(ref, refs,
                                              table) == want

    def _ptr(self, spikes: dict[str, torch.Tensor], name: str,
             refs: list[str | int]) -> str | int | None:
        """Referenced element or None when the pointer rejects."""
        vec = self._pop(spikes, name, len(refs))
        if vec is None:
            return None
        pick, _ = _winner(vec, self._ptr_threshold, self._tie_eps)
        return None if pick is None else refs[pick]

    def _emit(self, op: str, spikes: dict[str, torch.Tensor],
              refs: list[str | int], table: list[str]
              ) -> tuple[dict | None, dict]:
        """Build the winning action; pointers may still reject."""
        if op == INPUT:
            ref = self._ptr(spikes, "left", refs)
            if ref not in self._space.inputs():
                return None, {"verdict": "unknown-token"}
            return {"kind": "input", "ref": ref}, {"verdict": ""}
        if op == CONST:
            vec = self._pop(spikes, "value", len(CONSTS))
            if vec is None:
                return None, {"verdict": "unknown-token"}
            pick, reason = _winner(vec, self._ptr_threshold,
                                   self._tie_eps)
            if pick is None:
                return None, {"verdict": reason}
            return {"kind": "const", "value": CONSTS[pick]}, {
                "verdict": ""}
        if op == SELECT:
            cond = self._ptr(spikes, "cond", refs)
            left = self._ptr(spikes, "left", refs)
            right = self._ptr(spikes, "right", refs)
            if cond is None or left is None or right is None:
                return None, {"verdict": "tie"}
            if not self._typed(cond, refs, table, "Bool"):
                return None, {"verdict": "type-mismatch"}
            if not self._typed(left, refs, table, "Int"):
                return None, {"verdict": "type-mismatch"}
            if not self._typed(right, refs, table, "Int"):
                return None, {"verdict": "type-mismatch"}
            return {"kind": "apply", "op": SELECT, "cond": cond,
                    "left": left, "right": right}, {"verdict": ""}
        left = self._ptr(spikes, "left", refs)
        right = self._ptr(spikes, "right", refs)
        if left is None or right is None:
            return None, {"verdict": "tie"}
        if not self._typed(left, refs, table, "Int"):
            return None, {"verdict": "type-mismatch"}
        if not self._typed(right, refs, table, "Int"):
            return None, {"verdict": "type-mismatch"}
        return {"kind": "apply", "op": op, "left": left,
                "right": right}, {"verdict": ""}
