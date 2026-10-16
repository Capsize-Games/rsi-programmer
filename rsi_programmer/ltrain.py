#!/usr/bin/env python3
"""Bit-slice team training loop (lattice v32).

Positive solution triples target their opcode bits;
negatives target the reserved word (rejected at decode).
The struct gate targets solution membership. A worst-bit
joint term couples the four heads per word. The shared
encoder stays write-locked throughout; one lineage intent
is recorded per run.
"""
from __future__ import annotations

from typing import Any

import torch

from rsi_programmer.author import contrastive_groups
from rsi_programmer.lattice import EMIT, N_BITS, STOP, BitLattice
from rsi_programmer.lineage import mission_ledger
from rsi_programmer.modules import with_gap
from rsi_programmer.policy import OP_ORDER
from rsi_programmer.words import OPCODES, unpack


def _margins(out: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    """House margin: positives above negatives, or regress.

    With both classes present this is the pairwise margin
    rankers train on (policy/modular precedent): relative,
    skew-proof, and forgiving of operating point. With a
    constant target (an opcode bit no training triple sets)
    it regresses to that constant instead.
    """
    pos = out[want > 0.5]
    neg = out[want < 0.5]
    if len(pos) and len(neg):
        gap = pos.unsqueeze(1) - neg.unsqueeze(0)
        return torch.nn.functional.softplus(1.0 - gap).mean()
    const = want[0] if len(want) else torch.zeros(())
    return ((out - const) ** 2).mean()


def _head_rate(lat: BitLattice, mat: torch.Tensor, key: str,
               col: int) -> float:
    """Mean firing rate of one head on one batch, stateless."""
    lat.reset()
    with torch.no_grad():
        out = lat._drive(mat)
    return out[key][:, col].mean().item()


def _calibrate(lat: BitLattice, mat: torch.Tensor
               ) -> dict[str, float]:
    """Center every readout head at 50% firing (max-entropy).

    Fresh spiking heads land outside the surrogate-gradient
    band (measured exactly-zero gradients) and can never
    learn. Centering each head bias to half firing puts all
    heads where gradients flow; training does the rest. No
    hand-tuned constants: 50% is the maximum-slope point,
    found by binary search per head (firing is monotone in
    bias), deterministic on the training batch.
    """
    state = lat.weights()
    found: dict[str, float] = {}
    targets = [("b%d" % i, 0, "b%d_fc" % i)
               for i in range(N_BITS)]
    targets.append(("st", EMIT, "st_fc"))
    with torch.no_grad():
        for key, col, prefix in targets:
            size = state[prefix + ".bias"].numel()
            lo, hi = -10.0, 10.0
            for _ in range(12):
                mid = (lo + hi) / 2.0
                bias = torch.zeros(size)
                bias[col] = mid
                state[prefix + ".bias"] = bias
                lat._net.load_state_dict(state)
                rate = _head_rate(lat, mat, key, col)
                if abs(rate - 0.5) < 0.05:
                    break
                if rate < 0.5:
                    lo = mid
                else:
                    hi = mid
            found[key] = _head_rate(lat, mat, key, col)
        lat._net.load_state_dict(state)
    return found


def _bit_targets(groups: list[Any]
                 ) -> tuple[list[list[float]], list[int],
                            list[int]]:
    """Own-opcode bits for every triple; labels stay separate.

    Bits recognize opcodes (a linear readout of the op
    one-hot); solution membership belongs to the struct
    gate alone. Folding membership into bits demands an
    XOR-like rule no linear head can express, so negatives
    no longer target the reserved word here. Reserved words
    remain a decoder safety boundary only.
    """
    feats: list[list[float]] = []
    bits: list[int] = []
    labels: list[int] = []
    for group in groups:
        for row, label in group:
            op = OP_ORDER[max(range(len(OP_ORDER)),
                              key=lambda i: row[i])]
            feats.append(row)
            bits += unpack(OPCODES[op])
            labels.append(label)
    return feats, bits, labels


def train_lattice(lat: BitLattice, fixture: str, ledger: str,
                  epochs: int = 8,
                  lam: float = 0.2) -> dict[str, Any]:
    """Train heads and encoder jointly from the modular init.

    The encoder is deliberately NOT frozen: with frozen
    shared weights the fresh spiking heads sit outside the
    surrogate-gradient band (measured exactly-zero head and
    gate gradients) and nothing can learn. Joint training
    follows the house pattern (policy/modular); eval-time
    freezing via weights_hash still guards every benchmark.
    """
    groups = contrastive_groups(fixture)
    packs: list[tuple[Any, Any, Any]] = []
    for group in groups:
        feats, flat, labels = _bit_targets([group])
        mat = with_gap(feats, None)
        want = torch.tensor(flat, dtype=torch.float32)
        want = want.reshape(len(feats), N_BITS)
        packs.append((mat, want, torch.tensor(
            labels, dtype=torch.float32)))
    full = torch.cat([mat for mat, _, _ in packs])
    calibration = _calibrate(lat, full)
    for _ in range(epochs):
        lat._opt.zero_grad()
        total, count = torch.zeros(()), 0
        for mat, want, membership in packs:
            lat.reset()
            out = lat._drive(mat)
            terms = [_margins(out["b%d" % i][:, 0],
                              want[:, i])
                     for i in range(N_BITS)]
            terms.append(_margins(out["st"][:, EMIT],
                                  membership))
            terms.append(_margins(out["st"][:, STOP],
                                  1.0 - membership))
            joined = torch.stack(terms)
            total = total + joined.mean() + \
                lam * joined.max()
            count += 1
        loss = total / max(count, 1)
        loss.backward()
        lat._opt.step()
    report = _report(lat, full, torch.cat(
        [want.reshape(-1) for _, want, _ in packs]).reshape(
        -1, N_BITS), fixture, epochs, sum(
        mat.shape[0] for mat, _, _ in packs))
    report["calibration"] = calibration
    store = mission_ledger(ledger, "rsi-lattice")
    store.record_intent("train-seed-%d" % lat._seed,
                        "rsi-lattice", 0, 1, report)
    store.close()
    return report


def _report(lat: BitLattice, mat: torch.Tensor,
            want: torch.Tensor, fixture: str, epochs: int,
            rounds: int) -> dict[str, Any]:
    """Per-head and word-exact metrics, no gradients."""
    lat.reset()
    with torch.no_grad():
        out = lat._drive(mat)
    got = torch.stack([out["b%d" % i][:, 0] >= 0.5
                       for i in range(N_BITS)], dim=1)
    truth = want > 0.5
    acc = [(got[:, i] == truth[:, i]).float().mean().item()
           for i in range(N_BITS)]
    exact = (got == truth).all(dim=1).float().mean()
    return {"fixture": fixture, "epochs": epochs,
            "rounds": rounds, "head_accuracy": acc,
            "word_exact": exact.item(),
            "emit_rate": out["st"][:, EMIT].mean().item()}
