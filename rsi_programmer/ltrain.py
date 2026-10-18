#!/usr/bin/env python3
"""Bit-slice team training loop (lattice v32).

Every triple directly labels its own opcode bits (recognition);
membership labels apply to EMIT and the complementary STOP output.
Both struct outputs are fitted, then decoded by an EMIT-minus-STOP
threshold. STOP can also affect later outputs through recurrent
feedback. Each pack combines the mean of six aggregate head losses
with their maximum; the maximum is across pack-aggregate head losses,
not per-word errors. Mixed-class rate terms use a one-spike hinge
margin and have attainable floor zero. Thresholds are tuned on a
calibration slice and metrics are reported on a separate evaluation
slice.
"""
from __future__ import annotations

from typing import Any

import torch

from rsi_programmer.author import contrastive_groups
from rsi_programmer.lattice import EMIT, N_BITS, STOP, BitLattice
from rsi_programmer.lineage import mission_ledger
from rsi_programmer.modules import with_gap
from rsi_programmer.policy import OP_ORDER, STEPS
from rsi_programmer.words import OPCODES, unpack

RATE_TARGET = 0.5
RATE_TOLERANCE = 0.05


def _rate_margin(steps: int) -> float:
    """One binary firing step as a fraction of the rate window."""
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("lattice steps must be positive")
    return 1.0 / steps


def _margins(out: torch.Tensor, want: torch.Tensor,
             steps: int = STEPS) -> torch.Tensor:
    """One-spike hinge ranking for mixed labels; regress constants.

    Rates average ``steps`` binary spike steps, so the requested
    margin is one firing step (1/steps). Once every positive-negative pair is
    separated by that amount, the term is exactly zero and gives no
    further gradient toward saturated rates. With a constant target
    (an opcode bit no training triple sets), squared error regresses
    the channel to that constant instead.
    """
    margin = _rate_margin(steps)
    pos = out[want > 0.5]
    neg = out[want < 0.5]
    if len(pos) and len(neg):
        gap = pos.unsqueeze(1) - neg.unsqueeze(0)
        return torch.relu(margin - gap).mean()
    const = want[0] if len(want) else torch.zeros(())
    return ((out - const) ** 2).mean()


def _head_rate(lat: BitLattice, mat: torch.Tensor, key: str,
               col: int) -> float:
    """Mean firing rate of one head on one batch, stateless."""
    lat.reset()
    with torch.no_grad():
        out = lat._drive(mat)
    return out[key][:, col].mean().item()


def _measure_rates(lat: BitLattice,
                   mat: torch.Tensor) -> dict[str, float]:
    """Measure every bit and struct channel on one fresh batch."""
    lat.reset()
    with torch.no_grad():
        out = lat._drive(mat)
    rates = {"b%d" % i: out["b%d" % i][:, 0].mean().item()
             for i in range(N_BITS)}
    rates["st_emit"] = out["st"][:, EMIT].mean().item()
    rates["st_stop"] = out["st"][:, STOP].mean().item()
    return rates


def _calibrate(lat: BitLattice, mat: torch.Tensor
               ) -> dict[str, Any]:
    """Center all channels, stopping when their measured rates converge.

    Bias searches are repeated because struct outputs feed back
    into the encoder and can shift bit rates. This controls firing
    rates; it does not establish that any head separates labels.
    """
    state = lat.weights()
    targets = [("b%d" % i, 0, "b%d_fc" % i)
               for i in range(N_BITS)]
    targets += [("st", EMIT, "st_fc"),
                ("st", STOP, "st_fc")]
    cycles = 0
    with torch.no_grad():
        for cycle in range(1, 4):
            for key, col, prefix in targets:
                size = state[prefix + ".bias"].numel()
                lo, hi = -10.0, 10.0
                for _ in range(12):
                    mid = (lo + hi) / 2.0
                    bias = state[prefix + ".bias"].clone()
                    bias[col] = mid
                    state[prefix + ".bias"] = bias
                    lat._net.load_state_dict(state)
                    rate = _head_rate(lat, mat, key, col)
                    if abs(rate - RATE_TARGET) < RATE_TOLERANCE:
                        break
                    if rate < RATE_TARGET:
                        lo = mid
                    else:
                        hi = mid
            cycles = cycle
            rates = _measure_rates(lat, mat)
            deviations = {key: abs(value - RATE_TARGET)
                          for key, value in rates.items()}
            if all(value <= RATE_TOLERANCE
                   for value in deviations.values()):
                break
        lat._net.load_state_dict(state)
    return {"rates": rates, "converged": all(
                value <= RATE_TOLERANCE for value in deviations.values()),
            "cycles": cycles, "target": RATE_TARGET,
            "tolerance": RATE_TOLERANCE,
            "max_deviation": max(deviations.values(), default=0.0)}


def _bit_targets(groups: list[Any]
                 ) -> tuple[list[list[float]], list[int],
                            list[int]]:
    """Own-opcode bit labels; membership labels stay separate.

    Bits recognize opcodes (a linear readout of the op
    one-hot); direct membership targets go to the struct
    outputs. Those outputs feed back into later encoder
    states, so STOP can influence later bit rates indirectly.
    Folding membership into bits demands an
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


def pack_rows(group: list[Any]) -> tuple[Any, Any, Any]:
    """One contrastive group to training tensors."""
    feats, flat, labels = _bit_targets([group])
    mat = with_gap(feats, None)
    want = torch.tensor(flat, dtype=torch.float32)
    want = want.reshape(len(feats), N_BITS)
    return mat, want, torch.tensor(labels,
                                    dtype=torch.float32)


def _pack_objective(terms: torch.Tensor, lam: float) -> torch.Tensor:
    """Mean plus weighted maximum over per-head pack losses."""
    return terms.mean() + lam * terms.max()


def fit_packs(lat: BitLattice,
              packs: list[tuple[Any, Any, Any]], epochs: int,
              lam: float) -> None:
    """Fit mean-plus-maximum of six pack-aggregate head losses."""
    for _ in range(epochs):
        lat._opt.zero_grad()
        total, count = torch.zeros(()), 0
        for mat, want, membership in packs:
            lat.reset()
            out = lat._drive(mat)
            terms = [_margins(out["b%d" % i][:, 0],
                              want[:, i], lat._steps)
                     for i in range(N_BITS)]
            terms.append(_margins(out["st"][:, EMIT],
                                  membership, lat._steps))
            terms.append(_margins(out["st"][:, STOP],
                                  1.0 - membership, lat._steps))
            total = total + _pack_objective(torch.stack(terms), lam)
            count += 1
        loss = total / max(count, 1)
        loss.backward()
        lat._opt.step()


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
    Calibration and evaluation groups are split before either
    calibration or fitting. Rates are measured again after the
    final feedback-sensitive calibration. Threshold tuning uses
    calibration groups; reported bit metrics use evaluation groups.
    """
    from rsi_programmer.lprecision import refine, split_groups
    supervision_report: list[dict[str, Any]] = []
    split_report: dict[str, Any] = {}
    groups = contrastive_groups(fixture,
                                supervision_report=supervision_report)
    if not groups:
        raise ValueError("lattice training requires at least one group")
    train_groups, calibration_groups, eval_groups = split_groups(
        groups, report=split_report)
    packs = [pack_rows(group) for group in train_groups]
    full = torch.cat([mat for mat, _, _ in packs])
    calibration = _calibrate(lat, full)
    fit_packs(lat, packs, epochs, lam)
    postfit_rates = _measure_rates(lat, full)
    calibration_rates = calibration["rates"]
    rate_drift = {key: postfit_rates[key] - calibration_rates[key]
                  for key in calibration_rates}
    report = {"fixture": fixture, "epochs": epochs,
              "train_groups": len(train_groups),
              "calibration_groups": len(calibration_groups),
              "evaluation_groups": len(eval_groups),
              "split_unit": "contrastive_group",
              "group_split": split_report,
              "initial_calibration": calibration,
              "postfit_rates": postfit_rates,
              "postfit_rate_drift": rate_drift,
              "postfit_rates_in_band": all(
                  abs(value - RATE_TARGET) <= RATE_TOLERANCE
                  for value in postfit_rates.values()),
              "postfit_max_rate_deviation": max(
                  (abs(value - RATE_TARGET)
                   for value in postfit_rates.values()), default=0.0)}
    report["training_objective"] = {
        "mixed_class_loss": "pairwise_hinge",
        "mixed_class_rate_margin": _rate_margin(lat._steps),
        "rate_window_steps": lat._steps,
        "mixed_class_attainable_loss_floor": 0.0,
        "constant_target_loss": "mean_squared_error",
        "pack_aggregate": "mean_plus_lambda_max_of_six_head_losses",
        "stop_target": "complement_of_membership",
        "struct_decode": "emit_rate_ge_stop_rate_plus_calibrated_margin"}
    report["training_supervision"] = supervision_report
    if calibration_groups:
        report.update(refine(lat, calibration_groups))
        report["calibration_available"] = True
    else:
        report.update({"calibration_available": False,
                       "calibration_exact": None,
                       "calibration_f1": None,
                       "calibration_joint_word_emit_accuracy": None,
                       "calibration_rows": 0,
                       "val_exact": None, "val_f1": None,
                       "val_rows": 0})
    eval_packs = [pack_rows(group) for group in eval_groups]
    if eval_packs:
        eval_mat = torch.cat([mat for mat, _, _ in eval_packs])
        eval_want = torch.cat([want for _, want, _ in eval_packs])
        eval_memb = torch.cat([memb for _, _, memb in eval_packs])
        report["evaluation_available"] = True
    else:
        eval_mat = torch.empty((0, full.shape[1]))
        eval_want = torch.empty((0, N_BITS))
        eval_memb = torch.empty((0,))
        report["evaluation_available"] = False
    final = _report(lat, eval_mat, eval_want, fixture, epochs,
                    eval_mat.shape[0], eval_memb)
    report.update(final)
    store = mission_ledger(ledger, "rsi-lattice")
    store.record_intent("train-seed-%d" % lat._seed,
                        "rsi-lattice", 0, 1, report)
    store.close()
    return report


def _report(lat: BitLattice, mat: torch.Tensor,
            want: torch.Tensor, fixture: str, epochs: int,
            rounds: int, membership: torch.Tensor | None = None
            ) -> dict[str, Any]:
    """Per-bit and word-bit exact accuracy, no gradients.

    Bits decode at the learned thresholds, matching the
    author path; the struct emit rate stays raw.
    """
    if not rounds:
        return {"fixture": fixture, "epochs": epochs,
                "evaluation_rows": 0, "bit_accuracy": None,
                "word_bit_exact_accuracy": None,
                "rounds": 0, "head_accuracy": None,
                "word_exact": None, "emit_rate": None,
                "stop_rate": None,
                "emit_membership_accuracy": None,
                "joint_word_emit_accuracy": None}
    lat.reset()
    with torch.no_grad():
        out = lat._drive(mat)
    got = torch.stack([out["b%d" % i][:, 0] >=
                       lat.thresholds["b%d" % i]
                       for i in range(N_BITS)], dim=1)
    truth = want > 0.5
    acc = [(got[:, i] == truth[:, i]).float().mean().item()
           for i in range(N_BITS)]
    exact = (got == truth).all(dim=1).float().mean()
    emit_pred = (out["st"][:, EMIT] >=
                 out["st"][:, STOP] + lat.thresholds["st"])
    emit_accuracy: float | None = None
    joint_accuracy: float | None = None
    if membership is not None and membership.numel():
        membership_truth = membership > 0.5
        emit_accuracy = (emit_pred == membership_truth).float().mean().item()
        joint = (got == truth).all(dim=1)
        joint = joint & emit_pred
        joint_accuracy = (joint == membership_truth).float().mean().item()
    word_exact = exact.item()
    return {"fixture": fixture, "epochs": epochs,
            "evaluation_rows": rounds, "bit_accuracy": acc,
            "word_bit_exact_accuracy": word_exact,
            # Retained for callers of the original report schema.
            "rounds": rounds, "head_accuracy": acc,
            "word_exact": word_exact,
            "emit_rate": out["st"][:, EMIT].mean().item(),
            "stop_rate": out["st"][:, STOP].mean().item(),
            "emit_membership_accuracy": emit_accuracy,
            "joint_word_emit_accuracy": joint_accuracy}
