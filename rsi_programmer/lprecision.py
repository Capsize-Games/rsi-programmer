"""Precision pass: held-out split and operating sweep.

Phase 2 of lattice training (Issue #40). Fitting stays
in phase 1: continued refitting drifts the operating
point (measured emit collapse) and all-negative packs
regress the gate to silence, so this pass never fits.
It splits a seeded held-out slice per group and learns
per-head thresholds plus the struct margin by
coordinate descent there. Thresholds land in
lat.thresholds, which the author path decodes with.
"""
from __future__ import annotations

from random import Random
from typing import Any

import torch

from rsi_programmer.lattice import N_BITS, BitLattice
from rsi_programmer.ltrain import pack_rows

VAL_FRAC = 0.2
BIT_GRID = (0.5, 0.4, 0.6, 0.7)
MARGIN_GRID = (0.0, -0.2, 0.2)

Row = tuple[list[float], int]


def split_groups(groups: list[list[Row]], seed: int = 0
                 ) -> tuple[list[list[Row]], list[list[Row]],
                            list[list[Row]]]:
    """Seeded train/calibration/evaluation split by whole groups."""
    order = list(range(len(groups)))
    Random(seed).shuffle(order)
    n_hold = int(len(groups) * VAL_FRAC)
    if len(groups) >= 3:
        n_hold = max(1, n_hold)
    cal_ids = set(order[:n_hold])
    eval_ids = set(order[n_hold:2 * n_hold])
    train = [group for i, group in enumerate(groups)
             if i not in cal_ids and i not in eval_ids]
    calibration = [group for i, group in enumerate(groups)
                   if i in cal_ids]
    evaluation = [group for i, group in enumerate(groups)
                  if i in eval_ids]
    return train, calibration, evaluation


def _word_exact(lat: BitLattice, mat: torch.Tensor,
                want: torch.Tensor) -> float:
    """Word-exact rate at the current thresholds."""
    lat.reset()
    with torch.no_grad():
        out = lat._drive(mat)
    got = torch.stack(
        [out["b%d" % i][:, 0] >= lat.thresholds["b%d" % i]
         for i in range(N_BITS)], dim=1)
    return ((got == (want > 0.5)).all(dim=1).float().mean()
            .item())


def _struct_f1(lat: BitLattice, mat: torch.Tensor,
               memb: torch.Tensor) -> float:
    """Struct-gate F1 at the current margin."""
    lat.reset()
    with torch.no_grad():
        out = lat._drive(mat)
    margin = lat.thresholds["st"]
    pred = out["st"][:, 0] >= out["st"][:, 1] + margin
    truth = memb > 0.5
    tp = (pred & truth).sum().item()
    fp = (pred & ~truth).sum().item()
    fn = ((~pred) & truth).sum().item()
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2.0 * prec * rec / (prec + rec)


def _joint_word_emit_accuracy(lat: BitLattice, mat: torch.Tensor,
                              want: torch.Tensor,
                              memb: torch.Tensor) -> float:
    """Accuracy of word-exact AND EMIT as a membership prediction."""
    lat.reset()
    with torch.no_grad():
        out = lat._drive(mat)
    got = torch.stack(
        [out["b%d" % i][:, 0] >= lat.thresholds["b%d" % i]
         for i in range(N_BITS)], dim=1)
    word_exact = (got == (want > 0.5)).all(dim=1)
    emit = (out["st"][:, 0] >=
            out["st"][:, 1] + lat.thresholds["st"])
    return ((word_exact & emit) == (memb > 0.5)).float().mean().item()


def sweep_thresholds(lat: BitLattice,
                     valpack: tuple[Any, Any, Any]
                     ) -> dict[str, Any]:
    """Tune thresholds on calibration groups, in place.

    Each score is a full calibration-set drive with no gradients.
    Strict improvement wins; ties retain the current threshold,
    including values outside the search grids.
    """
    mat, want, memb = valpack
    for _ in range(2):
        for i in range(N_BITS):
            key = "b%d" % i
            best = lat.thresholds[key]
            best_v = _word_exact(lat, mat, want)
            for grid in BIT_GRID:
                lat.thresholds[key] = grid
                val = _word_exact(lat, mat, want)
                if val > best_v:
                    best, best_v = grid, val
            lat.thresholds[key] = best
    best_m = lat.thresholds["st"]
    best_f = _struct_f1(lat, mat, memb)
    for grid in MARGIN_GRID:
        lat.thresholds["st"] = grid
        val = _struct_f1(lat, mat, memb)
        if val > best_f:
            best_m, best_f = grid, val
    lat.thresholds["st"] = best_m
    exact = _word_exact(lat, mat, want)
    f1 = _struct_f1(lat, mat, memb)
    joint = _joint_word_emit_accuracy(lat, mat, want, memb)
    return {"bit_thresholds": [lat.thresholds["b%d" % i]
                               for i in range(N_BITS)],
            "st_margin": best_m,
            "calibration_available": True,
            "calibration_exact": exact,
            "calibration_f1": f1,
            "calibration_joint_word_emit_accuracy": joint,
            "calibration_rows": mat.shape[0],
            # Preserve the original report fields for existing consumers.
            "val_exact": exact, "val_f1": f1,
            "val_rows": mat.shape[0]}


def refine(lat: BitLattice,
           groups: list[list[Row]]) -> dict[str, Any]:
    """Tune thresholds on calibration groups. No fitting here."""
    out: dict[str, Any] = {}
    if groups:
        val = [row for group in groups for row in group]
        out.update(sweep_thresholds(lat, pack_rows(val)))
    return out
