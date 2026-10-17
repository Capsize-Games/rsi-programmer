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
                 ) -> tuple[list[list[Row]], list[Row]]:
    """Seeded 80/20 split per group; held-out rows pooled."""
    train: list[list[Row]] = []
    val: list[Row] = []
    for group in groups:
        idx = list(range(len(group)))
        Random(seed).shuffle(idx)
        keep = set(idx[:int(len(idx) * (1.0 - VAL_FRAC))])
        train.append([row for k, row in enumerate(group)
                      if k in keep])
        val += [row for k, row in enumerate(group)
                if k not in keep]
    return train, val


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


def sweep_thresholds(lat: BitLattice,
                     valpack: tuple[Any, Any, Any]
                     ) -> dict[str, Any]:
    """Coordinate descent on the held-out slice, in place.

    Grids start at the defaults so ties keep 0.5/0.0;
    every eval is a full held-out drive, no gradients.
    """
    mat, want, memb = valpack
    for _ in range(2):
        for i in range(N_BITS):
            key = "b%d" % i
            best, best_v = lat.thresholds[key], -1.0
            for grid in BIT_GRID:
                lat.thresholds[key] = grid
                val = _word_exact(lat, mat, want)
                if val > best_v:
                    best, best_v = grid, val
            lat.thresholds[key] = best
    best_m, best_f = lat.thresholds["st"], -1.0
    for grid in MARGIN_GRID:
        lat.thresholds["st"] = grid
        val = _struct_f1(lat, mat, memb)
        if val > best_f:
            best_m, best_f = grid, val
    lat.thresholds["st"] = best_m
    return {"bit_thresholds": [lat.thresholds["b%d" % i]
                               for i in range(N_BITS)],
            "st_margin": best_m,
            "val_exact": _word_exact(lat, mat, want),
            "val_f1": _struct_f1(lat, mat, memb),
            "val_rows": mat.shape[0]}


def refine(lat: BitLattice,
           groups: list[list[Row]]) -> dict[str, Any]:
    """Learn the decode operating point. No fitting here."""
    _, val = split_groups(groups)
    out: dict[str, Any] = {}
    if val:
        out.update(sweep_thresholds(lat, pack_rows(val)))
    return out
