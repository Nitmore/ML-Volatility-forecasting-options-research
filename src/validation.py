"""Purged expanding-window splits.

Fold k trains on [sample_start, Jan 1 of year k) and evaluates on year k.
Targets overlap: the label of a training row dated t uses returns t+1..t+h,
so rows whose `label_end` falls on or after the evaluation start would carry
evaluation-period returns into training. They are purged.

Because training always precedes evaluation, no post-evaluation embargo is
needed; the purge is the embargo of length h at the boundary.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    name: str
    split: str                     # "val" or "test"
    year: int
    train: pd.DatetimeIndex
    evaluate: pd.DatetimeIndex
    n_purged: int


def make_folds(index: pd.DatetimeIndex, label_end: pd.Series, sample_start: str,
               years: list[int], split: str, sample_end: str | None = None) -> list[Fold]:
    index = pd.DatetimeIndex(index)
    in_sample = index[(index >= sample_start) & ((index <= sample_end) if sample_end else True)]
    folds = []
    for y in years:
        ev_start, ev_end = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")
        cand = in_sample[in_sample < ev_start]
        ends = label_end.reindex(cand)
        keep = ends.notna() & (ends < ev_start)
        train = cand[keep.to_numpy()]
        evaluate = in_sample[(in_sample >= ev_start) & (in_sample <= ev_end)]
        folds.append(Fold(f"{split}_{y}", split, y, train, evaluate, int((~keep).sum())))
    return folds


def check_fold(f: Fold, label_end: pd.Series) -> None:
    """Hard guarantees: disjoint, ordered, and no training label reaches evaluation."""
    assert len(f.train) and len(f.evaluate), f.name
    assert f.train.max() < f.evaluate.min(), f"{f.name}: train overlaps evaluation"
    assert (label_end.reindex(f.train) < f.evaluate.min()).all(), f"{f.name}: purge failed"


def inner_calibration_split(train: pd.DatetimeIndex, label_end: pd.Series, n_calib: int = 252):
    """Last `n_calib` training rows form a calibration block; rows before it are purged the same way."""
    calib = train[-n_calib:]
    early = train[:-n_calib]
    early = early[(label_end.reindex(early) < calib.min()).to_numpy()]
    return early, calib
