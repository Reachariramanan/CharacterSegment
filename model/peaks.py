"""Shared 1D peak-picking, used by both the akshara-boundary model (peaks in
a boundary-logit signal) and the line-segmentation model (edge peaks in a
line-presence mask)."""

from __future__ import annotations

import numpy as np
import torch


def peaks_from_signal(signal: torch.Tensor, threshold: float, min_distance: int) -> list[int]:
    """A point is a peak if it's >= both neighbors and above `threshold`;
    peaks closer than `min_distance` are merged, keeping the higher one."""
    values = signal.detach().cpu().numpy()
    length = len(values)
    candidates = []
    for i in range(1, length - 1):
        if values[i] >= threshold and values[i] >= values[i - 1] and values[i] >= values[i + 1]:
            candidates.append(i)

    candidates.sort(key=lambda i: -values[i])
    kept: list[int] = []
    for i in candidates:
        if all(abs(i - k) >= min_distance for k in kept):
            kept.append(i)
    return sorted(kept)


def contiguous_runs_above_threshold(signal: torch.Tensor, threshold: float, min_run_length: int = 1) -> list[tuple[int, int]]:
    """Returns (start, end) index pairs (end exclusive) of contiguous runs
    where signal > threshold, used to turn a line-presence mask into
    (y0, y1) line bands."""
    values = signal.detach().cpu().numpy()
    is_on = values > threshold

    runs: list[tuple[int, int]] = []
    start = None
    for i, on in enumerate(is_on):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(is_on)))

    return [r for r in runs if (r[1] - r[0]) >= min_run_length]


def split_by_relative_minima(
    values: np.ndarray,
    start: int,
    end: int,
    min_sub_length: int = 10,
    min_prominence: float = 0.10,
) -> list[tuple[int, int]]:
    """Recursively splits (start, end) at its most prominent interior local
    minimum, if that minimum is at least `min_prominence` below the LOWER of
    the two surrounding local maxima -- catches real gaps that an absolute
    threshold misses.

    Needed because a single scalar presence-per-row target/prediction can't
    represent a diagonal or curved separator exactly: two lines whose flat
    bounding boxes overlap (e.g. from a hand-drawn curved split) only produce
    a PARTIAL dip in the row-averaged presence signal -- real, but often not
    deep enough to cross a fixed absolute threshold like 0.5. A relative/
    prominence-based check finds it anyway, the same way topographic
    prominence finds a real valley between two peaks regardless of the
    valley's absolute elevation.
    """
    segment = values[start:end]
    length = end - start
    if length < 2 * min_sub_length:
        return [(start, end)]

    interior = segment[min_sub_length : length - min_sub_length]
    if len(interior) == 0:
        return [(start, end)]

    min_idx = int(np.argmin(interior)) + min_sub_length
    min_val = segment[min_idx]
    left_max = segment[:min_idx].max()
    right_max = segment[min_idx:].max()
    prominence = min(left_max, right_max) - min_val

    if prominence < min_prominence:
        return [(start, end)]

    split_at = start + min_idx
    left = split_by_relative_minima(values, start, split_at, min_sub_length, min_prominence)
    right = split_by_relative_minima(values, split_at, end, min_sub_length, min_prominence)
    return left + right
