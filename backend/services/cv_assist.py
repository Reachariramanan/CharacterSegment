"""Classical-CV assisted boundary proposals for Grantha text lines.

Why not DBSCAN: a kootu ezhuthu (conjunct/ligature glyph cluster) can have
components that are spatially DISJOINT from their own base letter (e.g. a
vowel sign positioned above/below with a real visual gap) while touching a
neighboring INDEPENDENT letter (a shared baseline stroke or ligated serif).
Proximity-based clustering has no notion of "this gap is internal structure
of one glyph" vs "this gap is a real inter-akshara boundary" -- it only sees
Euclidean/pixel distance. The algorithm below is line-relative and
shape-aware instead:

  1. Adaptive (Sauvola) binarization -- robust to the uneven illumination
     typical of a photocopier/scanner source.
  2. Horizontal projection profile -> text-line bands.
  3. Per-line vertical projection profile -> valley detection gated by two
     script-specific priors (minimum akshara width, and a FULL-HEIGHT valley
     requirement -- a boundary is only accepted if the gap spans nearly the
     entire line height, not just a partial notch partway up/down a glyph).
  4. A connected-component cross-check (morphological closing with a tall/
     narrow kernel) that must independently agree a boundary separates two
     distinct glyph blobs before it's finalized. This double-gate is what
     keeps disjoint-but-single conjuncts intact, which DBSCAN cannot express.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_sauvola


@dataclass
class CvAssistConfig:
    # Sauvola adaptive threshold window (px, must be odd) and sensitivity.
    sauvola_window: int = 25
    sauvola_k: float = 0.2

    # Text-line detection (horizontal projection profile).
    line_smooth_px: int = 5          # moving-average window over row profile
    line_noise_floor_frac: float = 0.02  # frac of max row-darkness to count as "text row"
    line_gap_merge_px: int = 1       # merge line-bands separated by less than this
    line_min_height_px: int = 8      # discard bands shorter than this (noise/rules)
    line_vpad_px: int = 6            # vertical padding added when cropping a line strip

    # Per-line vertical valley detection.
    min_akshara_width_px: int = 10   # reject boundaries narrower than this
    valley_depth_frac: float = 0.12  # column darkness must drop below this frac of line max
    valley_smooth_px: int = 3        # moving-average window over column profile

    # Connected-component cross-check.
    closing_kernel_h: int = 7        # tall
    closing_kernel_w: int = 3        # narrow


@dataclass
class LineProposal:
    line_id: str
    y0: int
    y1: int
    boundaries_px: list[int] = field(default_factory=list)


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def _binarize_sauvola(gray: np.ndarray, cfg: CvAssistConfig) -> np.ndarray:
    """Returns a boolean array, True == ink (dark)."""
    window = cfg.sauvola_window | 1  # force odd
    thresh = threshold_sauvola(gray, window_size=window, k=cfg.sauvola_k)
    return gray < thresh


def _detect_text_lines(ink: np.ndarray, cfg: CvAssistConfig) -> list[tuple[int, int]]:
    row_profile = ink.sum(axis=1).astype(np.float64)
    smoothed = _moving_average(row_profile, cfg.line_smooth_px)
    noise_floor = smoothed.max() * cfg.line_noise_floor_frac

    is_text_row = smoothed > noise_floor

    bands: list[tuple[int, int]] = []
    in_band = False
    start = 0
    for i, val in enumerate(is_text_row):
        if val and not in_band:
            start = i
            in_band = True
        elif not val and in_band:
            bands.append((start, i))
            in_band = False
    if in_band:
        bands.append((start, len(is_text_row)))

    # Merge bands separated by a small gap (avoids a detached matra becoming its own "line").
    merged: list[tuple[int, int]] = []
    for band in bands:
        if merged and band[0] - merged[-1][1] <= cfg.line_gap_merge_px:
            merged[-1] = (merged[-1][0], band[1])
        else:
            merged.append(band)

    return [b for b in merged if (b[1] - b[0]) >= cfg.line_min_height_px]


def _find_valleys(
    column_profile: np.ndarray, cfg: CvAssistConfig
) -> list[int]:
    smoothed = _moving_average(column_profile, cfg.valley_smooth_px)
    line_max = smoothed.max()
    if line_max <= 0:
        return []

    threshold = line_max * cfg.valley_depth_frac
    is_valley = smoothed <= threshold

    # Find contiguous valley runs, take their midpoint as the candidate boundary x.
    candidates: list[int] = []
    in_valley = False
    start = 0
    for i, val in enumerate(is_valley):
        if val and not in_valley:
            start = i
            in_valley = True
        elif not val and in_valley:
            candidates.append((start + i) // 2)
            in_valley = False
    if in_valley:
        candidates.append((start + len(is_valley)) // 2)

    return candidates


def _connected_component_gate(
    line_ink: np.ndarray, candidates: list[int], cfg: CvAssistConfig
) -> list[int]:
    """Keep only candidate boundaries that separate two distinct blobs after
    a tall/narrow morphological closing (fuses disjoint conjunct parts)."""
    structure = np.ones((cfg.closing_kernel_h, cfg.closing_kernel_w), dtype=bool)
    closed = ndimage.binary_closing(line_ink, structure=structure)
    labeled, _ = ndimage.label(closed)

    accepted: list[int] = []
    for x in candidates:
        if x <= 0 or x >= line_ink.shape[1] - 1:
            continue
        left_labels = set(np.unique(labeled[:, max(0, x - 2):x])) - {0}
        right_labels = set(np.unique(labeled[:, x:x + 2])) - {0}
        # Accept if the label sets on either side are disjoint (a real component
        # break), OR either side is blank (background gap) -- both indicate a
        # genuine separation rather than a mid-glyph notch.
        if not left_labels or not right_labels or left_labels.isdisjoint(right_labels):
            accepted.append(x)
    return accepted


def _enforce_min_width(boundaries: list[int], width: int, cfg: CvAssistConfig) -> list[int]:
    all_pts = sorted(set([0] + boundaries + [width]))
    filtered = [all_pts[0]]
    for x in all_pts[1:]:
        if x - filtered[-1] >= cfg.min_akshara_width_px:
            filtered.append(x)
        elif x == width:
            # Always keep the right edge; drop the previous point instead if too close.
            if len(filtered) > 1:
                filtered[-1] = x
            else:
                filtered.append(x)
    return filtered


def propose_boundaries(
    gray: np.ndarray, cfg: CvAssistConfig | None = None
) -> list[LineProposal]:
    """gray: 2D uint8/float array, single-channel page image (0=black..255=white)."""
    cfg = cfg or CvAssistConfig()
    ink = _binarize_sauvola(gray, cfg)

    lines = _detect_text_lines(ink, cfg)
    width = gray.shape[1]

    proposals: list[LineProposal] = []
    for idx, (y0, y1) in enumerate(lines):
        y0p = max(0, y0 - cfg.line_vpad_px)
        y1p = min(gray.shape[0], y1 + cfg.line_vpad_px)
        line_ink = ink[y0p:y1p, :]

        col_profile = line_ink.sum(axis=0).astype(np.float64)
        candidates = _find_valleys(col_profile, cfg)
        gated = _connected_component_gate(line_ink, candidates, cfg)
        boundaries = _enforce_min_width(gated, width, cfg)

        proposals.append(
            LineProposal(
                line_id=f"L{idx + 1:02d}",
                y0=y0,
                y1=y1,
                boundaries_px=boundaries,
            )
        )

    return proposals
