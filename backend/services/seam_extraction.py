"""Text-line extraction via minimum-energy seams (Shi & Setlur / "Text line
extraction for historical document images"-style dynamic programming),
implemented as a classical-CV alternative to both cv_assist.py (projection
profiles) and the trained LineSegmentationModel -- so all three can be
compared directly on the same page.

Implements the paper's binary formulation (Eq. 4), which is described as
"the main formula" for this method:

    E(i,j)   = SDT(i,j)   -- signed distance transform: negative INSIDE ink
               (text) components, positive outside. A low-energy path
               therefore threads through the middle of the written text.

    SeamMap[i,j] = 2*E(i,j) + min_{l in {-1,0,1}} SeamMap[i+l, j-1]     (Eq. 4)

    SeamMap[i,0] = E(i,0)                                              (init)
    i* = argmin_i SeamMap[i, W-1]                                      (endpoint)
    backtrack i* -> column 0 following the stored argmin choices        (seam)

Multiple text lines are found by taking several well-separated local minima
of the last column's costs (analogous to picking several valleys instead of
just the single global minimum), then backtracking each independently.

Scope note: the paper's grayscale extension (Eq. 7, using a Gray-level
Distance Transform GDT with diagonal-move weights 1/sqrt(2)) is NOT
implemented this session -- it requires an approximate weighted-distance
solver (chamfer relaxation or Dijkstra) that's a meaningfully larger, slower
piece of work than the binary SDT version the user flagged as primary. The
`compute_seam_map` DP core here already accepts arbitrary per-direction
weights, so plugging in a GDT energy map later needs no change to the DP
itself -- only a new energy function.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from backend.services.cv_assist import CvAssistConfig, _binarize_sauvola


@dataclass
class SeamExtractionConfig:
    min_seam_separation_px: int = 15  # minimum row-distance between distinct line seams
    max_seams: int | None = None      # cap on number of lines to extract (None = no cap)


def compute_signed_distance_energy(ink: np.ndarray) -> np.ndarray:
    """SDT(i,j): negative inside ink (text) components, positive outside,
    zero at the ink/background boundary. `ink`: boolean array, True = ink."""
    inside_dist = distance_transform_edt(ink)       # 0 outside ink, grows inward
    outside_dist = distance_transform_edt(~ink)     # 0 inside ink, grows outward
    return outside_dist - inside_dist


def compute_normalized_energy_channel(gray: np.ndarray, scale_px: float = 20.0) -> np.ndarray:
    """SDT energy from a grayscale page image, normalized to roughly [-1,1]
    (via tanh-like clipping at `scale_px`) so it's on a comparable scale to
    a [0,1] grayscale channel -- meant to be stacked as an extra input
    channel for a model, giving it a direct, computed-not-learned signal for
    "where the ink actually is" rather than making a tiny network discover
    that from scratch out of a handful of training examples (see the
    "explainable activations fall on the characters" design discussion)."""
    ink = _binarize_sauvola(gray, CvAssistConfig())
    energy = compute_signed_distance_energy(ink)
    return np.clip(energy / scale_px, -1.0, 1.0).astype(np.float32)


def compute_seam_map(
    energy: np.ndarray, weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> tuple[np.ndarray, np.ndarray]:
    """Forward DP pass building the SeamMap (Eq. 4) column by column.

    weights = (w_up, w_mid, w_down) multiply the three candidate predecessor
    costs before taking the min -- (1,1,1) for the binary Eq. 4 formulation;
    Eq. 7's grayscale weights (1/sqrt2, 1, 1/sqrt2) would plug in here
    unchanged if a GDT energy map were supplied instead of SDT.

    Returns (seam_map, choice) where choice[i,j] in {-1,0,+1} records which
    row offset at column j-1 produced the minimum feeding into (i,j), for
    backtracking.
    """
    height, width = energy.shape
    w_up, w_mid, w_down = weights

    seam_map = np.zeros((height, width), dtype=np.float64)
    choice = np.zeros((height, width), dtype=np.int8)

    seam_map[:, 0] = energy[:, 0]

    inf = np.inf
    for j in range(1, width):
        prev = seam_map[:, j - 1]
        prev_padded = np.concatenate(([inf], prev, [inf]))  # length height+2

        opt_up = prev_padded[0:height]      # prev[i-1], offset -1 (row above at prev column)
        opt_mid = prev_padded[1 : height + 1]  # prev[i],   offset  0
        opt_down = prev_padded[2 : height + 2]  # prev[i+1], offset +1

        stacked = np.stack([w_up * opt_up, w_mid * opt_mid, w_down * opt_down], axis=0)
        min_vals = np.min(stacked, axis=0)
        min_idx = np.argmin(stacked, axis=0)  # 0,1,2 -> offset -1,0,+1

        seam_map[:, j] = 2.0 * energy[:, j] + min_vals
        choice[:, j] = min_idx - 1

    return seam_map, choice


def backtrack_seam(choice: np.ndarray, start_row: int) -> np.ndarray:
    """Follows backpointers from the last column to the first, returning the
    seam's row (y) at every column -- shape (width,)."""
    height, width = choice.shape
    seam = np.zeros(width, dtype=np.int32)
    seam[width - 1] = start_row

    row = int(start_row)
    for j in range(width - 1, 0, -1):
        offset = int(choice[row, j])  # choice is int8; must widen before arithmetic with `row`
        row = int(np.clip(row + offset, 0, height - 1))
        seam[j - 1] = row
    return seam


def find_ink_avoiding_path(energy_2d: np.ndarray, anchor_index: int, anchor_pull: float = 4.0) -> np.ndarray:
    """Classical (non-learned) DP seam through `energy_2d` (height, width;
    SDT-style energy, negative = inside ink, positive = background -- see
    compute_normalized_energy_channel) that stays in the BACKGROUND -- the
    opposite of compute_seam_map's usual ink-following medial seam above
    (built to extract a line's own center, i.e. the opposite goal). Negating
    the energy makes the same DP favor positions far from ink instead.

    A quadratic per-row bias toward `anchor_index` (the coarse line/
    boundary detector's own guess) is added to EVERY column, not just
    picked as a tie-break at the end -- an earlier version only biased the
    final endpoint choice, which let the seam wander almost anywhere across
    the whole strip for every other column (verified directly: a strip
    covering only 80px produced a seam ranging over 75 of them) since nothing
    stopped it chasing the single most background-heavy point in the entire
    strip. The bias is normalized by strip half-height so it auto-scales:
    `anchor_pull` is roughly how many units of ink-avoidance benefit
    (energy is ~[-1,1]) it costs to reach the strip's edge -- 1.0 means
    "about as much as the full possible benefit", so the seam only strays
    that far when there's a genuinely strong reason to, not a mild one.

    Used as model_runtime.py's DEFAULT seam generator (for both the line
    separator and the character separator) ahead of the learned
    LineSeparatorModel/CharacterSeparatorModel -- as of this writing those
    checkpoints have only ever been trained on FLAT targets (no curved
    ground truth exists yet, see model/character_seam_dataset.py), so a
    near-zero training loss there does not mean they've learned genuine
    ink-avoiding curvature. This DP is ink-avoiding by construction, not by
    a soft loss term, so it needs no training data at all to be useful
    immediately. Swap back to the DL models once they've been trained on
    real curved annotations and demonstrably generalize beyond flat.
    """
    height = energy_2d.shape[0]
    max_dist = max(1, height - 1)
    row_bias = anchor_pull * ((np.arange(height) - anchor_index) / max_dist) ** 2
    avoidance_energy = -energy_2d + row_bias[:, None]
    seam_map, choice = compute_seam_map(avoidance_energy)
    best_row = int(np.argmin(seam_map[:, -1]))
    return backtrack_seam(choice, best_row)


def _local_minima_with_separation(
    values: np.ndarray, min_separation: int, max_count: int | None
) -> list[int]:
    candidates = [
        i
        for i in range(1, len(values) - 1)
        if values[i] <= values[i - 1] and values[i] <= values[i + 1]
    ]
    candidates.sort(key=lambda i: values[i])  # lowest cost first

    kept: list[int] = []
    for i in candidates:
        if all(abs(i - k) >= min_separation for k in kept):
            kept.append(i)
        if max_count is not None and len(kept) >= max_count:
            break
    return sorted(kept)


def extract_medial_seams(
    seam_map: np.ndarray, choice: np.ndarray, cfg: SeamExtractionConfig | None = None
) -> list[np.ndarray]:
    """Finds several well-separated low-cost endpoints in the last column and
    backtracks each into a full medial seam. Returns seams sorted by mean row
    (top of page to bottom)."""
    cfg = cfg or SeamExtractionConfig()
    last_col_costs = seam_map[:, -1]
    start_rows = _local_minima_with_separation(last_col_costs, cfg.min_seam_separation_px, cfg.max_seams)

    seams = [backtrack_seam(choice, r) for r in start_rows]
    seams.sort(key=lambda s: s.mean())
    return seams


def seams_to_line_bands(seams: list[np.ndarray], page_height: int) -> list[dict]:
    """Converts medial seams into (y0, y1) line bands compatible with the
    existing annotation schema, by taking the per-column midpoint between
    each pair of consecutive seams and averaging it into one scalar boundary
    -- a deliberate simplification, since this document's lines are straight
    (the user confirmed the page isn't warped this session); the full
    per-column seam is kept too (under "seam_px") for future curved-line /
    "bad orientation" work, where the midpoint should be used per-column
    instead of averaged."""
    if not seams:
        return []

    gaps = [seams[i + 1].mean() - seams[i].mean() for i in range(len(seams) - 1)]
    default_half_height = (np.median(gaps) / 2) if gaps else 15.0

    bands = []
    for idx, seam in enumerate(seams):
        if idx > 0:
            y0 = float(((seam + seams[idx - 1]) / 2).mean())
        else:
            y0 = max(0.0, seam.mean() - default_half_height)

        if idx < len(seams) - 1:
            y1 = float(((seam + seams[idx + 1]) / 2).mean())
        else:
            y1 = min(page_height - 1, seam.mean() + default_half_height)

        bands.append(
            {
                "line_id": f"S{idx + 1:02d}",
                "y0": int(round(y0)),
                "y1": int(round(y1)),
                "seam_px": seam.tolist(),
            }
        )
    return bands


def extract_line_bands(gray: np.ndarray, cfg: SeamExtractionConfig | None = None) -> list[dict]:
    """End-to-end GLOBAL seam extraction: grayscale page -> Sauvola binarize
    -> SDT energy -> seam DP over the WHOLE page width -> multi-seam
    extraction -> line bands.

    Empirically, on a page like this one (a table-of-contents layout with a
    large blank horizontal gap between the left text column and the right
    page-number column) this drifts badly: with no ink to follow across the
    gap, a seam can wander into a neighboring line before re-entering ink on
    the other side, so seams don't track individual lines cleanly. See
    `extract_line_bands_windowed` for the practical fix (constrain the
    search to a local vertical strip per CV-assist-proposed line), which is
    what should actually be used for comparison on this document. This
    global version is kept because it's the paper's literal formulation and
    is a useful, honest data point on its own limitation.
    """
    cfg = cfg or SeamExtractionConfig()
    ink = _binarize_sauvola(gray, CvAssistConfig())
    energy = compute_signed_distance_energy(ink)
    seam_map, choice = compute_seam_map(energy)
    seams = extract_medial_seams(seam_map, choice, cfg)
    return seams_to_line_bands(seams, gray.shape[0])


def extract_line_bands_windowed(
    gray: np.ndarray, anchor_bands: list[tuple[int, int]], window_margin_px: int = 25
) -> list[dict]:
    """Runs the SAME Eq.4 SDT seam-map DP as `extract_line_bands`, but
    independently within a narrow vertical strip around each externally
    supplied anchor (y0, y1) -- e.g. from cv_assist's projection-profile line
    detector -- rather than globally across the whole page. This confines
    each line's seam search to its own neighborhood, which avoids the
    cross-line drift seen in the unconstrained global version on documents
    with wide blank gaps within a line (see `extract_line_bands` docstring).

    Within each strip, the single lowest-cost seam (global min of the last
    column, no multi-seam search needed since exactly one line is expected
    per anchor) is taken as that line's medial seam, and the strip's own
    (y0, y1) anchor is kept as the reported band (the DP is used here purely
    to produce the per-column `seam_px` trace for future curved-line work,
    not to re-derive the band extent, which the anchor already gives us).

    The margin is clamped so a strip never extends into a NEIGHBORING
    anchor's own band -- this document has several real inter-line gaps as
    small as 3px (much smaller than a naive fixed margin), and without this
    clamp two adjacent strips would overlap heavily and let one line's seam
    bleed into its neighbor's search space, defeating the point of windowing.
    """
    bands = []
    for idx, (anchor_y0, anchor_y1) in enumerate(anchor_bands):
        prev_limit = anchor_bands[idx - 1][1] if idx > 0 else 0
        next_limit = anchor_bands[idx + 1][0] if idx < len(anchor_bands) - 1 else gray.shape[0]

        strip_y0 = max(0, anchor_y0 - window_margin_px, prev_limit)
        strip_y1 = min(gray.shape[0], anchor_y1 + window_margin_px, next_limit)
        strip_gray = gray[strip_y0:strip_y1, :]

        ink = _binarize_sauvola(strip_gray, CvAssistConfig())
        energy = compute_signed_distance_energy(ink)
        seam_map, choice = compute_seam_map(energy)

        best_row_local = int(np.argmin(seam_map[:, -1]))
        seam_local = backtrack_seam(choice, best_row_local)
        seam_global = (seam_local + strip_y0).tolist()

        bands.append(
            {
                "line_id": f"S{idx + 1:02d}",
                "y0": anchor_y0,
                "y1": anchor_y1,
                "seam_px": seam_global,
            }
        )
    return bands
