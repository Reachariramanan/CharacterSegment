"""Shared "don't cut through ink" auxiliary loss for seam-generating models
(both LineSeparatorModel and the character-separator model reuse this).

Problem this fixes: the flow-matching training loss only regresses the
generated path toward the annotated ground-truth curve -- it has no notion
of "characters are here, don't cross them." With only a handful of curved
training examples, the model can end up cutting through a glyph in
configurations the sparse annotation doesn't cover. Fix: penalize the
generated path directly using the SDT energy channel it's already
conditioned on (negative = inside ink, positive = background) -- a
differentiable, directly-computed nudge away from ink, on top of whatever
the annotated curves alone teach.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def downsample_width(img_bhw: torch.Tensor, target_width: int) -> torch.Tensor:
    """img_bhw: (B, H, W) -> (B, H, target_width) via linear interpolation
    along width (H is treated as the "channel" dim for F.interpolate)."""
    return F.interpolate(img_bhw, size=target_width, mode="linear", align_corners=True)


def sample_along_path(image_bhw: torch.Tensor, pos_norm_bl: torch.Tensor) -> torch.Tensor:
    """Differentiably samples `image_bhw` (B, H, L) at row positions given by
    `pos_norm_bl` (B, L), a [0,1]-normalized position per column (0 = row 0,
    1 = row H-1), via linear interpolation between the two nearest rows.
    Gradients flow into `pos_norm_bl` (not into the row indices themselves --
    standard differentiable-sampling behavior, same idea as grid_sample)."""
    batch, height, length = image_bhw.shape
    pos = pos_norm_bl.clamp(0.0, 1.0) * (height - 1)
    row0 = pos.floor().long().clamp(0, height - 2)
    row1 = row0 + 1
    frac = pos - row0.float()

    batch_idx = torch.arange(batch, device=image_bhw.device).unsqueeze(-1).expand(batch, length)
    col_idx = torch.arange(length, device=image_bhw.device).unsqueeze(0).expand(batch, length)

    val0 = image_bhw[batch_idx, row0, col_idx]
    val1 = image_bhw[batch_idx, row1, col_idx]
    return val0 * (1 - frac) + val1 * frac


def energy_avoidance_loss(energy_bhw: torch.Tensor, pos_norm_bl: torch.Tensor) -> torch.Tensor:
    """Penalizes the path (pos_norm_bl) for passing through strongly
    negative energy (deep inside ink) -- zero penalty once the path is in
    background territory (energy >= 0), so this only pushes the path OUT of
    characters, it doesn't otherwise bias where in the background it sits
    (that's still governed by the main flow-matching regression loss)."""
    sampled = sample_along_path(energy_bhw, pos_norm_bl)
    return F.relu(-sampled).mean()
