"""LineSegmentationModel: learns to predict text-LINE y-bands for a whole
page, replacing the classical CV-assist row-projection heuristic with a
trained model -- the akshara-boundary BoundaryModel still handles x-axis
segmentation *within* a line, unchanged.

Unlike the akshara task (a partition of one line's width into adjacent
segments, cut by point boundaries), lines on a page are separate (y0, y1)
bands with real whitespace gaps between them -- not a full partition of the
page height. So instead of point-boundary logits, this model predicts a
continuous "line presence" mask p(y) in [0,1] along the page height (1 =
inside a text line, 0 = inter-line whitespace), reusing the exact same
backbone -> SSM -> diffusion-self-correction machinery as the akshara model,
just oriented along the page's y-axis (see LineSlimResNet18) and with a
different training target/decoding step:
  - Target: a soft rectangular mask per line (1 inside [y0,y1], smoothed at
    the edges), rather than gaussian bumps at cut-points.
  - Decoding: contiguous runs of the refined mask above a threshold become
    the predicted (y0,y1) line bands (mirrors cv_assist's row-projection
    run-finding, but on a learned signal).
  - The precise edges of each run (the y0/y1 transition points) are still
    exactly the same kind of "boundary" as the akshara task, so the existing
    soft-argmax localization loss is reused unchanged on the flattened list
    of all lines' y0s and y1s.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.config import LineModelConfig
from model.diffusion_head import DiffusionSelfCorrector
from model.line_backbone import LineSlimResNet18
from model.peaks import contiguous_runs_above_threshold, split_by_relative_minima
from model.ssm import S4DStack


class ColumnCollapsePool(nn.Module):
    """Mirror of RowCollapsePool (boundary_model.py) but collapses over W via
    softmax-over-width attention, producing a (B, C, H) sequence along the
    page's y-axis instead of (B, C, W) along x."""

    def __init__(self, channels: int):
        super().__init__()
        self.attn_conv = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_logits = self.attn_conv(x)  # (B, 1, H, W)
        attn = torch.softmax(attn_logits, dim=3)  # softmax over W
        return (x * attn).sum(dim=3)  # (B, C, H)


class LineSegmentationModel(nn.Module):
    def __init__(self, cfg: LineModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or LineModelConfig()

        self.backbone = LineSlimResNet18(self.cfg.line_backbone)
        self.pool = ColumnCollapsePool(self.backbone.out_channels)
        self.ssm = S4DStack(self.cfg.ssm)
        self.presence_head = nn.Linear(self.cfg.ssm.d_model, 1)
        self.diffusion = DiffusionSelfCorrector(
            self.cfg.diffusion, feature_channels=self.cfg.ssm.d_model
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, 1, H_page, W_page) -> (logits (B, H_page), hidden (B, d_model, H_page)).
        Runs in one shot (no sliding window) -- a full page height is small
        enough for the FFT-based SSM/diffusion to handle directly."""
        feat = self.backbone(x)  # (B, d_model, H_page, W_page/16)
        seq = self.pool(feat)  # (B, d_model, H_page)
        h = self.ssm(seq.transpose(1, 2))  # (B, H_page, d_model)
        logits = self.presence_head(h).squeeze(-1)  # (B, H_page)
        return logits, h.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        logits, hidden = self.encode(x)
        refined = self.diffusion.self_correct(logits.unsqueeze(1), hidden)
        return {"logits": logits, "refined": refined.squeeze(1), "hidden": hidden}

    @torch.no_grad()
    def predict_line_bands(
        self, x: torch.Tensor, presence_threshold: float = 0.5, min_line_height_px: int = 6
    ) -> list[tuple[int, int]]:
        """x: (1, 1, H_page, W_page) -> list of (y0, y1) line bands."""
        assert x.shape[0] == 1, "predict_line_bands expects a single-example batch"
        out = self.forward(x)
        # "refined" is the diffusion head's x0-parameterization regression output
        # (trained via MSE directly against a [0,1] target), not a logit -- no
        # sigmoid here, mirroring how BoundaryModel.predict_boundaries thresholds
        # its "refined" signal directly.
        refined = out["refined"][0]
        coarse_bands = contiguous_runs_above_threshold(refined, presence_threshold, min_line_height_px)

        # A coarse band can still contain multiple real lines: two lines
        # whose flat bounding boxes overlap (e.g. a hand-drawn curved split)
        # only depress the row-averaged presence signal partially, which may
        # not cross the absolute threshold above even though it's a real
        # gap. Recursively split each coarse band at any sufficiently
        # prominent internal dip to catch these.
        refined_np = refined.detach().cpu().numpy()
        bands: list[tuple[int, int]] = []
        for start, end in coarse_bands:
            bands.extend(split_by_relative_minima(refined_np, start, end))
        return bands
