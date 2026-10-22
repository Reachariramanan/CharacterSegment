"""Generates a single curved line-separator seam y(x) via flow matching,
given a coarse anchor position (from the existing presence-based
LineSegmentationModel/cv_assist detector).

Why this needs per-column conditioning: a seam y(x) needs one feature vector
PER COLUMN (the path's domain is the width axis) -- unlike
LineSegmentationModel's per-ROW presence task. That per-column pooling
(RowCollapsePool: collapse height, keep width) is the same one BoundaryModel
already uses.

Why the backbone is LineSlimResNet18 (width-downsampling), not the fully
resolution-preserving SlimResNet18 BoundaryModel uses: BoundaryModel's crops
are narrow (<=256px, a sliding window), so full resolution is cheap. A line
separator spans the FULL PAGE WIDTH (~900px) -- running that through a
resolution-preserving 4-stage backbone was measured directly and is far too
slow for practical training (single-digit epochs took minutes). A line
seam's curvature is a smooth, low-frequency signal -- it does not need
per-pixel width precision the way an akshara cut does -- so this generates
the path at 1/16 width resolution (same downsampling LineSlimResNet18
already does for the line-presence task) and upsamples only the FINAL
generated/target path back to full width via linear interpolation.

Pipeline: LineSlimResNet18 -> RowCollapsePool -> S4DStack -> PathFlowMatcher
generates the (downsampled-length) seam directly -- no threshold/peak-
picking decode step, that's the point of switching to generation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.boundary_model import RowCollapsePool
from model.config import LineBackboneConfig, LineModelConfig
from model.flow_matching import PathFlowMatcher
from model.line_backbone import LineSlimResNet18
from model.seam_energy_loss import downsample_width, energy_avoidance_loss
from model.ssm import S4DStack


def _resample_1d(path: torch.Tensor, target_length: int) -> torch.Tensor:
    """path: (B, L) -> (B, target_length) via linear interpolation."""
    return F.interpolate(
        path.unsqueeze(1), size=target_length, mode="linear", align_corners=True
    ).squeeze(1)


class LineSeparatorModel(nn.Module):
    """Input strip has 2 channels by default: grayscale + a normalized SDT
    energy channel (see seam_extraction.compute_normalized_energy_channel).
    The energy channel is a directly-computed (not learned) signal for where
    the ink actually is, which matters a lot given how little curved
    training data exists -- the network only has to learn to follow/adjust
    the given energy valley, not discover from scratch where the ink is."""

    def __init__(self, cfg: LineModelConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = LineModelConfig(line_backbone=LineBackboneConfig(in_channels=2))
        self.cfg = cfg

        self.backbone = LineSlimResNet18(self.cfg.line_backbone)
        self.pool = RowCollapsePool(self.backbone.out_channels)
        self.ssm = S4DStack(self.cfg.ssm)
        self.flow = PathFlowMatcher(self.cfg.diffusion, feature_channels=self.cfg.ssm.d_model)

    def encode(self, strip: torch.Tensor) -> torch.Tensor:
        """strip: (B, 2, H_strip, W) [grayscale, SDT energy] horizontal strip
        crop -> per-column conditioning features (B, d_model, W/16-ish)."""
        feat = self.backbone(strip)  # (B, d_model, H_strip, W_down)
        seq = self.pool(feat)  # (B, d_model, W_down)
        h = self.ssm(seq.transpose(1, 2))  # (B, W_down, d_model)
        return h.transpose(1, 2)  # (B, d_model, W_down)

    def training_loss(
        self,
        strip: torch.Tensor,
        y1_norm_full: torch.Tensor,
        y0_norm_full: torch.Tensor,
        energy_weight: float = 0.25,
    ) -> torch.Tensor:
        """strip: (B,2,H_strip,W) -- channel 0 grayscale, channel 1 SDT
        energy. y1_norm_full, y0_norm_full: (B, W) target/prior paths at
        FULL strip width, in [0,1] strip-local coordinates -- downsampled
        internally to match the backbone's reduced width before computing
        the flow-matching loss.

        energy_weight: weight on the auxiliary "don't cut through ink" term
        (see model/seam_energy_loss.py) -- computed on the path RECONSTRUCTED
        from the network's current velocity prediction at a random noise
        level, not a full generation pass, so it's cheap to include on every
        training step."""
        h = self.encode(strip)  # (B, d_model, W_down)
        w_down = h.shape[-1]
        y1_down = _resample_1d(y1_norm_full, w_down)
        y0_down = _resample_1d(y0_norm_full, w_down)

        fm_loss, y1_pred_down = self.flow.training_loss_with_pred(
            y1_down.unsqueeze(1), h, x0=y0_down.unsqueeze(1)
        )

        energy_channel = strip[:, 1, :, :]  # (B, H_strip, W) -- full resolution
        energy_down = downsample_width(energy_channel, w_down)  # (B, H_strip, W_down)
        energy_loss = energy_avoidance_loss(energy_down, y1_pred_down.squeeze(1))

        return fm_loss + energy_weight * energy_loss

    @torch.no_grad()
    def generate_path(
        self,
        strip: torch.Tensor,
        strip_y0_global: int,
        strip_height: int,
        anchor_y_norm: float,
        n_steps: int | None = None,
    ) -> torch.Tensor:
        """strip: (1,2,H_strip,W). anchor_y_norm: the coarse-detector's guess
        for this seam, as a strip-local [0,1] fraction (flat prior). Returns
        the generated seam as page-GLOBAL pixel y-coordinates, upsampled
        back to the strip's full width, shape (W,)."""
        assert strip.shape[0] == 1, "generate_path expects a single-example batch"
        full_width = strip.shape[-1]
        h = self.encode(strip)
        w_down = h.shape[-1]
        x0 = torch.full((1, 1, w_down), anchor_y_norm, device=strip.device, dtype=torch.float32)
        y_norm_down = self.flow.generate(x0, h, n_steps=n_steps)  # (1,1,w_down)
        y_norm_full = _resample_1d(y_norm_down.squeeze(1), full_width).squeeze(0)  # (W,)
        return y_norm_full * strip_height + strip_y0_global
