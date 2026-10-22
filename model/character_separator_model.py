"""Generates a single character-separator seam x(y) via flow matching, given
a coarse anchor x-position (from the existing BoundaryModel peak-picking
detector).

Mirrors LineSeparatorModel exactly, transposed: a character separator's
domain is the HEIGHT axis (x as a function of y, within one line's own
height), so it needs per-ROW conditioning features -- ColumnCollapsePool
(collapse width, keep height), the same pooling LineSegmentationModel uses.

Unlike LineSeparatorModel, no backbone downsampling is needed: both strip
dimensions here are already small (a narrow width window ~2*margin_px, and
height = one line's own height, typically 15-45px) -- smaller than
BoundaryModel's existing 256px sliding windows -- so the resolution-
preserving SlimResNet18 backbone is cheap to run directly.

Reuses the same "avoid ink" energy loss as LineSeparatorModel: a character
separator should thread the gap between two glyphs, not cut through either
one -- same principle, same SDT energy channel, just the transposed axis.

No downsampling anywhere in this model means the generated path is already
at full (per-row) resolution -- no upsample/downsample step needed the way
LineSeparatorModel needs for its width axis.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.backbone import SlimResNet18
from model.config import BackboneConfig, ModelConfig
from model.flow_matching import PathFlowMatcher
from model.line_segmenter import ColumnCollapsePool
from model.seam_energy_loss import energy_avoidance_loss
from model.ssm import S4DStack


class CharacterSeparatorModel(nn.Module):
    """Input strip has 2 channels by default: grayscale + normalized SDT
    energy (see seam_extraction.compute_normalized_energy_channel)."""

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig(backbone=BackboneConfig(in_channels=2))
        self.cfg = cfg

        self.backbone = SlimResNet18(self.cfg.backbone)
        self.pool = ColumnCollapsePool(self.backbone.out_channels)
        self.ssm = S4DStack(self.cfg.ssm)
        self.flow = PathFlowMatcher(self.cfg.diffusion, feature_channels=self.cfg.ssm.d_model)

    def encode(self, strip: torch.Tensor) -> torch.Tensor:
        """strip: (B, 2, H, W_strip) [grayscale, SDT energy] vertical strip
        crop -> per-row conditioning features (B, d_model, H)."""
        feat = self.backbone(strip)  # (B, d_model, H, W_strip)
        seq = self.pool(feat)  # (B, d_model, H)
        h = self.ssm(seq.transpose(1, 2))  # (B, H, d_model)
        return h.transpose(1, 2)  # (B, d_model, H)

    def training_loss(
        self,
        strip: torch.Tensor,
        x1_norm: torch.Tensor,
        x0_norm: torch.Tensor,
        energy_weight: float = 0.25,
    ) -> torch.Tensor:
        """strip: (B,2,H,W_strip). x1_norm, x0_norm: (B, H) target/prior
        paths across the full line height, in [0,1] strip-local (width)
        coordinates."""
        h = self.encode(strip)  # (B, d_model, H) -- full resolution, no downsampling
        fm_loss, x1_pred = self.flow.training_loss_with_pred(
            x1_norm.unsqueeze(1), h, x0=x0_norm.unsqueeze(1)
        )

        energy_channel = strip[:, 1, :, :]  # (B, H, W_strip)
        # sample_along_path samples along dim1 at a position given per dim2
        # ("one value per column"); here we want to sample along WIDTH at a
        # position given per ROW, so transpose to put width first.
        energy_transposed = energy_channel.transpose(1, 2)  # (B, W_strip, H)
        energy_loss = energy_avoidance_loss(energy_transposed, x1_pred.squeeze(1))

        return fm_loss + energy_weight * energy_loss

    @torch.no_grad()
    def generate_path(
        self,
        strip: torch.Tensor,
        strip_x0_global: int,
        strip_width: int,
        anchor_x_norm: float,
        n_steps: int | None = None,
    ) -> torch.Tensor:
        """strip: (1,2,H,W_strip). anchor_x_norm: the coarse-detector's guess
        for this boundary, as a strip-local [0,1] fraction (flat prior).
        Returns the generated separator as page-GLOBAL pixel x-coordinates,
        one per row, shape (H,)."""
        assert strip.shape[0] == 1, "generate_path expects a single-example batch"
        h = self.encode(strip)
        height = h.shape[-1]
        x0 = torch.full((1, 1, height), anchor_x_norm, device=strip.device, dtype=torch.float32)
        x_norm = self.flow.generate(x0, h, n_steps=n_steps).squeeze(0).squeeze(0)  # (H,)
        return x_norm * strip_width + strip_x0_global
