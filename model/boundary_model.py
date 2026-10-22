"""Composes SlimResNet18 -> row-collapse attention pool -> S4D SSM stack ->
boundary head -> sliding-window diffusion self-correction into the full
BoundaryModel used for both training and inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.backbone import SlimResNet18
from model.config import ModelConfig
from model.diffusion_head import DiffusionSelfCorrector, stitch_windows
from model.peaks import peaks_from_signal
from model.ssm import S4DStack


class RowCollapsePool(nn.Module):
    """Turns a (B, C, H, W) feature map into a (B, C, W) sequence via a
    learned softmax-over-height attention pool (1x1 conv -> softmax over H),
    rather than naive average pooling -- lets the network weight the akshara
    body row over sparse super/subscript matra strokes."""

    def __init__(self, channels: int):
        super().__init__()
        self.attn_conv = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_logits = self.attn_conv(x)  # (B, 1, H, W)
        attn = torch.softmax(attn_logits, dim=2)  # softmax over H
        return (x * attn).sum(dim=2)  # (B, C, W)


class BoundaryModel(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()

        self.backbone = SlimResNet18(self.cfg.backbone)
        self.pool = RowCollapsePool(self.backbone.out_channels)
        self.ssm = S4DStack(self.cfg.ssm)
        self.boundary_head = nn.Linear(self.cfg.ssm.d_model, 1)
        self.diffusion = DiffusionSelfCorrector(
            self.cfg.diffusion, feature_channels=self.cfg.ssm.d_model
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, 1, H, W) grayscale line crop, W <= diffusion window size for
        a single forward pass (see `run_sliding_window` for full-line inference).
        Returns (logits (B, W), hidden (B, d_model, W))."""
        feat = self.backbone(x)  # (B, d_model, H, W)
        seq = self.pool(feat)  # (B, d_model, W)
        h = self.ssm(seq.transpose(1, 2))  # (B, W, d_model)
        logits = self.boundary_head(h).squeeze(-1)  # (B, W)
        return logits, h.transpose(1, 2)  # hidden back to (B, d_model, W)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """One-shot forward + diffusion self-correction, for windows already
        sized to fit in memory in a single pass (<= diffusion.window_px)."""
        logits, hidden = self.encode(x)
        refined = self.diffusion.self_correct(logits.unsqueeze(1), hidden)
        return {"logits": logits, "refined": refined.squeeze(1), "hidden": hidden}

    @torch.no_grad()
    def predict_boundaries(
        self,
        x: torch.Tensor,
        peak_threshold: float = 0.3,
        min_peak_distance_px: int = 8,
    ) -> list[int]:
        """x: (1, 1, H, W) single line crop (any width -- sliced into
        overlapping diffusion windows internally if wider than one window).
        Returns boundary x-positions (ints), always including 0 and W-1 edges."""
        assert x.shape[0] == 1, "predict_boundaries expects a single-example batch"
        width = x.shape[-1]
        window = self.cfg.diffusion.window_px
        stride = self.cfg.diffusion.stride_px

        if width <= window:
            out = self.forward(x)
            refined = out["refined"][0]  # (W,)
        else:
            preds, starts = [], []
            start = 0
            while start < width:
                end = min(start + window, width)
                crop = x[:, :, :, start:end]
                out = self.forward(crop)
                preds.append(out["refined"][0].unsqueeze(0))
                starts.append(start)
                if end == width:
                    break
                start += stride
            refined = stitch_windows(preds, starts, width)

        boundaries = peaks_from_signal(refined, peak_threshold, min_peak_distance_px)
        boundaries = sorted(set([0] + boundaries + [width - 1]))
        return boundaries
