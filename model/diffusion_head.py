"""Sliding-window diffusion-style self-correction head for boundary logits.

Disclosed reinterpretation (this is NOT literal generative image diffusion):
  - Forward process (training only): a ground-truth boundary signal z_0 (sum
    of narrow Gaussian bumps at true boundary x-positions) is noised via the
    standard DDPM forward marginal q(z_t | z_0) = sqrt(abar_t) z_0 + sqrt(1 -
    abar_t) eps, eps ~ N(0, I), for t in {1..T}.
  - Reverse process (train AND inference): a small conditioning network
    eps_theta(z_t, t, h) predicts z_0 directly (x0-parameterization -- more
    stable than noise-prediction at very small T like T=4), conditioned on
    the SSM's per-position hidden features h.
  - Inference departs from generative sampling: instead of starting from
    pure noise z_T ~ N(0,I), z_T is initialized from the SSM head's one-shot
    boundary logits (already a decent estimate), then T deterministic
    DDIM-style reverse steps run to self-correct it. This matches the
    "self-correcting / autocorrecting" framing from the design brief, rather
    than sampling a novel signal from scratch.
  - Sliding window: operates on windows of width `window_px` (default 256)
    with 50% overlap stride, so correcting one region doesn't require
    recomputing the whole page; overlapping predictions are stitched by
    averaging in the overlap zone (see `stitch_windows`).

Variance schedule: linear beta_t over T=4 steps (few steps => a coarse
schedule is fine; this is a low-dimensional 1D signal, not an image).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.config import DiffusionConfig


def make_beta_schedule(n_steps: int, beta_start: float = 1e-4, beta_end: float = 0.2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, n_steps)


class TimestepEmbedding(nn.Module):
    def __init__(self, n_steps: int, dim: int):
        super().__init__()
        self.table = nn.Embedding(n_steps, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) long -> (B, dim)"""
        return self.table(t)


class DiffusionSelfCorrector(nn.Module):
    """Conditioning network eps_theta(z_t, t, h) -> predicted z_0, implemented
    as 3 dilated Conv1d layers (dilations 1/2/4 grow the receptive field
    within one window without adding depth)."""

    def __init__(self, cfg: DiffusionConfig | None = None, feature_channels: int = 128):
        super().__init__()
        cfg = cfg or DiffusionConfig()
        self.cfg = cfg
        self.timestep_embed = TimestepEmbedding(cfg.n_steps, cfg.timestep_embed_dim)

        in_ch = 1 + feature_channels + cfg.timestep_embed_dim
        pad = lambda k, d: (d * (k - 1)) // 2

        self.conv1 = nn.Conv1d(in_ch, cfg.hidden_channels, cfg.kernel_size,
                                dilation=cfg.dilations[0], padding=pad(cfg.kernel_size, cfg.dilations[0]))
        self.conv2 = nn.Conv1d(cfg.hidden_channels, cfg.hidden_channels, cfg.kernel_size,
                                dilation=cfg.dilations[1], padding=pad(cfg.kernel_size, cfg.dilations[1]))
        self.conv3 = nn.Conv1d(cfg.hidden_channels, 1, cfg.kernel_size,
                                dilation=cfg.dilations[2], padding=pad(cfg.kernel_size, cfg.dilations[2]))
        self.act = nn.GELU()

        betas = make_beta_schedule(cfg.n_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def predict_z0(self, z_t: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """z_t: (B, 1, L), t: (B,) long, h: (B, feature_channels, L) -> predicted z0 (B, 1, L)"""
        length = z_t.shape[-1]
        t_embed = self.timestep_embed(t)  # (B, embed_dim)
        t_embed = t_embed.unsqueeze(-1).expand(-1, -1, length)  # (B, embed_dim, L)

        x = torch.cat([z_t, h, t_embed], dim=1)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        z0_pred = self.conv3(x)
        return z0_pred

    @torch.no_grad()
    def self_correct(self, z_init: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Deterministic DDIM-style reverse process, starting from the SSM's
        one-shot logits `z_init` (not noise) and running `n_steps` refinement
        passes. z_init: (B, 1, L), h: (B, feature_channels, L) -> refined (B, 1, L)."""
        n_steps = self.cfg.n_steps
        z_t = z_init
        for step in reversed(range(n_steps)):
            t = torch.full((z_t.shape[0],), step, device=z_t.device, dtype=torch.long)
            z0_pred = self.predict_z0(z_t, t, h)
            if step == 0:
                z_t = z0_pred
            else:
                # DDIM deterministic update toward the next-lower noise level.
                abar_t = self.alphas_cumprod[step]
                abar_prev = self.alphas_cumprod[step - 1]
                eps_pred = (z_t - torch.sqrt(abar_t) * z0_pred) / torch.sqrt(1 - abar_t)
                z_t = torch.sqrt(abar_prev) * z0_pred + torch.sqrt(1 - abar_prev) * eps_pred
        return z_t

    def training_loss(self, z0: torch.Tensor, h: torch.Tensor, pos_weight: float = 15.0) -> torch.Tensor:
        """Standard x0-parameterization training objective: sample a random
        timestep per example, noise z0 to that level, predict z0 back out,
        weighted MSE loss. z0: (B, 1, L), h: (B, feature_channels, L).

        pos_weight: boundary-bump targets are extremely sparse (a handful of
        ~3px-wide bumps in a sequence hundreds of pixels long, so >90% of
        positions are pure background/zero). Plain MSE over the whole
        sequence lets the model achieve a deceptively low loss by predicting
        near-zero everywhere -- it never has to commit to a sharp, confident
        peak at the actual boundary positions, and (observed in practice)
        converges to a generic, position-independent pattern rather than
        genuinely reading each line's content. Weighting each position's
        squared error by (1 + pos_weight * z0) makes errors AT/NEAR a true
        boundary count much more than errors in the background, the same
        imbalance-correction purpose focal loss serves for the classification
        head, just applied to this regression target."""
        batch = z0.shape[0]
        t = torch.randint(0, self.cfg.n_steps, (batch,), device=z0.device)
        abar_t = self.alphas_cumprod[t].view(-1, 1, 1)
        eps = torch.randn_like(z0)
        z_t = torch.sqrt(abar_t) * z0 + torch.sqrt(1 - abar_t) * eps
        z0_pred = self.predict_z0(z_t, t, h)
        weight = 1.0 + pos_weight * z0
        return (weight * (z0_pred - z0) ** 2).mean()


def stitch_windows(window_preds: list[torch.Tensor], window_starts: list[int], total_length: int) -> torch.Tensor:
    """Average overlapping window predictions into one full-length signal.
    window_preds[i]: (1, L_window) tensor for the window starting at window_starts[i]."""
    device = window_preds[0].device
    acc = torch.zeros(total_length, device=device)
    counts = torch.zeros(total_length, device=device)
    for pred, start in zip(window_preds, window_starts):
        length = pred.shape[-1]
        end = min(start + length, total_length)
        acc[start:end] += pred.squeeze(0)[: end - start]
        counts[start:end] += 1
    counts = counts.clamp(min=1)
    return acc / counts
