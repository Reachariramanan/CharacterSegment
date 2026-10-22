"""Conditional Flow Matching (CFM) for generating a 1D path directly, as an
evolution of diffusion_head.py's DDPM-style corrector.

Why flow matching instead of extending the DDPM head: the DDPM head predicts
a SCALAR presence/likelihood value per position, decoded afterward via
thresholding/peak-picking -- that decoding step is what caps everything at
straight, axis-aligned cuts (see model/line_segmenter.py's predict_line_bands
docstring and the "seam generation" design discussion). Flow matching instead
learns to GENERATE the path itself: a length-L vector of positions along the
free axis (y-per-column for a line separator, x-per-row for a character
separator), so curvature is representable by construction, not lost at
decode time.

Math (rectified/linear-path conditional flow matching, Lipman et al. 2022):
  Given a prior sample x0 (a coarse initial guess, NOT pure noise -- same
  "self-correcting" framing as the DDPM head) and the true path x1, define
  the straight-line interpolant:
      x_t = (1 - t) * x0 + t * x1,   t in [0, 1]
  Its time-derivative is constant along the path:
      u_t = dx_t/dt = x1 - x0
  Train a network v_theta(x_t, t, h) to regress this constant velocity:
      L = E_{t~U(0,1), x0, x1} || v_theta(x_t, t, h) - (x1 - x0) ||^2
  At generation time, integrate the learned ODE dx/dt = v_theta(x, t, h)
  forward from x0 (t=0) to x1 (t=1) via a few Euler steps -- no noise
  schedule, no alpha_bar bookkeeping, generally fewer steps needed than DDPM
  for the same quality, and it composes naturally with the existing
  "start from a decent guess, not from scratch" self-correction framing.

Same architecture pattern as DiffusionSelfCorrector (dilated Conv1d stack
conditioned on SSM hidden features), swapped from a discrete DDPM timestep
embedding to a continuous-t embedding since t in [0,1] here.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from model.config import DiffusionConfig


class ContinuousTimeEmbedding(nn.Module):
    """Sinusoidal embedding for continuous t in [0,1] (standard for
    continuous-time diffusion/flow models -- t is not a discrete index here,
    so a lookup-table embedding like the DDPM head's doesn't apply)."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "ContinuousTimeEmbedding dim must be even"
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) float in [0,1] -> (B, dim)"""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0) * (2 * math.pi)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class PathFlowMatcher(nn.Module):
    """v_theta(x_t, t, h) -> predicted velocity, same dilated-Conv1d shape as
    DiffusionSelfCorrector's conditioning network, conditioned on the SSM's
    per-position hidden features h."""

    def __init__(self, cfg: DiffusionConfig | None = None, feature_channels: int = 128):
        super().__init__()
        cfg = cfg or DiffusionConfig()
        self.cfg = cfg
        self.time_embed = ContinuousTimeEmbedding(cfg.timestep_embed_dim)

        in_ch = 1 + feature_channels + cfg.timestep_embed_dim
        pad = lambda k, d: (d * (k - 1)) // 2

        self.conv1 = nn.Conv1d(in_ch, cfg.hidden_channels, cfg.kernel_size,
                                dilation=cfg.dilations[0], padding=pad(cfg.kernel_size, cfg.dilations[0]))
        self.conv2 = nn.Conv1d(cfg.hidden_channels, cfg.hidden_channels, cfg.kernel_size,
                                dilation=cfg.dilations[1], padding=pad(cfg.kernel_size, cfg.dilations[1]))
        self.conv3 = nn.Conv1d(cfg.hidden_channels, 1, cfg.kernel_size,
                                dilation=cfg.dilations[2], padding=pad(cfg.kernel_size, cfg.dilations[2]))
        self.act = nn.GELU()

    def velocity(self, x_t: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """x_t: (B, 1, L), t: (B,) float in [0,1], h: (B, feature_channels, L)
        -> predicted velocity (B, 1, L)."""
        length = x_t.shape[-1]
        t_embed = self.time_embed(t)  # (B, embed_dim)
        t_embed = t_embed.unsqueeze(-1).expand(-1, -1, length)  # (B, embed_dim, L)

        x = torch.cat([x_t, h, t_embed], dim=1)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return self.conv3(x)

    def training_loss_with_pred(
        self, x1: torch.Tensor, h: torch.Tensor, x0: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Like `training_loss`, but also returns a reconstructed estimate of
        x1 at the sampled (x_t, t) -- since x_t = (1-t)x0 + t*x1 and the
        network predicts v ~= x1-x0, algebra gives an exact reconstruction
        x1_pred = x_t + (1-t)*v_pred (exact if v_pred is exact, a useful
        estimate otherwise). Callers use this to attach an auxiliary loss on
        the (still-learning) predicted path -- e.g. LineSeparatorModel's
        "don't cut through ink" energy penalty -- without needing a full
        multi-step generation pass during training."""
        if x0 is None:
            x0 = torch.randn_like(x1)
        batch = x1.shape[0]
        t = torch.rand(batch, device=x1.device)
        t_broadcast = t.view(-1, 1, 1)
        x_t = (1 - t_broadcast) * x0 + t_broadcast * x1
        target_v = x1 - x0
        pred_v = self.velocity(x_t, t, h)
        loss = torch.nn.functional.mse_loss(pred_v, target_v)
        x1_pred = x_t + (1 - t_broadcast) * pred_v
        return loss, x1_pred

    def training_loss(
        self, x1: torch.Tensor, h: torch.Tensor, x0: torch.Tensor | None = None
    ) -> torch.Tensor:
        """x1: (B, 1, L) true path (e.g. real seam y-positions, normalized --
        see PathFlowMatcher module docstring / caller for normalization).
        x0: (B, 1, L) prior/coarse-guess path; defaults to Gaussian noise if
        not given, but callers should pass a warm-started coarse guess where
        available (matches the DDPM head's self-correcting framing)."""
        loss, _ = self.training_loss_with_pred(x1, h, x0)
        return loss

    @torch.no_grad()
    def generate(self, x0: torch.Tensor, h: torch.Tensor, n_steps: int | None = None) -> torch.Tensor:
        """Deterministic forward Euler integration of dx/dt = v_theta(x,t,h)
        from x0 (t=0) to x1 (t=1). x0: (B, 1, L) prior/coarse-guess path,
        h: (B, feature_channels, L) -> generated path (B, 1, L)."""
        n_steps = n_steps or self.cfg.n_steps
        dt = 1.0 / n_steps
        x = x0
        for step in range(n_steps):
            t = torch.full((x.shape[0],), step * dt, device=x.device, dtype=torch.float32)
            v = self.velocity(x, t, h)
            x = x + v * dt
        return x
